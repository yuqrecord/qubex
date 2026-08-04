"""
AC Stark spectroscopy with a flat-top resonator (readout) drive.

This module measures the AC Stark shift of a qubit induced by a resonator
(readout cavity) drive. The qubit and the resonator are driven
simultaneously, then the resonator is read out with the standard readout
pulse. The resonator drive is a ``FlatTop`` pulse parameterized by
amplitude, duration and ramp time, so its amplitude can be swept; see
``ac_stark_spectroscopy_arbitrary_readout_drive`` for the variant that
plays a caller-supplied waveform instead.

The pulse sequence, the timing conventions and the capture-window layout
are documented in ``_core``; the Lorentzian fit and the shift derivation
are documented in ``_analysis``.

Sweep axes
----------
``qubit_drive_detuning_range`` is the spectroscopy axis and is always
swept inside one trace. The outer axes are the preparation state, the
resonator drive amplitude and the qubit drive time offset.

The AC Stark shift is referenced to a state- and offset-resolved
*no-drive* measurement: for every (state, time offset) a spectroscopy
trace is acquired with a ``Blank`` in place of the resonator drive but
otherwise identical timing, and

    shift = f0_driven - f0_no_drive

This removes calibration offsets and offset-dependent effects (e.g. T1
decay of the |e> preparation) from the shift. The public
``resonator_drive_amplitude_range`` must contain only positive values;
the no-drive reference is acquired internally.

Photon number conversion
------------------------
``dispersive_shift`` is the *signed* dispersive shift ``chi`` (GHz) in
the convention where the qubit frequency shifts by ``2 * chi`` per
intracavity photon:

    delta_f_qubit(n) = 2 * chi * n_photon
    n_photon = delta_f_qubit / (2 * chi)

If the resonator drive overlaps the qubit probe only partially (time
offset sweep), the inferred photon number is an effective value averaged
over the qubit probe envelope, not an instantaneous photon number.
Both preparation states measure the shift of the same ge transition, so
the same ge chi applies to both.

Assumptions and caveats
-----------------------
- Dispersive regime, weak spectroscopy probe, and an approximately
  Lorentzian line. Strong drives cause power broadening,
  measurement-induced dephasing, and (at high photon number) departures
  from the linear dispersive relation.
- Frequencies are in GHz, durations in ns, and amplitudes in normalized
  hardware units.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam, Result
from qubex.pulse import Blank, FlatTop, PulseSchedule, Waveform

from . import _analysis, _core, _plotting

# A driven schedule plays exactly two pulses on the read channel (the
# resonator drive and the readout), while the no-drive reference replaces
# the drive with a Blank and therefore opens a single capture window.
DRIVEN_CAPTURE_COUNT = 2
REFERENCE_CAPTURE_COUNT = 1


def _validate_flat_top(
    name: str,
    *,
    duration: float,
    ramptime: float,
) -> None:
    if duration <= 0:
        raise ValueError(f"`{name}_duration` must be positive, got {duration}.")
    if ramptime <= 0:
        raise ValueError(f"`{name}_ramptime` must be positive, got {ramptime}.")
    if duration < 2 * ramptime:
        raise ValueError(
            f"`{name}_duration` ({duration} ns) must be at least twice "
            f"`{name}_ramptime` ({ramptime} ns)."
        )


def _validate_drive_parameters(
    *,
    amplitude_name: str,
    resonator_drive_amplitude: float,
    resonator_drive_duration: float,
    resonator_drive_ramptime: float,
    resonator_drive_detuning: float,
) -> None:
    _core.validate_amplitude(
        amplitude_name, resonator_drive_amplitude, allow_zero=False
    )
    _core.validate_time_on_grid("resonator_drive_duration", resonator_drive_duration)
    _core.validate_time_on_grid("resonator_drive_ramptime", resonator_drive_ramptime)
    _validate_flat_top(
        "resonator_drive",
        duration=resonator_drive_duration,
        ramptime=resonator_drive_ramptime,
    )
    _core.validate_finite_scalar("resonator_drive_detuning", resonator_drive_detuning)


def _resonator_drive_pulse(
    *,
    resonator_drive_amplitude: float,
    resonator_drive_duration: float,
    resonator_drive_ramptime: float,
    resonator_drive_detuning: float,
) -> Waveform:
    return FlatTop(
        duration=resonator_drive_duration,
        amplitude=resonator_drive_amplitude,
        tau=resonator_drive_ramptime,
    ).detuned(resonator_drive_detuning)


def _pulse_schedule_title(
    *,
    target: str,
    qubit_initial_state: str,
    resonator_drive_amplitude: float,
    qubit_drive_time_offset: float,
) -> str:
    return (
        f"AC Stark spectroscopy : {target} : "
        f"|{qubit_initial_state}> "
        f"(A_r={resonator_drive_amplitude:g}, "
        f"offset={qubit_drive_time_offset:g} ns)"
    )


def ac_stark_spectroscopy_flattop_readout_drive_sequence(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_detuning: float,
    qubit_drive_time_offset: float,
    resonator_drive_amplitude: float,
    resonator_drive_duration: float,
    resonator_drive_ramptime: float,
    resonator_drive_detuning: float,
    waittime_after_pi_pulse: float,
    waittime_before_readout: float,
) -> PulseSchedule:
    """
    Build the AC Stark spectroscopy pulse schedule for one sweep point.

    Sequence: state preparation (pi pulse for "e", duration-matched blank
    for "g") followed by ``waittime_after_pi_pulse``, then a flat-top
    resonator drive and the detuned pi pulse probe with relative time
    offset, then the standard readout starting at
    ``(resonator drive end) + waittime_before_readout``.

    Parameters
    ----------
    exp : Experiment
        Experiment object providing pulses and hardware context.
    target : str
        Target qubit name or label.
    qubit_initial_state : {"g", "e"}
        Qubit state prepared before the drives. "e" uses the calibrated
        ge pi pulse (``Experiment.x180``).
    qubit_drive_detuning : float
        Detuning of the qubit drive from the calibrated ge transition
        frequency (GHz). The qubit drive is the calibrated ge pi pulse
        detuned by this amount, always applied on the ge control channel
        for both preparation states.
    qubit_drive_time_offset : float
        Start time of the qubit drive relative to the resonator drive
        start (ns). Negative values start the qubit drive first.
    resonator_drive_amplitude : float
        Amplitude of the flat-top resonator drive.
    resonator_drive_duration : float
        Duration of the resonator drive in ns.
    resonator_drive_ramptime : float
        Ramp time (flat-top tau) of the resonator drive in ns.
    resonator_drive_detuning : float
        Detuning of the resonator drive from the calibrated readout
        frequency (GHz).
    waittime_after_pi_pulse : float
        Delay between the state-preparation slot and the drive section
        (ns). Applied for both preparation states to keep identical
        timing.
    waittime_before_readout : float
        Delay between the resonator drive end and the readout pulse
        object start (ns).

    Returns
    -------
    PulseSchedule
        Schedule with two pulses on the read channel (resonator drive and
        readout), producing two capture windows.
    """
    return _core.build_sequence(
        exp,
        target=target,
        qubit_initial_state=qubit_initial_state,
        qubit_drive_detuning=qubit_drive_detuning,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_pulse=_resonator_drive_pulse(
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            resonator_drive_ramptime=resonator_drive_ramptime,
            resonator_drive_detuning=resonator_drive_detuning,
        ),
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
    )


def run_ac_stark_spectroscopy_flattop_readout_drive(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_detuning_range: ArrayLike,
    qubit_drive_time_offset: float,
    resonator_drive_amplitude: float,
    resonator_drive_duration: float,
    resonator_drive_ramptime: float,
    resonator_drive_detuning: float,
    waittime_after_pi_pulse: float,
    waittime_before_readout: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> dict[str, Any]:
    """
    Acquire one ge spectroscopy trace for a single parameter set.

    Sweeps ``qubit_drive_detuning_range`` and measures the averaged,
    time-integrated readout IQ value for each point using
    ``Experiment.execute``.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    qubit_initial_state : {"g", "e"}
        Qubit state prepared before the drives. Both states probe the ge
        transition on the ge control channel: "g" measures ge absorption,
        "e" measures ge stimulated emission after a pi pulse.
    qubit_drive_detuning_range : ArrayLike
        Sweep values of the qubit drive detuning from the calibrated ge
        transition frequency (GHz). The probe is the calibrated ge pi
        pulse (``Experiment.x180``) detuned by each of these values. At
        least 5 distinct points are required.
    qubit_drive_time_offset : float
        Qubit drive start time relative to the resonator drive start (ns).
    resonator_drive_amplitude : float
        Amplitude of the flat-top resonator drive. Must be positive: the
        no-drive reference is acquired by the main entry point.
    resonator_drive_duration : float
        Duration of the resonator drive in ns.
    resonator_drive_ramptime : float
        Ramp time of the resonator drive in ns.
    resonator_drive_detuning : float
        Detuning of the resonator drive from the calibrated readout
        frequency (GHz).
    waittime_after_pi_pulse : float
        Delay between the state-preparation slot and the drive section
        (ns). Applied for both preparation states to keep identical
        timing.
    waittime_before_readout : float
        Delay between the resonator drive end and the readout start (ns).
    shot_interval : float
        Interval between shots in ns.
    n_shots : int
        Number of shots per sweep point.
    plot_pulse_schedule : bool, optional
        If True, plot the pulse schedule of the first sweep point.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    dict[str, Any]
        Raw acquisition data with keys ``detuning_range``,
        ``frequency_range``, ``iq``, ``measure_results`` (the list of raw
        ``MultipleMeasureResult`` objects returned by
        ``Experiment.execute``, one per detuning point), labels,
        ``nominal_frequency`` (the calibrated ge transition frequency),
        ``qubit_drive_pulse`` and ``resonator_drive_pulse`` (the waveforms
        that were played) and an echo of the drive parameters.
    """
    qubit_drive_duration = float(_core.qubit_drive_waveform(exp, target).duration)
    _core.validate_state("qubit_initial_state", qubit_initial_state)
    _validate_drive_parameters(
        amplitude_name="resonator_drive_amplitude",
        resonator_drive_amplitude=resonator_drive_amplitude,
        resonator_drive_duration=resonator_drive_duration,
        resonator_drive_ramptime=resonator_drive_ramptime,
        resonator_drive_detuning=resonator_drive_detuning,
    )
    _core.validate_common_parameters(
        qubit_drive_duration=qubit_drive_duration,
        qubit_drive_detuning_range=qubit_drive_detuning_range,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_duration=resonator_drive_duration,
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
        shot_interval=shot_interval,
        n_shots=n_shots,
    )

    run_result = _core.acquire_spectroscopy_trace(
        exp,
        target=target,
        qubit_initial_state=qubit_initial_state,
        qubit_drive_detuning_range=qubit_drive_detuning_range,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_pulse=_resonator_drive_pulse(
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            resonator_drive_ramptime=resonator_drive_ramptime,
            resonator_drive_detuning=resonator_drive_detuning,
        ),
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
        shot_interval=shot_interval,
        n_shots=n_shots,
        expected_capture_count=DRIVEN_CAPTURE_COUNT,
        strict_capture_count=True,
        plot_pulse_schedule=plot_pulse_schedule,
        pulse_schedule_title=_pulse_schedule_title(
            target=target,
            qubit_initial_state=qubit_initial_state,
            resonator_drive_amplitude=resonator_drive_amplitude,
            qubit_drive_time_offset=qubit_drive_time_offset,
        ),
        reset_awg_and_capunits=(
            bool(reset_awg_and_capunits)
            if reset_awg_and_capunits is not None
            else False
        ),
        progress_description=f"qubit detuning ({qubit_initial_state})",
    )
    run_result["resonator_drive_amplitude"] = resonator_drive_amplitude
    run_result["resonator_drive_duration"] = resonator_drive_duration
    run_result["resonator_drive_ramptime"] = resonator_drive_ramptime
    run_result["resonator_drive_detuning"] = resonator_drive_detuning
    return run_result


def analyze_ac_stark_spectroscopy_flattop_readout_drive(
    *,
    detuning_range: ArrayLike,
    iq: ArrayLike,
    nominal_frequency: float,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_duration: float,
    rabi_param: RabiParam,
) -> dict[str, Any]:
    """
    Fit one ge spectroscopy trace and extract the resonance frequency.

    Thin wrapper around the shared Lorentzian fit; see
    ``_analysis.fit_spectroscopy_trace`` for the normalization, the
    initial guess and the polarity convention.

    Parameters
    ----------
    detuning_range : ArrayLike
        Qubit drive detuning sweep (GHz), in acquisition order.
    iq : ArrayLike
        Complex readout IQ values for each detuning point.
    nominal_frequency : float
        Calibrated ge transition frequency (GHz).
    qubit_initial_state : {"g", "e"}
        Prepared qubit state; fixes the expected line polarity.
    qubit_drive_duration : float
        Duration of the pi pulse used as the probe (ns).
    rabi_param : RabiParam
        Calibrated GE Rabi parameters used to normalize the IQ data.

    Returns
    -------
    dict[str, Any]
        Fit results with keys ``normalized``, ``fit_result``, ``f0``,
        ``f0_err``, ``gamma``, ``r2`` and ``valid``.
    """
    return _analysis.fit_spectroscopy_trace(
        detuning_range=detuning_range,
        iq=iq,
        nominal_frequency=nominal_frequency,
        qubit_initial_state=qubit_initial_state,
        qubit_drive_duration=qubit_drive_duration,
        rabi_param=rabi_param,
    )


def ac_stark_spectroscopy_flattop_readout_drive(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_states: Collection[Literal["g", "e"]] | str,
    qubit_drive_detuning_range: ArrayLike,
    qubit_drive_time_offset_range: float | ArrayLike,
    resonator_drive_amplitude_range: float | ArrayLike,
    resonator_drive_duration: float,
    resonator_drive_ramptime: float,
    resonator_drive_detuning: float,
    waittime_after_pi_pulse: float,
    waittime_before_readout: float,
    dispersive_shift: float | None,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> Result:
    """
    Run an AC Stark spectroscopy sweep and extract shifts and photon numbers.

    For every combination of preparation state, resonator drive amplitude
    and qubit drive time offset, a ge spectroscopy trace (versus
    ``qubit_drive_detuning_range``) is acquired and fitted. Both states
    probe the ge transition on the ge control channel ("g": absorption
    dip, "e": stimulated-emission peak after a pi pulse). The AC Stark
    shift is referenced to a no-drive spectroscopy measured once per
    (state, offset) with a ``Blank`` in place of the resonator drive and
    otherwise identical timing: ``shift = f0_driven - f0_no_drive``. With
    ``dispersive_shift`` (signed ge chi, GHz) given, the shift is
    converted to the mean intracavity photon number
    ``n = shift / (2 * chi)``.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    qubit_initial_states : Collection[{"g", "e"}] | str
        Preparation states to measure, e.g. ``["g"]`` or ``["g", "e"]``.
        Both states probe the ge transition.
    qubit_drive_detuning_range : ArrayLike
        Sweep values of the qubit drive detuning from the calibrated ge
        transition frequency (GHz). The probe is the calibrated ge pi
        pulse (``Experiment.x180``) detuned by each of these values;
        its shape, amplitude and duration are not free parameters.
        Always swept; at least 5 distinct points are required.
    qubit_drive_time_offset_range : float | ArrayLike
        Sweep axis: qubit drive start relative to the resonator drive
        start (ns). A scalar or a list; negative values are allowed.
    resonator_drive_amplitude_range : float | ArrayLike
        Sweep axis: resonator drive amplitude. A scalar or a list;
        every value must be positive.
    resonator_drive_duration : float
        Duration of the resonator drive in ns.
    resonator_drive_ramptime : float
        Ramp time (flat-top tau) of the resonator drive in ns.
    resonator_drive_detuning : float
        Detuning of the resonator drive from the calibrated readout
        frequency (GHz).
    waittime_after_pi_pulse : float
        Delay between the state-preparation slot and the drive section
        (ns). Applied for both preparation states to keep identical
        timing.
    waittime_before_readout : float
        Delay between the resonator drive end and the readout start (ns).
    dispersive_shift : float | None
        Signed dispersive shift chi in GHz, in the convention
        ``delta_f_qubit = 2 * chi * n_photon``. Pass None to skip the
        photon number conversion.
    shot_interval : float
        Interval between shots in ns.
    n_shots : int
        Number of shots per sweep point.
    plot : bool, optional
        If True, display the result figures.
    plot_pulse_schedule : bool, optional
        If True, plot the pulse schedule of the first driven point of
        each preparation state.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    Result
        Result with ``data`` (axes, raw spectra, fits and derived
        quantities; see the module docstring) and ``figures``. The
        ``data["raw"]`` entry holds the unprocessed
        ``MultipleMeasureResult`` objects from ``Experiment.execute`` as
        object arrays: ``measure_results`` (state, amplitude, offset,
        detuning) and ``no_drive_measure_results`` (state, offset,
        detuning). ``data["parameters"]["qubit_drive_pulse"]`` holds the
        ``Experiment.x180`` waveform used as the probe (before detuning);
        its name, duration and samples are read off the waveform itself.
    """
    if isinstance(qubit_initial_states, str):
        states = [qubit_initial_states]
    else:
        states = list(qubit_initial_states)
    if len(states) == 0:
        raise ValueError("`qubit_initial_states` must not be empty.")
    if len(set(states)) != len(states):
        raise ValueError("`qubit_initial_states` must not contain duplicates.")
    for state in states:
        _core.validate_state("qubit_initial_states", state)

    _core.validate_axis(
        "resonator_drive_amplitude_range", resonator_drive_amplitude_range
    )
    _core.validate_axis("qubit_drive_time_offset_range", qubit_drive_time_offset_range)
    amplitude_range = np.atleast_1d(
        np.asarray(resonator_drive_amplitude_range, dtype=float)
    )
    offset_range = np.atleast_1d(np.asarray(qubit_drive_time_offset_range, dtype=float))
    # The probe is the calibrated ge pi pulse, so its duration is fixed by
    # the calibration and only enters the timing constraint. The waveform
    # itself is stored in the result so that the probe actually played can
    # be inspected afterwards.
    qubit_drive_pulse = _core.qubit_drive_waveform(exp, target)
    qubit_drive_duration = float(qubit_drive_pulse.duration)
    for amplitude in amplitude_range:
        _validate_drive_parameters(
            amplitude_name="resonator_drive_amplitude_range",
            resonator_drive_amplitude=float(amplitude),
            resonator_drive_duration=resonator_drive_duration,
            resonator_drive_ramptime=resonator_drive_ramptime,
            resonator_drive_detuning=resonator_drive_detuning,
        )
    for offset in offset_range:
        _core.validate_common_parameters(
            qubit_drive_duration=qubit_drive_duration,
            qubit_drive_detuning_range=qubit_drive_detuning_range,
            qubit_drive_time_offset=float(offset),
            resonator_drive_duration=resonator_drive_duration,
            waittime_after_pi_pulse=waittime_after_pi_pulse,
            waittime_before_readout=waittime_before_readout,
            shot_interval=shot_interval,
            n_shots=n_shots,
            time_offset_name="qubit_drive_time_offset_range",
        )
    detuning_range = np.atleast_1d(np.asarray(qubit_drive_detuning_range, dtype=float))
    if dispersive_shift is not None:
        _core.validate_finite_scalar("dispersive_shift", dispersive_shift)
        if dispersive_shift == 0.0:
            raise ValueError("`dispersive_shift` must be nonzero (signed chi in GHz).")
        dispersive_shift = float(dispersive_shift)

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    nominal_qubit_frequency_ge = float(exp.ctx.targets[qubit_label].frequency)
    resonator_frequency = float(exp.ctx.targets[read_label].frequency)
    rabi_param = _core.get_rabi_param(exp, qubit_label)

    n_states = len(states)
    n_amplitudes = amplitude_range.size
    n_offsets = offset_range.size
    n_detunings = detuning_range.size

    iq = np.full(
        (n_states, n_amplitudes, n_offsets, n_detunings), np.nan, dtype=complex
    )
    normalized = np.full((n_states, n_amplitudes, n_offsets, n_detunings), np.nan)
    f0 = np.full((n_states, n_amplitudes, n_offsets), np.nan)
    f0_err = np.full((n_states, n_amplitudes, n_offsets), np.nan)
    gamma = np.full((n_states, n_amplitudes, n_offsets), np.nan)
    r2 = np.full((n_states, n_amplitudes, n_offsets), np.nan)
    fit_valid = np.zeros((n_states, n_amplitudes, n_offsets), dtype=bool)
    fit_results = np.empty((n_states, n_amplitudes, n_offsets), dtype=object)
    raw_measure_results = np.empty(
        (n_states, n_amplitudes, n_offsets, n_detunings), dtype=object
    )
    no_drive_measure_results = np.empty(
        (n_states, n_offsets, n_detunings), dtype=object
    )
    no_drive_iq = np.full((n_states, n_offsets, n_detunings), np.nan, dtype=complex)
    no_drive_normalized = np.full((n_states, n_offsets, n_detunings), np.nan)
    reference_f0 = np.full((n_states, n_offsets), np.nan)
    reference_f0_err = np.full((n_states, n_offsets), np.nan)
    reference_valid = np.zeros((n_states, n_offsets), dtype=bool)
    no_drive_results = np.empty((n_states, n_offsets), dtype=object)

    progress = tqdm(
        total=n_states * n_offsets * (1 + n_amplitudes),
        desc=f"AC Stark spectroscopy : {target}",
    )
    is_first_point = True
    for state_index, state in enumerate(states):
        # No-drive reference: measured once per (state, offset) and reused
        # for every resonator drive amplitude at that offset. The timing is
        # identical to the driven measurements; only the resonator drive is
        # replaced by a Blank of the same duration.
        for offset_index, offset in enumerate(offset_range):
            reference_run = _core.acquire_spectroscopy_trace(
                exp,
                target=target,
                qubit_initial_state=state,  # type: ignore[arg-type]
                qubit_drive_detuning_range=detuning_range,
                qubit_drive_time_offset=float(offset),
                resonator_drive_pulse=Blank(resonator_drive_duration),
                waittime_after_pi_pulse=waittime_after_pi_pulse,
                waittime_before_readout=waittime_before_readout,
                shot_interval=shot_interval,
                n_shots=n_shots,
                expected_capture_count=REFERENCE_CAPTURE_COUNT,
                strict_capture_count=False,
                plot_pulse_schedule=False,
                pulse_schedule_title="",
                reset_awg_and_capunits=(
                    bool(reset_awg_and_capunits) if is_first_point else False
                ),
                progress_description=f"qubit detuning ({state}, no drive)",
            )
            is_first_point = False
            reference_analysis = _analysis.fit_spectroscopy_trace(
                detuning_range=reference_run["detuning_range"],
                iq=reference_run["iq"],
                nominal_frequency=reference_run["nominal_frequency"],
                qubit_initial_state=state,  # type: ignore[arg-type]
                qubit_drive_duration=qubit_drive_duration,
                rabi_param=rabi_param,
            )
            reference_point = (state_index, offset_index)
            no_drive_measure_results[reference_point] = reference_run["measure_results"]
            no_drive_iq[reference_point] = reference_run["iq"]
            no_drive_normalized[reference_point] = reference_analysis["normalized"]
            reference_f0[reference_point] = reference_analysis["f0"]
            reference_f0_err[reference_point] = reference_analysis["f0_err"]
            reference_valid[reference_point] = reference_analysis["valid"]
            no_drive_results[reference_point] = reference_analysis["fit_result"]
            progress.update(1)
        if not np.all(reference_valid[state_index]):
            print(
                f"[WARN] invalid no-drive reference fit for state {state}; "
                "the AC Stark shift is NaN at the affected offsets."
            )
        for amplitude_index, amplitude in enumerate(amplitude_range):
            for offset_index, offset in enumerate(offset_range):
                is_first_point_of_state = amplitude_index == 0 and offset_index == 0
                run_result = _core.acquire_spectroscopy_trace(
                    exp,
                    target=target,
                    qubit_initial_state=state,  # type: ignore[arg-type]
                    qubit_drive_detuning_range=detuning_range,
                    qubit_drive_time_offset=float(offset),
                    resonator_drive_pulse=_resonator_drive_pulse(
                        resonator_drive_amplitude=float(amplitude),
                        resonator_drive_duration=resonator_drive_duration,
                        resonator_drive_ramptime=resonator_drive_ramptime,
                        resonator_drive_detuning=resonator_drive_detuning,
                    ),
                    waittime_after_pi_pulse=waittime_after_pi_pulse,
                    waittime_before_readout=waittime_before_readout,
                    shot_interval=shot_interval,
                    n_shots=n_shots,
                    expected_capture_count=DRIVEN_CAPTURE_COUNT,
                    strict_capture_count=True,
                    plot_pulse_schedule=(
                        plot_pulse_schedule and is_first_point_of_state
                    ),
                    pulse_schedule_title=_pulse_schedule_title(
                        target=target,
                        qubit_initial_state=state,
                        resonator_drive_amplitude=float(amplitude),
                        qubit_drive_time_offset=float(offset),
                    ),
                    reset_awg_and_capunits=False,
                    progress_description=f"qubit detuning ({state})",
                )
                analysis = _analysis.fit_spectroscopy_trace(
                    detuning_range=run_result["detuning_range"],
                    iq=run_result["iq"],
                    nominal_frequency=run_result["nominal_frequency"],
                    qubit_initial_state=state,  # type: ignore[arg-type]
                    qubit_drive_duration=qubit_drive_duration,
                    rabi_param=rabi_param,
                )
                point = (state_index, amplitude_index, offset_index)
                raw_measure_results[point] = run_result["measure_results"]
                iq[point] = run_result["iq"]
                normalized[point] = analysis["normalized"]
                f0[point] = analysis["f0"]
                f0_err[point] = analysis["f0_err"]
                gamma[point] = analysis["gamma"]
                r2[point] = analysis["r2"]
                fit_valid[point] = analysis["valid"]
                fit_results[point] = analysis["fit_result"]
                progress.update(1)
    progress.close()

    derived = _analysis.derive_ac_stark_shift(
        f0=f0,
        f0_err=f0_err,
        fit_valid=fit_valid,
        reference_f0=reference_f0[:, np.newaxis, :],
        reference_f0_err=reference_f0_err[:, np.newaxis, :],
        reference_valid=reference_valid[:, np.newaxis, :],
        dispersive_shift=dispersive_shift,
    )

    data: dict[str, Any] = {
        "target": target,
        "axes": {
            "state": tuple(states),
            "resonator_drive_amplitude_range": amplitude_range,
            "qubit_drive_time_offset_range": offset_range,
            "qubit_drive_detuning_range": detuning_range,
            "qubit_frequency_range": nominal_qubit_frequency_ge + detuning_range,
        },
        "calibration": {
            "nominal_qubit_frequency_ge": nominal_qubit_frequency_ge,
            "no_drive_reference_frequency": reference_f0,
            "no_drive_reference_frequency_error": reference_f0_err,
            "no_drive_reference_fit_valid": reference_valid,
            "resonator_frequency": resonator_frequency,
            "dispersive_shift": dispersive_shift,
        },
        "parameters": {
            "qubit_drive_pulse": qubit_drive_pulse,
            "resonator_drive_duration": resonator_drive_duration,
            "resonator_drive_ramptime": resonator_drive_ramptime,
            "resonator_drive_detuning": resonator_drive_detuning,
            "waittime_after_pi_pulse": waittime_after_pi_pulse,
            "waittime_before_readout": waittime_before_readout,
            "shot_interval": shot_interval,
            "n_shots": n_shots,
        },
        "spectroscopy": {
            "iq": iq,
            "normalized": normalized,
            "no_drive_iq": no_drive_iq,
            "no_drive_normalized": no_drive_normalized,
        },
        "raw": {
            "measure_results": raw_measure_results,
            "no_drive_measure_results": no_drive_measure_results,
        },
        "fit": {
            "f0": f0,
            "f0_err": f0_err,
            "gamma": gamma,
            "r2": r2,
            "valid": fit_valid,
            "result": fit_results,
            "no_drive_result": no_drive_results,
        },
        "derived": derived,
    }

    figures = plot_ac_stark_spectroscopy_flattop_readout_drive(data)
    result = Result(data=data, figures=figures)
    display_ac_stark_spectroscopy_flattop_readout_drive(result, plot=plot)
    return result


def _grid_heatmap(
    *,
    title: str,
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    z_values: NDArray[np.float64],
    z_title: str,
) -> go.Figure:
    fig = viz.make_figure()
    fig.add_heatmap(
        z=z_values,
        x=x_values,
        y=y_values,
        colorscale="Viridis",
        colorbar=dict(title=dict(text=z_title, side="right")),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Qubit drive time offset (ns)",
        yaxis_title="Resonator drive amplitude",
        width=600,
        height=400,
    )
    return fig


def plot_ac_stark_spectroscopy_flattop_readout_drive(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the result figures for an AC Stark spectroscopy dataset.

    Per preparation state this produces up to two spectroscopy heatmaps
    (versus resonator drive amplitude and versus qubit drive time offset,
    with a slice-selection dropdown when the other axis is also swept), a
    shift summary plot (with a photon-number secondary axis when
    ``dispersive_shift`` is available), a shift-versus-drive-power plot
    (x axis: resonator drive amplitude squared, one trace per offset;
    linear in the dispersive regime) when the amplitude axis is swept
    and, for a 2D sweep, shift and photon-number heatmaps over the
    amplitude/offset grid.

    When the qubit drive time offset is the x axis, a secondary top x
    axis shows the probe start measured from the resonator drive end
    (``offset - resonator_drive_duration``); its positive side is the
    ring-down region, useful for reading off when the intracavity photon
    number decays to zero.

    Parameters
    ----------
    data : dict[str, Any]
        The ``Result.data`` payload produced by
        ``ac_stark_spectroscopy_flattop_readout_drive``.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by a stable name including the state label.
    """
    target = data["target"]
    states = data["axes"]["state"]
    amplitude_range = data["axes"]["resonator_drive_amplitude_range"]
    offset_range = data["axes"]["qubit_drive_time_offset_range"]
    frequency_range = data["axes"]["qubit_frequency_range"]
    normalized = data["spectroscopy"]["normalized"]
    f0 = data["fit"]["f0"]
    fit_valid = data["fit"]["valid"]
    ac_stark_shift = data["derived"]["ac_stark_shift"]
    ac_stark_shift_error = data["derived"]["ac_stark_shift_error"]
    photon_number = data["derived"]["photon_number"]
    dispersive_shift = data["calibration"]["dispersive_shift"]
    resonator_drive_duration = data["parameters"]["resonator_drive_duration"]
    # Convert the qubit drive time offset to the probe start measured from
    # the resonator drive end: probe_start - resonator_drive_end = offset -
    # resonator_drive_duration.
    time_from_drive_end_shift = -float(resonator_drive_duration)
    time_from_drive_end_title = "Time from resonator drive end (ns)"

    n_amplitudes = amplitude_range.size
    n_offsets = offset_range.size
    figures: dict[str, go.Figure] = {}

    for state_index, state in enumerate(states):
        # Display-only sign flip for |e> (CKP convention) so both states
        # show the resonance with the same visual polarity. The Lorentzian
        # fit and the stored data use the unflipped normalized signal.
        display_sign = -1.0 if state == "e" else 1.0
        display_signal = display_sign * normalized[state_index]
        if n_amplitudes > 1:
            figures[f"spectroscopy_vs_amplitude_{state}"] = (
                _plotting.spectroscopy_heatmap(
                    target=target,
                    state=state,
                    x_values=amplitude_range,
                    x_title="Resonator drive amplitude",
                    frequency_range=frequency_range,
                    signal_slices=[
                        display_signal[:, offset_index, :]
                        for offset_index in range(n_offsets)
                    ],
                    f0_slices=[
                        f0[state_index, :, offset_index]
                        for offset_index in range(n_offsets)
                    ],
                    valid_slices=[
                        fit_valid[state_index, :, offset_index]
                        for offset_index in range(n_offsets)
                    ],
                    slice_labels=[f"offset={offset:g} ns" for offset in offset_range],
                )
            )
        if n_offsets > 1:
            figures[f"spectroscopy_vs_time_offset_{state}"] = (
                _plotting.spectroscopy_heatmap(
                    target=target,
                    state=state,
                    x_values=offset_range,
                    x_title="Qubit drive time offset (ns)",
                    frequency_range=frequency_range,
                    signal_slices=[
                        display_signal[amplitude_index, :, :]
                        for amplitude_index in range(n_amplitudes)
                    ],
                    f0_slices=[
                        f0[state_index, amplitude_index, :]
                        for amplitude_index in range(n_amplitudes)
                    ],
                    valid_slices=[
                        fit_valid[state_index, amplitude_index, :]
                        for amplitude_index in range(n_amplitudes)
                    ],
                    slice_labels=[
                        f"A_r={amplitude:g}" for amplitude in amplitude_range
                    ],
                )
            )

        if n_amplitudes > 1 and n_offsets > 1:
            figures[f"summary_{state}"] = _plotting.summary_figure(
                title=f"AC Stark shift : {target} : |{state}>",
                x_values=offset_range,
                x_title="Qubit drive time offset (ns)",
                shift_traces=[
                    ac_stark_shift[state_index, amplitude_index, :]
                    for amplitude_index in range(n_amplitudes)
                ],
                shift_error_traces=[
                    ac_stark_shift_error[state_index, amplitude_index, :]
                    for amplitude_index in range(n_amplitudes)
                ],
                trace_names=[f"A_r={amplitude:g}" for amplitude in amplitude_range],
                dispersive_shift=dispersive_shift,
                secondary_x_shift=time_from_drive_end_shift,
                secondary_x_title=time_from_drive_end_title,
            )
            figures[f"shift_heatmap_{state}"] = _grid_heatmap(
                title=f"AC Stark shift (GHz) : {target} : |{state}>",
                x_values=offset_range,
                y_values=amplitude_range,
                z_values=ac_stark_shift[state_index],
                z_title="Shift (GHz)",
            )
            if photon_number is not None:
                figures[f"photon_number_heatmap_{state}"] = _grid_heatmap(
                    title=f"Mean photon number : {target} : |{state}>",
                    x_values=offset_range,
                    y_values=amplitude_range,
                    z_values=photon_number[state_index],
                    z_title="Photon number",
                )
        elif n_amplitudes > 1 or n_offsets > 1:
            if n_offsets > 1:
                x_values = offset_range
                x_title = "Qubit drive time offset (ns)"
                shift_trace = ac_stark_shift[state_index, 0, :]
                shift_error_trace = ac_stark_shift_error[state_index, 0, :]
                secondary_x_shift = time_from_drive_end_shift
                secondary_x_title = time_from_drive_end_title
            else:
                x_values = amplitude_range
                x_title = "Resonator drive amplitude"
                shift_trace = ac_stark_shift[state_index, :, 0]
                shift_error_trace = ac_stark_shift_error[state_index, :, 0]
                secondary_x_shift = None
                secondary_x_title = ""
            figures[f"summary_{state}"] = _plotting.summary_figure(
                title=f"AC Stark shift : {target} : |{state}>",
                x_values=x_values,
                x_title=x_title,
                shift_traces=[shift_trace],
                shift_error_traces=[shift_error_trace],
                trace_names=[f"|{state}>"],
                dispersive_shift=dispersive_shift,
                secondary_x_shift=secondary_x_shift,
                secondary_x_title=secondary_x_title,
            )
        else:
            figures[f"spectrum_{state}"] = _plotting.single_spectrum_figure(
                target=target,
                state=state,
                frequency_range=frequency_range,
                signal=display_signal[0, 0],
                f0=f0[state_index, 0, 0],
                fit_valid=bool(fit_valid[state_index, 0, 0]),
            )

        if n_amplitudes > 1:
            # In the dispersive regime the photon number scales with the
            # drive power, so the shift is linear in amplitude squared.
            if n_offsets > 1:
                power_trace_names = [f"offset={offset:g} ns" for offset in offset_range]
            else:
                power_trace_names = [f"|{state}>"]
            figures[f"summary_vs_amplitude_squared_{state}"] = _plotting.summary_figure(
                title=f"AC Stark shift vs drive power : {target} : |{state}>",
                x_values=amplitude_range**2,
                x_title="Resonator drive amplitude squared",
                shift_traces=[
                    ac_stark_shift[state_index, :, offset_index]
                    for offset_index in range(n_offsets)
                ],
                shift_error_traces=[
                    ac_stark_shift_error[state_index, :, offset_index]
                    for offset_index in range(n_offsets)
                ],
                trace_names=power_trace_names,
                dispersive_shift=dispersive_shift,
            )

    return figures


def display_ac_stark_spectroscopy_flattop_readout_drive(
    result: Result,
    *,
    plot: bool = True,
) -> None:
    """
    Show the figures and print a summary table of an AC Stark result.

    Parameters
    ----------
    result : Result
        Result returned by ``ac_stark_spectroscopy_flattop_readout_drive``.
    plot : bool, optional
        If True, display the stored figures.
    """
    data = result.data
    if plot and result.figures:
        for fig in result.figures.values():
            fig.show()

    states = data["axes"]["state"]
    amplitude_range = data["axes"]["resonator_drive_amplitude_range"]
    offset_range = data["axes"]["qubit_drive_time_offset_range"]
    fit_valid = data["fit"]["valid"]
    r2 = data["fit"]["r2"]
    ac_stark_shift = data["derived"]["ac_stark_shift"]
    ac_stark_shift_error = data["derived"]["ac_stark_shift_error"]
    photon_number = data["derived"]["photon_number"]
    dispersive_shift = data["calibration"]["dispersive_shift"]

    print(f"=== AC Stark spectroscopy : {data['target']} ===")
    print(
        "nominal qubit frequency (ge): "
        f"{data['calibration']['nominal_qubit_frequency_ge']:.6f} GHz"
    )
    qubit_drive_pulse = data["parameters"]["qubit_drive_pulse"]
    print(
        "qubit drive pulse  : "
        f"{qubit_drive_pulse.name} ({qubit_drive_pulse.duration:g} ns, "
        f"max |amplitude| = {np.max(np.abs(qubit_drive_pulse.values)):.4f})"
    )
    if dispersive_shift is not None:
        print(f"dispersive shift   : {1000 * dispersive_shift:+.3f} MHz (chi)")
    header = f"{'state':>5} {'A_r':>10} {'offset (ns)':>12} {'shift (MHz)':>16}"
    if photon_number is not None:
        header += f" {'n_photon':>12}"
    header += f" {'r2':>7} {'valid':>6}"
    print(header)
    for state_index, state in enumerate(states):
        for amplitude_index, amplitude in enumerate(amplitude_range):
            for offset_index, offset in enumerate(offset_range):
                point = (state_index, amplitude_index, offset_index)
                shift = ac_stark_shift[point]
                shift_error = ac_stark_shift_error[point]
                if np.isfinite(shift):
                    shift_text = f"{1000 * shift:+.3f}±{1000 * shift_error:.3f}"
                else:
                    shift_text = "nan"
                row = f"{state:>5} {amplitude:>10g} {offset:>12g} {shift_text:>16}"
                if photon_number is not None:
                    n_value = photon_number[point]
                    row += (
                        f" {n_value:>12.3f}"
                        if np.isfinite(n_value)
                        else f" {'nan':>12}"
                    )
                row += f" {r2[point]:>7.3f} {bool(fit_valid[point])!s:>6}"
                print(row)
