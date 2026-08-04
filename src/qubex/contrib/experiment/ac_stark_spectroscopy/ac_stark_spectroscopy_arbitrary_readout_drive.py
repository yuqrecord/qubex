"""
AC Stark spectroscopy with an arbitrary resonator (readout) drive.

This module measures the AC Stark shift of a qubit induced by a
caller-supplied resonator (readout cavity) drive waveform. The qubit and
the resonator are driven simultaneously, then the resonator is read out
with the standard readout pulse. Use
``ac_stark_spectroscopy_flattop_readout_drive`` when a flat-top drive
parameterized by amplitude / duration / ramp time is enough; use this
variant to play a shaped, measured or otherwise precomputed envelope.

The pulse sequence, the timing conventions and the capture-window layout
are documented in ``_core``; the Lorentzian fit and the shift derivation
are documented in ``_analysis``.

The resonator drive waveform
----------------------------
``resonator_drive_pulse`` is played on the read channel **exactly as
supplied**: this module never scales, detunes or otherwise modifies it.
Anything about the drive is therefore decided outside the experiment,
which is the point of this variant:

- a carrier offset from the calibrated readout frequency is applied by
  the caller, e.g. ``pulse.detuned(0.001)``, or encoded directly in the
  complex samples;
- the drive power is part of the waveform, so it is *not* a sweep axis
  here (sweep it with the flat-top variant, or call this function once
  per waveform).

Any ``Waveform`` works: ``Arbitrary(values)`` for raw complex samples,
a ``PulseArray`` for a composite envelope, or any library pulse.

Sweep axes
----------
``qubit_drive_detuning_range`` is the spectroscopy axis and is always
swept inside one trace. The only outer sweep axis is the qubit drive
time offset; the preparation states are measured as a set.

The AC Stark shift is referenced to a state- and offset-resolved
*no-drive* measurement: for every (state, time offset) a spectroscopy
trace is acquired with a ``Blank`` of the same duration in place of the
resonator drive but otherwise identical timing, and

    shift = f0_driven - f0_no_drive

This removes calibration offsets and offset-dependent effects (e.g. T1
decay of the |e> preparation) from the shift.

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
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam, Result
from qubex.pulse import Blank, PulseSchedule, Waveform

from . import _analysis, _core, _plotting

# A driven schedule normally plays the resonator drive and the readout on
# the read channel (two capture windows), and the no-drive reference plays
# only the readout (one window). A composite waveform may open a different
# number of windows, so a mismatch is reported but not raised: the readout
# is scheduled last, hence its data is always the last capture.
DRIVEN_CAPTURE_COUNT = 2
REFERENCE_CAPTURE_COUNT = 1


def _validate_resonator_drive_pulse(
    name: str,
    pulse: Waveform,
) -> None:
    if not isinstance(pulse, Waveform):
        raise TypeError(
            f"`{name}` must be a Waveform (e.g. Arbitrary, PulseArray or any "
            f"library pulse), got {type(pulse).__name__}."
        )
    duration = float(pulse.duration)
    if duration <= 0:
        raise ValueError(f"`{name}` must have a positive duration, got {duration} ns.")
    _core.validate_time_on_grid(f"{name} duration", duration)
    values = np.asarray(pulse.values)
    if values.ndim != 1:
        raise ValueError(f"`{name}` must sample to a 1D array, got {values.ndim}D.")
    if values.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"`{name}` must contain only finite samples.")
    peak = float(np.max(np.abs(values)))
    if peak > 1.0 + _core.AMPLITUDE_TOLERANCE:
        raise ValueError(
            f"`{name}` exceeds the normalized hardware range: "
            f"max |value| = {peak}, must be <= 1."
        )
    if peak == 0.0:
        raise ValueError(
            f"`{name}` is identically zero. The no-drive reference is acquired "
            "internally, so the supplied drive must be non-zero."
        )


def _pulse_schedule_title(
    *,
    target: str,
    qubit_initial_state: str,
    qubit_drive_time_offset: float,
) -> str:
    return (
        f"AC Stark spectroscopy : {target} : "
        f"|{qubit_initial_state}> "
        f"(arbitrary drive, offset={qubit_drive_time_offset:g} ns)"
    )


def ac_stark_spectroscopy_arbitrary_readout_drive_sequence(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_detuning: float,
    qubit_drive_time_offset: float,
    resonator_drive_pulse: Waveform,
    waittime_after_pi_pulse: float,
    waittime_before_readout: float,
) -> PulseSchedule:
    """
    Build the AC Stark spectroscopy pulse schedule for one sweep point.

    Sequence: state preparation (pi pulse for "e", duration-matched blank
    for "g") followed by ``waittime_after_pi_pulse``, then the supplied
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
    resonator_drive_pulse : Waveform
        Resonator drive, played as supplied on the read channel.
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
        Schedule with the resonator drive and the readout on the read
        channel.
    """
    return _core.build_sequence(
        exp,
        target=target,
        qubit_initial_state=qubit_initial_state,
        qubit_drive_detuning=qubit_drive_detuning,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_pulse=resonator_drive_pulse,
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
    )


def run_ac_stark_spectroscopy_arbitrary_readout_drive(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_detuning_range: ArrayLike,
    qubit_drive_time_offset: float,
    resonator_drive_pulse: Waveform,
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
    resonator_drive_pulse : Waveform
        Resonator drive, played as supplied. Must be non-zero: the
        no-drive reference is acquired by the main entry point.
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
        that were played) and an echo of the timing parameters.
    """
    qubit_drive_duration = float(_core.qubit_drive_waveform(exp, target).duration)
    _core.validate_state("qubit_initial_state", qubit_initial_state)
    _validate_resonator_drive_pulse("resonator_drive_pulse", resonator_drive_pulse)
    _core.validate_common_parameters(
        qubit_drive_duration=qubit_drive_duration,
        qubit_drive_detuning_range=qubit_drive_detuning_range,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_duration=float(resonator_drive_pulse.duration),
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
        shot_interval=shot_interval,
        n_shots=n_shots,
    )

    return _core.acquire_spectroscopy_trace(
        exp,
        target=target,
        qubit_initial_state=qubit_initial_state,
        qubit_drive_detuning_range=qubit_drive_detuning_range,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_pulse=resonator_drive_pulse,
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
        shot_interval=shot_interval,
        n_shots=n_shots,
        expected_capture_count=DRIVEN_CAPTURE_COUNT,
        strict_capture_count=False,
        plot_pulse_schedule=plot_pulse_schedule,
        pulse_schedule_title=_pulse_schedule_title(
            target=target,
            qubit_initial_state=qubit_initial_state,
            qubit_drive_time_offset=qubit_drive_time_offset,
        ),
        reset_awg_and_capunits=(
            bool(reset_awg_and_capunits)
            if reset_awg_and_capunits is not None
            else False
        ),
        progress_description=f"qubit detuning ({qubit_initial_state})",
    )


def analyze_ac_stark_spectroscopy_arbitrary_readout_drive(
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


def ac_stark_spectroscopy_arbitrary_readout_drive(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_states: Collection[Literal["g", "e"]] | str,
    qubit_drive_detuning_range: ArrayLike,
    qubit_drive_time_offset_range: float | ArrayLike,
    resonator_drive_pulse: Waveform,
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
    Run an AC Stark spectroscopy offset sweep with an arbitrary drive.

    For every combination of preparation state and qubit drive time
    offset, a ge spectroscopy trace (versus
    ``qubit_drive_detuning_range``) is acquired and fitted. Both states
    probe the ge transition on the ge control channel ("g": absorption
    dip, "e": stimulated-emission peak after a pi pulse). The AC Stark
    shift is referenced to a no-drive spectroscopy measured once per
    (state, offset) with a ``Blank`` in place of the resonator drive and
    otherwise identical timing: ``shift = f0_driven - f0_no_drive``. With
    ``dispersive_shift`` (signed ge chi, GHz) given, the shift is
    converted to the mean intracavity photon number
    ``n = shift / (2 * chi)``.

    Sweeping the offset while the drive waveform is fixed maps out when
    the qubit sees the cavity photons: the ring-down region (probe after
    the drive ends) shows the shift decaying as the cavity empties.

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
        The only outer sweep axis: qubit drive start relative to the
        resonator drive start (ns). A scalar or a list; negative values
        are allowed.
    resonator_drive_pulse : Waveform
        Resonator drive, played as supplied on the read channel. Its
        amplitude, shape, duration and carrier offset are all decided by
        the caller (see the module docstring); it must be non-zero.
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
        quantities; see the module docstring) and ``figures``. Spectra
        are indexed (state, offset, detuning) and fits (state, offset).
        The ``data["raw"]`` entry holds the unprocessed
        ``MultipleMeasureResult`` objects from ``Experiment.execute``, and
        ``data["parameters"]`` holds the ``qubit_drive_pulse`` and
        ``resonator_drive_pulse`` waveforms that were played; their names,
        durations and samples are read off the waveforms themselves.
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

    _validate_resonator_drive_pulse("resonator_drive_pulse", resonator_drive_pulse)
    resonator_drive_duration = float(resonator_drive_pulse.duration)
    _core.validate_axis("qubit_drive_time_offset_range", qubit_drive_time_offset_range)
    offset_range = np.atleast_1d(np.asarray(qubit_drive_time_offset_range, dtype=float))
    # The probe is the calibrated ge pi pulse, so its duration is fixed by
    # the calibration and only enters the timing constraint. The waveform
    # itself is stored in the result so that the probe actually played can
    # be inspected afterwards.
    qubit_drive_pulse = _core.qubit_drive_waveform(exp, target)
    qubit_drive_duration = float(qubit_drive_pulse.duration)
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
    n_offsets = offset_range.size
    n_detunings = detuning_range.size

    iq = np.full((n_states, n_offsets, n_detunings), np.nan, dtype=complex)
    normalized = np.full((n_states, n_offsets, n_detunings), np.nan)
    f0 = np.full((n_states, n_offsets), np.nan)
    f0_err = np.full((n_states, n_offsets), np.nan)
    gamma = np.full((n_states, n_offsets), np.nan)
    r2 = np.full((n_states, n_offsets), np.nan)
    fit_valid = np.zeros((n_states, n_offsets), dtype=bool)
    fit_results = np.empty((n_states, n_offsets), dtype=object)
    raw_measure_results = np.empty((n_states, n_offsets, n_detunings), dtype=object)
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
        total=2 * n_states * n_offsets,
        desc=f"AC Stark spectroscopy : {target}",
    )
    is_first_point = True
    for state_index, state in enumerate(states):
        # No-drive reference: measured once per (state, offset). The timing
        # is identical to the driven measurements; only the resonator drive
        # is replaced by a Blank of the same duration.
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
        for offset_index, offset in enumerate(offset_range):
            run_result = _core.acquire_spectroscopy_trace(
                exp,
                target=target,
                qubit_initial_state=state,  # type: ignore[arg-type]
                qubit_drive_detuning_range=detuning_range,
                qubit_drive_time_offset=float(offset),
                resonator_drive_pulse=resonator_drive_pulse,
                waittime_after_pi_pulse=waittime_after_pi_pulse,
                waittime_before_readout=waittime_before_readout,
                shot_interval=shot_interval,
                n_shots=n_shots,
                expected_capture_count=DRIVEN_CAPTURE_COUNT,
                strict_capture_count=False,
                plot_pulse_schedule=plot_pulse_schedule and offset_index == 0,
                pulse_schedule_title=_pulse_schedule_title(
                    target=target,
                    qubit_initial_state=state,
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
            point = (state_index, offset_index)
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
        reference_f0=reference_f0,
        reference_f0_err=reference_f0_err,
        reference_valid=reference_valid,
        dispersive_shift=dispersive_shift,
    )

    data: dict[str, Any] = {
        "target": target,
        "axes": {
            "state": tuple(states),
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
            "resonator_drive_pulse": resonator_drive_pulse,
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

    figures = plot_ac_stark_spectroscopy_arbitrary_readout_drive(data)
    result = Result(data=data, figures=figures)
    display_ac_stark_spectroscopy_arbitrary_readout_drive(result, plot=plot)
    return result


def plot_ac_stark_spectroscopy_arbitrary_readout_drive(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the result figures for an arbitrary-drive AC Stark dataset.

    Per preparation state this produces a spectroscopy heatmap versus the
    qubit drive time offset (or a single spectrum when only one offset was
    measured) and, when the offset axis is swept, a shift summary plot
    with a photon-number secondary axis (when ``dispersive_shift`` is
    available). A secondary top x axis shows the probe start measured from
    the resonator drive end (``offset - resonator drive duration``); its
    positive side is the ring-down region, useful for reading off when the
    intracavity photon number decays to zero. One extra figure shows the
    resonator drive envelope that was played.

    Parameters
    ----------
    data : dict[str, Any]
        The ``Result.data`` payload produced by
        ``ac_stark_spectroscopy_arbitrary_readout_drive``.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by a stable name including the state label.
    """
    target = data["target"]
    states = data["axes"]["state"]
    offset_range = data["axes"]["qubit_drive_time_offset_range"]
    frequency_range = data["axes"]["qubit_frequency_range"]
    normalized = data["spectroscopy"]["normalized"]
    f0 = data["fit"]["f0"]
    fit_valid = data["fit"]["valid"]
    ac_stark_shift = data["derived"]["ac_stark_shift"]
    ac_stark_shift_error = data["derived"]["ac_stark_shift_error"]
    dispersive_shift = data["calibration"]["dispersive_shift"]
    resonator_drive_pulse = data["parameters"]["resonator_drive_pulse"]
    # Convert the qubit drive time offset to the probe start measured from
    # the resonator drive end: probe_start - resonator_drive_end = offset -
    # resonator drive duration.
    time_from_drive_end_shift = -float(resonator_drive_pulse.duration)
    time_from_drive_end_title = "Time from resonator drive end (ns)"

    n_offsets = offset_range.size
    figures: dict[str, go.Figure] = {}

    for state_index, state in enumerate(states):
        # Display-only sign flip for |e> (CKP convention) so both states
        # show the resonance with the same visual polarity. The Lorentzian
        # fit and the stored data use the unflipped normalized signal.
        display_sign = -1.0 if state == "e" else 1.0
        display_signal = display_sign * normalized[state_index]
        if n_offsets > 1:
            figures[f"spectroscopy_vs_time_offset_{state}"] = (
                _plotting.spectroscopy_heatmap(
                    target=target,
                    state=state,
                    x_values=offset_range,
                    x_title="Qubit drive time offset (ns)",
                    frequency_range=frequency_range,
                    signal_slices=[display_signal],
                    f0_slices=[f0[state_index]],
                    valid_slices=[fit_valid[state_index]],
                    slice_labels=[f"|{state}>"],
                )
            )
            figures[f"summary_{state}"] = _plotting.summary_figure(
                title=f"AC Stark shift : {target} : |{state}>",
                x_values=offset_range,
                x_title="Qubit drive time offset (ns)",
                shift_traces=[ac_stark_shift[state_index]],
                shift_error_traces=[ac_stark_shift_error[state_index]],
                trace_names=[f"|{state}>"],
                dispersive_shift=dispersive_shift,
                secondary_x_shift=time_from_drive_end_shift,
                secondary_x_title=time_from_drive_end_title,
            )
        else:
            figures[f"spectrum_{state}"] = _plotting.single_spectrum_figure(
                target=target,
                state=state,
                frequency_range=frequency_range,
                signal=display_signal[0],
                f0=f0[state_index, 0],
                fit_valid=bool(fit_valid[state_index, 0]),
            )

    figures["resonator_drive_pulse"] = _plotting.waveform_figure(
        resonator_drive_pulse,
        title=f"Resonator drive : {target}",
    )
    return figures


def display_ac_stark_spectroscopy_arbitrary_readout_drive(
    result: Result,
    *,
    plot: bool = True,
) -> None:
    """
    Show the figures and print a summary table of an AC Stark result.

    Parameters
    ----------
    result : Result
        Result returned by
        ``ac_stark_spectroscopy_arbitrary_readout_drive``.
    plot : bool, optional
        If True, display the stored figures.
    """
    data = result.data
    if plot and result.figures:
        for fig in result.figures.values():
            fig.show()

    states = data["axes"]["state"]
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
    resonator_drive_pulse = data["parameters"]["resonator_drive_pulse"]
    print(
        "resonator drive    : "
        f"{resonator_drive_pulse.name} ({resonator_drive_pulse.duration:g} ns, "
        f"max |amplitude| = {np.max(np.abs(resonator_drive_pulse.values)):.4f})"
    )
    if dispersive_shift is not None:
        print(f"dispersive shift   : {1000 * dispersive_shift:+.3f} MHz (chi)")
    header = f"{'state':>5} {'offset (ns)':>12} {'shift (MHz)':>16}"
    if photon_number is not None:
        header += f" {'n_photon':>12}"
    header += f" {'r2':>7} {'valid':>6}"
    print(header)
    for state_index, state in enumerate(states):
        for offset_index, offset in enumerate(offset_range):
            point = (state_index, offset_index)
            shift = ac_stark_shift[point]
            shift_error = ac_stark_shift_error[point]
            if np.isfinite(shift):
                shift_text = f"{1000 * shift:+.3f}±{1000 * shift_error:.3f}"
            else:
                shift_text = "nan"
            row = f"{state:>5} {offset:>12g} {shift_text:>16}"
            if photon_number is not None:
                n_value = photon_number[point]
                row += f" {n_value:>12.3f}" if np.isfinite(n_value) else f" {'nan':>12}"
            row += f" {r2[point]:>7.3f} {bool(fit_valid[point])!s:>6}"
            print(row)
