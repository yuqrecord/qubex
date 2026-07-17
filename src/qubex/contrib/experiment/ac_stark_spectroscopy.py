"""
AC Stark spectroscopy utilities.

This module measures the AC Stark shift of a qubit induced by a resonator
(readout cavity) drive. The qubit and the resonator are driven
simultaneously, then the resonator is read out with the standard readout
pulse.

Pulse sequence
--------------
::

    qubit     : [pi or blank][wait] | [Blank(max(0, +offset))][qubit drive]
    resonator : [     blank      ] | [Blank(max(0, -offset))][resonator drive][Blank(waittime_before_readout)][readout]

- ``qubit_drive_time_offset = (qubit drive start) - (resonator drive start)``.
  Negative values start the qubit drive before the resonator drive.
- ``waittime_after_pi_pulse`` is inserted between the state-preparation
  slot and the drive section. It is applied for both preparation states
  (the "g" schedule reserves the same slot with blanks) so that the
  "g" and "e" schedules keep identical timing.
- The readout pulse always starts at
  ``(resonator drive end) + waittime_before_readout``, independent of
  the qubit drive offset, so the measurement conditions are constant
  across an offset sweep. ``waittime_before_readout`` ends at the
  beginning of the readout pulse object (which includes the calibrated
  readout pre-margin).
- The qubit drive must end no later than the readout start; this is
  validated and rejected otherwise.

Measurement and analysis
------------------------
``qubit_drive_detuning_range`` is always swept. For each sweep point the
schedule is executed with ``Experiment.execute`` using shot averaging
(``mode="avg"``) and hardware time integration
(``time_integration=True``). Any pulse on the read channel produces a
capture window, so the resonator drive produces capture #1 (discarded)
and the readout produces capture #2 (the data); the capture count is
validated at runtime.

The readout signal versus qubit drive frequency is fitted with a
Lorentzian to extract the (shifted) resonance ``f0``. The probed
transition and the shift reference depend on the preparation state:

- state "g": the spectroscopy drive is applied on the ge control channel
  and ``shift = f0 - f_ge_calibrated``
- state "e": the spectroscopy drive is applied on the ef control channel
  and ``shift = f0 - f_ef_calibrated``
  (``Qubit.control_frequency_ef``)

The resonator drive amplitude must be positive; a zero-amplitude
reference point is not supported.

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
For state "e" the measured shift is that of the ef transition, whose
per-photon shift coefficient generally differs from the ge one; supply
the chi appropriate for the probed transition.

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
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam, Result
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.pulse import Blank, FlatTop, PulseSchedule, Waveform

STATE_LABELS = ("g", "e")
MIN_QUBIT_DETUNING_POINTS = 5
# FWHM of the power spectrum of a flat-top probe of duration D is ~0.443/D.
LORENTZIAN_TIME_BANDWIDTH = 0.443
FIT_R2_THRESHOLD = 0.5
TIME_GRID_TOLERANCE = 1e-6


def _sampling_period() -> float:
    return float(Waveform.SAMPLING_PERIOD)


def _validate_finite_scalar(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"`{name}` must be a real number, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value!r}.")


def _validate_time_on_grid(name: str, value: float) -> None:
    _validate_finite_scalar(name, value)
    dt = _sampling_period()
    if abs(value - round(value / dt) * dt) > TIME_GRID_TOLERANCE:
        raise ValueError(
            f"`{name}` ({value} ns) must be a multiple of the {dt} ns sampling period."
        )


def _validate_amplitude(name: str, value: float, *, allow_zero: bool) -> None:
    _validate_finite_scalar(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be within [0, 1], got {value}.")
    if not allow_zero and value == 0.0:
        raise ValueError(f"`{name}` must be positive, got {value}.")


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


def _validate_state(name: str, value: str) -> None:
    if value not in STATE_LABELS:
        raise ValueError(f"`{name}` must be one of {STATE_LABELS}, got {value!r}.")


def _validate_axis(name: str, values: float | ArrayLike) -> None:
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a scalar or a 1D sequence.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.unique(array).size != array.size:
        raise ValueError(f"`{name}` must not contain duplicate values.")


def _validate_n_shots(n_shots: int) -> None:
    if isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer)):
        raise TypeError(f"`n_shots` must be an integer, got {n_shots!r}.")
    if n_shots <= 0:
        raise ValueError(f"`n_shots` must be positive, got {n_shots}.")


def _validate_timing_constraint(
    *,
    qubit_drive_duration: float,
    qubit_drive_time_offset: float,
    resonator_drive_duration: float,
    waittime_before_readout: float,
) -> None:
    qubit_start = max(0.0, qubit_drive_time_offset)
    resonator_start = max(0.0, -qubit_drive_time_offset)
    qubit_end = qubit_start + qubit_drive_duration
    readout_start = resonator_start + resonator_drive_duration + waittime_before_readout
    if qubit_end > readout_start + TIME_GRID_TOLERANCE:
        raise ValueError(
            "The qubit drive must end no later than the readout start: "
            f"qubit drive ends at {qubit_end} ns but the readout starts at "
            f"{readout_start} ns for qubit_drive_time_offset="
            f"{qubit_drive_time_offset} ns. Reduce the offset or the qubit "
            "drive duration, or increase waittime_before_readout."
        )


def _validate_run_parameters(
    *,
    qubit_initial_state: str,
    qubit_drive_amplitude: float,
    qubit_drive_duration: float,
    qubit_drive_ramptime: float,
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
) -> None:
    _validate_state("qubit_initial_state", qubit_initial_state)
    _validate_amplitude("qubit_drive_amplitude", qubit_drive_amplitude, allow_zero=True)
    _validate_amplitude(
        "resonator_drive_amplitude", resonator_drive_amplitude, allow_zero=False
    )
    _validate_time_on_grid("qubit_drive_duration", qubit_drive_duration)
    _validate_time_on_grid("qubit_drive_ramptime", qubit_drive_ramptime)
    _validate_time_on_grid("resonator_drive_duration", resonator_drive_duration)
    _validate_time_on_grid("resonator_drive_ramptime", resonator_drive_ramptime)
    _validate_time_on_grid("qubit_drive_time_offset", qubit_drive_time_offset)
    _validate_time_on_grid("waittime_after_pi_pulse", waittime_after_pi_pulse)
    _validate_time_on_grid("waittime_before_readout", waittime_before_readout)
    _validate_flat_top(
        "qubit_drive",
        duration=qubit_drive_duration,
        ramptime=qubit_drive_ramptime,
    )
    _validate_flat_top(
        "resonator_drive",
        duration=resonator_drive_duration,
        ramptime=resonator_drive_ramptime,
    )
    if waittime_after_pi_pulse < 0:
        raise ValueError(
            "`waittime_after_pi_pulse` must be nonnegative, "
            f"got {waittime_after_pi_pulse}."
        )
    if waittime_before_readout < 0:
        raise ValueError(
            "`waittime_before_readout` must be nonnegative, "
            f"got {waittime_before_readout}."
        )
    _validate_timing_constraint(
        qubit_drive_duration=qubit_drive_duration,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_duration=resonator_drive_duration,
        waittime_before_readout=waittime_before_readout,
    )
    _validate_finite_scalar("resonator_drive_detuning", resonator_drive_detuning)
    _validate_finite_scalar("shot_interval", shot_interval)
    if shot_interval <= 0:
        raise ValueError(f"`shot_interval` must be positive, got {shot_interval}.")
    _validate_n_shots(n_shots)

    _validate_axis("qubit_drive_detuning_range", qubit_drive_detuning_range)
    n_detuning_points = np.atleast_1d(
        np.asarray(qubit_drive_detuning_range, dtype=float)
    ).size
    if n_detuning_points < MIN_QUBIT_DETUNING_POINTS:
        raise ValueError(
            "`qubit_drive_detuning_range` must be a sweep with at least "
            f"{MIN_QUBIT_DETUNING_POINTS} distinct points to fit a Lorentzian, "
            f"got {n_detuning_points} point(s). The AC Stark shift cannot be "
            "measured without sweeping the qubit drive detuning."
        )


def ac_stark_spectroscopy_sequence(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_amplitude: float,
    qubit_drive_duration: float,
    qubit_drive_ramptime: float,
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
    for "g") followed by ``waittime_after_pi_pulse``, then a resonator
    drive and a qubit drive with relative time offset, then the standard
    readout starting at ``(resonator drive end) + waittime_before_readout``.

    Parameters
    ----------
    exp : Experiment
        Experiment object providing pulses and hardware context.
    target : str
        Target qubit name or label.
    qubit_initial_state : {"g", "e"}
        Qubit state prepared before the drives. "e" uses the calibrated
        DRAG pi pulse when available (``exp.pulse.x180``).
    qubit_drive_amplitude : float
        Amplitude of the qubit spectroscopy drive.
    qubit_drive_duration : float
        Duration of the qubit drive in ns.
    qubit_drive_ramptime : float
        Ramp time (flat-top tau) of the qubit drive in ns.
    qubit_drive_detuning : float
        Detuning of the qubit drive from the reference transition
        frequency (GHz): the calibrated ge frequency for state "g", the
        calibrated ef frequency for state "e". For state "e" the drive is
        applied on the EF control channel.
    qubit_drive_time_offset : float
        Start time of the qubit drive relative to the resonator drive
        start (ns). Negative values start the qubit drive first.
    resonator_drive_amplitude : float
        Amplitude of the resonator drive.
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
    qubit = exp.ctx.resolve_qubit_label(target)
    resonator = exp.ctx.resolve_read_label(target)
    if qubit_initial_state == "e":
        drive_label = exp.ctx.resolve_ef_label(target)
    else:
        drive_label = qubit

    qubit_delay = max(0.0, qubit_drive_time_offset)
    resonator_delay = max(0.0, -qubit_drive_time_offset)

    qubit_drive_pulse = FlatTop(
        duration=qubit_drive_duration,
        amplitude=qubit_drive_amplitude,
        tau=qubit_drive_ramptime,
    ).detuned(qubit_drive_detuning)
    resonator_drive_pulse = FlatTop(
        duration=resonator_drive_duration,
        amplitude=resonator_drive_amplitude,
        tau=resonator_drive_ramptime,
    ).detuned(resonator_drive_detuning)
    pi_pulse = exp.pulse.x180(qubit)

    with PulseSchedule() as ps:
        if qubit_initial_state == "e":
            ps.add(qubit, pi_pulse)
        else:
            ps.add(qubit, Blank(pi_pulse.duration))
        if waittime_after_pi_pulse > 0:
            ps.add(qubit, Blank(waittime_after_pi_pulse))
        ps.barrier()
        if resonator_delay > 0:
            ps.add(resonator, Blank(resonator_delay))
        ps.add(resonator, resonator_drive_pulse)
        if waittime_before_readout > 0:
            ps.add(resonator, Blank(waittime_before_readout))
        ps.add(resonator, exp.pulse.readout(target))
        if qubit_delay > 0:
            ps.add(drive_label, Blank(qubit_delay))
        ps.add(drive_label, qubit_drive_pulse)
    return ps


def _extract_readout_iq(
    measure_result: MultipleMeasureResult,
    *,
    read_label: str,
    target: str,
) -> complex:
    available = list(measure_result.data)
    if read_label in available:
        data_target = read_label
    elif target in available:
        data_target = target
    else:
        raise KeyError(
            f"Neither {read_label=} nor {target=} exists in the measurement "
            f"result. Available targets: {available}"
        )
    captures = measure_result.data[data_target]
    if len(captures) != 2:
        raise RuntimeError(
            f"Expected exactly 2 capture windows for {data_target} "
            f"(resonator drive + readout), got {len(captures)}. The second "
            "capture is the readout data; a different count indicates that "
            "the schedule was compiled differently than assumed."
        )
    return complex(captures[1].kerneled)


def _get_rabi_param(exp: Experiment, target: str) -> RabiParam:
    rabi_param = exp.pulse.ge_rabi_params.get(target)
    if rabi_param is None:
        rabi_param = exp.pulse.rabi_params.get(target)
    if rabi_param is None:
        raise ValueError(f"`{target}` does not have a RabiParam.")
    return rabi_param


def run_ac_stark_spectroscopy(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_amplitude: float,
    qubit_drive_duration: float,
    qubit_drive_ramptime: float,
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
    Acquire one qubit spectroscopy trace for a single parameter set.

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
        Qubit state prepared before the drives. For "e" the spectroscopy
        drive probes the ef transition (applied on the ef control
        channel).
    qubit_drive_amplitude : float
        Amplitude of the qubit spectroscopy drive.
    qubit_drive_duration : float
        Duration of the qubit drive in ns.
    qubit_drive_ramptime : float
        Ramp time of the qubit drive in ns.
    qubit_drive_detuning_range : ArrayLike
        Sweep values of the qubit drive detuning from the state-dependent
        reference transition frequency (GHz). At least 5 distinct points
        are required.
    qubit_drive_time_offset : float
        Qubit drive start time relative to the resonator drive start (ns).
    resonator_drive_amplitude : float
        Amplitude of the resonator drive.
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
        ``frequency_range``, ``iq``, labels, ``reference_frequency``
        (the calibrated frequency of the probed transition), and an echo
        of the drive parameters.
    """
    _validate_run_parameters(
        qubit_initial_state=qubit_initial_state,
        qubit_drive_amplitude=qubit_drive_amplitude,
        qubit_drive_duration=qubit_drive_duration,
        qubit_drive_ramptime=qubit_drive_ramptime,
        qubit_drive_detuning_range=qubit_drive_detuning_range,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_amplitude=resonator_drive_amplitude,
        resonator_drive_duration=resonator_drive_duration,
        resonator_drive_ramptime=resonator_drive_ramptime,
        resonator_drive_detuning=resonator_drive_detuning,
        waittime_after_pi_pulse=waittime_after_pi_pulse,
        waittime_before_readout=waittime_before_readout,
        shot_interval=shot_interval,
        n_shots=n_shots,
    )
    detuning_range = np.atleast_1d(np.asarray(qubit_drive_detuning_range, dtype=float))
    reset_first = (
        bool(reset_awg_and_capunits) if reset_awg_and_capunits is not None else False
    )

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    if qubit_initial_state == "e":
        reference_frequency = float(exp.ctx.qubits[qubit_label].control_frequency_ef)
    else:
        reference_frequency = float(exp.ctx.targets[qubit_label].frequency)
    resonator_frequency = float(exp.ctx.targets[read_label].frequency)

    iq_values: list[complex] = []
    for index, detuning in enumerate(
        tqdm(
            detuning_range,
            desc=f"qubit detuning ({qubit_initial_state})",
            leave=False,
        )
    ):
        schedule = ac_stark_spectroscopy_sequence(
            exp,
            target=target,
            qubit_initial_state=qubit_initial_state,
            qubit_drive_amplitude=qubit_drive_amplitude,
            qubit_drive_duration=qubit_drive_duration,
            qubit_drive_ramptime=qubit_drive_ramptime,
            qubit_drive_detuning=float(detuning),
            qubit_drive_time_offset=qubit_drive_time_offset,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            resonator_drive_ramptime=resonator_drive_ramptime,
            resonator_drive_detuning=resonator_drive_detuning,
            waittime_after_pi_pulse=waittime_after_pi_pulse,
            waittime_before_readout=waittime_before_readout,
        )
        if plot_pulse_schedule and index == 0:
            schedule.plot(
                title=(
                    f"AC Stark spectroscopy : {target} : "
                    f"|{qubit_initial_state}> "
                    f"(A_r={resonator_drive_amplitude:g}, "
                    f"offset={qubit_drive_time_offset:g} ns)"
                ),
            )
        measure_result = exp.execute(
            schedule=schedule,
            mode="avg",
            time_integration=True,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=reset_first and index == 0,
        )
        iq_values.append(
            _extract_readout_iq(
                measure_result,
                read_label=read_label,
                target=target,
            )
        )

    return {
        "target": target,
        "qubit_label": qubit_label,
        "read_label": read_label,
        "qubit_initial_state": qubit_initial_state,
        "reference_frequency": reference_frequency,
        "resonator_frequency": resonator_frequency,
        "detuning_range": detuning_range,
        "frequency_range": reference_frequency + detuning_range,
        "iq": np.asarray(iq_values, dtype=complex),
        "qubit_drive_amplitude": qubit_drive_amplitude,
        "qubit_drive_duration": qubit_drive_duration,
        "qubit_drive_ramptime": qubit_drive_ramptime,
        "qubit_drive_time_offset": qubit_drive_time_offset,
        "resonator_drive_amplitude": resonator_drive_amplitude,
        "resonator_drive_duration": resonator_drive_duration,
        "resonator_drive_ramptime": resonator_drive_ramptime,
        "resonator_drive_detuning": resonator_drive_detuning,
        "waittime_after_pi_pulse": waittime_after_pi_pulse,
        "waittime_before_readout": waittime_before_readout,
        "shot_interval": shot_interval,
        "n_shots": n_shots,
    }


def _smooth_for_initial_guess(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(signal, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _check_lorentzian_fit(
    fit_result: Any,
    *,
    x: NDArray[np.float64],
) -> tuple[bool, str]:
    if fit_result is None:
        return False, "fit raised an exception"
    if fit_result.status is not FitStatus.SUCCESS:
        return False, "fit did not converge"
    f0 = float(fit_result.get("f0", np.nan))
    f0_err = float(fit_result.get("f0_err", np.nan))
    gamma = float(fit_result.get("gamma", np.nan))
    r2 = float(fit_result.get("r2", np.nan))
    if not (
        np.isfinite(f0)
        and np.isfinite(f0_err)
        and np.isfinite(gamma)
        and np.isfinite(r2)
    ):
        return False, f"non-finite fit parameters (f0={f0}, err={f0_err}, r2={r2})"
    span = float(np.max(x) - np.min(x))
    spacing = float(np.median(np.diff(x)))
    if r2 <= FIT_R2_THRESHOLD:
        return False, f"poor fit quality (r2={r2:.3f})"
    if not gamma < span:
        return False, f"linewidth exceeds the sweep span (gamma={gamma:.6g} GHz)"
    if f0 < np.min(x) + spacing or f0 > np.max(x) - spacing:
        return False, f"fitted center is at the sweep edge (f0={f0:.6f} GHz)"
    if f0_err > span:
        return False, f"fitted center is unconstrained (f0_err={f0_err:.6g} GHz)"
    return True, ""


def analyze_ac_stark_spectroscopy(
    *,
    detuning_range: ArrayLike,
    iq: ArrayLike,
    reference_frequency: float,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_duration: float,
    rabi_param: RabiParam,
) -> dict[str, Any]:
    """
    Fit one spectroscopy trace and extract the AC Stark shift.

    The complex IQ trace is normalized with the calibrated Rabi
    parameters, then fitted with a Lorentzian over the absolute qubit
    drive frequency. The fit is retried with the opposite line polarity
    when the first attempt does not pass the quality gates.

    Parameters
    ----------
    detuning_range : ArrayLike
        Qubit drive detuning sweep (GHz), in acquisition order.
    iq : ArrayLike
        Complex readout IQ values for each detuning point.
    reference_frequency : float
        Reference transition frequency (GHz): the calibrated ge frequency
        for state "g", the calibrated ef frequency for state "e". Used as
        the frequency axis origin and as the shift reference.
    qubit_initial_state : {"g", "e"}
        Prepared qubit state (recorded in the output).
    qubit_drive_duration : float
        Qubit drive duration in ns, used for the linewidth initial guess.
    rabi_param : RabiParam
        Calibrated Rabi parameters used to normalize the IQ data.

    Returns
    -------
    dict[str, Any]
        Fit results with keys ``normalized``, ``fit_result``, ``f0``,
        ``f0_err``, ``gamma``, ``r2``, ``valid`` and ``ac_stark_shift``
        (calibrated reference; NaN when the fit is invalid).
    """
    detuning_range = np.asarray(detuning_range, dtype=float)
    iq_array = np.asarray(iq, dtype=complex)
    if detuning_range.shape != iq_array.shape:
        raise ValueError(
            "`detuning_range` and `iq` must have the same shape, got "
            f"{detuning_range.shape} and {iq_array.shape}."
        )

    normalized = np.asarray(rabi_param.normalize(iq_array), dtype=float)

    order = np.argsort(detuning_range)
    x = reference_frequency + detuning_range[order]
    y = normalized[order]

    smoothed = _smooth_for_initial_guess(y)
    n_edge = max(1, y.size // 10)
    baseline = float(np.mean(np.concatenate([smoothed[:n_edge], smoothed[-n_edge:]])))
    deviations = smoothed - baseline
    index_min = int(np.argmin(deviations))
    index_max = int(np.argmax(deviations))
    if abs(deviations[index_min]) >= abs(deviations[index_max]):
        extremum_indices = [index_min, index_max]
    else:
        extremum_indices = [index_max, index_min]

    span = float(np.max(x) - np.min(x))
    spacing = float(np.median(np.diff(x)))
    gamma_guess = LORENTZIAN_TIME_BANDWIDTH / qubit_drive_duration
    gamma_guess = float(np.clip(gamma_guess, spacing, span / 2))
    amplitude_scale = 2 * float(np.max(y) - np.min(y)) + 1e-12

    fit_result = None
    valid = False
    reason = "no fit attempted"
    for extremum_index in extremum_indices:
        amplitude_guess = float(deviations[extremum_index])
        if amplitude_guess == 0.0:
            continue
        if amplitude_guess < 0:
            amplitude_bounds = (-amplitude_scale, 0.0)
        else:
            amplitude_bounds = (0.0, amplitude_scale)
        p0 = (amplitude_guess, float(x[extremum_index]), gamma_guess, baseline)
        bounds = (
            (amplitude_bounds[0], float(np.min(x)), 0.0, -np.inf),
            (amplitude_bounds[1], float(np.max(x)), np.inf, np.inf),
        )
        try:
            candidate = fitting.fit_lorentzian(
                x=x,
                y=y,
                p0=p0,
                bounds=bounds,
                plot=False,
            )
        except Exception as exc:
            candidate = None
            reason = f"fit raised an exception ({exc})"
        if candidate is not None:
            candidate_valid, candidate_reason = _check_lorentzian_fit(candidate, x=x)
            if fit_result is None or candidate_valid:
                fit_result = candidate
                valid = candidate_valid
                reason = candidate_reason
            if candidate_valid:
                break

    if fit_result is not None:
        f0 = float(fit_result.get("f0", np.nan))
        f0_err = float(fit_result.get("f0_err", np.nan))
        gamma = float(fit_result.get("gamma", np.nan))
        r2 = float(fit_result.get("r2", np.nan))
    else:
        f0 = f0_err = gamma = r2 = np.nan
    if not valid:
        print(f"[WARN] invalid Lorentzian fit (state={qubit_initial_state}): {reason}")

    return {
        "detuning_range": detuning_range,
        "frequency_range": reference_frequency + detuning_range,
        "normalized": normalized,
        "fit_result": fit_result,
        "f0": f0,
        "f0_err": f0_err,
        "gamma": gamma,
        "r2": r2,
        "valid": valid,
        "ac_stark_shift": f0 - reference_frequency if valid else np.nan,
    }


def ac_stark_spectroscopy(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_states: Collection[Literal["g", "e"]] | str,
    qubit_drive_amplitude: float,
    qubit_drive_duration: float,
    qubit_drive_ramptime: float,
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
    and qubit drive time offset, a qubit spectroscopy trace (versus
    ``qubit_drive_detuning_range``) is acquired and fitted. The AC Stark shift
    is referenced to the calibrated ge transition frequency for state "g"
    and to the calibrated ef transition frequency for state "e" (the
    spectroscopy drive is applied on the corresponding control channel).
    With ``dispersive_shift``
    (signed chi, GHz) given, the shift is converted to the mean
    intracavity photon number ``n = shift / (2 * chi)``.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    qubit_initial_states : Collection[{"g", "e"}] | str
        Preparation states to measure, e.g. ``["g"]`` or ``["g", "e"]``.
        State "e" probes the ef transition.
    qubit_drive_amplitude : float
        Amplitude of the qubit spectroscopy drive.
    qubit_drive_duration : float
        Duration of the qubit drive in ns.
    qubit_drive_ramptime : float
        Ramp time (flat-top tau) of the qubit drive in ns.
    qubit_drive_detuning_range : ArrayLike
        Sweep values of the qubit drive detuning from the state-dependent
        reference transition frequency (GHz). Always swept; at least
        5 distinct points are required.
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
        If True, plot the pulse schedule of the first acquired point of
        each preparation state.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    Result
        Result with ``data`` (axes, raw spectra, fits and derived
        quantities; see the module docstring) and ``figures``.
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
        _validate_state("qubit_initial_states", state)

    _validate_axis("resonator_drive_amplitude_range", resonator_drive_amplitude_range)
    _validate_axis("qubit_drive_time_offset_range", qubit_drive_time_offset_range)
    amplitude_range = np.atleast_1d(
        np.asarray(resonator_drive_amplitude_range, dtype=float)
    )
    offset_range = np.atleast_1d(np.asarray(qubit_drive_time_offset_range, dtype=float))
    for amplitude in amplitude_range:
        _validate_amplitude(
            "resonator_drive_amplitude_range", float(amplitude), allow_zero=False
        )
    for offset in offset_range:
        _validate_time_on_grid("qubit_drive_time_offset_range", float(offset))
        _validate_timing_constraint(
            qubit_drive_duration=qubit_drive_duration,
            qubit_drive_time_offset=float(offset),
            resonator_drive_duration=resonator_drive_duration,
            waittime_before_readout=waittime_before_readout,
        )
    detuning_range = np.atleast_1d(np.asarray(qubit_drive_detuning_range, dtype=float))
    if dispersive_shift is not None:
        _validate_finite_scalar("dispersive_shift", dispersive_shift)
        if dispersive_shift == 0.0:
            raise ValueError("`dispersive_shift` must be nonzero (signed chi in GHz).")
        dispersive_shift = float(dispersive_shift)

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    qubit_frequency_ge = float(exp.ctx.targets[qubit_label].frequency)
    qubit_frequency_ef = float(exp.ctx.qubits[qubit_label].control_frequency_ef)
    reference_frequencies = np.array(
        [qubit_frequency_ef if state == "e" else qubit_frequency_ge for state in states]
    )
    resonator_frequency = float(exp.ctx.targets[read_label].frequency)
    rabi_param = _get_rabi_param(exp, qubit_label)

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

    progress = tqdm(
        total=n_states * n_amplitudes * n_offsets,
        desc=f"AC Stark spectroscopy : {target}",
    )
    is_first_point = True
    for state_index, state in enumerate(states):
        for amplitude_index, amplitude in enumerate(amplitude_range):
            for offset_index, offset in enumerate(offset_range):
                is_first_point_of_state = amplitude_index == 0 and offset_index == 0
                run_result = run_ac_stark_spectroscopy(
                    exp,
                    target=target,
                    qubit_initial_state=state,  # type: ignore[arg-type]
                    qubit_drive_amplitude=qubit_drive_amplitude,
                    qubit_drive_duration=qubit_drive_duration,
                    qubit_drive_ramptime=qubit_drive_ramptime,
                    qubit_drive_detuning_range=detuning_range,
                    qubit_drive_time_offset=float(offset),
                    resonator_drive_amplitude=float(amplitude),
                    resonator_drive_duration=resonator_drive_duration,
                    resonator_drive_ramptime=resonator_drive_ramptime,
                    resonator_drive_detuning=resonator_drive_detuning,
                    waittime_after_pi_pulse=waittime_after_pi_pulse,
                    waittime_before_readout=waittime_before_readout,
                    shot_interval=shot_interval,
                    n_shots=n_shots,
                    plot_pulse_schedule=plot_pulse_schedule and is_first_point_of_state,
                    reset_awg_and_capunits=(
                        reset_awg_and_capunits if is_first_point else False
                    ),
                )
                is_first_point = False
                analysis = analyze_ac_stark_spectroscopy(
                    detuning_range=run_result["detuning_range"],
                    iq=run_result["iq"],
                    reference_frequency=run_result["reference_frequency"],
                    qubit_initial_state=state,  # type: ignore[arg-type]
                    qubit_drive_duration=qubit_drive_duration,
                    rabi_param=rabi_param,
                )
                point = (state_index, amplitude_index, offset_index)
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

    ac_stark_shift = np.where(
        fit_valid, f0 - reference_frequencies[:, np.newaxis, np.newaxis], np.nan
    )
    ac_stark_shift_error = np.where(fit_valid, f0_err, np.nan)

    if dispersive_shift is not None:
        photon_number = ac_stark_shift / (2 * dispersive_shift)
        photon_number_error = ac_stark_shift_error / (2 * abs(dispersive_shift))
        photon_number_valid = (
            fit_valid
            & np.isfinite(photon_number)
            & ~(photon_number + 3 * photon_number_error < 0)
        )
        if np.any(fit_valid & np.isfinite(photon_number) & ~photon_number_valid):
            print(
                "[WARN] some inferred photon numbers are significantly "
                "negative; check the sign convention of `dispersive_shift` "
                "(signed chi with delta_f = 2 * chi * n)."
            )
    else:
        photon_number = None
        photon_number_error = None
        photon_number_valid = None

    data: dict[str, Any] = {
        "target": target,
        "axes": {
            "state": tuple(states),
            "resonator_drive_amplitude_range": amplitude_range,
            "qubit_drive_time_offset_range": offset_range,
            "qubit_drive_detuning_range": detuning_range,
            "qubit_frequency_range": (
                reference_frequencies[:, np.newaxis] + detuning_range[np.newaxis, :]
            ),
        },
        "calibration": {
            "qubit_frequency_ge": qubit_frequency_ge,
            "qubit_frequency_ef": qubit_frequency_ef,
            "reference_frequency": reference_frequencies,
            "resonator_frequency": resonator_frequency,
            "dispersive_shift": dispersive_shift,
        },
        "parameters": {
            "qubit_drive_amplitude": qubit_drive_amplitude,
            "qubit_drive_duration": qubit_drive_duration,
            "qubit_drive_ramptime": qubit_drive_ramptime,
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
        },
        "fit": {
            "f0": f0,
            "f0_err": f0_err,
            "gamma": gamma,
            "r2": r2,
            "valid": fit_valid,
            "result": fit_results,
        },
        "derived": {
            "ac_stark_shift": ac_stark_shift,
            "ac_stark_shift_error": ac_stark_shift_error,
            "photon_number": photon_number,
            "photon_number_error": photon_number_error,
            "photon_number_valid": photon_number_valid,
        },
    }

    figures = plot_ac_stark_spectroscopy(data)
    result = Result(data=data, figures=figures)
    display_ac_stark_spectroscopy(result, plot=plot)
    return result


def _add_slice_dropdown(
    fig: go.Figure,
    *,
    labels: list[str],
    traces_per_slice: int,
) -> None:
    n_slices = len(labels)
    if n_slices <= 1:
        return
    buttons = []
    for slice_index, label in enumerate(labels):
        visible = [False] * (n_slices * traces_per_slice)
        for trace_offset in range(traces_per_slice):
            visible[slice_index * traces_per_slice + trace_offset] = True
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visible}],
            )
        )
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ]
    )


def _spectroscopy_heatmap(
    *,
    target: str,
    state: str,
    x_values: NDArray[np.float64],
    x_title: str,
    frequency_range: NDArray[np.float64],
    signal_slices: list[NDArray[np.float64]],
    f0_slices: list[NDArray[np.float64]],
    valid_slices: list[NDArray[np.bool_]],
    slice_labels: list[str],
) -> go.Figure:
    order = np.argsort(frequency_range)
    fig = viz.make_figure()
    for slice_index, signal in enumerate(signal_slices):
        visible = slice_index == 0
        fig.add_heatmap(
            z=signal[:, order].T,
            x=x_values,
            y=frequency_range[order],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Signal", side="right")),
            visible=visible,
        )
        valid = valid_slices[slice_index]
        fig.add_scatter(
            x=x_values[valid],
            y=f0_slices[slice_index][valid],
            mode="markers",
            name="Lorentzian fit f0",
            marker=dict(color="white", size=6, line=dict(color="black", width=1)),
            visible=visible,
        )
    _add_slice_dropdown(fig, labels=slice_labels, traces_per_slice=2)
    fig.update_layout(
        title=f"AC Stark spectroscopy : {target} : |{state}>",
        xaxis_title=x_title,
        yaxis_title="Qubit drive frequency (GHz)",
        width=600,
        height=400,
    )
    return fig


def _summary_figure(
    *,
    target: str,
    state: str,
    x_values: NDArray[np.float64],
    x_title: str,
    shift_traces: list[NDArray[np.float64]],
    shift_error_traces: list[NDArray[np.float64]],
    trace_names: list[str],
    dispersive_shift: float | None,
) -> go.Figure:
    fig = viz.make_figure()
    for shift, shift_error, name in zip(
        shift_traces, shift_error_traces, trace_names, strict=True
    ):
        fig.add_scatter(
            x=x_values,
            y=shift,
            error_y=dict(type="data", array=shift_error, visible=True),
            mode="lines+markers",
            name=name,
        )
    fig.update_layout(
        title=f"AC Stark shift : {target} : |{state}>",
        xaxis_title=x_title,
        yaxis_title="AC Stark shift (GHz)",
        width=600,
        height=400,
        showlegend=True,
    )

    finite_values = []
    for shift, shift_error in zip(shift_traces, shift_error_traces, strict=True):
        error = np.where(np.isfinite(shift_error), shift_error, 0.0)
        finite_values.append(shift - error)
        finite_values.append(shift + error)
    stacked = np.concatenate(finite_values)
    stacked = stacked[np.isfinite(stacked)]
    if dispersive_shift is not None and stacked.size > 0:
        low = float(np.min(stacked))
        high = float(np.max(stacked))
        padding = 0.1 * (high - low) if high > low else max(abs(high), 1e-6) * 0.1
        low -= padding
        high += padding
        fig.add_scatter(
            x=[x_values[0]],
            y=[low / (2 * dispersive_shift)],
            yaxis="y2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.update_layout(
            yaxis=dict(range=[low, high]),
            yaxis2=dict(
                title="Mean photon number",
                overlaying="y",
                side="right",
                range=[low / (2 * dispersive_shift), high / (2 * dispersive_shift)],
                showgrid=False,
            ),
        )
    return fig


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


def plot_ac_stark_spectroscopy(data: dict[str, Any]) -> dict[str, go.Figure]:
    """
    Build the result figures for an AC Stark spectroscopy dataset.

    Per preparation state this produces up to two spectroscopy heatmaps
    (versus resonator drive amplitude and versus qubit drive time offset,
    with a slice-selection dropdown when the other axis is also swept), a
    shift summary plot (with a photon-number secondary axis when
    ``dispersive_shift`` is available) and, for a 2D sweep, shift and
    photon-number heatmaps over the amplitude/offset grid.

    Parameters
    ----------
    data : dict[str, Any]
        The ``Result.data`` payload produced by ``ac_stark_spectroscopy``.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by a stable name including the state label.
    """
    target = data["target"]
    states = data["axes"]["state"]
    amplitude_range = data["axes"]["resonator_drive_amplitude_range"]
    offset_range = data["axes"]["qubit_drive_time_offset_range"]
    frequency_ranges = data["axes"]["qubit_frequency_range"]
    normalized = data["spectroscopy"]["normalized"]
    f0 = data["fit"]["f0"]
    fit_valid = data["fit"]["valid"]
    ac_stark_shift = data["derived"]["ac_stark_shift"]
    ac_stark_shift_error = data["derived"]["ac_stark_shift_error"]
    photon_number = data["derived"]["photon_number"]
    dispersive_shift = data["calibration"]["dispersive_shift"]

    n_amplitudes = amplitude_range.size
    n_offsets = offset_range.size
    figures: dict[str, go.Figure] = {}

    for state_index, state in enumerate(states):
        if n_amplitudes > 1:
            figures[f"spectroscopy_vs_amplitude_{state}"] = _spectroscopy_heatmap(
                target=target,
                state=state,
                x_values=amplitude_range,
                x_title="Resonator drive amplitude",
                frequency_range=frequency_ranges[state_index],
                signal_slices=[
                    normalized[state_index, :, offset_index, :]
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
        if n_offsets > 1:
            figures[f"spectroscopy_vs_time_offset_{state}"] = _spectroscopy_heatmap(
                target=target,
                state=state,
                x_values=offset_range,
                x_title="Qubit drive time offset (ns)",
                frequency_range=frequency_ranges[state_index],
                signal_slices=[
                    normalized[state_index, amplitude_index, :, :]
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
                slice_labels=[f"A_r={amplitude:g}" for amplitude in amplitude_range],
            )

        if n_amplitudes > 1 and n_offsets > 1:
            figures[f"summary_{state}"] = _summary_figure(
                target=target,
                state=state,
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
            else:
                x_values = amplitude_range
                x_title = "Resonator drive amplitude"
                shift_trace = ac_stark_shift[state_index, :, 0]
                shift_error_trace = ac_stark_shift_error[state_index, :, 0]
            figures[f"summary_{state}"] = _summary_figure(
                target=target,
                state=state,
                x_values=x_values,
                x_title=x_title,
                shift_traces=[shift_trace],
                shift_error_traces=[shift_error_trace],
                trace_names=[f"|{state}>"],
                dispersive_shift=dispersive_shift,
            )
        else:
            frequency_range = frequency_ranges[state_index]
            order = np.argsort(frequency_range)
            fig = viz.make_figure()
            fig.add_scatter(
                x=frequency_range[order],
                y=normalized[state_index, 0, 0, order],
                mode="lines+markers",
                name="data",
            )
            if fit_valid[state_index, 0, 0]:
                fig.add_vline(x=f0[state_index, 0, 0], line_dash="dash")
            fig.update_layout(
                title=f"AC Stark spectroscopy : {target} : |{state}>",
                xaxis_title="Qubit drive frequency (GHz)",
                yaxis_title="Signal",
                width=600,
                height=400,
            )
            figures[f"spectrum_{state}"] = fig

    return figures


def display_ac_stark_spectroscopy(result: Result, *, plot: bool = True) -> None:
    """
    Show the figures and print a summary table of an AC Stark result.

    Parameters
    ----------
    result : Result
        Result returned by ``ac_stark_spectroscopy``.
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
    print(f"qubit frequency (ge): {data['calibration']['qubit_frequency_ge']:.6f} GHz")
    if "e" in states:
        print(
            f"qubit frequency (ef): {data['calibration']['qubit_frequency_ef']:.6f} GHz"
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
