"""
Shared scheduling and acquisition machinery for the AC Stark spectroscopy variants.

This module holds everything that does not depend on how the resonator
(readout cavity) drive is parameterized: input validation primitives, the
pulse schedule builder, and the detuning-sweep acquisition loop. The
resonator drive enters as a fully prepared :class:`Waveform`, so the
FlatTop variant and the arbitrary-waveform variant share one
implementation.

Pulse sequence
--------------
::

    qubit     : [pi or blank][wait] | [Blank(max(0, +offset))][qubit drive]
    resonator : [     blank      ] | [Blank(max(0, -offset))][resonator drive][Blank(waittime_before_readout)][readout]

- The qubit drive is the calibrated ge pi pulse (``Experiment.x180``)
  detuned by ``qubit_drive_detuning``, so its shape, amplitude and
  duration come from the calibration and are not free parameters.
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
  validated by the callers and rejected otherwise.

Capture windows
---------------
Every pulse played on the read channel opens a capture window, and the
readout pulse is scheduled last, so the readout data is always the *last*
capture. A driven schedule with a single resonator drive play therefore
produces two windows, while the no-drive reference (a ``Blank`` in place
of the drive) produces one. ``extract_readout_iq`` takes the expected
count so that an unexpected schedule compilation is reported.

Validation helpers only check and raise; they never convert. Callers
convert explicitly after validation, and validate exactly once per public
entry point: ``acquire_spectroscopy_trace`` assumes validated inputs.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.pulse import Blank, PulseSchedule, Waveform

STATE_LABELS = ("g", "e")
MIN_QUBIT_DETUNING_POINTS = 5
TIME_GRID_TOLERANCE = 1e-6
# Tolerance on the normalized hardware amplitude bound, to accept waveforms
# whose peak sample is 1 up to floating point rounding.
AMPLITUDE_TOLERANCE = 1e-9


def sampling_period() -> float:
    """Return the waveform sampling period in ns."""
    return float(Waveform.SAMPLING_PERIOD)


def validate_finite_scalar(name: str, value: float) -> None:
    """Raise unless the value is a finite real number."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"`{name}` must be a real number, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value!r}.")


def validate_time_on_grid(name: str, value: float) -> None:
    """Raise unless the duration is a multiple of the sampling period."""
    validate_finite_scalar(name, value)
    dt = sampling_period()
    if abs(value - round(value / dt) * dt) > TIME_GRID_TOLERANCE:
        raise ValueError(
            f"`{name}` ({value} ns) must be a multiple of the {dt} ns sampling period."
        )


def validate_amplitude(name: str, value: float, *, allow_zero: bool) -> None:
    """Raise unless the amplitude lies within the normalized hardware range."""
    validate_finite_scalar(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be within [0, 1], got {value}.")
    if not allow_zero and value == 0.0:
        raise ValueError(f"`{name}` must be positive, got {value}.")


def validate_state(name: str, value: str) -> None:
    """Raise unless the value is a supported preparation state label."""
    if value not in STATE_LABELS:
        raise ValueError(f"`{name}` must be one of {STATE_LABELS}, got {value!r}.")


def validate_axis(name: str, values: float | ArrayLike) -> None:
    """Raise unless the value is a non-empty 1D sweep without duplicates."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a scalar or a 1D sequence.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.unique(array).size != array.size:
        raise ValueError(f"`{name}` must not contain duplicate values.")


def validate_n_shots(n_shots: int) -> None:
    """Raise unless the shot count is a positive integer."""
    if isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer)):
        raise TypeError(f"`n_shots` must be an integer, got {n_shots!r}.")
    if n_shots <= 0:
        raise ValueError(f"`n_shots` must be positive, got {n_shots}.")


def validate_qubit_drive_duration(duration: float) -> None:
    """Raise unless the calibrated pi pulse has a usable duration."""
    # The duration is not a user parameter: it comes from the calibrated
    # pi pulse. It is still checked because it enters the timing constraint.
    validate_finite_scalar("qubit_drive_duration", duration)
    if duration <= 0:
        raise ValueError(
            f"The calibrated pi pulse has a non-positive duration ({duration} ns)."
        )


def validate_detuning_range(name: str, values: ArrayLike) -> None:
    """Raise unless the detuning sweep can support a Lorentzian fit."""
    validate_axis(name, values)
    n_points = np.atleast_1d(np.asarray(values, dtype=float)).size
    if n_points < MIN_QUBIT_DETUNING_POINTS:
        raise ValueError(
            f"`{name}` must be a sweep with at least "
            f"{MIN_QUBIT_DETUNING_POINTS} distinct points to fit a Lorentzian, "
            f"got {n_points} point(s). The AC Stark shift cannot be "
            "measured without sweeping the qubit drive detuning."
        )


def validate_timing_constraint(
    *,
    qubit_drive_duration: float,
    qubit_drive_time_offset: float,
    resonator_drive_duration: float,
    waittime_before_readout: float,
) -> None:
    """Raise unless the qubit drive ends no later than the readout start."""
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


def validate_common_parameters(
    *,
    qubit_drive_duration: float,
    qubit_drive_detuning_range: ArrayLike,
    qubit_drive_time_offset: float,
    resonator_drive_duration: float,
    waittime_after_pi_pulse: float,
    waittime_before_readout: float,
    shot_interval: float,
    n_shots: int,
    time_offset_name: str = "qubit_drive_time_offset",
) -> None:
    """
    Validate the parameters that are common to every AC Stark variant.

    The preparation state and the resonator drive are validated by the
    caller, which knows whether a single state or a list is expected and
    how the drive is parameterized. ``time_offset_name`` names the offset
    in error messages so that a sweep can report its own argument name.
    """
    validate_qubit_drive_duration(qubit_drive_duration)
    validate_time_on_grid(time_offset_name, qubit_drive_time_offset)
    validate_time_on_grid("waittime_after_pi_pulse", waittime_after_pi_pulse)
    validate_time_on_grid("waittime_before_readout", waittime_before_readout)
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
    validate_timing_constraint(
        qubit_drive_duration=qubit_drive_duration,
        qubit_drive_time_offset=qubit_drive_time_offset,
        resonator_drive_duration=resonator_drive_duration,
        waittime_before_readout=waittime_before_readout,
    )
    validate_finite_scalar("shot_interval", shot_interval)
    if shot_interval <= 0:
        raise ValueError(f"`shot_interval` must be positive, got {shot_interval}.")
    validate_n_shots(n_shots)
    validate_detuning_range("qubit_drive_detuning_range", qubit_drive_detuning_range)


def qubit_drive_waveform(exp: Experiment, target: str) -> Waveform:
    """Return the calibrated ge pi pulse used as the spectroscopy probe."""
    return exp.x180(exp.ctx.resolve_qubit_label(target))


def get_rabi_param(exp: Experiment, target: str) -> RabiParam:
    """Return the calibrated GE Rabi parameters of the target."""
    rabi_param = exp.pulse.ge_rabi_params.get(target)
    if rabi_param is None:
        rabi_param = exp.pulse.rabi_params.get(target)
    if rabi_param is None:
        raise ValueError(f"`{target}` does not have a RabiParam.")
    return rabi_param


def build_sequence(
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
    for "g") followed by ``waittime_after_pi_pulse``, then the resonator
    drive and the qubit drive with relative time offset, then the standard
    readout starting at ``(resonator drive end) + waittime_before_readout``.

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
        Fully prepared resonator drive. It is played on the read channel
        exactly as given: this function never scales, detunes or otherwise
        modifies it, so any such policy belongs to the caller.
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
        channel, and the preparation and probe pulses on the ge control
        channel. Inputs are assumed validated.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    resonator = exp.ctx.resolve_read_label(target)

    qubit_delay = max(0.0, qubit_drive_time_offset)
    resonator_delay = max(0.0, -qubit_drive_time_offset)

    pi_pulse = qubit_drive_waveform(exp, target)
    qubit_drive_pulse = pi_pulse.detuned(qubit_drive_detuning)

    with PulseSchedule() as ps:
        buffer = 8 - (resonator_delay % 8)
        ps.add(qubit, Blank(buffer))
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
            ps.add(qubit, Blank(qubit_delay))
        ps.add(qubit, qubit_drive_pulse)
    return ps


def extract_readout_iq(
    measure_result: MultipleMeasureResult,
    *,
    read_label: str,
    target: str,
    expected_capture_count: int,
    strict: bool,
) -> complex:
    """
    Return the time-integrated readout IQ value of one measurement.

    The readout pulse is scheduled last, so the readout data is always the
    last capture window. ``expected_capture_count`` is the number of
    windows the schedule should have opened; a mismatch means the schedule
    was compiled differently than assumed and is either raised (``strict``)
    or reported as a warning.
    """
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
    if len(captures) == 0:
        raise RuntimeError(f"No capture windows returned for {data_target}.")
    if len(captures) != expected_capture_count:
        message = (
            f"Expected {expected_capture_count} capture window(s) for "
            f"{data_target}, got {len(captures)}."
        )
        if strict:
            raise RuntimeError(
                message + " The last capture is the readout data; a "
                "different count indicates that the schedule was compiled "
                "differently than assumed."
            )
        print(f"[WARN] {message} Using the last capture as the readout data.")
    return complex(captures[-1].kerneled)


def acquire_spectroscopy_trace(
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
    expected_capture_count: int,
    strict_capture_count: bool,
    plot_pulse_schedule: bool,
    pulse_schedule_title: str,
    reset_awg_and_capunits: bool,
    progress_description: str,
) -> dict[str, Any]:
    """
    Acquire one ge spectroscopy trace for a single parameter set.

    Sweeps the qubit drive detuning and measures the averaged,
    time-integrated readout IQ value for each point with
    ``Experiment.execute`` (``mode="avg"``, ``time_integration=True``).

    Inputs are assumed validated: every public entry point validates once
    at its own boundary and then delegates here.

    Returns
    -------
    dict[str, Any]
        ``detuning_range``, ``frequency_range``, ``iq``,
        ``measure_results`` (the raw ``MultipleMeasureResult`` objects,
        one per detuning point), the resolved labels,
        ``nominal_frequency`` (the calibrated ge transition frequency),
        ``resonator_frequency``, and the pulses that were played.
    """
    detuning_range = np.atleast_1d(np.asarray(qubit_drive_detuning_range, dtype=float))

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    nominal_frequency = float(exp.ctx.targets[qubit_label].frequency)
    resonator_frequency = float(exp.ctx.targets[read_label].frequency)
    qubit_drive_pulse = qubit_drive_waveform(exp, target)

    iq_values: list[complex] = []
    measure_results: list[MultipleMeasureResult] = []
    for index, detuning in enumerate(
        tqdm(detuning_range, desc=progress_description, leave=False)
    ):
        schedule = build_sequence(
            exp,
            target=target,
            qubit_initial_state=qubit_initial_state,
            qubit_drive_detuning=float(detuning),
            qubit_drive_time_offset=qubit_drive_time_offset,
            resonator_drive_pulse=resonator_drive_pulse,
            waittime_after_pi_pulse=waittime_after_pi_pulse,
            waittime_before_readout=waittime_before_readout,
        )
        if plot_pulse_schedule and index == 0:
            schedule.plot(title=pulse_schedule_title)
        measure_result = exp.execute(
            schedule=schedule,
            mode="avg",
            time_integration=True,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=reset_awg_and_capunits and index == 0,
        )
        measure_results.append(measure_result)
        iq_values.append(
            extract_readout_iq(
                measure_result,
                read_label=read_label,
                target=target,
                expected_capture_count=expected_capture_count,
                strict=strict_capture_count,
            )
        )

    return {
        "target": target,
        "qubit_label": qubit_label,
        "read_label": read_label,
        "qubit_initial_state": qubit_initial_state,
        "nominal_frequency": nominal_frequency,
        "resonator_frequency": resonator_frequency,
        "detuning_range": detuning_range,
        "frequency_range": nominal_frequency + detuning_range,
        "iq": np.asarray(iq_values, dtype=complex),
        "measure_results": measure_results,
        "qubit_drive_pulse": qubit_drive_pulse,
        "resonator_drive_pulse": resonator_drive_pulse,
        "qubit_drive_time_offset": qubit_drive_time_offset,
        "waittime_after_pi_pulse": waittime_after_pi_pulse,
        "waittime_before_readout": waittime_before_readout,
        "shot_interval": shot_interval,
        "n_shots": n_shots,
    }
