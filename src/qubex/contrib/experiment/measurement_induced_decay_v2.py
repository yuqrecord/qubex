"""
Measurement-induced decay of a single qubit under a flat-top readout drive.

This module measures how much faster a qubit leaves ``|e>`` while its readout
resonator is driven. A calibrated pi pulse prepares ``|e>``, a ``FlatTop`` drive
is played on the read channel for a swept duration, the cavity is allowed to
ring down, and the standard readout measures the surviving excited-state
population.

Pulse sequence
--------------
::

    qubit     : [x180][Blank(waittime_after_ctrl_pulse)]
    resonator :                                         [drive][Blank(waittime_ringdown)][readout]

- The state preparation is the calibrated ge pi pulse (``Experiment.x180``), so
  its shape, amplitude and duration come from the calibration and are not free
  parameters.
- The drive is a raw ``FlatTop`` played on the read channel exactly as
  parameterized. It is deliberately *not* built with ``Experiment.readout``:
  the readout factory adds pre/post margins and applies the configured ramp and
  DRAG policy, which would make "drive duration" ambiguous and would couple the
  stimulus to readout calibration drift.
- The no-drive reference replaces the drive with a ``Blank`` of exactly the same
  duration, so its timing is identical.

What is measured
----------------
The total idle time between the pi pulse and the readout grows with the swept
drive duration, so a single exponential fit of the excited-state population
against that duration returns

::

    Gamma_driven = Gamma_1 + Gamma_MID

where ``Gamma_1`` is the intrinsic relaxation rate and ``Gamma_MID`` is the
measurement-induced contribution. The fixed waits contribute only a constant
survival factor, which the fit absorbs into its amplitude. With
``reference=True`` the same sweep is repeated with the drive replaced by a
``Blank``, giving ``Gamma_ref = Gamma_1`` on the identical axis, and

::

    Gamma_MID = Gamma_driven - Gamma_ref

Capture windows
---------------
Every contiguous non-zero waveform range on the read channel opens its own
capture window, and the readout is scheduled last, so the readout data is
always the *last* capture. A driven schedule therefore produces two windows and
the ``Blank`` reference produces one; ``_extract_readout_iq`` checks the count so
that an unexpected compilation is reported rather than silently mis-read.

Assumptions and caveats
-----------------------
- The intracavity photon number fills over roughly ``1/kappa`` and the flat-top
  has two ramps, so the decay is not a single exponential for drive durations
  comparable to ``1/kappa`` or ``2 * ramptime``. Start the sweep above
  ``3 / kappa + 2 * ramptime`` and check the residual plot. Do not add a free
  time offset to the exponential: it is degenerate with the fitted amplitude.
- ``waittime_ringdown`` should be several ``1/kappa`` (``5/kappa`` leaves below
  1% of the photons) so that the final readout is not distorted by residual
  photons and so that the decay during the ring-down stays duration independent.
- At strong drive the transmon can leave the computational subspace (``|f>`` and
  dressed-state ionization). The reported rate is then an effective
  computational-subspace decay rate, not a clean ``|e> -> |g>`` rate. The
  off-axis IQ excursion and the fit diagnostics flag this, but only single-shot
  ``g``/``e``/``f`` classification can rule it out.
- ``Gamma_MID`` versus the squared drive amplitude is proportional to the photon
  number only at fixed detuning, since
  ``n ~ |A|^2 / ((kappa / 2)^2 + detuning^2)``. Converting to an absolute photon
  number needs a separate calibration.

Durations are in ns, amplitudes in normalized hardware units, frequencies and
rates in GHz and ns^-1 respectively (printed in MHz).
"""

from __future__ import annotations

from typing import Any

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

# Waveform sample grid.
SAMPLING_PERIOD_NS = float(Waveform.SAMPLING_PERIOD)
# QuEL-1 capture word (WORD_LENGTH = 4 samples). The backend adapter rejects a
# gap between two capture windows that is not a positive multiple of one word.
WORD_DURATION_NS = 8.0
# With DSP decimation every capture sum section must span a whole number of
# 16-sample groups, otherwise the driver raises E7awgCaptureDataError. The drive
# duration is the length of the first capture window, so it inherits this grid.
CAPTURE_SECTION_GRANULARITY_NS = 32.0
TIME_GRID_TOLERANCE = 1e-6

# A driven schedule plays two pulses on the read channel (drive and readout);
# the no-drive reference replaces the drive with a Blank and opens one window.
DRIVEN_CAPTURE_COUNT = 2
REFERENCE_CAPTURE_COUNT = 1

# An exponential has three free parameters, so four points is the absolute
# minimum; five keeps at least one degree of freedom for the r2 to mean anything.
MIN_DURATION_POINTS = 5

# Diagnostic thresholds. The off-axis excursion is measured in units of the
# calibrated |g>-|e> separation, so 0.3 is a large departure from that axis.
MAX_OFFAXIS_EXCURSION = 0.3
POPULATION_TOLERANCE = 0.1
MIN_FIT_R2 = 0.8

REFERENCE_LABEL = "reference"


def _is_multiple(value: float, unit: float) -> bool:
    """Return whether a value is an integer multiple of a unit."""
    return abs(value - round(value / unit) * unit) <= TIME_GRID_TOLERANCE


def _validate_finite_scalar(name: str, value: float) -> None:
    """Raise unless the value is a finite real number."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"`{name}` must be a real number, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value!r}.")


def _validate_time_on_grid(name: str, value: float, *, grid: float) -> None:
    """Raise unless the duration is a multiple of the given time grid."""
    _validate_finite_scalar(name, value)
    if not _is_multiple(value, grid):
        raise ValueError(f"`{name}` ({value} ns) must be a multiple of {grid} ns.")


def _validate_axis(name: str, values: float | ArrayLike) -> NDArray[np.float64]:
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
    return array


def _validate_amplitude(name: str, value: float) -> None:
    """Raise unless the amplitude is positive and within the hardware range."""
    _validate_finite_scalar(name, value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"`{name}` must be within (0, 1], got {value}.")


def _validate_ramptime(readout_drive_ramptime: float) -> None:
    """Raise unless the flat-top ramp time is usable."""
    _validate_time_on_grid(
        "readout_drive_ramptime", readout_drive_ramptime, grid=SAMPLING_PERIOD_NS
    )
    if readout_drive_ramptime <= 0:
        raise ValueError(
            f"`readout_drive_ramptime` must be positive, got {readout_drive_ramptime}."
        )


def _validate_drive_duration(
    name: str,
    duration: float,
    *,
    readout_drive_ramptime: float,
) -> None:
    """Raise unless one drive duration is usable as a capture window."""
    _validate_finite_scalar(name, duration)
    if duration <= 0:
        raise ValueError(f"`{name}` must be positive, got {duration}.")
    if duration < 2 * readout_drive_ramptime:
        raise ValueError(
            f"`{name}` ({duration} ns) must be at least twice "
            f"`readout_drive_ramptime` ({readout_drive_ramptime} ns)."
        )
    if not _is_multiple(duration, CAPTURE_SECTION_GRANULARITY_NS):
        raise ValueError(
            f"`{name}` ({duration} ns) must be a multiple of "
            f"{CAPTURE_SECTION_GRANULARITY_NS} ns: the drive opens a capture "
            "window, and with DSP decimation every capture section must span a "
            "whole number of 16-sample groups."
        )


def _validate_waittimes(
    *,
    waittime_after_ctrl_pulse: float,
    waittime_ringdown: float,
    control_pulse_duration: float,
    readout_pre_margin: float,
) -> None:
    """
    Raise unless both waits keep the capture windows on the hardware grid.

    ``waittime_after_ctrl_pulse`` sets where the drive capture starts and
    ``waittime_ringdown`` sets the gap to the readout capture; the QuEL-1
    backend requires both to land on the capture word grid.
    """
    _validate_time_on_grid(
        "waittime_after_ctrl_pulse", waittime_after_ctrl_pulse, grid=SAMPLING_PERIOD_NS
    )
    _validate_time_on_grid(
        "waittime_ringdown", waittime_ringdown, grid=SAMPLING_PERIOD_NS
    )
    if waittime_after_ctrl_pulse < 0:
        raise ValueError(
            "`waittime_after_ctrl_pulse` must be nonnegative, "
            f"got {waittime_after_ctrl_pulse}."
        )
    if waittime_ringdown < 0:
        raise ValueError(
            f"`waittime_ringdown` must be nonnegative, got {waittime_ringdown}."
        )

    drive_start = control_pulse_duration + waittime_after_ctrl_pulse
    if not _is_multiple(drive_start, WORD_DURATION_NS):
        raise ValueError(
            "The readout drive capture must start on the "
            f"{WORD_DURATION_NS} ns capture word grid, but it starts at "
            f"{drive_start} ns (control pulse {control_pulse_duration} ns + "
            f"waittime_after_ctrl_pulse {waittime_after_ctrl_pulse} ns). "
            "Adjust `waittime_after_ctrl_pulse`."
        )

    readout_gap = waittime_ringdown + readout_pre_margin
    if not _is_multiple(readout_gap, WORD_DURATION_NS):
        raise ValueError(
            "The gap between the readout drive capture and the readout capture "
            f"must be a multiple of {WORD_DURATION_NS} ns, but it is "
            f"{readout_gap} ns (waittime_ringdown {waittime_ringdown} ns + "
            f"readout pre-margin {readout_pre_margin} ns). "
            "Adjust `waittime_ringdown`."
        )


def _validate_shot_options(*, n_shots: int, shot_interval: float) -> None:
    """Raise unless the shot count and interval are usable."""
    if isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer)):
        raise TypeError(f"`n_shots` must be an integer, got {n_shots!r}.")
    if n_shots <= 0:
        raise ValueError(f"`n_shots` must be positive, got {n_shots}.")
    _validate_finite_scalar("shot_interval", shot_interval)
    if shot_interval <= 0:
        raise ValueError(f"`shot_interval` must be positive, got {shot_interval}.")


def _validate_duration_range(
    readout_drive_duration_range: ArrayLike,
    *,
    readout_drive_ramptime: float,
    require_fit: bool,
) -> NDArray[np.float64]:
    """Validate the swept drive durations and return them as an array."""
    duration_range = _validate_axis(
        "readout_drive_duration_range", readout_drive_duration_range
    )
    if require_fit and duration_range.size < MIN_DURATION_POINTS:
        raise ValueError(
            "`readout_drive_duration_range` must have at least "
            f"{MIN_DURATION_POINTS} distinct points to fit an exponential decay, "
            f"got {duration_range.size} point(s)."
        )
    for duration in duration_range:
        _validate_drive_duration(
            "readout_drive_duration_range",
            float(duration),
            readout_drive_ramptime=readout_drive_ramptime,
        )
    return duration_range


def _control_pulse(exp: Experiment, target: str) -> Waveform:
    """Return the calibrated ge pi pulse used for state preparation."""
    return exp.pulse.x180(exp.ctx.resolve_qubit_label(target))


def _get_rabi_param(exp: Experiment, target: str) -> RabiParam:
    """Return the calibrated GE Rabi parameters of the target."""
    qubit = exp.ctx.resolve_qubit_label(target)
    rabi_param = exp.pulse.ge_rabi_params.get(qubit)
    if rabi_param is None:
        rabi_param = exp.pulse.rabi_params.get(qubit)
    if rabi_param is None:
        raise ValueError(
            f"`{qubit}` does not have a RabiParam. Run a Rabi experiment first; "
            "this measurement does not synthesize a fallback normalization."
        )
    if not np.isfinite(rabi_param.amplitude) or rabi_param.amplitude == 0.0:
        raise ValueError(
            f"`{qubit}` has an unusable RabiParam amplitude "
            f"({rabi_param.amplitude}); recalibrate before measuring."
        )
    return rabi_param


def _readout_drive_pulse(
    *,
    readout_drive_amplitude: float,
    readout_drive_duration: float,
    readout_drive_ramptime: float,
    readout_drive_detuning: float,
) -> Waveform:
    """Build the flat-top readout drive played on the read channel."""
    return FlatTop(
        duration=readout_drive_duration,
        amplitude=readout_drive_amplitude,
        tau=readout_drive_ramptime,
    ).detuned(readout_drive_detuning)


def _condition_label(readout_drive_amplitude: float | None) -> str:
    """Return a short label naming one drive condition."""
    if readout_drive_amplitude is None:
        return REFERENCE_LABEL
    return f"A={readout_drive_amplitude:g}"


def _pulse_schedule_title(
    *,
    target: str,
    readout_drive_amplitude: float | None,
    readout_drive_duration: float,
) -> str:
    """Return the title used when plotting one pulse schedule."""
    return (
        f"Measurement-induced decay : {target} : "
        f"{_condition_label(readout_drive_amplitude)}, "
        f"D={readout_drive_duration:g} ns"
    )


def measurement_induced_decay_v2_sequence(
    exp: Experiment,
    *,
    target: str,
    readout_drive_amplitude: float | None,
    readout_drive_duration: float,
    readout_drive_ramptime: float,
    readout_drive_detuning: float,
    waittime_after_ctrl_pulse: float,
    waittime_ringdown: float,
) -> PulseSchedule:
    """
    Build the measurement-induced decay pulse schedule for one sweep point.

    Sequence: the calibrated ge pi pulse, then ``waittime_after_ctrl_pulse``,
    then the flat-top readout drive on the read channel, then
    ``waittime_ringdown``, then the standard readout.

    Parameters
    ----------
    exp : Experiment
        Experiment object providing pulses and hardware context.
    target : str
        Target qubit name or label.
    readout_drive_amplitude : float | None
        Amplitude of the flat-top readout drive. Pass None to build the
        no-drive reference, which replaces the drive with a ``Blank`` of
        ``readout_drive_duration`` and therefore keeps identical timing.
    readout_drive_duration : float
        Duration of the readout drive in ns.
    readout_drive_ramptime : float
        Ramp time (flat-top tau) of the readout drive in ns.
    readout_drive_detuning : float
        Detuning of the readout drive from the calibrated readout frequency
        (GHz).
    waittime_after_ctrl_pulse : float
        Delay between the end of the pi pulse and the start of the readout
        drive (ns).
    waittime_ringdown : float
        Delay between the end of the readout drive and the readout pulse object
        start (ns). The readout pulse begins with its own pre-margin blank, so
        the two capture windows stay separate even at zero.

    Returns
    -------
    PulseSchedule
        Schedule with the readout drive and the readout on the read channel and
        the pi pulse on the ge control channel. Inputs are assumed validated.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    resonator = exp.ctx.resolve_read_label(target)

    if readout_drive_amplitude is None:
        drive_pulse: Waveform = Blank(readout_drive_duration)
    else:
        drive_pulse = _readout_drive_pulse(
            readout_drive_amplitude=readout_drive_amplitude,
            readout_drive_duration=readout_drive_duration,
            readout_drive_ramptime=readout_drive_ramptime,
            readout_drive_detuning=readout_drive_detuning,
        )

    with PulseSchedule() as ps:
        ps.add(qubit, _control_pulse(exp, target))
        if waittime_after_ctrl_pulse > 0:
            ps.add(qubit, Blank(waittime_after_ctrl_pulse))
        ps.barrier()
        ps.add(resonator, drive_pulse)
        if waittime_ringdown > 0:
            ps.add(resonator, Blank(waittime_ringdown))
        ps.add(resonator, exp.pulse.readout(target))
    return ps


def _extract_readout_iq(
    measure_result: MultipleMeasureResult,
    *,
    read_label: str,
    target: str,
    expected_capture_count: int,
) -> complex:
    """
    Return the time-integrated readout IQ value of one measurement.

    The readout is scheduled last, so its data is always the last capture
    window. ``expected_capture_count`` is the number of windows the schedule
    should have opened; a mismatch means the schedule compiled differently than
    assumed and is raised rather than silently mis-read.
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
    if len(captures) != expected_capture_count:
        raise RuntimeError(
            f"Expected {expected_capture_count} capture window(s) for "
            f"{data_target}, got {len(captures)}. The last capture is the "
            "readout data; a different count indicates that the schedule was "
            "compiled differently than assumed."
        )
    return complex(np.mean(np.asarray(captures[-1].kerneled)))


def _measure_point(
    exp: Experiment,
    *,
    target: str,
    read_label: str,
    readout_drive_amplitude: float | None,
    readout_drive_duration: float,
    readout_drive_ramptime: float,
    readout_drive_detuning: float,
    waittime_after_ctrl_pulse: float,
    waittime_ringdown: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool,
    reset_awg_and_capunits: bool,
) -> tuple[complex, MultipleMeasureResult]:
    """Acquire one (condition, duration) point and return its readout IQ."""
    schedule = measurement_induced_decay_v2_sequence(
        exp,
        target=target,
        readout_drive_amplitude=readout_drive_amplitude,
        readout_drive_duration=readout_drive_duration,
        readout_drive_ramptime=readout_drive_ramptime,
        readout_drive_detuning=readout_drive_detuning,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        waittime_ringdown=waittime_ringdown,
    )
    if plot_pulse_schedule:
        schedule.plot(
            title=_pulse_schedule_title(
                target=target,
                readout_drive_amplitude=readout_drive_amplitude,
                readout_drive_duration=readout_drive_duration,
            )
        )
    measure_result = exp.measurement_service.execute(
        schedule,
        mode="avg",
        time_integration=True,
        n_shots=n_shots,
        shot_interval=shot_interval,
        final_measurement=False,
        reset_awg_and_capunits=reset_awg_and_capunits,
        plot=False,
    )
    iq_value = _extract_readout_iq(
        measure_result,
        read_label=read_label,
        target=target,
        expected_capture_count=(
            REFERENCE_CAPTURE_COUNT
            if readout_drive_amplitude is None
            else DRIVEN_CAPTURE_COUNT
        ),
    )
    return iq_value, measure_result


def run_measurement_induced_decay_v2(
    exp: Experiment,
    *,
    target: str,
    readout_drive_amplitude: float | None,
    readout_drive_duration_range: ArrayLike,
    readout_drive_ramptime: float,
    readout_drive_detuning: float,
    waittime_after_ctrl_pulse: float,
    waittime_ringdown: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> dict[str, Any]:
    """
    Acquire one decay trace for a single drive condition.

    Sweeps ``readout_drive_duration_range`` and measures the averaged,
    time-integrated readout IQ value at each point with ``Experiment.execute``.
    This is the standalone building block; the main entry point interleaves the
    reference with the driven points instead of calling this per condition.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    readout_drive_amplitude : float | None
        Amplitude of the flat-top readout drive, or None for the no-drive
        reference trace.
    readout_drive_duration_range : ArrayLike
        Sweep values of the readout drive duration in ns. Each value must be
        positive, at least twice ``readout_drive_ramptime``, and a multiple of
        the capture section granularity.
    readout_drive_ramptime : float
        Ramp time (flat-top tau) of the readout drive in ns.
    readout_drive_detuning : float
        Detuning of the readout drive from the calibrated readout frequency
        (GHz).
    waittime_after_ctrl_pulse : float
        Delay between the pi pulse and the readout drive (ns).
    waittime_ringdown : float
        Delay between the readout drive and the readout (ns).
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
        Raw acquisition data with keys ``readout_drive_duration_range``,
        ``iq``, ``measure_results`` (the raw ``MultipleMeasureResult`` objects,
        one per duration), the resolved labels, the calibrated readout
        frequency, the control pulse waveform and an echo of the drive
        parameters.
    """
    if readout_drive_amplitude is not None:
        _validate_amplitude("readout_drive_amplitude", readout_drive_amplitude)
    _validate_finite_scalar("readout_drive_detuning", readout_drive_detuning)
    _validate_ramptime(readout_drive_ramptime)
    duration_range = _validate_duration_range(
        readout_drive_duration_range,
        readout_drive_ramptime=readout_drive_ramptime,
        require_fit=False,
    )
    control_pulse = _control_pulse(exp, target)
    _validate_waittimes(
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        waittime_ringdown=waittime_ringdown,
        control_pulse_duration=float(control_pulse.duration),
        readout_pre_margin=float(exp.ctx.readout_pre_margin),
    )
    _validate_shot_options(n_shots=n_shots, shot_interval=shot_interval)

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)

    iq_values: list[complex] = []
    measure_results: list[MultipleMeasureResult] = []
    for index, duration in enumerate(
        tqdm(
            duration_range,
            desc=f"drive duration ({_condition_label(readout_drive_amplitude)})",
            leave=False,
        )
    ):
        iq_value, measure_result = _measure_point(
            exp,
            target=target,
            read_label=read_label,
            readout_drive_amplitude=readout_drive_amplitude,
            readout_drive_duration=float(duration),
            readout_drive_ramptime=readout_drive_ramptime,
            readout_drive_detuning=readout_drive_detuning,
            waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
            waittime_ringdown=waittime_ringdown,
            shot_interval=shot_interval,
            n_shots=n_shots,
            plot_pulse_schedule=plot_pulse_schedule and index == 0,
            reset_awg_and_capunits=bool(reset_awg_and_capunits) and index == 0,
        )
        iq_values.append(iq_value)
        measure_results.append(measure_result)

    return {
        "target": target,
        "qubit_label": qubit_label,
        "read_label": read_label,
        "readout_drive_duration_range": duration_range,
        "iq": np.asarray(iq_values, dtype=complex),
        "measure_results": measure_results,
        "control_pulse": control_pulse,
        "readout_frequency": float(exp.ctx.targets[read_label].frequency),
        "readout_drive_amplitude": readout_drive_amplitude,
        "readout_drive_ramptime": readout_drive_ramptime,
        "readout_drive_detuning": readout_drive_detuning,
        "waittime_after_ctrl_pulse": waittime_after_ctrl_pulse,
        "waittime_ringdown": waittime_ringdown,
        "shot_interval": shot_interval,
        "n_shots": n_shots,
    }


def analyze_measurement_induced_decay_v2(
    *,
    readout_drive_duration_range: ArrayLike,
    iq: ArrayLike,
    rabi_param: RabiParam,
) -> dict[str, Any]:
    """
    Fit one decay trace and derive the decay rate.

    The complex readout IQ is normalized with the calibrated GE Rabi parameters
    and converted to the excited-state population ``(1 - normalized) / 2``,
    which is fitted with ``A * exp(-D / tau) + C`` against the drive duration.
    The fixed waits enter only through the constant ``A``, so ``tau`` is the
    decay time in the swept variable and ``1 / tau`` is the decay rate under the
    drive.

    The component of the IQ trace orthogonal to the calibrated ``|g>``-``|e>``
    axis is reported as ``offaxis_excursion`` (in units of the ``|g>``-``|e>``
    separation). A large excursion means the averaged signal left that axis, so
    the population conversion no longer holds -- typically leakage or
    dressed-state ionization at strong drive.

    Parameters
    ----------
    readout_drive_duration_range : ArrayLike
        Drive durations in ns, in acquisition order.
    iq : ArrayLike
        Complex readout IQ values for each duration, in the same order.
    rabi_param : RabiParam
        Calibrated GE Rabi parameters used to normalize the IQ data.

    Returns
    -------
    dict[str, Any]
        Fit results with keys ``duration_range``, ``normalized``,
        ``excited_population``, ``offaxis_excursion``, ``fit_result``, ``tau``,
        ``tau_err``, ``decay_rate``, ``decay_rate_error``, ``r2``, ``valid``
        and ``warnings``. Arrays are returned sorted by duration.
    """
    duration_range = np.atleast_1d(
        np.asarray(readout_drive_duration_range, dtype=float)
    )
    iq_array = np.atleast_1d(np.asarray(iq, dtype=complex))
    if duration_range.shape != iq_array.shape:
        raise ValueError(
            "`readout_drive_duration_range` and `iq` must have the same shape, "
            f"got {duration_range.shape} and {iq_array.shape}."
        )

    order = np.argsort(duration_range)
    duration_range = duration_range[order]
    iq_array = iq_array[order]

    normalized = np.asarray(rabi_param.normalize(iq_array), dtype=float)
    excited_population = 0.5 * (1.0 - normalized)
    # RabiParam models iq = (distance + 1j * projection) * exp(1j * angle), so
    # the real part after de-rotation is the calibrated off-axis coordinate.
    rotated = iq_array * np.exp(-1j * rabi_param.angle)
    offaxis_excursion = (np.real(rotated) - rabi_param.distance) / rabi_param.amplitude

    warnings: list[str] = []
    fit_result: Any = None
    try:
        fit_result = fitting.fit_exp_decay(
            x=duration_range,
            y=excited_population,
            plot=False,
        )
    except Exception as exc:
        # A failed fit is a diagnostic, not a reason to discard the acquisition.
        warnings.append(f"exponential fit raised an exception ({exc})")

    tau = np.nan
    tau_err = np.nan
    r2 = np.nan
    if fit_result is None:
        pass
    elif fit_result.status is not FitStatus.SUCCESS:
        warnings.append("exponential fit did not converge")
    else:
        tau = float(fit_result["tau"])
        tau_err = float(fit_result["tau_err"])
        r2 = float(fit_result["r2"])

    valid = bool(np.isfinite(tau) and tau > 0 and np.isfinite(tau_err))
    if fit_result is not None and fit_result.status is FitStatus.SUCCESS and not valid:
        warnings.append(f"fitted decay time is unusable (tau={tau}, err={tau_err})")
    if valid and not np.isfinite(r2):
        # A trace with no variance (a flat signal) still "converges", but its
        # coefficient of determination is undefined and the fitted decay time
        # carries no information.
        warnings.append(
            f"fit quality is undefined (r2={r2}); the trace shows no decay."
        )
        valid = False
    elif valid and r2 < MIN_FIT_R2:
        warnings.append(f"poor fit quality (r2={r2:.3f})")

    max_offaxis = float(np.max(np.abs(offaxis_excursion)))
    if max_offaxis > MAX_OFFAXIS_EXCURSION:
        warnings.append(
            f"IQ leaves the calibrated |g>-|e> axis by up to {max_offaxis:.3f} "
            "of the state separation; the population conversion may not hold "
            "(leakage or dressed-state ionization)."
        )
    if np.any(excited_population < -POPULATION_TOLERANCE) or np.any(
        excited_population > 1.0 + POPULATION_TOLERANCE
    ):
        warnings.append(
            "excited-state population falls outside [0, 1] beyond tolerance; "
            "check the Rabi calibration and the drive strength."
        )

    decay_rate = 1.0 / tau if valid else np.nan
    decay_rate_error = tau_err / tau**2 if valid else np.nan

    return {
        "duration_range": duration_range,
        "normalized": normalized,
        "excited_population": excited_population,
        "offaxis_excursion": offaxis_excursion,
        "max_offaxis_excursion": max_offaxis,
        "fit_result": fit_result,
        "tau": tau,
        "tau_err": tau_err,
        "decay_rate": decay_rate,
        "decay_rate_error": decay_rate_error,
        "r2": r2,
        "valid": valid,
        "warnings": warnings,
    }


def measurement_induced_decay_v2(
    exp: Experiment,
    *,
    target: str,
    readout_drive_amplitude_range: float | ArrayLike,
    readout_drive_duration_range: ArrayLike,
    readout_drive_ramptime: float,
    readout_drive_detuning: float,
    waittime_after_ctrl_pulse: float,
    waittime_ringdown: float,
    reference: bool,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> Result:
    """
    Measure the measurement-induced decay rate of one qubit.

    For every readout drive amplitude a decay trace is acquired against
    ``readout_drive_duration_range`` and fitted with a single exponential,
    giving ``Gamma_driven = Gamma_1 + Gamma_MID``. With ``reference=True`` a
    no-drive trace (a ``Blank`` in place of the drive, identical timing) is
    acquired on the same duration axis, giving ``Gamma_ref = Gamma_1``, and the
    measurement-induced rate is reported as
    ``Gamma_MID = Gamma_driven - Gamma_ref``.

    Acquisition interleaves the conditions: the duration is the outer loop and
    the conditions (reference first, then each amplitude) the inner loop, so the
    reference sits next to every driven point of the same duration in time and
    slow drift of T1, frequency or readout gain is common to both. One reference
    per duration is shared across amplitudes, since the reference does not
    depend on the drive amplitude.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label. This measurement drives and reads one qubit
        only; there is no simultaneous multi-qubit mode.
    readout_drive_amplitude_range : float | ArrayLike
        Sweep axis: amplitude of the flat-top readout drive. A scalar or a list;
        every value must be within (0, 1]. The no-drive case is not a member of
        this axis -- use ``reference=True``.
    readout_drive_duration_range : ArrayLike
        Sweep axis: duration of the readout drive in ns; this is the decay axis.
        Each value must be positive, at least twice ``readout_drive_ramptime``,
        and a multiple of the capture section granularity. At least
        ``MIN_DURATION_POINTS`` distinct points are required. Start well above
        ``3 / kappa + 2 * readout_drive_ramptime`` so that the cavity filling
        transient does not curve the exponential.
    readout_drive_ramptime : float
        Ramp time (flat-top tau) of the readout drive in ns. The flat part lasts
        ``readout_drive_duration - 2 * readout_drive_ramptime``.
    readout_drive_detuning : float
        Detuning of the readout drive from the calibrated readout frequency
        (GHz). The resonator frequency is state dependent, so this selects which
        dressed resonance is driven and therefore the photon number.
    waittime_after_ctrl_pulse : float
        Delay between the end of the pi pulse and the start of the readout drive
        (ns). It is constant across the sweep, so it only scales the fitted
        amplitude; keep it short to preserve contrast.
    waittime_ringdown : float
        Delay between the end of the readout drive and the readout (ns). Use
        several ``1 / kappa`` so that residual photons neither distort the
        readout nor add a duration-dependent decay.
    reference : bool
        Whether to acquire the interleaved no-drive reference trace. Without it
        only ``Gamma_driven`` is reported and ``Gamma_MID`` is NaN.
    shot_interval : float
        Interval between shots in ns.
    n_shots : int
        Number of shots per sweep point.
    plot : bool, optional
        If True, display the result figures.
    plot_pulse_schedule : bool, optional
        If True, plot the pulse schedule of the first driven point and, when
        enabled, of the first reference point.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    Result
        Result with ``data`` and ``figures``. ``data["signal"]`` holds the raw
        complex IQ and the normalized traces, ``data["fit"]`` the per-amplitude
        exponential fits, ``data["derived"]`` the rates in ns^-1 and
        ``data["diagnostics"]`` the collected warnings.
        ``data["raw"]["measure_results"]`` keeps the unprocessed
        ``MultipleMeasureResult`` objects as an object array of shape
        (amplitude, duration).

    Notes
    -----
    The uncertainty on ``Gamma_MID`` adds the driven and reference rate errors
    in quadrature, which ignores the drift covariance that the interleaving
    deliberately introduces; it is therefore a conservative estimate. A joint or
    bootstrap analysis over the paired blocks is better when rigorous errors
    matter.

    ``Gamma_MID`` versus the squared amplitude is proportional to the intracavity
    photon number only at fixed detuning; converting to an absolute photon
    number requires a separate calibration.
    """
    amplitude_range = _validate_axis(
        "readout_drive_amplitude_range", readout_drive_amplitude_range
    )
    for amplitude in amplitude_range:
        _validate_amplitude("readout_drive_amplitude_range", float(amplitude))
    _validate_finite_scalar("readout_drive_detuning", readout_drive_detuning)
    _validate_ramptime(readout_drive_ramptime)
    duration_range = _validate_duration_range(
        readout_drive_duration_range,
        readout_drive_ramptime=readout_drive_ramptime,
        require_fit=True,
    )
    control_pulse = _control_pulse(exp, target)
    _validate_waittimes(
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        waittime_ringdown=waittime_ringdown,
        control_pulse_duration=float(control_pulse.duration),
        readout_pre_margin=float(exp.ctx.readout_pre_margin),
    )
    _validate_shot_options(n_shots=n_shots, shot_interval=shot_interval)

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    rabi_param = _get_rabi_param(exp, target)

    n_amplitudes = amplitude_range.size
    n_durations = duration_range.size

    iq = np.full((n_amplitudes, n_durations), np.nan, dtype=complex)
    measure_results = np.empty((n_amplitudes, n_durations), dtype=object)
    reference_iq = np.full(n_durations, np.nan, dtype=complex)
    reference_measure_results = np.empty(n_durations, dtype=object)

    progress = tqdm(
        total=n_durations * (n_amplitudes + (1 if reference else 0)),
        desc=f"Measurement-induced decay : {target}",
    )
    is_first_execution = True
    # One block per duration, reference first: the reference then sits next to
    # every driven point of the same duration in time, so slow drift is common
    # to both and cancels in the rate difference. `None` marks the reference,
    # whose amplitude index is -1 because it is not part of the amplitude axis.
    conditions: list[tuple[int, float | None]] = [
        *([(-1, None)] if reference else []),
        *((index, float(amplitude)) for index, amplitude in enumerate(amplitude_range)),
    ]
    for duration_index, duration in enumerate(duration_range):
        for amplitude_index, amplitude in conditions:
            iq_value, measure_result = _measure_point(
                exp,
                target=target,
                read_label=read_label,
                readout_drive_amplitude=amplitude,
                readout_drive_duration=float(duration),
                readout_drive_ramptime=readout_drive_ramptime,
                readout_drive_detuning=readout_drive_detuning,
                waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
                waittime_ringdown=waittime_ringdown,
                shot_interval=shot_interval,
                n_shots=n_shots,
                plot_pulse_schedule=plot_pulse_schedule and duration_index == 0,
                reset_awg_and_capunits=(
                    bool(reset_awg_and_capunits) and is_first_execution
                ),
            )
            is_first_execution = False
            if amplitude is None:
                reference_iq[duration_index] = iq_value
                reference_measure_results[duration_index] = measure_result
            else:
                iq[amplitude_index, duration_index] = iq_value
                measure_results[amplitude_index, duration_index] = measure_result
            progress.update(1)
    progress.close()

    warnings: list[str] = []
    analyses: list[dict[str, Any]] = []
    for amplitude_index, amplitude in enumerate(amplitude_range):
        analysis = analyze_measurement_induced_decay_v2(
            readout_drive_duration_range=duration_range,
            iq=iq[amplitude_index],
            rabi_param=rabi_param,
        )
        analyses.append(analysis)
        warnings.extend(
            f"[A={amplitude:g}] {message}" for message in analysis["warnings"]
        )

    reference_analysis: dict[str, Any] | None = None
    if reference:
        reference_analysis = analyze_measurement_induced_decay_v2(
            readout_drive_duration_range=duration_range,
            iq=reference_iq,
            rabi_param=rabi_param,
        )
        warnings.extend(
            f"[{REFERENCE_LABEL}] {message}"
            for message in reference_analysis["warnings"]
        )

    sorted_duration_range = analyses[0]["duration_range"]
    normalized = np.stack([analysis["normalized"] for analysis in analyses])
    excited_population = np.stack(
        [analysis["excited_population"] for analysis in analyses]
    )
    offaxis_excursion = np.stack(
        [analysis["offaxis_excursion"] for analysis in analyses]
    )
    decay_rate = np.array([analysis["decay_rate"] for analysis in analyses])
    decay_rate_error = np.array([analysis["decay_rate_error"] for analysis in analyses])
    tau = np.array([analysis["tau"] for analysis in analyses])
    tau_err = np.array([analysis["tau_err"] for analysis in analyses])
    r2 = np.array([analysis["r2"] for analysis in analyses])
    fit_valid = np.array([analysis["valid"] for analysis in analyses], dtype=bool)
    # FitResult is a UserDict, so np.array() would iterate it into an array of
    # its keys; fill an object array explicitly to keep the objects themselves.
    fit_results = np.empty(len(analyses), dtype=object)
    for index, analysis in enumerate(analyses):
        fit_results[index] = analysis["fit_result"]

    if reference_analysis is None:
        reference_decay_rate = np.nan
        reference_decay_rate_error = np.nan
    else:
        reference_decay_rate = reference_analysis["decay_rate"]
        reference_decay_rate_error = reference_analysis["decay_rate_error"]

    measurement_induced_decay_rate = decay_rate - reference_decay_rate
    measurement_induced_decay_rate_error = np.hypot(
        decay_rate_error, reference_decay_rate_error
    )
    negative = measurement_induced_decay_rate < -measurement_induced_decay_rate_error
    warnings.extend(
        f"[A={amplitude_range[amplitude_index]:g}] measurement-induced rate is "
        "negative beyond its uncertainty; the drive appears to slow the decay, "
        "which usually means the reference and driven traces are not comparable."
        for amplitude_index in np.flatnonzero(negative)
    )

    data: dict[str, Any] = {
        "target": target,
        "qubit_label": qubit_label,
        "read_label": read_label,
        "axes": {
            "readout_drive_amplitude_range": amplitude_range,
            "readout_drive_duration_range": sorted_duration_range,
        },
        "parameters": {
            "control_pulse": control_pulse,
            "readout_drive_ramptime": readout_drive_ramptime,
            "readout_drive_detuning": readout_drive_detuning,
            "waittime_after_ctrl_pulse": waittime_after_ctrl_pulse,
            "waittime_ringdown": waittime_ringdown,
            "reference": reference,
            "shot_interval": shot_interval,
            "n_shots": n_shots,
            "readout_frequency": float(exp.ctx.targets[read_label].frequency),
            "rabi_param": rabi_param,
        },
        "signal": {
            "iq": iq,
            "normalized": normalized,
            "excited_population": excited_population,
            "offaxis_excursion": offaxis_excursion,
            "reference_iq": reference_iq,
            "reference_normalized": (
                None if reference_analysis is None else reference_analysis["normalized"]
            ),
            "reference_excited_population": (
                None
                if reference_analysis is None
                else reference_analysis["excited_population"]
            ),
            "reference_offaxis_excursion": (
                None
                if reference_analysis is None
                else reference_analysis["offaxis_excursion"]
            ),
        },
        "raw": {
            "measure_results": measure_results,
            "reference_measure_results": reference_measure_results,
        },
        "fit": {
            "tau": tau,
            "tau_err": tau_err,
            "r2": r2,
            "valid": fit_valid,
            "result": fit_results,
            "reference_tau": (
                np.nan if reference_analysis is None else reference_analysis["tau"]
            ),
            "reference_tau_err": (
                np.nan if reference_analysis is None else reference_analysis["tau_err"]
            ),
            "reference_r2": (
                np.nan if reference_analysis is None else reference_analysis["r2"]
            ),
            "reference_valid": (
                False if reference_analysis is None else reference_analysis["valid"]
            ),
            "reference_result": (
                None if reference_analysis is None else reference_analysis["fit_result"]
            ),
        },
        "derived": {
            "decay_rate": decay_rate,
            "decay_rate_error": decay_rate_error,
            "reference_decay_rate": reference_decay_rate,
            "reference_decay_rate_error": reference_decay_rate_error,
            "measurement_induced_decay_rate": measurement_induced_decay_rate,
            "measurement_induced_decay_rate_error": (
                measurement_induced_decay_rate_error
            ),
        },
        "diagnostics": {
            "max_offaxis_excursion": np.array(
                [analysis["max_offaxis_excursion"] for analysis in analyses]
            ),
            "reference_max_offaxis_excursion": (
                np.nan
                if reference_analysis is None
                else reference_analysis["max_offaxis_excursion"]
            ),
            "warnings": warnings,
        },
    }

    figures = plot_measurement_induced_decay_v2(data)
    result = Result(data=data, figures=figures)
    display_measurement_induced_decay_v2(result, plot=plot)
    return result


def _fit_curve(
    fit_result: Any, duration_range: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Return the fitted exponential evaluated on a dense duration grid."""
    if fit_result is None or fit_result.status is not FitStatus.SUCCESS:
        return None
    amplitude = float(fit_result["A"])
    tau = float(fit_result["tau"])
    offset = float(fit_result["C"])
    return amplitude * np.exp(-duration_range / tau) + offset


def _dense_durations(duration_range: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a dense duration grid spanning the sweep for smooth fit lines."""
    return np.linspace(
        float(np.min(duration_range)), float(np.max(duration_range)), 200
    )


def plot_measurement_induced_decay_v2(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the result figures for a measurement-induced decay dataset.

    Produces the decay curves with their fits, the fit residuals (which expose
    the cavity filling transient and a badly chosen fit window), the decay rate
    against the drive amplitude and against its square, and the IQ trajectories
    (whose departure from the calibrated axis is the leakage tell).

    Parameters
    ----------
    data : dict[str, Any]
        Payload built by ``measurement_induced_decay_v2``.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by ``decay_curves``, ``residuals``, ``decay_rate``,
        ``decay_rate_vs_power`` and ``iq``.
    """
    target = data["target"]
    amplitude_range = data["axes"]["readout_drive_amplitude_range"]
    duration_range = data["axes"]["readout_drive_duration_range"]
    excited_population = data["signal"]["excited_population"]
    reference_population = data["signal"]["reference_excited_population"]
    fit_results = data["fit"]["result"]
    reference_fit_result = data["fit"]["reference_result"]
    dense = _dense_durations(duration_range)

    figures: dict[str, go.Figure] = {}

    decay_fig = viz.make_figure()
    residual_fig = viz.make_figure()
    traces: list[tuple[str, NDArray[np.float64], Any]] = [
        (f"A={amplitude:g}", excited_population[index], fit_results[index])
        for index, amplitude in enumerate(amplitude_range)
    ]
    if reference_population is not None:
        traces.insert(0, (REFERENCE_LABEL, reference_population, reference_fit_result))
    for name, population, fit_result in traces:
        decay_fig.add_scatter(
            x=duration_range,
            y=population,
            mode="markers",
            name=name,
        )
        curve = _fit_curve(fit_result, dense)
        if curve is not None:
            decay_fig.add_scatter(
                x=dense,
                y=curve,
                mode="lines",
                name=f"{name} fit",
                line=dict(dash="dash"),
                showlegend=False,
            )
        fitted = _fit_curve(fit_result, duration_range)
        if fitted is not None:
            residual_fig.add_scatter(
                x=duration_range,
                y=population - fitted,
                mode="lines+markers",
                name=name,
            )
    decay_fig.update_layout(
        title=f"Measurement-induced decay : {target}",
        xaxis_title="Readout drive duration (ns)",
        yaxis_title="Excited state population",
        xaxis_type="log",
        width=600,
        height=400,
        showlegend=True,
    )
    residual_fig.add_hline(y=0.0, line_dash="dot")
    residual_fig.update_layout(
        title=f"Fit residuals : {target}",
        xaxis_title="Readout drive duration (ns)",
        yaxis_title="Residual (data - fit)",
        xaxis_type="log",
        width=600,
        height=400,
        showlegend=True,
    )
    figures["decay_curves"] = decay_fig
    figures["residuals"] = residual_fig

    # Rates are stored in ns^-1; 1 ns^-1 = 1000 MHz.
    decay_rate = 1e3 * data["derived"]["decay_rate"]
    decay_rate_error = 1e3 * data["derived"]["decay_rate_error"]
    mid_rate = 1e3 * data["derived"]["measurement_induced_decay_rate"]
    mid_rate_error = 1e3 * data["derived"]["measurement_induced_decay_rate_error"]
    reference_rate = 1e3 * data["derived"]["reference_decay_rate"]

    rate_fig = viz.make_figure()
    rate_fig.add_scatter(
        x=amplitude_range,
        y=decay_rate,
        error_y=dict(type="data", array=decay_rate_error, visible=True),
        mode="lines+markers",
        name="driven (Γ₁ + Γ_MID)",
    )
    if np.isfinite(reference_rate):
        rate_fig.add_hline(
            y=reference_rate,
            line_dash="dot",
            annotation_text="reference (Γ₁)",
        )
        rate_fig.add_scatter(
            x=amplitude_range,
            y=mid_rate,
            error_y=dict(type="data", array=mid_rate_error, visible=True),
            mode="lines+markers",
            name="measurement induced (Γ_MID)",
        )
    rate_fig.update_layout(
        title=f"Decay rate : {target}",
        xaxis_title="Readout drive amplitude",
        yaxis_title="Decay rate (MHz)",
        width=600,
        height=400,
        showlegend=True,
    )
    figures["decay_rate"] = rate_fig

    power_fig = viz.make_figure()
    power_rate = mid_rate if np.isfinite(reference_rate) else decay_rate
    power_rate_error = (
        mid_rate_error if np.isfinite(reference_rate) else decay_rate_error
    )
    power_name = "Γ_MID" if np.isfinite(reference_rate) else "Γ₁ + Γ_MID"
    power_fig.add_scatter(
        x=np.asarray(amplitude_range) ** 2,
        y=power_rate,
        error_y=dict(type="data", array=power_rate_error, visible=True),
        mode="lines+markers",
        name=power_name,
    )
    power_fig.update_layout(
        title=f"Decay rate vs drive power : {target}",
        xaxis_title="Readout drive amplitude squared",
        yaxis_title="Decay rate (MHz)",
        width=600,
        height=400,
        showlegend=True,
    )
    figures["decay_rate_vs_power"] = power_fig

    iq = data["signal"]["iq"]
    reference_iq = data["signal"]["reference_iq"]
    iq_fig = viz.make_figure()
    if reference_population is not None:
        iq_fig.add_scatter(
            x=np.real(reference_iq),
            y=np.imag(reference_iq),
            mode="markers",
            name=REFERENCE_LABEL,
        )
    for index, amplitude in enumerate(amplitude_range):
        iq_fig.add_scatter(
            x=np.real(iq[index]),
            y=np.imag(iq[index]),
            mode="markers",
            name=f"A={amplitude:g}",
        )
    iq_fig.update_layout(
        title=f"Readout IQ : {target}",
        xaxis_title="I",
        yaxis_title="Q",
        width=500,
        height=500,
        showlegend=True,
    )
    figures["iq"] = iq_fig

    return figures


def display_measurement_induced_decay_v2(
    result: Result,
    *,
    plot: bool = True,
) -> None:
    """
    Show the figures and print a summary of a measurement-induced decay result.

    Parameters
    ----------
    result : Result
        Result returned by ``measurement_induced_decay_v2``.
    plot : bool, optional
        If True, display the stored figures.
    """
    data = result.data
    if plot and result.figures:
        for figure in result.figures.values():
            figure.show()

    amplitude_range = data["axes"]["readout_drive_amplitude_range"]
    derived = data["derived"]
    fit = data["fit"]
    diagnostics = data["diagnostics"]

    print(f"=== Measurement-induced decay : {data['target']} ===")
    control_pulse = data["parameters"]["control_pulse"]
    print(
        "control pulse      : "
        f"{control_pulse.name} ({control_pulse.duration:g} ns, "
        f"max |amplitude| = {np.max(np.abs(control_pulse.values)):.4f})"
    )
    print(
        "readout drive      : "
        f"ramptime {data['parameters']['readout_drive_ramptime']:g} ns, "
        f"detuning {1e3 * data['parameters']['readout_drive_detuning']:+.3f} MHz"
    )
    print(
        "waits              : "
        f"after control pulse {data['parameters']['waittime_after_ctrl_pulse']:g} ns, "
        f"ringdown {data['parameters']['waittime_ringdown']:g} ns"
    )
    if data["parameters"]["reference"] and fit["reference_valid"]:
        print(
            "reference          : "
            f"T1 = {fit['reference_tau'] / 1e3:.3f} ± "
            f"{fit['reference_tau_err'] / 1e3:.3f} µs "
            f"(Γ₁ = {1e3 * derived['reference_decay_rate']:.4f} MHz, "
            f"r2 = {fit['reference_r2']:.3f})"
        )
    elif data["parameters"]["reference"]:
        print("reference          : fit failed; Γ_MID is NaN")
    else:
        print("reference          : not measured; Γ_MID is NaN")

    header = (
        f"{'A':>8} {'T1_driven (µs)':>18} {'Γ_driven (MHz)':>18} "
        f"{'Γ_MID (MHz)':>20} {'r2':>7} {'offaxis':>8} {'valid':>6}"
    )
    print(header)
    for index, amplitude in enumerate(amplitude_range):
        tau = fit["tau"][index]
        tau_err = fit["tau_err"][index]
        if np.isfinite(tau):
            tau_text = f"{tau / 1e3:.3f}±{tau_err / 1e3:.3f}"
            rate_text = (
                f"{1e3 * derived['decay_rate'][index]:.4f}"
                f"±{1e3 * derived['decay_rate_error'][index]:.4f}"
            )
        else:
            tau_text = "nan"
            rate_text = "nan"
        mid = derived["measurement_induced_decay_rate"][index]
        if np.isfinite(mid):
            mid_text = (
                f"{1e3 * mid:.4f}"
                f"±{1e3 * derived['measurement_induced_decay_rate_error'][index]:.4f}"
            )
        else:
            mid_text = "nan"
        print(
            f"{amplitude:>8g} {tau_text:>18} {rate_text:>18} {mid_text:>20} "
            f"{fit['r2'][index]:>7.3f} "
            f"{diagnostics['max_offaxis_excursion'][index]:>8.3f} "
            f"{bool(fit['valid'][index])!s:>6}"
        )

    for message in diagnostics["warnings"]:
        print(f"[WARN] {message}")
