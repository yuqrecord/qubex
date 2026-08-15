"""
Shared machinery for the ground-referenced 1Q transition experiments.

Every experiment in this package measures the same observable. The qubit is
climbed from |g> to the *lower* level of the target transition with a pi pulse on
every rung below it, the waveform under test is played on the target transition,
the same pi pulses are replayed in reverse order, and only then is the qubit read
out. Ideal un-preparation maps

    |k>   -> |g>,
    |k+1> -> |k+1>,

so the ground-state population after the sequence *is* the lower-level population
of the target transition's Bloch sphere. The readout therefore separates |g> from
the transition's upper level, and the IQ distance is measured against the ground
state no matter how high the rung sits.

The four amplitude calibrations additionally share the error-amplified
zero-crossing method: the Bloch vector is driven by (n - 1/4) and by (n + 1/4)
full turns, which ideally land on the same point of the equator, so the
*difference* of the two normalized signals crosses zero exactly at the correct
amplitude with a gain proportional to `n`.

Writing `A0` for the starting amplitude and `x` for the swept amplitude offset,
the pulse under test has rotation angle `theta_target + eps` with
`eps ~ theta_target * x / A0`, and

- half-pi family, `4n - 1` versus `4n + 1` repetitions::

      d(eps) = sin((4n - 1) eps) + sin((4n + 1) eps)
             = 2 sin(4 n eps) cos(eps)  ~  8 n eps

- pi family, a fixed `+-pi/2` tipping pulse followed by `2n` repetitions::

      d(delta) = sin(2 n delta - eta) + sin(2 n delta + eta)
               = 2 sin(2 n delta) cos(eta)  ~  4 n delta

where `eta` is the tipping pulse's own angle error. It cancels exactly, which is
what lets the pi calibration run while the half-pi pulse is still coarse.

Both families therefore share the difference slope `K = 4 pi n / A0` near the
root, and both develop extra roots once `|x - x*| / A0` reaches `1 / (2n)`;
`validate_amplitude_diff_range` enforces the conservative half of that bound.

Three systematics drive the implementation choices here:

- Any affine error in the normalization cancels in the difference, because
  `g z_minus + c = g z_plus + c` implies `z_minus = z_plus` for `g != 0`. Loss of
  contrast from imperfect ladder preparation is exactly such an affine error, so
  it costs precision but does not move the root. The two branches must therefore
  be normalized with the *same* `RabiParam`.
- A slow drift of the readout baseline does *not* cancel. The two branches are
  consequently interleaved point by point in ABBA order rather than measured as
  two consecutive sweeps.
- Leakage outside `{|k>, |k+1>}` and relaxation of the upper level do not cancel
  either, and they accumulate with the pulse count. The half-pi branches differ
  by two pulses, so the shorter branch is padded with an idle `Blank` placed
  *between the train and the un-preparation*: both branches then start every
  un-prep pulse, and the readout, at the same absolute time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam
from qubex.pulse import Blank, PulseArray, PulseSchedule, Waveform
from qubex.system.transition import (
    TRANSITION_NAMES,
    TransitionName,
    is_transition_name,
    transition_index,
)

Branch = Literal["minus", "plus"]

BRANCH_MINUS: Branch = "minus"
BRANCH_PLUS: Branch = "plus"
BRANCHES: tuple[Branch, Branch] = (BRANCH_MINUS, BRANCH_PLUS)

#: Fraction of the ideal slope below which the crossing fit is treated as a failure.
MIN_SLOPE_RATIO = 0.05

#: Fraction of the swept half-width the root must stay inside.
MAX_ROOT_POSITION = 0.9

#: Minimum number of distinct amplitude offsets needed for the linear fit.
MIN_SWEEP_POINTS = 4

#: Smallest Rabi arc amplitude that still supports normalization.
MIN_RABI_AMPLITUDE = 1e-12


@dataclass(frozen=True)
class TransitionLadder:
    """
    Labels needed to drive one transition from the ground state.

    Attributes
    ----------
    qubit : str
        Qubit label, e.g. `"Q01"`. Readout data and `sweep_parameter` results are
        keyed by this label.
    transition : TransitionName
        Transition under test, e.g. `"fh"`.
    target : str
        Target label of the transition, e.g. `"Q01-fh"`. Calibration entries are
        keyed by this label.
    lower_labels : tuple[str, ...]
        Target labels of every rung below the transition, in preparation order.
        Empty for `"ge"`.
    """

    qubit: str
    transition: TransitionName
    target: str
    lower_labels: tuple[str, ...]

    @property
    def level_index(self) -> int:
        """Return the lower level index `k` driven by the transition."""
        return transition_index(self.transition)


def sampling_period_ns(exp: Experiment) -> float:
    """Return the measurement sampling period in ns."""
    return exp.ctx.util.resolve_sampling_period(exp.ctx.measurement.sampling_period)


def validate_transition(transition: str) -> None:
    """Raise when the transition name is not part of the level ladder."""
    if not is_transition_name(transition):
        raise ValueError(
            f"`transition` {transition!r} is not a supported transition. "
            f"Supported transitions: {list(TRANSITION_NAMES)}."
        )


def validate_duration(duration: float, *, ramptime: float | None = None) -> None:
    """Raise when the pulse duration is unusable."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"`duration` must be a positive number, got {duration!r}.")
    period = Waveform.SAMPLING_PERIOD
    if not np.isclose(duration % period, 0) and not np.isclose(
        duration % period, period
    ):
        raise ValueError(
            f"`duration` must be a multiple of the sampling period ({period} ns), "
            f"got {duration} ns."
        )
    if ramptime is None:
        return
    if not np.isfinite(ramptime) or ramptime < 0:
        raise ValueError(f"`ramptime` must be a non-negative number, got {ramptime!r}.")
    if duration <= 2 * ramptime:
        raise ValueError(
            f"`duration` ({duration} ns) must be greater than `2 * ramptime` "
            f"({2 * ramptime} ns)."
        )


def validate_n_rotations(n_rotations: int) -> None:
    """Raise when the requested number of full rotations is unusable."""
    if not isinstance(n_rotations, (int, np.integer)) or isinstance(n_rotations, bool):
        raise TypeError(f"`n_rotations` must be an int, got {n_rotations!r}.")
    if n_rotations < 1:
        raise ValueError(f"`n_rotations` must be at least 1, got {n_rotations}.")


def validate_positive_int(value: int, name: str) -> None:
    """Raise when a count argument is not a positive int."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"`{name}` must be an int, got {value!r}.")
    if value < 1:
        raise ValueError(f"`{name}` must be at least 1, got {value}.")


def validate_n_shots(n_shots: int) -> None:
    """Raise when the shot count is unusable."""
    validate_positive_int(n_shots, "n_shots")


def validate_shot_interval(shot_interval: float) -> None:
    """Raise when the shot interval is unusable."""
    if not np.isfinite(shot_interval) or shot_interval < 0:
        raise ValueError(
            f"`shot_interval` must be a non-negative number, got {shot_interval!r}."
        )


def validate_amplitude(amplitude: float, name: str = "amplitude") -> None:
    """Raise when a drive amplitude is outside the usable range."""
    if not np.isfinite(amplitude):
        raise ValueError(f"`{name}` must be finite, got {amplitude!r}.")
    if not 0 < amplitude <= 1:
        raise ValueError(f"`{name}` must lie in (0, 1], got {amplitude}.")


def validate_initial_amplitude(initial_amplitude: float) -> None:
    """Raise when the starting amplitude is outside the usable range."""
    if not np.isfinite(initial_amplitude):
        raise ValueError(
            f"The starting amplitude is not finite ({initial_amplitude!r})."
        )
    if not 0 < initial_amplitude <= 1:
        raise ValueError(
            f"The starting amplitude must lie in (0, 1], got {initial_amplitude}."
        )


def validate_sweep_range(values: NDArray[np.float64], name: str) -> None:
    """Raise when a sweep array is empty or contains non-finite values."""
    if values.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError(f"`{name}` must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"`{name}` must contain only finite values.")


def validate_time_range(time_range: NDArray[np.float64]) -> None:
    """Raise when the drive-duration sweep is unusable."""
    validate_sweep_range(time_range, "time_range")
    if np.any(time_range < 0):
        raise ValueError("`time_range` must contain only non-negative durations.")


def validate_effective_durations(
    time_range: NDArray[np.float64],
    ramptime: float,
) -> None:
    """Raise when a swept `FlatTop` duration misses the sampling grid."""
    if not np.isfinite(ramptime) or ramptime < 0:
        raise ValueError(f"`ramptime` must be a non-negative number, got {ramptime!r}.")
    period = Waveform.SAMPLING_PERIOD
    durations = time_range + 2 * ramptime
    remainders = durations % period
    off_grid = durations[~(np.isclose(remainders, 0) | np.isclose(remainders, period))]
    if off_grid.size > 0:
        raise ValueError(
            "`time_range + 2 * ramptime` must be a multiple of the sampling "
            f"period ({period} ns); {off_grid.tolist()} are not."
        )


def validate_beta_range(beta_range: NDArray[np.float64], *, degree: int) -> None:
    """Raise when the DRAG beta sweep cannot support the polynomial fit."""
    validate_sweep_range(beta_range, "beta_range")
    n_distinct = len(np.unique(beta_range))
    if n_distinct <= degree:
        raise ValueError(
            f"`beta_range` needs more than `degree` ({degree}) distinct values "
            f"for the polynomial fit, got {n_distinct}."
        )


def validate_amplitude_diff_range(
    amplitude_diff_range: NDArray[np.float64],
    *,
    initial_amplitude: float,
    n_rotations: int,
) -> None:
    """Raise when the amplitude sweep is degenerate or reaches a second root."""
    validate_sweep_range(amplitude_diff_range, "amplitude_diff_range")
    n_distinct = len(np.unique(amplitude_diff_range))
    if n_distinct < MIN_SWEEP_POINTS:
        raise ValueError(
            f"`amplitude_diff_range` needs at least {MIN_SWEEP_POINTS} distinct "
            f"values for the linear fit, got {n_distinct}."
        )

    amplitudes = initial_amplitude + amplitude_diff_range
    if np.any(amplitudes <= 0) or np.any(amplitudes > 1):
        raise ValueError(
            "`amplitude_diff_range` drives the amplitude outside (0, 1] "
            f"(starting amplitude {initial_amplitude:.6g}, reached "
            f"[{amplitudes.min():.6g}, {amplitudes.max():.6g}])."
        )

    half_width = float(np.max(np.abs(amplitude_diff_range)))
    bound = initial_amplitude / (4 * n_rotations)
    if half_width >= bound:
        raise ValueError(
            f"`amplitude_diff_range` half-width ({half_width:.6g}) reaches the "
            f"monotonic bound A0 / (4 n) = {bound:.6g} for n_rotations="
            f"{n_rotations}, where the difference signal grows extra zero "
            "crossings. Narrow the range or lower `n_rotations`; roughly "
            f"{0.1 * initial_amplitude / n_rotations:.6g} keeps the response "
            "within ~10 % of linear."
        )


def resolve_sweep_range(values: ArrayLike) -> NDArray[np.float64]:
    """Normalize a scalar-or-sequence sweep argument to a 1-D float array."""
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def resolve_ladder(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
) -> TransitionLadder:
    """
    Resolve every label needed to drive one transition from the ground state.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the target registry.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive.

    Returns
    -------
    TransitionLadder
        Qubit label, transition label and the ordered lower-rung labels.
    """
    validate_transition(transition)
    qubit = exp.ctx.resolve_qubit_label(target)
    index = transition_index(transition)
    labels = [
        exp.ctx.resolve_transition_label(qubit, name)
        for name in TRANSITION_NAMES[: index + 1]
    ]
    for label in labels:
        if label not in exp.ctx.targets:
            raise ValueError(
                f"Target `{label}` is not registered. Register it with "
                "`Experiment.register_transition_target` before running this "
                "experiment."
            )
    return TransitionLadder(
        qubit=qubit,
        transition=transition,
        target=labels[-1],
        lower_labels=tuple(labels[:-1]),
    )


def validate_ladder_calibrations(exp: Experiment, ladder: TransitionLadder) -> None:
    """
    Raise when a rung below the transition has no pi-pulse calibration.

    `Experiment.pulse.x180` silently falls back to two half-pi pulses when no pi
    pulse is stored. That fallback would change both the fidelity and the
    duration of the ladder preparation without saying so, which is why it is
    rejected up front here.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibration note.
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    """
    missing = [
        label
        for label in ladder.lower_labels
        if exp.calib_note.get_drag_pi_param(label) is None
        and exp.calib_note.get_pi_param(label) is None
    ]
    if missing:
        raise ValueError(
            f"Pi pulses are not calibrated for {missing}, which the {ladder.target!r} "
            "ladder needs to prepare and un-prepare the ground reference. "
            "Calibrate every lower transition first."
        )


def get_transition_rabi_param(
    exp: Experiment,
    ladder: TransitionLadder,
) -> RabiParam:
    """
    Return the Rabi parameters stored for the transition, or raise.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibration note.
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.

    Returns
    -------
    RabiParam
        Rabi parameters keyed by the transition label. They normalize the signal
        and seed the amplitude estimate.
    """
    param = exp.ctx.get_rabi_param(ladder.target)
    if param is None:
        raise ValueError(
            f"Rabi parameters are not stored for {ladder.target!r}. Run "
            "`obtain_rabi_params_based_on_ground` first; this experiment needs "
            "them both to normalize the signal and to estimate a starting "
            "amplitude."
        )
    if not np.isfinite(param.amplitude) or abs(param.amplitude) < MIN_RABI_AMPLITUDE:
        raise ValueError(
            f"Rabi parameters stored for {ladder.target!r} are degenerate "
            f"(amplitude {param.amplitude!r}) and cannot normalize the signal. "
            "Re-run `obtain_rabi_params_based_on_ground`."
        )
    if not np.isfinite(param.angle) or not np.isfinite(param.offset):
        raise ValueError(
            f"Rabi parameters stored for {ladder.target!r} contain non-finite "
            "values. Re-run `obtain_rabi_params_based_on_ground`."
        )
    return param


def resolve_prep_pulses(
    exp: Experiment,
    ladder: TransitionLadder,
) -> tuple[Waveform, ...]:
    """
    Snapshot the pi pulses that climb the ladder, in preparation order.

    The waveforms are read once and then frozen into the schedule builder, so a
    calibration note that a later step updates cannot change the sequence in the
    middle of a sweep.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibrated pulses.
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.

    Returns
    -------
    tuple[Waveform, ...]
        One pi pulse per rung below the transition.
    """
    return tuple(exp.pulse.x180(label) for label in ladder.lower_labels)


def build_ground_referenced_schedule(
    *,
    ladder: TransitionLadder,
    prep_pulses: Sequence[Waveform],
    body: Waveform,
    pad_duration: float = 0.0,
) -> PulseSchedule:
    """
    Build `prep -> body -> pad -> un-prep` for one transition.

    Parameters
    ----------
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    prep_pulses : Sequence[Waveform]
        Pi pulses of the lower rungs, in preparation order.
    body : Waveform
        Waveform under test, played on the transition target.
    pad_duration : float, optional
        Idle time appended to the body so that branches of different length
        reach the un-preparation, and the readout, at the same absolute time.

    Returns
    -------
    PulseSchedule
        Schedule driving the ladder and the transition target.
    """
    if len(prep_pulses) != len(ladder.lower_labels):
        raise ValueError(
            f"`prep_pulses` has {len(prep_pulses)} entries but the {ladder.target!r} "
            f"ladder has {len(ladder.lower_labels)} lower rungs."
        )
    if not np.isfinite(pad_duration) or pad_duration < 0:
        raise ValueError(
            f"`pad_duration` must be a non-negative number, got {pad_duration!r}."
        )

    with PulseSchedule() as ps:
        for label, pulse in zip(ladder.lower_labels, prep_pulses, strict=True):
            ps.add(label, pulse)
            ps.barrier()
        ps.add(ladder.target, body)
        if pad_duration > 0:
            ps.add(ladder.target, Blank(pad_duration))
        ps.barrier()
        for label, pulse in zip(
            reversed(ladder.lower_labels),
            reversed(prep_pulses),
            strict=True,
        ):
            ps.add(label, pulse)
            ps.barrier()
    return ps


def sweep_ground_referenced(
    exp: Experiment,
    *,
    ladder: TransitionLadder,
    rabi_param: RabiParam | None,
    sequence: Callable[[float], PulseSchedule],
    sweep_range: ArrayLike,
    frequencies: dict[str, float] | None = None,
    n_shots: int,
    shot_interval: float,
) -> tuple[NDArray[np.complex128], NDArray[np.float64] | None]:
    """
    Run a ground-referenced sweep and return the raw and normalized signals.

    `MeasurementService.sweep_parameter` keys its result by qubit label and can
    only attach a ge or ef `RabiParam`, so the normalization is done here with
    the parameters stored under the *transition* label.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    rabi_param : RabiParam | None
        Parameters used to normalize the raw signal. `None` returns raw data
        only.
    sequence : Callable[[float], PulseSchedule]
        Sweep-value to schedule mapping.
    sweep_range : ArrayLike
        Values passed to `sweep_parameter`.
    frequencies : dict[str, float] | None, optional
        Control frequencies applied for the duration of the sweep.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.float64] | None]
        Raw kerneled IQ values and their normalized projection.
    """
    sweep_data = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        frequencies=frequencies,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
    ).data[ladder.qubit]
    raw = np.asarray(sweep_data.data, dtype=np.complex128)
    if rabi_param is None:
        return raw, None
    return raw, np.asarray(rabi_param.normalize(raw), dtype=np.float64)


def estimate_amplitude(
    exp: Experiment,
    *,
    target: str,
    unit_pulse: Waveform,
    pulse_type: Literal["hpi", "pi"],
) -> float:
    """
    Estimate a control amplitude from the stored Rabi rate.

    This reproduces the first guess of
    `CalibrationService.calibrate_default_pulse` and
    `CalibrationService.calibrate_drag_amplitude`: the target Rabi rate is set so
    that the unit-amplitude envelope's real area produces a quarter (or a half)
    rotation. `Experiment.calc_control_amplitude` is already transition aware.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the Rabi parameters.
    target : str
        Transition label.
    unit_pulse : Waveform
        The envelope built at `amplitude=1`.
    pulse_type : {"hpi", "pi"}
        Whether the pulse should realize a half-pi or a pi rotation.

    Returns
    -------
    float
        Estimated control amplitude.
    """
    area = float(unit_pulse.real.sum()) * sampling_period_ns(exp)
    rabi_rate = (0.25 if pulse_type == "hpi" else 0.5) / area
    return float(exp.calc_control_amplitude(target, rabi_rate))


def transition_frequency(
    exp: Experiment,
    *,
    qubit: str,
    index: int,
) -> float:
    """
    Return the control frequency of the `index`-th rung of the ladder.

    Rungs beyond the modelled ladder (the upper leakage channel of `"mn"`) are
    extrapolated with the spacing of the last modelled step.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the quantum system.
    qubit : str
        Qubit label.
    index : int
        Rung index, `0` for `"ge"`.

    Returns
    -------
    float
        Control frequency in GHz.
    """
    if index < 0:
        raise ValueError(f"`index` must be non-negative, got {index}.")
    qubit_object = exp.ctx.qubits[qubit]
    n_modelled = len(TRANSITION_NAMES)
    if index < n_modelled:
        return float(qubit_object.get_control_frequency(TRANSITION_NAMES[index]))
    last = float(qubit_object.get_control_frequency(TRANSITION_NAMES[-1]))
    previous = float(qubit_object.get_control_frequency(TRANSITION_NAMES[-2]))
    return last + (index - (n_modelled - 1)) * (last - previous)


def estimate_drag_beta(
    exp: Experiment,
    *,
    qubit: str,
    transition: TransitionName,
    drag_coeff: float,
) -> float:
    """
    Return a first guess for the DRAG beta of a transition.

    For the transition `k -> k+1` there are two leakage channels rather than one,
    with opposite-sign detunings

        delta_minus = w(k-1, k)   - w(k, k+1),
        delta_plus  = w(k+1, k+2) - w(k, k+1),

    and squared oscillator matrix elements `k` and `k + 2`. A weighted
    least-squares null of the two spectral components gives

        beta0 = -drag_coeff * (k dm + (k+2) dp) / (k dm^2 + (k+2) dp^2),

    which reduces to the familiar `-drag_coeff / alpha` at `k = 0`, where only
    the upper channel exists. On the default linear ladder
    `f_n = f_ge + n * alpha` it simplifies further to
    `-drag_coeff / ((k + 1) * alpha)`.

    This is only a seed; the beta calibration measures the real value.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the quantum system.
    qubit : str
        Qubit label.
    transition : TransitionName
        Transition to drive.
    drag_coeff : float
        DRAG coefficient.

    Returns
    -------
    float
        Estimated DRAG beta.
    """
    index = transition_index(transition)
    frequency = transition_frequency(exp, qubit=qubit, index=index)
    delta_plus = transition_frequency(exp, qubit=qubit, index=index + 1) - frequency
    if index == 0:
        weighted = -drag_coeff / delta_plus
    else:
        delta_minus = (
            transition_frequency(exp, qubit=qubit, index=index - 1) - frequency
        )
        numerator = index * delta_minus + (index + 2) * delta_plus
        denominator = index * delta_minus**2 + (index + 2) * delta_plus**2
        weighted = -drag_coeff * numerator / denominator
    if not np.isfinite(weighted):
        raise ValueError(
            f"The DRAG beta of {transition!r} on {qubit!r} could not be derived "
            "from the level ladder. Pass `beta` explicitly."
        )
    return float(weighted)


def build_acquisition_plan(
    amplitude_diff_range: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[Branch]]:
    """
    Interleave the two branches point by point in ABBA order.

    Returns one entry per acquisition, so the returned arrays are twice as long
    as `amplitude_diff_range`. Alternating the order within each pair keeps
    either branch from always being measured first, which cancels the linear part
    of a slow baseline drift.

    Parameters
    ----------
    amplitude_diff_range : NDArray[np.float64]
        Amplitude offsets to visit.

    Returns
    -------
    tuple[NDArray[np.float64], list[Branch]]
        The per-acquisition amplitude offset and branch label.
    """
    diffs: list[float] = []
    branches: list[Branch] = []
    for i, diff in enumerate(amplitude_diff_range):
        pair = BRANCHES if i % 2 == 0 else BRANCHES[::-1]
        for branch in pair:
            diffs.append(float(diff))
            branches.append(branch)
    return np.asarray(diffs, dtype=np.float64), branches


def build_branch_body(
    *,
    pulse: Waveform,
    branch: Branch,
    repetitions: dict[Branch, int],
    tipping_pulse: Waveform | None,
) -> Waveform:
    """
    Build the drive of one branch at one amplitude.

    Parameters
    ----------
    pulse : Waveform
        The pulse under test, already built at the swept amplitude.
    branch : Branch
        Which of the two rotation counts to realize.
    repetitions : dict[Branch, int]
        Repetition count per branch.
    tipping_pulse : Waveform | None
        Fixed `+-pi/2` pulse prepended to the train (pi family), or None when the
        pulse under test provides the quarter turn itself (half-pi family).

    Returns
    -------
    Waveform
        Concatenated drive of the branch.
    """
    segments: list[Waveform] = []
    if tipping_pulse is not None:
        segments.append(
            tipping_pulse.scaled(-1) if branch == BRANCH_MINUS else tipping_pulse
        )
    segments.append(pulse.repeated(repetitions[branch]))
    return PulseArray(segments)


def build_branch_schedule(
    *,
    ladder: TransitionLadder,
    prep_pulses: Sequence[Waveform],
    pulse: Waveform,
    branch: Branch,
    repetitions: dict[Branch, int],
    tipping_pulse: Waveform | None,
    pad_duration: float,
) -> PulseSchedule:
    """
    Build the ground-referenced schedule of one branch at one amplitude.

    Parameters
    ----------
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    prep_pulses : Sequence[Waveform]
        Pi pulses of the lower rungs, in preparation order.
    pulse : Waveform
        The pulse under test, already built at the swept amplitude.
    branch : Branch
        Which of the two rotation counts to realize.
    repetitions : dict[Branch, int]
        Repetition count per branch.
    tipping_pulse : Waveform | None
        Fixed `+-pi/2` pulse, or None.
    pad_duration : float
        Idle time appended to the `minus` branch's train so that both branches
        reach the un-preparation after the same elapsed time. Ignored for the
        `plus` branch.

    Returns
    -------
    PulseSchedule
        Ground-referenced schedule of the branch.
    """
    return build_ground_referenced_schedule(
        ladder=ladder,
        prep_pulses=prep_pulses,
        body=build_branch_body(
            pulse=pulse,
            branch=branch,
            repetitions=repetitions,
            tipping_pulse=tipping_pulse,
        ),
        pad_duration=pad_duration if branch == BRANCH_MINUS else 0.0,
    )


def build_branch_sequence(
    *,
    ladder: TransitionLadder,
    prep_pulses: Sequence[Waveform],
    make_pulse: Callable[[float], Waveform],
    initial_amplitude: float,
    acquisition_diffs: NDArray[np.float64],
    acquisition_branches: Sequence[Branch],
    repetitions: dict[Branch, int],
    tipping_pulse: Waveform | None,
    pad_duration: float,
) -> Callable[[float], PulseSchedule]:
    """
    Return the parametric sequence consumed by `sweep_parameter`.

    The sweep coordinate is an acquisition index rather than a physical
    parameter, because each index selects both an amplitude offset and a branch.
    Everything the closure needs is captured once, so the schedule never re-reads
    a calibration note that a later step may already have updated.

    Parameters
    ----------
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    prep_pulses : Sequence[Waveform]
        Pi pulses of the lower rungs, in preparation order.
    make_pulse : Callable[[float], Waveform]
        Builds the pulse under test at an absolute amplitude.
    initial_amplitude : float
        Amplitude the offsets are measured from.
    acquisition_diffs : NDArray[np.float64]
        Amplitude offset of every acquisition.
    acquisition_branches : Sequence[Branch]
        Branch of every acquisition.
    repetitions : dict[Branch, int]
        Repetition count per branch.
    tipping_pulse : Waveform | None
        Fixed `+-pi/2` pulse, or None.
    pad_duration : float
        Idle time appended to the `minus` branch's train.

    Returns
    -------
    Callable[[float], PulseSchedule]
        Maps an acquisition index to its schedule.
    """
    branches = list(acquisition_branches)
    pulses = tuple(prep_pulses)

    def sequence(index: float) -> PulseSchedule:
        i = round(float(index))
        return build_branch_schedule(
            ladder=ladder,
            prep_pulses=pulses,
            pulse=make_pulse(initial_amplitude + float(acquisition_diffs[i])),
            branch=branches[i],
            repetitions=repetitions,
            tipping_pulse=tipping_pulse,
            pad_duration=pad_duration,
        )

    return sequence


def split_branches(
    signal: NDArray[np.float64],
    acquisition_branches: Sequence[Branch],
    n_points: int,
) -> dict[Branch, NDArray[np.float64]]:
    """
    De-interleave an acquisition-ordered array back into the two branches.

    Parameters
    ----------
    signal : NDArray[np.float64]
        Values in acquisition order.
    acquisition_branches : Sequence[Branch]
        Branch of every acquisition.
    n_points : int
        Number of amplitude offsets.

    Returns
    -------
    dict[Branch, NDArray[np.float64]]
        One array per branch, indexed by amplitude offset.
    """
    split: dict[Branch, NDArray[np.float64]] = {
        branch: np.full(n_points, np.nan, dtype=np.float64) for branch in BRANCHES
    }
    for acquisition, branch in enumerate(acquisition_branches):
        split[branch][acquisition // 2] = signal[acquisition]
    return split


def fit_crossing(
    *,
    target: str,
    amplitude_diff_range: NDArray[np.float64],
    signal_minus: NDArray[np.float64],
    signal_plus: NDArray[np.float64],
    initial_amplitude: float,
    n_rotations: int,
    theta_target: float,
) -> dict[str, Any]:
    """
    Locate the amplitude offset where the two branches agree.

    The paired difference is fitted rather than the two branches separately: it
    carries the same information with two fewer nuisance parameters and it
    exploits the common-mode noise the interleaving creates.

    The true difference is `2 sin(4 n eps) cos(eps)`, not a straight line, so a
    crossing that sits away from the centre of the sweep is pulled slightly
    towards it by the cubic term. Re-running with the sweep re-centred on the
    reported amplitude removes it, and the residual is far below the shot-noise
    floor in any case.

    Parameters
    ----------
    target : str
        Transition label, used in error messages.
    amplitude_diff_range : NDArray[np.float64]
        Amplitude offsets.
    signal_minus, signal_plus : NDArray[np.float64]
        Normalized signal of the (n - 1/4) and (n + 1/4) branches.
    initial_amplitude : float
        Amplitude the offsets are measured from.
    n_rotations : int
        Number of full rotations.
    theta_target : float
        Ideal rotation angle of one pulse, `pi / 2` or `pi`.

    Returns
    -------
    dict[str, Any]
        Fit parameters, the located crossing and its uncertainty, and the
        implied per-pulse angle error.
    """
    difference = signal_minus - signal_plus
    fit = fitting.fit_linear(
        amplitude_diff_range,
        difference,
        intercept=True,
        plot=False,
        target=target,
    )
    slope = float(fit["a"])
    intercept = float(fit["b"])
    cov = np.asarray(fit["pcov"], dtype=np.float64)

    # d ~ 8 n eps (half-pi) or 4 n delta (pi), and eps = theta_target * x / A0
    # in both cases, so the two families share the same ideal slope.
    ideal_slope = 4 * np.pi * n_rotations / initial_amplitude

    if abs(slope) < MIN_SLOPE_RATIO * abs(ideal_slope):
        raise ValueError(
            f"The fitted slope for {target!r} ({slope:.6g}) is below "
            f"{MIN_SLOPE_RATIO:.0%} of the ideal {ideal_slope:.6g}. The "
            "difference signal carries almost no information: check that the "
            "Rabi parameters are current, that the ladder pi pulses still "
            "prepare the lower level, that `n_rotations` is not so large that "
            "decoherence has washed the contrast out, and - for the pi scripts "
            "- that the tipping pulse shares the drive phase of the train."
        )

    amplitude_diff = -intercept / slope
    variance = (
        (intercept / slope**2) ** 2 * cov[0, 0]
        + cov[1, 1] / slope**2
        - 2 * intercept * cov[0, 1] / slope**3
    )
    amplitude_diff_err = float(np.sqrt(variance)) if variance > 0 else float("nan")

    half_width = float(np.max(np.abs(amplitude_diff_range)))
    if abs(amplitude_diff) > MAX_ROOT_POSITION * half_width:
        raise ValueError(
            f"The crossing for {target!r} ({amplitude_diff:.6g}) lies outside "
            f"the central {MAX_ROOT_POSITION:.0%} of the swept half-width "
            f"({half_width:.6g}), so it is an extrapolation. Re-centre "
            "`amplitude_diff_range` on it, or widen the sweep."
        )

    fitted = slope * amplitude_diff_range + intercept
    return {
        "slope": slope,
        "slope_err": float(fit["a_err"]),
        "intercept": intercept,
        "intercept_err": float(fit["b_err"]),
        "cov": cov,
        "r2": float(fit["r2"]),
        "ideal_slope": float(ideal_slope),
        "slope_ratio": float(slope / ideal_slope),
        "residuals": difference - fitted,
        "fitted": fitted,
        "difference": difference,
        "amplitude_diff": float(amplitude_diff),
        "amplitude_diff_err": amplitude_diff_err,
        "amplitude": float(initial_amplitude + amplitude_diff),
        "relative_correction": float(amplitude_diff / initial_amplitude),
        "angle_error_per_pulse": float(
            theta_target * amplitude_diff / initial_amplitude
        ),
    }


def build_beta_body(
    *,
    pulse_type: Literal["hpi", "pi"],
    drive: Waveform,
    analyzer: Waveform,
    n_turns: int,
) -> Waveform:
    """
    Build the DRAG beta error-amplification drive.

    This mirrors `CalibrationService.calibrate_drag_beta`: alternating signs make
    the rotation-axis tilt caused by a wrong beta accumulate, and the closing
    `Y90-` analyzer maps it onto the measurement axis.

    Parameters
    ----------
    pulse_type : {"hpi", "pi"}
        Which alternating train to build.
    drive : Waveform
        The DRAG pulse at the swept beta.
    analyzer : Waveform
        Fixed `Y90-` pulse. It must not depend on the swept beta.
    n_turns : int
        Number of alternating pairs.

    Returns
    -------
    Waveform
        Concatenated drive of one beta point.
    """
    positive = drive
    negative = drive.scaled(-1)
    if pulse_type == "hpi":
        segments: list[Waveform] = [
            positive,
            PulseArray([negative, positive] * n_turns),
            analyzer,
        ]
    else:
        segments = [
            PulseArray([positive, negative] * n_turns),
            analyzer,
        ]
    return PulseArray(segments)


def run_beta_pass(
    exp: Experiment,
    *,
    ladder: TransitionLadder,
    prep_pulses: Sequence[Waveform],
    rabi_param: RabiParam,
    pulse_type: Literal["hpi", "pi"],
    make_drive: Callable[[float], Waveform],
    analyzer: Waveform,
    beta_range: NDArray[np.float64],
    n_turns: int,
    n_shots: int,
    shot_interval: float,
) -> dict[str, Any]:
    """
    Measure the ground-referenced DRAG beta sweep.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    ladder : TransitionLadder
        Ladder resolved by `resolve_ladder`.
    prep_pulses : Sequence[Waveform]
        Pi pulses of the lower rungs, in preparation order.
    rabi_param : RabiParam
        Parameters used to normalize the raw signal.
    pulse_type : {"hpi", "pi"}
        Which alternating train to build.
    make_drive : Callable[[float], Waveform]
        Builds the DRAG pulse at an absolute beta.
    analyzer : Waveform
        Fixed `Y90-` pulse.
    beta_range : NDArray[np.float64]
        Absolute beta values to visit.
    n_turns : int
        Number of alternating pairs.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_beta_pass`.
    """
    pulses = tuple(prep_pulses)

    def sequence(beta: float) -> PulseSchedule:
        return build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=pulses,
            body=build_beta_body(
                pulse_type=pulse_type,
                drive=make_drive(float(beta)),
                analyzer=analyzer,
                n_turns=n_turns,
            ),
        )

    raw, normalized = sweep_ground_referenced(
        exp,
        ladder=ladder,
        rabi_param=rabi_param,
        sequence=sequence,
        sweep_range=beta_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )
    return {
        "target": ladder.target,
        "pulse_type": pulse_type,
        "beta_range": beta_range,
        "n_turns": n_turns,
        "raw": raw,
        "normalized": normalized,
    }


def analyze_beta_pass(
    *,
    run_result: dict[str, Any],
    degree: int,
) -> dict[str, Any]:
    """
    Locate the beta that cancels the accumulated rotation-axis tilt.

    Unlike `CalibrationService.calibrate_drag_beta`, a fit whose root is
    non-finite or outside the swept range raises instead of silently becoming
    `beta = 0.0`.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_beta_pass`.
    degree : int
        Degree of the fitted polynomial.

    Returns
    -------
    dict[str, Any]
        The run payload plus the fit and the located beta.
    """
    target = run_result["target"]
    beta_range = run_result["beta_range"]
    fit = fitting.fit_polynomial(
        beta_range,
        run_result["normalized"],
        degree=degree,
        plot=False,
        target=target,
        title=f"DRAG {run_result['pulse_type']} beta calibration",
        xlabel="Beta",
        ylabel="Normalized signal",
    )
    beta = float(fit["root"])
    if not np.isfinite(beta):
        raise ValueError(
            f"The DRAG beta fit for {target!r} found no root inside "
            f"[{beta_range.min():.6g}, {beta_range.max():.6g}]. Widen "
            "`beta_range`, raise `n_turns`, or check the ladder preparation."
        )
    return {
        **run_result,
        "fit": fit,
        "beta": beta,
        "fitted": np.asarray(fit["fun"](beta_range), dtype=np.float64),
    }


def branches_figure(
    *,
    title: str,
    amplitude_diff_range: NDArray[np.float64],
    signal_minus: NDArray[np.float64],
    signal_plus: NDArray[np.float64],
    n_rotations: int,
) -> go.Figure:
    """Build the per-branch signal figure."""
    fig = viz.make_figure()
    fig.add_scatter(
        x=amplitude_diff_range,
        y=signal_minus,
        mode="lines+markers",
        name=f"{n_rotations} - 1/4 turns",
    )
    fig.add_scatter(
        x=amplitude_diff_range,
        y=signal_plus,
        mode="lines+markers",
        name=f"{n_rotations} + 1/4 turns",
    )
    fig.add_hline(y=0, line_dash="dot", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Amplitude difference (a.u.)",
        yaxis_title="Normalized signal",
        width=600,
        height=400,
    )
    return fig


def difference_figure(
    *,
    title: str,
    amplitude_diff_range: NDArray[np.float64],
    analysis: dict[str, Any],
) -> go.Figure:
    """Build the difference figure with its linear fit and the crossing."""
    fig = viz.make_figure()
    fig.add_scatter(
        x=amplitude_diff_range,
        y=analysis["difference"],
        mode="markers",
        name="difference",
    )
    fig.add_scatter(
        x=amplitude_diff_range,
        y=analysis["fitted"],
        mode="lines",
        name="linear fit",
    )
    fig.add_hline(y=0, line_dash="dot", line_width=1)
    fig.add_vline(x=analysis["amplitude_diff"], line_dash="dash")
    fig.add_annotation(
        x=analysis["amplitude_diff"],
        y=0,
        text=f"crossing: {analysis['amplitude_diff']:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Amplitude difference (a.u.)",
        yaxis_title="Signal difference",
        width=600,
        height=400,
    )
    return fig


def residuals_figure(
    *,
    title: str,
    amplitude_diff_range: NDArray[np.float64],
    analysis: dict[str, Any],
) -> go.Figure:
    """Build the fit-residual figure."""
    fig = viz.make_figure()
    fig.add_scatter(
        x=amplitude_diff_range,
        y=analysis["residuals"],
        mode="lines+markers",
        name="residual",
    )
    fig.add_hline(y=0, line_dash="dot", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Amplitude difference (a.u.)",
        yaxis_title="Residual",
        width=600,
        height=400,
    )
    return fig


def iq_figure(
    *,
    title: str,
    raw: NDArray[np.complex128],
) -> go.Figure:
    """Build the raw IQ figure, which exposes non-affine readout behaviour."""
    fig = viz.make_figure()
    fig.add_scatter(
        x=raw.real,
        y=raw.imag,
        mode="lines+markers",
        name="raw",
    )
    fig.update_layout(
        title=title,
        xaxis_title="I (arb. units)",
        yaxis_title="Q (arb. units)",
        width=500,
        height=400,
    )
    return fig


def beta_figure(
    *,
    title: str,
    analysis: dict[str, Any],
) -> go.Figure:
    """Build the DRAG beta sweep figure with its polynomial fit and root."""
    beta_range = analysis["beta_range"]
    fig = viz.make_figure()
    fig.add_scatter(
        x=beta_range,
        y=analysis["normalized"],
        mode="markers",
        name="signal",
    )
    fig.add_scatter(
        x=beta_range,
        y=analysis["fitted"],
        mode="lines",
        name="polynomial fit",
    )
    fig.add_vline(x=analysis["beta"], line_dash="dash")
    fig.add_annotation(
        x=analysis["beta"],
        y=0,
        text=f"beta: {analysis['beta']:.6g}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Beta",
        yaxis_title="Normalized signal",
        width=600,
        height=400,
    )
    return fig


def iterations_figure(
    *,
    title: str,
    iterations: list[dict[str, Any]],
) -> go.Figure:
    """Build the amplitude/beta history figure of the DRAG calibrations."""
    indices = list(range(1, len(iterations) + 1))
    fig = viz.make_figure()
    fig.add_scatter(
        x=indices,
        y=[iteration["amplitude"] for iteration in iterations],
        mode="lines+markers",
        name="amplitude",
    )
    fig.add_scatter(
        x=indices,
        y=[iteration["beta"] for iteration in iterations],
        mode="lines+markers",
        name="beta",
        yaxis="y2",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Amplitude (a.u.)",
        yaxis2=dict(title="Beta", overlaying="y", side="right"),
        xaxis=dict(tickmode="array", tickvals=indices),
        width=600,
        height=400,
    )
    return fig


def show_figures(figures: dict[str, go.Figure], *, prefix: str) -> None:
    """Show every figure with a download filename derived from `prefix`."""
    for key, figure in figures.items():
        figure.show(config=viz.get_config(filename=f"{prefix}_{key}"))


def plot_pulse_schedules(
    schedules: dict[str, PulseSchedule],
    *,
    title: str,
) -> None:
    """Draw one schedule per key, labelled by the key."""
    for key, schedule in schedules.items():
        schedule.plot(title=f"{title} : {key}")


def print_ladder(data: dict[str, Any]) -> None:
    """Print the ladder header shared by every `display_` function."""
    lower = list(data["lower_labels"])
    print(f"Target: {data['target']}")
    print(f"  qubit               : {data['qubit']}")
    print(f"  transition          : {data['transition']}")
    print(f"  ladder              : {' -> '.join([*lower, data['target']])}")


def print_summary(data: dict[str, Any]) -> None:
    """Print the common part of the four `display_` functions."""
    source = {
        "calib_note": "stored calibration",
        "rabi_estimate": "Rabi-rate estimate (no matching stored pulse)",
    }[data["initial_amplitude_source"]]
    print_ladder(data)
    print(f"  duration            : {data['duration']} ns")
    print(f"  n_rotations         : {data['n_rotations']}")
    print(
        f"  repetitions         : {data['repetitions']['minus']} (n - 1/4) / "
        f"{data['repetitions']['plus']} (n + 1/4)"
    )
    print(f"  initial amplitude   : {data['initial_amplitude']:.9g}  [{source}]")
    print(
        f"  amplitude offset    : {data['amplitude_diff']:.9g} "
        f"+- {data['amplitude_diff_err']:.3g}"
    )
    print(f"  calibrated amplitude: {data['amplitude']:.9g}")
    print(f"  relative correction : {data['relative_correction']:+.3%}")
    print(f"  angle error / pulse : {data['angle_error_per_pulse']:+.6g} rad")
    print(
        f"  fit                 : slope {data['fit']['slope']:.6g} "
        f"({data['fit']['slope_ratio']:.1%} of ideal), R2 {data['fit']['r2']:.4f}"
    )


def require_fit_success(fit: Any, *, target: str, what: str) -> None:
    """Raise when a `FitResult` reports an outright failure."""
    if getattr(fit, "status", FitStatus.SUCCESS) is FitStatus.ERROR:
        raise ValueError(
            f"The {what} for {target!r} failed: {getattr(fit, 'message', '')}"
        )
