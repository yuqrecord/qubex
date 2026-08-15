"""
Ground-referenced half-pi (FlatTop) amplitude calibration for any transition.

`Experiment.calibrate_hpi_pulse` locates the amplitude at the *extremum* of a
fitted cosine, which is flat and therefore imprecise, and it drives the target
straight from |g>, which is meaningless above ge. This script refines the
amplitude against a *zero crossing* while keeping the whole sequence referenced
to the ground state.

Method
------
The qubit is climbed to the lower level of the transition, the pulse under test
opens the train with a sign that selects the branch and is then repeated `4 n`
more times, the ladder is climbed back down and the qubit is read out::

    X_ge -> ... -> [-+hpi] -> [hpi] * 4n -> ... -> X_ge

The `minus` branch therefore realizes `-theta + 4n theta = (4n - 1) theta` and
the `plus` branch `(4n + 1) theta`, driving the Bloch vector by (n - 1/4) and
(n + 1/4) full turns. Both ideally land on the equator, so their normalized
signals are equal; with a per-pulse angle error `eps` they are
`sin((4n - 1) eps)` and `-sin((4n + 1) eps)`, whose difference

    d(eps) = 2 sin(4 n eps) cos(eps)  ~  8 n eps

crosses zero exactly at `eps = 0` with a gain proportional to `n`. Sweeping the
amplitude around its starting value and fitting `d` against a straight line
therefore locates the amplitude, amplified by `n`.

Inverting the leading pulse rather than dropping two pulses gives exactly the
same rotation angles, but leaves both branches with `4n + 1` pulses of identical
duration. That matters because a branch-dependent *gain* leaves the crossing
alone - both branch signals vanish there - while a branch-dependent *offset*
moves it by `dx / A0 = d(offset) / (4 pi n)`, and a difference in pulse count is
the easiest way to produce one: leakage and relaxation accumulate per pulse.

Starting point
--------------
The amplitude the offsets are measured from is the calibration stored under the
transition label when its `duration` and `tau` match the request, and otherwise
the same Rabi-rate first guess `Experiment.calibrate_hpi_pulse` uses, through
the transition-aware `Experiment.calc_control_amplitude`. In the latter case the
guess can be a few percent off, so use a wide `amplitude_diff_range` with
`n_rotations=1` first and re-run narrow.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, and the
  transition must have stored Rabi parameters. Both are checked, not worked
  around.
- `amplitude_diff_range` must stay inside `A0 / (4 n)`, beyond which the
  difference signal grows extra zero crossings. This is validated, not clipped.
- The two branches are interleaved point by point so that a slow drift of the
  readout baseline, which does *not* cancel, is suppressed.
- Loss of contrast from imperfect ladder preparation is an affine error and
  cancels in the difference; it costs precision, not accuracy.
- What the matched branches cannot cancel is an asymmetry between the `+X` and
  the `-X` leading pulse, such as an IQ-mixer offset, which biases the crossing
  by `mu / (2 pi n)`. Compare several `n_rotations`; a result that drifts
  systematically with `n` indicates such an effect rather than a pure pulse-area
  error.
- The calibrated amplitude is written to `exp.calib_note` under the transition
  label. `calib_note` does not autosave; call `Experiment.save_calib_note`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import FlatTop, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_calibrate_hpi_pulse_based_on_ground",
    "calibrate_hpi_pulse_based_on_ground",
    "display_calibrate_hpi_pulse_based_on_ground",
    "plot_calibrate_hpi_pulse_based_on_ground",
    "run_calibrate_hpi_pulse_based_on_ground",
]

THETA_TARGET = np.pi / 2


def _resolve_initial_amplitude(
    exp: Experiment,
    *,
    target: str,
    duration: float,
    ramptime: float,
) -> tuple[float, str]:
    """Return the starting amplitude and where it came from."""
    stored = exp.calib_note.get_hpi_param(target)
    if (
        stored is not None
        and stored.get("amplitude") is not None
        and np.isclose(stored["duration"], duration)
        and np.isclose(stored["tau"], ramptime)
    ):
        return float(stored["amplitude"]), "calib_note"
    unit_pulse = FlatTop(duration=duration, amplitude=1, tau=ramptime)
    amplitude = _core.estimate_amplitude(
        exp,
        target=target,
        unit_pulse=unit_pulse,
        pulse_type="hpi",
    )
    return amplitude, "rabi_estimate"


def _make_pulse(*, duration: float, ramptime: float) -> Callable[[float], Waveform]:
    """Return a builder of the pulse under test at a given amplitude."""

    def make_pulse(amplitude: float) -> Waveform:
        return FlatTop(duration=duration, amplitude=amplitude, tau=ramptime)

    return make_pulse


def run_calibrate_hpi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float,
    ramptime: float,
    n_rotations: int,
    amplitude_diff_range: ArrayLike,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
) -> dict[str, Any]:
    """
    Measure both rotation branches over the amplitude offsets.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to calibrate, `"ge"` through `"mn"`.
    duration : float
        Pulse duration in ns.
    ramptime : float
        `FlatTop` ramp time in ns.
    n_rotations : int
        Number of full rotations `n`; both branches use a signed leading pulse
        followed by `4n` more, realizing `4n - 1` and `4n + 1` rotations.
    amplitude_diff_range : ArrayLike
        Amplitude offsets from the starting amplitude. Scalar or sequence.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot_pulse_schedule : bool, optional
        Whether to draw the two branch schedules at zero offset.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_calibrate_hpi_pulse_based_on_ground`.
        It freezes the starting amplitude, the ladder pulses and the realized
        sweep, so the analysis never depends on a calibration note that a later
        step may have updated.
    """
    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    _core.validate_duration(duration, ramptime=ramptime)
    _core.validate_n_rotations(n_rotations)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    rabi_param = _core.get_transition_rabi_param(exp, ladder)
    initial_amplitude, source = _resolve_initial_amplitude(
        exp,
        target=ladder.target,
        duration=duration,
        ramptime=ramptime,
    )
    _core.validate_initial_amplitude(initial_amplitude)

    diffs = _core.resolve_sweep_range(amplitude_diff_range)
    _core.validate_amplitude_diff_range(
        diffs,
        initial_amplitude=initial_amplitude,
        n_rotations=n_rotations,
    )

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    repetitions = 4 * n_rotations
    make_pulse = _make_pulse(duration=duration, ramptime=ramptime)
    acquisition_diffs, acquisition_branches = _core.build_acquisition_plan(diffs)
    sequence = _core.build_branch_sequence(
        ladder=ladder,
        prep_pulses=prep_pulses,
        make_pulse=make_pulse,
        initial_amplitude=initial_amplitude,
        acquisition_diffs=acquisition_diffs,
        acquisition_branches=acquisition_branches,
        repetitions=repetitions,
        # The pulse under test tips onto the equator itself, so the leading
        # pulse is swept along with the train.
        make_tipping_pulse=make_pulse,
    )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {
                f"{branch} branch": _core.build_branch_schedule(
                    ladder=ladder,
                    prep_pulses=prep_pulses,
                    pulse=make_pulse(initial_amplitude),
                    branch=branch,
                    repetitions=repetitions,
                    tipping_pulse=make_pulse(initial_amplitude),
                )
                for branch in _core.BRANCHES
            },
            title=f"Ground-referenced hpi calibration : {ladder.target}",
        )

    raw, normalized = _core.sweep_ground_referenced(
        exp,
        ladder=ladder,
        rabi_param=rabi_param,
        sequence=sequence,
        sweep_range=np.arange(len(acquisition_diffs)),
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    return {
        "qubit": ladder.qubit,
        "transition": ladder.transition,
        "target": ladder.target,
        "lower_labels": ladder.lower_labels,
        "duration": duration,
        "ramptime": ramptime,
        "n_rotations": n_rotations,
        "repetitions": repetitions,
        "amplitude_diff_range": diffs,
        "amplitude_range": initial_amplitude + diffs,
        "initial_amplitude": initial_amplitude,
        "initial_amplitude_source": source,
        "acquisition_diffs": acquisition_diffs,
        "acquisition_branches": acquisition_branches,
        "rabi_param": rabi_param,
        "raw": raw,
        "normalized": normalized,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_calibrate_hpi_pulse_based_on_ground(
    *,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Locate the amplitude where the two branches agree.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_calibrate_hpi_pulse_based_on_ground`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the per-branch signals, the fit and the calibrated
        amplitude.
    """
    diffs = run_result["amplitude_diff_range"]
    signal = _core.split_branches(
        run_result["normalized"],
        run_result["acquisition_branches"],
        len(diffs),
    )
    fit = _core.fit_crossing(
        target=run_result["target"],
        amplitude_diff_range=diffs,
        signal_minus=signal[_core.BRANCH_MINUS],
        signal_plus=signal[_core.BRANCH_PLUS],
        initial_amplitude=run_result["initial_amplitude"],
        n_rotations=run_result["n_rotations"],
        theta_target=THETA_TARGET,
    )
    return {
        **run_result,
        "signal": signal,
        "difference": fit["difference"],
        "fit": fit,
        "amplitude_diff": fit["amplitude_diff"],
        "amplitude_diff_err": fit["amplitude_diff_err"],
        "amplitude": fit["amplitude"],
        "relative_correction": fit["relative_correction"],
        "angle_error_per_pulse": fit["angle_error_per_pulse"],
        "lobe_bound": run_result["initial_amplitude"] / (4 * run_result["n_rotations"]),
    }


def plot_calibrate_hpi_pulse_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_calibrate_hpi_pulse_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    title = f"Ground-referenced hpi calibration : {target}"
    return {
        "branches": _core.branches_figure(
            title=title,
            amplitude_diff_range=data["amplitude_diff_range"],
            signal_minus=data["signal"][_core.BRANCH_MINUS],
            signal_plus=data["signal"][_core.BRANCH_PLUS],
            n_rotations=data["n_rotations"],
        ),
        "difference": _core.difference_figure(
            title=f"{title} : difference",
            amplitude_diff_range=data["amplitude_diff_range"],
            analysis=data["fit"],
        ),
        "residuals": _core.residuals_figure(
            title=f"{title} : residuals",
            amplitude_diff_range=data["amplitude_diff_range"],
            analysis=data["fit"],
        ),
        "iq": _core.iq_figure(
            title=f"{title} : IQ",
            raw=data["raw"],
        ),
    }


def display_calibrate_hpi_pulse_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the calibration summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `calibrate_hpi_pulse_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"calibrate_hpi_pulse_based_on_ground_{data['target']}",
        )
    _core.print_summary(data)
    print(f"  ramptime            : {data['ramptime']} ns")


def calibrate_hpi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float,
    ramptime: float,
    n_rotations: int,
    amplitude_diff_range: ArrayLike,
    n_shots: int,
    shot_interval: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    """
    Calibrate a transition's half-pi pulse amplitude and store it.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to calibrate, `"ge"` through `"mn"`.
    duration : float
        Pulse duration in ns.
    ramptime : float
        `FlatTop` ramp time in ns.
    n_rotations : int
        Number of full rotations `n`.
    amplitude_diff_range : ArrayLike
        Amplitude offsets from the starting amplitude. Scalar or sequence.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot : bool, optional
        Whether to show the figures.
    plot_pulse_schedule : bool, optional
        Whether to draw the two branch schedules at zero offset.

    Returns
    -------
    Result
        Payload, figures and the calibrated amplitude. The amplitude is also
        written to `exp.calib_note` under the transition label.
    """
    run_result = run_calibrate_hpi_pulse_based_on_ground(
        exp,
        target=target,
        transition=transition,
        duration=duration,
        ramptime=ramptime,
        n_rotations=n_rotations,
        amplitude_diff_range=amplitude_diff_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    data = analyze_calibrate_hpi_pulse_based_on_ground(run_result=run_result)
    figures = plot_calibrate_hpi_pulse_based_on_ground(data)

    exp.calib_note.update_hpi_param(
        data["target"],
        {
            "target": data["target"],
            "duration": duration,
            "amplitude": data["amplitude"],
            "tau": ramptime,
        },
    )

    result = Result(data=data, figures=figures)
    display_calibrate_hpi_pulse_based_on_ground(result, plot)
    return result
