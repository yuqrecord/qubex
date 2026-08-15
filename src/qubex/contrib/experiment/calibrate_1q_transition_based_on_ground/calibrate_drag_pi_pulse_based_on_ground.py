"""
Ground-referenced DRAG pi calibration for any transition.

`Experiment.calibrate_drag_pi_pulse` locates the amplitude at the *extremum* of a
fitted cosine, drives the target straight from |g>, seeds `beta` from the ge
anharmonicity and delegates the beta step to `Experiment.calibrate_drag_beta`,
which normalizes through the ge Rabi arc and silently substitutes `beta = 0.0`
when its polynomial fit finds no root. None of that survives above ge, so both
passes are reimplemented here, referenced to the ground state.

Method
------
Amplitude and beta are correlated, so they are refined by coordinate descent over
`n_iterations` passes.

*Amplitude pass.* A train of identical pi rotations has no first-order
sensitivity to the pulse area, so a fixed `+-pi/2` *tipping* pulse is prepended
to bring the state onto the equator of the transition's Bloch sphere. Each branch
climbs the ladder, applies the tipping pulse and then `2 n` DRAG pi pulses,
realizing (n - 1/4) and (n + 1/4) full turns, climbs back down and is read out.
With a per-pulse angle error `delta` and a tipping-pulse angle error `eta` the
two normalized signals are `sin(2 n delta - eta)` and `-sin(2 n delta + eta)`,
whose difference

    d(delta) = 2 sin(2 n delta) cos(eta)  ~  4 n delta

crosses zero exactly at `delta = 0`. `eta` only scales the gain: the tipping
pulse's own amplitude error cancels *exactly*. Both branches contain the same
pulses in the same numbers, so they need no padding and their leakage and
relaxation match.

*Beta pass.* The same ladder wraps the error-amplification sequence of
`CalibrationService.calibrate_drag_beta`::

    [X180+ , X180-] * n_turns , Y90-

The closing `Y90-` analyzer is the transition's stored *FlatTop* half-pi, so it
does not depend on the beta being swept. The normalized signal is fitted with a
polynomial and its root is the calibrated beta; a non-finite root, or one outside
the swept range, raises rather than defaulting to zero.

Starting point
--------------
The amplitude and beta the sweep starts from are the calibration stored under the
transition label when its `duration` matches, and otherwise a Rabi-rate estimate
for the amplitude and `_core.estimate_drag_beta` for beta, which accounts for the
two leakage channels a higher rung has.

The tipping pulse is the transition's stored DRAG half-pi when there is one, and
otherwise its stored FlatTop half-pi. Either is fine — its error cancels — but
one of them must exist.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, and the
  transition must have stored Rabi parameters and a stored FlatTop half-pi pulse
  (the beta analyzer). Run `calibrate_hpi_pulse_based_on_ground` first.
- `amplitude_diff_range` must stay inside `A0 / (4 n)`, beyond which the
  difference signal grows extra zero crossings. This is validated, not clipped.
- `beta_range` is a set of offsets *from the current beta*, matching
  `Experiment.calibrate_drag_beta`.
- A phase offset between the tipping pulse and the train scales the gain by its
  cosine without moving the crossing, so a low fitted slope is a meaningful
  failure signal rather than mere noise.
- Coordinate descent ends each pass with beta, so the reported amplitude was
  measured at the *previous* beta. Use enough iterations for both to settle.
- Results are written to `exp.calib_note` under the transition label.
  `calib_note` does not autosave; call `Experiment.save_calib_note`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import Drag, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_calibrate_drag_pi_pulse_based_on_ground",
    "calibrate_drag_pi_pulse_based_on_ground",
    "display_calibrate_drag_pi_pulse_based_on_ground",
    "plot_calibrate_drag_pi_pulse_based_on_ground",
    "run_calibrate_drag_pi_pulse_based_on_ground",
]

THETA_TARGET = np.pi


def _resolve_initial_pulse(
    exp: Experiment,
    *,
    ladder: _core.TransitionLadder,
    duration: float,
    drag_coeff: float,
) -> tuple[float, float, str]:
    """Return the starting amplitude, the beta to use, and where they came from."""
    stored = exp.calib_note.get_drag_pi_param(ladder.target)
    if (
        stored is not None
        and stored.get("amplitude") is not None
        and stored.get("beta") is not None
        and np.isclose(stored["duration"], duration)
    ):
        return float(stored["amplitude"]), float(stored["beta"]), "calib_note"
    beta = _core.estimate_drag_beta(
        exp,
        qubit=ladder.qubit,
        transition=ladder.transition,
        drag_coeff=drag_coeff,
    )
    unit_pulse = Drag(duration=duration, amplitude=1, beta=beta)
    amplitude = _core.estimate_amplitude(
        exp,
        target=ladder.target,
        unit_pulse=unit_pulse,
        pulse_type="pi",
    )
    return amplitude, beta, "rabi_estimate"


def _resolve_tipping_pulse(exp: Experiment, *, target: str) -> Waveform:
    """Return the fixed half-pi pulse used to tip onto the equator."""
    if exp.calib_note.get_drag_hpi_param(target) is not None:
        return exp.pulse.get_drag_hpi_pulse(target)
    if exp.calib_note.get_hpi_param(target) is not None:
        return exp.pulse.get_hpi_pulse(target)
    raise ValueError(
        f"No half-pi pulse is calibrated for {target!r}, and this calibration "
        "needs one to tip the state onto the equator. Run "
        "`calibrate_hpi_pulse_based_on_ground` first."
    )


def _resolve_analyzer(exp: Experiment, *, target: str) -> Waveform:
    """Return the fixed `Y90-` analyzer of the beta pass."""
    if exp.calib_note.get_hpi_param(target) is None:
        raise ValueError(
            f"A FlatTop half-pi pulse is not calibrated for {target!r}, and the "
            "DRAG beta calibration needs it as the closing Y90- analyzer. Run "
            "`calibrate_hpi_pulse_based_on_ground` first."
        )
    return exp.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)


def _make_pulse(*, duration: float, beta: float) -> Callable[[float], Waveform]:
    """Return a builder of the pulse under test at a given amplitude."""

    def make_pulse(amplitude: float) -> Waveform:
        return Drag(duration=duration, amplitude=amplitude, beta=beta)

    return make_pulse


def run_calibrate_drag_pi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float,
    drag_coeff: float,
    n_rotations: int,
    amplitude_diff_range: ArrayLike,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
) -> dict[str, Any]:
    """
    Measure both rotation branches over the amplitude offsets, at fixed beta.

    This is one amplitude pass; `calibrate_drag_pi_pulse_based_on_ground`
    alternates it with the beta calibration.

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
    drag_coeff : float
        DRAG coefficient used to seed beta when no matching DRAG calibration is
        stored.
    n_rotations : int
        Number of full rotations `n`; both branches use `2n` pi pulses and differ
        in the sign of the tipping pulse.
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
        Raw payload consumed by
        `analyze_calibrate_drag_pi_pulse_based_on_ground`.
    """
    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    _core.validate_duration(duration)
    _core.validate_n_rotations(n_rotations)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    rabi_param = _core.get_transition_rabi_param(exp, ladder)
    tipping_pulse = _resolve_tipping_pulse(exp, target=ladder.target)
    initial_amplitude, beta, source = _resolve_initial_pulse(
        exp,
        ladder=ladder,
        duration=duration,
        drag_coeff=drag_coeff,
    )
    _core.validate_initial_amplitude(initial_amplitude)

    diffs = _core.resolve_sweep_range(amplitude_diff_range)
    _core.validate_amplitude_diff_range(
        diffs,
        initial_amplitude=initial_amplitude,
        n_rotations=n_rotations,
    )

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    repetitions: dict[_core.Branch, int] = {
        _core.BRANCH_MINUS: 2 * n_rotations,
        _core.BRANCH_PLUS: 2 * n_rotations,
    }
    make_pulse = _make_pulse(duration=duration, beta=beta)
    acquisition_diffs, acquisition_branches = _core.build_acquisition_plan(diffs)
    sequence = _core.build_branch_sequence(
        ladder=ladder,
        prep_pulses=prep_pulses,
        make_pulse=make_pulse,
        initial_amplitude=initial_amplitude,
        acquisition_diffs=acquisition_diffs,
        acquisition_branches=acquisition_branches,
        repetitions=repetitions,
        tipping_pulse=tipping_pulse,
        pad_duration=0.0,
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
                    tipping_pulse=tipping_pulse,
                    pad_duration=0.0,
                )
                for branch in _core.BRANCHES
            },
            title=f"Ground-referenced DRAG pi calibration : {ladder.target}",
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
        "ladder": ladder,
        "prep_pulses": prep_pulses,
        "duration": duration,
        "drag_coeff": drag_coeff,
        "beta": beta,
        "n_rotations": n_rotations,
        "repetitions": repetitions,
        "pad_duration": 0.0,
        "tipping_pulse": tipping_pulse,
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


def analyze_calibrate_drag_pi_pulse_based_on_ground(
    *,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Locate the amplitude where the two branches agree.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_calibrate_drag_pi_pulse_based_on_ground`.

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


def plot_calibrate_drag_pi_pulse_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_calibrate_drag_pi_pulse_based_on_ground`,
        or the assembled payload of `calibrate_drag_pi_pulse_based_on_ground`
        (which additionally carries the beta pass and the iteration history).

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    title = f"Ground-referenced DRAG pi calibration : {target}"
    figures = {
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
    beta_result = data.get("beta_result")
    if beta_result is not None:
        figures["beta"] = _core.beta_figure(
            title=f"{title} : beta",
            analysis=beta_result,
        )
    iterations = data.get("iterations")
    if iterations:
        figures["iterations"] = _core.iterations_figure(
            title=f"{title} : iterations",
            iterations=iterations,
        )
    return figures


def display_calibrate_drag_pi_pulse_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the calibration summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `calibrate_drag_pi_pulse_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"calibrate_drag_pi_pulse_based_on_ground_{data['target']}",
        )
    _core.print_summary(data)
    print(f"  beta                : {data['beta']:.9g}")
    print(f"  tipping pulse       : {data['tipping_pulse'].duration} ns")
    iterations = data.get("iterations")
    if iterations:
        print("  iterations:")
        for i, iteration in enumerate(iterations, start=1):
            print(
                f"    {i}: amplitude {iteration['amplitude']:.9g}  "
                f"beta {iteration['beta']:.9g}  "
                f"R2 {iteration['fit']['r2']:.4f}"
            )


def calibrate_drag_pi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float,
    drag_coeff: float,
    n_rotations: int,
    amplitude_diff_range: ArrayLike,
    n_iterations: int,
    calibrate_beta: bool,
    beta_range: ArrayLike,
    n_turns: int,
    degree: int,
    n_shots: int,
    shot_interval: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    """
    Calibrate a transition's DRAG pi amplitude and beta, and store both.

    Each iteration refines the amplitude against the zero crossing at the current
    beta, writes it to the calibration note, and then — when `calibrate_beta` is
    set — refines beta at that amplitude and writes it back.

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
    drag_coeff : float
        DRAG coefficient used to seed beta.
    n_rotations : int
        Number of full rotations `n`.
    amplitude_diff_range : ArrayLike
        Amplitude offsets from the starting amplitude. Scalar or sequence.
    n_iterations : int
        Number of amplitude/beta coordinate-descent passes.
    calibrate_beta : bool
        Whether to refine beta between amplitude passes.
    beta_range : ArrayLike
        Beta offsets from the current beta. Scalar or sequence.
    n_turns : int
        Number of `(X180+, X180-)` repetitions in the beta sequence.
    degree : int
        Degree of the polynomial fitted by the beta calibration.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot : bool, optional
        Whether to show the figures.
    plot_pulse_schedule : bool, optional
        Whether to draw the two branch schedules of the first pass.

    Returns
    -------
    Result
        Payload, figures, the calibrated amplitude and beta, and the
        per-iteration history. Both parameters are also written to
        `exp.calib_note` under the transition label.
    """
    _core.validate_positive_int(n_iterations, "n_iterations")
    _core.validate_positive_int(n_turns, "n_turns")
    _core.validate_positive_int(degree, "degree")
    betas = _core.resolve_sweep_range(beta_range)
    if calibrate_beta:
        _core.validate_beta_range(betas, degree=degree)

    iterations: list[dict[str, Any]] = []
    data: dict[str, Any] = {}
    beta_result: dict[str, Any] | None = None
    for i in range(n_iterations):
        run_result = run_calibrate_drag_pi_pulse_based_on_ground(
            exp,
            target=target,
            transition=transition,
            duration=duration,
            drag_coeff=drag_coeff,
            n_rotations=n_rotations,
            amplitude_diff_range=amplitude_diff_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot_pulse_schedule=plot_pulse_schedule and i == 0,
        )
        data = analyze_calibrate_drag_pi_pulse_based_on_ground(run_result=run_result)

        beta = float(data["beta"])
        amplitude = float(data["amplitude"])
        exp.calib_note.update_drag_pi_param(
            data["target"],
            {
                "target": data["target"],
                "duration": duration,
                "amplitude": amplitude,
                "beta": beta,
            },
        )

        if calibrate_beta:
            ladder: _core.TransitionLadder = data["ladder"]
            analyzer = _resolve_analyzer(exp, target=ladder.target)
            beta_result = _core.analyze_beta_pass(
                run_result=_core.run_beta_pass(
                    exp,
                    ladder=ladder,
                    prep_pulses=data["prep_pulses"],
                    rabi_param=data["rabi_param"],
                    pulse_type="pi",
                    make_drive=lambda value, amplitude=amplitude: Drag(
                        duration=duration,
                        amplitude=amplitude,
                        beta=float(value),
                    ),
                    analyzer=analyzer,
                    beta_range=beta + betas,
                    n_turns=n_turns,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                ),
                degree=degree,
            )
            beta = float(beta_result["beta"])
            exp.calib_note.update_drag_pi_param(
                data["target"],
                {
                    "target": data["target"],
                    "duration": duration,
                    "amplitude": amplitude,
                    "beta": beta,
                },
            )

        iterations.append(
            {
                "amplitude": amplitude,
                "beta": beta,
                "fit": data["fit"],
                "beta_result": beta_result,
                "data": data,
            }
        )

    data = {
        **data,
        "beta": iterations[-1]["beta"],
        "beta_result": beta_result,
        "iterations": iterations,
    }
    figures = plot_calibrate_drag_pi_pulse_based_on_ground(data)
    result = Result(data=data, figures=figures)
    display_calibrate_drag_pi_pulse_based_on_ground(result, plot)
    return result
