"""
Ground-referenced pi (FlatTop) amplitude calibration for any transition.

`Experiment.calibrate_pi_pulse` locates the amplitude at the *extremum* of a
fitted cosine, which is flat and therefore imprecise, and it drives the target
straight from |g>, which is meaningless above ge. This script refines the
amplitude against a *zero crossing* while keeping the whole sequence referenced
to the ground state.

Method
------
A train of identical pi rotations has no first-order sensitivity to the pulse
area — `d/d(delta) cos[m (pi + delta)]` vanishes at `delta = 0` — so a separate
`+-pi/2` *tipping* pulse is prepended to bring the state onto the equator of the
transition's Bloch sphere, where the accumulated error shows up directly along z.
Each branch climbs the ladder, applies the tipping pulse and then `2 n` pi
pulses, realizing (n - 1/4) and (n + 1/4) full turns, climbs back down and is
read out::

    X_ge -> ... -> [-+hpi] -> [pi] * 2n -> ... -> X_ge

With a per-pulse angle error `delta` and a tipping-pulse angle error `eta` the
two normalized signals are `sin(2 n delta - eta)` and `-sin(2 n delta + eta)`,
whose difference

    d(delta) = 2 sin(2 n delta) cos(eta)  ~  4 n delta

crosses zero exactly at `delta = 0`. Note that `eta` only scales the gain: the
tipping pulse's own amplitude error cancels *exactly*, not merely to first order.

Both branches contain the same pulses in the same numbers and differ only in the
sign of the tipping pulse, so they have identical duration and their leakage and
relaxation match. The half-pi calibrations are built the same way, with the
pulse under test acting as its own tipping pulse.

Starting point
--------------
The amplitude the offsets are measured from is the calibration stored under the
transition label when its `duration` and `tau` match the request, and otherwise
the Rabi-rate first guess through the transition-aware
`Experiment.calc_control_amplitude`.

The tipping pulse is the transition's stored FlatTop half-pi. Unlike the ge-only
reference implementation this script *requires* it: the uncalibrated fallback
amplitude `control_amplitude / sqrt(k + 1)` can be far off at a high rung, and a
badly wrong `eta` shrinks `cos(eta)` — the gain — unpredictably. Run
`calibrate_hpi_pulse_based_on_ground` first.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, and the
  transition must have stored Rabi parameters and a stored half-pi pulse.
- `amplitude_diff_range` must stay inside `A0 / (4 n)`, beyond which the
  difference signal grows extra zero crossings. This is validated, not clipped.
- The two branches are interleaved point by point so that a slow drift of the
  readout baseline, which does *not* cancel, is suppressed.
- A phase offset between the tipping pulse and the train scales the gain by its
  cosine without moving the crossing, so a low fitted slope is a meaningful
  failure signal rather than mere noise.
- The calibrated amplitude is written to `exp.calib_note` under the transition
  label. `calib_note` does not autosave; call `Experiment.save_calib_note`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
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
    "analyze_calibrate_pi_pulse_based_on_ground",
    "calibrate_pi_pulse_based_on_ground",
    "display_calibrate_pi_pulse_based_on_ground",
    "plot_calibrate_pi_pulse_based_on_ground",
    "run_calibrate_pi_pulse_based_on_ground",
]

THETA_TARGET = np.pi


def _resolve_initial_amplitude(
    exp: Experiment,
    *,
    target: str,
    duration: float,
    ramptime: float,
) -> tuple[float, str]:
    """Return the starting amplitude and where it came from."""
    stored = exp.calib_note.get_pi_param(target)
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
        pulse_type="pi",
    )
    return amplitude, "rabi_estimate"


def _resolve_tipping_pulse(exp: Experiment, *, target: str) -> Waveform:
    """Return the stored FlatTop half-pi used to tip onto the equator."""
    if exp.calib_note.get_hpi_param(target) is None:
        raise ValueError(
            f"A half-pi pulse is not calibrated for {target!r}, and this "
            "calibration needs it to tip the state onto the equator. Run "
            "`calibrate_hpi_pulse_based_on_ground` first."
        )
    return exp.pulse.get_hpi_pulse(target)


def _make_pulse(*, duration: float, ramptime: float) -> Callable[[float], Waveform]:
    """Return a builder of the pulse under test at a given amplitude."""

    def make_pulse(amplitude: float) -> Waveform:
        return FlatTop(duration=duration, amplitude=amplitude, tau=ramptime)

    return make_pulse


def run_calibrate_pi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float | None = None,
    ramptime: float | None = None,
    n_rotations: int | None = None,
    amplitude_diff_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot_pulse_schedule: bool | None = None,
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
    duration : float | None, optional
        Pulse duration in ns. Defaults to `_core.DEFAULT_PI_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_PI_RAMPTIME`.
    n_rotations : int | None, optional
        Number of full rotations `n`; both branches use `2n` pi pulses and differ
        in the sign of the tipping pulse.
        Defaults to `_core.DEFAULT_N_ROTATIONS`.
    amplitude_diff_range : ArrayLike | None, optional
        Amplitude offsets from the starting amplitude. Scalar or sequence.
        Defaults to a symmetric sweep around the starting amplitude, built by
        `_core.default_amplitude_diff_range`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the two branch schedules at zero offset. Defaults to
        True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_calibrate_pi_pulse_based_on_ground`.
        It freezes the starting amplitude, the ladder pulses and the realized
        sweep, so the analysis never depends on a calibration note that a later
        step may have updated.
    """
    if duration is None:
        duration = _core.DEFAULT_PI_DURATION
    if ramptime is None:
        ramptime = _core.DEFAULT_PI_RAMPTIME
    if n_rotations is None:
        n_rotations = _core.DEFAULT_N_ROTATIONS
    if n_shots is None:
        n_shots = _core.DEFAULT_N_SHOTS
    if shot_interval is None:
        shot_interval = _core.DEFAULT_SHOT_INTERVAL
    if plot_pulse_schedule is None:
        plot_pulse_schedule = True

    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    _core.validate_duration(duration, ramptime=ramptime)
    _core.validate_n_rotations(n_rotations)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    rabi_param = _core.get_transition_rabi_param(exp, ladder)
    tipping_pulse = _resolve_tipping_pulse(exp, target=ladder.target)
    initial_amplitude, source = _resolve_initial_amplitude(
        exp,
        target=ladder.target,
        duration=duration,
        ramptime=ramptime,
    )
    _core.validate_initial_amplitude(initial_amplitude)

    if amplitude_diff_range is None:
        amplitude_diff_range = _core.default_amplitude_diff_range(
            initial_amplitude=initial_amplitude,
            n_rotations=n_rotations,
        )
    diffs = _core.resolve_sweep_range(amplitude_diff_range)
    _core.validate_amplitude_diff_range(
        diffs,
        initial_amplitude=initial_amplitude,
        n_rotations=n_rotations,
    )

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    repetitions = 2 * n_rotations
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
        # The tipping pulse must not be swept: its own angle error only cancels
        # exactly while it stays fixed across the whole sweep.
        make_tipping_pulse=lambda _amplitude: tipping_pulse,
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
                )
                for branch in _core.BRANCHES
            },
            title=f"Ground-referenced pi calibration : {ladder.target}",
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


def analyze_calibrate_pi_pulse_based_on_ground(
    *,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Locate the amplitude where the two branches agree.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_calibrate_pi_pulse_based_on_ground`.

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


def plot_calibrate_pi_pulse_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_calibrate_pi_pulse_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    title = f"Ground-referenced pi calibration : {target}"
    return {
        "branches": _core.branches_figure(
            title=title,
            amplitude_range=data["amplitude_range"],
            initial_amplitude=data["initial_amplitude"],
            signal_minus=data["signal"][_core.BRANCH_MINUS],
            signal_plus=data["signal"][_core.BRANCH_PLUS],
            n_rotations=data["n_rotations"],
        ),
        "difference": _core.difference_figure(
            title=f"{title} : difference",
            amplitude_range=data["amplitude_range"],
            initial_amplitude=data["initial_amplitude"],
            analysis=data["fit"],
        ),
        "residuals": _core.residuals_figure(
            title=f"{title} : residuals",
            amplitude_range=data["amplitude_range"],
            initial_amplitude=data["initial_amplitude"],
            analysis=data["fit"],
        ),
        "iq": _core.iq_figure(
            title=f"{title} : IQ",
            raw=data["raw"],
        ),
    }


def display_calibrate_pi_pulse_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the calibration summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `calibrate_pi_pulse_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"calibrate_pi_pulse_based_on_ground_{data['target']}",
        )
    _core.print_summary(data)
    print(f"  ramptime            : {data['ramptime']} ns")
    print(f"  tipping pulse       : {data['tipping_pulse'].duration} ns FlatTop")


def calibrate_pi_pulse_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    duration: float | None = None,
    ramptime: float | None = None,
    n_rotations: int | None = None,
    amplitude_diff_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Calibrate a transition's pi pulse amplitude and store it.

    Every requested target is checked before the first acquisition, so an
    unregistered target or an uncalibrated lower rung stops the whole batch
    rather than leaving the calibration note half written. The targets are then
    calibrated one after another.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    targets : Collection[str] | str | None, optional
        Qubit labels, or any target labels that resolve to qubits. Defaults to
        every qubit that has the transition registered.
    transition : TransitionName
        Transition to calibrate, `"ge"` through `"mn"`.
    duration : float | None, optional
        Pulse duration in ns. Defaults to `_core.DEFAULT_PI_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_PI_RAMPTIME`.
    n_rotations : int | None, optional
        Number of full rotations `n`. Defaults to `_core.DEFAULT_N_ROTATIONS`.
    amplitude_diff_range : ArrayLike | None, optional
        Amplitude offsets from the starting amplitude. Scalar or sequence.
        Defaults to a symmetric sweep around each target's starting amplitude.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the two branch schedules at zero offset. Defaults to
        True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so
        the calibrated amplitude of one target is at
        `result.data[target]["amplitude"]`. It is also written to
        `exp.calib_note` under the transition label.
        A target whose analysis, storage, plotting or display failed still
        appears, carrying whatever the pipeline reached and the exception
        under `"error"`; a target that came through carries no `"error"`
        key at all. Measurement failures are not caught: they are raised.
    """
    if plot is None:
        plot = True

    ladders = _core.preflight_ladders(
        exp,
        targets=targets,
        transition=transition,
        require_rabi_param=True,
    )

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_calibrate_pi_pulse_based_on_ground(run_result=run_result)

    def _store(payload: dict[str, Any]) -> None:
        exp.calib_note.update_pi_param(
            payload["target"],
            {
                "target": payload["target"],
                "duration": payload["duration"],
                "amplitude": payload["amplitude"],
                "tau": payload["ramptime"],
            },
        )

    def _display(result: Result) -> None:
        display_calibrate_pi_pulse_based_on_ground(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        run_result = run_calibrate_pi_pulse_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=transition,
            duration=duration,
            ramptime=ramptime,
            n_rotations=n_rotations,
            amplitude_diff_range=amplitude_diff_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            store=_store,
            plot=plot_calibrate_pi_pulse_based_on_ground,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
