"""
Ground-referenced Rabi experiment for any transition of the level ladder.

`Experiment.obtain_rabi_params` drives the target straight from |g>, and
`obtain_ef_rabi_params` hard-codes the single ef rung and references the fit to
the |e> point of the ge arc. Neither works above ef.

Method
------
The qubit is climbed from |g> to the lower level of the transition with a pi
pulse on every rung below it, a `FlatTop` of swept duration is played on the
transition, the same pi pulses are replayed in reverse order, and only then is
the qubit read out::

    X_ge -> X_ef -> ... -> FlatTop(T) on the transition -> ... -> X_ef -> X_ge

Ideal un-preparation maps `|k> -> |g>` and leaves `|k+1>` alone, so the readout
separates |g> from the transition's *upper* level. The measured IQ trace is an
arc from the ground-state point to the upper level's point, and the fit is
referenced to a separately measured |g> point, exactly as
`MeasurementService.rabi_experiment` does for ge.

The fitted `RabiParam` is stored under the *transition* label (for example
`"Q00-fh"`), which `CalibrationNote` accepts unchanged, and it is what every
calibration in this package uses to normalize its signal.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse. That is
  checked, not silently worked around: `Experiment.pulse.x180` would otherwise
  fall back to two half-pi pulses and change both the fidelity and the duration
  of the ladder.
- Preparation and un-preparation errors reduce the contrast of the arc. That is
  an affine loss, so it costs precision but does not bias the fitted frequency.
- The transition target's registered frequency is used. Run
  `chevron_pattern_based_on_ground` first when it is not yet accurate.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.models import RabiParam, Result
from qubex.pulse import FlatTop, PulseSchedule, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_obtain_rabi_params_based_on_ground",
    "display_obtain_rabi_params_based_on_ground",
    "obtain_rabi_params_based_on_ground",
    "plot_obtain_rabi_params_based_on_ground",
    "run_obtain_rabi_params_based_on_ground",
]


def run_obtain_rabi_params_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    time_range: ArrayLike | None = None,
    amplitude: float | None = None,
    ramptime: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplification: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Measure the ground-referenced Rabi oscillation of one transition.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    time_range : ArrayLike | None, optional
        Flat-top durations in ns. Scalar or sequence. Defaults to
        `_core.DEFAULT_TIME_RANGE`.
    amplitude : float | None, optional
        Drive amplitude on the transition. Defaults to the qubit's control
        amplitude scaled for the transition.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. The played duration is `T + 2 * ramptime`.
        Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    readout_amplification : bool | None, optional
        Whether to insert readout amplification pulses into both the |g>
        reference measurement and the sweep, so the reference point shares the
        readout of the arc it anchors. `None` is passed on as is, so
        `MeasurementService.measure` resolves it to False.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the longest duration. Defaults to True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_obtain_rabi_params_based_on_ground`.
    """
    if time_range is None:
        time_range = _core.DEFAULT_TIME_RANGE
    if ramptime is None:
        ramptime = _core.DEFAULT_RABI_RAMPTIME
    if n_shots is None:
        n_shots = _core.DEFAULT_N_SHOTS
    if shot_interval is None:
        shot_interval = _core.DEFAULT_SHOT_INTERVAL
    if plot_pulse_schedule is None:
        plot_pulse_schedule = True

    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    if amplitude is None:
        amplitude = _core.resolve_amplitudes(
            exp,
            ladders=[ladder],
            amplitudes=None,
        )[ladder.target]
    _core.validate_amplitude(amplitude)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    times = _core.resolve_sweep_range(time_range)
    _core.validate_time_range(times)
    _core.validate_effective_durations(times, ramptime)

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    control_sampling_period = exp.ctx.measurement.sampling_period

    def make_body(duration: float) -> Waveform:
        return FlatTop(
            duration=duration + 2 * ramptime,
            amplitude=amplitude,
            tau=ramptime,
            sampling_period=control_sampling_period,
        )

    def sequence(duration: float) -> PulseSchedule:
        return _core.build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=prep_pulses,
            body=make_body(float(duration)),
        )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {"longest drive": sequence(float(np.max(times)))},
            title=f"Ground-referenced Rabi : {ladder.target}",
        )

    reference_points = exp.measurement_service.obtain_reference_points(
        [ladder.qubit],
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplification=readout_amplification,
    )["iq"]
    reference_point = reference_points[ladder.qubit]

    raw, _ = _core.sweep_ground_referenced(
        exp,
        ladder=ladder,
        rabi_param=None,
        sequence=sequence,
        sweep_range=times,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplification=readout_amplification,
    )

    return {
        "qubit": ladder.qubit,
        "transition": ladder.transition,
        "target": ladder.target,
        "lower_labels": ladder.lower_labels,
        "time_range": times,
        "effective_time_range": times + ramptime,
        "amplitude": amplitude,
        "ramptime": ramptime,
        "reference_point": reference_point,
        "raw": raw,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
        "readout_amplification": readout_amplification,
    }


def analyze_obtain_rabi_params_based_on_ground(
    *,
    run_result: dict[str, Any],
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Fit the ground-referenced Rabi oscillation.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_obtain_rabi_params_based_on_ground`.
    is_damped : bool | None, optional
        Whether to fit a damped cosine. Defaults to
        `_core.DEFAULT_IS_DAMPED`.
    fit_threshold : float | None, optional
        Minimum R² below which the parameters are reported as NaN, matching
        `MeasurementService.rabi_experiment`. Defaults to
        `_core.DEFAULT_FIT_THRESHOLD`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the fit and the resulting `RabiParam`.
    """
    if is_damped is None:
        is_damped = _core.DEFAULT_IS_DAMPED
    if fit_threshold is None:
        fit_threshold = _core.DEFAULT_FIT_THRESHOLD

    target = run_result["target"]
    fit = fitting.fit_rabi(
        target=target,
        times=run_result["effective_time_range"],
        data=run_result["raw"],
        reference_point=run_result["reference_point"],
        plot=False,
        is_damped=is_damped,
    )
    if fit.status is FitStatus.ERROR or float(fit["r2"]) < fit_threshold:
        rabi_param = RabiParam.nan(target=target)
    else:
        rabi_param = RabiParam(
            target=target,
            amplitude=float(fit["amplitude"]),
            frequency=float(fit["frequency"]),
            phase=float(fit["phase"]),
            offset=float(fit["offset"]),
            noise=float(fit["noise"]),
            angle=float(fit["angle"]),
            distance=float(fit["distance"]),
            r2=float(fit["r2"]),
            reference_phase=float(fit["reference_phase"]),
        )
    return {
        **run_result,
        "fit": fit,
        "fit_threshold": fit_threshold,
        "is_damped": is_damped,
        "rabi_param": rabi_param,
    }


def plot_obtain_rabi_params_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the Rabi experiment.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_obtain_rabi_params_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    figures: dict[str, go.Figure] = {
        "iq": _core.iq_figure(
            title=f"Ground-referenced Rabi : {data['target']} : IQ",
            raw=data["raw"],
        ),
    }
    rabi_figure = data["fit"].figure
    if rabi_figure is not None:
        figures["rabi"] = rabi_figure
    return figures


def display_obtain_rabi_params_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the Rabi summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `obtain_rabi_params_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"obtain_rabi_params_based_on_ground_{data['target']}",
        )
    rabi_param: RabiParam = data["rabi_param"]
    _core.print_ladder(data)
    print(f"  amplitude           : {data['amplitude']:.9g}")
    print(f"  ramptime            : {data['ramptime']} ns")
    print(f"  rabi frequency      : {rabi_param.frequency * 1e3:.6g} MHz")
    print(f"  arc amplitude       : {rabi_param.amplitude:.6g}")
    print(f"  fit R2              : {rabi_param.r2:.4f}")
    if not np.isfinite(rabi_param.r2) or rabi_param.r2 < data["fit_threshold"]:
        print(
            f"  Rabi parameters were NOT stored: R2 is below the threshold "
            f"({data['fit_threshold']:.3g})."
        )


def obtain_rabi_params_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    time_range: ArrayLike | None = None,
    amplitudes: Mapping[str, float] | float | None = None,
    ramptime: float | None = None,
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplification: bool | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Measure and store the ground-referenced Rabi parameters of one transition.

    Every requested target is checked before the first acquisition, so an
    unregistered target or an uncalibrated lower rung stops the whole batch
    rather than leaving the calibration note half written. The targets are then
    measured one after another.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    targets : Collection[str] | str | None, optional
        Qubit labels, or any target labels that resolve to qubits. Defaults to
        every qubit that has the transition registered.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    time_range : ArrayLike | None, optional
        Flat-top durations in ns. Scalar or sequence. Defaults to
        `_core.DEFAULT_TIME_RANGE`.
    amplitudes : Mapping[str, float] | float | None, optional
        Drive amplitude per target, one amplitude shared by every target, or
        `None` to derive each from the qubit's control amplitude. A mapping may
        be keyed by transition label or by qubit label.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    is_damped : bool | None, optional
        Whether to fit a damped cosine. Defaults to `_core.DEFAULT_IS_DAMPED`.
    fit_threshold : float | None, optional
        Minimum R² for the parameters to be stored. Defaults to
        `_core.DEFAULT_FIT_THRESHOLD`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    readout_amplification : bool | None, optional
        Whether to insert readout amplification pulses into every acquisition.
        Defaults to False.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the longest duration. Defaults to True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so the
        fitted parameters of one target are at
        `result.data[target]["rabi_param"]`. They are also written to
        `exp.calib_note` under the transition label when the fit clears
        `fit_threshold`.
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
        require_rabi_param=False,
    )
    resolved_amplitudes = _core.resolve_amplitudes(
        exp,
        ladders=ladders,
        amplitudes=amplitudes,
    )

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_obtain_rabi_params_based_on_ground(
            run_result=run_result,
            is_damped=is_damped,
            fit_threshold=fit_threshold,
        )

    def _store(payload: dict[str, Any]) -> None:
        exp.ctx.store_rabi_params(
            {payload["target"]: payload["rabi_param"]},
            r2_threshold=payload["fit_threshold"],
        )

    def _display(result: Result) -> None:
        display_obtain_rabi_params_based_on_ground(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        run_result = run_obtain_rabi_params_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=transition,
            time_range=time_range,
            amplitude=resolved_amplitudes[ladder.target],
            ramptime=ramptime,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_amplification=readout_amplification,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            store=_store,
            plot=plot_obtain_rabi_params_based_on_ground,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
