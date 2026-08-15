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
    time_range: ArrayLike,
    amplitude: float,
    ramptime: float,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
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
    time_range : ArrayLike
        Flat-top durations in ns. Scalar or sequence.
    amplitude : float
        Drive amplitude on the transition.
    ramptime : float
        `FlatTop` ramp time in ns. The played duration is `T + 2 * ramptime`.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot_pulse_schedule : bool, optional
        Whether to draw the schedule of the longest duration.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_obtain_rabi_params_based_on_ground`.
    """
    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
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
    }


def analyze_obtain_rabi_params_based_on_ground(
    *,
    run_result: dict[str, Any],
    is_damped: bool,
    fit_threshold: float,
) -> dict[str, Any]:
    """
    Fit the ground-referenced Rabi oscillation.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_obtain_rabi_params_based_on_ground`.
    is_damped : bool
        Whether to fit a damped cosine.
    fit_threshold : float
        Minimum R² below which the parameters are reported as NaN, matching
        `MeasurementService.rabi_experiment`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the fit and the resulting `RabiParam`.
    """
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
    target: str,
    transition: TransitionName,
    time_range: ArrayLike,
    amplitude: float,
    ramptime: float,
    is_damped: bool,
    fit_threshold: float,
    n_shots: int,
    shot_interval: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    """
    Measure and store the ground-referenced Rabi parameters of one transition.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    time_range : ArrayLike
        Flat-top durations in ns. Scalar or sequence.
    amplitude : float
        Drive amplitude on the transition.
    ramptime : float
        `FlatTop` ramp time in ns.
    is_damped : bool
        Whether to fit a damped cosine.
    fit_threshold : float
        Minimum R² for the parameters to be stored.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot : bool, optional
        Whether to show the figures.
    plot_pulse_schedule : bool, optional
        Whether to draw the schedule of the longest duration.

    Returns
    -------
    Result
        Payload, figures and the fitted `RabiParam`. The parameters are also
        written to `exp.calib_note` under the transition label when the fit
        clears `fit_threshold`.
    """
    run_result = run_obtain_rabi_params_based_on_ground(
        exp,
        target=target,
        transition=transition,
        time_range=time_range,
        amplitude=amplitude,
        ramptime=ramptime,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    data = analyze_obtain_rabi_params_based_on_ground(
        run_result=run_result,
        is_damped=is_damped,
        fit_threshold=fit_threshold,
    )
    figures = plot_obtain_rabi_params_based_on_ground(data)

    exp.ctx.store_rabi_params(
        {data["target"]: data["rabi_param"]},
        r2_threshold=fit_threshold,
    )

    result = Result(data=data, figures=figures)
    display_obtain_rabi_params_based_on_ground(result, plot)
    return result
