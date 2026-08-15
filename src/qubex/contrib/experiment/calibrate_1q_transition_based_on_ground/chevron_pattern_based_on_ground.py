"""
Ground-referenced chevron pattern for any transition of the level ladder.

`Experiment.chevron_pattern` cannot run on a transition label: it looks the drive
amplitude up in the qubit-keyed `params.control_amplitude` and it groups targets
with `create_qubit_subgroups`, both of which raise for `"Q00-hi"`. It also drives
the target straight from |g>.

Method
------
For every point of the `detuning x duration` grid the qubit is climbed to the
lower level of the transition, a detuned `FlatTop` is played on the transition,
the ladder is climbed back down, and the qubit is read out::

    X_ge -> X_ef -> ... -> FlatTop(T).detuned(d) -> ... -> X_ef -> X_ge

The carrier is set once through `frequencies={<transition label>: frequency}` and
the detuning is applied on the waveform, so the NCO is not retuned per point.

The 2-D grid is flattened into a 1-D acquisition-index sweep, then reshaped by
the analysis. Each detuning row is fitted with `fitting.fit_rabi` for its Rabi
rate, and the rates are fitted against
`sqrt(f_rabi^2 + (f_control - f_resonance)^2)` with `fitting.fit_detuned_rabi` to
locate the resonance.

This is the frequency-finding step of the bootstrap, so it deliberately does
*not* need a stored `RabiParam` for the transition: the heat map is a PCA
projection of the raw IQ grid.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse.
- Nothing is persisted. The transition frequency belongs to the registered
  `Target`, not to `calib_note`, and changing it can imply backend NCO state, so
  the fitted resonance is returned and printed and applying it stays an explicit
  act of the caller (`Experiment.register_transition_target(frequency=...)`).
- A detuned-Rabi fit that fails outright raises instead of reporting NaN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import FlatTop, PulseSchedule, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_chevron_pattern_based_on_ground",
    "chevron_pattern_based_on_ground",
    "display_chevron_pattern_based_on_ground",
    "plot_chevron_pattern_based_on_ground",
    "run_chevron_pattern_based_on_ground",
]

#: Detuned-Rabi fit quality below which the resonance estimate is called out.
RELIABLE_DETUNED_RABI_R2 = 0.5


def _project_grid(raw_grid: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Project a gridded IQ chevron onto its principal axis.

    Parameters
    ----------
    raw_grid : NDArray[np.complex128]
        Raw IQ values shaped `(detuning, time)`.

    Returns
    -------
    NDArray[np.float64]
        Projected values shaped `(time, detuning)`, ready for a heat map.
    """
    values = raw_grid.ravel()
    centered = values - np.mean(values)
    try:
        _, _, vh = np.linalg.svd(
            np.column_stack([centered.real, centered.imag]),
            full_matrices=False,
        )
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0])

    projected = axis[0] * (raw_grid.real - np.mean(values.real)) + axis[1] * (
        raw_grid.imag - np.mean(values.imag)
    )
    if np.mean(projected[:, 0]) < np.mean(projected):
        projected = -projected
    return np.asarray(projected.T, dtype=np.float64)


def run_chevron_pattern_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    detuning_range: ArrayLike,
    time_range: ArrayLike,
    amplitude: float,
    frequency: float,
    ramptime: float,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
) -> dict[str, Any]:
    """
    Measure the ground-referenced chevron grid of one transition.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    detuning_range : ArrayLike
        Drive detunings in GHz, relative to `frequency`. Scalar or sequence.
    time_range : ArrayLike
        Flat-top durations in ns. Scalar or sequence.
    amplitude : float
        Drive amplitude on the transition.
    frequency : float
        Carrier frequency of the transition drive in GHz.
    ramptime : float
        `FlatTop` ramp time in ns. The played duration is `T + 2 * ramptime`.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot_pulse_schedule : bool, optional
        Whether to draw the schedule of the longest, undetuned drive.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_chevron_pattern_based_on_ground`.
    """
    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    _core.validate_amplitude(amplitude)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"`frequency` must be a positive number, got {frequency!r}.")

    detunings = _core.resolve_sweep_range(detuning_range)
    _core.validate_sweep_range(detunings, "detuning_range")
    times = _core.resolve_sweep_range(time_range)
    _core.validate_time_range(times)
    _core.validate_effective_durations(times, ramptime)

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    control_sampling_period = exp.ctx.measurement.sampling_period
    sweep_points = [
        (float(detuning), float(duration))
        for detuning in detunings
        for duration in times
    ]

    def make_body(*, detuning: float, duration: float) -> Waveform:
        return FlatTop(
            duration=duration + 2 * ramptime,
            amplitude=amplitude,
            tau=ramptime,
            sampling_period=control_sampling_period,
        ).detuned(detuning)

    def sequence(point_index: float) -> PulseSchedule:
        detuning, duration = sweep_points[round(float(point_index))]
        return _core.build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=prep_pulses,
            body=make_body(detuning=detuning, duration=duration),
        )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {
                "longest drive": _core.build_ground_referenced_schedule(
                    ladder=ladder,
                    prep_pulses=prep_pulses,
                    body=make_body(detuning=0.0, duration=float(np.max(times))),
                )
            },
            title=f"Ground-referenced chevron : {ladder.target}",
        )

    raw, _ = _core.sweep_ground_referenced(
        exp,
        ladder=ladder,
        rabi_param=None,
        sequence=sequence,
        sweep_range=np.arange(len(sweep_points)),
        frequencies={ladder.target: frequency},
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    return {
        "qubit": ladder.qubit,
        "transition": ladder.transition,
        "target": ladder.target,
        "lower_labels": ladder.lower_labels,
        "detuning_range": detunings,
        "time_range": times,
        "effective_time_range": times + ramptime,
        "control_frequencies": frequency + detunings,
        "frequency": frequency,
        "amplitude": amplitude,
        "ramptime": ramptime,
        "raw": raw,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_chevron_pattern_based_on_ground(
    *,
    run_result: dict[str, Any],
    rabi_fit_r2_threshold: float,
) -> dict[str, Any]:
    """
    Fit the chevron grid and locate the resonance.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_chevron_pattern_based_on_ground`.
    rabi_fit_r2_threshold : float
        Per-detuning Rabi fits below this R² are dropped before the detuned-Rabi
        fit. Use `0.0` to keep every point.

    Returns
    -------
    dict[str, Any]
        The run payload plus the projected grid, the per-detuning Rabi rates and
        the resonance estimate.
    """
    target = run_result["target"]
    detunings = run_result["detuning_range"]
    times = run_result["effective_time_range"]
    raw_grid = np.asarray(run_result["raw"], dtype=np.complex128).reshape(
        detunings.size,
        times.size,
    )

    rabi_rates = np.full(detunings.size, np.nan, dtype=np.float64)
    rabi_fit_r2 = np.full(detunings.size, np.nan, dtype=np.float64)
    for index in range(detunings.size):
        fit = fitting.fit_rabi(
            target=target,
            times=times,
            data=raw_grid[index],
            plot=False,
            warn_low_r2=False,
        )
        if fit.status is FitStatus.ERROR:
            continue
        rabi_rates[index] = float(fit["frequency"])
        rabi_fit_r2[index] = float(fit["r2"])

    selected_rates = np.where(
        np.isfinite(rabi_fit_r2) & (rabi_fit_r2 >= rabi_fit_r2_threshold),
        rabi_rates,
        np.nan,
    )
    if np.count_nonzero(np.isfinite(selected_rates)) < 3:
        raise ValueError(
            f"Only {np.count_nonzero(np.isfinite(selected_rates))} detunings of "
            f"{target!r} produced a Rabi fit above R2 {rabi_fit_r2_threshold:.3g}. "
            "Lower the threshold, widen `time_range`, or raise `amplitude`."
        )

    detuned_fit = fitting.fit_detuned_rabi(
        target=target,
        control_frequencies=run_result["control_frequencies"],
        rabi_frequencies=selected_rates,
        plot=False,
        warn_low_r2=False,
    )
    _core.require_fit_success(detuned_fit, target=target, what="detuned Rabi fit")
    resonant_frequency = float(detuned_fit["f_resonance"])
    if not np.isfinite(resonant_frequency):
        raise ValueError(
            f"The detuned Rabi fit for {target!r} produced a non-finite "
            "resonance. Widen `detuning_range` or improve the ladder "
            "preparation."
        )

    return {
        **run_result,
        "raw_grid": raw_grid,
        "chevron_data": _project_grid(raw_grid),
        "rabi_rates": rabi_rates,
        "rabi_fit_r2": rabi_fit_r2,
        "selected_rabi_rates": selected_rates,
        "rabi_fit_r2_threshold": rabi_fit_r2_threshold,
        "detuned_fit": detuned_fit,
        "resonant_frequency": resonant_frequency,
        "resonant_frequency_err": float(detuned_fit["f_resonance_err"]),
        "detuned_fit_r2": float(detuned_fit["r2"]),
    }


def plot_chevron_pattern_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the chevron experiment.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_chevron_pattern_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=data["control_frequencies"],
            y=data["time_range"],
            z=data["chevron_data"],
            colorscale="Viridis",
        )
    )
    fig.add_vline(x=data["resonant_frequency"], line_dash="dash")
    fig.update_layout(
        title=dict(
            text=f"Ground-referenced chevron : {target}",
            subtitle=dict(
                text=(
                    f"amplitude={data['amplitude']:.6g}, "
                    f"resonance={data['resonant_frequency']:.6f} GHz"
                ),
                font=dict(size=13, family="monospace"),
            ),
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Time (ns)",
        width=600,
        height=400,
        margin=dict(t=80),
    )

    figures: dict[str, go.Figure] = {"chevron": fig}
    detuned_figure = data["detuned_fit"].figure
    if detuned_figure is not None:
        figures["detuned_rabi"] = detuned_figure
    return figures


def display_chevron_pattern_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the chevron summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `chevron_pattern_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"chevron_pattern_based_on_ground_{data['target']}",
        )
    _core.print_ladder(data)
    print(f"  amplitude           : {data['amplitude']:.9g}")
    print(f"  carrier frequency   : {data['frequency']:.6f} GHz")
    print(
        f"  resonant frequency  : {data['resonant_frequency']:.6f} "
        f"+- {data['resonant_frequency_err']:.1g} GHz"
    )
    print(
        f"  detuning from carrier: "
        f"{(data['resonant_frequency'] - data['frequency']) * 1e3:+.3f} MHz"
    )
    print(f"  detuned Rabi fit R2 : {data['detuned_fit_r2']:.4f}")
    print(
        f"  detunings used      : "
        f"{np.count_nonzero(np.isfinite(data['selected_rabi_rates']))}"
        f" / {data['detuning_range'].size}"
    )
    if data["detuned_fit_r2"] < RELIABLE_DETUNED_RABI_R2:
        print(
            "  The detuned Rabi fit is unreliable; treat the resonance as a "
            "rough estimate."
        )
    print(
        "  Nothing was stored. Apply the resonance explicitly, for example with "
        "`Experiment.register_transition_target(..., frequency=...)`."
    )


def chevron_pattern_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    detuning_range: ArrayLike,
    time_range: ArrayLike,
    amplitude: float,
    frequency: float,
    ramptime: float,
    rabi_fit_r2_threshold: float,
    n_shots: int,
    shot_interval: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    """
    Measure a ground-referenced chevron pattern and locate the resonance.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    detuning_range : ArrayLike
        Drive detunings in GHz, relative to `frequency`. Scalar or sequence.
    time_range : ArrayLike
        Flat-top durations in ns. Scalar or sequence.
    amplitude : float
        Drive amplitude on the transition.
    frequency : float
        Carrier frequency of the transition drive in GHz.
    ramptime : float
        `FlatTop` ramp time in ns.
    rabi_fit_r2_threshold : float
        Per-detuning Rabi fits below this R² are dropped before the detuned-Rabi
        fit.
    n_shots : int
        Shots per acquisition.
    shot_interval : float
        Interval between shots in ns.
    plot : bool, optional
        Whether to show the figures.
    plot_pulse_schedule : bool, optional
        Whether to draw the schedule of the longest, undetuned drive.

    Returns
    -------
    Result
        Payload, figures and the fitted resonance. Nothing is written to
        `exp.calib_note`.
    """
    run_result = run_chevron_pattern_based_on_ground(
        exp,
        target=target,
        transition=transition,
        detuning_range=detuning_range,
        time_range=time_range,
        amplitude=amplitude,
        frequency=frequency,
        ramptime=ramptime,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    data = analyze_chevron_pattern_based_on_ground(
        run_result=run_result,
        rabi_fit_r2_threshold=rabi_fit_r2_threshold,
    )
    figures = plot_chevron_pattern_based_on_ground(data)

    result = Result(data=data, figures=figures)
    display_chevron_pattern_based_on_ground(result, plot)
    return result
