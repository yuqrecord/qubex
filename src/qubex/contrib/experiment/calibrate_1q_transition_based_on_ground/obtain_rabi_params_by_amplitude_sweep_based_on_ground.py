"""
Ground-referenced Rabi experiment that sweeps the drive amplitude.

`obtain_rabi_params_based_on_ground` sweeps the drive *duration* at a fixed
amplitude over `DEFAULT_RABI_TIME_RANGE`, which reaches 400 ns. When the
coherence of the transition is short the arc has decayed long before the sweep
ends, and the fitted frequency is whatever the first surviving fringe implies.
This experiment fixes the duration instead and sweeps the amplitude, so every
point costs the same short dwell.

Nothing in the repository measured this before.
`Experiment.obtain_ampl_rabi_relation` recovers the same slope, but by running a
whole duration sweep at each of ten amplitudes and fitting a line through the
rates. `CalibrationService.calibrate_default_pulse` does sweep the amplitude at
a fixed duration, yet `fitting.fit_ampl_calib_data` keeps only the location of
the extremum and discards the oscillation frequency.

Method
------
The qubit is climbed from |g> to the lower level of the transition with a pi
pulse on every rung below it, a `FlatTop` of *fixed* duration and *swept*
amplitude is played on the transition, the same pi pulses are replayed in
reverse order, and only then is the qubit read out::

    X_ge -> X_ef -> ... -> FlatTop(A) on the transition -> ... -> X_ef -> X_ge

Ideal un-preparation maps `|k> -> |g>` and leaves `|k+1>` alone, so the readout
separates |g> from the transition's *upper* level, exactly as the duration sweep
does. The fit is referenced to a separately measured |g> point.

Writing `T_eff` for the effective duration -- the area of the unit-amplitude
envelope, which is `duration + ramptime` for a `FlatTop` -- the rotation in full
turns at amplitude `A` is `kappa * A * T_eff`, so the signal against amplitude
is a cosine of `f_A = kappa * T_eff` turns per unit amplitude. The fitted `f_A`
therefore gives the slope

    kappa = f_A / T_eff          (GHz per unit drive amplitude)

which is the physically primary result and is reported as
`"rabi_rate_per_amplitude"`.

The stored `RabiParam.frequency` is `kappa * A_cfg`, the rate at the target's
*configured* control amplitude. That is what `PulseService.calc_control_amplitude`
assumes: it forms `rabi_param.frequency / default_amplitude` with
`default_amplitude` taken from `ControlParameters`, never from the amplitude the
measurement happened to use. Because this experiment measures a slope, the
reported rate does not depend on where the sweep stopped.

The amplitudes of a pi and a half-pi pulse at this duration fall straight out of
the same fit as `0.5 / f_A` and `0.25 / f_A`. They are reported but deliberately
not stored: they belong to this envelope and this duration, and the amplitude
calibrations of this package measure their own.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, as for the
  duration sweep.
- This experiment does *not* escape the trade-off between drive strength and
  coherence; it only holds the dwell short and fixed while the drive strength
  varies. A duration sweep at the same raised amplitude reaches the same number
  of turns in the same time. What it buys is that the whole sweep pays one short
  dwell rather than an increasing one.
- For the same reason, shortening `duration` is not free. Holding the number of
  turns fixed while halving the duration doubles the amplitude the sweep has to
  reach, which makes every systematic below worse.
- The fitted slope is biased when the response is not linear in amplitude. The
  AC-Stark shift detunes the drive by roughly `A^2`, so the observed rate becomes
  `sqrt(Omega^2 + Delta^2)` and the oscillation is slightly chirped; leakage to
  the neighbouring levels grows with amplitude, and on an ef or higher rung there
  are two such channels; and the analog chain can compress well below a requested
  amplitude of 1. Fitting a free chirp over a couple of turns cannot separate
  these, so the fit stays a plain cosine. The way to detect the bias is to
  compare `"rabi_rate_per_amplitude"` across nested amplitude ranges.
- The default amplitude range is sized from a *prior* -- a stored `RabiParam`
  when there is one, otherwise the nominal `DEFAULT_RABI_FREQUENCY` -- so the
  number of turns it covers is an estimate, not a guarantee. The analysis reports
  the number actually fitted and the samples per turn, because a high R-squared
  does not rule out an aliased sweep.
- `"reference_signal"` places the independently measured |g> point on the
  normalized axis as a diagnostic of the ladder: it is 1 when the sweep really
  does return to |g>, and above 1 by the fraction of contrast that preparation,
  un-preparation and the fixed dwell cost. It does not constrain the fit. The
  signal at zero amplitude is deliberately *not* used for this -- normalization
  divides by the freely fitted arc amplitude, so that number stays near 1 however
  much contrast was lost.
- `obtain_rabi_params_based_on_ground` stores the rate at the `amplitude` it was
  given, which differs from the rate at `A_cfg` whenever a caller overrides that
  argument. This experiment always reports at `A_cfg`, so the two disagree in
  that case.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_RABI_FREQUENCY
from qubex.experiment.models import RabiParam, Result
from qubex.pulse import FlatTop, PulseSchedule, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground",
    "display_obtain_rabi_params_by_amplitude_sweep_based_on_ground",
    "obtain_rabi_params_by_amplitude_sweep_based_on_ground",
    "plot_obtain_rabi_params_by_amplitude_sweep_based_on_ground",
    "run_obtain_rabi_params_by_amplitude_sweep_based_on_ground",
]

#: Flat-top length of the fixed drive, in ns. Together with
#: `_core.DEFAULT_RABI_RAMPTIME` this makes the effective duration
#: `1 / DEFAULT_RABI_FREQUENCY`, so the configured control amplitude is
#: nominally one full turn.
DEFAULT_DURATION = 1 / DEFAULT_RABI_FREQUENCY - _core.DEFAULT_RABI_RAMPTIME

#: Full turns the derived amplitude range nominally covers.
DEFAULT_N_TURNS = 2.0

#: Points in the derived amplitude range. Same count as `DEFAULT_RABI_TIME_RANGE`.
DEFAULT_N_POINTS = 51

#: Whether to fit a damped cosine. This diverges from `_core.DEFAULT_IS_DAMPED`
#: on purpose: at a fixed duration the decay is not T2'. Decoherence costs a
#: contrast factor that is common to every point, and what does fall off with
#: amplitude -- leakage, compression -- is not an exponential in amplitude.
DEFAULT_IS_DAMPED = False

#: Samples per fitted turn below which the sweep is called out as aliasing-prone.
MIN_POINTS_PER_TURN = 8.0

#: Fitted turns below which the slope is called out as poorly constrained.
MIN_SWEPT_TURNS = 1.0


def _validate_duration(duration: float, ramptime: float) -> None:
    """
    Raise when the fixed drive duration is unusable.

    Parameters
    ----------
    duration : float
        Flat-top length in ns.
    ramptime : float
        `FlatTop` ramp time in ns.
    """
    _core.validate_duration(duration)
    if not np.isfinite(ramptime) or ramptime < 0:
        raise ValueError(f"`ramptime` must be a non-negative number, got {ramptime!r}.")
    period = Waveform.SAMPLING_PERIOD
    played = duration + 2 * ramptime
    remainder = played % period
    if not (np.isclose(remainder, 0) or np.isclose(remainder, period)):
        raise ValueError(
            f"`duration + 2 * ramptime` ({played} ns) must be a multiple of the "
            f"sampling period ({period} ns)."
        )


def _validate_amplitude_range(amplitude_range: NDArray[np.float64]) -> None:
    """
    Raise when the swept amplitudes cannot support the cosine fit.

    `_core.validate_amplitude` cannot be used here: it rejects zero, which this
    sweep deliberately includes.

    Parameters
    ----------
    amplitude_range : NDArray[np.float64]
        Swept drive amplitudes.
    """
    _core.validate_sweep_range(amplitude_range, "amplitude_range")
    if np.any(amplitude_range < 0) or np.any(amplitude_range > 1):
        raise ValueError(
            "`amplitude_range` must lie in [0, 1]; "
            f"got [{amplitude_range.min()}, {amplitude_range.max()}]."
        )
    if np.any(np.diff(amplitude_range) <= 0):
        raise ValueError("`amplitude_range` must be strictly increasing.")
    if amplitude_range.size > 2:
        steps = np.diff(amplitude_range)
        if not np.allclose(steps, steps[0]):
            raise ValueError(
                "`amplitude_range` must be evenly spaced: the fit seeds its "
                "frequency from an FFT, which assumes a uniform grid."
            )
    if amplitude_range.size < _core.MIN_SWEEP_POINTS:
        raise ValueError(
            f"`amplitude_range` must contain at least {_core.MIN_SWEEP_POINTS} "
            f"points, got {amplitude_range.size}."
        )


def _resolve_rate_prior(
    exp: Experiment,
    *,
    target: str,
    control_amplitude: float,
) -> tuple[float, str]:
    """
    Estimate the Rabi rate per unit amplitude before measuring it.

    A stored `RabiParam` is the better prior, and having one means this run is a
    refinement rather than a bootstrap. Falling back on
    `DEFAULT_RABI_FREQUENCY` assumes the configured control amplitude realizes
    the nominal rate.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the Rabi parameters.
    target : str
        Transition label.
    control_amplitude : float
        Configured control amplitude of the transition.

    Returns
    -------
    tuple[float, str]
        Prior rate in GHz per unit amplitude, and where it came from.
    """
    stored = exp.ctx.get_rabi_param(target)
    if stored is not None and np.isfinite(stored.frequency) and stored.frequency > 0:
        return float(stored.frequency) / control_amplitude, "calib_note"
    return DEFAULT_RABI_FREQUENCY / control_amplitude, "nominal_rabi_rate"


def run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    amplitude_range: ArrayLike | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Measure the ground-referenced Rabi oscillation against the drive amplitude.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    amplitude_range : ArrayLike | None, optional
        Drive amplitudes to sweep, in [0, 1]. Must be strictly increasing and
        evenly spaced. Defaults to a range derived from a stored `RabiParam`, or
        from `DEFAULT_RABI_FREQUENCY` when none is stored, that nominally covers
        `DEFAULT_N_TURNS` turns in `DEFAULT_N_POINTS` points, capped at 1.
    duration : float | None, optional
        Flat-top length of the fixed drive in ns. The played duration is
        `duration + 2 * ramptime`. Defaults to `DEFAULT_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the largest amplitude. Defaults to True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by
        `analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground`.
    """
    if duration is None:
        duration = DEFAULT_DURATION
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
    _validate_duration(duration, ramptime)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    # Snapshotted here: the conversion to `RabiParam.frequency` is tied to this
    # value, so a later configuration change must not silently reinterpret it.
    control_amplitude = _core.resolve_amplitudes(
        exp,
        ladders=[ladder],
        amplitudes=None,
    )[ladder.target]
    _core.validate_amplitude(control_amplitude, "control amplitude")

    control_sampling_period = exp.ctx.measurement.sampling_period
    played_duration = duration + 2 * ramptime

    def make_body(amplitude: float) -> Waveform:
        return FlatTop(
            duration=played_duration,
            amplitude=amplitude,
            tau=ramptime,
            sampling_period=control_sampling_period,
        )

    effective_duration = float(
        make_body(1.0).real.sum() * _core.sampling_period_ns(exp)
    )
    if not np.isfinite(effective_duration) or effective_duration <= 0:
        raise ValueError(
            f"The drive envelope of {ladder.target!r} has a non-positive area "
            f"({effective_duration} ns), so no Rabi rate can be derived from it."
        )

    rate_prior, rate_prior_source = _resolve_rate_prior(
        exp,
        target=ladder.target,
        control_amplitude=control_amplitude,
    )
    if amplitude_range is None:
        wanted = DEFAULT_N_TURNS / (rate_prior * effective_duration)
        amplitude_range_capped = wanted > 1.0
        amplitudes = np.linspace(0.0, min(1.0, wanted), DEFAULT_N_POINTS)
        amplitude_range_source = rate_prior_source
    else:
        amplitudes = _core.resolve_sweep_range(amplitude_range)
        amplitude_range_capped = False
        amplitude_range_source = "explicit"
    _validate_amplitude_range(amplitudes)

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)

    def sequence(amplitude: float) -> PulseSchedule:
        return _core.build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=prep_pulses,
            body=make_body(float(amplitude)),
        )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {"largest amplitude": sequence(float(np.max(amplitudes)))},
            title=f"Ground-referenced amplitude Rabi : {ladder.target}",
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
        sweep_range=amplitudes,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    span = float(amplitudes[-1] - amplitudes[0])
    return {
        "qubit": ladder.qubit,
        "transition": ladder.transition,
        "target": ladder.target,
        "lower_labels": ladder.lower_labels,
        "amplitude_range": amplitudes,
        "duration": duration,
        "ramptime": ramptime,
        "played_duration": played_duration,
        "effective_duration": effective_duration,
        "control_amplitude": control_amplitude,
        "amplitude_range_source": amplitude_range_source,
        "amplitude_range_capped": amplitude_range_capped,
        "nominal_rate_per_amplitude": rate_prior,
        "nominal_swept_turns": rate_prior * effective_duration * span,
        "reference_point": reference_point,
        "raw": raw,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
    *,
    run_result: dict[str, Any],
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Fit the amplitude sweep and convert its slope into Rabi parameters.

    The fitted frequency is in turns per unit amplitude, so it is divided by the
    effective duration to give the slope in GHz per unit amplitude and then
    multiplied by the configured control amplitude to give the rate that
    `PulseService.calc_control_amplitude` expects to find in
    `RabiParam.frequency`.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by
        `run_obtain_rabi_params_by_amplitude_sweep_based_on_ground`.
    is_damped : bool | None, optional
        Whether to fit a damped cosine. The decay length is then in units of
        amplitude and is phenomenological, not a coherence time. Defaults to
        `DEFAULT_IS_DAMPED`.
    fit_threshold : float | None, optional
        Minimum R-squared below which the Rabi parameters are reported as NaN,
        matching `obtain_rabi_params_based_on_ground`. Defaults to
        `_core.DEFAULT_FIT_THRESHOLD`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the fit, the fitted slope and the resulting
        `RabiParam`.
    """
    if is_damped is None:
        is_damped = DEFAULT_IS_DAMPED
    if fit_threshold is None:
        fit_threshold = _core.DEFAULT_FIT_THRESHOLD

    target = run_result["target"]
    amplitudes = np.asarray(run_result["amplitude_range"], dtype=np.float64)
    raw = np.asarray(run_result["raw"], dtype=np.complex128)
    effective_duration = float(run_result["effective_duration"])
    control_amplitude = float(run_result["control_amplitude"])
    span = float(amplitudes[-1] - amplitudes[0])

    fit = fitting.fit_rabi(
        target=target,
        times=amplitudes,
        data=raw,
        reference_point=run_result["reference_point"],
        plot=False,
        is_damped=is_damped,
        # The fitter's own default is 10 000, which is a time in ns. On an
        # amplitude axis the only sensible scale is the swept span.
        tau_est=span if is_damped else None,
    )

    normalized: NDArray[np.float64] | None = None
    residuals: NDArray[np.float64] | None = None
    reference_signal = np.nan
    turns_per_amplitude = np.nan
    turns_per_amplitude_err = np.nan
    swept_turns = np.nan
    points_per_turn = np.nan
    r2 = np.nan

    if fit.status is not FitStatus.ERROR:
        r2 = float(fit["r2"])
        turns_per_amplitude = float(fit["frequency"])
        turns_per_amplitude_err = float(fit["frequency_err"])
        swept_turns = turns_per_amplitude * span
        if swept_turns > 0:
            points_per_turn = amplitudes.size / swept_turns

        # `normalize` is (imag(rotated) - offset) / amplitude, so the normalized
        # model is the bare cosine. It is rebuilt from the fit rather than from
        # the `RabiParam` below, which is NaN once the threshold bites.
        rotated = raw * np.exp(-1j * float(fit["angle"]))
        normalized = np.asarray(
            (rotated.imag - float(fit["offset"])) / float(fit["amplitude"]),
            dtype=np.float64,
        )
        residuals = normalized - _normalized_model(
            amplitudes,
            turns_per_amplitude=turns_per_amplitude,
            phase=float(fit["phase"]),
            tau=float(fit["tau"]) if is_damped else None,
        )
        # Where the independently measured |g> point lands on the same
        # normalized axis. The signal at zero amplitude is *not* the
        # diagnostic to use here: the fit divides by its own freely
        # fitted arc amplitude, so that value is close to 1 however much
        # contrast the ladder lost. The reference point is outside the
        # fit, so it moves above 1 by exactly the fraction lost.
        reference_rotated = complex(run_result["reference_point"]) * np.exp(
            -1j * float(fit["angle"])
        )
        reference_signal = (reference_rotated.imag - float(fit["offset"])) / float(
            fit["amplitude"]
        )

    if fit.status is FitStatus.ERROR or r2 < fit_threshold:
        rabi_param = RabiParam.nan(target=target)
        rate_per_amplitude = np.nan
        rate_per_amplitude_err = np.nan
        pi_amplitude = np.nan
        hpi_amplitude = np.nan
    else:
        rate_per_amplitude = turns_per_amplitude / effective_duration
        rate_per_amplitude_err = turns_per_amplitude_err / effective_duration
        pi_amplitude = 0.5 / turns_per_amplitude
        hpi_amplitude = 0.25 / turns_per_amplitude
        rabi_param = RabiParam(
            target=target,
            amplitude=float(fit["amplitude"]),
            frequency=rate_per_amplitude * control_amplitude,
            phase=float(fit["phase"]),
            offset=float(fit["offset"]),
            noise=float(fit["noise"]),
            angle=float(fit["angle"]),
            distance=float(fit["distance"]),
            r2=r2,
            reference_phase=float(fit["reference_phase"]),
        )

    return {
        **run_result,
        "fit": fit,
        "fit_threshold": fit_threshold,
        "is_damped": is_damped,
        "r2": r2,
        "normalized": normalized,
        "residuals": residuals,
        "turns_per_amplitude": turns_per_amplitude,
        "turns_per_amplitude_err": turns_per_amplitude_err,
        "rabi_rate_per_amplitude": rate_per_amplitude,
        "rabi_rate_per_amplitude_err": rate_per_amplitude_err,
        "pi_amplitude": pi_amplitude,
        "hpi_amplitude": hpi_amplitude,
        "swept_turns": swept_turns,
        "points_per_turn": points_per_turn,
        "reference_signal": reference_signal,
        "rabi_param": rabi_param,
    }


def _normalized_model(
    amplitudes: NDArray[np.float64],
    *,
    turns_per_amplitude: float,
    phase: float,
    tau: float | None,
) -> NDArray[np.float64]:
    """
    Evaluate the fitted oscillation after normalization.

    Parameters
    ----------
    amplitudes : NDArray[np.float64]
        Drive amplitudes to evaluate at.
    turns_per_amplitude : float
        Fitted frequency in turns per unit amplitude.
    phase : float
        Fitted phase in radians.
    tau : float | None
        Fitted decay length in units of amplitude, or `None` when undamped.

    Returns
    -------
    NDArray[np.float64]
        Normalized signal the fit predicts.
    """
    cosine = np.cos(2 * np.pi * turns_per_amplitude * amplitudes + phase)
    if tau is None:
        return np.asarray(cosine, dtype=np.float64)
    return np.asarray(np.exp(-amplitudes / tau) * cosine, dtype=np.float64)


def plot_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the amplitude-swept Rabi experiment.

    The fitter's own figure is not reused: it labels its x axis
    "Drive duration (ns)" and prints the fitted frequency as MHz, both of which
    are wrong once the swept quantity is an amplitude.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by
        `analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    figures: dict[str, go.Figure] = {}
    amplitudes = np.asarray(data["amplitude_range"], dtype=np.float64)
    normalized = data["normalized"]

    if normalized is not None:
        fine = np.linspace(float(amplitudes[0]), float(amplitudes[-1]), 1000)
        model = _normalized_model(
            fine,
            turns_per_amplitude=data["turns_per_amplitude"],
            phase=float(data["fit"]["phase"]),
            tau=float(data["fit"]["tau"]) if data["is_damped"] else None,
        )
        fig = viz.make_figure()
        fig.add_scatter(x=fine, y=model, mode="lines", name="Fit")
        fig.add_scatter(
            x=amplitudes,
            y=normalized,
            mode="markers",
            name="Data",
            error_y=dict(
                type="constant",
                value=float(data["fit"]["noise"]) / float(data["fit"]["amplitude"]),
            ),
        )
        reference_signal = data["reference_signal"]
        if np.isfinite(reference_signal):
            fig.add_hline(
                y=reference_signal,
                line_dash="dot",
                line_width=1,
                annotation_text="|g>",
            )
        pi_amplitude = data["pi_amplitude"]
        if (
            np.isfinite(pi_amplitude)
            and amplitudes[0] <= pi_amplitude <= amplitudes[-1]
        ):
            fig.add_vline(x=pi_amplitude, line_dash="dash")
        rate = data["rabi_rate_per_amplitude"]
        fig.update_layout(
            title=dict(
                text=f"Ground-referenced amplitude Rabi : {target}",
                subtitle=dict(
                    text=(
                        f"{rate * 1e3:.4g} MHz per unit amplitude, "
                        f"{data['rabi_param'].frequency * 1e3:.4g} MHz at "
                        f"A={data['control_amplitude']:.4g}"
                    ),
                    font=dict(size=13, family="monospace"),
                ),
            ),
            xaxis_title="Drive amplitude (arb. units)",
            yaxis_title="Normalized signal",
            width=600,
            height=400,
            margin=dict(t=80),
        )
        figures["rabi"] = fig

        residual_fig = viz.make_figure()
        residual_fig.add_scatter(
            x=amplitudes,
            y=data["residuals"],
            mode="lines+markers",
            name="residual",
        )
        residual_fig.add_hline(y=0, line_dash="dot", line_width=1)
        residual_fig.update_layout(
            title=f"Ground-referenced amplitude Rabi : {target} : residuals",
            xaxis_title="Drive amplitude (arb. units)",
            yaxis_title="Residual",
            width=600,
            height=400,
        )
        figures["residuals"] = residual_fig

    figures["iq"] = _core.iq_figure(
        title=f"Ground-referenced amplitude Rabi : {target} : IQ",
        raw=data["raw"],
    )
    return figures


def display_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the amplitude-sweep summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by
        `obtain_rabi_params_by_amplitude_sweep_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=(
                "obtain_rabi_params_by_amplitude_sweep_based_on_ground_"
                f"{data['target']}"
            ),
        )
    rabi_param: RabiParam = data["rabi_param"]
    amplitudes = np.asarray(data["amplitude_range"], dtype=np.float64)
    source = {
        "explicit": "given by the caller",
        "calib_note": "derived from the stored Rabi parameters",
        "nominal_rabi_rate": "derived from the nominal Rabi rate",
    }[data["amplitude_range_source"]]

    _core.print_ladder(data)
    print(f"  duration            : {data['duration']} ns flat top")
    print(f"  played duration     : {data['played_duration']} ns")
    print(f"  effective duration  : {data['effective_duration']:.6g} ns")
    print(f"  control amplitude   : {data['control_amplitude']:.9g}")
    print(
        f"  amplitude range     : {amplitudes[0]:.6g} to {amplitudes[-1]:.6g} "
        f"in {amplitudes.size} points ({source})"
    )
    if data["amplitude_range_capped"]:
        print(
            "  The derived range was capped at an amplitude of 1, so it covers "
            "fewer turns than requested."
        )
    print(
        f"  turns per amplitude : {data['turns_per_amplitude']:.6g} "
        f"+- {data['turns_per_amplitude_err']:.1g}"
    )
    print(
        f"  rate per amplitude  : {data['rabi_rate_per_amplitude'] * 1e3:.6g} "
        f"+- {data['rabi_rate_per_amplitude_err'] * 1e3:.1g} MHz"
    )
    print(
        f"  rabi frequency      : {rabi_param.frequency * 1e3:.6g} MHz "
        f"at the control amplitude"
    )
    print(f"  pi amplitude        : {data['pi_amplitude']:.6g}")
    print(f"  half-pi amplitude   : {data['hpi_amplitude']:.6g}")
    print(f"  swept turns         : {data['swept_turns']:.4g}")
    print(f"  points per turn     : {data['points_per_turn']:.4g}")
    print(
        f"  reference signal    : {data['reference_signal']:.4f} "
        "(1 for an ideal ladder, above 1 once it loses contrast)"
    )
    print(f"  arc amplitude       : {rabi_param.amplitude:.6g}")
    print(f"  fit R2              : {data['r2']:.4f}")

    if data["swept_turns"] < MIN_SWEPT_TURNS:
        print(
            f"  The sweep covers less than {MIN_SWEPT_TURNS:.3g} turns, so the "
            "slope is poorly constrained. Widen `amplitude_range`, or lengthen "
            "`duration` if the amplitude is already at its ceiling."
        )
    if data["points_per_turn"] < MIN_POINTS_PER_TURN:
        print(
            f"  Fewer than {MIN_POINTS_PER_TURN:.3g} points per turn: the fit "
            "may be aliased, and a high R2 would not reveal it. Add points or "
            "narrow `amplitude_range`."
        )
    if not np.isfinite(rabi_param.r2) or rabi_param.r2 < data["fit_threshold"]:
        print(
            f"  Rabi parameters were NOT stored: R2 is below the threshold "
            f"({data['fit_threshold']:.3g})."
        )


def obtain_rabi_params_by_amplitude_sweep_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    amplitude_range: ArrayLike | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Measure and store the Rabi parameters of one transition from an amplitude sweep.

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
    amplitude_range : ArrayLike | None, optional
        Drive amplitudes to sweep, shared by every target. Defaults to a range
        derived per target from its stored `RabiParam`, or from
        `DEFAULT_RABI_FREQUENCY` when none is stored.
    duration : float | None, optional
        Flat-top length of the fixed drive in ns. Defaults to
        `DEFAULT_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    is_damped : bool | None, optional
        Whether to fit a damped cosine. Defaults to `DEFAULT_IS_DAMPED`, which
        is False: at a fixed duration the decay is not a coherence time.
    fit_threshold : float | None, optional
        Minimum R-squared for the parameters to be stored. Defaults to
        `_core.DEFAULT_FIT_THRESHOLD`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the largest amplitude. Defaults to True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so the
        fitted slope of one target is at
        `result.data[target]["rabi_rate_per_amplitude"]` and the parameters at
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

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
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
        display_obtain_rabi_params_by_amplitude_sweep_based_on_ground(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        run_result = run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=transition,
            amplitude_range=amplitude_range,
            duration=duration,
            ramptime=ramptime,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            store=_store,
            plot=plot_obtain_rabi_params_by_amplitude_sweep_based_on_ground,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
