"""
Ground-referenced chevron pattern measured by sweeping the drive amplitude.

`chevron_pattern_based_on_ground` sweeps `detuning x duration`, fits every
detuning row with `fitting.fit_rabi`, and feeds the per-row rates to
`fitting.fit_detuned_rabi`, which locates the resonance from the hyperbola
`sqrt(f_rabi^2 + (f_control - f_resonance)^2)`. The duration sweep reaches 400 ns,
so on a transition whose coherence is short the arc has decayed before the
resonance is resolved.

This experiment fixes the duration and sweeps the amplitude instead, as
`obtain_rabi_params_by_amplitude_sweep_based_on_ground` does for the Rabi rate.
Every point of the grid costs the same short dwell.

Why the analysis is not the same analysis
-----------------------------------------
The duration chevron works because in the time domain the row oscillation
frequency *is* the generalized rate `sqrt(Omega^2 + delta^2)`, which grows with
the detuning. In the amplitude domain the asymptotic slope is `kappa` whatever
the detuning, so the row frequency carries almost no detuning information while
the cosine model itself falls apart. Writing `kappa = 0.0625` GHz per unit
amplitude and `T_eff = 80` ns, a cosine fitted to each row of the exact
two-level response gives

    detuning    fitted turns per amplitude    kappa * T_eff    R^2
      0 MHz               5.0000                  5.0000      1.0000
      2 MHz               5.0148                  5.0000      0.9925
      4 MHz               5.0559                  5.0000      0.8882
      6 MHz               5.1049                  5.0000      0.6027

The frequency moves by one per cent while the fit quality collapses, and the
quantity has units of inverse amplitude rather than GHz. Passing it to
`fit_detuned_rabi` would be meaningless, so this module does not use that fitter
at all.

Two further estimators look plausible and are both wrong here, for the same
reason: the `sin^2` factor of

    z(A) = 1 - 2 * [Omega^2/(Omega^2+delta^2)] * sin^2(pi T sqrt(Omega^2+delta^2))

vanishes whenever the drive completes a whole turn, which inverts the sign of
the feature at resonance.

- The oscillation power of a row is *not* reliably peaked on resonance. Over an
  amplitude window of [0.18, 0.22] it peaks near `+-8.5` MHz instead, where it
  is fourteen times its on-resonance value. A peak fit would be confidently
  wrong by that much.
- A lineshape fitted along the detuning axis at one amplitude is not reliable
  either. `fitting.fit_sqrt_lorentzian` describes the steady state, whereas a
  fixed duration gives the Rabi lineshape, whose centre flips between a peak and
  a dip as the turn count passes each half integer. At exactly one turn the
  fitted centre lands 12.7 MHz away.

What survives is the symmetry itself: the pattern is even in the detuning about
the resonance, to machine precision, whatever the amplitude window. So the
resonance is located as the grid's line of mirror symmetry -- see
`_symmetry_centre`.

Method
------
For every point of the `detuning x amplitude` grid the qubit is climbed to the
lower level of the transition, a `FlatTop` of fixed duration and swept amplitude
is played on the transition, the ladder is climbed back down, and the qubit is
read out::

    X_ge -> X_ef -> ... -> FlatTop(A).detuned(d) -> ... -> X_ef -> X_ge

The carrier is set once through `frequencies={<transition label>: frequency}` and
the detuning is applied on the waveform, so the NCO is not retuned per point.
The 2-D grid is flattened into a 1-D acquisition-index sweep, then reshaped by
the analysis.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse.
- **This is a short-dwell diagnostic and a bootstrap, not the default precision
  frequency measurement.** A `51 x 51` grid is about fifty times the acquisitions
  of a 51-point spectroscopy scan at the same shots per point. What the expense
  buys is amplitude-resolved evidence -- whether coherent oscillations exist at
  all, and whether strong-drive shifts move the centre -- rather than one more
  resonance number. Weak-drive spectroscopy is the cheaper way to get only a
  frequency.
- A stronger drive makes the symmetry feature *broader*, the opposite of the
  duration chevron where more turns help. For a square pulse the half-width of
  the row-power feature goes roughly 3.7 MHz at half a turn, 3.7 at one turn,
  10.3 at two and 18.0 at four. The default therefore aims at one turn, not the
  two that the amplitude-swept Rabi uses.
- The reflection needs the same number of offsets on both sides of a candidate,
  so only the inner part of the swept detuning range can hold the centre. A
  resonance outside it is reported as a failure, not extrapolated.
- The quoted uncertainty comes from the curvature of the symmetry objective. It
  is a precision estimate, not an accuracy guarantee: the limit on accuracy is
  the amplitude-dependent Stark shift, which `"centre_drift"` is there to expose.
- Nothing is persisted. The transition frequency belongs to the registered
  `Target`, not to `calib_note`, so applying the result stays an explicit act of
  the caller (`Experiment.register_transition_target(frequency=...)`).
- A future refinement would port the matched transform of
  `contrib.experiment.chevron_matched_transform` with `tau` replaced by the fixed
  `T_eff` and trial `(f_resonance, kappa)`. It assumes a single resonance across
  the whole grid, so it would average the Stark shift away rather than show it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_RABI_FREQUENCY
from qubex.experiment.models import Result
from qubex.pulse import FlatTop, PulseSchedule, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_chevron_pattern_by_amplitude_sweep_based_on_ground",
    "chevron_pattern_by_amplitude_sweep_based_on_ground",
    "display_chevron_pattern_by_amplitude_sweep_based_on_ground",
    "plot_chevron_pattern_by_amplitude_sweep_based_on_ground",
    "run_chevron_pattern_by_amplitude_sweep_based_on_ground",
]

#: Flat-top length of the fixed drive, in ns. With `_core.DEFAULT_RABI_RAMPTIME`
#: this makes the effective duration `1 / DEFAULT_RABI_FREQUENCY`, the same
#: choice `obtain_rabi_params_by_amplitude_sweep_based_on_ground` makes.
DEFAULT_DURATION = 1 / DEFAULT_RABI_FREQUENCY - _core.DEFAULT_RABI_RAMPTIME

#: Full turns the derived amplitude range nominally covers. Deliberately lower
#: than the amplitude-swept Rabi's two: a stronger drive broadens the symmetry
#: feature, so more turns cost resolution here instead of buying it.
DEFAULT_N_TURNS = 1.0

#: Points along the amplitude axis of the derived grid.
DEFAULT_N_AMPLITUDES = 51

#: Points along the detuning axis of the derived grid.
DEFAULT_N_DETUNINGS = 51

#: Detuning half-span, in units of `max(1 / T_eff, kappa * A_max)`.
DEFAULT_DETUNING_SPAN_FACTOR = 2.0

#: Fraction of the detuning axis used as the reflection half-window. The centre
#: can therefore only be located within the middle half of the swept range.
DEFAULT_SYMMETRY_WINDOW_FRACTION = 0.25

#: Centre drift, in units of the quoted uncertainty, above which the resonance
#: is called out as amplitude dependent.
MAX_DRIFT_SIGMA = 3.0

#: Smallest detuning axis the symmetry estimator can work on: it needs the
#: reflection offsets on both sides of a candidate plus three interior
#: candidates for the parabola.
MIN_DETUNINGS = 5

#: Relative spread of the symmetry objective below which the grid is declared to
#: carry no line of symmetry at all. A featureless grid gives about 0.2, a real
#: chevron tens.
MIN_SYMMETRY_CONTRAST = 1.0


def _project_grid(raw_grid: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Project a gridded IQ chevron onto its principal axis.

    This mirrors the private helper of `chevron_pattern_based_on_ground`, which
    is deliberately left untouched.

    Parameters
    ----------
    raw_grid : NDArray[np.complex128]
        Raw IQ values shaped `(detuning, amplitude)`.

    Returns
    -------
    NDArray[np.float64]
        Projected values shaped `(amplitude, detuning)`, ready for a heat map.
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


def _symmetry_centre(
    *,
    frequencies: NDArray[np.float64],
    field: NDArray[np.float64],
    half_window: int,
) -> dict[str, Any]:
    """
    Locate the line of mirror symmetry of a grid along its frequency axis.

    The objective is the reflected difference

        L(c) = sum_k sum_A [ S(c + k dF, A) - S(c - k dF, A) ]^2

    evaluated with a *fixed* number of offsets `k`, so every candidate centre is
    supported by the same number of terms and an edge cannot look good merely by
    having less to disagree with. Each row has its mean over the amplitude axis
    removed, which suppresses an additive frequency-dependent background; the
    rows are deliberately not normalized individually, because their relative
    contrast is signal.

    Parameters
    ----------
    frequencies : NDArray[np.float64]
        Uniformly spaced control frequencies, in GHz.
    field : NDArray[np.float64]
        Projected signal shaped `(frequency, amplitude)`.
    half_window : int
        Number of reflection offsets `k`.

    Returns
    -------
    dict[str, Any]
        `"centre"`, `"centre_err"`, `"contrast"`, `"residual"`, `"objective"`
        and `"candidates"`.

    Raises
    ------
    ValueError
        When the grid carries no usable line of symmetry, or when the best
        candidate sits at the edge of the searchable range.
    """
    # Removing each row's mean over the amplitude axis suppresses an additive
    # frequency-dependent background. With a single amplitude that mean *is*
    # the signal, so the removal would zero the field entirely.
    if field.shape[1] > 1:
        signal = field - field.mean(axis=1, keepdims=True)
    else:
        signal = field
    n_frequencies = frequencies.size
    offsets = np.arange(1, half_window + 1)
    candidates = np.arange(half_window, n_frequencies - half_window)
    objective = np.array(
        [
            float(np.sum((signal[index + offsets] - signal[index - offsets]) ** 2))
            for index in candidates
        ],
        dtype=np.float64,
    )

    floor = float(objective.min())
    ceiling = float(objective.max())
    # An objective that is zero everywhere means the grid does not vary with
    # the detuning at all, which is the degenerate no-information case rather
    # than a perfect symmetry axis.
    contrast = (
        0.0 if ceiling <= 0 else (np.inf if floor <= 0 else ceiling / floor - 1.0)
    )
    if contrast < MIN_SYMMETRY_CONTRAST:
        raise ValueError(
            f"The grid carries no line of symmetry: the reflection objective "
            f"varies by only {contrast:.3g} across the searchable range, below "
            f"{MIN_SYMMETRY_CONTRAST:.3g}. The drive produced no detuning-"
            "dependent structure; raise the amplitude, lengthen `duration`, or "
            "check that the transition is driven at all."
        )

    best = int(np.argmin(objective))
    if best == 0 or best == objective.size - 1:
        raise ValueError(
            "The symmetry axis sits at the edge of the searchable detuning "
            f"range ({frequencies[candidates[best]]:.6f} GHz). The reflection "
            "needs the same number of offsets on both sides, so only the inner "
            "half of the sweep can hold the centre. Re-centre `frequency` on "
            "that value, or widen `detuning_range`."
        )

    low, middle, high = objective[best - 1], objective[best], objective[best + 1]
    curvature = 0.5 * (low - 2 * middle + high)
    step = float(frequencies[1] - frequencies[0])
    shift = 0.0 if curvature <= 0 else -0.5 * (high - low) / (2 * curvature)
    centre = float(frequencies[candidates[best]] + shift * step)

    # The residuals summed at the minimum are what estimate the noise, so the
    # degrees of freedom are those of a single candidate, not of every candidate.
    dof = max(half_window * field.shape[1] - 1, 1)
    variance = floor / dof
    centre_err = (
        float(np.sqrt(variance * step**2 / curvature)) if curvature > 0 else np.nan
    )
    return {
        "centre": centre,
        "centre_err": centre_err,
        "contrast": contrast,
        "residual": float(np.sqrt(floor / (half_window * field.shape[1]))),
        "objective": objective,
        "candidates": np.asarray(frequencies[candidates], dtype=np.float64),
    }


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


def _validate_uniform_range(values: NDArray[np.float64], name: str) -> None:
    """
    Raise when a swept axis is not a usable uniform grid.

    Both axes have to be uniform: the amplitude axis because the Rabi fit of the
    resonant row seeds its frequency from an FFT, and the detuning axis because
    the symmetry estimator reflects it by index.

    Parameters
    ----------
    values : NDArray[np.float64]
        Swept values.
    name : str
        Argument name, used in the messages.
    """
    _core.validate_sweep_range(values, name)
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"`{name}` must be strictly increasing.")
    if values.size > 2:
        steps = np.diff(values)
        if not np.allclose(steps, steps[0]):
            raise ValueError(
                f"`{name}` must be evenly spaced: the analysis reflects and "
                "Fourier-seeds it, and both assume a uniform grid."
            )
    if values.size < _core.MIN_SWEEP_POINTS:
        raise ValueError(
            f"`{name}` must contain at least {_core.MIN_SWEEP_POINTS} points, "
            f"got {values.size}."
        )


def _validate_amplitude_range(amplitude_range: NDArray[np.float64]) -> None:
    """
    Raise when the swept amplitudes are unusable.

    `_core.validate_amplitude` cannot be used: it rejects zero, which this grid
    deliberately includes.

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
    _validate_uniform_range(amplitude_range, "amplitude_range")


def _resolve_rate_prior(
    exp: Experiment,
    *,
    target: str,
    control_amplitude: float,
) -> tuple[float, str]:
    """
    Estimate the Rabi rate per unit amplitude before measuring anything.

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


def run_chevron_pattern_by_amplitude_sweep_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    detuning_range: ArrayLike | None = None,
    amplitude_range: ArrayLike | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    frequency: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Measure the `detuning x amplitude` grid of one transition at a fixed duration.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    detuning_range : ArrayLike | None, optional
        Drive detunings in GHz, relative to `frequency`. Must be strictly
        increasing and evenly spaced. Defaults to a symmetric range whose half
        span is `DEFAULT_DETUNING_SPAN_FACTOR * max(1 / T_eff, kappa * A_max)`.
    amplitude_range : ArrayLike | None, optional
        Drive amplitudes to sweep, in [0, 1]. Must be strictly increasing and
        evenly spaced. Defaults to a range that nominally covers
        `DEFAULT_N_TURNS` turns, capped at 1.
    duration : float | None, optional
        Flat-top length of the fixed drive in ns. The played duration is
        `duration + 2 * ramptime`. Defaults to `DEFAULT_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    frequency : float | None, optional
        Carrier frequency of the transition drive in GHz. Defaults to the
        frequency the transition target is registered with.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the largest, undetuned drive. Defaults
        to True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by
        `analyze_chevron_pattern_by_amplitude_sweep_based_on_ground`.
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
    if frequency is None:
        frequency = _core.resolve_frequencies(
            exp,
            ladders=[ladder],
            frequencies=None,
        )[ladder.target]
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"`frequency` must be a positive number, got {frequency!r}.")

    control_amplitude = _core.resolve_amplitudes(
        exp,
        ladders=[ladder],
        amplitudes=None,
    )[ladder.target]
    _core.validate_amplitude(control_amplitude, "control amplitude")

    control_sampling_period = exp.ctx.measurement.sampling_period
    played_duration = duration + 2 * ramptime

    def make_body(*, detuning: float, amplitude: float) -> Waveform:
        return FlatTop(
            duration=played_duration,
            amplitude=amplitude,
            tau=ramptime,
            sampling_period=control_sampling_period,
        ).detuned(detuning)

    effective_duration = float(
        make_body(detuning=0.0, amplitude=1.0).real.sum()
        * _core.sampling_period_ns(exp)
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
    wanted_amplitude = DEFAULT_N_TURNS / (rate_prior * effective_duration)
    if amplitude_range is None:
        amplitude_range_capped = wanted_amplitude > 1.0
        amplitudes = np.linspace(
            0.0,
            min(1.0, wanted_amplitude),
            DEFAULT_N_AMPLITUDES,
        )
        amplitude_range_source = rate_prior_source
    else:
        amplitudes = _core.resolve_sweep_range(amplitude_range)
        amplitude_range_capped = False
        amplitude_range_source = "explicit"
    _validate_amplitude_range(amplitudes)

    if detuning_range is None:
        half_span = DEFAULT_DETUNING_SPAN_FACTOR * max(
            1.0 / effective_duration,
            rate_prior * float(amplitudes[-1]),
        )
        detunings = np.linspace(-half_span, half_span, DEFAULT_N_DETUNINGS)
        detuning_range_source = rate_prior_source
    else:
        detunings = _core.resolve_sweep_range(detuning_range)
        detuning_range_source = "explicit"
    _validate_uniform_range(detunings, "detuning_range")
    if detunings.size < MIN_DETUNINGS:
        raise ValueError(
            f"`detuning_range` must contain at least {MIN_DETUNINGS} points "
            f"for the symmetry estimator, got {detunings.size}."
        )

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    sweep_points = [
        (float(detuning), float(amplitude))
        for detuning in detunings
        for amplitude in amplitudes
    ]

    def sequence(point_index: float) -> PulseSchedule:
        detuning, amplitude = sweep_points[round(float(point_index))]
        return _core.build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=prep_pulses,
            body=make_body(detuning=detuning, amplitude=amplitude),
        )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {
                "largest amplitude": _core.build_ground_referenced_schedule(
                    ladder=ladder,
                    prep_pulses=prep_pulses,
                    body=make_body(
                        detuning=0.0,
                        amplitude=float(np.max(amplitudes)),
                    ),
                )
            },
            title=f"Ground-referenced amplitude chevron : {ladder.target}",
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
        "amplitude_range": amplitudes,
        "control_frequencies": frequency + detunings,
        "frequency": frequency,
        "duration": duration,
        "ramptime": ramptime,
        "played_duration": played_duration,
        "effective_duration": effective_duration,
        "control_amplitude": control_amplitude,
        "amplitude_range_source": amplitude_range_source,
        "amplitude_range_capped": amplitude_range_capped,
        "detuning_range_source": detuning_range_source,
        "nominal_rate_per_amplitude": rate_prior,
        "raw": raw,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_chevron_pattern_by_amplitude_sweep_based_on_ground(
    *,
    run_result: dict[str, Any],
    symmetry_window: int | None = None,
) -> dict[str, Any]:
    """
    Locate the resonance from the mirror symmetry of the grid.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by
        `run_chevron_pattern_by_amplitude_sweep_based_on_ground`.
    symmetry_window : int | None, optional
        Number of reflection offsets. Larger values use more of the grid but
        shrink the range of candidate centres. Defaults to
        `DEFAULT_SYMMETRY_WINDOW_FRACTION` of the detuning axis.

    Returns
    -------
    dict[str, Any]
        The run payload plus the projected grid, the resonance and the
        diagnostics.

    Raises
    ------
    ValueError
        When the grid carries no usable line of symmetry, or when the symmetry
        axis falls outside the searchable part of the swept range.
    """
    target = run_result["target"]
    detunings = np.asarray(run_result["detuning_range"], dtype=np.float64)
    amplitudes = np.asarray(run_result["amplitude_range"], dtype=np.float64)
    control_frequencies = np.asarray(
        run_result["control_frequencies"], dtype=np.float64
    )
    effective_duration = float(run_result["effective_duration"])
    raw_grid = np.asarray(run_result["raw"], dtype=np.complex128).reshape(
        detunings.size,
        amplitudes.size,
    )
    # `(amplitude, detuning)`, which is also the orientation a heat map wants.
    chevron_data = _project_grid(raw_grid)
    field = np.asarray(chevron_data.T, dtype=np.float64)

    if symmetry_window is None:
        symmetry_window = round(DEFAULT_SYMMETRY_WINDOW_FRACTION * detunings.size)
    symmetry_window = max(1, min(symmetry_window, (detunings.size - 3) // 2))

    symmetry = _symmetry_centre(
        frequencies=control_frequencies,
        field=field,
        half_window=symmetry_window,
    )
    resonant_frequency = symmetry["centre"]

    # Diagnostics. The row power is reported because it is informative about
    # where the signal is, but it is NOT the estimator: on a narrow amplitude
    # window its maximum can sit several MHz off resonance.
    row_power = np.var(field, axis=1)

    nested_centres: dict[str, float] = {"full": resonant_frequency}
    half = amplitudes.size // 2
    if half >= 2:
        try:
            nested_centres["lower_half"] = _symmetry_centre(
                frequencies=control_frequencies,
                field=field[:, :half],
                half_window=symmetry_window,
            )["centre"]
        except ValueError:
            nested_centres["lower_half"] = np.nan
    else:
        nested_centres["lower_half"] = np.nan
    centre_drift = nested_centres["lower_half"] - resonant_frequency

    # The amplitude sweep is a true cosine only on resonance, so the Rabi slope
    # is read off the row nearest the located centre and nowhere else.
    resonant_index = int(np.argmin(np.abs(control_frequencies - resonant_frequency)))
    rate_per_amplitude = np.nan
    row_fit: Any = None
    if amplitudes.size >= _core.MIN_SWEEP_POINTS:
        row_fit = fitting.fit_rabi(
            target=target,
            times=amplitudes,
            data=raw_grid[resonant_index],
            plot=False,
            warn_low_r2=False,
        )
        if row_fit.status is not FitStatus.ERROR:
            rate_per_amplitude = float(row_fit["frequency"]) / effective_duration

    return {
        **run_result,
        "raw_grid": raw_grid,
        "chevron_data": chevron_data,
        "symmetry_window": symmetry_window,
        "symmetry_objective": symmetry["objective"],
        "symmetry_candidates": symmetry["candidates"],
        "symmetry_contrast": symmetry["contrast"],
        "reflection_residual": symmetry["residual"],
        "resonant_frequency": resonant_frequency,
        "resonant_frequency_err": symmetry["centre_err"],
        "detuning_from_carrier": resonant_frequency - float(run_result["frequency"]),
        "row_power": row_power,
        "nested_centres": nested_centres,
        "centre_drift": centre_drift,
        "resonant_row_index": resonant_index,
        "resonant_row_fit": row_fit,
        "rabi_rate_per_amplitude": rate_per_amplitude,
    }


def plot_chevron_pattern_by_amplitude_sweep_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the amplitude-swept chevron.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by
        `analyze_chevron_pattern_by_amplitude_sweep_based_on_ground`.

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
            y=data["amplitude_range"],
            z=data["chevron_data"],
            colorscale="Viridis",
        )
    )
    fig.add_vline(x=data["resonant_frequency"], line_dash="dash")
    fig.update_layout(
        title=dict(
            text=f"Ground-referenced amplitude chevron : {target}",
            subtitle=dict(
                text=(
                    f"duration={data['duration']:.6g} ns, "
                    f"resonance={data['resonant_frequency']:.6f} GHz "
                    "(projected IQ, not population)"
                ),
                font=dict(size=13, family="monospace"),
            ),
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Drive amplitude (arb. units)",
        width=600,
        height=400,
        margin=dict(t=80),
    )

    symmetry_fig = viz.make_figure()
    symmetry_fig.add_scatter(
        x=data["symmetry_candidates"],
        y=data["symmetry_objective"],
        mode="lines+markers",
        name="reflected difference",
    )
    symmetry_fig.add_vline(x=data["resonant_frequency"], line_dash="dash")
    symmetry_fig.update_layout(
        title=dict(
            text=f"Ground-referenced amplitude chevron : {target} : symmetry",
            subtitle=dict(
                text=(
                    f"contrast={data['symmetry_contrast']:.3g}, "
                    f"residual={data['reflection_residual']:.3g}"
                ),
                font=dict(size=13, family="monospace"),
            ),
        ),
        xaxis_title="Candidate resonance (GHz)",
        yaxis_title="Reflected difference",
        width=600,
        height=400,
        margin=dict(t=80),
    )

    return {
        "chevron": fig,
        "symmetry": symmetry_fig,
        "iq": _core.iq_figure(
            title=f"Ground-referenced amplitude chevron : {target} : IQ",
            raw=data["raw"],
        ),
    }


def display_chevron_pattern_by_amplitude_sweep_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the amplitude-chevron summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by
        `chevron_pattern_by_amplitude_sweep_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=(
                f"chevron_pattern_by_amplitude_sweep_based_on_ground_{data['target']}"
            ),
        )
    sources = {
        "explicit": "given by the caller",
        "calib_note": "derived from the stored Rabi parameters",
        "nominal_rabi_rate": "derived from the nominal Rabi rate",
    }
    amplitudes = np.asarray(data["amplitude_range"], dtype=np.float64)
    detunings = np.asarray(data["detuning_range"], dtype=np.float64)

    _core.print_ladder(data)
    print(f"  duration            : {data['duration']} ns flat top")
    print(f"  played duration     : {data['played_duration']} ns")
    print(f"  effective duration  : {data['effective_duration']:.6g} ns")
    print(f"  control amplitude   : {data['control_amplitude']:.9g}")
    print(
        f"  amplitude range     : {amplitudes[0]:.6g} to {amplitudes[-1]:.6g} "
        f"in {amplitudes.size} points "
        f"({sources[data['amplitude_range_source']]})"
    )
    if data["amplitude_range_capped"]:
        print(
            "  The derived amplitude range was capped at 1, so it covers fewer "
            "turns than requested."
        )
    print(
        f"  detuning range      : {detunings[0] * 1e3:+.3f} to "
        f"{detunings[-1] * 1e3:+.3f} MHz in {detunings.size} points "
        f"({sources[data['detuning_range_source']]})"
    )
    print(f"  carrier frequency   : {data['frequency']:.6f} GHz")
    print(
        f"  resonant frequency  : {data['resonant_frequency']:.6f} "
        f"+- {data['resonant_frequency_err']:.1g} GHz"
    )
    print(f"  detuning from carrier: {data['detuning_from_carrier'] * 1e3:+.3f} MHz")
    print(f"  symmetry contrast   : {data['symmetry_contrast']:.4g}")
    print(f"  reflection residual : {data['reflection_residual']:.4g}")
    print(
        f"  centre drift        : {data['centre_drift'] * 1e3:+.3f} MHz "
        "(lower half of the amplitude range vs all of it)"
    )
    print(
        f"  rate per amplitude  : "
        f"{data['rabi_rate_per_amplitude'] * 1e3:.6g} MHz per unit amplitude "
        "(from the resonant row only)"
    )
    drift = float(data["centre_drift"])
    error = float(data["resonant_frequency_err"])
    if np.isfinite(drift) and np.isfinite(error) and error > 0:
        if abs(drift) > MAX_DRIFT_SIGMA * error:
            print(
                f"  The centre moves by {abs(drift) / error:.1f} times its own uncertainty "
                "between the lower half of the amplitude range and all of it, so "
                "the resonance is amplitude dependent -- most likely a Stark "
                "shift. The number above is a drive-weighted average, not the "
                "undriven transition frequency. Re-run over a lower amplitude "
                "range to approach it."
            )
    print(
        "  The uncertainty above is the precision of the symmetry axis, not the "
        "accuracy of the transition frequency: an amplitude-dependent Stark "
        "shift moves the axis itself, which is what the centre drift exposes."
    )
    print(
        "  `row_power` is reported for inspection but is NOT the estimator: on a "
        "narrow amplitude window its maximum can sit several MHz off resonance."
    )
    print(
        "  Nothing was stored. Apply the resonance explicitly, for example with "
        "`Experiment.register_transition_target(..., frequency=...)`."
    )


def chevron_pattern_by_amplitude_sweep_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    detuning_range: ArrayLike | None = None,
    amplitude_range: ArrayLike | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    frequencies: Mapping[str, float] | float | None = None,
    symmetry_window: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Measure the amplitude-swept chevron pattern of one transition.

    Every requested target is checked before the first acquisition, so an
    unregistered target or an uncalibrated lower rung stops the whole batch. The
    targets are then measured one after another.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    targets : Collection[str] | str | None, optional
        Qubit labels, or any target labels that resolve to qubits. Defaults to
        every qubit that has the transition registered.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    detuning_range : ArrayLike | None, optional
        Drive detunings in GHz, shared by every target. Defaults to a range
        derived per target.
    amplitude_range : ArrayLike | None, optional
        Drive amplitudes to sweep, shared by every target. Defaults to a range
        derived per target.
    duration : float | None, optional
        Flat-top length of the fixed drive in ns. Defaults to
        `DEFAULT_DURATION`.
    ramptime : float | None, optional
        `FlatTop` ramp time in ns. Defaults to `_core.DEFAULT_RABI_RAMPTIME`.
    frequencies : Mapping[str, float] | float | None, optional
        Carrier frequency in GHz per target, one frequency shared by every
        target, or `None` to use the frequency each transition target is
        registered with.
    symmetry_window : int | None, optional
        Number of reflection offsets used by the analysis. Defaults to
        `DEFAULT_SYMMETRY_WINDOW_FRACTION` of the detuning axis.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the largest, undetuned drive. Defaults
        to True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so the
        resonance of one target is at
        `result.data[target]["resonant_frequency"]`.
        Nothing is written to `exp.calib_note`.
        A target whose analysis, plotting or display failed still appears,
        carrying whatever the pipeline reached and the exception under
        `"error"`; a target that came through carries no `"error"` key at all.
        Measurement failures are not caught: they are raised.
    """
    if plot is None:
        plot = True

    ladders = _core.preflight_ladders(
        exp,
        targets=targets,
        transition=transition,
        require_rabi_param=False,
    )
    resolved_frequencies = _core.resolve_frequencies(
        exp,
        ladders=ladders,
        frequencies=frequencies,
    )

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_chevron_pattern_by_amplitude_sweep_based_on_ground(
            run_result=run_result,
            symmetry_window=symmetry_window,
        )

    def _display(result: Result) -> None:
        display_chevron_pattern_by_amplitude_sweep_based_on_ground(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        run_result = run_chevron_pattern_by_amplitude_sweep_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=transition,
            detuning_range=detuning_range,
            amplitude_range=amplitude_range,
            duration=duration,
            ramptime=ramptime,
            frequency=resolved_frequencies[ladder.target],
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            # The transition frequency belongs to the registered target, so this
            # experiment deliberately persists nothing.
            store=None,
            plot=plot_chevron_pattern_by_amplitude_sweep_based_on_ground,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
