"""
Ground-referenced DRAG half-pi calibration for any transition.

`Experiment.calibrate_drag_hpi_pulse` locates the amplitude at the *extremum* of
a fitted cosine, drives the target straight from |g>, seeds `beta` from the ge
anharmonicity and delegates the beta step to `Experiment.calibrate_drag_beta`,
which normalizes through the ge Rabi arc and silently substitutes `beta = 0.0`
when its polynomial fit finds no root. None of that survives above ge, so both
passes are reimplemented here, referenced to the ground state.

Method
------
Amplitude and beta are correlated, so they are refined by coordinate descent over
`n_iterations` passes.

*Amplitude pass.* The qubit is climbed to the lower level of the transition, the
DRAG pulse under test opens the train with a sign that selects the branch and is
then repeated `4 n` more times, the ladder is climbed back down and the qubit is
read out::

    X_ge -> ... -> [-+drag] -> [drag] * 4n -> ... -> X_ge

The two branches therefore realize `(4n - 1) theta` and `(4n + 1) theta`. With a
per-pulse angle error `eps` the normalized signals are `sin((4n - 1) eps)` and
`-sin((4n + 1) eps)`, whose difference

    d(eps) = 2 sin(4 n eps) cos(eps)  ~  8 n eps

crosses zero exactly at `eps = 0`. Inverting the leading pulse rather than
dropping two keeps `4n + 1` pulses of identical duration in both branches, which
matters here more than anywhere else in the package: leakage accumulates per
pulse and a branch-dependent *offset* is exactly what moves the crossing.

*Beta pass.* The same ladder wraps the error-amplification sequence of
`CalibrationService.calibrate_drag_beta`::

    X90+ , [X90- , X90+] * n_turns , Y90-

The closing `Y90-` analyzer is the transition's stored DRAG half-pi when there is
one, and otherwise its uncorrected half-pi. It is resolved once per pass, before
the sweep, so it does not depend on the beta being swept. The normalized signal
is fitted with a polynomial and its root is the calibrated beta; a non-finite
root, or one outside the swept range, raises rather than defaulting to zero.

The analyzer's own axis error does not merely cost contrast: it offsets the
signal, and the fit takes that signal's root, so to first order the beta comes
back biased by `(phi - a) / ((2n + 1) k)` for an in-plane phase error `phi` and
an out-of-plane tilt `a`. An uncorrected half-pi carries `beta = 0`, the largest
beta error there is, which is why the corrected one is preferred.

Here the analyzer is the very pulse under calibration, so each pass analyzes
with the beta the previous one found. That is a contraction rather than a
runaway: in the symmetric model the error obeys `e_(j+1) = -e_j / (2n + 1)`, so
the default two passes at `n_turns = 1` leave a ninth of the seed error. Note
that the first pass analyzes with the amplitude it has just measured but only
the *seed* beta - the entry it reads was written moments earlier by the
amplitude pass.

Starting point
--------------
The amplitude and beta the sweep starts from are the calibration stored under the
transition label when its `duration` matches, and otherwise a Rabi-rate estimate
for the amplitude and `_core.estimate_drag_beta` for beta. The stored amplitude
additionally has to have been measured with the requested `type`: the rotation
angle follows the area under the envelope, so an amplitude from another envelope
lands far outside the sweep. A stored beta carries over either way. For a transition
`k -> k+1` there are two leakage channels rather than one, with opposite-sign
detunings, so the ge formula `-drag_coeff / alpha` does not carry over; see
`_core.estimate_drag_beta`. It is only a seed — the beta pass measures the real
value.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, and the
  transition must have stored Rabi parameters and some stored half-pi pulse,
  corrected or not, to analyze the beta pass with. Run
  `calibrate_hpi_pulse_based_on_ground` first.
- `amplitude_diff_range` must stay inside `A0 / (4 n)`, beyond which the
  difference signal grows extra zero crossings. This is validated, not clipped.
- `beta_range` is a set of offsets *from the current beta*, matching
  `Experiment.calibrate_drag_beta`.
- Matching the branches pulse for pulse removes the leading source of a
  branch-dependent offset, but it does not stop leakage from eating the contrast
  as `n_rotations` grows. That is what the interleaved beta calibration is for;
  keep `n_rotations` moderate. What remains uncancelled is any asymmetry between
  the `+X` and the `-X` leading pulse, which biases the crossing by
  `mu / (2 pi n)`.
- Coordinate descent ends each pass with beta, so the reported amplitude was
  measured at the *previous* beta. Use enough iterations for both to settle.
- Results are written to `exp.calib_note` under the transition label.
  `calib_note` does not autosave; call `Experiment.save_calib_note`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import Drag, RampType, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_calibrate_drag_hpi_pulse_based_on_ground",
    "calibrate_drag_hpi_pulse_based_on_ground",
    "display_calibrate_drag_hpi_pulse_based_on_ground",
    "plot_calibrate_drag_hpi_pulse_based_on_ground",
    "run_calibrate_drag_hpi_pulse_based_on_ground",
]

THETA_TARGET = np.pi / 2


def _resolve_initial_pulse(
    exp: Experiment,
    *,
    ladder: _core.TransitionLadder,
    duration: float,
    drag_coeff: float,
    beta: float | None,
    type: RampType,
) -> tuple[float, float, str, str]:
    """
    Return the starting amplitude, the beta to use, and where each came from.

    The two are resolved independently. A given `beta` replaces only the beta:
    the rotation angle is set by the real part of the envelope and `beta` only
    enters the imaginary part, so an amplitude measured at some other beta is
    still the best seed available.

    The amplitude, on the other hand, is reused only when the stored
    calibration was measured with the same envelope, because the rotation angle
    follows the area under it: the same peak amplitude spans a range of about
    -11 % to +27 % of a rotation across the envelopes `Drag` can build, far
    outside the amplitude sweep this calibration then runs. A stored beta
    survives a change of envelope; a stored amplitude does not.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibration note.
    ladder : _core.TransitionLadder
        Ladder resolved by `_core.resolve_ladder`.
    duration : float
        Pulse duration in ns.
    drag_coeff : float
        DRAG coefficient used to seed beta from the level ladder.
    beta : float | None
        Beta to use, or `None` to take the stored or the derived one.
    type : RampType
        Envelope of the pulse under test, already resolved and validated.

    Returns
    -------
    tuple[float, float, str, str]
        Starting amplitude, beta, amplitude source and beta source.
    """
    stored = exp.calib_note.get_drag_hpi_param(ladder.target)
    has_stored = (
        stored is not None
        and stored.get("amplitude") is not None
        and stored.get("beta") is not None
        and np.isclose(stored["duration"], duration)
    )
    # A stored calibration without a `"type"` was measured with a Gaussian.
    has_stored_type = (
        has_stored
        and stored is not None
        and (stored.get("type") or _core.DEFAULT_DRAG_TYPE) == type
    )

    if beta is not None:
        beta_source = "argument"
    elif has_stored:
        beta, beta_source = float(stored["beta"]), "calib_note"  # type: ignore[index]
    else:
        beta = _core.estimate_drag_beta(
            exp,
            qubit=ladder.qubit,
            transition=ladder.transition,
            drag_coeff=drag_coeff,
        )
        beta_source = "ladder_estimate"

    if has_stored_type:
        return float(stored["amplitude"]), beta, "calib_note", beta_source  # type: ignore[index]

    unit_pulse = Drag(duration=duration, amplitude=1, beta=beta, type=type)
    amplitude = _core.estimate_amplitude(
        exp,
        target=ladder.target,
        unit_pulse=unit_pulse,
        pulse_type="hpi",
    )
    return amplitude, beta, "rabi_estimate", beta_source


def _resolve_analyzer(exp: Experiment, *, target: str) -> Waveform:
    """
    Return the fixed `Y90-` analyzer of the beta pass.

    The corrected half-pi is preferred where one is stored. The analyzer's own
    axis error offsets the signal, and the fit takes the *root* of that signal,
    so the offset lands in the calibrated beta rather than washing out: to first
    order the half-pi sequence is biased by `(phi - a) / ((2n + 1) k)` for an
    in-plane phase error `phi` and an out-of-plane tilt `a`. An uncorrected
    half-pi carries `beta = 0`, which is the largest beta error available.

    It is resolved once per pass, before the sweep, so it stays fixed while beta
    is swept - which is what the amplification sequence needs of it.

    The closing `raise` is defensive: the amplitude pass writes a DRAG entry
    before this runs, so this flow never reaches it.
    """
    if exp.calib_note.get_drag_hpi_param(target) is not None:
        return exp.pulse.get_drag_hpi_pulse(target).shifted(-np.pi / 2)
    if exp.calib_note.get_hpi_param(target) is not None:
        return exp.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)
    raise ValueError(
        f"No half-pi pulse is calibrated for {target!r}, and the DRAG beta "
        "calibration needs one as the closing Y90- analyzer. Run "
        "`calibrate_hpi_pulse_based_on_ground` first."
    )


def _make_pulse(
    *,
    duration: float,
    beta: float,
    type: RampType,
) -> Callable[[float], Waveform]:
    """Return a builder of the pulse under test at a given amplitude."""

    def make_pulse(amplitude: float) -> Waveform:
        return Drag(duration=duration, amplitude=amplitude, beta=beta, type=type)

    return make_pulse


def run_calibrate_drag_hpi_pulse_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    duration: float | None = None,
    drag_coeff: float | None = None,
    beta: float | None = None,
    type: RampType | None = None,
    n_rotations: int | None = None,
    amplitude_diff_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Measure both rotation branches over the amplitude offsets, at fixed beta.

    This is one amplitude pass; `calibrate_drag_hpi_pulse_based_on_ground`
    alternates it with the beta calibration.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to calibrate, `"ge"` through `"mn"`.
    duration : float | None, optional
        Pulse duration in ns. Defaults to `_core.DEFAULT_DRAG_HPI_DURATION`.
    drag_coeff : float | None, optional
        DRAG coefficient used to seed beta when no matching DRAG calibration is
        stored. Defaults to `_core.DEFAULT_DRAG_COEFF`.
    beta : float | None, optional
        DRAG correction coefficient in ns. Defaults to the stored calibration
        when one matches, and otherwise to the estimate derived from the level
        ladder. Pass `0` together with `calibrate_beta=False` to run the
        calibration with the correction turned off.
    type : RampType | None, optional
        Envelope of the pulse under test. Defaults to
        `_core.DEFAULT_DRAG_TYPE`. `"Squad"` is part of `RampType` but `Drag`
        cannot build it, so it is rejected.
    n_rotations : int | None, optional
        Number of full rotations `n`; both branches use a signed leading pulse
        followed by `4n` more, realizing `4n - 1` and `4n + 1` rotations.
        Defaults to `_core.DEFAULT_DRAG_N_ROTATIONS`.
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
        Raw payload consumed by
        `analyze_calibrate_drag_hpi_pulse_based_on_ground`.
    """
    if duration is None:
        duration = _core.DEFAULT_DRAG_HPI_DURATION
    if drag_coeff is None:
        drag_coeff = _core.DEFAULT_DRAG_COEFF
    if type is None:
        type = _core.DEFAULT_DRAG_TYPE
    if n_rotations is None:
        n_rotations = _core.DEFAULT_DRAG_N_ROTATIONS
    if n_shots is None:
        n_shots = _core.DEFAULT_N_SHOTS
    if shot_interval is None:
        shot_interval = _core.DEFAULT_SHOT_INTERVAL
    if plot_pulse_schedule is None:
        plot_pulse_schedule = True

    _core.validate_drag_type(type)
    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    _core.validate_duration(duration)
    if beta is not None:
        _core.validate_beta(beta)
    _core.validate_n_rotations(n_rotations)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)

    rabi_param = _core.get_transition_rabi_param(exp, ladder)
    initial_amplitude, beta, source, beta_source = _resolve_initial_pulse(
        exp,
        ladder=ladder,
        duration=duration,
        drag_coeff=drag_coeff,
        beta=beta,
        type=type,
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
    repetitions = 4 * n_rotations
    make_pulse = _make_pulse(duration=duration, beta=beta, type=type)
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
            title=f"Ground-referenced DRAG hpi calibration : {ladder.target}",
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
        "beta_source": beta_source,
        "type": type,
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


def analyze_calibrate_drag_hpi_pulse_based_on_ground(
    *,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Locate the amplitude where the two branches agree.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_calibrate_drag_hpi_pulse_based_on_ground`.

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


def plot_calibrate_drag_hpi_pulse_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_calibrate_drag_hpi_pulse_based_on_ground`,
        or the assembled payload of `calibrate_drag_hpi_pulse_based_on_ground`
        (which additionally carries the beta pass and the iteration history).

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    target = data["target"]
    title = f"Ground-referenced DRAG hpi calibration : {target}"
    figures = {
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


def display_calibrate_drag_hpi_pulse_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the calibration summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `calibrate_drag_hpi_pulse_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"calibrate_drag_hpi_pulse_based_on_ground_{data['target']}",
        )
    _core.print_summary(data)
    print(f"  type                : {data['type']}")
    beta_source = _core.describe_beta_source(data["beta_source"])
    print(f"  beta                : {data['beta']:.9g}  [{beta_source}]")
    iterations = data.get("iterations")
    if iterations:
        print("  iterations:")
        for i, iteration in enumerate(iterations, start=1):
            print(
                f"    {i}: amplitude {iteration['amplitude']:.9g}  "
                f"beta {iteration['beta']:.9g}  "
                f"R2 {iteration['fit']['r2']:.4f}"
            )


def _calibrate_one_drag_hpi(
    exp: Experiment,
    *,
    ladder: _core.TransitionLadder,
    duration: float | None,
    drag_coeff: float | None,
    beta: float | None,
    type: RampType | None,
    n_rotations: int | None,
    amplitude_diff_range: ArrayLike | None,
    n_iterations: int,
    calibrate_beta: bool,
    betas: NDArray[np.float64],
    n_turns: int,
    degree: int,
    n_shots: int | None,
    shot_interval: float | None,
    plot: bool,
    plot_pulse_schedule: bool | None,
) -> tuple[dict[str, Any], dict[str, go.Figure]]:
    """
    Run the amplitude/beta coordinate descent of one transition.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    ladder : _core.TransitionLadder
        Ladder returned by `_core.preflight_ladders`.
    duration, drag_coeff, beta, type, n_rotations, amplitude_diff_range : optional
        Forwarded to `run_calibrate_drag_hpi_pulse_based_on_ground`, which
        resolves any of them that is `None`.
    n_iterations : int
        Number of amplitude/beta coordinate-descent passes.
    calibrate_beta : bool
        Whether to refine beta between amplitude passes.
    betas : NDArray[np.float64]
        Beta offsets from the current beta, already resolved and validated.
    n_turns : int
        Number of `(X90-, X90+)` repetitions in the beta sequence.
    degree : int
        Degree of the polynomial fitted by the beta calibration.
    n_shots, shot_interval : optional
        Forwarded to the run stage.
    plot : bool
        Whether to show the figures of this target.
    plot_pulse_schedule : bool | None
        Whether to draw the two branch schedules of the first pass.

    Returns
    -------
    tuple[dict[str, Any], dict[str, go.Figure]]
        The payload of this target and its figures.
    """
    iterations: list[dict[str, Any]] = []
    analyzed: dict[str, Any] | None = None
    beta_result: dict[str, Any] | None = None
    failed: dict[str, Any] | None = None
    # The loop rebinds `beta` to the value of the pass under way, so hold on to
    # the argument. It seeds the first pass only: after that the descent has to
    # continue from the beta it refined, which it reads back from the note.
    seed_beta = beta
    seed_beta_source: str | None = None
    for i in range(n_iterations):
        run_result = run_calibrate_drag_hpi_pulse_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=ladder.transition,
            duration=duration,
            drag_coeff=drag_coeff,
            beta=seed_beta if i == 0 else None,
            type=type,
            n_rotations=n_rotations,
            amplitude_diff_range=amplitude_diff_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot_pulse_schedule=plot_pulse_schedule and i == 0,
        )
        if seed_beta_source is None:
            seed_beta_source = str(run_result["beta_source"])
        try:
            data = analyze_calibrate_drag_hpi_pulse_based_on_ground(
                run_result=run_result
            )

            # The run stage owns the defaults, so read what it actually played.
            resolved_duration = float(data["duration"])
            resolved_type: RampType = data["type"]
            resolved_n_shots = int(data["n_shots"])
            resolved_shot_interval = float(data["shot_interval"])
            beta = float(data["beta"])
            amplitude = float(data["amplitude"])
            exp.calib_note.update_drag_hpi_param(
                data["target"],
                {
                    "target": data["target"],
                    "duration": resolved_duration,
                    "amplitude": amplitude,
                    "beta": beta,
                    "type": resolved_type,
                },
            )

            if calibrate_beta:
                analyzer = _resolve_analyzer(exp, target=ladder.target)
                beta_result = _core.analyze_beta_pass(
                    run_result=_core.run_beta_pass(
                        exp,
                        ladder=ladder,
                        prep_pulses=data["prep_pulses"],
                        rabi_param=data["rabi_param"],
                        pulse_type="hpi",
                        make_drive=lambda value, amplitude=amplitude, type=resolved_type: (
                            Drag(
                                duration=resolved_duration,
                                amplitude=amplitude,
                                beta=float(value),
                                type=type,
                            )
                        ),
                        analyzer=analyzer,
                        beta_range=beta + betas,
                        n_turns=n_turns,
                        n_shots=resolved_n_shots,
                        shot_interval=resolved_shot_interval,
                    ),
                    degree=degree,
                )
                beta = float(beta_result["beta"])
                exp.calib_note.update_drag_hpi_param(
                    data["target"],
                    {
                        "target": data["target"],
                        "duration": resolved_duration,
                        "amplitude": amplitude,
                        "beta": beta,
                        "type": resolved_type,
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
            analyzed = data
        except Exception as error:  # noqa: BLE001 - the raw data must survive
            failed = _core.record_stage_failure(
                run_result,
                target=ladder.target,
                stage=f"analysis (iteration {i + 1}/{n_iterations})",
                error=error,
            )
            break

    if analyzed is None:
        # Not one pass got far enough to fit, so the raw sweep is all there is.
        return failed if failed is not None else {}, {}

    data = {
        **analyzed,
        "beta": iterations[-1]["beta"],
        # Report where the descent started, not what the last pass read back.
        "beta_source": seed_beta_source,
        "beta_result": beta_result,
        "iterations": iterations,
    }
    if failed is not None:
        # An earlier pass succeeded; keep it and say why the descent stopped.
        data["error"] = failed["error"]

    def _display(result: Result) -> None:
        display_calibrate_drag_hpi_pulse_based_on_ground(result, plot)

    return _core.run_analysis_stages(
        run_result=data,
        # The passes above already analyzed and stored; only the figures and
        # the summary are left to guard.
        analyze=lambda payload: payload,
        store=None,
        plot=plot_calibrate_drag_hpi_pulse_based_on_ground,
        display=_display,
    )


def calibrate_drag_hpi_pulse_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    duration: float | None = None,
    drag_coeff: float | None = None,
    beta: float | None = None,
    type: RampType | None = None,
    n_rotations: int | None = None,
    amplitude_diff_range: ArrayLike | None = None,
    n_iterations: int | None = None,
    calibrate_beta: bool | None = None,
    beta_range: ArrayLike | None = None,
    n_turns: int | None = None,
    degree: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Calibrate a transition's DRAG half-pi amplitude and beta, and store both.

    Each iteration refines the amplitude against the zero crossing at the current
    beta, writes it to the calibration note, and then — when `calibrate_beta` is
    set — refines beta at that amplitude and writes it back.

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
        Pulse duration in ns. Defaults to `_core.DEFAULT_DRAG_HPI_DURATION`.
    drag_coeff : float | None, optional
        DRAG coefficient used to seed beta. Defaults to
        `_core.DEFAULT_DRAG_COEFF`.
    beta : float | None, optional
        DRAG correction coefficient in ns, shared by every target. Defaults to
        each target's stored calibration when one matches, and otherwise to the
        estimate derived from the level ladder. Pass `0` together with
        `calibrate_beta=False` to calibrate the amplitude with the correction
        turned off.
    type : RampType | None, optional
        Envelope of the pulse under test, shared by every target, and stored
        alongside the amplitude and beta so the calibrated pulse is rebuilt with
        it. Defaults to `_core.DEFAULT_DRAG_TYPE`. A stored amplitude is reused
        as the starting point only when it was measured with this same
        envelope. `"Squad"` is part of `RampType` but `Drag` cannot build it, so
        it is rejected.
    n_rotations : int | None, optional
        Number of full rotations `n`. Defaults to
        `_core.DEFAULT_DRAG_N_ROTATIONS`, which assumes the amplitude is already
        close; run `calibrate_hpi_pulse_based_on_ground` first, or lower it, when
        starting from a coarse estimate.
    amplitude_diff_range : ArrayLike | None, optional
        Amplitude offsets from the starting amplitude. Scalar or sequence.
        Defaults to a symmetric sweep around each target's starting amplitude.
    n_iterations : int | None, optional
        Number of amplitude/beta coordinate-descent passes. Defaults to
        `_core.DEFAULT_N_ITERATIONS`.
    calibrate_beta : bool | None, optional
        Whether to refine beta between amplitude passes. Defaults to
        `_core.DEFAULT_CALIBRATE_BETA`.
    beta_range : ArrayLike | None, optional
        Beta offsets from the current beta. Scalar or sequence. Defaults to
        `_core.DEFAULT_BETA_RANGE`.
    n_turns : int | None, optional
        Number of `(X90-, X90+)` repetitions in the beta sequence. Defaults to `_core.DEFAULT_N_TURNS`.
    degree : int | None, optional
        Degree of the polynomial fitted by the beta calibration. Defaults to
        `_core.DEFAULT_DEGREE`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the two branch schedules of the first pass. Defaults to
        True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so
        the calibrated amplitude and beta of one target are at
        `result.data[target]["amplitude"]` and `result.data[target]["beta"]`.
        Both are also written to `exp.calib_note` under the transition label.
        A target whose analysis, storage, plotting or display failed still
        appears, carrying whatever the pipeline reached and the exception
        under `"error"`; a target that came through carries no `"error"`
        key at all. Measurement failures are not caught: they are raised.
    """
    if plot is None:
        plot = True
    if n_iterations is None:
        n_iterations = _core.DEFAULT_N_ITERATIONS
    if calibrate_beta is None:
        calibrate_beta = _core.DEFAULT_CALIBRATE_BETA
    if beta_range is None:
        beta_range = _core.DEFAULT_BETA_RANGE
    if n_turns is None:
        n_turns = _core.DEFAULT_N_TURNS
    if degree is None:
        degree = _core.DEFAULT_DEGREE

    _core.validate_positive_int(n_iterations, "n_iterations")
    _core.validate_positive_int(n_turns, "n_turns")
    _core.validate_positive_int(degree, "degree")
    betas = _core.resolve_sweep_range(beta_range)
    if calibrate_beta:
        _core.validate_beta_range(betas, degree=degree)

    ladders = _core.preflight_ladders(
        exp,
        targets=targets,
        transition=transition,
        require_rabi_param=True,
    )

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        payload, target_figures = _calibrate_one_drag_hpi(
            exp,
            ladder=ladder,
            duration=duration,
            drag_coeff=drag_coeff,
            beta=beta,
            type=type,
            n_rotations=n_rotations,
            amplitude_diff_range=amplitude_diff_range,
            n_iterations=n_iterations,
            calibrate_beta=calibrate_beta,
            betas=betas,
            n_turns=n_turns,
            degree=degree,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
