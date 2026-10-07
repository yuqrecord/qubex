"""
Ground-referenced Ramsey experiment for any transition of the level ladder.

`Experiment.ramsey_experiment` is ge only: it reads the drive frequency out of
the qubit-keyed registry, groups targets with `create_qubit_subgroups`, and
drives the target straight from |g>. The contrib `ef_ramsey_experiment` and
`fh_ramsey_experiment` hard-code their single rung and project the raw IQ trace
themselves instead of normalizing against the transition's Rabi arc.

Method
------
The qubit is climbed from |g> to the lower level of the transition with a pi
pulse on every rung below it, a Ramsey sequence is played on the transition, the
same pi pulses are replayed in reverse order, and only then is the qubit read
out::

    X_ge -> X_ef -> ... -> [X90 -> wait(T) -> -X90] -> ... -> X_ef -> X_ge

Both half-pi pulses are the transition's calibrated *DRAG* half-pi when one is
stored. Driving `k <-> k+1` leaks into `k-1` and `k+2`, and the two pulses sit
on either side of a wait that can run into microseconds, so the leakage they
seed decays along with the fringe. When no DRAG half-pi is stored, the
uncorrected half-pi is played instead. It seeds more of that leakage and can
shift the fitted T2*, so the payload records which of the two was played under
`"hpi_source"` and the summary prints it.

Ideal un-preparation maps `|k> -> |g>` and leaves `|k+1>` alone, so the readout
separates |g> from the transition's *upper* level and the normalized signal is
the lower-level population of the transition's Bloch sphere, exactly as the rest
of this package measures it.

The fringe comes from a deliberate carrier detuning, applied through
`frequencies={<transition label>: frequency + detuning}` rather than on the
waveform. `Waveform.detuned` would rotate each element against its *own* time
base, so the second half-pi would restart its phase ramp at zero and no fringe
would appear at all; retuning the channel makes the modulator run on
schedule-absolute time, and the second half-pi picks up `2 pi * detuning * T`.
Only the transition target is retuned, so the pi pulses of the lower rungs keep
their own carriers.

Assumptions and caveats
-----------------------
- Every rung below the transition must have a calibrated pi pulse, the
  transition must have a calibrated half-pi pulse, DRAG or uncorrected, and it
  must have stored Rabi parameters to normalize against. Run
  `calibrate_drag_hpi_pulse_based_on_ground` before this experiment, or at
  least `calibrate_hpi_pulse_based_on_ground`.
- The sequence closes with `-X90` rather than `Y90`, so the fringe is a pure
  cosine and its phase cannot say which side of the drive the resonance sits on.
  The resonance is reported as `drive - sign(detuning) * f_ramsey`, which is
  only right while `|f_resonance - frequency| < |detuning|`. The mirror solution
  is reported alongside it, and a fringe that reaches `2 |detuning|` says the
  assumption has broken down.
- The decay is fitted from a starting guess, `t2_estimation`, which defaults
  to the 10 us that `fitting.fit_ramsey` itself assumes. A sweep that runs
  well past that can settle on a shallow local minimum with a visibly low
  R2 and a biased fringe frequency, so pass the coherence time you expect
  when it is far from the default.
- Nothing is persisted. The transition frequency belongs to the registered
  `Target`, not to `calib_note`, so applying the result stays an explicit act of
  the caller (`Experiment.register_transition_target(frequency=...)`), and T2*
  has no home in the note at all.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import Blank, PulseArray, PulseSchedule, Waveform
from qubex.system.transition import TransitionName

from . import _core

__all__ = [
    "analyze_ramsey_experiment_based_on_ground",
    "display_ramsey_experiment_based_on_ground",
    "plot_ramsey_experiment_based_on_ground",
    "ramsey_experiment_based_on_ground",
    "run_ramsey_experiment_based_on_ground",
]

#: Wait times swept between the two half-pi pulses, in ns. Same as the core
#: `Experiment.ramsey_experiment`.
DEFAULT_TIME_RANGE = np.arange(0, 10001, 100)

#: Carrier detuning that produces the fringe, in GHz. Same as the core
#: `Experiment.ramsey_experiment`.
DEFAULT_DETUNING = 0.001

#: Starting guess for the decay of the fringe, in ns. Same as the one
#: `fitting.fit_ramsey` falls back to.
DEFAULT_T2_ESTIMATION = 10_000.0

#: Ramsey fit quality below which the result is called out.
RELIABLE_RAMSEY_R2 = 0.5

#: Fringe frequency, in units of `|detuning|`, at which the sign rule that picks
#: the resonance out of the two mirrored solutions stops being justified.
SIGN_RULE_FRINGE_LIMIT = 2.0


def _resolve_hpi_pulse(exp: Experiment, *, target: str) -> tuple[Waveform, str]:
    """
    Return the half-pi pulse of the transition and which entry it came from.

    The stored DRAG half-pi is preferred, and the uncorrected one is used only
    when no DRAG half-pi is stored. The note is checked explicitly before
    `get_hpi_pulse` is called, because that getter silently hands back a nominal
    pulse when nothing is stored, which must never stand in for a calibration.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibration note.
    target : str
        Transition label.

    Returns
    -------
    tuple[Waveform, str]
        The half-pi pulse, and `"drag_hpi"` or `"hpi"` for the entry it was
        rebuilt from.
    """
    if exp.calib_note.get_drag_hpi_param(target) is not None:
        return exp.pulse.get_drag_hpi_pulse(target), "drag_hpi"
    if exp.calib_note.get_hpi_param(target) is not None:
        return exp.pulse.get_hpi_pulse(target), "hpi"
    raise ValueError(
        f"No half-pi pulse is calibrated for {target!r}, and a Ramsey sequence "
        "is built from two of them. Run "
        "`calibrate_drag_hpi_pulse_based_on_ground` (preferred) or "
        "`calibrate_hpi_pulse_based_on_ground` first."
    )


def _validate_wait_times(time_range: NDArray[np.float64]) -> None:
    """Raise when a swept wait time misses the sampling grid."""
    period = Waveform.SAMPLING_PERIOD
    remainders = time_range % period
    off_grid = time_range[~(np.isclose(remainders, 0) | np.isclose(remainders, period))]
    if off_grid.size > 0:
        raise ValueError(
            "`time_range` must contain only multiples of the sampling period "
            f"({period} ns); {off_grid.tolist()} are not."
        )


def _validate_detuning(detuning: float) -> None:
    """Raise when the carrier detuning cannot produce a signed fringe."""
    if not np.isfinite(detuning) or detuning == 0:
        raise ValueError(
            f"`detuning` must be a non-zero finite number, got {detuning!r}. "
            "The fringe is the beat against the carrier offset, and its sign is "
            "what picks the resonance out of the two mirrored solutions."
        )


def _validate_t2_estimation(t2_estimation: float) -> None:
    """Raise when the decay guess cannot seed the fit."""
    if not np.isfinite(t2_estimation) or t2_estimation <= 0:
        raise ValueError(
            "`t2_estimation` must be a positive, finite number of ns, got "
            f"{t2_estimation!r}."
        )


def run_ramsey_experiment_based_on_ground(
    exp: Experiment,
    *,
    target: str,
    transition: TransitionName,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    frequency: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplification: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Measure the ground-referenced Ramsey fringe of one transition.

    The half-pi pulses are the transition's stored DRAG half-pi, or its
    uncorrected half-pi when no DRAG one is stored.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    transition : TransitionName
        Transition to drive, `"ge"` through `"mn"`.
    time_range : ArrayLike | None, optional
        Wait times between the two half-pi pulses in ns. Scalar or sequence.
        Defaults to `DEFAULT_TIME_RANGE`.
    detuning : float | None, optional
        Carrier detuning in GHz, applied to the transition target only.
        Defaults to `DEFAULT_DETUNING`.
    frequency : float | None, optional
        Undetuned carrier frequency of the transition in GHz. Defaults to the
        frequency the transition target is registered with.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    readout_amplification : bool | None, optional
        Whether to insert readout amplification pulses into the sweep. The
        signal is normalized with the stored Rabi parameters, so they should
        have been measured with the same setting. `None` is passed on as is,
        so `MeasurementService.measure` resolves it to False.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the longest wait. Defaults to True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_ramsey_experiment_based_on_ground`.
        `"hpi_source"` is `"drag_hpi"` or `"hpi"`, naming the half-pi entry
        that was played.
    """
    if time_range is None:
        time_range = DEFAULT_TIME_RANGE
    if detuning is None:
        detuning = DEFAULT_DETUNING
    if n_shots is None:
        n_shots = _core.DEFAULT_N_SHOTS
    if shot_interval is None:
        shot_interval = _core.DEFAULT_SHOT_INTERVAL
    if plot_pulse_schedule is None:
        plot_pulse_schedule = True

    ladder = _core.resolve_ladder(exp, target=target, transition=transition)
    _core.validate_ladder_calibrations(exp, ladder)
    rabi_param = _core.get_transition_rabi_param(exp, ladder)
    hpi_pulse, hpi_source = _resolve_hpi_pulse(exp, target=ladder.target)
    if frequency is None:
        frequency = _core.resolve_frequencies(
            exp,
            ladders=[ladder],
            frequencies=None,
        )[ladder.target]
    _validate_detuning(detuning)
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"`frequency` must be a positive number, got {frequency!r}.")

    times = _core.resolve_sweep_range(time_range)
    _core.validate_time_range(times)
    _validate_wait_times(times)

    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    # The second half-pi is -X90: a pi phase shift inverts the envelope, which
    # brings the Bloch vector back to |g> at T = 0 and makes the fringe a cosine.
    second_hpi_pulse = hpi_pulse.shifted(np.pi)

    def make_body(wait: float) -> Waveform:
        return PulseArray([hpi_pulse, Blank(duration=wait), second_hpi_pulse])

    def sequence(wait: float) -> PulseSchedule:
        return _core.build_ground_referenced_schedule(
            ladder=ladder,
            prep_pulses=prep_pulses,
            body=make_body(float(wait)),
        )

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {"longest wait": sequence(float(np.max(times)))},
            title=f"Ground-referenced Ramsey : {ladder.target}",
        )

    drive_frequency = frequency + detuning
    raw, normalized = _core.sweep_ground_referenced(
        exp,
        ladder=ladder,
        rabi_param=rabi_param,
        sequence=sequence,
        sweep_range=times,
        frequencies={ladder.target: drive_frequency},
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
        "detuning": detuning,
        "frequency": frequency,
        "drive_frequency": drive_frequency,
        "hpi_duration": hpi_pulse.duration,
        "hpi_source": hpi_source,
        "raw": raw,
        "normalized": normalized,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
        "readout_amplification": readout_amplification,
    }


def analyze_ramsey_experiment_based_on_ground(
    *,
    run_result: dict[str, Any],
    t2_estimation: float | None = None,
) -> dict[str, Any]:
    """
    Fit the Ramsey fringe and turn it into a resonance estimate.

    The sequence closes with `-X90`, so the fringe is a cosine and the fitted
    phase carries no sign information. The resonance is therefore taken to lie
    on the `frequency` side of the drive, which is the branch
    `drive - sign(detuning) * f_ramsey`, and the mirrored branch is reported
    next to it.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_ramsey_experiment_based_on_ground`.
    t2_estimation : float | None, optional
        Starting guess for the decay in ns. It only seeds the fit, but a
        sweep that runs well past the default can otherwise settle on a
        shallow local minimum. Defaults to `DEFAULT_T2_ESTIMATION`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the fit, T2* and the two mirrored resonances.
    """
    if t2_estimation is None:
        t2_estimation = DEFAULT_T2_ESTIMATION
    _validate_t2_estimation(t2_estimation)

    target = run_result["target"]
    fit = fitting.fit_ramsey(
        target=target,
        times=run_result["time_range"],
        data=run_result["normalized"],
        amplitude_est=1.0,
        offset_est=0.0,
        tau_est=t2_estimation,
        plot=False,
        title="Ground-referenced Ramsey",
    )
    _core.require_fit_success(fit, target=target, what="Ramsey fit")

    detuning = float(run_result["detuning"])
    drive_frequency = float(run_result["drive_frequency"])
    ramsey_frequency = float(fit["f"])
    branch = float(np.sign(detuning))
    resonant_frequency = drive_frequency - branch * ramsey_frequency
    mirror_frequency = drive_frequency + branch * ramsey_frequency

    return {
        **run_result,
        "t2_estimation": t2_estimation,
        "fit": fit,
        "t2": float(fit["tau"]),
        "t2_err": float(fit["tau_err"]),
        "ramsey_frequency": ramsey_frequency,
        "ramsey_frequency_err": float(fit["f_err"]),
        "phi": float(fit["phi"]),
        "r2": float(fit["r2"]),
        "resonant_frequency": resonant_frequency,
        "mirror_frequency": mirror_frequency,
        "detuning_from_carrier": resonant_frequency - float(run_result["frequency"]),
    }


def plot_ramsey_experiment_based_on_ground(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the Ramsey experiment.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_ramsey_experiment_based_on_ground`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    figures: dict[str, go.Figure] = {}
    fringe_figure = data["fit"].figure
    if fringe_figure is not None:
        figures["ramsey"] = fringe_figure
    figures["iq"] = _core.iq_figure(
        title=f"Ground-referenced Ramsey : {data['target']}",
        raw=data["raw"],
    )
    return figures


def display_ramsey_experiment_based_on_ground(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the Ramsey summary and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `ramsey_experiment_based_on_ground`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"ramsey_experiment_based_on_ground_{data['target']}",
        )
    hpi_label = {
        "drag_hpi": "DRAG",
        "hpi": "uncorrected (no DRAG half-pi stored)",
    }[data["hpi_source"]]
    _core.print_ladder(data)
    print(f"  half-pi pulse       : {hpi_label}, {data['hpi_duration']} ns")
    print(f"  carrier frequency   : {data['frequency']:.6f} GHz")
    print(f"  applied detuning    : {data['detuning'] * 1e3:+.3f} MHz")
    print(f"  drive frequency     : {data['drive_frequency']:.6f} GHz")
    print(
        f"  T2*                 : {data['t2'] * 1e-3:.3f} "
        f"+- {data['t2_err'] * 1e-3:.1g} us"
    )
    print(
        f"  fringe frequency    : {data['ramsey_frequency'] * 1e3:.4f} "
        f"+- {data['ramsey_frequency_err'] * 1e3:.1g} MHz"
    )
    print(f"  resonant frequency  : {data['resonant_frequency']:.6f} GHz")
    print(f"  mirror solution     : {data['mirror_frequency']:.6f} GHz")
    print(f"  detuning from carrier: {data['detuning_from_carrier'] * 1e3:+.3f} MHz")
    print(f"  Ramsey fit R2       : {data['r2']:.4f}")
    if data["r2"] < RELIABLE_RAMSEY_R2:
        print(
            "  The Ramsey fit is unreliable; seed it with `t2_estimation` (it "
            f"started from {data['t2_estimation']:.0f} ns), shorten "
            "`time_range` towards T2*, or improve the ladder preparation."
        )
    if data["ramsey_frequency"] >= SIGN_RULE_FRINGE_LIMIT * abs(data["detuning"]):
        print(
            "  The fringe is at least twice the applied detuning, so the "
            "resonance is not necessarily the branch printed above. Re-run with "
            "a larger `detuning` to tell the two solutions apart."
        )
    print(
        "  Nothing was stored. Apply the resonance explicitly, for example with "
        "`Experiment.register_transition_target(..., frequency=...)`."
    )


def ramsey_experiment_based_on_ground(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    transition: TransitionName,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    frequencies: Mapping[str, float] | float | None = None,
    t2_estimation: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplification: bool | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Measure the ground-referenced Ramsey fringe of one transition.

    Every requested target is checked before the first acquisition, so an
    unregistered target, an uncalibrated lower rung, a missing half-pi pulse
    or missing Rabi parameters stop the whole batch. A target without a DRAG
    half-pi falls back to its uncorrected one. The targets are then measured
    one after another.

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
        Wait times between the two half-pi pulses in ns. Scalar or sequence.
        Defaults to `DEFAULT_TIME_RANGE`.
    detuning : float | None, optional
        Carrier detuning in GHz, applied to the transition target only.
        Defaults to `DEFAULT_DETUNING`.
    frequencies : Mapping[str, float] | float | None, optional
        Undetuned carrier frequency in GHz per target, one frequency shared by
        every target, or `None` to use the frequency each transition target is
        registered with.
    t2_estimation : float | None, optional
        Starting guess for the decay in ns, shared by every target. Defaults
        to `DEFAULT_T2_ESTIMATION`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to
        `_core.DEFAULT_SHOT_INTERVAL`.
    readout_amplification : bool | None, optional
        Whether to insert readout amplification pulses into the sweep. The
        stored Rabi parameters should have been measured with the same
        setting. Defaults to False.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the schedule of the longest wait. Defaults to True.

    Returns
    -------
    Result
        Payload keyed by transition label, and figures keyed by
        `"{target}:{name}"`. Each payload is the single-target payload, so T2*
        of one target is at `result.data[target]["t2"]`, and the half-pi it
        was measured with at `result.data[target]["hpi_source"]`.
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
        require_rabi_param=True,
    )
    for ladder in ladders:
        _resolve_hpi_pulse(exp, target=ladder.target)
    resolved_frequencies = _core.resolve_frequencies(
        exp,
        ladders=ladders,
        frequencies=frequencies,
    )

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_ramsey_experiment_based_on_ground(
            run_result=run_result,
            t2_estimation=t2_estimation,
        )

    def _display(result: Result) -> None:
        display_ramsey_experiment_based_on_ground(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for ladder in ladders:
        run_result = run_ramsey_experiment_based_on_ground(
            exp,
            target=ladder.qubit,
            transition=transition,
            time_range=time_range,
            detuning=detuning,
            frequency=resolved_frequencies[ladder.target],
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_amplification=readout_amplification,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            # The transition frequency belongs to the registered target and T2*
            # has no entry in the note, so this experiment persists nothing.
            store=None,
            plot=plot_ramsey_experiment_based_on_ground,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
