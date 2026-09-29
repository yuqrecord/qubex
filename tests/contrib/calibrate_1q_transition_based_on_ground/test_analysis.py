"""Tests for the shared machinery of the ground-referenced transition package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast, get_args

import numpy as np
import plotly.graph_objects as go
import pytest

from qubex.contrib.experiment.calibrate_1q_transition_based_on_ground import (
    _core,
    analyze_chevron_pattern_based_on_ground,
    analyze_measure_dispersive_shift_extended,
)
from qubex.experiment import experiment_constants as constants
from qubex.pulse import Drag, FlatTop, RampType
from qubex.system.transition import (
    LEVEL_NAMES,
    TRANSITION_NAMES,
    TransitionName,
)

QUBIT = "Q00"
ALPHA = -0.35
DURATION = 32.0
RAMPTIME = 12.0
TRANSITIONS = list(TRANSITION_NAMES)


def _build_experiment(
    factory: Callable[..., Any],
    transition: TransitionName,
) -> Any:
    """Return a stub configured for a FlatTop half-pi on one transition."""
    return factory(
        transition=transition,
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.4,
        pulse_type="hpi",
    )


def _pulse_bounds(values: np.ndarray) -> tuple[int, int]:
    """Return the first and last sample index where a channel is driven."""
    driven = np.flatnonzero(np.abs(values) > 0)
    return int(driven[0]), int(driven[-1])


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_ladder_ascends_then_descends(
    stub_experiment_factory: Callable[..., Any],
    transition: TransitionName,
) -> None:
    """Given a transition, when a schedule is built, then rungs rise then fall."""
    exp = _build_experiment(stub_experiment_factory, transition)
    ladder = _core.resolve_ladder(exp, target=QUBIT, transition=transition)
    schedule = _core.build_ground_referenced_schedule(
        ladder=ladder,
        prep_pulses=_core.resolve_prep_pulses(exp, ladder),
        body=exp.pulse.get_hpi_pulse(ladder.target),
    )

    sequences = schedule.get_sampled_sequences()
    starts = [_pulse_bounds(sequences[label])[0] for label in ladder.lower_labels]
    ends = [_pulse_bounds(sequences[label])[1] for label in ladder.lower_labels]
    body_start, body_end = _pulse_bounds(sequences[ladder.target])

    assert starts == sorted(starts)
    assert ends == sorted(ends, reverse=True)
    if ladder.lower_labels:
        assert starts[-1] < body_start
        assert body_end < ends[-1]


def test_ge_needs_no_ladder(stub_experiment_factory: Callable[..., Any]) -> None:
    """Given the ge transition, when resolved, then there is no preparation."""
    exp = _build_experiment(stub_experiment_factory, "ge")
    ladder = _core.resolve_ladder(exp, target=QUBIT, transition="ge")

    assert ladder.target == QUBIT
    assert ladder.lower_labels == ()

    schedule = _core.build_ground_referenced_schedule(
        ladder=ladder,
        prep_pulses=(),
        body=exp.pulse.get_hpi_pulse(ladder.target),
    )
    assert schedule.labels == [QUBIT]


def test_mn_uses_every_lower_rung(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the mn transition, when resolved, then all eight rungs are used."""
    exp = _build_experiment(stub_experiment_factory, "mn")
    ladder = _core.resolve_ladder(exp, target=QUBIT, transition="mn")

    assert ladder.target == f"{QUBIT}-mn"
    assert ladder.lower_labels == (
        QUBIT,
        f"{QUBIT}-ef",
        f"{QUBIT}-fh",
        f"{QUBIT}-hi",
        f"{QUBIT}-ij",
        f"{QUBIT}-jk",
        f"{QUBIT}-kl",
        f"{QUBIT}-lm",
    )


def test_missing_ladder_pi_pulse_raises(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an uncalibrated rung, when validated, then it raises."""
    exp = _build_experiment(stub_experiment_factory, "fh")
    ladder = _core.resolve_ladder(exp, target=QUBIT, transition="fh")
    exp.calib_note.pi.pop(f"{QUBIT}-ef")

    with pytest.raises(ValueError, match="Pi pulses are not calibrated"):
        _core.validate_ladder_calibrations(exp, ladder)


def test_unregistered_transition_target_raises(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an unregistered rung, when resolved, then it raises."""
    exp = _build_experiment(stub_experiment_factory, "fh")
    exp.ctx.targets.pop(f"{QUBIT}-fh")

    with pytest.raises(ValueError, match="is not registered"):
        _core.resolve_ladder(exp, target=QUBIT, transition="fh")


def test_unsupported_transition_raises(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an unknown transition name, when resolved, then it raises."""
    exp = _build_experiment(stub_experiment_factory, "ef")

    with pytest.raises(ValueError, match="not a supported transition"):
        _core.resolve_ladder(
            exp,
            target=QUBIT,
            transition=cast("TransitionName", "no"),
        )


def test_acquisition_plan_interleaves_in_abba_order() -> None:
    """Given amplitude offsets, when planned, then branches alternate in pairs."""
    diffs = np.array([-0.02, -0.01, 0.01, 0.02])
    acquisition_diffs, branches = _core.build_acquisition_plan(diffs)

    assert acquisition_diffs.tolist() == [
        -0.02,
        -0.02,
        -0.01,
        -0.01,
        0.01,
        0.01,
        0.02,
        0.02,
    ]
    assert branches == [
        _core.BRANCH_MINUS,
        _core.BRANCH_PLUS,
        _core.BRANCH_PLUS,
        _core.BRANCH_MINUS,
        _core.BRANCH_MINUS,
        _core.BRANCH_PLUS,
        _core.BRANCH_PLUS,
        _core.BRANCH_MINUS,
    ]


def test_split_branches_undoes_the_interleaving() -> None:
    """Given an acquisition-ordered signal, when split, then each branch is restored."""
    diffs = np.array([-0.02, -0.01, 0.01, 0.02])
    _, branches = _core.build_acquisition_plan(diffs)
    signal = np.arange(len(branches), dtype=np.float64)

    split = _core.split_branches(signal, branches, len(diffs))

    assert split[_core.BRANCH_MINUS].tolist() == [0.0, 3.0, 4.0, 7.0]
    assert split[_core.BRANCH_PLUS].tolist() == [1.0, 2.0, 5.0, 6.0]


def _ideal_branches(
    diffs: np.ndarray,
    *,
    initial_amplitude: float,
    n_rotations: int,
    root: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two ideal half-pi branch signals for a known crossing."""
    eps = (np.pi / 2) * (diffs - root) / initial_amplitude
    minus = np.sin((4 * n_rotations - 1) * eps)
    plus = -np.sin((4 * n_rotations + 1) * eps)
    return minus, plus


def _fit(minus: np.ndarray, plus: np.ndarray, diffs: np.ndarray) -> float:
    """Return the crossing located by the shared fit."""
    return _core.fit_crossing(
        target=f"{QUBIT}-fh",
        amplitude_diff_range=diffs,
        signal_minus=minus,
        signal_plus=plus,
        initial_amplitude=0.4,
        n_rotations=2,
        theta_target=np.pi / 2,
    )["amplitude_diff"]


def test_crossing_finds_the_true_amplitude() -> None:
    """Given ideal branches, when fitted, then the crossing is where they agree."""
    diffs = np.linspace(-0.01, 0.01, 21)
    minus, plus = _ideal_branches(diffs, initial_amplitude=0.4, n_rotations=2, root=0.0)

    assert _fit(minus, plus, diffs) == pytest.approx(0.0, abs=1e-9)


def test_an_off_centre_crossing_is_pulled_slightly_towards_the_centre() -> None:
    """Given a crossing away from the centre, when fitted, then the cubic term biases it."""
    diffs = np.linspace(-0.01, 0.01, 21)
    root = 0.002
    minus, plus = _ideal_branches(
        diffs, initial_amplitude=0.4, n_rotations=2, root=root
    )

    fitted = _fit(minus, plus, diffs)

    assert fitted == pytest.approx(root, rel=1e-2)
    assert abs(fitted) < abs(root)


@pytest.mark.parametrize(("gain", "offset"), [(0.3, 0.0), (1.0, 0.4), (-0.6, -0.2)])
def test_crossing_is_invariant_under_affine_normalization_errors(
    gain: float,
    offset: float,
) -> None:
    """Given a shared affine distortion, when fitted, then the crossing is unmoved."""
    diffs = np.linspace(-0.01, 0.01, 21)
    minus, plus = _ideal_branches(
        diffs, initial_amplitude=0.4, n_rotations=2, root=0.002
    )
    baseline = _fit(minus, plus, diffs)

    distorted = _fit(gain * minus + offset, gain * plus + offset, diffs)

    assert distorted == pytest.approx(baseline, rel=1e-9)


def test_branch_dependent_offset_biases_the_crossing() -> None:
    """Given a per-branch offset, when fitted, then the crossing does move."""
    diffs = np.linspace(-0.01, 0.01, 21)
    minus, plus = _ideal_branches(
        diffs, initial_amplitude=0.4, n_rotations=2, root=0.002
    )
    baseline = _fit(minus, plus, diffs)

    biased = _fit(minus, plus + 0.05, diffs)

    assert abs(biased - baseline) > 1e-4


def test_fit_crossing_rejects_a_flat_difference() -> None:
    """Given a difference with no slope, when fitted, then it raises."""
    diffs = np.linspace(-0.01, 0.01, 21)
    flat = np.zeros_like(diffs)

    with pytest.raises(ValueError, match="below 5%"):
        _fit(flat, flat + 1e-6, diffs)


def test_fit_crossing_rejects_an_extrapolated_root() -> None:
    """Given a crossing outside the sweep, when fitted, then it raises."""
    diffs = np.linspace(-0.01, 0.01, 21)
    minus, plus = _ideal_branches(
        diffs, initial_amplitude=0.4, n_rotations=2, root=0.0098
    )

    with pytest.raises(ValueError, match="outside the central"):
        _fit(minus, plus, diffs)


def test_amplitude_diff_range_must_stay_inside_the_monotonic_bound() -> None:
    """Given a sweep reaching A0 / (4 n), when validated, then it raises."""
    diffs = np.linspace(-0.06, 0.06, 21, dtype=np.float64)

    with pytest.raises(ValueError, match="monotonic bound"):
        _core.validate_amplitude_diff_range(
            diffs,
            initial_amplitude=0.4,
            n_rotations=2,
        )


def test_amplitude_diff_range_must_stay_inside_the_unit_interval() -> None:
    """Given a sweep leaving (0, 1], when validated, then it raises."""
    diffs = np.linspace(-0.02, 0.02, 21, dtype=np.float64)

    with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
        _core.validate_amplitude_diff_range(
            diffs,
            initial_amplitude=0.99,
            n_rotations=1,
        )


def test_amplitude_diff_range_needs_enough_points() -> None:
    """Given fewer than four offsets, when validated, then it raises."""
    diffs = np.array([-0.001, 0.001])

    with pytest.raises(ValueError, match="at least 4 distinct"):
        _core.validate_amplitude_diff_range(
            diffs,
            initial_amplitude=0.4,
            n_rotations=1,
        )


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_drag_beta_seed_follows_the_linear_ladder(
    stub_experiment_factory: Callable[..., Any],
    transition: TransitionName,
) -> None:
    """Given a linear ladder, when seeded, then beta is -lambda / ((k + 1) alpha)."""
    exp = _build_experiment(stub_experiment_factory, transition)
    index = TRANSITIONS.index(transition)

    beta = _core.estimate_drag_beta(
        exp,
        qubit=QUBIT,
        transition=transition,
        drag_coeff=0.5,
    )

    assert beta == pytest.approx(-0.5 / ((index + 1) * ALPHA))


def test_analyze_beta_pass_returns_the_root() -> None:
    """Given a beta sweep crossing zero, when analyzed, then the root is returned."""
    beta_range = np.linspace(-1.0, 1.0, 21)
    run_result = {
        "target": f"{QUBIT}-fh",
        "pulse_type": "hpi",
        "beta_range": beta_range,
        "n_turns": 1,
        "raw": np.zeros_like(beta_range, dtype=np.complex128),
        "normalized": 0.3 * (beta_range - 0.25),
    }

    analysis = _core.analyze_beta_pass(run_result=run_result, degree=1)

    assert analysis["beta"] == pytest.approx(0.25)


def test_analyze_beta_pass_raises_without_a_root_in_range() -> None:
    """Given a beta sweep with no root, when analyzed, then it raises."""
    beta_range = np.linspace(-1.0, 1.0, 21)
    run_result = {
        "target": f"{QUBIT}-fh",
        "pulse_type": "hpi",
        "beta_range": beta_range,
        "n_turns": 1,
        "raw": np.zeros_like(beta_range, dtype=np.complex128),
        "normalized": 0.3 * (beta_range - 10.0),
    }

    with pytest.raises(ValueError, match="found no root"):
        _core.analyze_beta_pass(run_result=run_result, degree=1)


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_beta_pass_wraps_the_body_in_the_ladder(
    stub_experiment_factory: Callable[..., Any],
    pulse_type: Literal["hpi", "pi"],
) -> None:
    """Given a beta sweep, when run, then every schedule keeps the full ladder."""
    exp = _build_experiment(stub_experiment_factory, "fh")
    ladder = _core.resolve_ladder(exp, target=QUBIT, transition="fh")
    prep_pulses = _core.resolve_prep_pulses(exp, ladder)
    analyzer = exp.pulse.get_hpi_pulse(ladder.target).shifted(-np.pi / 2)

    _core.run_beta_pass(
        exp,
        ladder=ladder,
        prep_pulses=prep_pulses,
        rabi_param=exp.ctx.get_rabi_param(ladder.target),
        pulse_type=pulse_type,
        make_drive=lambda beta: Drag(duration=16.0, amplitude=0.2, beta=beta),
        analyzer=analyzer,
        beta_range=np.linspace(-1.0, 1.0, 5, dtype=np.float64),
        n_turns=1,
        n_shots=1,
        shot_interval=0.0,
    )

    schedules = exp.sweep_calls[-1]["schedules"]
    assert len(schedules) == 5
    for schedule in schedules:
        assert set(schedule.labels) == {*ladder.lower_labels, ladder.target}


def _chevron_run_result(
    *,
    carrier: float,
    resonance_offset: float,
    rabi_rate: float,
    detunings: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    """Return a synthetic chevron payload with a known resonance."""
    delta = detunings[:, None] - resonance_offset
    effective = np.sqrt(rabi_rate**2 + delta**2)
    weight = (rabi_rate / effective) ** 2
    z = 1.0 - 2.0 * weight * np.sin(np.pi * effective * times[None, :]) ** 2
    raw = (0.3 + 1j * (0.5 * z + 0.1)) * np.exp(1j * 0.7)
    return {
        "target": f"{QUBIT}-fh",
        "detuning_range": detunings,
        "time_range": times,
        "effective_time_range": times,
        "control_frequencies": carrier + detunings,
        "frequency": carrier,
        "amplitude": 0.1,
        "raw": raw.ravel(),
    }


def test_chevron_analysis_recovers_the_resonance() -> None:
    """Given a synthetic chevron, when analyzed, then the resonance is recovered."""
    carrier = 8.0
    resonance_offset = 0.004
    run_result = _chevron_run_result(
        carrier=carrier,
        resonance_offset=resonance_offset,
        rabi_rate=0.02,
        detunings=np.linspace(-0.02, 0.02, 21),
        times=np.arange(0, 201, 8, dtype=np.float64),
    )

    analysis = analyze_chevron_pattern_based_on_ground(
        run_result=run_result,
        rabi_fit_r2_threshold=0.9,
    )

    assert analysis["resonant_frequency"] == pytest.approx(
        carrier + resonance_offset, abs=1e-4
    )
    assert analysis["chevron_data"].shape == (
        run_result["time_range"].size,
        run_result["detuning_range"].size,
    )
    assert analysis["raw_grid"].shape == (
        run_result["detuning_range"].size,
        run_result["time_range"].size,
    )


def test_chevron_analysis_raises_when_too_few_rows_fit() -> None:
    """Given an unusable R² threshold, when analyzed, then it raises."""
    run_result = _chevron_run_result(
        carrier=8.0,
        resonance_offset=0.004,
        rabi_rate=0.02,
        detunings=np.linspace(-0.02, 0.02, 21),
        times=np.arange(0, 201, 8, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="produced a Rabi fit above"):
        analyze_chevron_pattern_based_on_ground(
            run_result=run_result,
            rabi_fit_r2_threshold=1.5,
        )


@pytest.mark.parametrize(
    "pulse",
    [
        FlatTop(duration=DURATION, amplitude=0.3, tau=RAMPTIME),
        Drag(duration=16.0, amplitude=0.3, beta=0.7),
    ],
    ids=["flat_top", "drag"],
)
def test_inverted_leading_pulse_matches_the_dropped_repetitions(pulse: Any) -> None:
    """Given a signed leading pulse, when built, then the rotation is unchanged."""
    n_rotations = 3
    repetitions = 4 * n_rotations
    bodies = {
        branch: _core.build_branch_body(
            pulse=pulse,
            branch=branch,
            repetitions=repetitions,
            tipping_pulse=pulse,
        )
        for branch in _core.BRANCHES
    }

    # The total rotation is what the unequal-repetition construction would give.
    assert float(bodies[_core.BRANCH_MINUS].real.sum()) == pytest.approx(
        float(pulse.repeated(repetitions - 1).real.sum())
    )
    assert float(bodies[_core.BRANCH_PLUS].real.sum()) == pytest.approx(
        float(pulse.repeated(repetitions + 1).real.sum())
    )

    # ...but the branches now carry the same pulse count and the same duration,
    # so nothing that accumulates per pulse can bias the crossing.
    assert len(bodies[_core.BRANCH_MINUS].values) == len(
        bodies[_core.BRANCH_PLUS].values
    )
    assert bodies[_core.BRANCH_MINUS].duration == bodies[_core.BRANCH_PLUS].duration
    assert len(bodies[_core.BRANCH_PLUS].values) == len(
        pulse.repeated(repetitions + 1).values
    )


def test_iq_figure_matches_the_reference_iq_plane_styling() -> None:
    """Given IQ samples, when the figure is built, then it uses the IQPlotter styling."""
    raw = np.array([1 + 2j, 3 + 1j, -1 - 0.5j])
    fig = _core.iq_figure(title="IQ", raw=raw)

    # No connecting line between the swept points.
    assert fig.data[0].mode == "markers"

    # Both axes span the same symmetric range, so the origin sits at the centre.
    assert fig.layout.xaxis.range == fig.layout.yaxis.range
    assert fig.layout.xaxis.range[0] == -fig.layout.xaxis.range[1]

    # I = 0 and Q = 0 are drawn in black.
    assert fig.layout.xaxis.zeroline
    assert fig.layout.xaxis.zerolinecolor == "black"
    assert fig.layout.yaxis.zeroline
    assert fig.layout.yaxis.zerolinecolor == "black"

    # One unit of I is as long as one unit of Q.
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


@pytest.mark.parametrize("initial_amplitude", [0.05, 0.3, 0.9, 0.999])
@pytest.mark.parametrize("n_rotations", [1, 4, 16])
def test_default_amplitude_diff_range_is_always_accepted_by_its_validator(
    initial_amplitude: float,
    n_rotations: int,
) -> None:
    """Given any usable starting amplitude, when defaulted, then the sweep is valid."""
    diffs = _core.default_amplitude_diff_range(
        initial_amplitude=initial_amplitude,
        n_rotations=n_rotations,
    )

    # Symmetric about zero, so the starting amplitude stays at the centre.
    assert diffs[0] == pytest.approx(-diffs[-1])
    assert len(diffs) == _core.DEFAULT_N_POINTS
    # The swept amplitudes never leave (0, 1].
    amplitudes = initial_amplitude + diffs
    assert amplitudes.min() > 0
    assert amplitudes.max() <= 1
    _core.validate_amplitude_diff_range(
        diffs,
        initial_amplitude=initial_amplitude,
        n_rotations=n_rotations,
    )


def test_default_amplitude_diff_range_rejects_a_saturated_amplitude() -> None:
    """Given a starting amplitude of 1, when defaulted, then it refuses to guess."""
    with pytest.raises(ValueError, match="no room to sweep upwards"):
        _core.default_amplitude_diff_range(initial_amplitude=1.0, n_rotations=1)


def test_defaults_are_taken_from_the_core_experiment_constants() -> None:
    """Given the defaults, when compared, then they are the `Experiment` values."""
    assert _core.DEFAULT_N_SHOTS is constants.CALIBRATION_SHOTS
    assert _core.DEFAULT_SHOT_INTERVAL is constants.DEFAULT_INTERVAL
    assert _core.DEFAULT_TIME_RANGE is constants.DEFAULT_RABI_TIME_RANGE
    assert _core.DEFAULT_DRAG_HPI_DURATION is constants.DRAG_HPI_DURATION
    assert _core.DEFAULT_DRAG_PI_DURATION is constants.DRAG_PI_DURATION
    assert _core.DEFAULT_DRAG_COEFF is constants.DRAG_COEFF
    # Not an `Experiment` constant: `Drag` itself falls back to a Gaussian.
    assert _core.DEFAULT_DRAG_TYPE == "Gaussian"
    # `MeasurementService.obtain_rabi_params` derives its ramp the same way.
    assert (
        _core.DEFAULT_RABI_RAMPTIME == constants.HPI_DURATION - constants.HPI_RAMPTIME
    )


def _run_result(target: str = "Q00-fh") -> dict[str, Any]:
    """Return a minimal run payload for the stage runner."""
    return {"target": target, "raw": np.array([1 + 1j])}


def test_analysis_stages_return_every_stage_when_all_succeed() -> None:
    """Given healthy stages, when run, then the payload carries no error key."""
    seen: list[str] = []

    payload, figures = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=lambda run_result: {**run_result, "amplitude": 0.3},
        store=lambda _: seen.append("store"),
        plot=lambda _: {"branches": go.Figure()},
        display=lambda _: seen.append("display"),
    )

    assert seen == ["store", "display"]
    assert payload["amplitude"] == 0.3
    assert "error" not in payload
    assert list(figures) == ["branches"]


def test_a_failed_analysis_keeps_the_raw_data_and_skips_the_rest() -> None:
    """Given a failing analysis, when run, then the run payload comes back."""
    seen: list[str] = []
    error = ValueError("no crossing")

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        raise error

    payload, figures = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=_analyze,
        store=lambda _: seen.append("store"),
        plot=lambda _: seen.append("plot") or {},  # type: ignore[func-returns-value]
        display=lambda _: seen.append("display"),
    )

    # Nothing downstream runs: they all read keys only the analysis produces.
    assert seen == []
    assert payload["error"] is error
    assert payload["raw"] is not None
    assert figures == {}


def test_a_failed_plot_keeps_the_analysis_and_the_stored_calibration() -> None:
    """Given a failing plot, when run, then the analysis and the store survive."""
    seen: list[str] = []
    error = RuntimeError("no renderer")

    def _plot(payload: dict[str, Any]) -> dict[str, Any]:
        raise error

    payload, figures = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=lambda run_result: {**run_result, "amplitude": 0.3},
        store=lambda _: seen.append("store"),
        plot=_plot,
        display=lambda _: seen.append("display"),
    )

    assert seen == ["store"]
    assert payload["amplitude"] == 0.3
    assert payload["error"] is error
    assert figures == {}


def test_a_failed_store_still_builds_the_figures() -> None:
    """Given a failing store, when run, then plotting and display still happen."""
    seen: list[str] = []
    error = KeyError("calib note")

    def _store(payload: dict[str, Any]) -> None:
        raise error

    payload, figures = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=lambda run_result: {**run_result, "amplitude": 0.3},
        store=_store,
        plot=lambda _: {"branches": go.Figure()},
        display=lambda _: seen.append("display"),
    )

    assert seen == ["display"]
    assert payload["error"] is error
    assert list(figures) == ["branches"]


def test_a_failed_display_keeps_the_payload_and_the_figures() -> None:
    """Given a failing display, when run, then everything before it survives."""
    error = RuntimeError("no notebook")

    def _display(result: Any) -> None:
        raise error

    payload, figures = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=lambda run_result: {**run_result, "amplitude": 0.3},
        store=None,
        plot=lambda _: {"branches": go.Figure()},
        display=_display,
    )

    assert payload["amplitude"] == 0.3
    assert payload["error"] is error
    assert list(figures) == ["branches"]


def test_only_the_first_failure_is_kept() -> None:
    """Given two failing stages, when run, then the upstream error is the one kept."""
    first = KeyError("store")
    second = RuntimeError("display")

    def _store(payload: dict[str, Any]) -> None:
        raise first

    def _display(result: Any) -> None:
        raise second

    payload, _ = _core.run_analysis_stages(
        run_result=_run_result(),
        analyze=lambda run_result: {**run_result, "amplitude": 0.3},
        store=_store,
        plot=lambda _: {"branches": go.Figure()},
        display=_display,
    )

    assert payload["error"] is first


@pytest.mark.parametrize("beta", [0.0, -0.5, 1.4286, 1e-12])
def test_validate_beta_accepts_any_finite_value(beta: float) -> None:
    """Given a finite beta, when validated, then it passes - zero included."""
    assert _core.validate_beta(beta) is None


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_validate_beta_rejects_a_non_finite_value(beta: float) -> None:
    """Given a non-finite beta, when validated, then it raises."""
    with pytest.raises(ValueError, match="must be a finite number"):
        _core.validate_beta(beta)


@pytest.mark.parametrize("type", _core.DRAG_TYPES)
def test_validate_drag_type_accepts_every_envelope_drag_can_build(
    type: RampType,
) -> None:
    """Given a supported envelope, when validated, then it passes."""
    assert _core.validate_drag_type(type) is None
    # The allowlist is only trustworthy if `Drag` really builds each of them.
    assert Drag(duration=16, amplitude=0.1, beta=0.0, type=type).length > 0


@pytest.mark.parametrize("type", ["Squad", "Blackmann", ""])
def test_validate_drag_type_rejects_an_envelope_drag_cannot_build(type: str) -> None:
    """Given an unsupported envelope, when validated, then it raises."""
    with pytest.raises(ValueError, match="is not a supported DRAG envelope"):
        _core.validate_drag_type(cast(RampType, type))


def test_every_ramp_type_is_either_supported_or_deliberately_left_out() -> None:
    """Given `RampType`, when a member is added, then this test must be revisited."""
    assert set(get_args(RampType)) - set(_core.DRAG_TYPES) == {"Squad"}


def _readout_scan_run_result(
    *,
    signals: np.ndarray,
    frequency_range: np.ndarray | None = None,
    center_frequency: float | None = None,
) -> dict[str, Any]:
    """Return the payload `run_measure_dispersive_shift_extended` would produce."""
    max_level = signals.shape[0] - 1
    if frequency_range is None:
        frequency_range = np.linspace(10.0, 10.02, signals.shape[1])
    if center_frequency is None:
        center_frequency = float(np.mean(frequency_range))
    labels = tuple(
        QUBIT if name == "ge" else f"{QUBIT}-{name}"
        for name in TRANSITION_NAMES[:max_level]
    )
    return {
        "target": QUBIT,
        "qubit": QUBIT,
        "read_label": f"R{QUBIT}",
        "max_level": max_level,
        "levels": tuple(range(max_level + 1)),
        "state_labels": tuple(LEVEL_NAMES[: max_level + 1]),
        "labels": labels,
        "frequency_range": frequency_range,
        "center_frequency": center_frequency,
        "readout_amplitude": 0.05,
        "electrical_delay": 0.0,
        "alternate_state_order": False,
        "acquisition_order": (),
        "signals": signals,
        "n_shots": 128,
        "shot_interval": 1024.0,
    }


def _phasors(phases: np.ndarray, amplitudes: np.ndarray | None = None) -> np.ndarray:
    """Return unit-amplitude phasors unless amplitudes are given."""
    if amplitudes is None:
        amplitudes = np.ones_like(phases)
    return amplitudes * np.exp(1j * phases)


def _two_level_scan(peak: int, n_points: int = 21) -> np.ndarray:
    """Return a g/e scan whose phase separation spikes at one frequency."""
    phases = np.full((2, n_points), 0.0)
    phases[1] = 0.1
    phases[1, peak] = 2.0
    return _phasors(phases)


def test_readout_scan_optimum_is_the_widest_phase_separation() -> None:
    """Given a spiked separation, when analyzed, then that frequency is chosen."""
    peak = 7
    run_result = _readout_scan_run_result(signals=_two_level_scan(peak))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    subset = data["subsets"][0]
    assert subset["optimal_index"] == peak
    assert subset["optimal_frequency"] == run_result["frequency_range"][peak]
    assert subset["optimal_phase_separation"] == pytest.approx(2.0)
    assert subset["resolved"]


def test_readout_scan_phase_distance_folds_across_pi() -> None:
    """Given phases beyond pi apart, when analyzed, then the distance folds back."""
    n_points = 21
    peak = 5
    phases = np.zeros((2, n_points))
    phases[1] = 3 * np.pi / 2
    phases[1, peak] = np.pi
    run_result = _readout_scan_run_result(signals=_phasors(phases))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    subset = data["subsets"][0]
    separation = subset["phase_separation"]
    assert np.all(separation <= np.pi + 1e-12)
    assert separation[0] == pytest.approx(np.pi / 2)
    assert subset["optimal_index"] == peak
    assert subset["optimal_phase_separation"] == pytest.approx(np.pi)


def test_readout_scan_subsets_are_named_after_their_levels() -> None:
    """Given a three-transition ladder, when analyzed, then subsets nest by name."""
    rng = np.random.default_rng(0)
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(4, 15)))
    run_result = _readout_scan_run_result(signals=signals)

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    subsets = data["subsets"]
    assert [subset["name"] for subset in subsets] == ["ge", "gef", "gefh"]
    assert [len(subset["levels"]) for subset in subsets] == [2, 3, 4]
    assert [subset["ideal_separation"] for subset in subsets] == pytest.approx(
        [np.pi, 2 * np.pi / 3, np.pi / 2]
    )


def test_readout_scan_ignores_points_where_a_level_vanishes() -> None:
    """Given a vanishing level, when analyzed, then that frequency is not chosen."""
    n_points = 21
    best, runner_up = 3, 11
    phases = np.zeros((2, n_points))
    phases[1] = 0.1
    phases[1, best] = 3.0
    phases[1, runner_up] = 2.0
    amplitudes = np.ones((2, n_points))
    amplitudes[1, best] = 1e-3
    run_result = _readout_scan_run_result(signals=_phasors(phases, amplitudes))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    assert not data["valid_mask"][best]
    assert data["subsets"][0]["optimal_index"] == runner_up


def test_readout_scan_ignores_non_finite_points() -> None:
    """Given a non-finite point, when analyzed, then it is not chosen."""
    n_points = 21
    best, runner_up = 4, 12
    phases = np.zeros((2, n_points))
    phases[1] = 0.1
    phases[1, runner_up] = 2.0
    signals = _phasors(phases)
    signals[1, best] = np.nan

    data = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals)
    )

    assert not data["valid_mask"][best]
    assert data["subsets"][0]["optimal_index"] == runner_up


def test_readout_scan_breaks_ties_towards_the_center_frequency() -> None:
    """Given two equal maxima, when analyzed, then the nearer one to center wins."""
    n_points = 21
    near, far = 9, 1
    phases = np.zeros((2, n_points))
    phases[1] = 0.1
    phases[1, [near, far]] = 2.0
    frequencies = np.linspace(10.0, 10.02, n_points)
    run_result = _readout_scan_run_result(
        signals=_phasors(phases),
        frequency_range=frequencies,
        center_frequency=float(frequencies[10]),
    )

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    assert data["subsets"][0]["optimal_index"] == near


def test_readout_scan_flags_a_subset_that_never_separates() -> None:
    """Given coincident levels, when analyzed, then the subset is unresolved."""
    n_points = 11
    phases = np.zeros((2, n_points))
    phases[1] = 1e-3
    run_result = _readout_scan_run_result(signals=_phasors(phases))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    assert not data["subsets"][0]["resolved"]


def test_readout_scan_reports_the_pair_that_limits_the_optimum() -> None:
    """Given an uneven triple, when analyzed, then the closest pair is reported."""
    n_points = 11
    peak = 4
    phases = np.zeros((3, n_points))
    phases[1] = 0.05
    phases[2] = 0.1
    phases[1, peak] = 2.0
    phases[2, peak] = 2.5
    run_result = _readout_scan_run_result(signals=_phasors(phases))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    subset = data["subsets"][1]
    assert subset["name"] == "gef"
    assert subset["optimal_index"] == peak
    # |e> and |f> sit 0.5 rad apart, closer than either is to |g>.
    assert subset["limiting_pair"] == (1, 2)
    assert subset["optimal_phase_separation"] == pytest.approx(0.5)


def test_readout_scan_keeps_the_unwrapped_absolute_phase() -> None:
    """Given a winding phase, when analyzed, then it is kept and unwrapped."""
    n_points = 41
    # A phase that runs well past 2 pi, as a cable delay makes it do.
    ramp = np.linspace(0.0, 8 * np.pi, n_points)
    signals = _phasors(np.stack([ramp, ramp + 0.3]))

    data = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals)
    )

    absolute = data["absolute_phases"]
    # Continuous: no wrap survives, so no step reaches pi.
    assert np.all(np.abs(np.diff(absolute, axis=1)) < np.pi)
    # Still the measured phase, up to the turns the wrap had removed.
    assert np.allclose(np.angle(np.exp(1j * (absolute - data["phases"]))), 0.0)
    # The winding is recovered rather than folded away.
    assert np.allclose(absolute[0] - absolute[0][0], ramp - ramp[0])
    assert absolute[0][-1] == pytest.approx(8 * np.pi)


def test_readout_scan_phases_are_referenced_to_the_ground_state() -> None:
    """Given any scan, when analyzed, then the ground-state trace is flat at zero."""
    rng = np.random.default_rng(1)
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(3, 9)))

    data = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals)
    )

    assert np.allclose(data["relative_phases"][0], 0.0)


def test_readout_scan_is_invariant_to_a_common_phase_factor() -> None:
    """Given a cable delay, when analyzed, then neither optimum moves."""
    rng = np.random.default_rng(2)
    n_points = 17
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(3, n_points)))
    delay = np.exp(1j * np.linspace(0.0, 40.0, n_points))

    plain = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals)
    )
    delayed = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals * delay)
    )

    for left, right in zip(plain["subsets"], delayed["subsets"], strict=True):
        assert left["optimal_frequency"] == right["optimal_frequency"]
        assert (
            left["optimal_frequency_by_distance"]
            == (right["optimal_frequency_by_distance"])
        )
        assert np.allclose(left["phase_separation"], right["phase_separation"])
        assert np.allclose(left["distance_separation"], right["distance_separation"])


def test_readout_scan_distance_optimum_can_differ_from_the_phase_optimum() -> None:
    """Given contrast that peaks elsewhere, when analyzed, then the optima split."""
    n_points = 21
    phase_peak, distance_peak = 5, 15
    phases = np.zeros((2, n_points))
    phases[1] = 0.2
    phases[1, phase_peak] = 1.0
    amplitudes = np.ones((2, n_points))
    amplitudes[1, distance_peak] = 4.0
    run_result = _readout_scan_run_result(signals=_phasors(phases, amplitudes))

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    subset = data["subsets"][0]
    assert subset["optimal_index"] == phase_peak
    assert subset["optimal_index_by_distance"] == distance_peak


def test_readout_scan_raises_when_no_point_is_usable() -> None:
    """Given an all non-finite scan, when analyzed, then it raises."""
    signals = np.full((2, 5), np.nan, dtype=np.complex128)

    with pytest.raises(ValueError, match="No frequency of the scan is usable"):
        analyze_measure_dispersive_shift_extended(
            run_result=_readout_scan_run_result(signals=signals)
        )


def test_readout_scan_corrects_the_signals_by_the_electrical_delay() -> None:
    """Given a delay, when analyzed, then the corrected signals de-embed it."""
    n_points = 13
    tau = 250.0
    signals = _phasors(np.zeros((2, n_points)))
    run_result = _readout_scan_run_result(signals=signals)
    run_result["electrical_delay"] = tau

    data = analyze_measure_dispersive_shift_extended(run_result=run_result)

    frequencies = run_result["frequency_range"]
    assert np.allclose(
        data["corrected_signals"],
        signals * np.exp(2j * np.pi * frequencies * tau),
    )
    # The raw signals survive beside the corrected ones.
    assert np.array_equal(data["signals"], signals)


def test_readout_scan_leaves_the_signals_alone_at_zero_delay() -> None:
    """Given no delay, when analyzed, then corrected and raw agree exactly."""
    rng = np.random.default_rng(4)
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(2, 9)))

    data = analyze_measure_dispersive_shift_extended(
        run_result=_readout_scan_run_result(signals=signals)
    )

    assert np.array_equal(data["corrected_signals"], signals)


def test_readout_scan_de_embeds_the_cable_from_the_absolute_phase() -> None:
    """Given a cable ramp, when corrected, then the absolute phase goes flat."""
    n_points = 41
    frequencies = np.linspace(10.0, 10.01, n_points)
    tau = 300.0
    offsets = (0.0, 0.4)
    # A flat resonator response seen through a long cable.
    signals = np.stack(
        [
            np.exp(1j * offset) * np.exp(-2j * np.pi * frequencies * tau)
            for offset in offsets
        ]
    )
    run_result = _readout_scan_run_result(
        signals=signals,
        frequency_range=frequencies,
    )

    uncorrected = analyze_measure_dispersive_shift_extended(run_result=run_result)
    corrected = analyze_measure_dispersive_shift_extended(
        run_result={**run_result, "electrical_delay": tau}
    )

    # Without the correction the trace winds through many turns.
    winding = uncorrected["absolute_phases"][0]
    assert abs(winding[-1] - winding[0]) > 2 * np.pi
    # With it, only the constant the resonator set is left.
    flat = corrected["absolute_phases"]
    assert np.allclose(flat[0], flat[0][0])
    assert np.allclose(flat[1] - flat[0], offsets[1] - offsets[0])


def test_readout_scan_optimum_does_not_move_with_the_electrical_delay() -> None:
    """Given any delay, when analyzed, then neither metric changes at all."""
    rng = np.random.default_rng(5)
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(3, 17)))
    run_result = _readout_scan_run_result(signals=signals)

    plain = analyze_measure_dispersive_shift_extended(run_result=run_result)
    delayed = analyze_measure_dispersive_shift_extended(
        run_result={**run_result, "electrical_delay": 137.0}
    )

    for left, right in zip(plain["subsets"], delayed["subsets"], strict=True):
        # The metrics come from the raw signals, so this is exact, not close.
        assert left["optimal_frequency"] == right["optimal_frequency"]
        assert left["optimal_phase_separation"] == right["optimal_phase_separation"]
        assert np.array_equal(left["phase_separation"], right["phase_separation"])
        assert np.array_equal(left["distance_separation"], right["distance_separation"])


def test_readout_scan_analysis_is_idempotent() -> None:
    """Given an analyzed payload, when analyzed again, then nothing changes."""
    rng = np.random.default_rng(6)
    signals = _phasors(rng.uniform(0, 2 * np.pi, size=(3, 15)))
    run_result = {
        **_readout_scan_run_result(signals=signals),
        "electrical_delay": 90.0,
    }

    once = analyze_measure_dispersive_shift_extended(run_result=run_result)
    twice = analyze_measure_dispersive_shift_extended(run_result=once)

    for key in (
        "signals",
        "corrected_signals",
        "phases",
        "absolute_phases",
        "relative_phases",
        "amplitudes",
        "valid_mask",
    ):
        assert np.array_equal(once[key], twice[key]), key
    for left, right in zip(once["subsets"], twice["subsets"], strict=True):
        assert left["optimal_frequency"] == right["optimal_frequency"]
        assert left["limiting_pair"] == right["limiting_pair"]


def test_readout_scan_requires_the_electrical_delay_to_be_present() -> None:
    """Given a payload without the key, when analyzed, then it says so."""
    run_result = _readout_scan_run_result(signals=_phasors(np.zeros((2, 5))))
    del run_result["electrical_delay"]

    with pytest.raises(KeyError, match="electrical_delay"):
        analyze_measure_dispersive_shift_extended(run_result=run_result)
