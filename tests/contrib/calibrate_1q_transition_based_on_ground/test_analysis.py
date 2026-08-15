"""Tests for the shared machinery of the ground-referenced transition package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pytest

from qubex.contrib.experiment.calibrate_1q_transition_based_on_ground import (
    _core,
    analyze_chevron_pattern_based_on_ground,
)
from qubex.pulse import Drag, FlatTop
from qubex.system.transition import TRANSITION_NAMES, TransitionName

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
