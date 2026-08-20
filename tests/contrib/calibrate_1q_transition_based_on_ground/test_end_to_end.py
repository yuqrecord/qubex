"""End-to-end tests driving the experiments against a simulating stub."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.calibrate_1q_transition_based_on_ground import (
    _core,
    calibrate_drag_hpi_pulse_based_on_ground,
    calibrate_drag_pi_pulse_based_on_ground,
    calibrate_hpi_pulse_based_on_ground,
    calibrate_pi_pulse_based_on_ground,
    chevron_pattern_based_on_ground,
    obtain_rabi_params_based_on_ground,
    run_chevron_pattern_based_on_ground,
)

from .conftest import CONTROL_AMPLITUDE

QUBIT = "Q00"
QUBIT_2 = "Q01"
QUBIT_3 = "Q02"
DURATION = 32.0
RAMPTIME = 12.0
DRAG_HPI_DURATION = 16.0
DRAG_PI_DURATION = 24.0
DRAG_COEFF = 0.5
N_SHOTS = 1024
SHOT_INTERVAL = 10_000.0


def test_rabi_params_are_stored_under_the_transition_label(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated fh transition, when measured, then params land on the label."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.4,
        pulse_type="pi",
    )
    amplitude = 0.1
    # z = cos(2 pi * rate * amplitude * (T + ramptime)) in the stub.
    expected_frequency = exp.rabi_rate_per_amplitude * amplitude

    result = obtain_rabi_params_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        time_range=np.arange(0, 401, 8),
        amplitudes=amplitude,
        ramptime=RAMPTIME,
        is_damped=False,
        fit_threshold=0.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    target = f"{QUBIT}-fh"
    assert list(result.data) == [target]
    assert result.data[target]["target"] == target
    assert result.data[target]["lower_labels"] == (QUBIT, f"{QUBIT}-ef")
    rabi_param = result.data[target]["rabi_param"]
    assert rabi_param.frequency == pytest.approx(expected_frequency, rel=1e-3)
    assert exp.ctx.stored_rabi_params[f"{QUBIT}-fh"] is rabi_param
    assert f"{QUBIT}-fh" in exp.calib_note.rabi
    assert exp.reference_calls[0]["targets"] == [QUBIT]


def test_hpi_calibration_finds_the_simulated_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated fh transition, when calibrated, then the amplitude is found."""
    stored_amplitude = 0.30
    true_amplitude = 0.3012
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=true_amplitude,
        pulse_type="hpi",
        hpi_amplitude=stored_amplitude,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    target = f"{QUBIT}-fh"
    assert result.data[target]["initial_amplitude"] == pytest.approx(stored_amplitude)
    assert result.data[target]["initial_amplitude_source"] == "calib_note"
    assert result.data[target]["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.hpi[target]["amplitude"] == pytest.approx(
        result.data[target]["amplitude"]
    )
    assert exp.calib_note.hpi[target]["duration"] == DURATION
    assert exp.calib_note.hpi[target]["tau"] == RAMPTIME


def test_hpi_branches_are_matched(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the two branches, when swept, then they match in time and in drive."""
    n_rotations = 2
    exp = stub_experiment_factory(
        transition="ef",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ef",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=n_rotations,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    schedules = exp.sweep_calls[-1]["schedules"]
    target = f"{QUBIT}-ef"
    assert len({schedule.duration for schedule in schedules}) == 1
    # No branch is padded with an idle Blank: every schedule drives the
    # transition channel for the same number of samples, so leakage and
    # relaxation accumulate equally on both branches.
    driven = {
        int(np.count_nonzero(schedule.get_sampled_sequences()[target]))
        for schedule in schedules
    }
    assert len(driven) == 1
    assert driven.pop() == (4 * n_rotations + 1) * int(DURATION / 2)


def test_pi_calibration_finds_the_simulated_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated ef transition, when calibrated, then the amplitude is found."""
    stored_amplitude = 0.60
    true_amplitude = 0.6018
    exp = stub_experiment_factory(
        transition="ef",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=true_amplitude,
        pulse_type="pi",
    )
    target = f"{QUBIT}-ef"
    exp.calib_note.update_pi_param(
        target,
        {
            "target": target,
            "duration": DURATION,
            "amplitude": stored_amplitude,
            "tau": RAMPTIME,
        },
    )

    result = calibrate_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ef",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.006, 0.006, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert result.data[target]["initial_amplitude"] == pytest.approx(stored_amplitude)
    assert result.data[target]["initial_amplitude_source"] == "calib_note"
    assert result.data[target]["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.pi[target]["amplitude"] == pytest.approx(
        result.data[target]["amplitude"]
    )


def test_pi_calibration_requires_a_half_pi_pulse(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given no stored half-pi, when calibrated, then it raises before measuring."""
    exp = stub_experiment_factory(
        transition="ef",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.6,
        pulse_type="pi",
        with_hpi=False,
    )

    with pytest.raises(ValueError, match="half-pi pulse is not calibrated"):
        calibrate_pi_pulse_based_on_ground(
            exp,
            targets=QUBIT,
            transition="ef",
            duration=DURATION,
            ramptime=RAMPTIME,
            n_rotations=3,
            amplitude_diff_range=np.linspace(-0.006, 0.006, 11),
            n_shots=N_SHOTS,
            shot_interval=SHOT_INTERVAL,
            plot=False,
            plot_pulse_schedule=False,
        )

    assert exp.sweep_calls == []
    assert exp.calib_note.pi.get(f"{QUBIT}-ef") is None


def test_drag_hpi_calibration_finds_the_simulated_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated hi transition, when calibrated, then DRAG amplitude is found."""
    stored_amplitude = 0.20
    true_amplitude = 0.2009
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=true_amplitude,
        pulse_type="hpi",
        beta=0.0,
    )
    target = f"{QUBIT}-hi"
    exp.calib_note.update_drag_hpi_param(
        target,
        {
            "target": target,
            "duration": DRAG_HPI_DURATION,
            "amplitude": stored_amplitude,
            "beta": 0.0,
        },
    )

    result = calibrate_drag_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        drag_coeff=DRAG_COEFF,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_iterations=2,
        calibrate_beta=False,
        beta_range=np.linspace(-0.5, 0.5, 11),
        n_turns=1,
        degree=3,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert result.data[target]["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert len(result.data[target]["iterations"]) == 2
    assert exp.calib_note.drag_hpi[target]["amplitude"] == pytest.approx(
        result.data[target]["amplitude"]
    )
    assert exp.calib_note.drag_hpi[target]["beta"] == 0.0
    assert exp.calib_note.drag_hpi[target]["duration"] == DRAG_HPI_DURATION


def test_drag_pi_calibration_finds_the_simulated_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated fh transition, when calibrated, then DRAG amplitude is found."""
    stored_amplitude = 0.40
    true_amplitude = 0.4013
    exp = stub_experiment_factory(
        transition="fh",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=true_amplitude,
        pulse_type="pi",
        beta=0.0,
    )
    target = f"{QUBIT}-fh"
    exp.calib_note.update_drag_pi_param(
        target,
        {
            "target": target,
            "duration": DRAG_PI_DURATION,
            "amplitude": stored_amplitude,
            "beta": 0.0,
        },
    )

    result = calibrate_drag_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DRAG_PI_DURATION,
        drag_coeff=DRAG_COEFF,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_iterations=1,
        calibrate_beta=False,
        beta_range=np.linspace(-0.5, 0.5, 11),
        n_turns=1,
        degree=3,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert result.data[target]["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.drag_pi[target]["amplitude"] == pytest.approx(
        result.data[target]["amplitude"]
    )


def test_missing_rabi_params_raise_before_measuring(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given no transition Rabi params, when calibrated, then nothing is measured."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3,
        pulse_type="hpi",
        hpi_amplitude=0.3,
    )
    exp.ctx.stored_rabi_params.clear()
    before = dict(exp.calib_note.hpi)

    with pytest.raises(ValueError, match="Rabi parameters are not stored"):
        calibrate_hpi_pulse_based_on_ground(
            exp,
            targets=QUBIT,
            transition="fh",
            duration=DURATION,
            ramptime=RAMPTIME,
            n_rotations=3,
            amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
            n_shots=N_SHOTS,
            shot_interval=SHOT_INTERVAL,
            plot=False,
            plot_pulse_schedule=False,
        )

    assert exp.sweep_calls == []
    assert exp.calib_note.hpi == before


def test_ge_calibration_needs_no_ladder(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the ge transition, when calibrated, then only the qubit is driven."""
    exp = stub_experiment_factory(
        transition="ge",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ge",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert result.data[QUBIT]["lower_labels"] == ()
    assert result.data[QUBIT]["amplitude"] == pytest.approx(0.3012, rel=1e-4)
    for schedule in exp.sweep_calls[-1]["schedules"]:
        assert schedule.labels == [QUBIT]
    assert exp.calib_note.hpi[QUBIT]["amplitude"] == pytest.approx(
        result.data[QUBIT]["amplitude"]
    )


def test_chevron_run_visits_every_grid_point_inside_the_ladder(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a detuning x time grid, when run, then each point keeps the ladder."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.4,
        pulse_type="pi",
    )
    detunings = np.linspace(-0.01, 0.01, 5)
    times = np.arange(0, 41, 8, dtype=np.float64)

    run_result = run_chevron_pattern_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        detuning_range=detunings,
        time_range=times,
        amplitude=0.1,
        frequency=7.5,
        ramptime=0.0,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    call = exp.sweep_calls[-1]
    assert call["frequencies"] == {f"{QUBIT}-fh": 7.5}
    assert len(call["schedules"]) == detunings.size * times.size
    assert run_result["raw"].size == detunings.size * times.size
    for schedule in call["schedules"]:
        assert set(schedule.labels) == {QUBIT, f"{QUBIT}-ef", f"{QUBIT}-fh"}


def test_two_targets_are_calibrated_one_after_another(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given two qubits, when calibrated, then each is swept and stored on its own."""
    # Both sit inside the swept half-width of the shared stored amplitude.
    true_amplitudes = {QUBIT: 0.3012, QUBIT_2: 0.2988}
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.30,
        pulse_type="hpi",
        hpi_amplitude=0.30,
        qubits=[QUBIT, QUBIT_2],
        true_amplitudes=true_amplitudes,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2],
        transition="fh",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    targets = [f"{QUBIT}-fh", f"{QUBIT_2}-fh"]
    assert list(result.data) == targets
    # One sweep per target, in the requested order, each driving only its own qubit.
    assert len(exp.sweep_calls) == 2
    for target, call in zip(targets, exp.sweep_calls, strict=True):
        for schedule in call["schedules"]:
            assert target in schedule.labels
    for target in targets:
        qubit = target.split("-")[0]
        assert result.data[target]["amplitude"] == pytest.approx(
            true_amplitudes[qubit], rel=1e-4
        )
        assert exp.calib_note.hpi[target]["amplitude"] == pytest.approx(
            result.data[target]["amplitude"]
        )
        assert f"{target}:branches" in result.figures


def test_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when run, then the payload records the defaults."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=_core.DEFAULT_HPI_DURATION,
        ramptime=_core.DEFAULT_HPI_RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(exp, transition="fh", plot=False)

    payload = result.data[f"{QUBIT}-fh"]
    assert payload["duration"] == _core.DEFAULT_HPI_DURATION
    assert payload["ramptime"] == _core.DEFAULT_HPI_RAMPTIME
    assert payload["n_rotations"] == _core.DEFAULT_N_ROTATIONS
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    assert payload["shot_interval"] == _core.DEFAULT_SHOT_INTERVAL
    assert len(payload["amplitude_diff_range"]) == _core.DEFAULT_N_POINTS
    # `DEFAULT_N_ROTATIONS` is 1, so the crossing is only coarsely amplified.
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-2)


def test_rabi_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when the Rabi sweep runs, then it uses the defaults."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )

    result = obtain_rabi_params_based_on_ground(
        exp,
        transition="fh",
        is_damped=False,
        plot=False,
        plot_pulse_schedule=False,
    )

    payload = result.data[f"{QUBIT}-fh"]
    assert payload["ramptime"] == _core.DEFAULT_RABI_RAMPTIME
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    assert payload["shot_interval"] == _core.DEFAULT_SHOT_INTERVAL
    assert payload["fit_threshold"] == _core.DEFAULT_FIT_THRESHOLD
    np.testing.assert_allclose(
        payload["time_range"],
        np.asarray(_core.DEFAULT_TIME_RANGE, dtype=np.float64),
    )
    # `transition_index("fh") == 2`, so the accessor divides by sqrt(3).
    assert payload["amplitude"] == pytest.approx(CONTROL_AMPLITUDE / np.sqrt(3))


def test_chevron_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when the chevron runs, then it uses the defaults."""
    exp = stub_experiment_factory(
        transition="ef",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )

    result = chevron_pattern_based_on_ground(
        exp,
        transition="ef",
        detuning_range=np.linspace(-0.01, 0.01, 5),
        time_range=np.arange(0, 65, 8),
        rabi_fit_r2_threshold=0.0,
        plot=False,
        plot_pulse_schedule=False,
    )

    payload = result.data[f"{QUBIT}-ef"]
    assert payload["ramptime"] == _core.DEFAULT_RABI_RAMPTIME
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    # The registered target frequency, not the qubit's bare frequency.
    assert payload["frequency"] == pytest.approx(
        exp.ctx.targets[f"{QUBIT}-ef"].frequency
    )
    assert payload["amplitude"] == pytest.approx(CONTROL_AMPLITUDE / np.sqrt(2))


@pytest.mark.parametrize(
    ("amplitudes", "expected"),
    [
        pytest.param({f"{QUBIT}-fh": 0.11}, 0.11, id="mapping-by-target"),
        pytest.param({QUBIT: 0.12}, 0.12, id="mapping-by-qubit"),
        pytest.param(0.13, 0.13, id="scalar"),
        pytest.param(None, None, id="accessor"),
    ],
)
def test_rabi_amplitudes_accept_a_mapping_a_scalar_or_none(
    stub_experiment_factory: Callable[..., Any],
    amplitudes: Any,
    expected: float | None,
) -> None:
    """Given each spelling of `amplitudes`, when run, then the drive matches it."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )

    result = obtain_rabi_params_based_on_ground(
        exp,
        transition="fh",
        time_range=np.arange(0, 129, 8),
        amplitudes=amplitudes,
        is_damped=False,
        plot=False,
        plot_pulse_schedule=False,
    )

    if expected is None:
        expected = CONTROL_AMPLITUDE / np.sqrt(3)
    assert result.data[f"{QUBIT}-fh"]["amplitude"] == pytest.approx(expected)


def test_targets_default_to_the_qubits_that_have_the_transition_registered(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a partly registered chip, when targets is None, then the rest is skipped."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        qubits=[QUBIT, QUBIT_2, QUBIT_3],
        registered_qubits=[QUBIT, QUBIT_3],
        calibrated_qubits=[QUBIT, QUBIT_3],
    )

    result = obtain_rabi_params_based_on_ground(
        exp,
        transition="fh",
        time_range=np.arange(0, 129, 8),
        is_damped=False,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert list(result.data) == [f"{QUBIT}-fh", f"{QUBIT_3}-fh"]


def test_an_unusable_target_stops_the_batch_before_it_measures(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given one uncalibrated qubit, when calibrated, then nothing is measured."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
        qubits=[QUBIT, QUBIT_2],
        calibrated_qubits=[QUBIT],
    )
    before = dict(exp.calib_note.hpi)

    with pytest.raises(ValueError):
        calibrate_hpi_pulse_based_on_ground(
            exp,
            targets=[QUBIT, QUBIT_2],
            transition="fh",
            duration=DURATION,
            ramptime=RAMPTIME,
            n_rotations=3,
            amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
            n_shots=N_SHOTS,
            shot_interval=SHOT_INTERVAL,
            plot=False,
            plot_pulse_schedule=False,
        )

    # The healthy first target is not measured either, so nothing is half written.
    assert exp.sweep_calls == []
    assert exp.calib_note.hpi == before


def test_drag_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when the DRAG amplitude runs, then it uses the defaults."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="drag",
        duration=_core.DEFAULT_DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )

    # The stub integrates only the in-phase quadrature, so its signal carries no
    # beta dependence and the beta pass has no root to find. The amplitude pass
    # is what the defaults are checked on.
    result = calibrate_drag_hpi_pulse_based_on_ground(
        exp,
        transition="fh",
        calibrate_beta=False,
        plot=False,
        plot_pulse_schedule=False,
    )

    payload = result.data[f"{QUBIT}-fh"]
    assert payload["duration"] == _core.DEFAULT_DRAG_HPI_DURATION
    assert payload["drag_coeff"] == _core.DEFAULT_DRAG_COEFF
    assert payload["n_rotations"] == _core.DEFAULT_DRAG_N_ROTATIONS
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    assert len(payload["iterations"]) == _core.DEFAULT_N_ITERATIONS
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


def test_calibration_figures_are_plotted_against_the_amplitude_itself(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a calibration, when plotted, then the x axis is the amplitude, not the offset."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    target = f"{QUBIT}-fh"
    payload = result.data[target]
    initial_amplitude = payload["initial_amplitude"]
    for name in ("branches", "difference", "residuals"):
        figure = result.figures[f"{target}:{name}"]
        layout = figure.layout
        assert layout.xaxis.title.text == "Amplitude (a.u.)"
        for trace in figure.data:
            # The offset axis is drawn by an invisible placeholder trace.
            if trace.xaxis == "x2":
                continue
            np.testing.assert_allclose(
                np.asarray(trace.x, dtype=np.float64),
                payload["amplitude_range"],
            )

        # The top axis relabels the same range as offsets from the start.
        assert layout.xaxis2.title.text == "Amplitude difference (a.u.)"
        assert layout.xaxis2.side == "top"
        assert layout.xaxis2.overlaying == "x"
        np.testing.assert_allclose(
            np.asarray(layout.xaxis2.range, dtype=np.float64),
            np.asarray(layout.xaxis.range, dtype=np.float64) - initial_amplitude,
        )

    # The crossing is marked at the calibrated amplitude, not at the offset.
    difference = result.figures[f"{target}:difference"]
    assert difference.layout.shapes[-1].x0 == pytest.approx(payload["amplitude"])
    assert f"{payload['amplitude']:.6g}" in difference.layout.annotations[0].text


def _module(name: str) -> Any:
    """Return an experiment submodule, which the package shadows with its function."""
    return importlib.import_module(
        f"qubex.contrib.experiment.calibrate_1q_transition_based_on_ground.{name}"
    )


def test_a_failing_analysis_keeps_the_measurement_and_the_other_targets(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given one qubit the fit cannot solve, when calibrated, then the rest still runs."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
        qubits=[QUBIT, QUBIT_2, QUBIT_3],
        # Q01 sits far outside the swept window, so its crossing is an
        # extrapolation and `_core.fit_crossing` refuses it.
        true_amplitudes={QUBIT: 0.3012, QUBIT_2: 0.2513, QUBIT_3: 0.2988},
    )

    seeded = {target: dict(param) for target, param in exp.calib_note.hpi.items()}

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2, QUBIT_3],
        transition="fh",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    good, bad, also_good = (f"{q}-fh" for q in (QUBIT, QUBIT_2, QUBIT_3))
    assert list(result.data) == [good, bad, also_good]

    # The target that failed keeps its raw sweep and reports why.
    assert isinstance(result.data[bad]["error"], ValueError)
    assert result.data[bad]["raw"].size > 0
    assert "amplitude" not in result.data[bad]
    assert bad not in result.figures
    # Its stored calibration is left exactly as it was.
    assert exp.calib_note.hpi[bad] == seeded[bad]

    # Every other target is measured and stored as if nothing had happened.
    for target in (good, also_good):
        assert "error" not in result.data[target]
        assert exp.calib_note.hpi[target]["amplitude"] == pytest.approx(
            result.data[target]["amplitude"]
        )
        assert f"{target}:branches" in result.figures
    assert len(exp.sweep_calls) == 3


def test_a_failing_plot_keeps_the_stored_calibration(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a plotting failure, when calibrated, then the amplitude is still stored."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    def _boom(data: Any) -> Any:
        raise RuntimeError("no renderer")

    monkeypatch.setattr(
        _module("calibrate_hpi_pulse_based_on_ground"),
        "plot_calibrate_hpi_pulse_based_on_ground",
        _boom,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    target = f"{QUBIT}-fh"
    assert isinstance(result.data[target]["error"], RuntimeError)
    # The analysis and the calibration-note write happened before the figures.
    assert result.data[target]["amplitude"] == pytest.approx(0.3012, rel=1e-4)
    assert exp.calib_note.hpi[target]["amplitude"] == pytest.approx(
        result.data[target]["amplitude"]
    )
    assert result.figures == {}


def test_a_failing_measurement_is_raised(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a measurement failure, when calibrated, then it is not swallowed."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("box is gone")

    monkeypatch.setattr(
        _module("calibrate_hpi_pulse_based_on_ground"),
        "run_calibrate_hpi_pulse_based_on_ground",
        _boom,
    )

    with pytest.raises(ConnectionError, match="box is gone"):
        calibrate_hpi_pulse_based_on_ground(
            exp,
            targets=QUBIT,
            transition="fh",
            plot=False,
            plot_pulse_schedule=False,
        )


def _drag_stub(stub_experiment_factory: Callable[..., Any]) -> Any:
    """Return a DRAG stub seeded so that the amplitude pass converges."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )
    target = f"{QUBIT}-hi"
    exp.calib_note.update_drag_hpi_param(
        target,
        {
            "target": target,
            "duration": DRAG_HPI_DURATION,
            "amplitude": 0.30,
            "beta": 0.0,
        },
    )
    return exp


def _fail_drag_analysis_on_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_call: int,
) -> None:
    """Make the DRAG analysis raise on the n-th call, counting from one."""
    module = _module("calibrate_drag_hpi_pulse_based_on_ground")
    original = module.analyze_calibrate_drag_hpi_pulse_based_on_ground
    calls = itertools.count(1)

    def _maybe_fail(**kwargs: Any) -> Any:
        if next(calls) == failing_call:
            raise ValueError("no crossing")
        return original(**kwargs)

    monkeypatch.setattr(
        module,
        "analyze_calibrate_drag_hpi_pulse_based_on_ground",
        _maybe_fail,
    )


def _run_drag(exp: Any) -> Any:
    return calibrate_drag_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        drag_coeff=DRAG_COEFF,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_iterations=2,
        calibrate_beta=False,
        beta_range=np.linspace(-0.5, 0.5, 11),
        n_turns=1,
        degree=3,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )


def test_a_drag_pass_that_fails_late_keeps_the_earlier_passes(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a failing second pass, when calibrated, then the first pass survives."""
    exp = _drag_stub(stub_experiment_factory)
    _fail_drag_analysis_on_call(monkeypatch, failing_call=2)

    payload = _run_drag(exp).data[f"{QUBIT}-hi"]

    assert isinstance(payload["error"], ValueError)
    # The descent stopped, but the pass that did converge is the reported result.
    assert len(payload["iterations"]) == 1
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)
    assert exp.calib_note.drag_hpi[f"{QUBIT}-hi"]["amplitude"] == pytest.approx(
        payload["amplitude"]
    )


def test_a_drag_pass_that_fails_first_returns_the_raw_sweep(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a failing first pass, when calibrated, then it does not index an empty list."""
    exp = _drag_stub(stub_experiment_factory)
    _fail_drag_analysis_on_call(monkeypatch, failing_call=1)

    payload = _run_drag(exp).data[f"{QUBIT}-hi"]

    assert isinstance(payload["error"], ValueError)
    assert "iterations" not in payload
    assert payload["raw"].size > 0
    # The seeded calibration is untouched, because no pass ever converged.
    assert exp.calib_note.drag_hpi[f"{QUBIT}-hi"]["amplitude"] == pytest.approx(0.30)
