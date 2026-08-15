"""End-to-end tests driving the experiments against a simulating stub."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.calibrate_1q_transition_based_on_ground import (
    calibrate_drag_hpi_pulse_based_on_ground,
    calibrate_drag_pi_pulse_based_on_ground,
    calibrate_hpi_pulse_based_on_ground,
    calibrate_pi_pulse_based_on_ground,
    obtain_rabi_params_based_on_ground,
    run_chevron_pattern_based_on_ground,
)

QUBIT = "Q00"
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
        target=QUBIT,
        transition="fh",
        time_range=np.arange(0, 401, 8),
        amplitude=amplitude,
        ramptime=RAMPTIME,
        is_damped=False,
        fit_threshold=0.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert result.data["target"] == f"{QUBIT}-fh"
    assert result.data["lower_labels"] == (QUBIT, f"{QUBIT}-ef")
    rabi_param = result.data["rabi_param"]
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
        target=QUBIT,
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
    assert result.data["initial_amplitude"] == pytest.approx(stored_amplitude)
    assert result.data["initial_amplitude_source"] == "calib_note"
    assert result.data["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.hpi[target]["amplitude"] == pytest.approx(
        result.data["amplitude"]
    )
    assert exp.calib_note.hpi[target]["duration"] == DURATION
    assert exp.calib_note.hpi[target]["tau"] == RAMPTIME


def test_hpi_branches_have_the_same_duration(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the padded short branch, when swept, then both branches match in time."""
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
        target=QUBIT,
        transition="ef",
        duration=DURATION,
        ramptime=RAMPTIME,
        n_rotations=2,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    durations = {schedule.duration for schedule in exp.sweep_calls[-1]["schedules"]}
    assert len(durations) == 1


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
        target=QUBIT,
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

    assert result.data["initial_amplitude"] == pytest.approx(stored_amplitude)
    assert result.data["initial_amplitude_source"] == "calib_note"
    assert result.data["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.pi[target]["amplitude"] == pytest.approx(
        result.data["amplitude"]
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
            target=QUBIT,
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
        target=QUBIT,
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

    assert result.data["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert len(result.data["iterations"]) == 2
    assert exp.calib_note.drag_hpi[target]["amplitude"] == pytest.approx(
        result.data["amplitude"]
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
        target=QUBIT,
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

    assert result.data["amplitude"] == pytest.approx(true_amplitude, rel=1e-4)
    assert exp.calib_note.drag_pi[target]["amplitude"] == pytest.approx(
        result.data["amplitude"]
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
            target=QUBIT,
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
        target=QUBIT,
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

    assert result.data["lower_labels"] == ()
    assert result.data["amplitude"] == pytest.approx(0.3012, rel=1e-4)
    for schedule in exp.sweep_calls[-1]["schedules"]:
        assert schedule.labels == [QUBIT]
    assert exp.calib_note.hpi[QUBIT]["amplitude"] == pytest.approx(
        result.data["amplitude"]
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
