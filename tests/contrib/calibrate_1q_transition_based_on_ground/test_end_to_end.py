"""End-to-end tests driving the experiments against a simulating stub."""

from __future__ import annotations

import dataclasses
import importlib
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qubex.contrib.experiment.calibrate_1q_transition_based_on_ground import (
    _core,
    analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground,
    apply_electrical_delay,
    apply_phase_shift,
    calibrate_drag_hpi_pulse_based_on_ground,
    calibrate_drag_pi_pulse_based_on_ground,
    calibrate_hpi_pulse_based_on_ground,
    calibrate_pi_pulse_based_on_ground,
    chevron_pattern_based_on_ground,
    chevron_pattern_by_amplitude_sweep_based_on_ground,
    measure_dispersive_shift_extended,
    obtain_rabi_params_based_on_ground,
    obtain_rabi_params_by_amplitude_sweep_based_on_ground,
    ramsey_experiment_based_on_ground,
    run_chevron_pattern_based_on_ground,
    run_chevron_pattern_by_amplitude_sweep_based_on_ground,
    run_obtain_rabi_params_by_amplitude_sweep_based_on_ground,
    run_ramsey_experiment_based_on_ground,
)
from qubex.experiment import experiment_constants as constants
from qubex.experiment.models import Result
from qubex.pulse import Drag, Waveform
from qubex.system.transition import TRANSITION_NAMES

from .conftest import CONTROL_AMPLITUDE, DISPERSIVE_SHIFT, READOUT_FREQUENCY

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
        envelope="drag",
        duration=DURATION,
        true_amplitude=true_amplitude,
        pulse_type="hpi",
        hpi_amplitude=stored_amplitude,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DURATION,
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
    # A null `tau` is what tells `PulseService` to rebuild this as an envelope.
    assert exp.calib_note.hpi[target]["tau"] is None
    assert exp.calib_note.hpi[target]["beta"] == 0.0
    assert exp.calib_note.hpi[target]["type"] == _core.DEFAULT_DRAG_TYPE


def test_hpi_branches_are_matched(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the two branches, when swept, then they match in time and in drive."""
    n_rotations = 2
    exp = stub_experiment_factory(
        transition="ef",
        envelope="drag",
        duration=DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ef",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
            "tau": None,
            "beta": 0.0,
        },
    )

    result = calibrate_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ef",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="ge",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
        envelope="drag",
        duration=_core.DEFAULT_DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(exp, transition="fh", plot=False)

    payload = result.data[f"{QUBIT}-fh"]
    assert payload["duration"] == _core.DEFAULT_DRAG_HPI_DURATION
    assert payload["type"] == _core.DEFAULT_DRAG_TYPE
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
    # Readout amplification is left for `MeasurementService.measure` to resolve.
    assert payload["readout_amplification"] is None
    assert exp.reference_calls[-1]["readout_amplification"] is None
    assert exp.sweep_calls[-1]["readout_amplification"] is None


@pytest.mark.parametrize("readout_amplification", [True, False])
def test_rabi_forwards_readout_amplification_to_every_acquisition(
    stub_experiment_factory: Callable[..., Any],
    readout_amplification: bool,
) -> None:
    """Given readout amplification, when the Rabi runs, then the reference and the sweep use it."""
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
        targets=QUBIT,
        transition="fh",
        time_range=np.arange(0, 401, 8),
        is_damped=False,
        readout_amplification=readout_amplification,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert exp.reference_calls[-1]["readout_amplification"] is readout_amplification
    assert exp.sweep_calls[-1]["readout_amplification"] is readout_amplification
    payload = result.data[f"{QUBIT}-fh"]
    assert payload["readout_amplification"] is readout_amplification


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
        envelope="drag",
        duration=DURATION,
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
    assert payload["type"] == _core.DEFAULT_DRAG_TYPE
    assert payload["n_rotations"] == _core.DEFAULT_DRAG_N_ROTATIONS
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    assert len(payload["iterations"]) == _core.DEFAULT_N_ITERATIONS
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)
    # The envelope is stored, so `get_drag_hpi_pulse` rebuilds the same pulse.
    assert exp.calib_note.drag_hpi[f"{QUBIT}-fh"]["type"] == _core.DEFAULT_DRAG_TYPE


def test_calibration_figures_are_plotted_against_the_amplitude_itself(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a calibration, when plotted, then the x axis is the amplitude, not the offset."""
    exp = stub_experiment_factory(
        transition="fh",
        envelope="drag",
        duration=DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        hpi_amplitude=0.30,
    )

    result = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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
        envelope="drag",
        duration=DURATION,
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


def _drag_stub(
    stub_experiment_factory: Callable[..., Any],
    *,
    stored_type: str | None = None,
) -> Any:
    """Return a DRAG stub seeded so that the amplitude pass converges."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )
    target = f"{QUBIT}-hi"
    stored: dict[str, Any] = {
        "target": target,
        "duration": DRAG_HPI_DURATION,
        "amplitude": 0.30,
        "beta": 0.0,
    }
    if stored_type is not None:
        stored["type"] = stored_type
    exp.calib_note.update_drag_hpi_param(target, stored)
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


def _run_drag_with_beta(
    exp: Any,
    *,
    beta: float | None,
    calibrate_beta: bool = False,
) -> Any:
    """Run the DRAG half-pi calibration with an explicit starting beta."""
    return calibrate_drag_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        drag_coeff=DRAG_COEFF,
        beta=beta,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_iterations=2,
        calibrate_beta=calibrate_beta,
        beta_range=np.linspace(-0.5, 0.5, 11),
        n_turns=1,
        degree=3,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )


def test_a_given_beta_of_zero_is_not_mistaken_for_an_omitted_one(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given beta=0, when calibrated, then the correction is turned off, not defaulted."""
    exp = _drag_stub(stub_experiment_factory)
    target = f"{QUBIT}-hi"

    payload = _run_drag_with_beta(exp, beta=0).data[target]

    assert payload["beta"] == 0.0
    assert payload["beta_source"] == "argument"
    assert exp.calib_note.drag_hpi[target]["beta"] == 0.0
    # Every pass keeps it: there is nothing refining beta.
    assert [iteration["beta"] for iteration in payload["iterations"]] == [0.0, 0.0]


def test_a_given_beta_replaces_only_the_beta(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a stored calibration, when beta is given, then the amplitude seed survives."""
    exp = _drag_stub(stub_experiment_factory)
    target = f"{QUBIT}-hi"
    stored_amplitude = exp.calib_note.drag_hpi[target]["amplitude"]

    payload = _run_drag_with_beta(exp, beta=0.2).data[target]

    assert payload["beta"] == pytest.approx(0.2)
    assert payload["beta_source"] == "argument"
    # The rotation angle is set by the real part alone, so an amplitude measured
    # at some other beta is still the best seed there is.
    assert payload["initial_amplitude_source"] == "calib_note"
    assert payload["initial_amplitude"] == pytest.approx(stored_amplitude, rel=1e-2)


@pytest.mark.parametrize(
    ("stored_beta", "expected_beta", "expected_source"),
    [
        pytest.param(0.85, 0.85, "calib_note", id="stored"),
        pytest.param(None, None, "ladder_estimate", id="derived"),
    ],
)
def test_an_omitted_beta_keeps_the_previous_behaviour(
    stub_experiment_factory: Callable[..., Any],
    stored_beta: float | None,
    expected_beta: float | None,
    expected_source: str,
) -> None:
    """Given no beta, when calibrated, then the stored or derived value is used."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )
    target = f"{QUBIT}-hi"
    if stored_beta is not None:
        exp.calib_note.update_drag_hpi_param(
            target,
            {
                "target": target,
                "duration": DRAG_HPI_DURATION,
                "amplitude": 0.30,
                "beta": stored_beta,
            },
        )
    if expected_beta is None:
        expected_beta = _core.estimate_drag_beta(
            exp,
            qubit=QUBIT,
            transition="hi",
            drag_coeff=DRAG_COEFF,
        )

    payload = _run_drag_with_beta(exp, beta=None).data[target]

    assert payload["beta"] == pytest.approx(expected_beta)
    assert payload["beta_source"] == expected_source


def test_a_given_beta_centres_the_beta_sweep(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given beta, when the beta pass runs, then the sweep is centred on it."""
    exp = _drag_stub(stub_experiment_factory)

    # The stub carries no beta dependence, so the pass finds no root and the
    # analysis raises. The sweep it emitted is what is under test here.
    _run_drag_with_beta(exp, beta=0.2, calibrate_beta=True)

    beta_sweep = exp.sweep_calls[-1]["sweep_range"]
    np.testing.assert_allclose(beta_sweep, 0.2 + np.linspace(-0.5, 0.5, 11))


def test_a_non_finite_beta_is_rejected(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a non-finite beta, when calibrated, then it is refused up front."""
    exp = _drag_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match="must be a finite number"):
        _run_drag_with_beta(exp, beta=float("nan"))


READOUT_DF = 0.0001
READOUT_WIDTH = 0.01


def _readout_scan(
    experiment: Any,
    *,
    max_level: int,
    **kwargs: Any,
) -> Any:
    """Run a readout scan with a fine grid and no figures shown."""
    kwargs.setdefault("frequency_width", READOUT_WIDTH)
    kwargs.setdefault("df", READOUT_DF)
    return measure_dispersive_shift_extended(
        experiment,
        max_level=max_level,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
        **kwargs,
    )


def test_readout_scan_finds_the_frequency_between_two_levels(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given two levels, when scanned, then the optimum sits between them."""
    exp = readout_scan_experiment_factory(max_level=1)

    result = _readout_scan(exp, max_level=1)

    subset = result.data[QUBIT]["subsets"][0]
    midpoint = READOUT_FREQUENCY + DISPERSIVE_SHIFT / 2
    assert subset["name"] == "ge"
    assert subset["limiting_pair"] == (0, 1)
    assert subset["resolved"]
    assert abs(subset["optimal_frequency"] - midpoint) <= READOUT_DF


def test_readout_scan_reports_both_readout_criteria(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a three-rung ladder, when scanned, then both criteria are reported."""
    exp = readout_scan_experiment_factory(max_level=3)

    result = _readout_scan(exp, max_level=3)

    payload = result.data[QUBIT]
    assert "error" not in payload
    assert [subset["name"] for subset in payload["subsets"]] == ["ge", "geL"]
    assert payload["signals"].shape == (4, payload["frequency_range"].size)
    assert set(result.figures) == {
        f"{QUBIT}:{name}"
        for name in (
            "absolute_phase",
            "phase",
            "phase_separation",
            "distance_separation",
            "amplitude",
            "iq",
        )
    }


def test_readout_scan_measures_every_level_at_every_frequency(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a scan, when run, then each level is acquired once per frequency."""
    exp = readout_scan_experiment_factory(max_level=2)

    result = _readout_scan(exp, max_level=2)

    n_points = result.data[QUBIT]["frequency_range"].size
    assert len(exp.measure_calls) == 3 * n_points
    assert all(call["mode"] == "avg" for call in exp.measure_calls)
    assert all(call["n_shots"] == N_SHOTS for call in exp.measure_calls)
    assert all(not call["reset_awg_and_capunits"] for call in exp.measure_calls)


def test_readout_scan_leaves_the_hardware_alone_by_default(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given no request, when scanned, then the AWG is not reset."""
    exp = readout_scan_experiment_factory(max_level=1)

    _readout_scan(exp, max_level=1)

    assert exp.ctx.reset_calls == []


def test_readout_scan_resets_the_hardware_once_when_asked(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given the flag, when scanned, then the AWG is reset once per target."""
    exp = readout_scan_experiment_factory(max_level=1)

    _readout_scan(exp, max_level=1, reset_awg_and_capunits=True)

    assert exp.ctx.reset_calls == [(QUBIT,)]


def test_readout_scan_alternates_the_level_order_when_asked(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given the flag, when scanned, then the level order flips every point."""
    exp = readout_scan_experiment_factory(max_level=2)

    result = _readout_scan(exp, max_level=2, alternate_state_order=True)

    order = result.data[QUBIT]["acquisition_order"]
    assert order[0] == (0, 1, 2)
    assert order[1] == (2, 1, 0)
    assert order[2] == (0, 1, 2)


def test_readout_scan_sweeps_a_given_range_in_the_order_given(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given an explicit range, when scanned, then it is used and not sorted."""
    exp = readout_scan_experiment_factory(max_level=1)
    given = [10.003, 10.001, 10.002]

    result = _readout_scan(exp, max_level=1, frequency_range=given)

    assert np.array_equal(result.data[QUBIT]["frequency_range"], given)


def test_readout_scan_refuses_a_sweep_outside_the_tuning_window(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given too wide a sweep, when scanned, then it raises before measuring."""
    exp = readout_scan_experiment_factory(max_level=1)

    with pytest.raises(ValueError, match="tuning window"):
        _readout_scan(exp, max_level=1, frequency_width=0.8, df=0.01)

    assert exp.measure_calls == []


def test_readout_scan_refuses_an_uncalibrated_rung_before_measuring(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a missing pi pulse, when scanned, then it raises before measuring."""
    exp = readout_scan_experiment_factory(
        max_level=2,
        qubits=[QUBIT, QUBIT_2],
        calibrated_qubits=[QUBIT],
    )

    with pytest.raises(ValueError, match="Pi pulses are not calibrated"):
        _readout_scan(exp, max_level=2)

    assert exp.measure_calls == []


def test_readout_scan_refuses_an_unregistered_rung_before_measuring(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a missing target, when scanned, then it raises before measuring."""
    exp = readout_scan_experiment_factory(max_level=1)

    with pytest.raises(ValueError, match="not registered"):
        _readout_scan(exp, max_level=2, targets=QUBIT)

    assert exp.measure_calls == []


def test_readout_scan_keys_the_result_by_qubit_label(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given several qubits, when scanned, then each is keyed by its own label."""
    exp = readout_scan_experiment_factory(max_level=1, qubits=[QUBIT, QUBIT_2])

    result = _readout_scan(exp, max_level=1)

    assert set(result.data) == {QUBIT, QUBIT_2}
    for qubit in (QUBIT, QUBIT_2):
        assert result.data[qubit]["qubit"] == qubit
        assert result.data[qubit]["read_label"] == f"R{qubit}"
        assert f"{qubit}:phase" in result.figures


@pytest.mark.parametrize("max_level", [0, len(TRANSITION_NAMES) + 1])
def test_readout_scan_rejects_a_level_outside_the_ladder(
    readout_scan_experiment_factory: Callable[..., Any],
    max_level: int,
) -> None:
    """Given an impossible level, when scanned, then it raises before measuring."""
    exp = readout_scan_experiment_factory(max_level=1)

    with pytest.raises(ValueError, match="`max_level` must lie in"):
        _readout_scan(exp, max_level=max_level)

    assert exp.measure_calls == []


DELAY = 250.0


def test_apply_electrical_delay_rederives_without_touching_hardware(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a finished scan, when a delay is applied, then nothing is measured."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)
    acquisitions = len(exp.measure_calls)

    updated = apply_electrical_delay(result, DELAY)

    assert len(exp.measure_calls) == acquisitions
    payload = updated.data[QUBIT]
    assert payload["electrical_delay"] == DELAY
    assert np.allclose(
        payload["corrected_signals"],
        payload["signals"] * np.exp(2j * np.pi * payload["frequency_range"] * DELAY),
    )
    assert set(updated.figures or {}) == set(result.figures or {})


def test_apply_electrical_delay_leaves_the_original_result_alone(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a result, when a delay is applied, then the input is not modified."""
    exp = readout_scan_experiment_factory(max_level=1)
    result = _readout_scan(exp, max_level=1)
    before = np.array(result.data[QUBIT]["corrected_signals"], copy=True)

    updated = apply_electrical_delay(result, DELAY)

    assert result.data[QUBIT]["electrical_delay"] == 0.0
    assert np.array_equal(result.data[QUBIT]["corrected_signals"], before)
    assert updated.data[QUBIT] is not result.data[QUBIT]


def test_apply_electrical_delay_does_not_move_the_optimum(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a delay, when applied, then every reported optimum is unchanged."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)

    updated = apply_electrical_delay(result, DELAY)

    for left, right in zip(
        result.data[QUBIT]["subsets"],
        updated.data[QUBIT]["subsets"],
        strict=True,
    ):
        assert left["optimal_frequency"] == right["optimal_frequency"]
        assert left["optimal_phase_separation"] == right["optimal_phase_separation"]
        assert (
            left["optimal_frequency_by_distance"]
            == right["optimal_frequency_by_distance"]
        )


def test_apply_electrical_delay_takes_one_value_per_qubit(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a mapping, when applied, then each qubit gets its own delay."""
    exp = readout_scan_experiment_factory(max_level=1, qubits=[QUBIT, QUBIT_2])
    result = _readout_scan(exp, max_level=1)

    updated = apply_electrical_delay(result, {QUBIT: 100.0, QUBIT_2: 200.0})

    assert updated.data[QUBIT]["electrical_delay"] == 100.0
    assert updated.data[QUBIT_2]["electrical_delay"] == 200.0


def test_apply_electrical_delay_rejects_an_incomplete_mapping(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a mapping missing a qubit, when applied, then it raises."""
    exp = readout_scan_experiment_factory(max_level=1, qubits=[QUBIT, QUBIT_2])
    result = _readout_scan(exp, max_level=1)

    with pytest.raises(ValueError, match="has no entry for"):
        apply_electrical_delay(result, {QUBIT: 100.0})


def test_apply_electrical_delay_rejects_a_non_finite_delay(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a non-finite delay, when applied, then it raises."""
    exp = readout_scan_experiment_factory(max_level=1)
    result = _readout_scan(exp, max_level=1)

    with pytest.raises(ValueError, match="must be a finite number"):
        apply_electrical_delay(result, float("nan"))


def test_apply_electrical_delay_is_idempotent(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given the same delay twice, when applied, then the result is unchanged."""
    exp = readout_scan_experiment_factory(max_level=1)
    result = _readout_scan(exp, max_level=1)

    once = apply_electrical_delay(result, DELAY)
    twice = apply_electrical_delay(once, DELAY)

    assert np.array_equal(
        once.data[QUBIT]["corrected_signals"],
        twice.data[QUBIT]["corrected_signals"],
    )
    assert np.array_equal(
        once.data[QUBIT]["absolute_phases"],
        twice.data[QUBIT]["absolute_phases"],
    )


def test_apply_electrical_delay_can_be_undone_with_zero(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a corrected result, when zero is applied, then the raw data is back."""
    exp = readout_scan_experiment_factory(max_level=1)
    result = _readout_scan(exp, max_level=1)

    restored = apply_electrical_delay(apply_electrical_delay(result, DELAY), 0.0)

    assert np.array_equal(
        restored.data[QUBIT]["corrected_signals"],
        restored.data[QUBIT]["signals"],
    )


def test_apply_electrical_delay_passes_a_failed_target_through() -> None:
    """Given a target whose analysis failed, when applied, then it is untouched."""
    error = ValueError("the analysis never got there")
    result = Result(data={QUBIT: {"target": QUBIT, "error": error}}, figures={})

    updated = apply_electrical_delay(result, DELAY)

    assert updated.data[QUBIT]["error"] is error
    assert "corrected_signals" not in updated.data[QUBIT]


def test_apply_phase_shift_moves_only_inside_the_frequency_window(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a window, when a shift is applied, then only that band moves."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)
    before = np.array(result.data[QUBIT]["absolute_phases"], copy=True)
    frequencies = result.data[QUBIT]["frequency_range"]
    low, high = frequencies[10], frequencies[25]

    updated = apply_phase_shift(
        result, {QUBIT: {"f": {"frequency_range": [low, high], "turns": -1}}}
    )

    difference = updated.data[QUBIT]["absolute_phases"] - before
    inside = (frequencies >= low) & (frequencies <= high)
    assert np.allclose(difference[2][inside], -2 * np.pi)
    assert np.allclose(difference[2][~inside], 0.0)
    assert np.allclose(difference[:2], 0.0)


def test_apply_phase_shift_leaves_the_original_result_alone(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a result, when a shift is applied, then the input is not modified."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)
    before = np.array(result.data[QUBIT]["absolute_phases"], copy=True)

    updated = apply_phase_shift(result, {QUBIT: {"f": {"turns": -1}}})

    assert np.array_equal(result.data[QUBIT]["absolute_phases"], before)
    assert updated.data[QUBIT] is not result.data[QUBIT]


def test_apply_phase_shift_does_not_move_the_optimum(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a shift, when applied, then every reported optimum is unchanged."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)

    updated = apply_phase_shift(result, {QUBIT: {"e": {"turns": 2}}})

    for left, right in zip(
        result.data[QUBIT]["subsets"],
        updated.data[QUBIT]["subsets"],
        strict=True,
    ):
        assert left["optimal_frequency"] == right["optimal_frequency"]
        assert left["optimal_phase_separation"] == right["optimal_phase_separation"]
        assert (
            left["optimal_frequency_by_distance"]
            == right["optimal_frequency_by_distance"]
        )


def test_apply_phase_shift_composes_with_the_delay_in_either_order(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given both corrections, when applied, then the order does not matter."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)
    config = {QUBIT: {"f": {"turns": -1}}}

    delay_first = apply_phase_shift(apply_electrical_delay(result, DELAY), config)
    shift_first = apply_electrical_delay(apply_phase_shift(result, config), DELAY)

    assert np.allclose(
        delay_first.data[QUBIT]["absolute_phases"],
        shift_first.data[QUBIT]["absolute_phases"],
    )
    assert delay_first.data[QUBIT]["electrical_delay"] == DELAY
    assert shift_first.data[QUBIT]["phase_shift_config"] == config[QUBIT]


def test_apply_phase_shift_takes_a_config_per_qubit(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given two qubits, when a shift is applied, then each gets its own config."""
    exp = readout_scan_experiment_factory(max_level=2, qubits=[QUBIT, QUBIT_2])
    result = _readout_scan(exp, max_level=2)

    updated = apply_phase_shift(
        result,
        {QUBIT: {"f": {"turns": -1}}, QUBIT_2: {"e": {"turns": -2}}},
    )

    first = updated.data[QUBIT]["absolute_phases"]
    second = updated.data[QUBIT_2]["absolute_phases"]
    assert np.allclose(first[2] - result.data[QUBIT]["absolute_phases"][2], -2 * np.pi)
    assert np.allclose(
        second[1] - result.data[QUBIT_2]["absolute_phases"][1], -4 * np.pi
    )


def test_apply_phase_shift_keeps_a_qubit_left_out_of_the_config(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a config naming one qubit, when applied, then the other keeps its shift."""
    exp = readout_scan_experiment_factory(max_level=2, qubits=[QUBIT, QUBIT_2])
    result = _readout_scan(exp, max_level=2)

    both = apply_phase_shift(
        result, {QUBIT: {"f": {"turns": -1}}, QUBIT_2: {"f": {"turns": -1}}}
    )
    retuned = apply_phase_shift(both, {QUBIT: {"f": {"turns": -2}}})

    assert retuned.data[QUBIT]["phase_shift_config"] == {"f": {"turns": -2}}
    assert retuned.data[QUBIT_2]["phase_shift_config"] == {"f": {"turns": -1}}
    assert np.array_equal(
        retuned.data[QUBIT_2]["absolute_phases"],
        both.data[QUBIT_2]["absolute_phases"],
    )


def test_apply_phase_shift_clears_a_qubit_with_an_empty_config(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given an empty mapping, when applied, then that qubit's shift is dropped."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)

    shifted = apply_phase_shift(result, {QUBIT: {"f": {"turns": -1}}})
    cleared = apply_phase_shift(shifted, {QUBIT: {}})

    assert np.array_equal(
        cleared.data[QUBIT]["absolute_phases"],
        result.data[QUBIT]["absolute_phases"],
    )


def test_apply_phase_shift_rejects_an_unknown_qubit(
    readout_scan_experiment_factory: Callable[..., Any],
) -> None:
    """Given a qubit the result lacks, when applied, then it says so."""
    exp = readout_scan_experiment_factory(max_level=2)
    result = _readout_scan(exp, max_level=2)

    with pytest.raises(ValueError, match="this result does not hold"):
        apply_phase_shift(result, {QUBIT_2: {"f": {"turns": -1}}})


def _ramsey_stub(
    stub_experiment_factory: Callable[..., Any],
    *,
    transition: str = "fh",
    with_drag_hpi: bool = True,
    **kwargs: Any,
) -> Any:
    """Return a stub calibrated for a Ramsey sequence on one transition."""
    exp = stub_experiment_factory(
        transition=transition,
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        beta=DRAG_COEFF,
        true_amplitude=0.3012,
        pulse_type="hpi",
        **kwargs,
    )
    # The factory only seeds the uncorrected half-pi, and a Ramsey sequence
    # prefers the DRAG one, so that is added here on top of it.
    if with_drag_hpi:
        for target in exp.ctx.stored_rabi_params:
            exp.calib_note.update_drag_hpi_param(
                target,
                {
                    "target": target,
                    "duration": DRAG_HPI_DURATION,
                    "amplitude": 0.30,
                    "beta": DRAG_COEFF,
                },
            )
    return exp


def _ramsey_body(values: np.ndarray, *, hpi_duration: float, wait: float) -> tuple:
    """Split a transition channel into `(first half-pi, wait, second half-pi)`."""
    period = Waveform.SAMPLING_PERIOD
    n_hpi = round(hpi_duration / period)
    n_wait = round(wait / period)
    start = int(np.flatnonzero(np.abs(values) > 0)[0])
    first = values[start : start + n_hpi]
    gap = values[start + n_hpi : start + n_hpi + n_wait]
    second = values[start + n_hpi + n_wait : start + 2 * n_hpi + n_wait]
    tail = values[start + 2 * n_hpi + n_wait :]
    return first, gap, second, tail


def test_ramsey_run_brackets_every_wait_with_a_half_pi_and_its_negation(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a wait sweep, when run, then each point plays X90, wait, -X90."""
    exp = _ramsey_stub(stub_experiment_factory)
    times = np.arange(0, 41, 8, dtype=np.float64)

    run_result = run_ramsey_experiment_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        time_range=times,
        detuning=0.002,
        frequency=7.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    call = exp.sweep_calls[-1]
    # Only the transition target is retuned; the lower rungs keep their carriers.
    assert call["frequencies"] == {f"{QUBIT}-fh": 7.502}
    assert run_result["drive_frequency"] == pytest.approx(7.502)
    assert len(call["schedules"]) == times.size
    # Both half-pi pulses are stored, and the DRAG one wins.
    assert run_result["hpi_source"] == "drag_hpi"

    for wait, schedule in zip(times, call["schedules"], strict=True):
        values = schedule.get_sampled_sequences()[f"{QUBIT}-fh"]
        first, gap, second, tail = _ramsey_body(
            values,
            hpi_duration=run_result["hpi_duration"],
            wait=float(wait),
        )
        assert gap.size == round(wait / Waveform.SAMPLING_PERIOD)
        assert not np.any(gap)
        assert not np.any(tail)
        # The DRAG half-pi is played, not the uncorrected one: only DRAG carries
        # a quadrature component, and the uncorrected envelope is purely real.
        assert np.abs(first.imag).max() > 0
        # `-X90` is the half-pi shifted by pi, so it is the exact negation of
        # both components.
        np.testing.assert_allclose(second, -first, atol=1e-12)


def test_ramsey_run_keeps_the_fringe_inside_a_symmetric_ladder(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a high rung, when run, then the ladder brackets the Ramsey body."""
    exp = _ramsey_stub(stub_experiment_factory)

    run_ramsey_experiment_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        time_range=np.array([16.0]),
        frequency=7.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    schedule = exp.sweep_calls[-1]["schedules"][0]
    assert set(schedule.labels) == {QUBIT, f"{QUBIT}-ef", f"{QUBIT}-fh"}
    sequences = schedule.get_sampled_sequences()
    for label in (QUBIT, f"{QUBIT}-ef"):
        driven = np.flatnonzero(np.abs(sequences[label]) > 0)
        # Two blocks of equal length: the preparation and its mirror image.
        breaks = np.flatnonzero(np.diff(driven) > 1)
        assert breaks.size == 1
        prep = driven[: breaks[0] + 1]
        un_prep = driven[breaks[0] + 1 :]
        assert prep.size == un_prep.size
        np.testing.assert_allclose(
            sequences[label][prep],
            sequences[label][un_prep],
        )


@pytest.mark.parametrize(
    "stored_hpi",
    [
        # The factory seeds a `Drag` envelope at beta = 0, marked by a null `tau`.
        pytest.param(None, id="uncorrected_envelope"),
        pytest.param(
            {"duration": DURATION, "amplitude": 0.28, "tau": RAMPTIME},
            id="legacy_flat_top",
        ),
    ],
)
def test_ramsey_falls_back_to_the_uncorrected_half_pi_without_a_drag_one(
    stub_experiment_factory: Callable[..., Any],
    stored_hpi: dict[str, Any] | None,
) -> None:
    """Given no stored DRAG half-pi, when run, then the uncorrected one is played."""
    exp = _ramsey_stub(stub_experiment_factory, with_drag_hpi=False)
    target = f"{QUBIT}-fh"
    if stored_hpi is not None:
        exp.calib_note.update_hpi_param(target, {"target": target, **stored_hpi})
    expected = exp.pulse.get_hpi_pulse(target)
    times = np.arange(0, 41, 8, dtype=np.float64)

    run_result = run_ramsey_experiment_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        time_range=times,
        detuning=0.002,
        frequency=7.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    assert run_result["hpi_source"] == "hpi"
    assert run_result["hpi_duration"] == expected.duration
    call = exp.sweep_calls[-1]
    # Only the transition target is retuned, whichever half-pi is played.
    assert call["frequencies"] == {target: 7.502}
    for wait, schedule in zip(times, call["schedules"], strict=True):
        values = schedule.get_sampled_sequences()[target]
        first, gap, second, tail = _ramsey_body(
            values,
            hpi_duration=expected.duration,
            wait=float(wait),
        )
        np.testing.assert_allclose(first, expected.values, atol=1e-12)
        np.testing.assert_allclose(second, -expected.values, atol=1e-12)
        assert not np.any(gap)
        assert not np.any(tail)


def test_ramsey_requires_some_calibrated_half_pi_pulse(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a target with no half-pi at all, when run, then no target is measured."""
    exp = _ramsey_stub(stub_experiment_factory, qubits=[QUBIT, QUBIT_2])
    missing = f"{QUBIT_2}-fh"
    del exp.calib_note.hpi[missing]
    del exp.calib_note.drag_hpi[missing]

    # `get_hpi_pulse` would still hand back a nominal pulse for it, which must
    # not be played in place of a calibrated one.
    with pytest.raises(
        ValueError,
        match=rf"'{missing}'.*calibrate_hpi_pulse_based_on_ground",
    ):
        ramsey_experiment_based_on_ground(
            exp,
            targets=[QUBIT, QUBIT_2],
            transition="fh",
            time_range=np.array([16.0]),
            plot=False,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_ramsey_run_requires_some_calibrated_half_pi_pulse(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given neither half-pi stored, when run directly, then it raises before measuring."""
    exp = _ramsey_stub(stub_experiment_factory, with_drag_hpi=False, with_hpi=False)

    with pytest.raises(ValueError, match="calibrate_hpi_pulse_based_on_ground"):
        run_ramsey_experiment_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            time_range=np.array([16.0]),
            frequency=7.5,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_ramsey_requires_stored_rabi_params(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given no stored Rabi arc, when run, then it names the experiment to run."""
    exp = _ramsey_stub(stub_experiment_factory, transition="ge", calibrated_qubits=[])

    with pytest.raises(ValueError, match="obtain_rabi_params_based_on_ground"):
        ramsey_experiment_based_on_ground(
            exp,
            targets=QUBIT,
            transition="ge",
            time_range=np.array([16.0]),
            plot=False,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_ramsey_rejects_wait_times_off_the_sampling_grid(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an off-grid wait, when run, then it raises before measuring."""
    exp = _ramsey_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match="multiples of the sampling period"):
        run_ramsey_experiment_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            time_range=np.array([0.0, 8.0, 13.0]),
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_ramsey_rejects_a_detuning_that_cannot_produce_a_fringe(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a zero detuning, when run, then it says why a fringe needs one."""
    exp = _ramsey_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match="`detuning` must be a non-zero"):
        run_ramsey_experiment_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            time_range=np.array([16.0]),
            detuning=0.0,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_ramsey_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when run, then the payload records the defaults."""
    module = _module("ramsey_experiment_based_on_ground")
    exp = _ramsey_stub(stub_experiment_factory, transition="ef")

    payload = run_ramsey_experiment_based_on_ground(
        exp,
        target=QUBIT,
        transition="ef",
        plot_pulse_schedule=False,
    )

    assert payload["detuning"] == module.DEFAULT_DETUNING
    assert payload["n_shots"] == _core.DEFAULT_N_SHOTS
    assert payload["shot_interval"] == _core.DEFAULT_SHOT_INTERVAL
    np.testing.assert_allclose(
        payload["time_range"],
        np.asarray(module.DEFAULT_TIME_RANGE, dtype=np.float64),
    )
    # The registered target frequency, not the qubit's bare frequency.
    assert payload["frequency"] == pytest.approx(
        exp.ctx.targets[f"{QUBIT}-ef"].frequency
    )
    assert payload["drive_frequency"] == pytest.approx(
        payload["frequency"] + module.DEFAULT_DETUNING
    )
    # Readout amplification is left for `MeasurementService.measure` to resolve.
    assert payload["readout_amplification"] is None
    assert exp.sweep_calls[-1]["readout_amplification"] is None


@pytest.mark.parametrize("readout_amplification", [True, False])
def test_ramsey_forwards_readout_amplification_to_the_sweep(
    stub_experiment_factory: Callable[..., Any],
    readout_amplification: bool,
) -> None:
    """Given readout amplification, when the Ramsey runs, then the sweep uses it."""
    exp = _ramsey_stub(stub_experiment_factory)

    result = ramsey_experiment_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        time_range=np.arange(0, 401, 8),
        readout_amplification=readout_amplification,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert exp.sweep_calls[-1]["readout_amplification"] is readout_amplification
    payload = result.data[f"{QUBIT}-fh"]
    assert payload["readout_amplification"] is readout_amplification


def test_ramsey_entry_point_keys_by_transition_label_and_stores_nothing(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given two qubits, when run, then each is swept on its own and nothing is saved."""
    exp = _ramsey_stub(stub_experiment_factory, qubits=[QUBIT, QUBIT_2])
    seeded = {target: dict(param) for target, param in exp.calib_note.hpi.items()}
    seeded_drag = {
        target: dict(param) for target, param in exp.calib_note.drag_hpi.items()
    }

    result = ramsey_experiment_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2],
        transition="fh",
        time_range=np.arange(0, 401, 8),
        plot=False,
        plot_pulse_schedule=False,
    )

    targets = [f"{QUBIT}-fh", f"{QUBIT_2}-fh"]
    assert list(result.data) == targets
    assert len(exp.sweep_calls) == 2
    for target in targets:
        # The stub cannot free-evolve, so the fringe itself is not meaningful
        # here; what is pinned is that every stage ran and kept its payload.
        assert "error" not in result.data[target]
        assert result.data[target]["hpi_source"] == "drag_hpi"
        assert f"{target}:ramsey" in result.figures
        assert f"{target}:iq" in result.figures
    assert exp.calib_note.hpi == seeded
    assert exp.calib_note.drag_hpi == seeded_drag
    assert exp.calib_note.rabi == {}


def test_ramsey_entry_point_threads_the_decay_guess_to_the_analysis(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a decay guess, when run, then the analysis of every target gets it."""
    exp = _ramsey_stub(stub_experiment_factory, qubits=[QUBIT, QUBIT_2])

    result = ramsey_experiment_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2],
        transition="fh",
        time_range=np.arange(0, 401, 8),
        t2_estimation=750.0,
        plot=False,
        plot_pulse_schedule=False,
    )

    for target in (f"{QUBIT}-fh", f"{QUBIT_2}-fh"):
        assert result.data[target]["t2_estimation"] == 750.0


@pytest.mark.parametrize(
    ("stored_type", "expected_source"),
    [
        pytest.param(None, "calib_note", id="untyped_note_is_a_gaussian"),
        pytest.param("Gaussian", "calib_note", id="same_envelope"),
        pytest.param("Blackman", "rabi_estimate", id="other_envelope"),
    ],
)
def test_a_stored_amplitude_seeds_the_sweep_only_for_the_same_envelope(
    stub_experiment_factory: Callable[..., Any],
    stored_type: str | None,
    expected_source: str,
) -> None:
    """Given a stored envelope, when it differs, then the amplitude is re-estimated."""
    exp = _drag_stub(stub_experiment_factory, stored_type=stored_type)
    target = f"{QUBIT}-hi"

    payload = _run_drag(exp).data[target]

    # The rotation angle follows the area under the envelope, so an amplitude
    # measured with another one lands far outside the sweep this then runs.
    # Only the first pass sees the seeded note: it stores its own result under
    # the requested envelope, which every later pass then reuses.
    iterations = payload["iterations"]
    assert iterations[0]["data"]["initial_amplitude_source"] == expected_source
    assert iterations[1]["data"]["initial_amplitude_source"] == "calib_note"
    # Beta is resolved independently: it survives a change of envelope.
    assert payload["beta_source"] == "calib_note"
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


def test_the_requested_envelope_is_played_and_stored(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an envelope, when calibrated, then it drives the qubit and is stored."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        type="Blackman",
    )
    target = f"{QUBIT}-hi"

    payload = calibrate_drag_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        drag_coeff=DRAG_COEFF,
        beta=0.0,
        type="Blackman",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_iterations=1,
        calibrate_beta=False,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    assert payload["type"] == "Blackman"
    assert exp.calib_note.drag_hpi[target]["type"] == "Blackman"
    # The stub's Rabi rate is derived from the area under a *Blackman* unit
    # pulse, so converging on its true amplitude is only possible if the
    # calibration really played that envelope: a Gaussian of the same peak
    # amplitude covers about 27 % more area.
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


def test_the_pi_calibration_stores_the_requested_envelope(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an envelope, when the pi pulse is calibrated, then it is stored too."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=0.3012,
        pulse_type="pi",
        type="RaisedCosine",
    )
    target = f"{QUBIT}-hi"

    payload = calibrate_drag_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_PI_DURATION,
        drag_coeff=DRAG_COEFF,
        beta=0.0,
        type="RaisedCosine",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_iterations=1,
        calibrate_beta=False,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    assert payload["type"] == "RaisedCosine"
    assert exp.calib_note.drag_pi[target]["type"] == "RaisedCosine"
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


@pytest.mark.parametrize("type", ["Squad", "Gauss"])
def test_an_unsupported_envelope_is_rejected_before_measuring(
    stub_experiment_factory: Callable[..., Any],
    type: str,
) -> None:
    """Given an envelope `Drag` cannot build, when requested, then nothing is measured."""
    exp = _drag_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match="is not a supported DRAG envelope"):
        calibrate_drag_hpi_pulse_based_on_ground(
            exp,
            targets=QUBIT,
            transition="hi",
            type=type,  # type: ignore[arg-type]
            plot=False,
            plot_pulse_schedule=False,
        )

    assert exp.sweep_calls == []


def test_the_plain_hpi_calibration_plays_and_stores_the_requested_envelope(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an envelope, when the half-pi is calibrated, then it is played and stored."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        type="Blackman",
    )
    target = f"{QUBIT}-hi"

    payload = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        type="Blackman",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    assert payload["type"] == "Blackman"
    stored = exp.calib_note.hpi[target]
    assert stored["type"] == "Blackman"
    # A null `tau` is what keeps `PulseService` from rebuilding this as a flat
    # top, and it has to be written rather than omitted because entries merge.
    assert stored["tau"] is None
    assert stored["beta"] == 0.0
    # The stub's Rabi rate comes from the area under a *Blackman* unit pulse,
    # so converging on its true amplitude is only possible if that envelope was
    # really played: a Gaussian of the same peak amplitude covers ~27 % more.
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


def test_the_plain_pi_calibration_stores_the_requested_envelope(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given an envelope, when the pi pulse is calibrated, then it is stored too."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=0.3012,
        pulse_type="pi",
        type="RaisedCosine",
    )
    target = f"{QUBIT}-hi"

    payload = calibrate_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_PI_DURATION,
        type="RaisedCosine",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    assert payload["type"] == "RaisedCosine"
    assert exp.calib_note.pi[target]["type"] == "RaisedCosine"
    assert exp.calib_note.pi[target]["tau"] is None
    assert payload["amplitude"] == pytest.approx(0.3012, rel=1e-3)


def test_the_pi_calibration_tips_with_the_calibrated_envelope(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a stored envelope half-pi, when the pi runs, then it tips with it."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=0.3012,
        pulse_type="pi",
        type="RaisedCosine",
    )

    payload = calibrate_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_PI_DURATION,
        type="RaisedCosine",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[f"{QUBIT}-hi"]

    # The half-pi the fixture seeded is an envelope, so `get_hpi_pulse` has to
    # hand back that family rather than a flat top.
    assert isinstance(payload["tipping_pulse"], Drag)
    assert payload["tipping_pulse"].type == "RaisedCosine"


@pytest.mark.parametrize(
    "calibrate",
    [calibrate_hpi_pulse_based_on_ground, calibrate_pi_pulse_based_on_ground],
)
@pytest.mark.parametrize("type", ["Squad", "Gauss"])
def test_the_plain_calibrations_reject_an_unsupported_envelope(
    stub_experiment_factory: Callable[..., Any],
    calibrate: Callable[..., Any],
    type: str,
) -> None:
    """Given an envelope Drag cannot build, when calibrated, then nothing is measured."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
    )

    with pytest.raises(ValueError, match="is not a supported DRAG envelope"):
        calibrate(
            exp,
            targets=QUBIT,
            transition="hi",
            duration=DRAG_HPI_DURATION,
            type=type,
            plot=False,
            plot_pulse_schedule=False,
        )

    assert exp.sweep_calls == []


@pytest.mark.parametrize(
    ("stored_type", "expected_source"),
    [
        pytest.param("Blackman", "calib_note", id="same-envelope"),
        pytest.param("Gaussian", "rabi_estimate", id="other-envelope"),
    ],
)
def test_a_stored_amplitude_seeds_the_plain_sweep_only_for_the_same_envelope(
    stub_experiment_factory: Callable[..., Any],
    stored_type: str,
    expected_source: str,
) -> None:
    """Given a stored amplitude, when the envelope differs, then it is not reused."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        type="Blackman",
        with_hpi=False,
    )
    target = f"{QUBIT}-hi"
    exp.calib_note.update_hpi_param(
        target,
        {
            "target": target,
            "duration": DRAG_HPI_DURATION,
            "amplitude": 0.30,
            "tau": None,
            "beta": 0.0,
            "type": stored_type,
        },
    )

    payload = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        type="Blackman",
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    assert payload["initial_amplitude_source"] == expected_source


def test_a_stored_flat_top_does_not_seed_the_plain_sweep(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a stored flat top, when an envelope is calibrated, then it is not reused."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_HPI_DURATION,
        true_amplitude=0.3012,
        pulse_type="hpi",
        with_hpi=False,
    )
    target = f"{QUBIT}-hi"
    exp.calib_note.update_hpi_param(
        target,
        {
            "target": target,
            "duration": DRAG_HPI_DURATION,
            "amplitude": 0.30,
            "tau": RAMPTIME,
        },
    )

    payload = calibrate_hpi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_HPI_DURATION,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.003, 0.003, 11),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    ).data[target]

    # The area under a flat top and under an envelope differ, so its amplitude
    # is not a usable seed even when the duration matches.
    assert payload["initial_amplitude_source"] == "rabi_estimate"


def _capture_analyzers(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the analyzer handed to every beta pass."""
    captured: list[Any] = []
    original = _core.run_beta_pass

    def _spy(exp: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["analyzer"])
        return original(exp, **kwargs)

    monkeypatch.setattr(_core, "run_beta_pass", _spy)
    return captured


def _run_drag_pi_beta_pass(exp: Any) -> Any:
    """Run the DRAG pi calibration with the beta pass enabled."""
    return calibrate_drag_pi_pulse_based_on_ground(
        exp,
        targets=QUBIT,
        transition="hi",
        duration=DRAG_PI_DURATION,
        drag_coeff=DRAG_COEFF,
        n_rotations=3,
        amplitude_diff_range=np.linspace(-0.004, 0.004, 11),
        n_iterations=1,
        calibrate_beta=True,
        beta_range=np.linspace(-0.5, 0.5, 11),
        n_turns=1,
        degree=3,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )


def test_the_beta_analyzer_prefers_a_calibrated_drag_half_pi(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a DRAG half-pi, when the beta pass runs, then it analyzes with that."""
    exp = _drag_stub(stub_experiment_factory)
    target = f"{QUBIT}-hi"
    analyzers = _capture_analyzers(monkeypatch)

    _run_drag_with_beta(exp, beta=0.2, calibrate_beta=True)

    assert analyzers
    # The uncorrected half-pi is the pulse with the largest possible beta error,
    # and the analyzer's own axis error biases the root the fit reports.
    expected = exp.pulse.get_drag_hpi_pulse(target).shifted(-np.pi / 2)
    np.testing.assert_allclose(analyzers[-1].values, expected.values)


def test_the_beta_analyzer_falls_back_to_the_uncorrected_half_pi(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no DRAG half-pi, when the beta pass runs, then the plain one analyzes."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=0.3012,
        pulse_type="pi",
    )
    target = f"{QUBIT}-hi"
    assert exp.calib_note.get_drag_hpi_param(target) is None
    analyzers = _capture_analyzers(monkeypatch)

    _run_drag_pi_beta_pass(exp)

    assert analyzers
    expected = exp.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)
    np.testing.assert_allclose(analyzers[-1].values, expected.values)


def test_a_drag_half_pi_alone_is_enough_for_the_beta_pass(
    stub_experiment_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given only a DRAG half-pi, when the pi beta pass runs, then it is not refused."""
    exp = stub_experiment_factory(
        transition="hi",
        envelope="drag",
        duration=DRAG_PI_DURATION,
        true_amplitude=0.3012,
        pulse_type="pi",
        with_hpi=False,
    )
    target = f"{QUBIT}-hi"
    exp.calib_note.update_drag_hpi_param(
        target,
        {
            "target": target,
            "duration": DRAG_HPI_DURATION,
            "amplitude": 0.30,
            "beta": 0.1,
        },
    )
    assert exp.calib_note.get_hpi_param(target) is None
    analyzers = _capture_analyzers(monkeypatch)

    _run_drag_pi_beta_pass(exp)

    # It used to refuse here and tell the caller to run the uncorrected
    # calibration first, even with a perfectly good DRAG half-pi in the note.
    assert analyzers
    expected = exp.pulse.get_drag_hpi_pulse(target).shifted(-np.pi / 2)
    np.testing.assert_allclose(analyzers[-1].values, expected.values)


def _amplitude_sweep_stub(
    stub_experiment_factory: Callable[..., Any],
    *,
    transition: str = "fh",
    true_amplitude: float = 0.4,
    **kwargs: Any,
) -> Any:
    """Return a stub whose transition has a known Rabi rate per unit amplitude."""
    return stub_experiment_factory(
        transition=transition,
        envelope="flat_top",
        duration=DURATION,
        ramptime=RAMPTIME,
        true_amplitude=true_amplitude,
        pulse_type="pi",
        **kwargs,
    )


def test_amplitude_sweep_recovers_the_simulated_rate_per_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a simulated sweep, when run, then the rate per amplitude is recovered."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    target = f"{QUBIT}-fh"
    # The stub rotates by 2*pi*rate*area, so its rate *is* the slope dOmega/dA.
    expected_rate = exp.rabi_rate_per_amplitude

    result = obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        amplitude_range=np.linspace(0.0, 0.4, 51),
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot=False,
        plot_pulse_schedule=False,
    )

    payload = result.data[target]
    assert "error" not in payload
    assert payload["rabi_rate_per_amplitude"] == pytest.approx(expected_rate, rel=1e-4)
    # Stored under the transition label, at the configured control amplitude.
    control_amplitude = payload["control_amplitude"]
    assert control_amplitude == pytest.approx(CONTROL_AMPLITUDE / np.sqrt(3))
    assert exp.ctx.stored_rabi_params[target].frequency == pytest.approx(
        expected_rate * control_amplitude, rel=1e-4
    )


def test_amplitude_sweep_holds_the_drive_fixed_and_scales_only_its_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a swept amplitude, when run, then only the drive amplitude changes."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    amplitudes = np.linspace(0.0, 0.4, 11)

    run_result = run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        amplitude_range=amplitudes,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    call = exp.sweep_calls[-1]
    assert len(call["schedules"]) == amplitudes.size
    # A fixed duration is the whole point: every point must cost the same time.
    lengths = set()
    for amplitude, schedule in zip(amplitudes, call["schedules"], strict=True):
        assert set(schedule.labels) == {QUBIT, f"{QUBIT}-ef", f"{QUBIT}-fh"}
        sequences = schedule.get_sampled_sequences()
        lengths.add(len(sequences[f"{QUBIT}-fh"]))
        peak = float(np.abs(sequences[f"{QUBIT}-fh"]).max())
        assert peak == pytest.approx(amplitude, abs=1e-12)
    assert len(lengths) == 1
    assert (
        run_result["played_duration"]
        == run_result["duration"] + 2 * (run_result["ramptime"])
    )
    assert run_result["effective_duration"] == pytest.approx(
        run_result["duration"] + run_result["ramptime"]
    )


def test_amplitude_sweep_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when run, then the payload records the defaults."""
    module = _module("obtain_rabi_params_by_amplitude_sweep_based_on_ground")
    # On ge the nominal prior and the stub agree exactly, so the default range
    # covers the two turns it is designed for.
    exp = _amplitude_sweep_stub(
        stub_experiment_factory,
        transition="ge",
        calibrated_qubits=[],
    )

    run_result = run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="ge",
        plot_pulse_schedule=False,
    )

    assert run_result["duration"] == module.DEFAULT_DURATION
    assert run_result["ramptime"] == _core.DEFAULT_RABI_RAMPTIME
    assert run_result["n_shots"] == _core.DEFAULT_N_SHOTS
    assert run_result["shot_interval"] == _core.DEFAULT_SHOT_INTERVAL
    assert run_result["effective_duration"] == pytest.approx(
        1 / constants.DEFAULT_RABI_FREQUENCY
    )
    # No stored Rabi parameters, so the range falls back to the nominal rate.
    assert run_result["amplitude_range_source"] == "nominal_rabi_rate"
    assert run_result["amplitude_range_capped"] is False
    amplitudes = run_result["amplitude_range"]
    assert amplitudes.size == module.DEFAULT_N_POINTS
    assert amplitudes[0] == 0.0
    # At the default duration the nominal range is n_turns * A_cfg.
    assert amplitudes[-1] == pytest.approx(module.DEFAULT_N_TURNS * CONTROL_AMPLITUDE)

    payload = analyze_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        run_result=run_result
    )
    assert payload["swept_turns"] == pytest.approx(module.DEFAULT_N_TURNS, rel=1e-3)


def test_amplitude_sweep_derives_its_range_from_a_stored_rabi_param(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given stored Rabi parameters, when run, then they size the default range."""
    module = _module("obtain_rabi_params_by_amplitude_sweep_based_on_ground")
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    stored = exp.ctx.stored_rabi_params[f"{QUBIT}-fh"]

    run_result = run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        plot_pulse_schedule=False,
    )

    assert run_result["amplitude_range_source"] == "calib_note"
    control_amplitude = run_result["control_amplitude"]
    rate_prior = stored.frequency / control_amplitude
    expected_max = module.DEFAULT_N_TURNS / (
        rate_prior * run_result["effective_duration"]
    )
    assert run_result["amplitude_range"][-1] == pytest.approx(expected_max)


def test_amplitude_sweep_caps_a_derived_range_at_the_amplitude_ceiling(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a very slow prior, when run, then the derived range stops at one."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    target = f"{QUBIT}-fh"
    # A tiny stored rate asks for far more amplitude than the hardware allows.
    exp.ctx.stored_rabi_params[target] = dataclasses.replace(
        exp.ctx.stored_rabi_params[target],
        frequency=1e-5,
    )

    run_result = run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        plot_pulse_schedule=False,
    )

    assert run_result["amplitude_range"][-1] == pytest.approx(1.0)
    assert run_result["amplitude_range_capped"] is True


@pytest.mark.parametrize(
    ("amplitude_range", "match"),
    [
        pytest.param(np.array([0.0, 0.1, 1.4]), "must lie in", id="above-ceiling"),
        pytest.param(np.array([0.0, 0.1]), "at least", id="too-few-points"),
        pytest.param(np.array([0.0, 0.2, 0.1, 0.3]), "increasing", id="not-monotonic"),
        pytest.param(
            np.array([0.0, 0.1, 0.3, 0.4, 0.5]), "evenly spaced", id="not-uniform"
        ),
    ],
)
def test_amplitude_sweep_rejects_an_unusable_amplitude_range(
    stub_experiment_factory: Callable[..., Any],
    amplitude_range: np.ndarray,
    match: str,
) -> None:
    """Given an unusable range, when run, then it raises before measuring."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match=match):
        run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            amplitude_range=amplitude_range,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


@pytest.mark.parametrize("duration", [0.0, -8.0, 61.0])
def test_amplitude_sweep_rejects_an_unusable_duration(
    stub_experiment_factory: Callable[..., Any],
    duration: float,
) -> None:
    """Given an unusable duration, when run, then it raises before measuring."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match="`duration`"):
        run_obtain_rabi_params_by_amplitude_sweep_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            duration=duration,
            plot_pulse_schedule=False,
        )
    assert exp.sweep_calls == []


def test_amplitude_sweep_does_not_store_below_the_fit_threshold(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a threshold the fit cannot clear, when run, then nothing is stored."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    target = f"{QUBIT}-fh"
    seeded = exp.ctx.stored_rabi_params[target]

    result = obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        amplitude_range=np.linspace(0.0, 0.4, 51),
        fit_threshold=1.5,
        plot=False,
        plot_pulse_schedule=False,
    )

    assert not np.isfinite(result.data[target]["rabi_param"].frequency)
    assert exp.ctx.stored_rabi_params[target] is seeded


def test_amplitude_sweep_measures_two_targets_one_after_another(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given two qubits, when run, then each is swept and stored on its own."""
    true_amplitudes = {QUBIT: 0.4, QUBIT_2: 0.5}
    exp = _amplitude_sweep_stub(
        stub_experiment_factory,
        qubits=[QUBIT, QUBIT_2],
        true_amplitudes=true_amplitudes,
    )

    result = obtain_rabi_params_by_amplitude_sweep_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2],
        transition="fh",
        amplitude_range=np.linspace(0.0, 0.4, 51),
        plot=False,
        plot_pulse_schedule=False,
    )

    targets = [f"{QUBIT}-fh", f"{QUBIT_2}-fh"]
    figures = dict(result.figures or {})
    assert list(result.data) == targets
    assert len(exp.sweep_calls) == 2
    for target in targets:
        qubit = target.split("-")[0]
        payload = result.data[target]
        assert "error" not in payload
        assert payload["rabi_rate_per_amplitude"] == pytest.approx(
            exp.rabi_rates[qubit], rel=1e-4
        )
        assert f"{target}:rabi" in figures
        assert f"{target}:residuals" in figures
        assert f"{target}:iq" in figures


def test_amplitude_chevron_run_visits_every_grid_point_inside_the_ladder(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a detuning x amplitude grid, when run, then each point keeps the ladder."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    detunings = np.linspace(-0.01, 0.01, 5)
    amplitudes = np.linspace(0.0, 0.4, 7)

    run_result = run_chevron_pattern_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        detuning_range=detunings,
        amplitude_range=amplitudes,
        frequency=7.5,
        n_shots=N_SHOTS,
        shot_interval=SHOT_INTERVAL,
        plot_pulse_schedule=False,
    )

    call = exp.sweep_calls[-1]
    # The carrier is set once; the detuning rides on the waveform.
    assert call["frequencies"] == {f"{QUBIT}-fh": 7.5}
    assert len(call["schedules"]) == detunings.size * amplitudes.size
    assert run_result["raw"].size == detunings.size * amplitudes.size
    lengths = set()
    for schedule in call["schedules"]:
        assert set(schedule.labels) == {QUBIT, f"{QUBIT}-ef", f"{QUBIT}-fh"}
        lengths.add(len(schedule.get_sampled_sequences()[f"{QUBIT}-fh"]))
    # A fixed duration is the point: every grid point costs the same time.
    assert len(lengths) == 1


def test_amplitude_chevron_run_is_detuning_major_and_scales_the_amplitude(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given the flattened grid, when run, then it reshapes back to (detuning, amplitude)."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    detunings = np.linspace(-0.01, 0.01, 5)
    amplitudes = np.linspace(0.1, 0.4, 4)

    run_obtain = run_chevron_pattern_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="fh",
        detuning_range=detunings,
        amplitude_range=amplitudes,
        frequency=7.5,
        plot_pulse_schedule=False,
    )

    schedules = exp.sweep_calls[-1]["schedules"]
    peaks = np.array(
        [
            float(np.abs(s.get_sampled_sequences()[f"{QUBIT}-fh"]).max())
            for s in schedules
        ]
    ).reshape(detunings.size, amplitudes.size)
    # Detuning-major flattening: every row must repeat the same amplitude ramp.
    for row in peaks:
        np.testing.assert_allclose(row, amplitudes, atol=1e-12)
    assert run_obtain["detuning_range"].size == detunings.size
    assert run_obtain["amplitude_range"].size == amplitudes.size


def test_amplitude_chevron_defaults_are_resolved_when_every_argument_is_omitted(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given only a transition, when run, then the payload records the defaults."""
    module = _module("chevron_pattern_by_amplitude_sweep_based_on_ground")
    exp = _amplitude_sweep_stub(
        stub_experiment_factory, transition="ge", calibrated_qubits=[]
    )

    run_result = run_chevron_pattern_by_amplitude_sweep_based_on_ground(
        exp,
        target=QUBIT,
        transition="ge",
        plot_pulse_schedule=False,
    )

    assert run_result["duration"] == module.DEFAULT_DURATION
    assert run_result["ramptime"] == _core.DEFAULT_RABI_RAMPTIME
    assert run_result["effective_duration"] == pytest.approx(
        1 / constants.DEFAULT_RABI_FREQUENCY
    )
    assert run_result["amplitude_range"].size == module.DEFAULT_N_AMPLITUDES
    assert run_result["detuning_range"].size == module.DEFAULT_N_DETUNINGS
    assert run_result["amplitude_range_source"] == "nominal_rabi_rate"
    # One nominal turn at the configured amplitude, so A_max is A_cfg itself.
    assert run_result["amplitude_range"][-1] == pytest.approx(
        module.DEFAULT_N_TURNS * CONTROL_AMPLITUDE
    )
    # Half span = factor * max(1/T_eff, kappa * A_max); both are 1/T_eff here.
    assert run_result["detuning_range"][-1] == pytest.approx(
        module.DEFAULT_DETUNING_SPAN_FACTOR / run_result["effective_duration"]
    )
    assert run_result["detuning_range"][0] == pytest.approx(
        -run_result["detuning_range"][-1]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"amplitude_range": np.array([0.0, 0.1, 1.4])},
            "must lie in",
            id="amplitude-above-ceiling",
        ),
        pytest.param(
            {"amplitude_range": np.array([0.0, 0.1, 0.3, 0.4, 0.5])},
            "evenly spaced",
            id="amplitude-not-uniform",
        ),
        pytest.param(
            {"detuning_range": np.array([-0.01, 0.0, 0.004, 0.01])},
            "evenly spaced",
            id="detuning-not-uniform",
        ),
        pytest.param(
            {"detuning_range": np.array([0.0, -0.01, 0.01, 0.02])},
            "increasing",
            id="detuning-not-monotonic",
        ),
        pytest.param({"duration": 61.0}, "`duration`", id="duration-off-grid"),
    ],
)
def test_amplitude_chevron_rejects_unusable_axes(
    stub_experiment_factory: Callable[..., Any],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Given an unusable axis, when run, then it raises before measuring."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)

    with pytest.raises(ValueError, match=match):
        run_chevron_pattern_by_amplitude_sweep_based_on_ground(
            exp,
            target=QUBIT,
            transition="fh",
            frequency=7.5,
            plot_pulse_schedule=False,
            **kwargs,
        )
    assert exp.sweep_calls == []


def test_amplitude_chevron_finds_the_carrier_on_a_symmetric_stub_response(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given a response even in the detuning, when run, then the carrier is found."""
    exp = _amplitude_sweep_stub(stub_experiment_factory)
    target = f"{QUBIT}-fh"

    result = chevron_pattern_by_amplitude_sweep_based_on_ground(
        exp,
        targets=QUBIT,
        transition="fh",
        detuning_range=np.linspace(-0.01, 0.01, 21),
        amplitude_range=np.linspace(0.0, 0.4, 9),
        frequencies=7.5,
        plot=False,
        plot_pulse_schedule=False,
    )

    # The stub integrates the real part of a detuned envelope, so its area is an
    # even function of the detuning. It is not a physical chevron, but it is
    # genuinely symmetric about the carrier, which is what the estimator claims to
    # find -- and it exercises the flattening, the reshape and the projection.
    payload = result.data[target]
    assert "error" not in payload
    assert payload["raw"].size == 21 * 9
    assert payload["raw_grid"].shape == (21, 9)
    assert payload["resonant_frequency"] == pytest.approx(7.5, abs=2e-4)
    assert payload["chevron_data"].shape == (9, 21)


def test_amplitude_chevron_measures_two_targets_one_after_another(
    stub_experiment_factory: Callable[..., Any],
) -> None:
    """Given two qubits, when run, then each gets its own grid and nothing is stored."""
    exp = _amplitude_sweep_stub(
        stub_experiment_factory,
        qubits=[QUBIT, QUBIT_2],
    )
    seeded = dict(exp.ctx.stored_rabi_params)

    result = chevron_pattern_by_amplitude_sweep_based_on_ground(
        exp,
        targets=[QUBIT, QUBIT_2],
        transition="fh",
        detuning_range=np.linspace(-0.01, 0.01, 5),
        amplitude_range=np.linspace(0.0, 0.4, 5),
        plot=False,
        plot_pulse_schedule=False,
    )

    targets = [f"{QUBIT}-fh", f"{QUBIT_2}-fh"]
    assert list(result.data) == targets
    assert len(exp.sweep_calls) == 2
    # The transition frequency belongs to the registered target, so nothing is saved.
    assert exp.ctx.stored_rabi_params == seeded
    assert exp.calib_note.rabi == {}
