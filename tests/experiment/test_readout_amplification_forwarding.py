"""Tests for forwarding `readout_amplification` down to `MeasurementService.measure`."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.experiment.models.experiment_result import ExperimentResult
from qubex.experiment.services.characterization_service import (
    CharacterizationService,
)
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.experiment.services.pulse_service import PulseService

TARGET = "Q00"


def _sweep_service(captured: list[dict[str, Any]]) -> Any:
    """Return a `MeasurementService` whose `measure` records its keyword arguments."""

    @contextmanager
    def _modified_frequencies(_frequencies: dict[str, float] | None) -> Any:
        yield

    ctx = SimpleNamespace(
        targets={TARGET: SimpleNamespace(is_ge=True, is_ef=False)},
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        modified_frequencies=_modified_frequencies,
        get_rabi_param=lambda _target: None,
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = PulseService(cast(Any, ctx))

    def _measure(self: MeasurementService, _seq: object, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(data={TARGET: SimpleNamespace(kerneled=1.0 + 0.0j)})

    service.measure = MethodType(_measure, service)
    return service


def _sequence(_sweep_value: float) -> PulseSchedule:
    with PulseSchedule([TARGET]) as schedule:
        schedule.add(TARGET, Blank(8))
    return schedule


@pytest.mark.parametrize("readout_amplification", [True, False])
def test_sweep_parameter_forwards_readout_amplification_to_measure(
    readout_amplification: bool,
) -> None:
    """Given readout amplification, when sweeping, then every point is measured with it."""
    captured: list[dict[str, Any]] = []
    service = _sweep_service(captured)

    service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0, 1.0]),
        readout_amplification=readout_amplification,
        plot=False,
    )

    assert [kwargs["readout_amplification"] for kwargs in captured] == [
        readout_amplification,
        readout_amplification,
    ]


def test_sweep_parameter_leaves_omitted_readout_amplification_to_measure() -> None:
    """Given no readout amplification, when sweeping, then None reaches measure unresolved."""
    captured: list[dict[str, Any]] = []
    service = _sweep_service(captured)

    service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        plot=False,
    )

    assert captured[0]["readout_amplification"] is None


def _reference_service(captured: list[dict[str, Any]]) -> Any:
    """Return a `MeasurementService` whose `measure_state` records its keyword arguments."""
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(qubit_labels=[TARGET])

    def _measure_state(
        self: MeasurementService,
        _states: dict[str, str],
        **kwargs: Any,
    ) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(data={TARGET: SimpleNamespace(kerneled=1.0 + 0.0j)})

    service.measure_state = MethodType(_measure_state, service)
    return service


@pytest.mark.parametrize("readout_amplification", [True, False])
def test_obtain_reference_points_forwards_readout_amplification_to_measure_state(
    readout_amplification: bool,
) -> None:
    """Given readout amplification, when obtaining reference points, then the state measurement uses it."""
    captured: list[dict[str, Any]] = []
    service = _reference_service(captured)

    service.obtain_reference_points(
        [TARGET],
        readout_amplification=readout_amplification,
        store_reference_points=False,
    )

    assert captured[0]["readout_amplification"] is readout_amplification


def test_obtain_reference_points_leaves_omitted_readout_amplification_to_measure() -> (
    None
):
    """Given no readout amplification, when obtaining reference points, then None is passed on."""
    captured: list[dict[str, Any]] = []
    service = _reference_service(captured)

    service.obtain_reference_points([TARGET], store_reference_points=False)

    assert captured[0]["readout_amplification"] is None


def _characterization_service(captured: list[dict[str, Any]]) -> Any:
    """Return a `CharacterizationService` whose sweeps record their keyword arguments."""

    def _discretize_time_range(
        time_range: np.ndarray,
        *,
        sampling_period: float,
    ) -> np.ndarray:
        return np.asarray(time_range)

    def _sweep_parameter(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(data={})

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=[TARGET],
        measurement=SimpleNamespace(sampling_period=0.4),
        util=SimpleNamespace(
            discretize_time_range=_discretize_time_range,
            create_qubit_subgroups=lambda targets: [list(targets)],
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        sweep_parameter=_sweep_parameter
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        validate_rabi_params=lambda _targets: None
    )
    return service


@pytest.mark.parametrize("experiment", ["t1_experiment", "t2_experiment"])
@pytest.mark.parametrize("readout_amplification", [True, False, None])
def test_coherence_experiments_forward_readout_amplification_to_the_sweep(
    experiment: str,
    readout_amplification: bool | None,
) -> None:
    """Given readout amplification, when T1 or T2 runs, then the sweep receives it unchanged."""
    captured: list[dict[str, Any]] = []
    service = _characterization_service(captured)

    result = getattr(service, experiment)(
        targets=[TARGET],
        time_range=np.array([800.0, 1600.0]),
        readout_amplification=readout_amplification,
        plot=False,
    )

    assert isinstance(result, ExperimentResult)
    assert captured[0]["readout_amplification"] is readout_amplification


@pytest.mark.parametrize("experiment", ["t1_experiment", "t2_experiment"])
def test_coherence_experiments_leave_omitted_readout_amplification_to_measure(
    experiment: str,
) -> None:
    """Given no readout amplification, when T1 or T2 runs, then None is passed to the sweep."""
    captured: list[dict[str, Any]] = []
    service = _characterization_service(captured)

    getattr(service, experiment)(
        targets=[TARGET],
        time_range=np.array([800.0, 1600.0]),
        plot=False,
    )

    assert captured[0]["readout_amplification"] is None
