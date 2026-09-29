"""Tests for rebuilding stored half-pi and pi pulses in the right pulse family."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Drag, FlatTop

TARGET = "Q00-fh"

FLAT_TOP_PARAM = {
    "target": TARGET,
    "duration": 32.0,
    "amplitude": 0.30,
    "tau": 12.0,
}

DRAG_PARAM = {
    "target": TARGET,
    "duration": 16.0,
    "amplitude": 0.22,
    "tau": None,
    "beta": 0.0,
    "type": "Blackman",
}


def _service(param: dict[str, Any] | None) -> PulseService:
    """Return a PulseService whose note holds one half-pi and one pi entry."""
    ctx = SimpleNamespace(
        calib_note=SimpleNamespace(
            get_hpi_param=lambda target, valid_days=None: param,
            get_pi_param=lambda target, valid_days=None: param,
        ),
        calibration_valid_days=7,
        ge_targets=[TARGET],
        ef_targets=[TARGET],
    )
    return PulseService(experiment_context=cast(Any, ctx))


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_a_stored_flat_top_is_rebuilt_as_a_flat_top(pulse_type: str) -> None:
    """PulseService should rebuild an entry with a numeric `tau` as a FlatTop."""
    service = _service(FLAT_TOP_PARAM)

    pulse = getattr(service, f"get_{pulse_type}_pulse")(TARGET)

    assert isinstance(pulse, FlatTop)
    assert pulse.duration == 32.0
    assert pulse.amplitude == 0.30
    assert pulse.tau == 12.0


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_a_stored_envelope_is_rebuilt_as_a_drag(pulse_type: str) -> None:
    """PulseService should rebuild an entry with a null `tau` as a Drag."""
    service = _service(DRAG_PARAM)

    pulse = getattr(service, f"get_{pulse_type}_pulse")(TARGET)

    assert isinstance(pulse, Drag)
    assert pulse.duration == 16.0
    assert pulse.amplitude == 0.22
    assert pulse.beta == 0.0
    assert pulse.type == "Blackman"


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_an_envelope_without_a_type_falls_back_to_the_drag_default(
    pulse_type: str,
) -> None:
    """PulseService should leave `type` unset when the entry does not carry one."""
    param = {key: value for key, value in DRAG_PARAM.items() if key != "type"}
    service = _service(param)

    pulse = getattr(service, f"get_{pulse_type}_pulse")(TARGET)

    assert isinstance(pulse, Drag)
    assert pulse.type is None


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_an_envelope_without_a_beta_carries_no_correction(pulse_type: str) -> None:
    """PulseService should read a missing `beta` as an uncorrected envelope."""
    param = {key: value for key, value in DRAG_PARAM.items() if key != "beta"}
    service = _service(param)

    pulse = getattr(service, f"get_{pulse_type}_pulse")(TARGET)

    assert isinstance(pulse, Drag)
    assert pulse.beta == 0.0


@pytest.mark.parametrize(
    "prop", ["hpi_pulse", "pi_pulse", "ef_hpi_pulse", "ef_pi_pulse"]
)
def test_the_bulk_properties_agree_with_the_getters(prop: str) -> None:
    """PulseService should rebuild bulk entries in the same family as the getters."""
    assert isinstance(getattr(_service(FLAT_TOP_PARAM), prop)[TARGET], FlatTop)
    # A null `tau` must not be mistaken for a missing value and skipped.
    assert isinstance(getattr(_service(DRAG_PARAM), prop)[TARGET], Drag)


@pytest.mark.parametrize(
    "prop", ["hpi_pulse", "pi_pulse", "ef_hpi_pulse", "ef_pi_pulse"]
)
def test_the_bulk_properties_skip_an_entry_missing_a_required_value(
    prop: str,
) -> None:
    """PulseService should leave out an entry whose own family needs a missing value."""
    param = {**FLAT_TOP_PARAM, "amplitude": None}

    assert TARGET not in getattr(_service(param), prop)
