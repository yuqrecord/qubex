"""Tests for the export surface of the ground-referenced transition package."""

from __future__ import annotations

import qubex.contrib as contrib
import qubex.contrib.experiment as contrib_experiment
from qubex.contrib.experiment import calibrate_1q_transition_based_on_ground as package

ENTRY_POINTS = (
    "calibrate_drag_hpi_pulse_based_on_ground",
    "calibrate_drag_pi_pulse_based_on_ground",
    "calibrate_hpi_pulse_based_on_ground",
    "calibrate_pi_pulse_based_on_ground",
    "chevron_pattern_based_on_ground",
    "measure_dispersive_shift_extended",
    "obtain_rabi_params_based_on_ground",
)

STAGE_PREFIXES = ("run_", "analyze_", "plot_", "display_")


def test_entry_points_are_exported_from_contrib() -> None:
    """Given the contrib package, when imported, then the entry points exist."""
    for name in ENTRY_POINTS:
        assert callable(getattr(contrib, name))
        assert name in contrib.__all__


def test_entry_points_are_exported_from_experiment() -> None:
    """Given the experiment package, when imported, then it exposes the same objects."""
    for name in ENTRY_POINTS:
        assert getattr(contrib_experiment, name) is getattr(contrib, name)
        assert name in contrib_experiment.__all__


def test_all_stage_functions_are_exported_from_the_package() -> None:
    """Given the package, when imported, then every run/analyze/plot/display exists."""
    expected = {f"{prefix}{name}" for prefix in STAGE_PREFIXES for name in ENTRY_POINTS}
    assert expected | set(ENTRY_POINTS) == set(package.__all__)
    for name in package.__all__:
        assert callable(getattr(package, name))
