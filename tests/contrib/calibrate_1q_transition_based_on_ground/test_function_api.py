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
    "obtain_rabi_params_by_amplitude_sweep_based_on_ground",
)

STAGE_PREFIXES = ("run_", "analyze_", "plot_", "display_")

#: Public names that are not an entry point and not a stage of one. They are
#: declared here so the export surface stays pinned rather than open-ended.
EXTRA_EXPORTS = ("apply_electrical_delay",)


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
    assert expected | set(ENTRY_POINTS) | set(EXTRA_EXPORTS) == set(package.__all__)
    for name in package.__all__:
        assert callable(getattr(package, name))


def test_extra_exports_are_reachable_from_every_tier() -> None:
    """Given a non-entry-point export, when imported, then all three tiers agree."""
    for name in EXTRA_EXPORTS:
        assert getattr(contrib, name) is getattr(package, name)
        assert getattr(contrib_experiment, name) is getattr(package, name)
        assert name in contrib.__all__
        assert name in contrib_experiment.__all__
