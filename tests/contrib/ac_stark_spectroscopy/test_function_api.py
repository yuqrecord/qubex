"""Tests for functional APIs in `qubex.contrib.experiment.ac_stark_spectroscopy`."""

from __future__ import annotations

from qubex.contrib import (
    ac_stark_spectroscopy_arbitrary_readout_drive,
    ac_stark_spectroscopy_flattop_readout_drive,
)
from qubex.contrib.experiment.ac_stark_spectroscopy import (
    analyze_ac_stark_spectroscopy_arbitrary_readout_drive,
    analyze_ac_stark_spectroscopy_flattop_readout_drive,
    display_ac_stark_spectroscopy_arbitrary_readout_drive,
    display_ac_stark_spectroscopy_flattop_readout_drive,
    plot_ac_stark_spectroscopy_arbitrary_readout_drive,
    plot_ac_stark_spectroscopy_flattop_readout_drive,
    run_ac_stark_spectroscopy_arbitrary_readout_drive,
    run_ac_stark_spectroscopy_flattop_readout_drive,
)


def test_ac_stark_spectroscopy_entry_points_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then both variants are available."""
    assert callable(ac_stark_spectroscopy_flattop_readout_drive)
    assert callable(ac_stark_spectroscopy_arbitrary_readout_drive)


def test_ac_stark_spectroscopy_subfunctions_are_exported_from_the_package() -> None:
    """Given the subpackage, when imported, then run/analyze/plot/display exist."""
    assert callable(run_ac_stark_spectroscopy_flattop_readout_drive)
    assert callable(analyze_ac_stark_spectroscopy_flattop_readout_drive)
    assert callable(plot_ac_stark_spectroscopy_flattop_readout_drive)
    assert callable(display_ac_stark_spectroscopy_flattop_readout_drive)
    assert callable(run_ac_stark_spectroscopy_arbitrary_readout_drive)
    assert callable(analyze_ac_stark_spectroscopy_arbitrary_readout_drive)
    assert callable(plot_ac_stark_spectroscopy_arbitrary_readout_drive)
    assert callable(display_ac_stark_spectroscopy_arbitrary_readout_drive)
