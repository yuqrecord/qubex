"""
Ground-referenced characterization and calibration of any 1Q transition.

The core experiment layer stops at ef: `obtain_rabi_params` has no ladder state
preparation, `chevron_pattern` breaks on transition labels, `calibrate_hpi_pulse`
drives the target straight from |g>, and `sweep_parameter` can only normalize
against a ge or an ef Rabi arc.

Every experiment here is instead referenced to the ground state. The qubit is
climbed from |g> to the *lower* level of the target transition with a pi pulse on
each rung below it, the waveform under test is played on the transition, the same
pi pulses are replayed in reverse order, and only then is the qubit read out. For
`"fh"` that is::

    X_ge -> X_ef -> <waveform under test> -> X_ef -> X_ge -> readout

Ideal un-preparation maps `|k> -> |g>` and leaves `|k+1>` alone, so the readout
separates |g> from the transition's upper level and the IQ distance is measured
against the ground state at every rung of the ladder.

- `obtain_rabi_params_based_on_ground`: Rabi oscillation, referenced to a
  measured |g> point, stored under the transition label.
- `chevron_pattern_based_on_ground`: `detuning x duration` grid and the
  resonance it implies.
- `calibrate_hpi_pulse_based_on_ground`: FlatTop half-pi amplitude.
- `calibrate_pi_pulse_based_on_ground`: FlatTop pi amplitude, using a fixed
  `+-pi/2` tipping pulse whose own error cancels exactly.
- `calibrate_drag_hpi_pulse_based_on_ground`: DRAG half-pi amplitude and beta.
- `calibrate_drag_pi_pulse_based_on_ground`: DRAG pi amplitude and beta.

The four calibrations locate the amplitude at a *zero crossing* rather than at
the extremum of a fitted cosine: the Bloch vector is driven by (n - 1/4) and by
(n + 1/4) full turns, which ideally give the same normalized signal, and the
amplitude where the difference of the two crosses zero is the calibrated one. The
crossing is sharp where an extremum is flat, the sensitivity grows linearly with
`n`, and any affine error in the normalization — including the loss of contrast
that imperfect ladder preparation causes — drops out of the difference.

All of them write their result to `exp.calib_note` under the *transition* label,
for example `"Q00-fh"`. `chevron_pattern_based_on_ground` deliberately persists
nothing, because the transition frequency belongs to the registered `Target`.
`calib_note` does not autosave; call `Experiment.save_calib_note` when done.

Bootstrap order for a new rung::

    exp.register_transition_target(qubit=..., transition=..., channel_number=...)
    # calibrate the pi pulse of every lower rung first
    chevron_pattern_based_on_ground(...)     # -> resonance, re-register with it
    obtain_rabi_params_based_on_ground(...)  # -> transition RabiParam
    calibrate_hpi_pulse_based_on_ground(..., n_rotations=1, wide sweep)
    calibrate_hpi_pulse_based_on_ground(..., larger n_rotations, narrow sweep)
    calibrate_pi_pulse_based_on_ground(...)
    calibrate_drag_hpi_pulse_based_on_ground(...)
    calibrate_drag_pi_pulse_based_on_ground(...)
    exp.save_calib_note()
"""

from __future__ import annotations

from .calibrate_drag_hpi_pulse_based_on_ground import (
    analyze_calibrate_drag_hpi_pulse_based_on_ground,
    calibrate_drag_hpi_pulse_based_on_ground,
    display_calibrate_drag_hpi_pulse_based_on_ground,
    plot_calibrate_drag_hpi_pulse_based_on_ground,
    run_calibrate_drag_hpi_pulse_based_on_ground,
)
from .calibrate_drag_pi_pulse_based_on_ground import (
    analyze_calibrate_drag_pi_pulse_based_on_ground,
    calibrate_drag_pi_pulse_based_on_ground,
    display_calibrate_drag_pi_pulse_based_on_ground,
    plot_calibrate_drag_pi_pulse_based_on_ground,
    run_calibrate_drag_pi_pulse_based_on_ground,
)
from .calibrate_hpi_pulse_based_on_ground import (
    analyze_calibrate_hpi_pulse_based_on_ground,
    calibrate_hpi_pulse_based_on_ground,
    display_calibrate_hpi_pulse_based_on_ground,
    plot_calibrate_hpi_pulse_based_on_ground,
    run_calibrate_hpi_pulse_based_on_ground,
)
from .calibrate_pi_pulse_based_on_ground import (
    analyze_calibrate_pi_pulse_based_on_ground,
    calibrate_pi_pulse_based_on_ground,
    display_calibrate_pi_pulse_based_on_ground,
    plot_calibrate_pi_pulse_based_on_ground,
    run_calibrate_pi_pulse_based_on_ground,
)
from .chevron_pattern_based_on_ground import (
    analyze_chevron_pattern_based_on_ground,
    chevron_pattern_based_on_ground,
    display_chevron_pattern_based_on_ground,
    plot_chevron_pattern_based_on_ground,
    run_chevron_pattern_based_on_ground,
)
from .obtain_rabi_params_based_on_ground import (
    analyze_obtain_rabi_params_based_on_ground,
    display_obtain_rabi_params_based_on_ground,
    obtain_rabi_params_based_on_ground,
    plot_obtain_rabi_params_based_on_ground,
    run_obtain_rabi_params_based_on_ground,
)

__all__ = [
    "analyze_calibrate_drag_hpi_pulse_based_on_ground",
    "analyze_calibrate_drag_pi_pulse_based_on_ground",
    "analyze_calibrate_hpi_pulse_based_on_ground",
    "analyze_calibrate_pi_pulse_based_on_ground",
    "analyze_chevron_pattern_based_on_ground",
    "analyze_obtain_rabi_params_based_on_ground",
    "calibrate_drag_hpi_pulse_based_on_ground",
    "calibrate_drag_pi_pulse_based_on_ground",
    "calibrate_hpi_pulse_based_on_ground",
    "calibrate_pi_pulse_based_on_ground",
    "chevron_pattern_based_on_ground",
    "display_calibrate_drag_hpi_pulse_based_on_ground",
    "display_calibrate_drag_pi_pulse_based_on_ground",
    "display_calibrate_hpi_pulse_based_on_ground",
    "display_calibrate_pi_pulse_based_on_ground",
    "display_chevron_pattern_based_on_ground",
    "display_obtain_rabi_params_based_on_ground",
    "obtain_rabi_params_based_on_ground",
    "plot_calibrate_drag_hpi_pulse_based_on_ground",
    "plot_calibrate_drag_pi_pulse_based_on_ground",
    "plot_calibrate_hpi_pulse_based_on_ground",
    "plot_calibrate_pi_pulse_based_on_ground",
    "plot_chevron_pattern_based_on_ground",
    "plot_obtain_rabi_params_based_on_ground",
    "run_calibrate_drag_hpi_pulse_based_on_ground",
    "run_calibrate_drag_pi_pulse_based_on_ground",
    "run_calibrate_hpi_pulse_based_on_ground",
    "run_calibrate_pi_pulse_based_on_ground",
    "run_chevron_pattern_based_on_ground",
    "run_obtain_rabi_params_based_on_ground",
]
