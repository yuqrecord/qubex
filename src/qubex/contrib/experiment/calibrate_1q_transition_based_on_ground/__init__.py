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
- `ramsey_experiment_based_on_ground`: Ramsey fringe against a deliberate
  carrier detuning, for T2* and a sharper resonance. Built from the
  transition's DRAG half-pi, so it comes after that calibration.
- `calibrate_hpi_pulse_based_on_ground`: FlatTop half-pi amplitude.
- `calibrate_pi_pulse_based_on_ground`: FlatTop pi amplitude, using a fixed
  `+-pi/2` tipping pulse whose own error cancels exactly.
- `calibrate_drag_hpi_pulse_based_on_ground`: DRAG half-pi amplitude and beta.
- `calibrate_drag_pi_pulse_based_on_ground`: DRAG pi amplitude and beta.
- `measure_dispersive_shift_extended`: readout phase of every level against
  the readout frequency, and the frequency that separates the levels best.

The four calibrations locate the amplitude at a *zero crossing* rather than at
the extremum of a fitted cosine: the Bloch vector is driven by (n - 1/4) and by
(n + 1/4) full turns, which ideally give the same normalized signal, and the
amplitude where the difference of the two crosses zero is the calibrated one. The
crossing is sharp where an extremum is flat, the sensitivity grows linearly with
`n`, and any affine error in the normalization — including the loss of contrast
that imperfect ladder preparation causes — drops out of the difference.

All four build the two branches as a signed `+-pi/2` tipping pulse followed by
the *same* number of repetitions of the pulse under test, so the branches carry
the same pulse count and the same duration and everything that accumulates per
pulse — leakage, relaxation, transients — is matched between them. The half-pi
calibrations use the pulse under test as its own tipping pulse: inverting the
leading pulse realizes `(4n - 1) theta` exactly as dropping two would, without
the imbalance.

The four calibrations write their result to `exp.calib_note` under the
*transition* label, for example `"Q00-fh"`. `chevron_pattern_based_on_ground`
and `ramsey_experiment_based_on_ground` deliberately persist nothing, because
the transition frequency belongs to the registered `Target`, and
`measure_dispersive_shift_extended` persists nothing either, because the readout
frequency belongs to the configuration.
`calib_note` does not autosave; call `Experiment.save_calib_note` when done.

Each entry point follows the argument conventions of the core `Experiment` API.
`targets` accepts one label, several labels, or `None` for every qubit that has
the transition registered, and every other argument accepts `None` and resolves
its default internally from `_core` — which in turn takes the values the core
`Experiment` uses. Only `transition` has to be given, because guessing which
rung to calibrate would silently calibrate the wrong one. The requested targets
are all checked before the first acquisition and then measured one after
another, so a single unregistered target or uncalibrated rung cannot leave the
calibration note half written.

The result is keyed by transition label whatever the number of targets::

    result = calibrate_hpi_pulse_based_on_ground(exp, transition="fh")
    result.data["Q00-fh"]["amplitude"]
    result.figures["Q00-fh:branches"]

`measure_dispersive_shift_extended` is the one exception: no single transition
owns a readout scan, so it keys its result by qubit label instead. It also
takes `max_level` rather than `transition`, for the same reason `transition` is
required elsewhere - guessing how far up the ladder to go would silently
measure the wrong states.

A fit that does not converge must not cost the measurement that produced it, so
the analysis, the calibration-note write, the plotting and the display are each
guarded per target. Whatever the pipeline reached is kept, the exception is
printed and stored under `"error"`, and the remaining targets still run::

    if "error" in result.data["Q01-fh"]:
        raise result.data["Q01-fh"]["error"]   # the original traceback

A target that came through carries no `"error"` key at all. Failures of the
measurement itself are *not* caught - there is nothing to keep, and the run may
have left the hardware in a state worth looking at.

Bootstrap order for a new rung::

    exp.register_transition_target(qubit=..., transition=..., channel_number=...)
    # calibrate the pi pulse of every lower rung first
    chevron_pattern_based_on_ground(exp, transition="fh")     # -> resonance
    obtain_rabi_params_based_on_ground(exp, transition="fh")  # -> RabiParam
    calibrate_hpi_pulse_based_on_ground(exp, transition="fh")
    # then amplify: larger n_rotations over a narrower sweep
    calibrate_hpi_pulse_based_on_ground(exp, transition="fh", n_rotations=4)
    calibrate_pi_pulse_based_on_ground(exp, transition="fh")
    calibrate_drag_hpi_pulse_based_on_ground(exp, transition="fh")
    calibrate_drag_pi_pulse_based_on_ground(exp, transition="fh")
    ramsey_experiment_based_on_ground(exp, transition="fh")   # -> T2*, resonance
    exp.save_calib_note()

The DRAG calibrations default to `n_rotations = 4`, as the core
`CalibrationService` does, which assumes the amplitude is already close. Run the
FlatTop calibrations first, or pass a smaller `n_rotations`, when starting from
a coarse estimate.

The DRAG calibrations also take `type` to choose the envelope from `RampType`,
defaulting to `"Gaussian"`. It is stored beside the amplitude and beta, so
`Experiment.get_drag_hpi_pulse` rebuilds the calibrated pulse with it::

    calibrate_drag_hpi_pulse_based_on_ground(
        exp, transition="fh", type="Blackman")

A stored amplitude seeds the sweep only when it was measured with the same
envelope, since the rotation angle follows the area under it.

The DRAG calibrations also take `beta` as the starting value of the correction
coefficient, which otherwise comes from the stored calibration or, failing that,
from the level ladder. Passing it turns off the correction outright::

    calibrate_drag_hpi_pulse_based_on_ground(
        exp, transition="fh", beta=0, calibrate_beta=False)

At `beta = 0` the pulse has no quadrature component left, so this calibrates the
amplitude of the bare envelope. That is worth having above ge: driving
`k <-> k+1` has two leakage channels rather than one, at roughly `-alpha` and
`+alpha`, and a single first-derivative correction cannot null both - the beta
that cancels one doubles the other. The compromise the ladder estimate settles
on shrinks as `1 / (k + 1)`, so DRAG buys progressively less the higher the rung,
and `beta = 0` is the honest comparison point.
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
from .measure_dispersive_shift_extended import (
    analyze_measure_dispersive_shift_extended,
    apply_electrical_delay,
    apply_phase_shift,
    display_measure_dispersive_shift_extended,
    measure_dispersive_shift_extended,
    plot_measure_dispersive_shift_extended,
    run_measure_dispersive_shift_extended,
)
from .obtain_rabi_params_based_on_ground import (
    analyze_obtain_rabi_params_based_on_ground,
    display_obtain_rabi_params_based_on_ground,
    obtain_rabi_params_based_on_ground,
    plot_obtain_rabi_params_based_on_ground,
    run_obtain_rabi_params_based_on_ground,
)
from .ramsey_experiment_based_on_ground import (
    analyze_ramsey_experiment_based_on_ground,
    display_ramsey_experiment_based_on_ground,
    plot_ramsey_experiment_based_on_ground,
    ramsey_experiment_based_on_ground,
    run_ramsey_experiment_based_on_ground,
)

__all__ = [
    "analyze_calibrate_drag_hpi_pulse_based_on_ground",
    "analyze_calibrate_drag_pi_pulse_based_on_ground",
    "analyze_calibrate_hpi_pulse_based_on_ground",
    "analyze_calibrate_pi_pulse_based_on_ground",
    "analyze_chevron_pattern_based_on_ground",
    "analyze_measure_dispersive_shift_extended",
    "analyze_obtain_rabi_params_based_on_ground",
    "analyze_ramsey_experiment_based_on_ground",
    "apply_electrical_delay",
    "apply_phase_shift",
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
    "display_measure_dispersive_shift_extended",
    "display_obtain_rabi_params_based_on_ground",
    "display_ramsey_experiment_based_on_ground",
    "measure_dispersive_shift_extended",
    "obtain_rabi_params_based_on_ground",
    "plot_calibrate_drag_hpi_pulse_based_on_ground",
    "plot_calibrate_drag_pi_pulse_based_on_ground",
    "plot_calibrate_hpi_pulse_based_on_ground",
    "plot_calibrate_pi_pulse_based_on_ground",
    "plot_chevron_pattern_based_on_ground",
    "plot_measure_dispersive_shift_extended",
    "plot_obtain_rabi_params_based_on_ground",
    "plot_ramsey_experiment_based_on_ground",
    "ramsey_experiment_based_on_ground",
    "run_calibrate_drag_hpi_pulse_based_on_ground",
    "run_calibrate_drag_pi_pulse_based_on_ground",
    "run_calibrate_hpi_pulse_based_on_ground",
    "run_calibrate_pi_pulse_based_on_ground",
    "run_chevron_pattern_based_on_ground",
    "run_measure_dispersive_shift_extended",
    "run_obtain_rabi_params_based_on_ground",
    "run_ramsey_experiment_based_on_ground",
]
