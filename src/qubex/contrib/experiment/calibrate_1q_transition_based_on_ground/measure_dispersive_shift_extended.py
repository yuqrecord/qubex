"""
Readout-frequency scan of the level ladder, referenced to the ground state.

`Experiment.measure_dispersive_shift` stops at |e>. It runs
`measure_reflection_coefficient` twice, once per state, and that function hands
the state preparation to `PulseService.get_pulse_for_state`, which accepts
`"0"`, `"1"` and the four cardinal superpositions and raises for anything else.
It also drives a single channel, `{qubit: waveform}`, so there is nowhere to put
the ef or fh pulse of a ladder. Neither limit can be worked around from outside,
which is why this experiment builds its own acquisition.

The question here is not "what is chi" but "which readout frequency separates
these levels best", and that is a different optimum for every number of levels
one wants to tell apart. So the dispersive shift is never computed: what is
measured is the readout phase of each level against the readout frequency, and
what is reported is the frequency that separates them.

Method
------
Level `k` is reached by playing a pi pulse on each rung below it and stopping
there. The qubit must still be in |k> when the readout starts, so nothing is
played back down::

    |g> :                                        readout
    |e> : X_ge                                -> readout
    |f> : X_ge -> X_ef                        -> readout
    |h> : X_ge -> X_ef -> X_fh                -> readout

At every readout frequency all the levels are measured back to back before the
frequency moves on. The quantity of interest compares levels *at one frequency*,
so interleaving keeps slow drift common to the traces being compared, and it
retunes the readout once per group rather than once per level.

For every nested subset `S_m = {|g>, ..., |m>}` the separation at a frequency is
the smallest pairwise phase distance inside it::

    sep_m(f) = min over i < j in S_m of |arg(z_i(f) conj(z_j(f)))|

and the reported optimum is the frequency where that minimum is largest. Phase
is a circular quantity, so the distance is taken as the argument of
`z_i conj(z_j)` rather than as a difference of two `angle` calls, and it lands
in [0, pi]. The pair that realizes the minimum is reported too, because that
pair is what limits the readout.

The same max-min is reported on the Euclidean IQ distance `|z_i - z_j|`, which
weighs amplitude contrast as well as phase. That is a distance between the
*centroids* of the levels, not a discrimination fidelity: averaged data carries
no covariance, so nothing here is noise-aware.

`M` phasors on a circle can be spaced by at most `2 pi / M`, so the achievable
separation shrinks as levels are added. The ideal bound reported next to each
measured value is `2 pi / M`, not the two-level bound of `pi`.

Assumptions and caveats
-----------------------
- Every rung up to and including the top one must have a calibrated pi pulse.
  `PulseService.x180` silently falls back to two half-pi pulses otherwise, which
  would prepare a different state without saying so.
- The traces are averaged IQ, so each one is the response of the *prepared*
  ensemble rather than of a pure eigenstate: preparation infidelity and
  relaxation during the readout mix lower levels in. `arg(E[z])` is non-linear,
  so that mixture can pull the phase substantially wherever the contributions
  cancel. The optimum is therefore the best frequency for the states this setup
  can actually prepare. There is no four-state classifier to correct with -
  `Experiment.build_classifier` takes `n_states` of 2 or 3 - so nothing is
  corrected here.
- Phase is meaningless where the response vanishes, so a frequency at which any
  level drops below `MIN_AMPLITUDE_FRACTION` of the scan's median amplitude is
  dropped from the search. Such points are still plotted.
- The cable delay is a common factor `exp(-2 pi i f tau)` at a given frequency,
  so it cancels exactly from both metrics and is never measured here. The
  optional `electrical_delay` only straightens the frequency-wise trajectories
  for the eye. Drift *between* the per-level acquisitions does not cancel; run
  with `alternate_state_order=True` to expose it.
- The sweep has to stay inside the readout NCO window, 250 MHz around the
  channel's fine frequency. A wider scan needs the LO and CNCO retuning that
  `Experiment.scan_resonator_frequencies` does.
- No hardware is touched implicitly: the AWG and capture units are only reset
  when `reset_awg_and_capunits` asks for it, and the electrical delay is never
  measured. The core `measure_reflection_coefficient` does both unconditionally.
- Nothing is persisted. The readout frequency belongs to the configuration, so
  applying the result stays an explicit act of the caller.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import Blank, PulseSchedule, Waveform
from qubex.system.transition import LEVEL_NAMES, TRANSITION_NAMES

from . import _core

__all__ = [
    "analyze_measure_dispersive_shift_extended",
    "display_measure_dispersive_shift_extended",
    "measure_dispersive_shift_extended",
    "plot_measure_dispersive_shift_extended",
    "run_measure_dispersive_shift_extended",
]

#: Frequency step of the sweep in GHz, as `measure_reflection_coefficient` uses.
DEFAULT_DF = 0.0005

#: Full width of the sweep in GHz, as `measure_reflection_coefficient` uses.
DEFAULT_FREQUENCY_WIDTH = 0.05

#: Half-width of the readout tuning window in GHz. `Target.is_available` calls a
#: target unreachable once it sits further than this from the channel's fine
#: frequency, and the sweep moves the readout target rather than the NCO.
NCO_TUNING_HALF_WIDTH = 0.25

#: Fraction of the scan's median amplitude below which a level's phase is taken
#: to be undefined. The phase of a vanishing phasor is noise, and one such level
#: would drag the minimum of its whole subset to an arbitrary value.
MIN_AMPLITUDE_FRACTION = 0.1

#: Smallest phase separation in rad that still counts as a resolved optimum.
#: Below it the levels are coincident to within the noise and the argmax is an
#: arbitrary grid point rather than a readout frequency worth using.
MIN_RESOLVED_SEPARATION = 0.05

#: Relative tolerance within which two separations count as tied. Ties are
#: broken deterministically, towards the current readout frequency.
SEPARATION_TIE_RTOL = 1e-9


def _validate_max_level(max_level: int) -> None:
    """Raise when the requested top level is not a preparable rung."""
    if isinstance(max_level, bool) or not isinstance(max_level, int):
        raise TypeError(f"`max_level` must be an int, got {type(max_level).__name__}.")
    if not 1 <= max_level <= len(TRANSITION_NAMES):
        raise ValueError(
            f"`max_level` must lie in [1, {len(TRANSITION_NAMES)}], got {max_level}. "
            "It is the index of the highest level to prepare, so 1 measures |g> "
            "and |e>, and 3 measures |g>, |e>, |f> and |h>."
        )


def _resolve_ladder_labels(
    exp: Experiment,
    *,
    target: str,
    max_level: int,
) -> tuple[str, tuple[str, ...]]:
    """
    Resolve the qubit and the ordered transition labels that climb to a level.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the target registry.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    max_level : int
        Index of the highest level to prepare.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        Qubit label, and the transition labels from `"ge"` up to the rung that
        reaches `max_level`. `_core.resolve_ladder` raises for any of them that
        is not registered.
    """
    ladder = _core.resolve_ladder(
        exp,
        target=target,
        transition=TRANSITION_NAMES[max_level - 1],
    )
    return ladder.qubit, (*ladder.lower_labels, ladder.target)


def _validate_pi_pulses(exp: Experiment, *, labels: Sequence[str]) -> None:
    """
    Raise when a rung of the ladder has no pi-pulse calibration.

    `_core.validate_ladder_calibrations` checks only the rungs *below* the
    transition, because there the top rung is the one being calibrated. Here
    every rung is played, so every rung has to be calibrated.

    Parameters
    ----------
    exp : Experiment
        Experiment holding the calibration note.
    labels : Sequence[str]
        Transition labels of the ladder, in preparation order.
    """
    missing = [
        label
        for label in labels
        if exp.calib_note.get_drag_pi_param(label) is None
        and exp.calib_note.get_pi_param(label) is None
    ]
    if missing:
        raise ValueError(
            f"Pi pulses are not calibrated for {missing}, which this experiment "
            "needs to climb the ladder. `Experiment.pulse.x180` would silently "
            "fall back to two half-pi pulses and prepare a different state. "
            "Calibrate every rung first."
        )


def _resolve_frequency_range(
    *,
    frequency_range: ArrayLike | None,
    center_frequency: float,
    frequency_width: float,
    df: float,
) -> NDArray[np.float64]:
    """Return the swept readout frequencies, given or built around the center."""
    if frequency_range is not None:
        return _core.resolve_sweep_range(frequency_range)
    if not np.isfinite(frequency_width) or frequency_width <= 0:
        raise ValueError(
            f"`frequency_width` must be a positive number, got {frequency_width!r}."
        )
    if not np.isfinite(df) or df <= 0:
        raise ValueError(f"`df` must be a positive number, got {df!r}.")
    if not np.isfinite(center_frequency) or center_frequency <= 0:
        raise ValueError(
            f"`center_frequency` must be a positive number, got {center_frequency!r}."
        )
    return np.arange(
        center_frequency - frequency_width / 2,
        center_frequency + frequency_width / 2,
        df,
        dtype=np.float64,
    )


def _validate_frequency_range(frequency_range: NDArray[np.float64]) -> None:
    """Raise when the swept frequencies cannot be measured or compared."""
    _core.validate_sweep_range(frequency_range, "frequency_range")
    if np.any(frequency_range <= 0):
        raise ValueError("`frequency_range` must contain only positive frequencies.")
    if np.unique(frequency_range).size != frequency_range.size:
        raise ValueError(
            "`frequency_range` must not repeat a frequency. Each entry is one "
            "acquisition, and a duplicate would enter the optimum twice."
        )


def _validate_nco_window(
    exp: Experiment,
    *,
    read_label: str,
    frequency_range: NDArray[np.float64],
) -> None:
    """
    Raise when the sweep leaves the readout tuning window.

    The sweep moves the readout target frequency and leaves the NCO where it is,
    so a point further than `NCO_TUNING_HALF_WIDTH` from the channel's fine
    frequency is not reachable. Backends that cannot report a fine frequency are
    left alone: this is an early warning, and refusing to run because a channel
    is silent about its NCO would be worse than measuring.
    """
    try:
        nco_frequency = float(exp.ctx.targets[read_label].fine_frequency)
    except (AttributeError, TypeError):
        return
    if not np.isfinite(nco_frequency):
        return
    offset = float(np.max(np.abs(frequency_range - nco_frequency)))
    if offset >= NCO_TUNING_HALF_WIDTH:
        raise ValueError(
            f"The sweep reaches {offset * 1e3:.1f} MHz from the readout NCO of "
            f"{read_label!r}, outside the {NCO_TUNING_HALF_WIDTH * 1e3:.0f} MHz "
            "tuning window. Narrow `frequency_width`, or use "
            "`Experiment.scan_resonator_frequencies`, which retunes the LO and "
            "the CNCO per subrange."
        )


def _build_state_schedules(
    *,
    labels: Sequence[str],
    prep_pulses: Sequence[Waveform],
) -> list[PulseSchedule]:
    """
    Build one climb-only schedule per level, from |g> to the top of the ladder.

    Unlike `_core.build_ground_referenced_schedule` the ladder is not climbed
    back down: the qubit has to still be in the prepared level when the readout
    starts. The barriers keep consecutive rungs from overlapping, as they are
    played on different transition channels.

    Parameters
    ----------
    labels : Sequence[str]
        Transition labels of the ladder, in preparation order.
    prep_pulses : Sequence[Waveform]
        Pi pulse of each rung, in the same order.

    Returns
    -------
    list[PulseSchedule]
        Schedule `k` prepares level `k`, so the list is one longer than the
        ladder.
    """
    if len(prep_pulses) != len(labels):
        raise ValueError(
            f"`prep_pulses` has {len(prep_pulses)} entries but the ladder has "
            f"{len(labels)} rungs."
        )

    schedules: list[PulseSchedule] = []
    for level in range(len(labels) + 1):
        with PulseSchedule() as ps:
            if level == 0:
                ps.add(labels[0], Blank(0))
            for index in range(level):
                if index > 0:
                    ps.barrier()
                ps.add(labels[index], prep_pulses[index])
        schedules.append(ps)
    return schedules


def _extract_signal(result: Any, *, qubit: str) -> complex:
    """Return the averaged IQ point of one qubit, or say what came back instead."""
    data = result.data
    if qubit not in data:
        raise KeyError(
            f"The measurement returned no data for {qubit!r}; it carries "
            f"{sorted(data)}. The readout target may be registered under another "
            "label than the qubit."
        )
    value = np.asarray(data[qubit].kerneled)
    return complex(value if value.ndim == 0 else np.mean(value))


def _phase_distance(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """
    Return the circular distance between the phases of two signals.

    Taking the argument of `a conj(b)` rather than the difference of two
    `np.angle` calls keeps the quantity circular by construction: it needs no
    unwrapping, it cannot report two identical phasors as `2 pi` apart, and it
    lands in [0, pi] without a manual fold.
    """
    return np.abs(np.angle(np.asarray(a) * np.conj(np.asarray(b))))


def _select_optimum(
    *,
    separation: NDArray[np.float64],
    frequency_range: NDArray[np.float64],
    valid: NDArray[np.bool_],
    center_frequency: float,
    what: str,
) -> int:
    """
    Return the index of the best separation, breaking ties deterministically.

    Among the frequencies whose separation is within `SEPARATION_TIE_RTOL` of
    the best one, the frequency closest to the current readout frequency wins,
    and the lowest frequency breaks any remaining tie. A flat maximum therefore
    does not depend on the grid's direction.
    """
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        raise ValueError(
            f"No frequency of the scan is usable for the {what}: every point has "
            "a level whose response is non-finite or below "
            f"{MIN_AMPLITUDE_FRACTION:.3g} of the median amplitude. Raise the "
            "readout amplitude, or narrow the sweep to the resonance."
        )
    best = float(np.max(separation[candidates]))
    tied = candidates[separation[candidates] >= best - abs(best) * SEPARATION_TIE_RTOL]
    offsets = np.abs(frequency_range[tied] - center_frequency)
    closest = tied[offsets == offsets.min()]
    return int(closest[np.argmin(frequency_range[closest])])


def _resolve_per_qubit(
    *,
    qubits: Sequence[str],
    values: Mapping[str, float] | float | None,
    default: Callable[[str], float],
    name: str,
) -> dict[str, float]:
    """Resolve a per-qubit scalar that may be mapped, shared, or left to default."""
    resolved: dict[str, float] = {}
    for qubit in qubits:
        if values is None:
            value = default(qubit)
        elif isinstance(values, Mapping):
            if qubit not in values:
                raise ValueError(f"`{name}` has no entry for {qubit!r}.")
            value = values[qubit]
        else:
            value = values
        resolved[qubit] = float(value)
    return resolved


def _separation_figure(
    *,
    title: str,
    yaxis_title: str,
    frequency_range: NDArray[np.float64],
    subsets: Sequence[Mapping[str, Any]],
    separation_key: str,
    frequency_key: str,
    ideal_key: str | None,
) -> go.Figure:
    """Build a max-min separation figure, one trace per nested subset."""
    fig = viz.make_figure()
    for subset in subsets:
        fig.add_scatter(
            x=frequency_range,
            y=np.asarray(subset[separation_key], dtype=np.float64),
            mode="lines+markers",
            name=subset["name"],
        )
        fig.add_vline(x=subset[frequency_key], line_dash="dash", line_width=1)
        if ideal_key is not None:
            fig.add_hline(y=subset[ideal_key], line_dash="dot", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Readout frequency (GHz)",
        yaxis_title=yaxis_title,
        width=600,
        height=400,
    )
    return fig


def run_measure_dispersive_shift_extended(
    exp: Experiment,
    *,
    target: str,
    max_level: int,
    frequency_range: ArrayLike | None = None,
    center_frequency: float | None = None,
    frequency_width: float | None = None,
    df: float | None = None,
    readout_amplitude: float | None = None,
    electrical_delay: float | None = None,
    alternate_state_order: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> dict[str, Any]:
    """
    Scan the readout frequency with the qubit prepared in each level in turn.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    target : str
        Qubit label, or any target label that resolves to a qubit.
    max_level : int
        Index of the highest level to prepare, so `3` measures |g>, |e>, |f> and
        |h>. Every transition up to that rung must be registered and have a
        calibrated pi pulse.
    frequency_range : ArrayLike | None, optional
        Readout frequencies in GHz. Scalar or sequence. When given it takes
        precedence over `center_frequency`, `frequency_width` and `df`, and it
        is swept in the order it is given. Defaults to a grid built around the
        readout target's current frequency.
    center_frequency : float | None, optional
        Center of the built sweep in GHz. Defaults to the frequency the readout
        target is registered with. It is also the frequency that breaks ties
        between equally good optima.
    frequency_width : float | None, optional
        Full width of the built sweep in GHz. Defaults to
        `DEFAULT_FREQUENCY_WIDTH`.
    df : float | None, optional
        Step of the built sweep in GHz. Defaults to `DEFAULT_DF`.
    readout_amplitude : float | None, optional
        Readout amplitude. Defaults to the qubit's configured amplitude.
    electrical_delay : float | None, optional
        Cable delay in ns, used only to straighten the frequency-wise
        trajectories of the figures. It cancels from both metrics, and it is
        never measured here. Defaults to no correction.
    alternate_state_order : bool | None, optional
        Whether to reverse the order of the levels at every other frequency. It
        exposes a systematic drift that would otherwise sit on the levels in a
        fixed order. Defaults to False.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to `_core.DEFAULT_SHOT_INTERVAL`.
    reset_awg_and_capunits : bool | None, optional
        Whether to reset the AWG and capture units once before the sweep.
        Defaults to False, so the experiment touches no hardware that the
        measurement itself does not.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the preparation schedule of each level. Defaults to True.

    Returns
    -------
    dict[str, Any]
        Raw payload consumed by `analyze_measure_dispersive_shift_extended`.
    """
    if frequency_width is None:
        frequency_width = DEFAULT_FREQUENCY_WIDTH
    if df is None:
        df = DEFAULT_DF
    if electrical_delay is None:
        electrical_delay = 0.0
    if alternate_state_order is None:
        alternate_state_order = False
    if n_shots is None:
        n_shots = _core.DEFAULT_N_SHOTS
    if shot_interval is None:
        shot_interval = _core.DEFAULT_SHOT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = False
    if plot_pulse_schedule is None:
        plot_pulse_schedule = True

    _validate_max_level(max_level)
    qubit, labels = _resolve_ladder_labels(exp, target=target, max_level=max_level)
    _validate_pi_pulses(exp, labels=labels)

    read_label = exp.ctx.resolve_read_label(qubit)
    if center_frequency is None:
        center_frequency = float(exp.ctx.targets[read_label].frequency)
    if readout_amplitude is None:
        readout_amplitude = float(exp.ctx.params.get_readout_amplitude(qubit))

    frequencies = _resolve_frequency_range(
        frequency_range=frequency_range,
        center_frequency=center_frequency,
        frequency_width=frequency_width,
        df=df,
    )
    _validate_frequency_range(frequencies)
    _validate_nco_window(exp, read_label=read_label, frequency_range=frequencies)
    _core.validate_amplitude(readout_amplitude, "readout_amplitude")
    _core.validate_n_shots(n_shots)
    _core.validate_shot_interval(shot_interval)
    if not np.isfinite(electrical_delay):
        raise ValueError(
            f"`electrical_delay` must be a finite number, got {electrical_delay!r}."
        )

    levels = tuple(range(max_level + 1))
    state_labels = tuple(LEVEL_NAMES[level] for level in levels)
    prep_pulses = tuple(exp.pulse.x180(label) for label in labels)
    schedules = _build_state_schedules(labels=labels, prep_pulses=prep_pulses)

    if plot_pulse_schedule:
        _core.plot_pulse_schedules(
            {f"|{state_labels[level]}>": schedules[level] for level in levels[1:]},
            title=f"Ladder preparation : {qubit}",
        )

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[qubit])

    signals = np.full((len(levels), frequencies.size), np.nan, dtype=np.complex128)
    acquisition_order: list[tuple[int, ...]] = []
    for index, frequency in enumerate(
        tqdm(frequencies, desc=f"readout scan for {qubit}")
    ):
        order = levels[::-1] if alternate_state_order and index % 2 else levels
        acquisition_order.append(order)
        with exp.modified_frequencies({read_label: float(frequency)}):
            for level in order:
                result = exp.measure(
                    schedules[level],
                    mode="avg",
                    readout_amplitudes={qubit: readout_amplitude},
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    reset_awg_and_capunits=False,
                    plot=False,
                )
                signals[level, index] = _extract_signal(result, qubit=qubit)

    if electrical_delay:
        signals = signals * np.exp(2j * np.pi * frequencies * electrical_delay)

    return {
        "target": qubit,
        "qubit": qubit,
        "read_label": read_label,
        "max_level": max_level,
        "levels": levels,
        "state_labels": state_labels,
        "labels": labels,
        "frequency_range": frequencies,
        "center_frequency": center_frequency,
        "readout_amplitude": readout_amplitude,
        "electrical_delay": electrical_delay,
        "alternate_state_order": alternate_state_order,
        "acquisition_order": tuple(acquisition_order),
        "signals": signals,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_measure_dispersive_shift_extended(
    *,
    run_result: dict[str, Any],
    min_amplitude_fraction: float | None = None,
    min_resolved_separation: float | None = None,
) -> dict[str, Any]:
    """
    Locate the readout frequency that best separates each nested set of levels.

    Parameters
    ----------
    run_result : dict[str, Any]
        Payload returned by `run_measure_dispersive_shift_extended`.
    min_amplitude_fraction : float | None, optional
        Fraction of the scan's median amplitude below which a level's phase is
        treated as undefined, dropping that frequency from the search. Use `0.0`
        to keep every point. Defaults to `MIN_AMPLITUDE_FRACTION`.
    min_resolved_separation : float | None, optional
        Phase separation in rad below which a subset is reported as unresolved.
        Defaults to `MIN_RESOLVED_SEPARATION`.

    Returns
    -------
    dict[str, Any]
        The run payload plus the phases, the amplitudes, the usable-point mask
        and one entry per nested subset under `"subsets"`.
    """
    if min_amplitude_fraction is None:
        min_amplitude_fraction = MIN_AMPLITUDE_FRACTION
    if min_resolved_separation is None:
        min_resolved_separation = MIN_RESOLVED_SEPARATION

    signals = np.asarray(run_result["signals"], dtype=np.complex128)
    frequencies = np.asarray(run_result["frequency_range"], dtype=np.float64)
    center_frequency = float(run_result["center_frequency"])
    max_level = int(run_result["max_level"])
    state_labels = tuple(run_result["state_labels"])

    amplitudes = np.abs(signals)
    phases = np.angle(signals)
    # The ground state is the reference, so the common cable-delay slope drops
    # out of the plotted traces; unwrapping the relative phase directly is more
    # robust than unwrapping each level and subtracting afterwards.
    relative_phases = np.unwrap(np.angle(signals * np.conj(signals[0])), axis=1)

    finite = np.isfinite(signals)
    usable = amplitudes[finite]
    median_amplitude = float(np.median(usable)) if usable.size else float("nan")
    amplitude_floor = median_amplitude * min_amplitude_fraction
    valid = np.all(finite & (amplitudes >= amplitude_floor), axis=0)

    subsets: list[dict[str, Any]] = []
    for top in range(1, max_level + 1):
        levels = tuple(range(top + 1))
        pairs = tuple(itertools.combinations(levels, 2))
        phase_pairs = np.asarray(
            [_phase_distance(signals[i], signals[j]) for i, j in pairs]
        )
        distance_pairs = np.asarray(
            [np.abs(signals[i] - signals[j]) for i, j in pairs], dtype=np.float64
        )
        phase_separation = phase_pairs.min(axis=0)
        distance_separation = distance_pairs.min(axis=0)
        name = "".join(state_labels[level] for level in levels)

        phase_index = _select_optimum(
            separation=phase_separation,
            frequency_range=frequencies,
            valid=valid,
            center_frequency=center_frequency,
            what=f"phase separation of {name!r}",
        )
        distance_index = _select_optimum(
            separation=distance_separation,
            frequency_range=frequencies,
            valid=valid,
            center_frequency=center_frequency,
            what=f"IQ distance of {name!r}",
        )
        optimal_phase_separation = float(phase_separation[phase_index])
        subsets.append(
            {
                "name": name,
                "levels": levels,
                "state_labels": tuple(state_labels[level] for level in levels),
                "pairs": pairs,
                "phase_separation": phase_separation,
                "optimal_index": phase_index,
                "optimal_frequency": float(frequencies[phase_index]),
                "optimal_phase_separation": optimal_phase_separation,
                "ideal_separation": 2 * np.pi / len(levels),
                "limiting_pair": pairs[int(np.argmin(phase_pairs[:, phase_index]))],
                "resolved": optimal_phase_separation >= min_resolved_separation,
                "distance_separation": distance_separation,
                "optimal_index_by_distance": distance_index,
                "optimal_frequency_by_distance": float(frequencies[distance_index]),
                "optimal_distance_separation": float(
                    distance_separation[distance_index]
                ),
                "limiting_pair_by_distance": pairs[
                    int(np.argmin(distance_pairs[:, distance_index]))
                ],
            }
        )

    return {
        **run_result,
        "phases": phases,
        "relative_phases": relative_phases,
        "amplitudes": amplitudes,
        "median_amplitude": median_amplitude,
        "amplitude_floor": amplitude_floor,
        "valid_mask": valid,
        "min_amplitude_fraction": min_amplitude_fraction,
        "min_resolved_separation": min_resolved_separation,
        "subsets": subsets,
    }


def plot_measure_dispersive_shift_extended(
    data: dict[str, Any],
) -> dict[str, go.Figure]:
    """
    Build the figures of the readout-frequency scan.

    Parameters
    ----------
    data : dict[str, Any]
        Payload returned by `analyze_measure_dispersive_shift_extended`.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by name. They are built but not shown.
    """
    qubit = data["qubit"]
    frequencies = np.asarray(data["frequency_range"], dtype=np.float64)
    state_labels = tuple(data["state_labels"])
    signals = np.asarray(data["signals"], dtype=np.complex128)
    relative_phases = np.asarray(data["relative_phases"], dtype=np.float64)
    amplitudes = np.asarray(data["amplitudes"], dtype=np.float64)
    subsets = data["subsets"]

    phase_figure = viz.make_figure()
    for level, label in enumerate(state_labels):
        phase_figure.add_scatter(
            x=frequencies,
            y=relative_phases[level],
            mode="lines+markers",
            name=f"|{label}>",
        )
    for subset in subsets:
        phase_figure.add_vline(
            x=subset["optimal_frequency"],
            line_dash="dash",
            line_width=1,
            annotation_text=subset["name"],
            annotation_position="top",
        )
    phase_figure.update_layout(
        title=f"Readout phase of the level ladder : {qubit}",
        xaxis_title="Readout frequency (GHz)",
        yaxis_title="Phase relative to |g> (rad)",
        width=600,
        height=400,
    )

    amplitude_figure = viz.make_figure()
    for level, label in enumerate(state_labels):
        amplitude_figure.add_scatter(
            x=frequencies,
            y=amplitudes[level],
            mode="lines+markers",
            name=f"|{label}>",
        )
    amplitude_figure.update_layout(
        title=f"Readout amplitude of the level ladder : {qubit}",
        xaxis_title="Readout frequency (GHz)",
        yaxis_title="|IQ| (arb. units)",
        width=600,
        height=400,
    )

    return {
        "phase": phase_figure,
        "phase_separation": _separation_figure(
            title=f"Minimum pairwise phase separation : {qubit}",
            yaxis_title="min |dphi| (rad)",
            frequency_range=frequencies,
            subsets=subsets,
            separation_key="phase_separation",
            frequency_key="optimal_frequency",
            ideal_key="ideal_separation",
        ),
        "distance_separation": _separation_figure(
            title=f"Minimum pairwise centroid IQ distance : {qubit}",
            yaxis_title="min |dz| (arb. units)",
            frequency_range=frequencies,
            subsets=subsets,
            separation_key="distance_separation",
            frequency_key="optimal_frequency_by_distance",
            ideal_key=None,
        ),
        "amplitude": amplitude_figure,
        "iq": viz.make_iq_scatter_figure(
            {f"|{label}>": signals[level] for level, label in enumerate(state_labels)},
            mode="lines+markers",
            title=f"Readout response : {qubit}",
            xlabel="I (arb. units)",
            ylabel="Q (arb. units)",
        ),
    }


def display_measure_dispersive_shift_extended(
    result: Result,
    plot: bool = True,
) -> None:
    """
    Print the optimal readout frequencies and optionally show the figures.

    Parameters
    ----------
    result : Result
        Result assembled by `measure_dispersive_shift_extended`.
    plot : bool, optional
        Whether to show the figures.
    """
    data = dict(result.data)
    if plot and result.figures:
        _core.show_figures(
            dict(result.figures),
            prefix=f"measure_dispersive_shift_extended_{data['qubit']}",
        )

    state_labels = tuple(data["state_labels"])
    frequencies = np.asarray(data["frequency_range"], dtype=np.float64)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    print(f"Target: {data['target']}")
    print(f"  qubit               : {data['qubit']}")
    print(
        f"  max level           : {data['max_level']}  "
        f"({', '.join(f'|{label}>' for label in state_labels)})"
    )
    print(f"  ladder              : {' -> '.join(data['labels'])}")
    print(f"  readout target      : {data['read_label']}")
    print(
        f"  sweep               : {frequencies[0]:.6f} .. {frequencies[-1]:.6f} GHz, "
        f"{frequencies.size} points"
    )
    print(f"  readout amplitude   : {data['readout_amplitude']:.9g}")
    print(f"  shots               : {data['n_shots']}")
    print(f"  usable points       : {int(np.count_nonzero(valid))} / {valid.size}")

    print("  optimal readout frequency (max-min phase separation)")
    for subset in data["subsets"]:
        separation = subset["optimal_phase_separation"]
        first, second = subset["limiting_pair"]
        print(
            f"    {subset['name']:<6}: {subset['optimal_frequency']:.6f} GHz  "
            f"min |dphi| = {separation:.4f} rad ({np.degrees(separation):6.1f} deg)  "
            f"{separation / subset['ideal_separation']:6.1%} of 2pi/"
            f"{len(subset['levels'])}  "
            f"limited by |{state_labels[first]}>-|{state_labels[second]}>"
        )
        if not subset["resolved"]:
            print(
                f"    {subset['name']:<6}: the levels stay within "
                f"{data['min_resolved_separation']:.3g} rad of each other "
                "everywhere; there is no resolved optimum for this subset."
            )

    print("  optimal readout frequency (max-min centroid IQ distance)")
    for subset in data["subsets"]:
        first, second = subset["limiting_pair_by_distance"]
        print(
            f"    {subset['name']:<6}: "
            f"{subset['optimal_frequency_by_distance']:.6f} GHz  "
            f"min |dz| = {subset['optimal_distance_separation']:.4g}  "
            f"limited by |{state_labels[first]}>-|{state_labels[second]}>"
        )
    print(
        "  The centroid distance carries no noise model, so it ranks contrast "
        "rather than discrimination fidelity."
    )
    print(
        "  Nothing was stored. The readout frequency belongs to the "
        "configuration; apply it explicitly."
    )


def measure_dispersive_shift_extended(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    max_level: int,
    frequency_range: ArrayLike | None = None,
    center_frequencies: Mapping[str, float] | float | None = None,
    frequency_width: float | None = None,
    df: float | None = None,
    readout_amplitudes: Mapping[str, float] | float | None = None,
    electrical_delay: float | None = None,
    alternate_state_order: bool | None = None,
    min_amplitude_fraction: float | None = None,
    min_resolved_separation: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
    plot_pulse_schedule: bool | None = None,
) -> Result:
    """
    Find the readout frequency that best separates the levels of the ladder.

    Every level from |g> to `max_level` is prepared in turn while the readout
    frequency is swept, and the frequency that maximizes the smallest pairwise
    phase separation is reported for every nested subset: for `max_level=3` that
    is one frequency for `"ge"`, one for `"gef"` and one for `"gefh"`.

    Every requested target is checked before the first acquisition, so an
    unregistered rung or an uncalibrated pi pulse stops the whole batch. The
    targets are then measured one after another.

    Parameters
    ----------
    exp : Experiment
        Experiment to measure with.
    targets : Collection[str] | str | None, optional
        Qubit labels, or any target labels that resolve to qubits. Defaults to
        every qubit that has the top transition of the ladder registered.
    max_level : int
        Index of the highest level to prepare, so `3` measures |g>, |e>, |f> and
        |h>.
    frequency_range : ArrayLike | None, optional
        Readout frequencies in GHz, shared by every target. When given it takes
        precedence over `center_frequencies`, `frequency_width` and `df`.
        Defaults to a grid built per target around its readout frequency.
    center_frequencies : Mapping[str, float] | float | None, optional
        Center of the built sweep in GHz per qubit, one center shared by every
        qubit, or `None` to use each readout target's registered frequency.
    frequency_width : float | None, optional
        Full width of the built sweep in GHz. Defaults to
        `DEFAULT_FREQUENCY_WIDTH`.
    df : float | None, optional
        Step of the built sweep in GHz. Defaults to `DEFAULT_DF`.
    readout_amplitudes : Mapping[str, float] | float | None, optional
        Readout amplitude per qubit, one amplitude shared by every qubit, or
        `None` to use each qubit's configured amplitude.
    electrical_delay : float | None, optional
        Cable delay in ns, used only to straighten the figures. Defaults to no
        correction, and it is never measured here.
    alternate_state_order : bool | None, optional
        Whether to reverse the order of the levels at every other frequency.
        Defaults to False.
    min_amplitude_fraction : float | None, optional
        Fraction of the median amplitude below which a level's phase is treated
        as undefined. Defaults to `MIN_AMPLITUDE_FRACTION`.
    min_resolved_separation : float | None, optional
        Phase separation in rad below which a subset is reported as unresolved.
        Defaults to `MIN_RESOLVED_SEPARATION`.
    n_shots : int | None, optional
        Shots per acquisition. Defaults to `_core.DEFAULT_N_SHOTS`.
    shot_interval : float | None, optional
        Interval between shots in ns. Defaults to `_core.DEFAULT_SHOT_INTERVAL`.
    reset_awg_and_capunits : bool | None, optional
        Whether to reset the AWG and capture units once per target before its
        sweep. Defaults to False.
    plot : bool | None, optional
        Whether to show the figures. Defaults to True.
    plot_pulse_schedule : bool | None, optional
        Whether to draw the preparation schedule of each level. Defaults to True.

    Returns
    -------
    Result
        Payload keyed by qubit label, and figures keyed by `"{qubit}:{name}"`.
        No single transition owns this experiment, which is why the key is the
        qubit rather than a transition label. The optimum of one target is at
        `result.data[qubit]["subsets"][m - 1]["optimal_frequency"]`, where `m`
        is the number of transitions the subset spans.
        Nothing is written to `exp.calib_note`.
        A target whose analysis, plotting or display failed still appears,
        carrying whatever the pipeline reached and the exception under
        `"error"`; a target that came through carries no `"error"` key at all.
        Measurement failures are not caught: they are raised.
    """
    if plot is None:
        plot = True

    _validate_max_level(max_level)
    qubits = _core.resolve_target_labels(
        exp,
        targets=targets,
        transition=TRANSITION_NAMES[max_level - 1],
    )
    for qubit in qubits:
        _, labels = _resolve_ladder_labels(exp, target=qubit, max_level=max_level)
        _validate_pi_pulses(exp, labels=labels)

    resolved_amplitudes = _resolve_per_qubit(
        qubits=qubits,
        values=readout_amplitudes,
        default=lambda qubit: float(exp.ctx.params.get_readout_amplitude(qubit)),
        name="readout_amplitudes",
    )
    resolved_centers = _resolve_per_qubit(
        qubits=qubits,
        values=center_frequencies,
        default=lambda qubit: float(
            exp.ctx.targets[exp.ctx.resolve_read_label(qubit)].frequency
        ),
        name="center_frequencies",
    )

    def _analyze(run_result: dict[str, Any]) -> dict[str, Any]:
        return analyze_measure_dispersive_shift_extended(
            run_result=run_result,
            min_amplitude_fraction=min_amplitude_fraction,
            min_resolved_separation=min_resolved_separation,
        )

    def _display(result: Result) -> None:
        display_measure_dispersive_shift_extended(result, plot)

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for qubit in qubits:
        run_result = run_measure_dispersive_shift_extended(
            exp,
            target=qubit,
            max_level=max_level,
            frequency_range=frequency_range,
            center_frequency=resolved_centers[qubit],
            frequency_width=frequency_width,
            df=df,
            readout_amplitude=resolved_amplitudes[qubit],
            electrical_delay=electrical_delay,
            alternate_state_order=alternate_state_order,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot_pulse_schedule=plot_pulse_schedule,
        )
        payload, target_figures = _core.run_analysis_stages(
            run_result=run_result,
            analyze=_analyze,
            # The readout frequency belongs to the configuration, so this
            # experiment deliberately persists nothing.
            store=None,
            plot=plot_measure_dispersive_shift_extended,
            display=_display,
        )
        data[payload["target"]] = payload
        figures.update(
            {f"{payload['target']}:{name}": fig for name, fig in target_figures.items()}
        )

    return Result(data=data, figures=figures)
