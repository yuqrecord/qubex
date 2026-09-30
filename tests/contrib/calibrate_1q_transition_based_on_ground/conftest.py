"""Duck-typed `Experiment` stub for the ground-referenced transition tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from qubex.experiment.models.rabi_param import RabiParam
from qubex.pulse import Drag, FlatTop, PulseSchedule, RampType
from qubex.system.transition import (
    TRANSITION_NAMES,
    TransitionName,
    transition_index,
)

QUBIT = "Q00"
SAMPLING_PERIOD = 2.0
ALPHA = -0.35
GE_FREQUENCY = 8.0

#: Control amplitude of every stub qubit, which the transition accessor scales.
CONTROL_AMPLITUDE = 0.2

# Deliberately non-trivial so that the normalization path is exercised rather
# than accidentally reducing to the identity.
RABI_ANGLE = 0.7
RABI_OFFSET = 0.1
RABI_AMPLITUDE = 0.8
RABI_DISTANCE = 0.3
RABI_FREQUENCY = 0.01

PREP_PI = FlatTop(duration=32, amplitude=0.25, tau=12)


def transition_label(transition: TransitionName, qubit: str = QUBIT) -> str:
    """Return the target label of a transition of a stub qubit."""
    return qubit if transition == "ge" else f"{qubit}-{transition}"


def lower_labels(transition: TransitionName, qubit: str = QUBIT) -> tuple[str, ...]:
    """Return the ordered lower-rung labels of a transition."""
    return tuple(
        transition_label(name, qubit)
        for name in TRANSITION_NAMES[: transition_index(transition)]
    )


def rabi_param(target: str) -> RabiParam:
    """Return the Rabi parameters the stub normalizes with."""
    return RabiParam(
        target=target,
        amplitude=RABI_AMPLITUDE,
        frequency=RABI_FREQUENCY,
        phase=0.0,
        offset=RABI_OFFSET,
        noise=0.0,
        angle=RABI_ANGLE,
        distance=RABI_DISTANCE,
        r2=1.0,
        reference_phase=0.0,
    )


def iq_from_z(z: float) -> complex:
    """Invert `RabiParam.normalize` so that the stub can emit a wanted z."""
    rotated = complex(RABI_DISTANCE, z * RABI_AMPLITUDE + RABI_OFFSET)
    return complex(rotated * np.exp(1j * RABI_ANGLE))


class _SweepPoint:
    """Minimal stand-in for `SweepData`."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _SweepResult:
    """Minimal stand-in for `ExperimentResult[SweepData]`."""

    def __init__(self, data: dict[str, _SweepPoint]) -> None:
        self.data = data


class _Target:
    """Minimal stand-in for a registered `Target`."""

    def __init__(self, label: str) -> None:
        self.label = label
        index = transition_index(label.split("-")[1]) if "-" in label else 0
        self.frequency = GE_FREQUENCY + index * ALPHA


class _Qubit:
    """Minimal stand-in for a qubit model with a linear level ladder."""

    def __init__(self, label: str, alpha: float) -> None:
        self.label = label
        self.alpha = alpha
        self.frequency = GE_FREQUENCY

    def get_control_frequency(self, transition: TransitionName) -> float:
        """Return `f_ge + k * alpha` for the transition `k -> k+1`."""
        return self.frequency + transition_index(transition) * self.alpha


class _ControlParameters:
    """Minimal stand-in for `ControlParameters`."""

    @staticmethod
    def get_transition_control_amplitude(
        qubit: str,
        transition: TransitionName,
    ) -> float:
        """Return the control amplitude scaled by the transition matrix element."""
        return CONTROL_AMPLITUDE / ((transition_index(transition) + 1) ** 0.5)


class _Util:
    """Minimal stand-in for `ExperimentUtil`."""

    @staticmethod
    def resolve_sampling_period(sampling_period: float) -> float:
        """Return the sampling period unchanged."""
        return sampling_period


class _Measurement:
    """Minimal stand-in for `Measurement`."""

    sampling_period = SAMPLING_PERIOD


class _CalibNote:
    """Records and serves the calibration parameters of the stub."""

    def __init__(self) -> None:
        self.hpi: dict[str, Any] = {}
        self.pi: dict[str, Any] = {}
        self.drag_hpi: dict[str, Any] = {}
        self.drag_pi: dict[str, Any] = {}
        self.rabi: dict[str, Any] = {}

    def get_hpi_param(self, target: str) -> dict[str, Any] | None:
        """Return a stored FlatTop half-pi parameter set."""
        return self.hpi.get(target)

    def get_pi_param(self, target: str) -> dict[str, Any] | None:
        """Return a stored FlatTop pi parameter set."""
        return self.pi.get(target)

    def get_drag_hpi_param(self, target: str) -> dict[str, Any] | None:
        """Return a stored DRAG half-pi parameter set."""
        return self.drag_hpi.get(target)

    def get_drag_pi_param(self, target: str) -> dict[str, Any] | None:
        """Return a stored DRAG pi parameter set."""
        return self.drag_pi.get(target)

    def update_hpi_param(self, target: str, param: dict[str, Any]) -> None:
        """Record a FlatTop half-pi parameter set."""
        self.hpi[target] = param

    def update_pi_param(self, target: str, param: dict[str, Any]) -> None:
        """Record a FlatTop pi parameter set."""
        self.pi[target] = param

    def update_drag_hpi_param(self, target: str, param: dict[str, Any]) -> None:
        """Record a DRAG half-pi parameter set."""
        self.drag_hpi[target] = param

    def update_drag_pi_param(self, target: str, param: dict[str, Any]) -> None:
        """Record a DRAG pi parameter set."""
        self.drag_pi[target] = param

    def update_rabi_param(self, target: str, param: dict[str, Any]) -> None:
        """Record a Rabi parameter set."""
        self.rabi[target] = param


class _Context:
    """Minimal stand-in for `ExperimentContext`."""

    def __init__(
        self,
        *,
        qubit_labels: Sequence[str],
        labels: Sequence[str],
        calib_note: _CalibNote,
    ) -> None:
        self.qubit_labels = list(qubit_labels)
        self.qubits = {qubit: _Qubit(qubit, ALPHA) for qubit in self.qubit_labels}
        self.targets = {label: _Target(label) for label in labels}
        self.params = _ControlParameters()
        self.util = _Util()
        self.measurement = _Measurement()
        self.calib_note = calib_note
        self.stored_rabi_params: dict[str, RabiParam] = {}

    def resolve_qubit_label(self, label: str) -> str:
        """Return the qubit label of any target label."""
        return label.split("-")[0]

    def resolve_transition_label(
        self,
        label: str,
        transition: TransitionName,
    ) -> str:
        """Return the target label of a transition."""
        qubit = self.resolve_qubit_label(label)
        return qubit if transition == "ge" else f"{qubit}-{transition}"

    def get_rabi_param(self, target: str) -> RabiParam | None:
        """Return the Rabi parameters stored for a target."""
        return self.stored_rabi_params.get(target)

    def store_rabi_params(
        self,
        params: dict[str, RabiParam],
        r2_threshold: float | None = None,
    ) -> None:
        """Store Rabi parameters that clear the quality threshold."""
        threshold = 0.5 if r2_threshold is None else r2_threshold
        for label, param in params.items():
            if np.isfinite(param.r2) and param.r2 >= threshold:
                self.stored_rabi_params[label] = param
                self.calib_note.update_rabi_param(label, {"target": label})


def _rebuild_hpi_or_pi_pulse(param: dict[str, Any]) -> Any:
    """Rebuild a stored half-pi or pi pulse the way `PulseService` does."""
    tau = param.get("tau")
    if tau is None:
        return Drag(
            duration=param["duration"],
            amplitude=param["amplitude"],
            beta=param.get("beta") or 0.0,
            type=param.get("type"),
        )
    return FlatTop(
        duration=param["duration"],
        amplitude=param["amplitude"],
        tau=tau,
    )


class _Pulse:
    """Minimal stand-in for `PulseService`."""

    def __init__(self, calib_note: _CalibNote) -> None:
        self._calib_note = calib_note

    def x180(self, target: str) -> Any:
        """Return the pi pulse of a target, DRAG first."""
        param = self._calib_note.get_drag_pi_param(target)
        if param is not None:
            return Drag(
                duration=param["duration"],
                amplitude=param["amplitude"],
                beta=param["beta"],
                type=param.get("type"),
            )
        param = self._calib_note.get_pi_param(target)
        if param is not None:
            return _rebuild_hpi_or_pi_pulse(param)
        return PREP_PI.scaled(2)

    def get_hpi_pulse(self, target: str) -> Any:
        """Return the stored half-pi pulse, or a nominal one."""
        param = self._calib_note.get_hpi_param(target)
        if param is None:
            return FlatTop(duration=32, amplitude=0.3, tau=12)
        return _rebuild_hpi_or_pi_pulse(param)

    def get_drag_hpi_pulse(self, target: str) -> Any:
        """Return the stored DRAG half-pi pulse."""
        param = self._calib_note.get_drag_hpi_param(target)
        if param is None:
            raise KeyError(target)
        return Drag(
            duration=param["duration"],
            amplitude=param["amplitude"],
            beta=param["beta"],
            type=param.get("type"),
        )


class _MeasurementService:
    """Minimal stand-in for `MeasurementService`."""

    def __init__(self, experiment: StubExperiment) -> None:
        self._experiment = experiment

    def obtain_reference_points(
        self,
        targets: Sequence[str],
        *,
        n_shots: int,
        shot_interval: float,
    ) -> dict[str, dict[str, complex]]:
        """Return the |g> IQ point of every target."""
        self._experiment.reference_calls.append(
            {
                "targets": list(targets),
                "n_shots": n_shots,
                "shot_interval": shot_interval,
            }
        )
        return {"iq": dict.fromkeys(targets, iq_from_z(1.0))}

    def sweep_parameter(
        self,
        *,
        sequence: Callable[[float], PulseSchedule],
        sweep_range: Any,
        frequencies: dict[str, float] | None = None,
        n_shots: int,
        shot_interval: float,
        plot: bool,
    ) -> _SweepResult:
        """Simulate one sweep by integrating each emitted waveform."""
        experiment = self._experiment
        schedules = [sequence(point) for point in np.asarray(sweep_range)]
        experiment.sweep_calls.append(
            {
                "sweep_range": np.asarray(sweep_range),
                "frequencies": frequencies,
                "n_shots": n_shots,
                "shot_interval": shot_interval,
                "plot": plot,
                "schedules": schedules,
            }
        )
        simulated = [experiment.simulate(schedule) for schedule in schedules]
        driven = {qubit for qubit, _ in simulated}
        assert len(driven) == 1, "A sweep must drive exactly one qubit."
        qubit = driven.pop()
        values = [iq_from_z(z) for _, z in simulated]
        return _SweepResult(
            {qubit: _SweepPoint(np.asarray(values, dtype=np.complex128))}
        )


class StubExperiment:
    """
    An `Experiment` look-alike that simulates the waveforms it is handed.

    The simulation integrates the real part of the samples emitted on the
    *transition* channel and rotates the Bloch vector about x by an angle
    proportional to that area, which is exact for these sequences because every
    pulse on that channel shares the drive axis. The ladder preparation on the
    lower channels is treated as ideal, which is what the ground-referenced
    observable assumes. Because the simulation consumes the emitted samples, it
    exercises the repetition counts, the idle padding and the sign of the
    tipping pulse rather than re-deriving them.
    """

    def __init__(
        self,
        *,
        transition: TransitionName,
        rabi_rates: dict[str, float],
        registered_qubits: Sequence[str] | None = None,
    ) -> None:
        self.transition = transition
        self.rabi_rates = dict(rabi_rates)
        self.qubits = list(self.rabi_rates)
        registered = (
            self.qubits if registered_qubits is None else list(registered_qubits)
        )
        self.targets_by_qubit = {
            qubit: transition_label(transition, qubit) for qubit in self.qubits
        }
        self.lower_labels_by_qubit = {
            qubit: lower_labels(transition, qubit) for qubit in self.qubits
        }
        self.calib_note = _CalibNote()
        self.ctx = _Context(
            qubit_labels=self.qubits,
            labels=[
                label
                for qubit in registered
                for label in (
                    *self.lower_labels_by_qubit[qubit],
                    self.targets_by_qubit[qubit],
                )
            ],
            calib_note=self.calib_note,
        )
        self.pulse = _Pulse(self.calib_note)
        self.measurement_service = _MeasurementService(self)
        self.sweep_calls: list[dict[str, Any]] = []
        self.reference_calls: list[dict[str, Any]] = []

    @property
    def qubit(self) -> str:
        """Return the first stub qubit."""
        return self.qubits[0]

    @property
    def target(self) -> str:
        """Return the transition label of the first stub qubit."""
        return self.targets_by_qubit[self.qubit]

    @property
    def lower_labels(self) -> tuple[str, ...]:
        """Return the lower-rung labels of the first stub qubit."""
        return self.lower_labels_by_qubit[self.qubit]

    @property
    def rabi_rate_per_amplitude(self) -> float:
        """Return the Rabi rate per unit amplitude of the first stub qubit."""
        return self.rabi_rates[self.qubit]

    def simulate(self, schedule: PulseSchedule) -> tuple[str, float]:
        """Return the qubit the schedule drives and the signal it produces."""
        sequences = schedule.get_sampled_sequences()
        for qubit, target in self.targets_by_qubit.items():
            if target not in sequences:
                continue
            area = float(np.real(sequences[target]).sum()) * SAMPLING_PERIOD
            angle = 2 * np.pi * self.rabi_rates[qubit] * area
            return qubit, float(np.cos(angle))
        raise AssertionError("The schedule drives no known transition target.")

    def calc_control_amplitude(self, target: str, rabi_rate: float) -> float:
        """Invert the stub's linear amplitude-to-Rabi-rate relation."""
        for qubit, label in self.targets_by_qubit.items():
            if label == target:
                return rabi_rate / self.rabi_rates[qubit]
        raise AssertionError(f"Unknown target {target!r}.")


@pytest.fixture
def stub_experiment_factory() -> Callable[..., StubExperiment]:
    """
    Return a factory building a `StubExperiment` with a known true amplitude.

    The factory takes the transition, the pulse family under test, its shape and
    the amplitude that realizes the ideal rotation, derives the stub's linear
    amplitude-to-Rabi-rate relation from that, and seeds the calibrations every
    script requires: a pi pulse on each lower rung, the transition's Rabi
    parameters and — optionally — the transition's FlatTop half-pi.

    Passing `qubits` builds the same stub for several qubits at once. A qubit
    may then be given its own true amplitude through `true_amplitudes`, may be
    left out of `registered_qubits` so that its transition target is missing,
    and may be left out of `calibrated_qubits` so that its ladder and Rabi
    parameters are missing.
    """

    def factory(
        *,
        transition: TransitionName,
        envelope: str,
        duration: float,
        true_amplitude: float,
        pulse_type: str,
        ramptime: float = 12.0,
        beta: float = 0.0,
        type: RampType | None = None,
        with_hpi: bool = True,
        hpi_amplitude: float | None = None,
        qubits: Sequence[str] | None = None,
        true_amplitudes: dict[str, float] | None = None,
        registered_qubits: Sequence[str] | None = None,
        calibrated_qubits: Sequence[str] | None = None,
    ) -> StubExperiment:
        if envelope == "flat_top":
            unit = FlatTop(duration=duration, amplitude=1, tau=ramptime)
        else:
            unit = Drag(duration=duration, amplitude=1, beta=beta, type=type)
        unit_area = float(unit.real.sum()) * SAMPLING_PERIOD
        turns = 0.25 if pulse_type == "hpi" else 0.5

        labels = [QUBIT] if qubits is None else list(qubits)
        amplitudes = {
            qubit: (true_amplitudes or {}).get(qubit, true_amplitude)
            for qubit in labels
        }
        experiment = StubExperiment(
            transition=transition,
            rabi_rates={
                qubit: turns / (unit_area * amplitude)
                for qubit, amplitude in amplitudes.items()
            },
            registered_qubits=registered_qubits,
        )

        calibrated = labels if calibrated_qubits is None else list(calibrated_qubits)
        # Seed the half-pi in the same family as the pulse under test, so that
        # a calibration of that family can reuse it as its starting amplitude.
        if envelope == "flat_top":
            hpi_unit = FlatTop(duration=32, amplitude=1, tau=12)
            hpi_shape: dict[str, Any] = {"duration": 32, "tau": 12}
        else:
            hpi_unit = Drag(duration=duration, amplitude=1, beta=0.0, type=type)
            hpi_shape = {
                "duration": duration,
                "tau": None,
                "beta": 0.0,
                "type": type,
            }
        hpi_area = float(hpi_unit.real.sum()) * SAMPLING_PERIOD
        for qubit in calibrated:
            target = experiment.targets_by_qubit[qubit]
            for label in experiment.lower_labels_by_qubit[qubit]:
                experiment.calib_note.update_pi_param(
                    label,
                    {
                        "target": label,
                        "duration": PREP_PI.duration,
                        "amplitude": PREP_PI.amplitude,
                        "tau": PREP_PI.tau,
                    },
                )
            experiment.ctx.stored_rabi_params[target] = rabi_param(target)
            if with_hpi:
                experiment.calib_note.update_hpi_param(
                    target,
                    {
                        "target": target,
                        "amplitude": (
                            hpi_amplitude
                            if hpi_amplitude is not None
                            else 0.25 / (hpi_area * experiment.rabi_rates[qubit])
                        ),
                        **hpi_shape,
                    },
                )
        return experiment

    return factory


#: Resting readout frequency of the stub resonator, in GHz.
READOUT_FREQUENCY = 10.0

#: Fine (NCO) frequency of the stub readout channel, in GHz.
READOUT_NCO_FREQUENCY = 10.0

#: Linewidth of the stub resonator, in GHz.
RESONATOR_KAPPA = 0.002

#: Frequency the stub resonator moves by for each level of the qubit, in GHz.
DISPERSIVE_SHIFT = 0.0006

#: Configured readout amplitude of every stub qubit.
READOUT_AMPLITUDE = 0.05


class _ReadoutTarget:
    """Minimal stand-in for a `Target` that carries a tunable frequency."""

    def __init__(self, label: str, frequency: float, fine_frequency: float) -> None:
        self.label = label
        self.frequency = frequency
        self.fine_frequency = fine_frequency


class _ReadoutParameters:
    """Minimal stand-in for `ControlParameters` on the readout side."""

    @staticmethod
    def get_readout_amplitude(qubit: str) -> float:
        """Return the configured readout amplitude of a qubit."""
        return READOUT_AMPLITUDE


class _ReadoutContext:
    """Minimal stand-in for `ExperimentContext` for a readout-frequency scan."""

    def __init__(
        self,
        *,
        qubit_labels: Sequence[str],
        labels: Mapping[str, float],
    ) -> None:
        self.qubit_labels = list(qubit_labels)
        self.params = _ReadoutParameters()
        self.reset_calls: list[tuple[str, ...]] = []
        self.targets: dict[str, _ReadoutTarget] = {
            label: _ReadoutTarget(label, frequency, 0.0)
            for label, frequency in labels.items()
        }
        for qubit in self.qubit_labels:
            read_label = f"R{qubit}"
            self.targets[read_label] = _ReadoutTarget(
                read_label,
                READOUT_FREQUENCY,
                READOUT_NCO_FREQUENCY,
            )

    def resolve_qubit_label(self, label: str) -> str:
        """Return the qubit label of any target label."""
        return label.split("-")[0].removeprefix("R")

    def resolve_transition_label(
        self,
        label: str,
        transition: TransitionName,
    ) -> str:
        """Return the target label of a transition."""
        qubit = self.resolve_qubit_label(label)
        return qubit if transition == "ge" else f"{qubit}-{transition}"

    def resolve_read_label(self, label: str) -> str:
        """Return the readout target label of a qubit."""
        return f"R{self.resolve_qubit_label(label)}"

    def reset_awg_and_capunits(
        self,
        box_ids: Any = None,
        qubits: Sequence[str] | None = None,
    ) -> None:
        """Record a reset request instead of touching anything."""
        self.reset_calls.append(tuple(qubits or ()))


class _MeasureData:
    """Minimal stand-in for `MeasureData` in average mode."""

    def __init__(self, value: complex) -> None:
        self.kerneled = np.asarray(value)


class _MeasureResult:
    """Minimal stand-in for `MeasureResult`, keyed by qubit label."""

    def __init__(self, data: dict[str, _MeasureData]) -> None:
        self.data = data


class ReadoutScanExperiment:
    """
    An `Experiment` look-alike whose resonator moves with the qubit level.

    The response is the single-pole reflection `1 / (1 + 2 i d / kappa)` around a
    resonance that sits `k * chi` above the resting one for level `k`, so the
    phase of each level crosses zero at its own resonance and the levels can be
    separated. The prepared level is read back out of the schedule - it is the
    number of channels the schedule actually drives - so the ladder the
    experiment builds is exercised rather than re-derived.
    """

    def __init__(
        self,
        *,
        qubits: Sequence[str],
        registered_labels: Mapping[str, float],
        chi: float = DISPERSIVE_SHIFT,
        kappa: float = RESONATOR_KAPPA,
    ) -> None:
        self.chi = chi
        self.kappa = kappa
        self.calib_note = _CalibNote()
        self.ctx = _ReadoutContext(qubit_labels=qubits, labels=registered_labels)
        self.pulse = _Pulse(self.calib_note)
        self.measure_calls: list[dict[str, Any]] = []

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]) -> Iterator[None]:
        """Override target frequencies for the duration of the context."""
        original = {label: self.ctx.targets[label].frequency for label in frequencies}
        for label, frequency in frequencies.items():
            self.ctx.targets[label].frequency = frequency
        try:
            yield
        finally:
            for label, frequency in original.items():
                self.ctx.targets[label].frequency = frequency

    def measure(self, sequence: PulseSchedule, **kwargs: Any) -> _MeasureResult:
        """Return the reflection of the resonator at the applied frequency."""
        self.measure_calls.append(dict(kwargs))
        sequences = sequence.get_sampled_sequences()
        qubits = {self.ctx.resolve_qubit_label(label) for label in sequences}
        if len(qubits) != 1:
            raise AssertionError(f"The schedule spans {sorted(qubits)}.")
        qubit = qubits.pop()
        level = sum(1 for values in sequences.values() if np.any(np.abs(values) > 0))
        applied = self.ctx.targets[f"R{qubit}"].frequency
        detuning = applied - (READOUT_FREQUENCY + level * self.chi)
        return _MeasureResult(
            {qubit: _MeasureData(1.0 / (1.0 + 2j * detuning / self.kappa))}
        )


@pytest.fixture
def readout_scan_experiment_factory() -> Callable[..., ReadoutScanExperiment]:
    """
    Return a factory building a `ReadoutScanExperiment` with a calibrated ladder.

    The factory registers every transition target up to `max_level` and seeds a
    pi pulse on each rung, which is what a readout scan needs. A qubit left out
    of `registered_qubits` has no transition targets, and one left out of
    `calibrated_qubits` has no pi pulses, so both preflight failures can be
    provoked.
    """

    def factory(
        *,
        max_level: int,
        qubits: Sequence[str] | None = None,
        registered_qubits: Sequence[str] | None = None,
        calibrated_qubits: Sequence[str] | None = None,
        chi: float = DISPERSIVE_SHIFT,
        kappa: float = RESONATOR_KAPPA,
    ) -> ReadoutScanExperiment:
        labels = [QUBIT] if qubits is None else list(qubits)
        registered = labels if registered_qubits is None else list(registered_qubits)
        rungs = TRANSITION_NAMES[:max_level]
        experiment = ReadoutScanExperiment(
            qubits=labels,
            registered_labels={
                transition_label(name, qubit): GE_FREQUENCY
                + transition_index(name) * ALPHA
                for qubit in registered
                for name in rungs
            },
            chi=chi,
            kappa=kappa,
        )
        calibrated = labels if calibrated_qubits is None else list(calibrated_qubits)
        for qubit in calibrated:
            for name in rungs:
                label = transition_label(name, qubit)
                experiment.calib_note.update_pi_param(
                    label,
                    {
                        "target": label,
                        "duration": PREP_PI.duration,
                        "amplitude": PREP_PI.amplitude,
                        "tau": PREP_PI.tau,
                    },
                )
        return experiment

    return factory
