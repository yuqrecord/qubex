"""Duck-typed `Experiment` stub for the ground-referenced transition tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from qubex.experiment.models.rabi_param import RabiParam
from qubex.pulse import Drag, FlatTop, PulseSchedule
from qubex.system.transition import (
    TRANSITION_NAMES,
    TransitionName,
    transition_index,
)

QUBIT = "Q00"
SAMPLING_PERIOD = 2.0
ALPHA = -0.35
GE_FREQUENCY = 8.0

# Deliberately non-trivial so that the normalization path is exercised rather
# than accidentally reducing to the identity.
RABI_ANGLE = 0.7
RABI_OFFSET = 0.1
RABI_AMPLITUDE = 0.8
RABI_DISTANCE = 0.3
RABI_FREQUENCY = 0.01

PREP_PI = FlatTop(duration=32, amplitude=0.25, tau=12)


def transition_label(transition: TransitionName) -> str:
    """Return the target label of a transition of the stub qubit."""
    return QUBIT if transition == "ge" else f"{QUBIT}-{transition}"


def lower_labels(transition: TransitionName) -> tuple[str, ...]:
    """Return the ordered lower-rung labels of a transition."""
    return tuple(
        transition_label(name)
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


class _Qubit:
    """Minimal stand-in for a qubit model with a linear level ladder."""

    def __init__(self, label: str, alpha: float) -> None:
        self.label = label
        self.alpha = alpha
        self.frequency = GE_FREQUENCY

    def get_control_frequency(self, transition: TransitionName) -> float:
        """Return `f_ge + k * alpha` for the transition `k -> k+1`."""
        return self.frequency + transition_index(transition) * self.alpha


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

    def __init__(self, *, labels: Sequence[str], calib_note: _CalibNote) -> None:
        self.qubit_labels = [QUBIT]
        self.qubits = {QUBIT: _Qubit(QUBIT, ALPHA)}
        self.targets = dict.fromkeys(labels, object())
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
            )
        param = self._calib_note.get_pi_param(target)
        if param is not None:
            return FlatTop(
                duration=param["duration"],
                amplitude=param["amplitude"],
                tau=param["tau"],
            )
        return PREP_PI.scaled(2)

    def get_hpi_pulse(self, target: str) -> Any:
        """Return the stored FlatTop half-pi pulse, or a nominal one."""
        param = self._calib_note.get_hpi_param(target)
        if param is None:
            return FlatTop(duration=32, amplitude=0.3, tau=12)
        return FlatTop(
            duration=param["duration"],
            amplitude=param["amplitude"],
            tau=param["tau"],
        )

    def get_drag_hpi_pulse(self, target: str) -> Any:
        """Return the stored DRAG half-pi pulse."""
        param = self._calib_note.get_drag_hpi_param(target)
        if param is None:
            raise KeyError(target)
        return Drag(
            duration=param["duration"],
            amplitude=param["amplitude"],
            beta=param["beta"],
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
        values = [iq_from_z(experiment.simulate(schedule)) for schedule in schedules]
        return _SweepResult(
            {QUBIT: _SweepPoint(np.asarray(values, dtype=np.complex128))}
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
        rabi_rate_per_amplitude: float,
    ) -> None:
        self.transition = transition
        self.target = transition_label(transition)
        self.lower_labels = lower_labels(transition)
        self.rabi_rate_per_amplitude = rabi_rate_per_amplitude
        self.calib_note = _CalibNote()
        self.ctx = _Context(
            labels=[*self.lower_labels, self.target],
            calib_note=self.calib_note,
        )
        self.pulse = _Pulse(self.calib_note)
        self.measurement_service = _MeasurementService(self)
        self.sweep_calls: list[dict[str, Any]] = []
        self.reference_calls: list[dict[str, Any]] = []

    def simulate(self, schedule: PulseSchedule) -> float:
        """Return the normalized signal the schedule would produce."""
        waveform = schedule.get_sampled_sequences()[self.target]
        area = float(np.real(waveform).sum()) * SAMPLING_PERIOD
        angle = 2 * np.pi * self.rabi_rate_per_amplitude * area
        return float(np.cos(angle))

    def calc_control_amplitude(self, target: str, rabi_rate: float) -> float:
        """Invert the stub's linear amplitude-to-Rabi-rate relation."""
        assert target == self.target
        return rabi_rate / self.rabi_rate_per_amplitude


@pytest.fixture
def stub_experiment_factory() -> Callable[..., StubExperiment]:
    """
    Return a factory building a `StubExperiment` with a known true amplitude.

    The factory takes the transition, the pulse family under test, its shape and
    the amplitude that realizes the ideal rotation, derives the stub's linear
    amplitude-to-Rabi-rate relation from that, and seeds the calibrations every
    script requires: a pi pulse on each lower rung, the transition's Rabi
    parameters and — optionally — the transition's FlatTop half-pi.
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
        with_hpi: bool = True,
        hpi_amplitude: float | None = None,
    ) -> StubExperiment:
        if envelope == "flat_top":
            unit = FlatTop(duration=duration, amplitude=1, tau=ramptime)
        else:
            unit = Drag(duration=duration, amplitude=1, beta=beta)
        unit_area = float(unit.real.sum()) * SAMPLING_PERIOD
        turns = 0.25 if pulse_type == "hpi" else 0.5
        experiment = StubExperiment(
            transition=transition,
            rabi_rate_per_amplitude=turns / (unit_area * true_amplitude),
        )

        for label in experiment.lower_labels:
            experiment.calib_note.update_pi_param(
                label,
                {
                    "target": label,
                    "duration": PREP_PI.duration,
                    "amplitude": PREP_PI.amplitude,
                    "tau": PREP_PI.tau,
                },
            )
        experiment.ctx.stored_rabi_params[experiment.target] = rabi_param(
            experiment.target
        )
        if with_hpi:
            flat_top_hpi = FlatTop(duration=32, amplitude=1, tau=12)
            hpi_area = float(flat_top_hpi.real.sum()) * SAMPLING_PERIOD
            experiment.calib_note.update_hpi_param(
                experiment.target,
                {
                    "target": experiment.target,
                    "duration": 32,
                    "amplitude": (
                        hpi_amplitude
                        if hpi_amplitude is not None
                        else 0.25 / (hpi_area * experiment.rabi_rate_per_amplitude)
                    ),
                    "tau": 12,
                },
            )
        return experiment

    return factory
