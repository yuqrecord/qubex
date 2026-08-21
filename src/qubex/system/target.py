"""Target and target label helpers for control and readout channels."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any, Literal

from pydantic import Field

from qubex.core import MutableModel

from .control_system import CapChannel, GenChannel
from .quantum_system import Mux, Qubit, Resonator
from .target_type import (
    TRANSITION_BY_TARGET_TYPE,
    TRANSITION_TARGET_TYPES,
    TargetType,
)
from .transition import TransitionName

PhysicalObject = Qubit | Resonator | Mux


class CapTarget(MutableModel):
    """Capture target metadata for readout channels."""

    label: str
    frequency: float
    object: PhysicalObject
    channel: CapChannel
    type: TargetType

    @classmethod
    def new_read_target(
        cls,
        *,
        resonator: Resonator,
        channel: CapChannel,
    ) -> CapTarget:
        """Create a readout capture target for a resonator."""
        return cls(
            label=Target.read_label(resonator.label),
            object=resonator,
            frequency=resonator.frequency,
            channel=channel,
            type=TargetType.READ,
        )


class Target(MutableModel):
    """Generator target metadata for control and readout channels."""

    label: str
    frequency: float
    object: PhysicalObject
    channel: GenChannel
    type: TargetType
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        """Return the debug representation of the target."""
        return f"Target(label={self.label}, frequency={self.frequency}, channel={self.channel.id}, object={self.object.label})"

    @property
    def qubit(self) -> str:
        """Return the associated qubit label."""
        if isinstance(self.object, Qubit):
            return self.object.label
        elif isinstance(self.object, Resonator):
            return self.object.qubit
        elif isinstance(self.object, Mux):
            return ""
        else:
            raise TypeError("Invalid quantum object.")

    @property
    def sideband(self) -> Literal["U", "L"] | None:
        """Return the sideband setting of the channel."""
        return self.channel.port.sideband

    @property
    def coarse_frequency(self) -> float:
        """Return the channel coarse frequency in GHz."""
        if isinstance(self.channel, GenChannel):
            return self.channel.coarse_frequency * 1e-9
        else:
            raise TypeError("Invalid channel.")

    @property
    def fine_frequency(self) -> float:
        """Return the channel fine frequency in GHz."""
        if isinstance(self.channel, GenChannel):
            return self.channel.fine_frequency * 1e-9
        else:
            raise TypeError("Invalid channel.")

    @property
    def awg_frequency(self) -> float:
        """Return the AWG frequency in GHz for the target."""
        if isinstance(self.channel, GenChannel):
            if self.sideband == "L":
                return self.fine_frequency - self.frequency
            else:
                return self.frequency - self.fine_frequency
        else:
            raise TypeError("Invalid channel.")

    @property
    def is_available(self) -> bool:
        """Report whether the target frequency is within the tuning range."""
        return abs(self.frequency - self.fine_frequency) < 250 * 1e-3  # 250 MHz

    @property
    def is_ge(self) -> bool:
        """Report whether this is a GE control target."""
        return self.type == TargetType.CTRL_GE

    @property
    def is_ef(self) -> bool:
        """Report whether this is an EF control target."""
        return self.type == TargetType.CTRL_EF

    @property
    def is_fh(self) -> bool:
        """Report whether this is an FH control target."""
        return self.type == TargetType.CTRL_FH

    @property
    def is_hi(self) -> bool:
        """Report whether this is an HI control target."""
        return self.type == TargetType.CTRL_HI

    @property
    def is_ij(self) -> bool:
        """Report whether this is an IJ control target."""
        return self.type == TargetType.CTRL_IJ

    @property
    def is_jk(self) -> bool:
        """Report whether this is a JK control target."""
        return self.type == TargetType.CTRL_JK

    @property
    def is_kl(self) -> bool:
        """Report whether this is a KL control target."""
        return self.type == TargetType.CTRL_KL

    @property
    def is_lm(self) -> bool:
        """Report whether this is an LM control target."""
        return self.type == TargetType.CTRL_LM

    @property
    def is_mn(self) -> bool:
        """Report whether this is an MN control target."""
        return self.type == TargetType.CTRL_MN

    @property
    def transition(self) -> TransitionName | None:
        """Return the driven transition, or None for non-transition targets."""
        return TRANSITION_BY_TARGET_TYPE.get(self.type)

    @property
    def is_transition(self) -> bool:
        """Report whether this target drives a single-qubit transition."""
        return self.type in TRANSITION_BY_TARGET_TYPE

    @property
    def is_cr(self) -> bool:
        """Report whether this is a CR control target."""
        return self.type == TargetType.CTRL_CR

    @property
    def is_bswap(self) -> bool:
        """Report whether this is a bSWAP control target."""
        return self.type == TargetType.CTRL_2Q and self.metadata.get("gate") == "BSWAP"

    @property
    def is_2q(self) -> bool:
        """Report whether this target drives a two-qubit interaction."""
        return self.type in (
            TargetType.CTRL_2Q,
            TargetType.CTRL_CR,
        )

    @property
    def is_read(self) -> bool:
        """Report whether this is a readout target."""
        return self.type == TargetType.READ

    @property
    def is_pump(self) -> bool:
        """Report whether this is a pump target."""
        return self.type == TargetType.PUMP

    def is_related_to_qubits(self, qubits: Collection[str]) -> bool:
        """Return whether the target relates to the provided qubits."""
        if isinstance(self.object, Qubit):
            return self.object.label in qubits
        elif isinstance(self.object, Resonator):
            return self.object.qubit in qubits
        elif isinstance(self.object, Mux):
            return any(
                qubit in qubits
                for qubit in [resonator.qubit for resonator in self.object.resonators]
            )
        else:
            raise TypeError("Invalid quantum object.")

    @classmethod
    def new_target(
        cls,
        *,
        label: str,
        frequency: float,
        object: PhysicalObject,
        channel: GenChannel,
        type: TargetType = TargetType.UNKNOWN,
        metadata: dict[str, Any] | None = None,
    ) -> Target:
        """Create a generic target from metadata."""
        return cls(
            label=label,
            frequency=frequency,
            object=object,
            channel=channel,
            type=type,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def new_transition_target(
        cls,
        *,
        transition: TransitionName,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create a transition control target for a qubit."""
        return cls(
            label=Target.transition_label(qubit.label, transition),
            object=qubit,
            frequency=qubit.get_control_frequency(transition),
            channel=channel,
            type=TRANSITION_TARGET_TYPES[transition],
        )

    @classmethod
    def new_ge_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create a GE control target for a qubit."""
        return cls.new_transition_target(
            transition="ge",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_ef_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an EF control target for a qubit."""
        return cls.new_transition_target(
            transition="ef",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_fh_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an FH control target for a qubit."""
        return cls.new_transition_target(
            transition="fh",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_hi_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an HI control target for a qubit."""
        return cls.new_transition_target(
            transition="hi",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_ij_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an IJ control target for a qubit."""
        return cls.new_transition_target(
            transition="ij",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_jk_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create a JK control target for a qubit."""
        return cls.new_transition_target(
            transition="jk",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_kl_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create a KL control target for a qubit."""
        return cls.new_transition_target(
            transition="kl",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_lm_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an LM control target for a qubit."""
        return cls.new_transition_target(
            transition="lm",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_mn_target(
        cls,
        *,
        qubit: Qubit,
        channel: GenChannel,
    ) -> Target:
        """Create an MN control target for a qubit."""
        return cls.new_transition_target(
            transition="mn",
            qubit=qubit,
            channel=channel,
        )

    @classmethod
    def new_cr_target(
        cls,
        *,
        control_qubit: Qubit,
        target_qubit: Qubit | None = None,
        channel: GenChannel,
    ) -> Target:
        """Create a CR control target for a qubit pair."""
        if target_qubit is not None:
            return cls(
                label=Target.cr_label(control_qubit.label, target_qubit.label),
                frequency=target_qubit.frequency,
                object=control_qubit,
                channel=channel,
                type=TargetType.CTRL_CR,
                metadata={
                    "control_qubit": control_qubit.label,
                    "target_qubit": target_qubit.label,
                },
            )
        else:
            return cls(
                label=Target.cr_label(control_qubit.label),
                frequency=round(channel.fine_frequency * 1e-9, 6),
                object=control_qubit,
                channel=channel,
                type=TargetType.CTRL_CR,
                metadata={
                    "control_qubit": control_qubit.label,
                    "target_qubit": "CR",
                },
            )

    @classmethod
    def new_read_target(
        cls,
        *,
        resonator: Resonator,
        channel: GenChannel,
    ) -> Target:
        """Create a readout target for a resonator."""
        return cls(
            label=Target.read_label(resonator.label),
            object=resonator,
            frequency=resonator.frequency,
            channel=channel,
            type=TargetType.READ,
        )

    @classmethod
    def new_pump_target(
        cls,
        *,
        mux: Mux,
        frequency: float,
        channel: GenChannel,
    ) -> Target:
        """Create a pump target for a mux."""
        return cls(
            label=mux.label,
            object=mux,
            frequency=frequency,
            channel=channel,
            type=TargetType.PUMP,
        )

    @classmethod
    def qubit_label(
        cls,
        label: str,
    ) -> str:
        """Extract a qubit label from a target label."""
        if (
            (match := re.match(r"^R(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)-ef$", label))
            or (match := re.match(r"^(Q\d+)-fh$", label))
            or (match := re.match(r"^(Q\d+)-CR$", label))
            or (match := re.match(r"^(Q\d+)-(Q\d+)$", label))
            or (match := re.match(r"^(Q\d+)(-|_)[a-zA-Z0-9]+$", label))
        ):
            qubit_label = match.group(1)
        else:
            raise ValueError(f"Invalid target label `{label}`.")
        return qubit_label

    @classmethod
    def transition_label(cls, label: str, transition: TransitionName) -> str:
        """Return the transition target label for a qubit label."""
        qubit = cls.qubit_label(label)
        if transition == "ge":
            return f"{qubit}"
        return f"{qubit}-{transition}"

    @classmethod
    def ge_label(cls, label: str) -> str:
        """Return the GE target label for a qubit label."""
        return cls.transition_label(label, "ge")

    @classmethod
    def ef_label(cls, label: str) -> str:
        """Return the EF target label for a qubit label."""
        return cls.transition_label(label, "ef")

    @classmethod
    def fh_label(cls, label: str) -> str:
        """Return the FH target label for a qubit label."""
        return cls.transition_label(label, "fh")

    @classmethod
    def hi_label(cls, label: str) -> str:
        """Return the HI target label for a qubit label."""
        return cls.transition_label(label, "hi")

    @classmethod
    def ij_label(cls, label: str) -> str:
        """Return the IJ target label for a qubit label."""
        return cls.transition_label(label, "ij")

    @classmethod
    def jk_label(cls, label: str) -> str:
        """Return the JK target label for a qubit label."""
        return cls.transition_label(label, "jk")

    @classmethod
    def kl_label(cls, label: str) -> str:
        """Return the KL target label for a qubit label."""
        return cls.transition_label(label, "kl")

    @classmethod
    def lm_label(cls, label: str) -> str:
        """Return the LM target label for a qubit label."""
        return cls.transition_label(label, "lm")

    @classmethod
    def mn_label(cls, label: str) -> str:
        """Return the MN target label for a qubit label."""
        return cls.transition_label(label, "mn")

    @classmethod
    def cr_label(cls, control_label: str, target_label: str | None = None) -> str:
        """Return the CR target label for a control/target pair."""
        control_qubit = cls.qubit_label(control_label)
        if target_label is None:
            target_label = "CR"
        else:
            target_label = cls.qubit_label(target_label)
        return f"{control_qubit}-{target_label}"

    @classmethod
    def read_label(cls, label: str) -> str:
        """Return the readout target label for a qubit label."""
        qubit = cls.qubit_label(label)
        return f"R{qubit}"
