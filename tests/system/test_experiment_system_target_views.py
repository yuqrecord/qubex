"""Tests for ExperimentSystem target collections."""

from __future__ import annotations

from typing import Literal

from qubex.system import Qubit, Resonator, Target
from qubex.system.control_system import GenChannel, GenPort, PortType
from qubex.system.experiment_system import ExperimentSystem
from qubex.system.target_registry import TargetRegistry


def _make_gen_channel(*, sideband: Literal["U", "L"] = "U") -> GenChannel:
    port = GenPort(
        id="B0-P0",
        box_id="B0",
        number=0,
        type=PortType.CTRL,
        channels=(),
        sideband=sideband,
        lo_freq=9_000_000_000,
        cnco_freq=1_000_000_000,
    )
    return GenChannel(id="B0-P0-CH0", number=0, _port=port, fnco_freq=0)


def _make_qubit(label: str) -> Qubit:
    return Qubit(
        index=int(label[1:]),
        label=label,
        chip_id="chip",
        resonator=f"R{label}",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
    )


def _make_system(targets: list[Target]) -> ExperimentSystem:
    system = object.__new__(ExperimentSystem)
    system.__dict__["_target_registry"] = TargetRegistry(gen_targets=targets)
    return system


def test_transition_targets_include_higher_transitions() -> None:
    """Given higher transition targets, when listing them, then all transitions appear."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel()
    resonator = Resonator(
        index=0,
        label="RQ00",
        chip_id="chip",
        qubit="Q00",
        _readout_frequency=6.5,
    )
    targets = [
        Target.new_hi_target(qubit=qubit, channel=channel),
        Target.new_ge_target(qubit=qubit, channel=channel),
        Target.new_read_target(resonator=resonator, channel=channel),
        Target.new_mn_target(qubit=qubit, channel=channel),
        Target.new_ef_target(qubit=qubit, channel=channel),
    ]

    system = _make_system(targets)

    # Grouped by transition regardless of registration order, and readout excluded.
    assert [target.label for target in system.transition_targets] == [
        "Q00",
        "Q00-ef",
        "Q00-hi",
        "Q00-mn",
    ]


def test_ctrl_targets_include_higher_transitions() -> None:
    """Given a higher transition target, when listing control targets, then it is kept."""
    qubit0 = _make_qubit("Q00")
    qubit1 = _make_qubit("Q01")
    channel = _make_gen_channel()
    targets = [
        Target.new_ge_target(qubit=qubit0, channel=channel),
        Target.new_ef_target(qubit=qubit0, channel=channel),
        Target.new_fh_target(qubit=qubit0, channel=channel),
        Target.new_hi_target(qubit=qubit0, channel=channel),
        Target.new_cr_target(
            control_qubit=qubit0,
            target_qubit=qubit1,
            channel=channel,
        ),
    ]

    system = _make_system(targets)

    assert [target.label for target in system.ctrl_targets] == [
        "Q00",
        "Q00-ef",
        "Q00-fh",
        "Q00-hi",
        "Q00-Q01",
    ]


def test_ctrl_targets_keep_legacy_order_without_higher_transitions() -> None:
    """Given only ge/ef/fh targets, when listing control targets, then the order is unchanged."""
    qubit0 = _make_qubit("Q00")
    qubit1 = _make_qubit("Q01")
    channel = _make_gen_channel()
    targets = [
        Target.new_ge_target(qubit=qubit0, channel=channel),
        Target.new_ef_target(qubit=qubit0, channel=channel),
        Target.new_ge_target(qubit=qubit1, channel=channel),
        Target.new_ef_target(qubit=qubit1, channel=channel),
    ]

    system = _make_system(targets)

    # The pre-existing contract grouped all ge targets before all ef targets.
    assert [target.label for target in system.ctrl_targets] == [
        "Q00",
        "Q01",
        "Q00-ef",
        "Q01-ef",
    ]
