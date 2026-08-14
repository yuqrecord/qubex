"""Tests for backend target helpers."""

from __future__ import annotations

from typing import Literal

import pytest

from qubex.system import Mux, Qubit, Resonator, Target, TargetType
from qubex.system.control_system import GenChannel, GenPort, PortType
from qubex.system.transition import (
    LEVEL_NAMES,
    TRANSITION_NAMES,
    TransitionName,
    transition_index,
)


def _make_gen_channel(*, sideband: Literal["U", "L"]) -> GenChannel:
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
        index=0,
        label=label,
        chip_id="chip",
        resonator=f"R{label}",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
    )


def _make_resonator(label: str, qubit: str) -> Resonator:
    return Resonator(
        index=0,
        label=label,
        chip_id="chip",
        qubit=qubit,
        _readout_frequency=6.5,
    )


def test_target_label_helpers():
    """Target label helpers should parse and format labels."""
    assert Target.qubit_label("Q00") == "Q00"
    assert Target.qubit_label("Q00-ef") == "Q00"
    assert Target.qubit_label("Q00-fh") == "Q00"
    assert Target.qubit_label("Q00-CR") == "Q00"
    assert Target.qubit_label("Q00-Q01") == "Q00"
    assert Target.qubit_label("Q00_read") == "Q00"
    assert Target.fh_label("Q00") == "Q00-fh"
    assert Target.read_label("Q00") == "RQ00"
    assert Target.cr_label("Q00", "Q01") == "Q00-Q01"
    with pytest.raises(ValueError, match="Invalid target label"):
        Target.qubit_label("BAD")


def test_cr_target_stores_qubit_roles_in_metadata() -> None:
    """CR targets should carry qubit roles as metadata."""
    control = _make_qubit("Q00")
    target_qubit = _make_qubit("Q01")
    channel = _make_gen_channel(sideband="U")

    cr_pair = Target.new_cr_target(
        control_qubit=control,
        target_qubit=target_qubit,
        channel=channel,
    )
    cr_default = Target.new_cr_target(control_qubit=control, channel=channel)

    assert cr_pair.metadata == {"control_qubit": "Q00", "target_qubit": "Q01"}
    assert cr_default.metadata == {"control_qubit": "Q00", "target_qubit": "CR"}


def test_target_frequency_and_availability():
    """Target should compute AWG frequency and availability."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel(sideband="U")
    target = Target.new_ge_target(qubit=qubit, channel=channel)
    target.frequency = 9.95  # GHz
    assert target.awg_frequency == pytest.approx(-0.05, abs=1e-6)
    assert target.is_available is True


def test_target_related_qubits_and_qubit_property():
    """Target should resolve related qubits for objects."""
    qubit = _make_qubit("Q00")
    resonator = _make_resonator("RQ00", "Q00")
    mux = Mux(index=0, label="M0", chip_id="chip", resonators=(resonator,))
    channel = _make_gen_channel(sideband="L")

    qubit_target = Target.new_ge_target(qubit=qubit, channel=channel)
    read_target = Target.new_read_target(resonator=resonator, channel=channel)
    pump_target = Target.new_pump_target(mux=mux, frequency=9.0, channel=channel)

    assert qubit_target.qubit == "Q00"
    assert read_target.qubit == "Q00"
    assert pump_target.qubit == ""
    assert qubit_target.is_related_to_qubits(["Q00"]) is True
    assert read_target.is_related_to_qubits(["Q00"]) is True
    assert pump_target.is_related_to_qubits(["Q00"]) is True
    assert qubit_target.type == TargetType.CTRL_GE


def test_fh_target_uses_qubit_fh_frequency():
    """FH target should use the qubit FH control frequency."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel(sideband="L")

    target = Target.new_fh_target(qubit=qubit, channel=channel)

    assert target.label == "Q00-fh"
    assert target.type == TargetType.CTRL_FH
    assert target.is_fh is True
    assert target.frequency == pytest.approx(4.4)


def test_transition_names_match_level_ladder():
    """Transition names should be the adjacent level-name pairs."""
    assert len(TRANSITION_NAMES) == len(LEVEL_NAMES) - 1
    for index, transition in enumerate(TRANSITION_NAMES):
        assert transition == LEVEL_NAMES[index] + LEVEL_NAMES[index + 1]
        assert transition_index(transition) == index


def test_transition_index_rejects_unknown_transition():
    """Unknown transition names should raise a descriptive error."""
    with pytest.raises(ValueError, match="Unsupported transition"):
        transition_index("no")  # type: ignore[arg-type]


@pytest.mark.parametrize("transition", TRANSITION_NAMES)
def test_transition_label_helpers_cover_every_transition(transition: TransitionName):
    """Every transition should build a label and parse back to its qubit."""
    expected = "Q00" if transition == "ge" else f"Q00-{transition}"

    assert Target.transition_label("Q00", transition) == expected
    assert Target.transition_label("Q00-ef", transition) == expected
    assert Target.qubit_label(expected) == "Q00"


def test_named_transition_label_helpers_match_generic_helper():
    """Named label helpers should agree with the generic helper."""
    assert Target.hi_label("Q00") == "Q00-hi"
    assert Target.ij_label("Q00") == "Q00-ij"
    assert Target.jk_label("Q00") == "Q00-jk"
    assert Target.kl_label("Q00") == "Q00-kl"
    assert Target.lm_label("Q00") == "Q00-lm"
    assert Target.mn_label("Q00") == "Q00-mn"


@pytest.mark.parametrize("transition", TRANSITION_NAMES)
def test_new_transition_target_uses_transition_metadata(transition: TransitionName):
    """Transition targets should carry the matching label, type, and frequency."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel(sideband="L")

    target = Target.new_transition_target(
        transition=transition,
        qubit=qubit,
        channel=channel,
    )

    assert target.label == Target.transition_label("Q00", transition)
    assert target.type is TargetType[f"CTRL_{transition.upper()}"]
    assert target.transition == transition
    assert target.is_transition is True
    assert target.object is qubit
    assert target.channel is channel
    assert target.frequency == pytest.approx(qubit.get_control_frequency(transition))


def test_named_transition_target_predicates():
    """Named transition predicates should follow the target type."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel(sideband="L")

    hi_target = Target.new_hi_target(qubit=qubit, channel=channel)
    mn_target = Target.new_mn_target(qubit=qubit, channel=channel)

    assert hi_target.is_hi is True
    assert hi_target.is_mn is False
    assert mn_target.is_mn is True
    assert mn_target.is_ij is False


def test_non_transition_target_has_no_transition():
    """Readout targets should not report a transition."""
    resonator = _make_resonator("RQ00", "Q00")
    channel = _make_gen_channel(sideband="L")

    target = Target.new_read_target(resonator=resonator, channel=channel)

    assert target.transition is None
    assert target.is_transition is False


def test_qubit_control_frequency_derives_higher_transitions():
    """Higher transitions should follow `f_ge + n * (f_ef - f_ge)` by default."""
    qubit = _make_qubit("Q00")

    assert qubit.get_control_frequency("ge") == pytest.approx(5.0)
    assert qubit.get_control_frequency("ef") == pytest.approx(4.7)
    assert qubit.get_control_frequency("fh") == pytest.approx(4.4)
    assert qubit.control_frequency_hi == pytest.approx(4.1)
    assert qubit.control_frequency_ij == pytest.approx(3.8)
    assert qubit.control_frequency_jk == pytest.approx(3.5)
    assert qubit.control_frequency_kl == pytest.approx(3.2)
    assert qubit.control_frequency_lm == pytest.approx(2.9)
    assert qubit.control_frequency_mn == pytest.approx(2.6)


def test_qubit_control_frequency_prefers_stored_override():
    """A stored override should win over the derived transition frequency."""
    qubit = _make_qubit("Q00")
    qubit.transition_control_frequency_values = {"hi": 4.05, "mn": 2.55}

    assert qubit.control_frequency_hi == pytest.approx(4.05)
    assert qubit.control_frequency_mn == pytest.approx(2.55)
    assert qubit.control_frequency_ij == pytest.approx(3.8)
    assert qubit.frequency == pytest.approx(5.0)
    assert qubit.control_frequency_ef == pytest.approx(4.7)
    assert qubit.control_frequency_fh == pytest.approx(4.4)


def test_qubit_control_frequency_falls_back_to_bare_frequency():
    """Derived higher frequencies should use the bare frequency when GE is unset."""
    qubit = Qubit(
        index=0,
        label="Q00",
        chip_id="chip",
        resonator="RQ00",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
    )

    assert qubit.get_control_frequency("ef") == pytest.approx(4.7)
    assert qubit.get_control_frequency("hi") == pytest.approx(4.1)
