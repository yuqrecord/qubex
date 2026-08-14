"""Target-type enumeration shared across system models."""

from __future__ import annotations

from enum import Enum

from .transition import TRANSITION_NAMES, TransitionName


class TargetType(Enum):
    """Enumerate supported logical target types."""

    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_FH = "CTRL_FH"
    CTRL_HI = "CTRL_HI"
    CTRL_IJ = "CTRL_IJ"
    CTRL_JK = "CTRL_JK"
    CTRL_KL = "CTRL_KL"
    CTRL_LM = "CTRL_LM"
    CTRL_MN = "CTRL_MN"
    CTRL_2Q = "CTRL_2Q"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    PUMP = "PUMP"
    UNKNOWN = "UNKNOWN"


TRANSITION_TARGET_TYPES: dict[TransitionName, TargetType] = {
    transition: TargetType[f"CTRL_{transition.upper()}"]
    for transition in TRANSITION_NAMES
}
"""Control target type for each transition name."""

TRANSITION_BY_TARGET_TYPE: dict[TargetType, TransitionName] = {
    target_type: transition
    for transition, target_type in TRANSITION_TARGET_TYPES.items()
}
"""Transition name for each transition-control target type."""
