"""Qubit level ladder and transition-name helpers."""

from __future__ import annotations

from typing import Literal, TypeAlias

LEVEL_NAMES: tuple[str, ...] = (
    "g",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
)
"""Level names ordered by energy, so `LEVEL_NAMES[n]` names level `n`."""

TransitionName: TypeAlias = Literal[
    "ge",
    "ef",
    "fh",
    "hi",
    "ij",
    "jk",
    "kl",
    "lm",
    "mn",
]
"""Supported single-photon transition names."""

TRANSITION_NAMES: tuple[TransitionName, ...] = (
    "ge",
    "ef",
    "fh",
    "hi",
    "ij",
    "jk",
    "kl",
    "lm",
    "mn",
)
"""Transition names ordered by energy, so `TRANSITION_NAMES[n]` drives level `n` to `n + 1`."""

_TRANSITION_INDEXES: dict[str, int] = {
    transition: index for index, transition in enumerate(TRANSITION_NAMES)
}


def transition_index(transition: TransitionName) -> int:
    """Return the lower level index driven by a transition."""
    try:
        return _TRANSITION_INDEXES[transition]
    except KeyError:
        raise ValueError(f"Unsupported transition `{transition}`.") from None


def is_transition_name(value: str) -> bool:
    """Report whether a string is a supported transition name."""
    return value in _TRANSITION_INDEXES
