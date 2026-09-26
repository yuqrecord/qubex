"""Tests for the Blackman pulse."""

import numpy as np
import pytest
from qxpulse import Blackman, Drag, Pulse

import qubex as qx

dt = qx.pulse.get_sampling_period()


def test_inheritance():
    """Blackman should inherit from Pulse."""
    assert issubclass(Blackman, Pulse)


def test_empty_init():
    """Blackman should raise a TypeError if no duration is provided."""
    with pytest.raises(TypeError):
        Blackman()  # type: ignore


def test_init():
    """Blackman should be initialized with valid parameters."""
    pulse = Blackman(duration=5 * dt, amplitude=1, beta=1)
    assert pulse.name == "Blackman"
    assert pulse.length == 5
    assert pulse.duration == 5 * dt
    assert pulse.amplitude == 1
    assert pulse.beta == 1
    assert pulse.values == pytest.approx(
        [
            0.04021286 + 0.08904755j,
            0.50978714 + 0.35787384j,
            1.0 + 0.0j,
            0.50978714 - 0.35787384j,
            0.04021286 - 0.08904755j,
        ]
    )


def test_zero_duration():
    """Blackman should be initialized with zero duration."""
    pulse = Blackman(duration=0, amplitude=1, beta=1)
    assert pulse.duration == 0
    assert pulse.values == pytest.approx([])


def test_invalid_duration():
    """Blackman should raise a ValueError if duration is not a multiple of the sampling period."""
    with pytest.raises(
        ValueError, match=r"Duration must be a multiple of the sampling period"
    ):
        Blackman(duration=5 * dt + np.pi, amplitude=0.1)


def test_endpoints_are_zero():
    """Blackman should vanish at both ends of its support."""
    values = Blackman.func(t=[0.0, 100.0], duration=100, amplitude=1.0, beta=1.0)
    assert values == pytest.approx([0.0, 0.0], abs=1e-12)


def test_drag_dispatch():
    """Drag should reproduce the Blackman pulse when type is Blackman."""
    pulse = Blackman(duration=5 * dt, amplitude=1, beta=1)
    drag = Drag(duration=5 * dt, amplitude=1, beta=1, type="Blackman")
    assert drag.values == pytest.approx(pulse.values)
