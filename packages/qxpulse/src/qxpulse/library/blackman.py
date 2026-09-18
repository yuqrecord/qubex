"""Blackman pulse shape helpers."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from qxpulse.pulse import Pulse

# Classic Blackman window coefficients (w(0) = w(T) = 0, w(T/2) = 1).
A0: Final = 0.42
A1: Final = 0.5
A2: Final = 0.08


class Blackman(Pulse):
    """
    A class to represent a Blackman window pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    beta : float, optional
        DRAG correction coefficient. Default is None.

    Examples
    --------
    >>> pulse = Blackman(
    ...     duration=100,
    ...     amplitude=1.0,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        beta: float | None = None,
        **kwargs,
    ):
        super().__init__(
            duration=duration,
            **kwargs,
        )

        self.amplitude: Final = amplitude
        self.beta: Final = beta
        self._finalize_initialization()

    @override
    def _sample_values(self) -> NDArray[np.complex128]:
        """Return sampled values for the Blackman pulse."""
        if self.length == 0:
            return np.array([], dtype=np.complex128)
        duration = self.duration
        return Blackman.func(
            t=self._sampling_points(duration),
            duration=duration,
            amplitude=self.amplitude,
            beta=self.beta,
        )

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        beta: float | None = None,
    ) -> NDArray:
        """
        Compute a Blackman window pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        beta : float, optional
            DRAG correction coefficient. Default is None.
        """
        t = np.asarray(t)

        if duration == 0:
            return np.zeros_like(t, dtype=np.complex128)

        theta = 2 * np.pi * t / duration
        Omega = amplitude * (A0 - A1 * np.cos(theta) + A2 * np.cos(2 * theta))
        if beta is None:
            values = Omega
        else:
            dOmega = (
                amplitude
                * (2 * np.pi / duration)
                * (A1 * np.sin(theta) - 2 * A2 * np.sin(2 * theta))
            )
            values = Omega + beta * 1j * dOmega
        return np.where(
            (t >= 0) & (t <= duration),
            values,
            0,
        ).astype(np.complex128)
