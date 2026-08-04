"""
Shared analysis for the AC Stark spectroscopy variants.

Both variants probe the same ge transition and reference the driven
resonance against a no-drive measurement, so the single-trace Lorentzian
fit and the shift / photon-number derivation are identical; only the
array shapes differ, and those are handled by the callers.

Line polarity follows the CKP convention (see ``ckp_characterization``):

- state "g": ge absorption |g> -> |e>, observed as a dip
- state "e": a ge pi pulse prepares |e>, then ge stimulated emission
  |e> -> |g> is observed as a peak

Because the probe is a pi pulse, the observed line is strongly power
broadened: its width is set by the pi pulse Rabi frequency (of order
``1 / duration``) rather than by the qubit linewidth.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubex.analysis import FitStatus, fitting
from qubex.experiment.models import RabiParam

# FWHM of the power spectrum of a flat-top probe of duration D is ~0.443/D.
LORENTZIAN_TIME_BANDWIDTH = 0.443
# Kept at 0.5 to match the ckp_characterization gating (r2 > 0.5); the
# polarity check makes a stricter threshold unnecessary for rejecting
# noise-only traces, and raising it without hardware data would risk
# rejecting genuinely broadened lines at high photon number.
FIT_R2_THRESHOLD = 0.5


def smooth_for_initial_guess(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a lightly smoothed copy used only for the fit initial guess."""
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(signal, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def check_lorentzian_fit(
    fit_result: Any,
    *,
    x: NDArray[np.float64],
    expected_amplitude_sign: int,
) -> tuple[bool, str]:
    """Return whether a Lorentzian fit is usable, with the reason if not."""
    if fit_result is None:
        return False, "fit raised an exception"
    if fit_result.status is not FitStatus.SUCCESS:
        return False, "fit did not converge"
    amplitude = float(fit_result.get("A", np.nan))
    f0 = float(fit_result.get("f0", np.nan))
    f0_err = float(fit_result.get("f0_err", np.nan))
    gamma = float(fit_result.get("gamma", np.nan))
    r2 = float(fit_result.get("r2", np.nan))
    if not (
        np.isfinite(amplitude)
        and np.isfinite(f0)
        and np.isfinite(f0_err)
        and np.isfinite(gamma)
        and np.isfinite(r2)
    ):
        return False, f"non-finite fit parameters (f0={f0}, err={f0_err}, r2={r2})"
    if amplitude * expected_amplitude_sign <= 0:
        return False, (
            f"line polarity mismatch (A={amplitude:.6g}, expected "
            f"{'positive' if expected_amplitude_sign > 0 else 'negative'})"
        )
    span = float(np.max(x) - np.min(x))
    spacing = float(np.median(np.diff(x)))
    if r2 <= FIT_R2_THRESHOLD:
        return False, f"poor fit quality (r2={r2:.3f})"
    if not gamma < span:
        return False, f"linewidth exceeds the sweep span (gamma={gamma:.6g} GHz)"
    if f0 < np.min(x) + spacing or f0 > np.max(x) - spacing:
        return False, f"fitted center is at the sweep edge (f0={f0:.6f} GHz)"
    if f0_err > span:
        return False, f"fitted center is unconstrained (f0_err={f0_err:.6g} GHz)"
    return True, ""


def fit_spectroscopy_trace(
    *,
    detuning_range: ArrayLike,
    iq: ArrayLike,
    nominal_frequency: float,
    qubit_initial_state: Literal["g", "e"],
    qubit_drive_duration: float,
    rabi_param: RabiParam,
) -> dict[str, Any]:
    """
    Fit one ge spectroscopy trace and extract the resonance frequency.

    The complex IQ trace is normalized with the calibrated GE Rabi
    parameters, then fitted with a Lorentzian over the absolute qubit
    drive frequency. The line polarity is fixed by the preparation state
    (CKP convention): an absorption dip for "g" and a
    stimulated-emission peak for "e".

    Parameters
    ----------
    detuning_range : ArrayLike
        Qubit drive detuning sweep (GHz), in acquisition order.
    iq : ArrayLike
        Complex readout IQ values for each detuning point.
    nominal_frequency : float
        Calibrated ge transition frequency (GHz), used as the frequency
        axis origin for both preparation states.
    qubit_initial_state : {"g", "e"}
        Prepared qubit state; fixes the expected line polarity.
    qubit_drive_duration : float
        Duration of the pi pulse used as the probe (ns), used for the
        linewidth initial guess.
    rabi_param : RabiParam
        Calibrated GE Rabi parameters used to normalize the IQ data.

    Returns
    -------
    dict[str, Any]
        Fit results with keys ``normalized``, ``fit_result``, ``f0``,
        ``f0_err``, ``gamma``, ``r2`` and ``valid``.
    """
    detuning_range = np.asarray(detuning_range, dtype=float)
    iq_array = np.asarray(iq, dtype=complex)
    if detuning_range.shape != iq_array.shape:
        raise ValueError(
            "`detuning_range` and `iq` must have the same shape, got "
            f"{detuning_range.shape} and {iq_array.shape}."
        )

    normalized = np.asarray(rabi_param.normalize(iq_array), dtype=float)

    order = np.argsort(detuning_range)
    x = nominal_frequency + detuning_range[order]
    y = normalized[order]

    smoothed = smooth_for_initial_guess(y)
    n_edge = max(1, y.size // 10)
    baseline = float(np.mean(np.concatenate([smoothed[:n_edge], smoothed[-n_edge:]])))
    deviations = smoothed - baseline

    # CKP polarity convention with GE Rabi normalization: the |g> resonance
    # is an absorption dip, the |e> resonance is a stimulated-emission peak.
    # The polarity is fixed per state (no opposite-polarity retry) so that
    # noise or background variations are never adopted as the resonance.
    if qubit_initial_state == "g":
        expected_amplitude_sign = -1
        extremum_index = int(np.argmin(deviations))
    else:
        expected_amplitude_sign = 1
        extremum_index = int(np.argmax(deviations))

    span = float(np.max(x) - np.min(x))
    spacing = float(np.median(np.diff(x)))
    gamma_guess = LORENTZIAN_TIME_BANDWIDTH / qubit_drive_duration
    gamma_guess = float(np.clip(gamma_guess, spacing, span / 2))
    amplitude_scale = 2 * float(np.max(y) - np.min(y)) + 1e-12

    amplitude_guess = float(deviations[extremum_index])
    if expected_amplitude_sign > 0:
        amplitude_guess = max(amplitude_guess, 1e-12)
        amplitude_bounds = (0.0, amplitude_scale)
    else:
        amplitude_guess = min(amplitude_guess, -1e-12)
        amplitude_bounds = (-amplitude_scale, 0.0)
    p0 = (amplitude_guess, float(x[extremum_index]), gamma_guess, baseline)
    bounds = (
        (amplitude_bounds[0], float(np.min(x)), 0.0, -np.inf),
        (amplitude_bounds[1], float(np.max(x)), np.inf, np.inf),
    )
    try:
        fit_result = fitting.fit_lorentzian(
            x=x,
            y=y,
            p0=p0,
            bounds=bounds,
            plot=False,
        )
        valid, reason = check_lorentzian_fit(
            fit_result,
            x=x,
            expected_amplitude_sign=expected_amplitude_sign,
        )
    except Exception as exc:
        fit_result = None
        valid = False
        reason = f"fit raised an exception ({exc})"

    if fit_result is not None:
        f0 = float(fit_result.get("f0", np.nan))
        f0_err = float(fit_result.get("f0_err", np.nan))
        gamma = float(fit_result.get("gamma", np.nan))
        r2 = float(fit_result.get("r2", np.nan))
    else:
        f0 = f0_err = gamma = r2 = np.nan
    if not valid:
        print(f"[WARN] invalid Lorentzian fit (state={qubit_initial_state}): {reason}")

    return {
        "detuning_range": detuning_range,
        "frequency_range": nominal_frequency + detuning_range,
        "normalized": normalized,
        "fit_result": fit_result,
        "f0": f0,
        "f0_err": f0_err,
        "gamma": gamma,
        "r2": r2,
        "valid": valid,
    }


def derive_ac_stark_shift(
    *,
    f0: NDArray[np.float64],
    f0_err: NDArray[np.float64],
    fit_valid: NDArray[np.bool_],
    reference_f0: NDArray[np.float64],
    reference_f0_err: NDArray[np.float64],
    reference_valid: NDArray[np.bool_],
    dispersive_shift: float | None,
) -> dict[str, Any]:
    """
    Convert fitted resonances into AC Stark shifts and photon numbers.

    ``shift = f0_driven - f0_no_drive`` removes calibration offsets and
    offset-dependent effects (e.g. T1 decay of the |e> preparation) from
    the shift. With ``dispersive_shift`` (signed chi in GHz, convention
    ``delta_f_qubit = 2 * chi * n_photon``) the shift is converted to the
    mean intracavity photon number ``n = shift / (2 * chi)``.

    All array arguments must broadcast against each other; callers
    with an extra sweep axis broadcast the reference arrays themselves.
    """
    shift_mask = fit_valid & reference_valid
    ac_stark_shift = np.where(shift_mask, f0 - reference_f0, np.nan)
    ac_stark_shift_error = np.where(
        shift_mask,
        np.sqrt(f0_err**2 + reference_f0_err**2),
        np.nan,
    )

    if dispersive_shift is not None:
        photon_number = ac_stark_shift / (2 * dispersive_shift)
        photon_number_error = ac_stark_shift_error / (2 * abs(dispersive_shift))
        photon_number_valid = (
            shift_mask
            & np.isfinite(photon_number)
            & ~(photon_number + 3 * photon_number_error < 0)
        )
        if np.any(shift_mask & np.isfinite(photon_number) & ~photon_number_valid):
            print(
                "[WARN] some inferred photon numbers are significantly "
                "negative; check the sign convention of `dispersive_shift` "
                "(signed chi with delta_f = 2 * chi * n)."
            )
    else:
        photon_number = None
        photon_number_error = None
        photon_number_valid = None

    return {
        "ac_stark_shift": ac_stark_shift,
        "ac_stark_shift_error": ac_stark_shift_error,
        "photon_number": photon_number,
        "photon_number_error": photon_number_error,
        "photon_number_valid": photon_number_valid,
    }
