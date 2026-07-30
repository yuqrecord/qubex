"""
Post-selected quantum-efficiency measurement.

This module measures the quantum efficiency of the dispersive readout chain
following arXiv:1711.05336 (Bultink et al.), combining

1. a readout-induced dephasing measurement: a Ramsey sequence with a readout
   drive injected between the two pi/2 pulses,

   ``x90 -- [readout drive + wait] -- x90(phase) -- readout``

   where the injected drive is followed by a wait time so that the resonator
   rings down before the second pi/2 pulse, and

2. a post-selected single-shot SNR measurement reusing
   :mod:`.assignment_fidelity`; the identical "readout drive + wait" waveform
   serves as both the prior (post-selection) and posterior readout pulse, so
   the capture window includes the ring-down tail and the classifier trained
   on posterior data transfers to the prior readout.

Both measurements sweep the readout amplitude ``epsilon`` and the results are
combined through the fitted scalings

- ``|rho01|(epsilon) = b * exp(-epsilon^2 / (2 * sigma_m^2))``
- ``SNR(epsilon) = a * epsilon``
- ``eta = a^2 * sigma_m^2``

The SNR convention is the one used in :mod:`.assignment_fidelity`:

    ``SNR = |mu_e - mu_g| / sqrt(var_g + var_e)``

evaluated on the post-selected test set with the moment-based estimator.
Note that :mod:`qubex.contrib.experiment.quantum_efficiency_measurement`
uses ``SNR = |mu_e - mu_g| / (0.5 * (sigma_g + sigma_e))`` and accordingly
``eta = a^2 * sigma_m^2 / 2``; for equal state variances both conventions
give the same ``eta``. The pointwise diagnostic in this module is
``eta_i = SNR_i^2 / (2 * beta_m_i)`` with ``beta_m_i = -log(rho01_i / b)``
(not the ``SNR^2 / (4 * beta_m)`` expression of the other convention).

Differences from ``quantum_efficiency_measurement.py``:

- the injected readout drive is followed by an explicit ring-down wait time
  (``waittime_after_readout_drive``), and the identical padded waveform is
  used as the SNR readout pulse;
- the SNR is computed after post-selection on the prior readout instead of
  absorbing the thermal population into a double-Gaussian fit;
- the readout drive duration and ramptime can be overridden (``None`` keeps
  the calibrated values).

Caveat — post-selection at low amplitude: the prior readout necessarily uses
the same weak pulse as the posterior readout, so as ``epsilon -> 0`` the
prior classification approaches chance and post-selection becomes
ineffective. No automatic filtering is applied; exclude unreliable
low-amplitude points via ``readout_amplitude_range`` and inspect the
``post_selection_rate`` figure and the retained shot counts.

The error of ``eta`` is propagated as
``eta_err = sqrt((2*a*sigma_m^2*a_err)^2 + (2*a^2*sigma_m*sigma_m_err)^2)``,
treating the two fits as uncorrelated.
"""

import numpy as np
import plotly.graph_objects as go
from qxpulse import Waveform
from scipy.optimize import curve_fit

import qubex as qx
import qubex.visualization as viz
from qubex.experiment.experiment_constants import SAMPLING_PERIOD
from qubex.experiment.models import Result

from .assignment_fidelity import (
    analyze_post_selected_assignment_fidelity,
    run_post_selected_assignment_fidelity,
    _validate_readout_pulse_start_position as _validate_readout_pulse_start_position_snr,
)

__all__ = [
    "analyze_post_selected_quantum_efficiency",
    "display_post_selected_quantum_efficiency",
    "plot_post_selected_quantum_efficiency",
    "post_selected_quantum_efficiency_experiment",
    "run_post_selected_snr",
    "run_readout_induced_dephasing",
]

SNR_CONVENTION = "abs(mu_e - mu_g) / sqrt(var_g + var_e)"
GRID_TOLERANCE = 1e-9
MIN_READOUT_AMPLITUDES = 3
MIN_RAMSEY_PHASES = 8
MIN_RAMSEY_PHASE_SPAN = 1.5 * np.pi


def _validate_readout_pulse_start_position(
    exp: qx.Experiment,
    *,
    target: str,
    readout_duration: float | None,
    waittime_after_readout_drive: float,
    waittime_after_capture_window: float,
    waittime_after_ctrl_pulse: float,
    capture_sampling_period: float,
):
    _validate_readout_pulse_start_position_snr(
        exp,
        target=target,
        readout_duration=readout_duration,
        waittime_after_readout_pulse=waittime_after_readout_drive
        + waittime_after_capture_window,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        capture_sampling_period=capture_sampling_period,
    )
    qubit = exp.ctx.qubits[target].label
    drag_hpi_pulse_duration = exp.drag_hpi_pulse[qubit].duration
    readout_drive_start = drag_hpi_pulse_duration + waittime_after_ctrl_pulse
    if readout_drive_start % capture_sampling_period != 0:
        raise ValueError(
            f"The readout drive start time ({readout_drive_start} ns) must be a "
            f"multiple of the capture sampling period ({capture_sampling_period} ns)."
        )
    final_readout_start = (
        drag_hpi_pulse_duration
        + waittime_after_ctrl_pulse
        + readout_duration
        + waittime_after_readout_drive
        + waittime_after_capture_window
        + drag_hpi_pulse_duration
        + waittime_after_ctrl_pulse
    )
    if final_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The final readout start time ({final_readout_start} ns) must be a "
            f"multiple of the capture sampling period ({capture_sampling_period} ns)."
        )


def _validate_time_on_grid(value, *, name, minimum=0.0):
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be a finite number.")
    if value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}.")
    ratio = value / SAMPLING_PERIOD
    if abs(ratio - round(ratio)) > GRID_TOLERANCE:
        raise ValueError(f"`{name}` must be a multiple of {SAMPLING_PERIOD} ns.")


def _validate_optional_time_on_grid(value, *, name, minimum=0.0):
    if value is None:
        return
    _validate_time_on_grid(value, name=name, minimum=minimum)


def _validate_positive_int(value, *, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}.")


def _validate_readout_amplitude(value, *, name="readout_amplitude"):
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be a finite number.")
    if value <= 0:
        raise ValueError(f"`{name}` must be > 0.")


def _validate_shot_interval(value):
    if not np.isfinite(value):
        raise ValueError("`shot_interval` must be a finite number.")
    if value <= 0:
        raise ValueError("`shot_interval` must be > 0.")


def _validate_ramsey_phase_range(phases):
    if phases.ndim != 1:
        raise ValueError("`ramsey_phase_range` must be a 1D array.")
    if not np.all(np.isfinite(phases)):
        raise ValueError("`ramsey_phase_range` must contain only finite values.")
    if len(np.unique(phases)) < MIN_RAMSEY_PHASES:
        raise ValueError(
            f"`ramsey_phase_range` must contain at least {MIN_RAMSEY_PHASES} "
            "distinct phases."
        )
    span = float(np.max(phases) - np.min(phases))
    if span < MIN_RAMSEY_PHASE_SPAN:
        raise ValueError(
            f"`ramsey_phase_range` must span at least {MIN_RAMSEY_PHASE_SPAN:.6g} rad."
        )


def _validate_readout_amplitude_range(amplitudes):
    if amplitudes.ndim != 1:
        raise ValueError("`readout_amplitude_range` must be a scalar or a 1D array.")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("`readout_amplitude_range` must contain only finite values.")
    if np.any(amplitudes <= 0):
        raise ValueError("`readout_amplitude_range` values must be > 0.")
    if len(amplitudes) < MIN_READOUT_AMPLITUDES:
        raise ValueError(
            "`readout_amplitude_range` must contain at least "
            f"{MIN_READOUT_AMPLITUDES} amplitudes."
        )


def _build_readout_pulse(
    exp: qx.Experiment,
    *,
    target: str,
    readout_amplitude: float,
    readout_duration: float | None,
    readout_ramptime: float | None,
    waittime_after_readout_drive: float,
) -> Waveform:
    """
    Build the shared "readout drive + ring-down wait" waveform.

    The same waveform is used as the injected drive of the dephasing sequence
    and as the prior/posterior readout pulse of the SNR measurement. `None`
    for `readout_duration` / `readout_ramptime` keeps the calibrated values.
    The wait time is appended as zero padding; `add_readout_post_margin`
    appends `exp.readout_post_margin` after the wait (the margin is handled
    here only, so the sibling run function is called with
    `add_readout_post_margin=False`).
    """
    pulse = exp.readout(
        target,
        amplitude=float(readout_amplitude),
        duration=readout_duration,
        ramptime=readout_ramptime,
    )
    pulse = pulse.padded(pulse.duration + waittime_after_readout_drive)
    return pulse


def run_readout_induced_dephasing(
    exp: qx.Experiment,
    *,
    target: str,
    readout_amplitude: float,
    readout_duration: float | None,
    readout_ramptime: float | None,
    ramsey_phase_range,
    waittime_after_readout_drive: float,
    waittime_after_ctrl_pulse: float,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
):
    """
    Acquire one readout-induced-dephasing Ramsey fringe at one readout amplitude.

    For each phase in `ramsey_phase_range` the sequence is

        x90 -- [readout drive + wait] -- x90(phase) -- wait -- calibrated readout

    built from the DRAG half-pi pulse. The injected drive generates a capture
    window that is ignored by the analysis; the last capture is the final
    readout. With `plot_pulse_schedule=True` only the schedule of the first
    phase is plotted.
    """
    _validate_readout_amplitude(readout_amplitude)
    _validate_optional_time_on_grid(
        readout_duration,
        name="readout_duration",
        minimum=SAMPLING_PERIOD,
    )
    _validate_optional_time_on_grid(readout_ramptime, name="readout_ramptime")
    phase_array = np.asarray(ramsey_phase_range, dtype=np.float64)
    _validate_ramsey_phase_range(phase_array)
    _validate_time_on_grid(
        waittime_after_readout_drive,
        name="waittime_after_readout_drive",
    )
    _validate_positive_int(n_shots, name="n_shots")
    _validate_shot_interval(shot_interval)

    qubit = exp.ctx.qubits[target].label
    resonator = exp.ctx.resonators[target].label
    if qubit not in exp.drag_hpi_pulse:
        raise ValueError(f"DRAG half-pi pulse is not calibrated for {qubit}.")
    hpi = exp.drag_hpi_pulse[qubit]
    readout_pulse = _build_readout_pulse(
        exp,
        target=target,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        readout_ramptime=readout_ramptime,
        waittime_after_readout_drive=waittime_after_readout_drive,
    )
    final_readout_pulse = exp.readout(target)

    results = []
    for index, phase in enumerate(phase_array):
        schedule = qx.PulseSchedule()
        with schedule as s:
            s.add(qubit, hpi)
            s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
            s.barrier()
            s.add(resonator, readout_pulse)
            s.barrier()
            s.add(qubit, hpi.shifted(float(phase)))
            s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
            s.barrier()
            s.add(resonator, final_readout_pulse)
        if plot_pulse_schedule and index == 0:
            schedule.plot(
                title=(
                    f"Readout-induced dephasing : {target} "
                    f"(amplitude={float(readout_amplitude):.6g}, first phase)"
                )
            )
        results.append(
            exp.execute(
                schedule=schedule,
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )
        )
    return {
        "target": target,
        "readout_amplitude": float(readout_amplitude),
        "readout_pulse": readout_pulse,
        "ramsey_phase_range": phase_array,
        "results": results,
    }


def run_post_selected_snr(
    exp: qx.Experiment,
    *,
    target: str,
    readout_amplitude: float,
    readout_duration: float | None,
    readout_ramptime: float | None,
    waittime_after_readout_drive: float,
    waittime_after_capture_window: float,
    waittime_after_ctrl_pulse: float,
    n_train: int,
    n_test: int,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
):
    """
    Acquire the post-selected single-shot data at one readout amplitude.

    Delegates to `run_post_selected_assignment_fidelity` with the shared
    "readout drive + wait" waveform as the readout pulse (prior and
    posterior). The post margin is already included by `_build_readout_pulse`
    when requested, so the sibling is called with
    `add_readout_post_margin=False`.
    """
    _validate_readout_amplitude(readout_amplitude)
    _validate_optional_time_on_grid(
        readout_duration,
        name="readout_duration",
        minimum=SAMPLING_PERIOD,
    )
    _validate_optional_time_on_grid(readout_ramptime, name="readout_ramptime")
    _validate_time_on_grid(
        waittime_after_readout_drive,
        name="waittime_after_readout_drive",
    )
    _validate_time_on_grid(
        waittime_after_capture_window, name="waittime_after_capture_window"
    )
    _validate_time_on_grid(waittime_after_ctrl_pulse, name="waittime_after_ctrl_pulse")
    _validate_positive_int(n_train, name="n_train", minimum=2)
    _validate_positive_int(n_test, name="n_test", minimum=2)
    _validate_shot_interval(shot_interval)

    readout_pulse = _build_readout_pulse(
        exp,
        target=target,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        readout_ramptime=readout_ramptime,
        waittime_after_readout_drive=waittime_after_readout_drive,
    )
    run_result = run_post_selected_assignment_fidelity(
        exp,
        target=target,
        readout_pulse=readout_pulse,
        waittime_after_readout_pulse=waittime_after_capture_window,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        pump_amplitude=pump_amplitude,
        shot_interval=shot_interval,
        n_shots=n_train + n_test,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    return {
        "target": target,
        "readout_amplitude": float(readout_amplitude),
        "readout_pulse": readout_pulse,
        "result": run_result,
    }


def _validate_run_result_consistency(
    *,
    target,
    dephasing_run_results,
    snr_run_results,
):
    if len(dephasing_run_results) != len(snr_run_results):
        raise ValueError(
            "`dephasing_run_results` and `snr_run_results` must have the same length."
        )
    if len(dephasing_run_results) < MIN_READOUT_AMPLITUDES:
        raise ValueError(
            f"At least {MIN_READOUT_AMPLITUDES} readout amplitudes are required."
        )
    reference_phases = np.asarray(
        dephasing_run_results[0]["ramsey_phase_range"], dtype=np.float64
    )
    durations = []
    for dephasing_run, snr_run in zip(
        dephasing_run_results, snr_run_results, strict=True
    ):
        if dephasing_run["target"] != target or snr_run["target"] != target:
            raise ValueError("All run results must belong to the same target.")
        if float(dephasing_run["readout_amplitude"]) != float(
            snr_run["readout_amplitude"]
        ):
            raise ValueError(
                "Dephasing and SNR run results must use identical readout "
                "amplitudes in identical order."
            )
        phases = np.asarray(dephasing_run["ramsey_phase_range"], dtype=np.float64)
        if not np.array_equal(phases, reference_phases):
            raise ValueError(
                "All dephasing run results must share the same `ramsey_phase_range`."
            )
        durations.append(float(dephasing_run["readout_pulse"].duration))
        durations.append(float(snr_run["readout_pulse"].duration))
    if len(set(durations)) != 1:
        raise ValueError(
            "All readout pulses must have the same duration; the Ramsey "
            "window must not change with the readout amplitude."
        )


def _extract_excited_probabilities(*, target, dephasing_run_result):
    phases = dephasing_run_result["ramsey_phase_range"]
    results = dephasing_run_result["results"]
    if len(results) != len(phases):
        raise ValueError("Each phase must have exactly one measurement result.")
    excited_probabilities = []
    for measurement in results:
        if target not in measurement.data:
            raise ValueError(f"Measurement data for `{target}` not found.")
        readouts = measurement.data[target]
        if len(readouts) != 2:
            raise ValueError(
                "Expected exactly 2 capture windows (injected drive + final "
                f"readout) but found {len(readouts)}."
            )
        probabilities = np.asarray(readouts[-1].probabilities, dtype=np.float64)
        if probabilities.size < 2:
            raise ValueError(
                "The final readout must provide at least two state probabilities."
            )
        excited_probabilities.append(float(probabilities[1]))
    return np.asarray(excited_probabilities, dtype=np.float64)


def _fit_ramsey_fringe(*, phases, excited_probabilities):
    """
    Fit one Ramsey fringe with `sigma_z(phi) = c + A*cos(phi) + B*sin(phi)`.

    Returns `rho01 = 0.5*sqrt(A^2+B^2)` with its propagated error,
    `phi0 = atan2(-B, A)`, the offset `c` and fit diagnostics.
    """
    p_e = np.asarray(excited_probabilities, dtype=np.float64)
    if p_e.shape != phases.shape:
        raise ValueError(
            "`excited_probabilities` must have the same shape as `phases`."
        )
    if len(phases) < 4:
        raise ValueError("At least 4 phase points are required for fringe fitting.")
    sigma_z = 1.0 - 2.0 * p_e
    design = np.column_stack([np.ones_like(phases), np.cos(phases), np.sin(phases)])
    solution, _, rank, _ = np.linalg.lstsq(design, sigma_z, rcond=None)
    if rank < 3:
        raise ValueError(
            "Ramsey fringe fitting is rank-deficient; check the phase range."
        )
    offset, cosine, sine = (float(value) for value in solution)
    sigma_z_fit = design @ solution
    residuals = sigma_z - sigma_z_fit
    residual_sum = float(np.sum(residuals**2))
    degrees_of_freedom = len(phases) - 3
    residual_variance = residual_sum / degrees_of_freedom
    covariance = residual_variance * np.linalg.inv(design.T @ design)

    rho01 = 0.5 * float(np.hypot(cosine, sine))
    if rho01 > 0:
        jacobian = np.array([cosine, sine], dtype=np.float64) / (4.0 * rho01)
        rho01_variance = float(jacobian @ covariance[1:, 1:] @ jacobian)
        rho01_err = float(np.sqrt(max(rho01_variance, 0.0)))
    else:
        rho01_err = float("nan")
    total_sum = float(np.sum((sigma_z - np.mean(sigma_z)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return {
        "rho01": rho01,
        "rho01_err": rho01_err,
        "phi0": float(np.arctan2(-sine, cosine)),
        "offset": offset,
        "cosine": cosine,
        "sine": sine,
        "covariance": covariance,
        "r_squared": float(r_squared),
        "excited_probabilities": p_e,
        "excited_probabilities_fit": 0.5 * (1.0 - sigma_z_fit),
        "sigma_z": sigma_z,
        "sigma_z_fit": sigma_z_fit,
        "p_e_min": float(np.min(p_e)),
        "p_e_max": float(np.max(p_e)),
    }


def _fit_dephasing_envelope(*, readout_amplitudes, rho01_values):
    """Fit `rho01(epsilon) = b * exp(-epsilon^2 / (2 * sigma_m^2))`."""
    if np.any(~np.isfinite(rho01_values)) or np.any(rho01_values <= 0):
        raise ValueError(
            "`rho01_values` must be finite and strictly positive for envelope fitting."
        )

    def model(epsilon, b, sigma_m):
        return b * np.exp(-(epsilon**2) / (2.0 * sigma_m**2))

    initial_b = float(np.max(rho01_values))
    nearest_index = int(np.argmin(np.abs(rho01_values - initial_b / np.e)))
    initial_sigma = float(abs(readout_amplitudes[nearest_index]))
    if initial_sigma <= 0:
        initial_sigma = max(float(np.max(readout_amplitudes)) / 2.0, 1e-6)
    try:
        popt, pcov = curve_fit(
            model,
            readout_amplitudes,
            rho01_values,
            p0=[initial_b, initial_sigma],
            bounds=([0.0, 1e-12], [np.inf, np.inf]),
            maxfev=20_000,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Measurement-induced dephasing fitting failed: {exc}"
        ) from exc
    b, sigma_m = (float(value) for value in popt)
    b_err, sigma_m_err = (float(value) for value in np.sqrt(np.diag(pcov)))
    if not all(np.isfinite([b, sigma_m, b_err, sigma_m_err])):
        raise RuntimeError(
            "Measurement-induced dephasing fitting returned non-finite "
            "parameters or covariance."
        )
    fit_amplitudes = np.linspace(0.0, float(np.max(readout_amplitudes)), 500)
    residuals = rho01_values - model(readout_amplitudes, b, sigma_m)
    total_sum = float(np.sum((rho01_values - np.mean(rho01_values)) ** 2))
    residual_sum = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return {
        "b": b,
        "b_err": b_err,
        "sigma_m": sigma_m,
        "sigma_m_err": sigma_m_err,
        "covariance": np.asarray(pcov, dtype=np.float64),
        "fit_amplitudes": fit_amplitudes,
        "rho01_fit": model(fit_amplitudes, b, sigma_m),
        "residuals": residuals,
        "r_squared": float(r_squared),
    }


def _fit_snr_scaling(*, readout_amplitudes, snr_values):
    """Fit the origin-constrained scaling `SNR(epsilon) = a * epsilon`."""
    if np.any(~np.isfinite(snr_values)):
        raise ValueError("`snr_values` must contain only finite values.")
    denominator = float(np.sum(readout_amplitudes**2))
    if denominator <= 0:
        raise ValueError("`readout_amplitudes` must contain nonzero values.")
    a = float(np.sum(readout_amplitudes * snr_values) / denominator)
    residuals = snr_values - a * readout_amplitudes
    degrees_of_freedom = len(readout_amplitudes) - 1
    if degrees_of_freedom < 1:
        raise ValueError("At least 2 amplitudes are required for SNR scaling fitting.")
    residual_variance = float(np.sum(residuals**2)) / degrees_of_freedom
    a_err = float(np.sqrt(residual_variance / denominator))
    if not all(np.isfinite([a, a_err])):
        raise RuntimeError("SNR scaling fitting returned non-finite parameters.")
    total_sum = float(np.sum((snr_values - np.mean(snr_values)) ** 2))
    residual_sum = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    fit_amplitudes = np.linspace(0.0, float(np.max(readout_amplitudes)), 500)
    return {
        "a": a,
        "a_err": a_err,
        "fit_amplitudes": fit_amplitudes,
        "snr_fit": a * fit_amplitudes,
        "residuals": residuals,
        "r_squared": float(r_squared),
    }


def analyze_post_selected_quantum_efficiency(
    *,
    target: str,
    dephasing_run_results,
    snr_run_results,
    n_train: int,
    n_test: int,
    n_max_epoch: int,
):
    """
    Analyze the dephasing and SNR run results and extract the quantum efficiency.

    `dephasing_run_results` and `snr_run_results` are lists of the dicts
    returned by `run_readout_induced_dephasing` and `run_post_selected_snr`,
    one entry per readout amplitude (identical amplitudes in identical order).
    The eta fit uses the moment-based post-selected SNR on the test set.
    """
    _validate_positive_int(n_max_epoch, name="n_max_epoch")
    _validate_run_result_consistency(
        target=target,
        dephasing_run_results=dephasing_run_results,
        snr_run_results=snr_run_results,
    )
    amplitudes = np.asarray(
        [float(run["readout_amplitude"]) for run in dephasing_run_results],
        dtype=np.float64,
    )
    phases = np.asarray(
        dephasing_run_results[0]["ramsey_phase_range"], dtype=np.float64
    )

    fringe_summaries = []
    excited_probabilities = np.empty((len(amplitudes), len(phases)), dtype=np.float64)
    for index, dephasing_run in enumerate(dephasing_run_results):
        p_e = _extract_excited_probabilities(
            target=target,
            dephasing_run_result=dephasing_run,
        )
        excited_probabilities[index] = p_e
        fringe_summaries.append(
            _fit_ramsey_fringe(phases=phases, excited_probabilities=p_e)
        )
    rho01 = np.asarray(
        [summary["rho01"] for summary in fringe_summaries], dtype=np.float64
    )
    rho01_err = np.asarray(
        [summary["rho01_err"] for summary in fringe_summaries], dtype=np.float64
    )
    phi0 = np.asarray(
        [summary["phi0"] for summary in fringe_summaries], dtype=np.float64
    )
    fringe_offset = np.asarray(
        [summary["offset"] for summary in fringe_summaries], dtype=np.float64
    )
    envelope_fit = _fit_dephasing_envelope(
        readout_amplitudes=amplitudes,
        rho01_values=rho01,
    )
    beta_m = -np.log(rho01 / envelope_fit["b"])

    snr_analyses = [
        analyze_post_selected_assignment_fidelity(
            target=target,
            run_result=snr_run["result"],
            n_train=n_train,
            n_test=n_test,
            n_max_epoch=n_max_epoch,
            verbose=False,
        )
        for snr_run in snr_run_results
    ]
    snr_values = {
        w: {
            d: np.asarray(
                [analysis["snr"][w][d] for analysis in snr_analyses],
                dtype=np.float64,
            )
            for d in ["train", "test"]
        }
        for w in ["before_post_selection", "after_post_selection"]
    }
    gaussian_fit_snr_after_test = [
        analysis["weighted_signal_histogram"]["gaussian_fit_snr"][
            "after_post_selection"
        ]["test"]
        for analysis in snr_analyses
    ]
    post_selection_rate = {
        s: {
            d: np.asarray(
                [analysis["post_selection_rate"][s][d] for analysis in snr_analyses],
                dtype=np.float64,
            )
            for d in ["train", "test"]
        }
        for s in ["g", "e"]
    }
    retained_shots = {
        s: {
            d: np.asarray(
                [
                    int(np.sum(analysis["post_selection"][s][d]))
                    for analysis in snr_analyses
                ],
                dtype=np.int64,
            )
            for d in ["train", "test"]
        }
        for s in ["g", "e"]
    }
    snr_fit = _fit_snr_scaling(
        readout_amplitudes=amplitudes,
        snr_values=snr_values["after_post_selection"]["test"],
    )

    a = float(snr_fit["a"])
    a_err = float(snr_fit["a_err"])
    sigma_m = float(envelope_fit["sigma_m"])
    sigma_m_err = float(envelope_fit["sigma_m_err"])
    quantum_efficiency = a**2 * sigma_m**2
    quantum_efficiency_err = float(
        np.sqrt(
            (2.0 * a * sigma_m**2 * a_err) ** 2
            + (2.0 * a**2 * sigma_m * sigma_m_err) ** 2
        )
    )
    pointwise_quantum_efficiency = np.full_like(amplitudes, np.nan)
    valid = beta_m > 0
    pointwise_quantum_efficiency[valid] = snr_values["after_post_selection"]["test"][
        valid
    ] ** 2 / (2.0 * beta_m[valid])

    return {
        "target": target,
        "snr_convention": SNR_CONVENTION,
        "quantum_efficiency": float(quantum_efficiency),
        "quantum_efficiency_err": quantum_efficiency_err,
        "pointwise_quantum_efficiency": pointwise_quantum_efficiency,
        "readout_amplitudes": amplitudes,
        "dephasing": {
            "ramsey_phase_range": phases,
            "excited_probabilities": excited_probabilities,
            "fringe": fringe_summaries,
            "rho01": rho01,
            "rho01_err": rho01_err,
            "phi0": phi0,
            "phi0_unwrapped": np.unwrap(phi0),
            "fringe_offset": fringe_offset,
            "beta_m": beta_m,
            "fit": envelope_fit,
        },
        "snr": {
            "analysis": snr_analyses,
            "snr": snr_values,
            "gaussian_fit_snr_after_test": gaussian_fit_snr_after_test,
            "post_selection_rate": post_selection_rate,
            "retained_shots": retained_shots,
            "fit": snr_fit,
        },
    }


def _title_with_subtitle(text, subtitle):
    return {
        "text": text,
        "subtitle": {
            "text": subtitle,
            "font": {"size": 11, "family": "monospace"},
        },
    }


def _annotate(figure, text):
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=text,
        bgcolor="rgba(255, 255, 255, 0.8)",
        showarrow=False,
    )


def _pi_ticks(phases):
    tick_start = int(np.floor(np.min(phases) / np.pi))
    tick_stop = int(np.ceil(np.max(phases) / np.pi))
    tickvals = [index * np.pi for index in range(tick_start, tick_stop + 1)]
    ticktext = []
    for index in range(tick_start, tick_stop + 1):
        if index == 0:
            ticktext.append("0")
        elif index == 1:
            ticktext.append("π")
        elif index == -1:
            ticktext.append("-π")
        else:
            ticktext.append(f"{index}π")
    return tickvals, ticktext


def plot_post_selected_quantum_efficiency(*, analysis_result):
    """Build the plotly figures for one analysis result."""
    target = analysis_result["target"]
    amplitudes = analysis_result["readout_amplitudes"]
    dephasing = analysis_result["dephasing"]
    snr = analysis_result["snr"]
    envelope_fit = dephasing["fit"]
    snr_fit = snr["fit"]
    eta = analysis_result["quantum_efficiency"]
    eta_err = analysis_result["quantum_efficiency_err"]
    figures = {}

    overview = viz.make_figure()
    overview.set_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    overview.add_trace(
        go.Scatter(
            x=amplitudes,
            y=dephasing["rho01"],
            error_y={"type": "data", "array": dephasing["rho01_err"]},
            mode="markers",
            name="ρ<sub>01</sub> data",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    overview.add_trace(
        go.Scatter(
            x=envelope_fit["fit_amplitudes"],
            y=envelope_fit["rho01_fit"],
            mode="lines",
            name="ρ<sub>01</sub> fit",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    overview.add_trace(
        go.Scatter(
            x=amplitudes,
            y=snr["snr"]["after_post_selection"]["test"],
            mode="markers",
            name="SNR data",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    overview.add_trace(
        go.Scatter(
            x=snr_fit["fit_amplitudes"],
            y=snr_fit["snr_fit"],
            mode="lines",
            name="SNR fit",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    overview.update_xaxes(title_text="Readout amplitude", row=1, col=1)
    overview.update_yaxes(
        title_text="|ρ<sub>01</sub>|", row=1, col=1, secondary_y=False
    )
    overview.update_yaxes(
        title_text="Signal-to-noise ratio", row=1, col=1, secondary_y=True
    )
    overview.update_layout(
        title=_title_with_subtitle(
            f"Post-selected quantum efficiency : {target}",
            "η = a²σ<sub>m</sub>², SNR after post-selection (test set)",
        ),
        width=600,
        height=400,
        showlegend=True,
    )
    _annotate(overview, f"η = {eta:.3g} ± {eta_err:.2g}")
    figures["overview"] = overview

    dephasing_figure = viz.make_figure()
    dephasing_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=dephasing["rho01"],
            error_y={"type": "data", "array": dephasing["rho01_err"]},
            mode="markers",
            name="data",
        )
    )
    dephasing_figure.add_trace(
        go.Scatter(
            x=envelope_fit["fit_amplitudes"],
            y=envelope_fit["rho01_fit"],
            mode="lines",
            name="fit",
        )
    )
    dephasing_figure.update_layout(
        title=f"Readout-induced dephasing : {target}",
        xaxis_title="Readout amplitude",
        yaxis_title="|ρ<sub>01</sub>|",
        width=600,
        height=400,
        showlegend=True,
    )
    _annotate(
        dephasing_figure,
        (
            f"σ<sub>m</sub> = {envelope_fit['sigma_m']:.3g} "
            f"± {envelope_fit['sigma_m_err']:.2g}"
        ),
    )
    figures["dephasing"] = dephasing_figure

    snr_figure = viz.make_figure()
    snr_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=snr["snr"]["before_post_selection"]["test"],
            mode="markers",
            name="before post-selection",
        )
    )
    snr_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=snr["snr"]["after_post_selection"]["test"],
            mode="markers",
            name="after post-selection",
        )
    )
    snr_figure.add_trace(
        go.Scatter(
            x=snr_fit["fit_amplitudes"],
            y=snr_fit["snr_fit"],
            mode="lines",
            name="fit (after)",
        )
    )
    snr_figure.update_layout(
        title=_title_with_subtitle(
            f"Post-selected readout SNR : {target}",
            "test set",
        ),
        xaxis_title="Readout amplitude",
        yaxis_title="Signal-to-noise ratio",
        width=600,
        height=400,
        showlegend=True,
    )
    _annotate(snr_figure, f"a = {snr_fit['a']:.3g} ± {snr_fit['a_err']:.2g}")
    figures["snr"] = snr_figure

    rate_figure = viz.make_figure()
    rate_figure.set_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    for state in ["g", "e"]:
        rate_figure.add_trace(
            go.Scatter(
                x=amplitudes,
                y=snr["post_selection_rate"][state]["test"],
                mode="lines+markers",
                name=f"rate |{state}⟩",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        rate_figure.add_trace(
            go.Scatter(
                x=amplitudes,
                y=snr["retained_shots"][state]["test"],
                mode="lines+markers",
                line={"dash": "dot"},
                name=f"retained |{state}⟩",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    rate_figure.update_xaxes(title_text="Readout amplitude", row=1, col=1)
    rate_figure.update_yaxes(
        title_text="Post-selection rate", row=1, col=1, secondary_y=False
    )
    rate_figure.update_yaxes(
        title_text="Retained shots", row=1, col=1, secondary_y=True
    )
    rate_figure.update_layout(
        title=_title_with_subtitle(
            f"Post-selection diagnostics : {target}",
            "test set; near-chance rates indicate ineffective post-selection",
        ),
        width=600,
        height=400,
        showlegend=True,
    )
    figures["post_selection_rate"] = rate_figure

    pointwise_figure = viz.make_figure()
    pointwise_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=analysis_result["pointwise_quantum_efficiency"],
            mode="markers",
            name="pointwise η",
        )
    )
    pointwise_figure.add_hline(y=eta, line_dash="dash", line_color="gray")
    pointwise_figure.update_layout(
        title=_title_with_subtitle(
            f"Pointwise quantum efficiency : {target}",
            "diagnostic only; η_i = SNR_i²/(2β_m,i), NaN where β_m ≤ 0",
        ),
        xaxis_title="Readout amplitude",
        yaxis_title="η",
        width=600,
        height=400,
        showlegend=True,
    )
    _annotate(pointwise_figure, f"fit η = {eta:.3g}")
    figures["pointwise_eta"] = pointwise_figure

    phi0_figure = viz.make_figure()
    phi0_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=dephasing["phi0"],
            mode="markers",
            name="φ₀ (wrapped)",
        )
    )
    phi0_figure.add_trace(
        go.Scatter(
            x=amplitudes,
            y=dephasing["phi0_unwrapped"],
            mode="lines+markers",
            name="φ₀ (unwrapped)",
        )
    )
    phi0_figure.update_layout(
        title=_title_with_subtitle(
            f"Ramsey fringe phase : {target}",
            "AC-Stark phase diagnostic",
        ),
        xaxis_title="Readout amplitude",
        yaxis_title="φ₀ (rad)",
        width=600,
        height=400,
        showlegend=True,
    )
    figures["phi0"] = phi0_figure

    phases = dephasing["ramsey_phase_range"]
    phase_fine = np.linspace(float(np.min(phases)), float(np.max(phases)), 500)
    tickvals, ticktext = _pi_ticks(phases)
    for index, summary in enumerate(dephasing["fringe"]):
        fringe_figure = viz.make_figure()
        fringe_figure.add_trace(
            go.Scatter(
                x=phases,
                y=summary["excited_probabilities"],
                mode="markers",
                name="data",
            )
        )
        sigma_z_fine = (
            summary["offset"]
            + summary["cosine"] * np.cos(phase_fine)
            + summary["sine"] * np.sin(phase_fine)
        )
        fringe_figure.add_trace(
            go.Scatter(
                x=phase_fine,
                y=0.5 * (1.0 - sigma_z_fine),
                mode="lines",
                name="fit",
            )
        )
        fringe_figure.update_xaxes(tickvals=tickvals, ticktext=ticktext)
        fringe_figure.update_layout(
            title=_title_with_subtitle(
                f"Ramsey fringe : {target}",
                f"readout_amplitude={amplitudes[index]:.6g}",
            ),
            xaxis_title="Phase",
            yaxis_title="Excited-state probability",
            width=600,
            height=400,
            showlegend=True,
        )
        _annotate(
            fringe_figure,
            (f"|ρ<sub>01</sub>| = {summary['rho01']:.3g} ± {summary['rho01_err']:.2g}"),
        )
        figures[f"ramsey_{index:02d}"] = fringe_figure

    return figures


def display_post_selected_quantum_efficiency(result: Result, plot: bool = True):
    """Print the fitted quantities and show the figures."""
    data = result.data
    target = data["target"]
    amplitudes = data["readout_amplitudes"]
    dephasing = data["dephasing"]
    snr = data["snr"]
    envelope_fit = dephasing["fit"]
    snr_fit = snr["fit"]

    print(f"Post-selected quantum efficiency : {target}")
    print(f"  snr_convention = {data['snr_convention']}")
    print(
        f"  eta = {data['quantum_efficiency']:.6g}"
        f" ± {data['quantum_efficiency_err']:.2g}"
    )
    print(f"  a = {snr_fit['a']:.6g} ± {snr_fit['a_err']:.2g}")
    print(
        f"  sigma_m = {envelope_fit['sigma_m']:.6g} ± {envelope_fit['sigma_m_err']:.2g}"
    )
    print(f"  b = {envelope_fit['b']:.6g} ± {envelope_fit['b_err']:.2g}")
    print(
        "  amplitude | rho01 | snr_before | snr_after | rate_g | rate_e"
        " | retained_g | retained_e (test set)"
    )
    for index, amplitude in enumerate(amplitudes):
        rho01 = dephasing["rho01"][index]
        rho01_err = dephasing["rho01_err"][index]
        snr_before = snr["snr"]["before_post_selection"]["test"][index]
        snr_after = snr["snr"]["after_post_selection"]["test"][index]
        rate_g = snr["post_selection_rate"]["g"]["test"][index]
        rate_e = snr["post_selection_rate"]["e"]["test"][index]
        retained_g = snr["retained_shots"]["g"]["test"][index]
        retained_e = snr["retained_shots"]["e"]["test"][index]
        print(
            f"  {amplitude:.6g} | {rho01:.4g}±{rho01_err:.2g}"
            f" | {snr_before:.4g} | {snr_after:.4g}"
            f" | {rate_g:.3f} | {rate_e:.3f}"
            f" | {retained_g} | {retained_e}"
        )
    if plot:
        for key, figure in (result.figures or {}).items():
            figure.show(
                config=viz.get_config(
                    filename=f"post_selected_quantum_efficiency_{target}_{key}",
                )
            )


def post_selected_quantum_efficiency_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    readout_amplitude_range,
    readout_duration: float | None,
    readout_ramptime: float | None,
    ramsey_phase_range,
    waittime_after_readout_drive: float,
    waittime_after_capture_window: float,
    waittime_after_ctrl_pulse: float,
    dephasing_n_shots: int,
    n_train: int,
    n_test: int,
    n_max_epoch: int = 10,
    shot_interval: float,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    """
    Measure the quantum efficiency with post-selected SNR extraction.

    `readout_amplitude_range` accepts a list (at least 3
    amplitudes). For each amplitude the dephasing and SNR acquisitions run
    sequentially, then the analysis fits

    - ``|rho01|(epsilon) = b * exp(-epsilon^2 / (2 * sigma_m^2))``
    - ``SNR(epsilon) = a * epsilon`` (after post-selection, test set)

    and reports ``eta = a^2 * sigma_m^2`` (see the module docstring for the
    SNR convention and the low-amplitude post-selection caveat).
    """
    _validate_readout_amplitude_range(readout_amplitude_range)
    phase_array = np.asarray(ramsey_phase_range, dtype=np.float64)

    pulse_schedule_is_plotted = False
    dephasing_run_results = []
    for amplitude in readout_amplitude_range:
        dephasing_run_results.append(
            run_readout_induced_dephasing(
                exp,
                target=target,
                readout_amplitude=float(amplitude),
                readout_duration=readout_duration,
                readout_ramptime=readout_ramptime,
                ramsey_phase_range=phase_array,
                waittime_after_readout_drive=waittime_after_readout_drive,
                waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
                n_shots=dephasing_n_shots,
                shot_interval=shot_interval,
                plot_pulse_schedule=plot_pulse_schedule
                and not pulse_schedule_is_plotted,
            )
        )
        pulse_schedule_is_plotted = True
    pulse_schedule_is_plotted = False
    snr_run_results = []
    for amplitude in readout_amplitude_range:
        snr_run_results.append(
            run_post_selected_snr(
                exp,
                target=target,
                readout_amplitude=float(amplitude),
                readout_duration=readout_duration,
                readout_ramptime=readout_ramptime,
                waittime_after_readout_drive=waittime_after_readout_drive,
                waittime_after_capture_window=waittime_after_capture_window,
                waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
                n_train=n_train,
                n_test=n_test,
                pump_buffer_time=pump_buffer_time,
                pump_ramptime=pump_ramptime,
                pump_amplitude=pump_amplitude,
                shot_interval=shot_interval,
                plot_pulse_schedule=plot_pulse_schedule
                and not pulse_schedule_is_plotted,
            )
        )
        pulse_schedule_is_plotted = True
    analysis_result = analyze_post_selected_quantum_efficiency(
        target=target,
        dephasing_run_results=dephasing_run_results,
        snr_run_results=snr_run_results,
        n_train=n_train,
        n_test=n_test,
        n_max_epoch=n_max_epoch,
    )
    figures = plot_post_selected_quantum_efficiency(analysis_result=analysis_result)
    data = {
        **analysis_result,
        "parameters": {
            "target": target,
            "readout_amplitude_range": readout_amplitude_range,
            "readout_duration": readout_duration,
            "readout_ramptime": readout_ramptime,
            "ramsey_phase_range": phase_array,
            "waittime_after_readout_drive": waittime_after_readout_drive,
            "waittime_after_capture_window": waittime_after_capture_window,
            "waittime_after_ctrl_pulse": waittime_after_ctrl_pulse,
            "dephasing_n_shots": dephasing_n_shots,
            "n_train": n_train,
            "n_test": n_test,
            "n_max_epoch": n_max_epoch,
            "shot_interval": shot_interval,
        },
        "raw": {
            "dephasing": dephasing_run_results,
            "snr": snr_run_results,
        },
    }
    result = Result(
        data=data,
        figure=figures["overview"],
        figures=figures,
    )
    display_post_selected_quantum_efficiency(result, plot)
    return result
