"""
This module provides functions to perform IMPA pump pulse calibration.

The pump pulse frequency and amplitude are swept on a 2D grid and the
readout SNR gain of pump-on over pump-off is measured for each point.

Method
------
For each parameter set, four pulse schedules are measured with
single-shot, time-integrated readout
(``execute(mode="single", time_integration=True)``), one for every
combination of readout pulse on/off and pump pulse on/off:

- readout off / pump off
- readout on  / pump off
- readout off / pump on
- readout on  / pump on

"No pulse" is realized as a zero-amplitude pulse so that all four
schedules have identical timing and capture windows. The readout-on
schedules use ``readout_amplitude`` as the readout pulse amplitude
(``None`` uses the calibrated readout amplitude).

- Signal: distance in the IQ plane between the shot means of the
  readout-on and readout-off measurements, evaluated separately for
  pump off and pump on:
  ``signal = |mean(readout on) - mean(readout off)|``.
- Noise: the IQ-plane RMS scatter of the readout-on shots, evaluated
  separately for pump off and pump on. This is the IQ-plane RMS SNR
  convention, not the projected discrimination SNR. Two estimators are
  computed (see below).
- SNR: ``signal / noise`` (amplitude SNR). The SNR gain is reported in
  dB as ``20 * log10(snr_on / snr_off)``, which equals ``10 * log10``
  of the power SNR ratio.

SNR estimators
--------------
Two noise estimators are computed from the same shots and combined with
the same signal, so they differ *only* in how the noise is estimated.
Results carry a ``_gauss`` suffix for the second one, and the best point
is selected independently for each.

- ``moment`` (keys without a suffix): the sample standard deviation of
  the readout-on shots, ``sqrt(Var_I + Var_Q)`` with ``ddof=1``.
- ``gaussian fit`` (keys with a ``_gauss`` suffix): the width of a
  Gaussian fitted to the readout-on cloud, ``sqrt(trace(Sigma))``.

The second estimator exists because there is no pi pulse in this
experiment, so the qubit is at thermal equilibrium and a readout-on
cloud is a mixture of the |0> and |1> readout responses with weights
``(1 - p, p)``. Its sample standard deviation therefore picks up the
blob separation ``d`` on top of the true noise::

    sigma_mix = sigma_c * sqrt(1 + p * (1 - p) * (d / sigma_c) ** 2)

At ``p = 5%`` this inflates the noise by 17% at ``d = 4 sigma_c`` and by
57% at ``d = 8 sigma_c``. Because ``d`` grows with the amplifier gain,
the inflation is stronger with the pump on and does *not* cancel in the
``snr_on / snr_off`` ratio; it compresses the reported SNR gain.

The fit uses ``covariance_type="tied"``, so every component shares one
covariance. That matches the physics (both blobs are read out through
the same amplifier chain) and means the fit yields a single sigma, so
the |0>/|1> components never have to be told apart. Only the noise is
taken from the fit; the signal keeps using the sample means, because the
mean bias is only ``O(p)`` and its sign depends on the IQ geometry.

One- and two-component models are fitted and selected by BIC
(``GAUSS_FIT_BIC_THRESHOLD``). Where the blobs are not resolved - the
pump-off reference in particular - an unconstrained two-component fit
would split a single blob and report too small a sigma, which would
inflate ``snr_off`` and deflate the gain. BIC rejects the second
component there, and the one-component fit reproduces the sample
covariance exactly, so those points fall back to the moment value. This
is the right behaviour rather than a fudge: the moment estimator is only
wrong once the blobs are resolved, which is exactly when the
two-component fit succeeds. ``n_components_*_gauss`` records which model
was used at every point (0 = not fitted, moment value used), and
``p_thermal_*_gauss`` records the fitted minority weight.

The pump-off reference is measured once per pump frequency row, so
``snr_off`` is a vector over frequency and slow readout drift between
rows is compensated. The pump frequency is applied per row with
``exp.modified_frequencies({mux: pump_frequency})``, where the mux label
is the PUMP channel label; the pump-on schedules depend only on the pump
amplitude and are reused across rows.

The DC bias voltage is the outermost sweep axis: ``voltage_range``
accepts a scalar or a list, and the full pump frequency x amplitude
sweep (including the pump-off reference) is repeated for each voltage.
The best (voltage, frequency, amplitude) combination over all sweeps is
reported.

Per-voltage SNR heatmaps are produced for both estimators (the
Gaussian-fit ones under ``*_gauss`` keys), together with two diagnostic
heatmaps of the fit itself, ``p_thermal_gauss`` and
``n_components_gauss``.

An IQ-plane scatter figure (``iq_best``) of the four single-shot clouds
at the best point (readout off/on x pump off/on) is produced, with each
cloud mean marked by a black cross and the fitted noise drawn as a
dotted 2-sigma circle on the readout-on clouds; the figure is omitted
when the best point is undefined. ``iq_best_gauss`` is added when the
Gaussian-fit estimator picks a different best point.

The same figure is available for *any* swept point through
``display_impa_pump_pulse_calibration_iq_shots``, which takes the result
and the index of each swept parameter, shows the IQ scatter, and reports
the parameter values and the SNR at that point
(``plot_impa_pump_pulse_calibration_iq_shots`` returns the figure
instead of showing it). These per-point figures are built on demand and
are deliberately not stored in the result, because keeping one figure
per (voltage, frequency, amplitude) combination would make the result
very large; the single shots they are built from are already stored.

Only ``run_impa_pump_pulse_calibration`` needs an ``Experiment``. The
later steps are closed over the data the run step returns:
``analyze_impa_pump_pulse_calibration(run_result=...)``,
``plot_impa_pump_pulse_calibration(run_result=..., analysis_result=...)``
and ``display_impa_pump_pulse_calibration(result=...)`` take nothing
else, so they can be re-run offline on a stored ``run_result`` without a
connection to the hardware and without repeating the sweep axes. The
qubit label and the sweep axes are resolved once during the run and
carried in ``run_result``.

Pump LO control
---------------
The sweep frequencies are grouped into consecutive subranges such that
every point stays within +/-200 MHz of the actual mixing frequency
(LO + CNCO) programmed for its group; the pump port LO/CNCO is set
explicitly per group with ``modified_backend_settings`` and restored
afterwards. ``pump_frequency_range`` must therefore be non-empty,
finite, and strictly increasing. The pump port has a dedicated LO, so
retuning does not affect the readout channel; note, however, that on
some box types the pump LO chip is shared with the monitor-in line, and
each retune re-initializes the AWG/capture units of the whole box, so
avoid concurrent use of the same box. The QuEL-3 backend provides no
pump LO control and is not supported (RuntimeError).

Timing
------
The pump pulse is a flat-top on the mux (pump) channel of duration
``readout_pulse.duration + 2 * pump_buffer_time`` with ramp time
``pump_ramptime``; the readout pulse starts after ``pump_buffer_time``,
so the pump envelope covers the readout with a margin on both sides.
Choose ``pump_buffer_time`` large enough that the pump ramp and the
amplifier settling complete before the capture window starts. The
flat-top requires ``duration >= 2 * tau`` and the capture start must lie
on the capture word grid; both are enforced when the schedule is built
or compiled. Frequencies are in GHz, durations in ns, and amplitudes in
normalized hardware units.

This module assumes that an external instance `dc_bias_controller` is provided, which has the following methods:
- `validate_voltage(voltage: float)`: Validates if the given voltage is within the acceptable range for the DC bias controller.
- `voltage(mux: str, voltage_in_volts: float = None)`: Gets or sets the voltage for the specified multiplexer (mux).
- `voltage_sweep_rate(mux: str, rate_in_volts_per_second: float = None)`: Gets or sets the voltage sweep rate for the specified multiplexer (mux).

Example
-------
::

    result = impa_pump_pulse_calibration(
        exp,
        target="Q00",
        pump_frequency_range=np.arange(15.0, 16.0, 0.02),
        pump_amplitude_range=np.linspace(0.1, 1.0, 10),
        pump_buffer_time=128.0,
        pump_ramptime=32.0,
        readout_amplitude=None,
        dc_bias_controller=dc_bias_controller,
        voltage_range=0.1,
        shot_interval=150 * 1024,
        n_shots=1024,
    )

    from qubex.contrib.experiment.impa_calibration.impa_pump_pulse_calibration import (
        display_impa_pump_pulse_calibration_iq_shots,
    )

    display_impa_pump_pulse_calibration_iq_shots(
        result=result,
        voltage_index=0,
        pump_frequency_index=3,
        pump_amplitude_index=2,
    )
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

import qubex as qx
import qubex.visualization as viz
from qubex.backend.backend_controller import BACKEND_KIND_QUEL3
from qubex.experiment.models import Result
from qubex.system import MixingUtil
from qubex.system.quel1.quel1_system_constants import CNCO_CENTER_READ_HZ

# The pump LO/CNCO is retuned whenever the next sweep point would sit
# more than this far from the actual programmed mixing frequency.
PUMP_LO_MAX_DETUNING_HZ = 200_000_000
# Mirrors ExperimentSystem._configure_pump_port.
PUMP_SSB = "U"
# Hardware LO limits; outside this range the LO is silently mis-programmed.
PUMP_LO_MIN_HZ = 1_875_000_000
PUMP_LO_MAX_HZ = 15_000_000_000

# Gaussian-fit noise estimator (see the "SNR estimators" section above).
# A two-component fit is accepted only when it beats the one-component fit by
# more than this much BIC; 10 is the conventional "strong evidence" threshold.
# The one/two-component parameter counts differ by 3 (one extra mean and one
# extra weight), so the complexity penalty alone is 3 * ln(n_shots) ~ 21 at
# 1024 shots, which is what an unresolved fit has to overcome.
GAUSS_FIT_BIC_THRESHOLD = 10.0
# n_init=1 already matches n_init=10 on resolved mixtures; 2 is a cheap hedge
# for small excited-state populations, where the k-means seed is less reliable.
GAUSS_FIT_N_INIT = 2
GAUSS_FIT_RANDOM_STATE = 42
# All components share one covariance, so the fit yields a single sigma and the
# |0>/|1> component labels never have to be resolved.
GAUSS_FIT_COVARIANCE_TYPE = "tied"


def _validate_dc_bias_controller(dc_bias_controller) -> None:
    if not hasattr(dc_bias_controller, "validate_voltage"):
        raise ValueError(
            "The provided dc_bias_controller does not have a 'validate_voltage' method."
        )
    if not hasattr(dc_bias_controller, "voltage"):
        raise ValueError(
            "The provided dc_bias_controller does not have a 'voltage' method."
        )
    if not hasattr(dc_bias_controller, "voltage_sweep_rate"):
        raise ValueError(
            "The provided dc_bias_controller does not have a 'voltage_sweep_rate' method."
        )


def _validate_voltage_range(dc_bias_controller, voltage_range: ArrayLike) -> None:
    for voltage in voltage_range:
        dc_bias_controller.validate_voltage(voltage)


def _validate_sweep_index(*, name: str, value, size: int) -> int:
    # Sweep indices are user-facing coordinates, so Python-style negative
    # indexing is rejected instead of silently selecting another point.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    index = int(value)
    if not 0 <= index < size:
        raise ValueError(f"{name} must be in the range [0, {size - 1}], got {index}.")
    return index


def _build_pulse_schedule(
    exp: qx.Experiment,
    *,
    target: str,
    send_readout_pulse: bool,
    send_pump_pulse: bool,
    pump_buffer_time: float,
    pump_amplitude: float,
    pump_ramptime: float,
    readout_amplitude: float | None,
    plot_pulse_schedule: bool = True,
) -> qx.PulseSchedule:
    qubit = exp.ctx.resolve_qubit_label(target)
    readout = exp.ctx.resolve_read_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    readout_pulse = exp.readout(
        target,
        amplitude=readout_amplitude if send_readout_pulse else 0.0,
    )
    schedule = qx.PulseSchedule()
    with schedule as ps:
        ps.add(
            mux,
            qx.pulse.FlatTop(
                amplitude=pump_amplitude if send_pump_pulse else 0.0,
                duration=readout_pulse.duration + 2 * pump_buffer_time,
                tau=pump_ramptime,
            ),
        )
        ps.add(readout, qx.pulse.Blank(duration=pump_buffer_time))
        ps.add(readout, readout_pulse)
    if plot_pulse_schedule:
        schedule.plot(
            title=(
                "IMPA Pump Pulse Calibration : "
                f"readout={'on' if send_readout_pulse else 'off'}, "
                f"pump={'on' if send_pump_pulse else 'off'}"
            )
        )
    return schedule


def _extract_iq_shots(measure_result, *, qubit: str):
    if qubit not in measure_result.data:
        raise KeyError(
            f"Measurement data for {qubit} not found. "
            f"Available targets: {list(measure_result.data)}"
        )
    captures = measure_result.data[qubit]
    if len(captures) == 0:
        raise RuntimeError(f"No capture windows returned for {qubit}.")
    # The last capture is the readout window; on QuEL-1 a workaround
    # capture may be prepended at t=0.
    return np.atleast_1d(np.asarray(captures[-1].kerneled))


def _sample_iq_rms(iq_shots) -> float:
    # Complex std = sqrt(Var_I + Var_Q) (IQ-plane RMS noise).
    shots = np.asarray(iq_shots)
    if shots.size < 2:
        return float("nan")
    return float(np.std(shots, ddof=1))


def _fit_cloud_noise(iq_shots) -> dict:
    """
    Estimate the IQ-plane RMS noise of one shot cloud by Gaussian fitting.

    Without a pi pulse the qubit sits at thermal equilibrium, so a readout-on
    cloud is a mixture of the |0> and |1> readout responses. Its sample
    standard deviation therefore picks up the blob separation ``d`` on top of
    the true noise, ``sigma_mix = sigma_c * sqrt(1 + p(1-p)(d/sigma_c)**2)``.

    A tied-covariance Gaussian mixture is fitted with one and with two
    components and the better one is chosen by BIC. ``tied`` means every
    component shares a single covariance, which is what the physics says
    anyway (both blobs are read out through the same amplifier chain) and
    which removes any need to decide *which* component is the ground state.
    The one-component fit reproduces the sample covariance exactly, so an
    unresolved cloud simply falls back to the moment estimate.
    """
    shots = np.asarray(iq_shots)
    # n_components=0 flags "not fitted, moment value used".
    not_fitted = {
        "noise": _sample_iq_rms(shots),
        "n_components": 0,
        "delta_bic": float("nan"),
        "p_thermal": float("nan"),
    }
    finite = shots[np.isfinite(shots)]
    # Two components need enough shots to be meaningful at all.
    if finite.size < 8:
        return not_fitted
    features = np.column_stack((np.real(finite), np.imag(finite)))
    # sklearn's reg_covar is an absolute floor, so a cloud with small IQ
    # magnitudes would be dominated by it. Mirrors StateClassifierGMM.fit.
    extent = max(float(np.ptp(features[:, 0])), float(np.ptp(features[:, 1])))
    if not np.isfinite(extent) or extent <= 0.0:
        return not_fitted
    scale = 1.0 / extent
    scaled = features * scale

    def fit(n_components: int) -> GaussianMixture:
        return GaussianMixture(
            n_components=n_components,
            covariance_type=GAUSS_FIT_COVARIANCE_TYPE,
            n_init=GAUSS_FIT_N_INIT,
            random_state=GAUSS_FIT_RANDOM_STATE,
        ).fit(scaled)

    model_1 = fit(1)
    model_2 = fit(2)
    # Both models share the covariance family, so their BICs are comparable.
    delta_bic = float(model_1.bic(scaled) - model_2.bic(scaled))
    if delta_bic > GAUSS_FIT_BIC_THRESHOLD:
        model = model_2
        n_components = 2
        p_thermal = float(np.min(np.asarray(model_2.weights_)))
    else:
        model = model_1
        n_components = 1
        p_thermal = float("nan")
    noise = float(np.sqrt(np.trace(np.asarray(model.covariances_))) / scale)
    return {
        "noise": noise,
        "n_components": n_components,
        "delta_bic": delta_bic,
        "p_thermal": p_thermal,
    }


def _fit_cloud_noise_grid(iq_clouds, *, desc: str) -> dict:
    # Fit every cloud of an N-dimensional grid whose last axis is the shots.
    clouds = np.asarray(iq_clouds)
    grid_shape = clouds.shape[:-1]
    noise = np.empty(grid_shape, dtype=float)
    n_components = np.empty(grid_shape, dtype=int)
    delta_bic = np.empty(grid_shape, dtype=float)
    p_thermal = np.empty(grid_shape, dtype=float)
    for index in tqdm(list(np.ndindex(grid_shape)), desc=desc):
        fit = _fit_cloud_noise(clouds[index])
        noise[index] = fit["noise"]
        n_components[index] = fit["n_components"]
        delta_bic[index] = fit["delta_bic"]
        p_thermal[index] = fit["p_thermal"]
    return {
        "noise": noise,
        "n_components": n_components,
        "delta_bic": delta_bic,
        "p_thermal": p_thermal,
    }


def run_impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    pump_buffer_time: float,
    pump_ramptime: float,
    readout_amplitude: float | None,
    dc_bias_controller,
    voltage_range: ArrayLike,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
):
    _validate_dc_bias_controller(dc_bias_controller)
    voltage_range = np.atleast_1d(np.asarray(voltage_range, dtype=float))
    if voltage_range.size == 0:
        raise ValueError("voltage_range must not be empty.")
    if not np.all(np.isfinite(voltage_range)):
        raise ValueError("voltage_range must contain only finite values.")
    _validate_voltage_range(dc_bias_controller, voltage_range)
    pump_frequency_range = np.atleast_1d(np.asarray(pump_frequency_range, dtype=float))
    pump_amplitude_range = np.atleast_1d(np.asarray(pump_amplitude_range, dtype=float))
    if exp.system_manager.backend_kind == BACKEND_KIND_QUEL3:
        raise RuntimeError(
            "Explicit pump LO control is not supported on the QuEL-3 backend."
        )
    if pump_frequency_range.size == 0:
        raise ValueError("pump_frequency_range must not be empty.")
    if not np.all(np.isfinite(pump_frequency_range)):
        raise ValueError("pump_frequency_range must contain only finite values.")
    if pump_frequency_range.ndim != 1 or not np.all(np.diff(pump_frequency_range) > 0):
        raise ValueError("pump_frequency_range must be strictly increasing.")

    # Group consecutive frequencies so that every point stays within
    # PUMP_LO_MAX_DETUNING_HZ of the actual mixing frequency (LO + CNCO)
    # programmed for its group; the pump LO is retuned once per group.
    pump_lo_groups = []
    start = 0
    while start < len(pump_frequency_range):
        end = start + 1
        lo, cnco, f_mix = MixingUtil.calc_lo_cnco(
            float(pump_frequency_range[start]) * 1e9,
            ssb=PUMP_SSB,
            cnco_center=CNCO_CENTER_READ_HZ,
        )
        while end < len(pump_frequency_range):
            f_center = (pump_frequency_range[start] + pump_frequency_range[end]) / 2
            candidate = MixingUtil.calc_lo_cnco(
                float(f_center) * 1e9,
                ssb=PUMP_SSB,
                cnco_center=CNCO_CENTER_READ_HZ,
            )
            # The range is ascending, so checking both end points suffices.
            detuning = max(
                abs(pump_frequency_range[start] * 1e9 - candidate[2]),
                abs(pump_frequency_range[end] * 1e9 - candidate[2]),
            )
            if detuning > PUMP_LO_MAX_DETUNING_HZ:
                break
            lo, cnco, f_mix = candidate
            end += 1
        pump_lo_groups.append((pump_frequency_range[start:end], lo, cnco, f_mix))
        start = end
    for subrange, lo, cnco, f_mix in pump_lo_groups:
        if (
            lo is None
            or not np.isfinite(lo)
            or not np.isfinite(cnco)
            or not np.isfinite(f_mix)
        ):
            raise ValueError(
                f"Invalid pump LO settings (lo={lo}, cnco={cnco}, f_mix={f_mix}) "
                f"for {subrange[0]}-{subrange[-1]} GHz."
            )
        if not PUMP_LO_MIN_HZ <= lo <= PUMP_LO_MAX_HZ:
            raise ValueError(
                f"The pump LO frequency {lo / 1e9} GHz for "
                f"{subrange[0]}-{subrange[-1]} GHz is outside the supported range "
                f"[{PUMP_LO_MIN_HZ / 1e9}, {PUMP_LO_MAX_HZ / 1e9}] GHz."
            )

    qubit = exp.ctx.resolve_qubit_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)

    ps_readout_off_pump_off = _build_pulse_schedule(
        exp,
        target=target,
        send_readout_pulse=False,
        send_pump_pulse=False,
        pump_buffer_time=pump_buffer_time,
        pump_amplitude=0.0,
        pump_ramptime=pump_ramptime,
        readout_amplitude=readout_amplitude,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    ps_readout_on_pump_off = _build_pulse_schedule(
        exp,
        target=target,
        send_readout_pulse=True,
        send_pump_pulse=False,
        pump_buffer_time=pump_buffer_time,
        pump_amplitude=0.0,
        pump_ramptime=pump_ramptime,
        readout_amplitude=readout_amplitude,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    # The pump-on schedules depend only on the amplitude; the pump
    # frequency is applied at execution time, so they are reused across
    # frequency rows.
    ps_readout_off_pump_on_list = [
        _build_pulse_schedule(
            exp,
            target=target,
            send_readout_pulse=False,
            send_pump_pulse=True,
            pump_buffer_time=pump_buffer_time,
            pump_amplitude=float(amplitude),
            pump_ramptime=pump_ramptime,
            readout_amplitude=readout_amplitude,
            plot_pulse_schedule=plot_pulse_schedule and index == 0,
        )
        for index, amplitude in enumerate(pump_amplitude_range)
    ]
    ps_readout_on_pump_on_list = [
        _build_pulse_schedule(
            exp,
            target=target,
            send_readout_pulse=True,
            send_pump_pulse=True,
            pump_buffer_time=pump_buffer_time,
            pump_amplitude=float(amplitude),
            pump_ramptime=pump_ramptime,
            readout_amplitude=readout_amplitude,
            plot_pulse_schedule=plot_pulse_schedule and index == 0,
        )
        for index, amplitude in enumerate(pump_amplitude_range)
    ]

    def execute(schedule):
        return exp.execute(
            schedule=schedule,
            mode="single",
            time_integration=True,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )

    results_readout_off_pump_off = []
    results_readout_on_pump_off = []
    results_readout_off_pump_on = []
    results_readout_on_pump_on = []
    original_voltage = dc_bias_controller.voltage(mux=mux)
    print(f"Original voltage for mux '{mux}': {original_voltage} V")
    print(
        f"Voltage sweep rate for mux '{mux}': {dc_bias_controller.voltage_sweep_rate(mux=mux)} V/s"
    )
    try:
        for voltage in voltage_range:
            dc_bias_controller.voltage(mux=mux, voltage_in_volts=float(voltage))
            v_results_readout_off_pump_off = []
            v_results_readout_on_pump_off = []
            v_results_readout_off_pump_on = []
            v_results_readout_on_pump_on = []
            with tqdm(
                total=len(pump_frequency_range),
                desc=f"IMPA pump pulse calibration : {target}, {voltage} V",
            ) as progress_bar:
                for subrange, lo, cnco, f_mix in pump_lo_groups:
                    print(
                        f"Pump LO = {lo / 1e9} GHz, CNCO = {cnco / 1e9} GHz "
                        f"(mixing frequency {f_mix / 1e9} GHz) for "
                        f"{subrange[0]}-{subrange[-1]} GHz"
                    )
                    with exp.system_manager.modified_backend_settings(
                        label=mux,
                        lo_freq=lo,
                        cnco_freq=cnco,
                        fnco_freq=0,
                    ):
                        for frequency in subrange:
                            with exp.modified_frequencies({mux: float(frequency)}):
                                # Pump-off reference per frequency row
                                # (drift tracking).
                                v_results_readout_off_pump_off.append(
                                    execute(ps_readout_off_pump_off)
                                )
                                v_results_readout_on_pump_off.append(
                                    execute(ps_readout_on_pump_off)
                                )
                                row_readout_off_pump_on = []
                                row_readout_on_pump_on = []
                                for (
                                    ps_readout_off_pump_on,
                                    ps_readout_on_pump_on,
                                ) in zip(
                                    ps_readout_off_pump_on_list,
                                    ps_readout_on_pump_on_list,
                                    strict=True,
                                ):
                                    row_readout_off_pump_on.append(
                                        execute(ps_readout_off_pump_on)
                                    )
                                    row_readout_on_pump_on.append(
                                        execute(ps_readout_on_pump_on)
                                    )
                                v_results_readout_off_pump_on.append(
                                    row_readout_off_pump_on
                                )
                                v_results_readout_on_pump_on.append(
                                    row_readout_on_pump_on
                                )
                            progress_bar.update(1)
            results_readout_off_pump_off.append(v_results_readout_off_pump_off)
            results_readout_on_pump_off.append(v_results_readout_on_pump_off)
            results_readout_off_pump_on.append(v_results_readout_off_pump_on)
            results_readout_on_pump_on.append(v_results_readout_on_pump_on)
    finally:
        dc_bias_controller.voltage(mux=mux, voltage_in_volts=original_voltage)
        print(
            f"Voltage for mux '{mux}' restored to original value: {dc_bias_controller.voltage(mux=mux)} V"
        )
    return {
        "target": target,
        "qubit": qubit,
        "mux": mux,
        "pump_frequency_range": pump_frequency_range,
        "pump_amplitude_range": pump_amplitude_range,
        "pump_buffer_time": pump_buffer_time,
        "pump_ramptime": pump_ramptime,
        "readout_amplitude": readout_amplitude,
        "voltage_range": voltage_range,
        "shot_interval": shot_interval,
        "n_shots": n_shots,
        "pump_lo_subranges": [
            {
                "lo_freq": lo,
                "cnco_freq": cnco,
                "f_mix": f_mix,
                "frequency_start": float(subrange[0]),
                "frequency_stop": float(subrange[-1]),
            }
            for subrange, lo, cnco, f_mix in pump_lo_groups
        ],
        "results_readout_off_pump_off": results_readout_off_pump_off,
        "results_readout_on_pump_off": results_readout_on_pump_off,
        "results_readout_off_pump_on": results_readout_off_pump_on,
        "results_readout_on_pump_on": results_readout_on_pump_on,
    }


def _compute_snr_gain(
    *,
    signal_off,
    signal_on,
    noise_off,
    noise_on,
    voltage_range,
    pump_frequency_range,
    pump_amplitude_range,
    method_label: str,
) -> dict:
    # Shared by both noise estimators; only `noise_off` / `noise_on` differ.
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_off = signal_off / noise_off
        snr_on = signal_on / noise_on
        snr_gain = snr_on / snr_off[:, :, np.newaxis]
        snr_gain_db = 20 * np.log10(snr_gain)
    invalid_reference = ~np.isfinite(snr_off) | (snr_off <= 0)
    if np.any(invalid_reference):
        invalid_points = [
            (float(voltage_range[i]), float(pump_frequency_range[j]))
            for i, j in zip(*np.where(invalid_reference), strict=True)
        ]
        print(
            f"[WARN] invalid pump-off reference SNR ({method_label}) at "
            f"(voltage V, frequency GHz) = {invalid_points}; "
            "the SNR gain is NaN at the affected points."
        )
        snr_gain[invalid_reference] = np.nan
        snr_gain_db[invalid_reference] = np.nan
    finite_snr_gain_db = np.where(np.isfinite(snr_gain_db), snr_gain_db, np.nan)
    if np.all(np.isnan(finite_snr_gain_db)):
        print(
            f"[WARN] no finite SNR gain value ({method_label}); "
            "the best point is undefined."
        )
        best_index = None
        best_voltage = None
        best_pump_frequency = None
        best_pump_amplitude = None
        best_snr_gain_db = None
        best_snr_on = None
    else:
        best_index = np.unravel_index(
            np.nanargmax(finite_snr_gain_db), finite_snr_gain_db.shape
        )
        best_voltage = float(voltage_range[best_index[0]])
        best_pump_frequency = float(pump_frequency_range[best_index[1]])
        best_pump_amplitude = float(pump_amplitude_range[best_index[2]])
        best_snr_gain_db = float(snr_gain_db[best_index])
        best_snr_on = float(snr_on[best_index])
    return {
        "snr_off": snr_off,
        "snr_on": snr_on,
        "snr_gain": snr_gain,
        "snr_gain_db": snr_gain_db,
        "best_index": best_index,
        "best_voltage": best_voltage,
        "best_pump_frequency": best_pump_frequency,
        "best_pump_amplitude": best_pump_amplitude,
        "best_snr_gain_db": best_snr_gain_db,
        "best_snr_on": best_snr_on,
    }


def _sweep_ranges(run_result):
    # The run step stores the sweep axes it actually used, so the analysis and
    # the plots read them from there instead of taking them as arguments.
    return (
        np.atleast_1d(np.asarray(run_result["voltage_range"], dtype=float)),
        np.atleast_1d(np.asarray(run_result["pump_frequency_range"], dtype=float)),
        np.atleast_1d(np.asarray(run_result["pump_amplitude_range"], dtype=float)),
    )


def analyze_impa_pump_pulse_calibration(
    *,
    run_result,
):
    voltage_range, pump_frequency_range, pump_amplitude_range = _sweep_ranges(
        run_result
    )
    # The qubit label was already resolved by the run step, so the analysis
    # needs no Experiment and can be re-run offline on a stored run_result.
    qubit = run_result["qubit"]
    iq_readout_off_pump_off = np.array(
        [
            [_extract_iq_shots(res, qubit=qubit) for res in rows]
            for rows in run_result["results_readout_off_pump_off"]
        ]
    )
    iq_readout_on_pump_off = np.array(
        [
            [_extract_iq_shots(res, qubit=qubit) for res in rows]
            for rows in run_result["results_readout_on_pump_off"]
        ]
    )
    iq_readout_off_pump_on = np.array(
        [
            [[_extract_iq_shots(res, qubit=qubit) for res in row] for row in rows]
            for rows in run_result["results_readout_off_pump_on"]
        ]
    )
    iq_readout_on_pump_on = np.array(
        [
            [[_extract_iq_shots(res, qubit=qubit) for res in row] for row in rows]
            for rows in run_result["results_readout_on_pump_on"]
        ]
    )

    mean_readout_off_pump_off = iq_readout_off_pump_off.mean(axis=-1)
    mean_readout_on_pump_off = iq_readout_on_pump_off.mean(axis=-1)
    mean_readout_off_pump_on = iq_readout_off_pump_on.mean(axis=-1)
    mean_readout_on_pump_on = iq_readout_on_pump_on.mean(axis=-1)
    signal_off = np.abs(mean_readout_on_pump_off - mean_readout_off_pump_off)
    signal_on = np.abs(mean_readout_on_pump_on - mean_readout_off_pump_on)
    # Complex std = sqrt(Var_I + Var_Q) (IQ-plane RMS noise).
    noise_off = iq_readout_on_pump_off.std(axis=-1, ddof=1)
    noise_on = iq_readout_on_pump_on.std(axis=-1, ddof=1)
    moment = _compute_snr_gain(
        signal_off=signal_off,
        signal_on=signal_on,
        noise_off=noise_off,
        noise_on=noise_on,
        voltage_range=voltage_range,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        method_label="moment",
    )

    # Gaussian-fit noise, evaluated on the same shots and combined with the
    # same signal, so the two methods differ only in the noise estimator.
    fit_off = _fit_cloud_noise_grid(
        iq_readout_on_pump_off,
        desc="Gaussian fit : pump off",
    )
    fit_on = _fit_cloud_noise_grid(
        iq_readout_on_pump_on,
        desc="Gaussian fit : pump on",
    )
    gauss = _compute_snr_gain(
        signal_off=signal_off,
        signal_on=signal_on,
        noise_off=fit_off["noise"],
        noise_on=fit_on["noise"],
        voltage_range=voltage_range,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        method_label="gaussian fit",
    )
    return {
        "iq_readout_off_pump_off": iq_readout_off_pump_off,
        "iq_readout_on_pump_off": iq_readout_on_pump_off,
        "iq_readout_off_pump_on": iq_readout_off_pump_on,
        "iq_readout_on_pump_on": iq_readout_on_pump_on,
        "mean_readout_off_pump_off": mean_readout_off_pump_off,
        "mean_readout_on_pump_off": mean_readout_on_pump_off,
        "mean_readout_off_pump_on": mean_readout_off_pump_on,
        "mean_readout_on_pump_on": mean_readout_on_pump_on,
        "signal_off": signal_off,
        "signal_on": signal_on,
        "noise_off": noise_off,
        "noise_on": noise_on,
        "snr_off": moment["snr_off"],
        "snr_on": moment["snr_on"],
        "snr_gain": moment["snr_gain"],
        "snr_gain_db": moment["snr_gain_db"],
        "best_index": moment["best_index"],
        "best_voltage": moment["best_voltage"],
        "best_pump_frequency": moment["best_pump_frequency"],
        "best_pump_amplitude": moment["best_pump_amplitude"],
        "best_snr_gain_db": moment["best_snr_gain_db"],
        "best_snr_on": moment["best_snr_on"],
        "noise_off_gauss": fit_off["noise"],
        "noise_on_gauss": fit_on["noise"],
        "n_components_off_gauss": fit_off["n_components"],
        "n_components_on_gauss": fit_on["n_components"],
        "delta_bic_off_gauss": fit_off["delta_bic"],
        "delta_bic_on_gauss": fit_on["delta_bic"],
        "p_thermal_off_gauss": fit_off["p_thermal"],
        "p_thermal_on_gauss": fit_on["p_thermal"],
        "snr_off_gauss": gauss["snr_off"],
        "snr_on_gauss": gauss["snr_on"],
        "snr_gain_gauss": gauss["snr_gain"],
        "snr_gain_db_gauss": gauss["snr_gain_db"],
        "best_index_gauss": gauss["best_index"],
        "best_voltage_gauss": gauss["best_voltage"],
        "best_pump_frequency_gauss": gauss["best_pump_frequency"],
        "best_pump_amplitude_gauss": gauss["best_pump_amplitude"],
        "best_snr_gain_db_gauss": gauss["best_snr_gain_db"],
        "best_snr_on_gauss": gauss["best_snr_on"],
    }


def _select_iq_point(
    analysis_result,
    *,
    voltage_range: ArrayLike,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    index,
):
    # Collect everything needed to plot one swept point; the pump-off
    # quantities are indexed by (voltage, frequency) only.
    v_idx, f_idx, a_idx = (int(i) for i in index)
    return {
        "index": (v_idx, f_idx, a_idx),
        "voltage": float(voltage_range[v_idx]),
        "pump_frequency": float(pump_frequency_range[f_idx]),
        "pump_amplitude": float(pump_amplitude_range[a_idx]),
        "iq_clouds": {
            "readout off / pump off": analysis_result["iq_readout_off_pump_off"][
                v_idx, f_idx
            ],
            "readout on / pump off": analysis_result["iq_readout_on_pump_off"][
                v_idx, f_idx
            ],
            "readout off / pump on": analysis_result["iq_readout_off_pump_on"][
                v_idx, f_idx, a_idx
            ],
            "readout on / pump on": analysis_result["iq_readout_on_pump_on"][
                v_idx, f_idx, a_idx
            ],
        },
        "cloud_means": {
            "readout off / pump off": analysis_result["mean_readout_off_pump_off"][
                v_idx, f_idx
            ],
            "readout on / pump off": analysis_result["mean_readout_on_pump_off"][
                v_idx, f_idx
            ],
            "readout off / pump on": analysis_result["mean_readout_off_pump_on"][
                v_idx, f_idx, a_idx
            ],
            "readout on / pump on": analysis_result["mean_readout_on_pump_on"][
                v_idx, f_idx, a_idx
            ],
        },
        # The fitted noise applies to the readout-on clouds only.
        "cloud_noise_gauss": {
            "readout on / pump off": float(
                analysis_result["noise_off_gauss"][v_idx, f_idx]
            ),
            "readout on / pump on": float(
                analysis_result["noise_on_gauss"][v_idx, f_idx, a_idx]
            ),
        },
        "snr_off": float(analysis_result["snr_off"][v_idx, f_idx]),
        "snr_on": float(analysis_result["snr_on"][v_idx, f_idx, a_idx]),
        "snr_gain_db": float(analysis_result["snr_gain_db"][v_idx, f_idx, a_idx]),
        "snr_off_gauss": float(analysis_result["snr_off_gauss"][v_idx, f_idx]),
        "snr_on_gauss": float(analysis_result["snr_on_gauss"][v_idx, f_idx, a_idx]),
        "snr_gain_db_gauss": float(
            analysis_result["snr_gain_db_gauss"][v_idx, f_idx, a_idx]
        ),
        "n_components_off_gauss": int(
            analysis_result["n_components_off_gauss"][v_idx, f_idx]
        ),
        "n_components_on_gauss": int(
            analysis_result["n_components_on_gauss"][v_idx, f_idx, a_idx]
        ),
        "p_thermal_off_gauss": float(
            analysis_result["p_thermal_off_gauss"][v_idx, f_idx]
        ),
        "p_thermal_on_gauss": float(
            analysis_result["p_thermal_on_gauss"][v_idx, f_idx, a_idx]
        ),
    }


def _select_iq_point_from_result(
    result: Result,
    *,
    voltage_index,
    pump_frequency_index,
    pump_amplitude_index,
):
    voltage_range = np.atleast_1d(np.asarray(result.data["voltage_range"], dtype=float))
    pump_frequency_range = np.atleast_1d(
        np.asarray(result.data["pump_frequency_range"], dtype=float)
    )
    pump_amplitude_range = np.atleast_1d(
        np.asarray(result.data["pump_amplitude_range"], dtype=float)
    )
    index = (
        _validate_sweep_index(
            name="voltage_index", value=voltage_index, size=len(voltage_range)
        ),
        _validate_sweep_index(
            name="pump_frequency_index",
            value=pump_frequency_index,
            size=len(pump_frequency_range),
        ),
        _validate_sweep_index(
            name="pump_amplitude_index",
            value=pump_amplitude_index,
            size=len(pump_amplitude_range),
        ),
    )
    point = _select_iq_point(
        result.data["analysis"],
        voltage_range=voltage_range,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        index=index,
    )
    return result.data["raw"]["target"], point


def _make_iq_shots_figure(*, target: str, point, at_best: bool, method: str):
    label = "at Best Point " if at_best else ""
    method_label = "Gaussian fit" if method == "gauss" else "moment"
    fig = viz.make_iq_scatter_figure(
        point["iq_clouds"],
        title=(
            f"IMPA Pump Pulse Calibration - IQ Shots {label}for {target}"
            f"<br><sup>V = {point['voltage']:.4g} V, "
            f"pump frequency = {point['pump_frequency']:.6g} GHz, "
            f"pump amplitude = {point['pump_amplitude']:.4g} a.u.<br>"
            f"moment : SNR = {point['snr_off']:.4g} (pump off) / "
            f"{point['snr_on']:.4g} (pump on), "
            f"gain = {point['snr_gain_db']:.4g} dB<br>"
            f"Gaussian fit : SNR = {point['snr_off_gauss']:.4g} (pump off) / "
            f"{point['snr_on_gauss']:.4g} (pump on), "
            f"gain = {point['snr_gain_db_gauss']:.4g} dB "
            f"[{point['n_components_off_gauss']}/"
            f"{point['n_components_on_gauss']} components]"
            f"<br><i>best point selected by {method_label}</i></sup>"
        ),
        width=600,
        height=460,
    )
    # The signal is defined from the cloud means; mark them so the
    # readout-on/off displacement is visible at a glance.
    for name, mean in point["cloud_means"].items():
        fig.add_scatter(
            x=[float(np.real(mean))],
            y=[float(np.imag(mean))],
            mode="markers",
            marker=dict(size=10, color="black", symbol="x"),
            name=f"mean : {name}",
            showlegend=False,
        )
    # Draw the fitted noise as a 2-sigma circle on the readout-on clouds, so
    # the gap between the raw scatter and the fitted noise is visible. Follows
    # the convention of viz.make_classification_figure. The fitted sigma is the
    # IQ-plane RMS, so the per-axis radius is sigma / sqrt(2).
    theta = np.linspace(0, 2 * np.pi, 100)
    for name, noise in point["cloud_noise_gauss"].items():
        mean = point["cloud_means"][name]
        if not np.isfinite(noise):
            continue
        radius = 2 * noise / np.sqrt(2)
        fig.add_scatter(
            x=float(np.real(mean)) + radius * np.cos(theta),
            y=float(np.imag(mean)) + radius * np.sin(theta),
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            name=f"fit ± 2σ : {name}",
            showlegend=False,
        )
    return fig


def _make_snr_figures(
    *,
    target: str,
    voltage_range,
    pump_frequency_range,
    pump_amplitude_range,
    snr_gain_db,
    snr_on,
    snr_off,
    best_index,
    best_pump_frequency,
    best_pump_amplitude,
    method_label: str,
) -> dict:
    # Clamp the color range at 0 dB so that all non-positive gains share
    # the same color; the range is shared by all voltages for
    # comparability.
    finite_gain_db = snr_gain_db[np.isfinite(snr_gain_db)]
    gain_zmax = float(finite_gain_db.max()) if finite_gain_db.size > 0 else 1.0
    if gain_zmax <= 0.0:
        gain_zmax = 1.0
    # One figure per voltage, collected as lists under a single key.
    figures_snr_gain = []
    figures_snr_on = []
    figures_snr_off = []
    for voltage_index, voltage in enumerate(voltage_range):
        # Plot SNR gain heatmap
        fig_snr_gain = viz.make_figure()
        fig_snr_gain.add_heatmap(
            x=pump_amplitude_range,
            y=pump_frequency_range,
            z=snr_gain_db[voltage_index],
            zmin=0.0,
            zmax=gain_zmax,
            colorscale="Viridis",
            colorbar=dict(title=dict(text="SNR gain (dB)", side="right")),
        )
        if best_index is not None and best_index[0] == voltage_index:
            fig_snr_gain.add_scatter(
                x=[best_pump_amplitude],
                y=[best_pump_frequency],
                mode="markers",
                marker=dict(symbol="x", color="red", size=10),
                name="Best",
                showlegend=False,
            )
        fig_snr_gain.update_layout(
            title=(
                f"IMPA Pump Pulse Calibration - SNR Gain ({method_label}) for "
                f"{target} at {voltage} V"
            ),
            xaxis_title="Pump amplitude (a.u.)",
            yaxis_title="Pump frequency (GHz)",
            width=600,
            height=400,
        )
        # Plot pump-on SNR heatmap
        fig_snr_on = viz.make_figure()
        fig_snr_on.add_heatmap(
            x=pump_amplitude_range,
            y=pump_frequency_range,
            z=snr_on[voltage_index],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="SNR (pump on)", side="right")),
        )
        fig_snr_on.update_layout(
            title=(
                f"IMPA Pump Pulse Calibration - Pump-on SNR ({method_label}) for "
                f"{target} at {voltage} V"
            ),
            xaxis_title="Pump amplitude (a.u.)",
            yaxis_title="Pump frequency (GHz)",
            width=600,
            height=400,
        )
        # Plot pump-off reference SNR
        fig_snr_off = viz.make_figure()
        fig_snr_off.add_scatter(
            x=pump_frequency_range,
            y=snr_off[voltage_index],
            mode="lines+markers",
            name="SNR (pump off)",
        )
        fig_snr_off.update_layout(
            title=(
                f"IMPA Pump Pulse Calibration - Pump-off Reference SNR "
                f"({method_label}) for {target} at {voltage} V"
            ),
            xaxis_title="Pump frequency (GHz)",
            yaxis_title="SNR (pump off)",
            width=600,
            height=400,
        )
        figures_snr_gain.append(fig_snr_gain)
        figures_snr_on.append(fig_snr_on)
        figures_snr_off.append(fig_snr_off)
    return {
        "snr_gain": figures_snr_gain,
        "snr_on": figures_snr_on,
        "snr_off": figures_snr_off,
    }


def _make_gauss_diagnostic_figures(
    *,
    target: str,
    voltage_range,
    pump_frequency_range,
    pump_amplitude_range,
    analysis_result,
) -> dict:
    # How the Gaussian fit behaved at every pump-on point: the fitted thermal
    # population and whether the two-component model was accepted at all.
    p_thermal = analysis_result["p_thermal_on_gauss"]
    n_components = analysis_result["n_components_on_gauss"]
    figures_p_thermal = []
    figures_n_components = []
    for voltage_index, voltage in enumerate(voltage_range):
        fig_p_thermal = viz.make_figure()
        fig_p_thermal.add_heatmap(
            x=pump_amplitude_range,
            y=pump_frequency_range,
            z=p_thermal[voltage_index],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="fitted p_thermal", side="right")),
        )
        fig_p_thermal.update_layout(
            title=(
                f"IMPA Pump Pulse Calibration - Fitted Thermal Population for "
                f"{target} at {voltage} V"
            ),
            xaxis_title="Pump amplitude (a.u.)",
            yaxis_title="Pump frequency (GHz)",
            width=600,
            height=400,
        )
        fig_n_components = viz.make_figure()
        fig_n_components.add_heatmap(
            x=pump_amplitude_range,
            y=pump_frequency_range,
            z=n_components[voltage_index],
            zmin=0,
            zmax=2,
            colorscale="Viridis",
            colorbar=dict(title=dict(text="components", side="right")),
        )
        fig_n_components.update_layout(
            title=(
                f"IMPA Pump Pulse Calibration - Accepted Gaussian Components for "
                f"{target} at {voltage} V"
            ),
            xaxis_title="Pump amplitude (a.u.)",
            yaxis_title="Pump frequency (GHz)",
            width=600,
            height=400,
        )
        figures_p_thermal.append(fig_p_thermal)
        figures_n_components.append(fig_n_components)
    return {
        "p_thermal_gauss": figures_p_thermal,
        "n_components_gauss": figures_n_components,
    }


def plot_impa_pump_pulse_calibration(
    *,
    run_result,
    analysis_result,
):
    voltage_range, pump_frequency_range, pump_amplitude_range = _sweep_ranges(
        run_result
    )
    target = run_result["target"]
    best_index = analysis_result["best_index"]
    best_index_gauss = analysis_result["best_index_gauss"]

    figures = _make_snr_figures(
        target=target,
        voltage_range=voltage_range,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        snr_gain_db=analysis_result["snr_gain_db"],
        snr_on=analysis_result["snr_on"],
        snr_off=analysis_result["snr_off"],
        best_index=best_index,
        best_pump_frequency=analysis_result["best_pump_frequency"],
        best_pump_amplitude=analysis_result["best_pump_amplitude"],
        method_label="moment",
    )
    figures.update(
        {
            f"{key}_gauss": value
            for key, value in _make_snr_figures(
                target=target,
                voltage_range=voltage_range,
                pump_frequency_range=pump_frequency_range,
                pump_amplitude_range=pump_amplitude_range,
                snr_gain_db=analysis_result["snr_gain_db_gauss"],
                snr_on=analysis_result["snr_on_gauss"],
                snr_off=analysis_result["snr_off_gauss"],
                best_index=best_index_gauss,
                best_pump_frequency=analysis_result["best_pump_frequency_gauss"],
                best_pump_amplitude=analysis_result["best_pump_amplitude_gauss"],
                method_label="Gaussian fit",
            ).items()
        }
    )
    figures.update(
        _make_gauss_diagnostic_figures(
            target=target,
            voltage_range=voltage_range,
            pump_frequency_range=pump_frequency_range,
            pump_amplitude_range=pump_amplitude_range,
            analysis_result=analysis_result,
        )
    )

    # Plot the IQ shot clouds at the best point of each method; omitted when
    # the best point is undefined, and the Gaussian-fit figure is omitted when
    # both methods agree on the same point.
    def iq_figure(index, method: str):
        return _make_iq_shots_figure(
            target=target,
            point=_select_iq_point(
                analysis_result,
                voltage_range=voltage_range,
                pump_frequency_range=pump_frequency_range,
                pump_amplitude_range=pump_amplitude_range,
                index=index,
            ),
            at_best=True,
            method=method,
        )

    if best_index is not None:
        figures["iq_best"] = iq_figure(best_index, "moment")
    if best_index_gauss is not None and best_index_gauss != best_index:
        figures["iq_best_gauss"] = iq_figure(best_index_gauss, "gauss")
    return figures


def display_impa_pump_pulse_calibration(
    *,
    result: Result,
    plot: bool = True,
):
    if plot:
        # Values are either a single figure or a list of per-voltage figures.
        for entry in (result.figures or {}).values():
            for fig in entry if isinstance(entry, list) else [entry]:
                fig.show()
    analysis = result.data["analysis"]
    print("IMPA Pump Pulse Calibration Result:")
    _display_best_point(analysis, method="moment")
    _display_best_point(analysis, method="gauss")
    _display_gauss_fit_summary(analysis)


def _display_best_point(analysis, *, method: str) -> None:
    suffix = "_gauss" if method == "gauss" else ""
    method_label = "Gaussian fit" if method == "gauss" else "moment"
    print(f"[{method_label}]")
    best_index = analysis[f"best_index{suffix}"]
    if best_index is None:
        print("  No finite SNR gain value; the best point is undefined.")
        return
    print(f"  Best Voltage (V): {analysis[f'best_voltage{suffix}']}")
    print(f"  Best Pump Frequency (GHz): {analysis[f'best_pump_frequency{suffix}']}")
    print(f"  Best Pump Amplitude (a.u.): {analysis[f'best_pump_amplitude{suffix}']}")
    print(f"  Best SNR Gain (dB): {analysis[f'best_snr_gain_db{suffix}']:.3f}")
    print(f"  SNR (pump on) at Best Point: {analysis[f'best_snr_on{suffix}']:.4f}")
    print(
        "  SNR (pump off) at Best Point: "
        f"{analysis[f'snr_off{suffix}'][best_index[0], best_index[1]]:.4f}"
    )
    n_voltages, n_frequencies, n_amplitudes = analysis[f"snr_gain_db{suffix}"].shape
    if best_index[1] in (0, n_frequencies - 1) or best_index[2] in (
        0,
        n_amplitudes - 1,
    ):
        print(
            "  [WARN] The best point lies on the sweep boundary; "
            "consider extending the sweep range."
        )
    if n_voltages > 1 and best_index[0] in (0, n_voltages - 1):
        print(
            "  [WARN] The best voltage lies on the sweep boundary; "
            "consider extending the voltage range."
        )


def _display_gauss_fit_summary(analysis) -> None:
    # A point where BIC rejected the two-component model carries the moment
    # noise, so knowing how often that happened tells you how much of the
    # Gaussian-fit map is actually different from the moment map.
    print("[Gaussian fit diagnostics]")
    for name, key in (
        ("pump off", "n_components_off_gauss"),
        ("pump on", "n_components_on_gauss"),
    ):
        n_components = np.asarray(analysis[key])
        total = n_components.size
        if total == 0:
            continue
        resolved = int(np.count_nonzero(n_components == 2))
        skipped = int(np.count_nonzero(n_components == 0))
        print(
            f"  {name}: two components accepted at {resolved}/{total} points "
            f"({100 * resolved / total:.1f}%)"
            + (f", {skipped} not fitted" if skipped else "")
        )
    p_thermal = np.asarray(analysis["p_thermal_on_gauss"], dtype=float)
    finite = p_thermal[np.isfinite(p_thermal)]
    if finite.size > 0:
        print(
            f"  fitted p_thermal (pump on): median {np.median(finite):.4f}, "
            f"range {finite.min():.4f}-{finite.max():.4f}"
        )


def plot_impa_pump_pulse_calibration_iq_shots(
    *,
    result: Result,
    voltage_index: int,
    pump_frequency_index: int,
    pump_amplitude_index: int,
):
    # Built on demand from the stored single shots; the figure is not
    # added to `result.figures` (see the module description).
    target, point = _select_iq_point_from_result(
        result,
        voltage_index=voltage_index,
        pump_frequency_index=pump_frequency_index,
        pump_amplitude_index=pump_amplitude_index,
    )
    return _make_iq_shots_figure(
        target=target, point=point, at_best=False, method="moment"
    )


def display_impa_pump_pulse_calibration_iq_shots(
    *,
    result: Result,
    voltage_index: int,
    pump_frequency_index: int,
    pump_amplitude_index: int,
    plot: bool = True,
):
    target, point = _select_iq_point_from_result(
        result,
        voltage_index=voltage_index,
        pump_frequency_index=pump_frequency_index,
        pump_amplitude_index=pump_amplitude_index,
    )
    if plot:
        _make_iq_shots_figure(
            target=target, point=point, at_best=False, method="moment"
        ).show()
    print("IMPA Pump Pulse Calibration - IQ Shots:")
    print(f"Index (voltage, frequency, amplitude): {point['index']}")
    print(f"Voltage (V): {point['voltage']}")
    print(f"Pump Frequency (GHz): {point['pump_frequency']}")
    print(f"Pump Amplitude (a.u.): {point['pump_amplitude']}")
    print("[moment]")
    print(f"  SNR (pump off): {point['snr_off']:.4f}")
    print(f"  SNR (pump on): {point['snr_on']:.4f}")
    print(f"  SNR Gain (dB): {point['snr_gain_db']:.3f}")
    print("[Gaussian fit]")
    print(
        f"  SNR (pump off): {point['snr_off_gauss']:.4f} "
        f"(n_components={point['n_components_off_gauss']})"
    )
    print(
        f"  SNR (pump on): {point['snr_on_gauss']:.4f} "
        f"(n_components={point['n_components_on_gauss']})"
    )
    print(f"  SNR Gain (dB): {point['snr_gain_db_gauss']:.3f}")
    print(
        f"  fitted p_thermal: {point['p_thermal_off_gauss']:.4f} (pump off), "
        f"{point['p_thermal_on_gauss']:.4f} (pump on)"
    )


def impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    pump_buffer_time: float,
    pump_ramptime: float,
    readout_amplitude: float | None,
    dc_bias_controller,
    voltage_range: ArrayLike,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(
        dc_bias_controller, np.atleast_1d(np.asarray(voltage_range, dtype=float))
    )
    run_result = run_impa_pump_pulse_calibration(
        exp=exp,
        target=target,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        readout_amplitude=readout_amplitude,
        dc_bias_controller=dc_bias_controller,
        voltage_range=voltage_range,
        shot_interval=shot_interval,
        n_shots=n_shots,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    analysis_result = analyze_impa_pump_pulse_calibration(run_result=run_result)
    figures = plot_impa_pump_pulse_calibration(
        run_result=run_result,
        analysis_result=analysis_result,
    )
    data = {
        "pump_frequency_range": run_result["pump_frequency_range"],
        "pump_amplitude_range": run_result["pump_amplitude_range"],
        "voltage_range": run_result["voltage_range"],
        "raw": run_result,
        "analysis": analysis_result,
    }
    result = Result(
        data=data,
        figures=figures,
    )
    display_impa_pump_pulse_calibration(result=result, plot=plot)
    return result
