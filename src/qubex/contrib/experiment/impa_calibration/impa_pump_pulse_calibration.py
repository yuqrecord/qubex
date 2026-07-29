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
schedules have identical timing and capture windows.

- Signal: distance in the IQ plane between the shot means of the
  readout-on and readout-off measurements, evaluated separately for
  pump off and pump on:
  ``signal = |mean(readout on) - mean(readout off)|``.
- Noise: complex RMS scatter of the readout-on shots around their shot
  mean (IQ-plane RMS, ``sqrt(Var_I + Var_Q)`` with ``ddof=1``),
  evaluated separately for pump off and pump on. This is the IQ-plane
  RMS SNR convention, not the projected discrimination SNR.
- SNR: ``signal / noise`` (amplitude SNR). The SNR gain is reported in
  dB as ``20 * log10(snr_on / snr_off)``, which equals ``10 * log10``
  of the power SNR ratio.

The pump-off reference (A, B) is measured once per pump frequency row,
so ``snr_off`` is a vector over frequency and slow readout drift between
rows is compensated. The pump frequency is applied per row with
``exp.modified_frequencies({mux: pump_frequency})``, where the mux label
is the PUMP channel label; the pump-on schedules depend only on the pump
amplitude and are reused across rows.

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
        dc_bias_controller=dc_bias_controller,
        voltage=0.1,
        shot_interval=150 * 1024,
        n_shots=1024,
    )
"""

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

import qubex as qx
import qubex.visualization as viz
from qubex.experiment.models import Result


def _validate_dc_bias_controller(dc_bias_controller) -> None:
    if not hasattr(dc_bias_controller, "validate_voltage"):
        raise ValueError(
            "The provided dc_bias_controller does not have a 'validate_voltage' method."
        )
    if not hasattr(dc_bias_controller, "voltage"):
        raise ValueError(
            "The provided dc_bias_controller does not have a 'voltage' method."
        )


def _validate_voltage_range(dc_bias_controller, voltage_range: ArrayLike) -> None:
    for voltage in voltage_range:
        dc_bias_controller.validate_voltage(voltage)


def _build_pulse_schedule(
    exp: qx.Experiment,
    *,
    target: str,
    send_readout_pulse: bool,
    send_pump_pulse: bool,
    pump_buffer_time: float,
    pump_amplitude: float,
    pump_ramptime: float,
    plot_pulse_schedule: bool = True,
) -> qx.PulseSchedule:
    qubit = exp.ctx.resolve_qubit_label(target)
    readout = exp.ctx.resolve_read_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    readout_pulse = exp.readout(
        target,
        amplitude=None if send_readout_pulse else 0.0,
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


def run_impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    pump_buffer_time: float,
    pump_ramptime: float,
    dc_bias_controller,
    voltage: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, [voltage])
    pump_frequency_range = np.atleast_1d(np.asarray(pump_frequency_range, dtype=float))
    pump_amplitude_range = np.atleast_1d(np.asarray(pump_amplitude_range, dtype=float))
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
    try:
        dc_bias_controller.voltage(mux=mux, voltage_in_volts=voltage)
        for frequency in tqdm(
            pump_frequency_range,
            desc=f"IMPA pump pulse calibration : {target}",
        ):
            with exp.modified_frequencies({mux: float(frequency)}):
                # Pump-off reference per frequency row (drift tracking).
                results_readout_off_pump_off.append(execute(ps_readout_off_pump_off))
                results_readout_on_pump_off.append(execute(ps_readout_on_pump_off))
                row_readout_off_pump_on = []
                row_readout_on_pump_on = []
                for ps_readout_off_pump_on, ps_readout_on_pump_on in zip(
                    ps_readout_off_pump_on_list,
                    ps_readout_on_pump_on_list,
                    strict=True,
                ):
                    row_readout_off_pump_on.append(execute(ps_readout_off_pump_on))
                    row_readout_on_pump_on.append(execute(ps_readout_on_pump_on))
                results_readout_off_pump_on.append(row_readout_off_pump_on)
                results_readout_on_pump_on.append(row_readout_on_pump_on)
    finally:
        dc_bias_controller.voltage(mux=mux, voltage_in_volts=original_voltage)
    return {
        "target": target,
        "qubit": qubit,
        "mux": mux,
        "pump_frequency_range": pump_frequency_range,
        "pump_amplitude_range": pump_amplitude_range,
        "pump_buffer_time": pump_buffer_time,
        "pump_ramptime": pump_ramptime,
        "voltage": voltage,
        "shot_interval": shot_interval,
        "n_shots": n_shots,
        "results_readout_off_pump_off": results_readout_off_pump_off,
        "results_readout_on_pump_off": results_readout_on_pump_off,
        "results_readout_off_pump_on": results_readout_off_pump_on,
        "results_readout_on_pump_on": results_readout_on_pump_on,
    }


def analyze_impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    run_result,
):
    pump_frequency_range = np.atleast_1d(np.asarray(pump_frequency_range, dtype=float))
    pump_amplitude_range = np.atleast_1d(np.asarray(pump_amplitude_range, dtype=float))
    qubit = exp.ctx.resolve_qubit_label(target)
    iq_readout_off_pump_off = np.array(
        [
            _extract_iq_shots(res, qubit=qubit)
            for res in run_result["results_readout_off_pump_off"]
        ]
    )
    iq_readout_on_pump_off = np.array(
        [
            _extract_iq_shots(res, qubit=qubit)
            for res in run_result["results_readout_on_pump_off"]
        ]
    )
    iq_readout_off_pump_on = np.array(
        [
            [_extract_iq_shots(res, qubit=qubit) for res in row]
            for row in run_result["results_readout_off_pump_on"]
        ]
    )
    iq_readout_on_pump_on = np.array(
        [
            [_extract_iq_shots(res, qubit=qubit) for res in row]
            for row in run_result["results_readout_on_pump_on"]
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
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_off = signal_off / noise_off
        snr_on = signal_on / noise_on
        snr_gain = snr_on / snr_off[:, np.newaxis]
        snr_gain_db = 20 * np.log10(snr_gain)
    invalid_reference = ~np.isfinite(snr_off) | (snr_off <= 0)
    if np.any(invalid_reference):
        print(
            "[WARN] invalid pump-off reference SNR at "
            f"{pump_frequency_range[invalid_reference]} GHz; "
            "the SNR gain is NaN at the affected frequencies."
        )
        snr_gain[invalid_reference, :] = np.nan
        snr_gain_db[invalid_reference, :] = np.nan
    finite_snr_gain_db = np.where(np.isfinite(snr_gain_db), snr_gain_db, np.nan)
    if np.all(np.isnan(finite_snr_gain_db)):
        print("[WARN] no finite SNR gain value; the best point is undefined.")
        best_index = None
        best_pump_frequency = None
        best_pump_amplitude = None
        best_snr_gain_db = None
        best_snr_on = None
    else:
        best_index = np.unravel_index(
            np.nanargmax(finite_snr_gain_db), finite_snr_gain_db.shape
        )
        best_pump_frequency = float(pump_frequency_range[best_index[0]])
        best_pump_amplitude = float(pump_amplitude_range[best_index[1]])
        best_snr_gain_db = float(snr_gain_db[best_index])
        best_snr_on = float(snr_on[best_index])
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
        "snr_off": snr_off,
        "snr_on": snr_on,
        "snr_gain": snr_gain,
        "snr_gain_db": snr_gain_db,
        "best_index": best_index,
        "best_pump_frequency": best_pump_frequency,
        "best_pump_amplitude": best_pump_amplitude,
        "best_snr_gain_db": best_snr_gain_db,
        "best_snr_on": best_snr_on,
    }


def plot_impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    analysis_result,
):
    pump_frequency_range = np.atleast_1d(np.asarray(pump_frequency_range, dtype=float))
    pump_amplitude_range = np.atleast_1d(np.asarray(pump_amplitude_range, dtype=float))
    snr_gain_db = analysis_result["snr_gain_db"]
    snr_on = analysis_result["snr_on"]
    snr_off = analysis_result["snr_off"]
    best_pump_frequency = analysis_result["best_pump_frequency"]
    best_pump_amplitude = analysis_result["best_pump_amplitude"]

    # Plot SNR gain heatmap
    fig_snr_gain = viz.make_figure()
    fig_snr_gain.add_heatmap(
        x=pump_amplitude_range,
        y=pump_frequency_range,
        z=snr_gain_db,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="SNR gain (dB)", side="right")),
    )
    if best_pump_amplitude is not None:
        fig_snr_gain.add_scatter(
            x=[best_pump_amplitude],
            y=[best_pump_frequency],
            mode="markers",
            marker=dict(symbol="x", color="red", size=10),
            name="Best",
            showlegend=False,
        )
    fig_snr_gain.update_layout(
        title=f"IMPA Pump Pulse Calibration - SNR Gain for {target}",
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
        z=snr_on,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="SNR (pump on)", side="right")),
    )
    fig_snr_on.update_layout(
        title=f"IMPA Pump Pulse Calibration - Pump-on SNR for {target}",
        xaxis_title="Pump amplitude (a.u.)",
        yaxis_title="Pump frequency (GHz)",
        width=600,
        height=400,
    )
    # Plot pump-off reference SNR
    fig_snr_off = viz.make_figure()
    fig_snr_off.add_scatter(
        x=pump_frequency_range,
        y=snr_off,
        mode="lines+markers",
        name="SNR (pump off)",
    )
    fig_snr_off.update_layout(
        title=f"IMPA Pump Pulse Calibration - Pump-off Reference SNR for {target}",
        xaxis_title="Pump frequency (GHz)",
        yaxis_title="SNR (pump off)",
        width=600,
        height=400,
    )
    return {
        "snr_gain": fig_snr_gain,
        "snr_on": fig_snr_on,
        "snr_off": fig_snr_off,
    }


def display_impa_pump_pulse_calibration(
    *,
    result: Result,
    plot: bool = True,
):
    if plot:
        for fig in (result.figures or {}).values():
            fig.show()
    analysis = result.data["analysis"]
    print("IMPA Pump Pulse Calibration Result:")
    best_index = analysis["best_index"]
    if best_index is None:
        print("No finite SNR gain value; the best point is undefined.")
        return
    print(f"Best Pump Frequency (GHz): {analysis['best_pump_frequency']}")
    print(f"Best Pump Amplitude (a.u.): {analysis['best_pump_amplitude']}")
    print(f"Best SNR Gain (dB): {analysis['best_snr_gain_db']:.3f}")
    print(f"SNR (pump on) at Best Point: {analysis['best_snr_on']:.4f}")
    print(f"SNR (pump off) at Best Frequency: {analysis['snr_off'][best_index[0]]:.4f}")
    n_frequencies, n_amplitudes = analysis["snr_gain_db"].shape
    if best_index[0] in (0, n_frequencies - 1) or best_index[1] in (
        0,
        n_amplitudes - 1,
    ):
        print(
            "[WARN] The best point lies on the sweep boundary; "
            "consider extending the sweep range."
        )


def impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    pump_buffer_time: float,
    pump_ramptime: float,
    dc_bias_controller,
    voltage: float,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, [voltage])
    run_result = run_impa_pump_pulse_calibration(
        exp=exp,
        target=target,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        dc_bias_controller=dc_bias_controller,
        voltage=voltage,
        shot_interval=shot_interval,
        n_shots=n_shots,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    pump_frequency_range = run_result["pump_frequency_range"]
    pump_amplitude_range = run_result["pump_amplitude_range"]
    analysis_result = analyze_impa_pump_pulse_calibration(
        exp=exp,
        target=target,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        run_result=run_result,
    )
    figures = plot_impa_pump_pulse_calibration(
        exp=exp,
        target=target,
        pump_frequency_range=pump_frequency_range,
        pump_amplitude_range=pump_amplitude_range,
        analysis_result=analysis_result,
    )
    data = {
        "pump_frequency_range": pump_frequency_range,
        "pump_amplitude_range": pump_amplitude_range,
        "voltage": voltage,
        "raw": run_result,
        "analysis": analysis_result,
    }
    result = Result(
        data=data,
        figures=figures,
    )
    display_impa_pump_pulse_calibration(result=result, plot=plot)
    return result
