"""
This module provides functions to perform IMPA voltage calibration.

This module assumes that an external instance `dc_bias_controller` is provided, which has the following methods:
- `validate_voltage(voltage: float)`: Validates if the given voltage is within the acceptable range for the DC bias controller.
- `voltage(mux: str, voltage_in_volts: float = None)`: Gets or sets the voltage for the specified multiplexer (mux).
"""

import qubex as qx
from qubex.experiment.models import Result
import qubex.visualization as viz
import numpy as np
from numpy.typing import ArrayLike
import scipy


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


def run_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    dc_bias_controller,
    voltage_range: ArrayLike,
    shot_interval: float,
    n_shots: int,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, voltage_range)
    qubit = exp.ctx.resolve_qubit_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    blank_schedule = qx.PulseSchedule()
    with blank_schedule as ps:
        ps.add(qubit, qx.pulse.Blank(0))
    original_voltage = dc_bias_controller.voltage(mux=mux)
    results = []
    for voltage in voltage_range:
        dc_bias_controller.voltage(mux=mux, voltage_in_volts=voltage)
        result = exp.measure(
            sequence=blank_schedule,
            mode="avg",
            time_integration=True,
            shot_interval=shot_interval,
            n_shots=n_shots,
        )
        results.append(result)
    dc_bias_controller.voltage(mux=mux, voltage_in_volts=original_voltage)
    return results


def analyze_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    voltage_range: ArrayLike,
    run_result,
):
    iq_data = np.array([res.data[target].raw for res in run_result])
    amplitude = np.abs(iq_data)
    phase = np.angle(iq_data)
    min_val = 1e20
    phase_offset = 0.0
    for start_phase in np.arange(-np.pi, np.pi, np.pi / 32):
        buf = np.abs(
            np.diff(np.where(phase > start_phase, phase - 2 * np.pi, phase))
        ).sum()
        if buf < min_val:
            min_val = buf
            phase_offset = start_phase
    shifted_phase = np.where(phase > phase_offset, phase - 2 * np.pi, phase)
    peak_idxes, _ = scipy.signal.find_peaks(
        -shifted_phase, height=-0.5 * (np.max(shifted_phase) + np.min(shifted_phase))
    )
    num_peaks = len(peak_idxes)
    peak_shifted_phases = shifted_phase[peak_idxes] if num_peaks > 0 else np.array([])
    peak_voltages = voltage_range[peak_idxes] if num_peaks > 0 else np.array([])
    peak_phases = phase[peak_idxes] if num_peaks > 0 else np.array([])
    best_peak_idx = np.argmin(np.abs(peak_voltages)) if num_peaks > 0 else None
    best_peak_shifted_phase = (
        peak_shifted_phases[best_peak_idx] if best_peak_idx is not None else None
    )
    best_peak_voltage = (
        peak_voltages[best_peak_idx] if best_peak_idx is not None else None
    )
    best_peak_phase = peak_phases[best_peak_idx] if best_peak_idx is not None else None
    return {
        "iq_data": iq_data,
        "amplitude": amplitude,
        "phase": phase,
        "phase_offset": phase_offset,
        "shifted_phase": shifted_phase,
        "peak_idxes": peak_idxes,
        "peak_shifted_phases": peak_shifted_phases,
        "peak_voltages": peak_voltages,
        "peak_phases": peak_phases,
        "best_peak_idx": best_peak_idx,
        "best_peak_shifted_phase": best_peak_shifted_phase,
        "best_peak_voltage": best_peak_voltage,
        "best_peak_phase": best_peak_phase,
    }


def plot_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    voltage_range: ArrayLike,
    analysis_result,
):
    amplitude = analysis_result["amplitude"]
    phase = analysis_result["phase"]
    phase_offset = analysis_result["phase_offset"]
    shifted_phase = analysis_result["shifted_phase"]
    best_peak_voltage = analysis_result["best_peak_voltage"]

    # Plot amplitude
    fig_amplitude = viz.make_figure()
    fig_amplitude.add_scatter(
        x=voltage_range,
        y=amplitude,
        mode="lines+markers",
        name="Amplitude (a.u.)",
    )
    fig_amplitude.update_layout(
        title=f"IMPA Voltage Calibration - Amplitude for {target}",
        xaxis_title="Voltage (V)",
        yaxis_title="Amplitude (a.u.)",
        width=600,
        height=400,
    )
    # Plot IQ-plane
    fig_iq = viz.make_figure()
    fig_iq.add_scatter(
        x=amplitude * np.cos(phase),
        y=amplitude * np.sin(phase),
        mode="lines+markers",
        name="IQ-plane",
    )
    fig_iq.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=np.max(amplitude) * np.cos(phase_offset),
        y1=np.max(amplitude) * np.sin(phase_offset),
        line=dict(color="black", dash="dash"),
        name="Phase Offset",
    )
    fig_iq.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_iq.update_layout(
        title=f"IMPA Voltage Calibration - IQ-plane for {target}",
        xaxis_title="I (a.u.)",
        yaxis_title="Q (a.u.)",
        width=600,
        height=400,
    )
    # Plot phase
    fig_phase = viz.make_figure()
    fig_phase.add_scatter(
        x=voltage_range,
        y=shifted_phase,
        mode="lines+markers",
        name="Phase (rad)",
    )
    if best_peak_voltage is not None:
        fig_phase.add_vline(
            x=best_peak_voltage,
            line=dict(color="black", dash="dash"),
            annotation_text=f"Best Peak Voltage: {best_peak_voltage:.3f} V",
            annotation_position="top right",
        )
    fig_phase.update_layout(
        title=f"IMPA Voltage Calibration - Phase for {target}",
        xaxis_title="Voltage (V)",
        yaxis_title="Phase (rad)",
        width=600,
        height=400,
    )
    # Plot phase difference
    fig_phase_diff = viz.make_figure()
    fig_phase_diff.add_scatter(
        x=0.5 * (voltage_range[:-1] + voltage_range[1:]),
        y=np.diff(shifted_phase),
        mode="lines+markers",
        name="Phase Difference (rad)",
    )
    fig_phase_diff.update_layout(
        title=f"IMPA Voltage Calibration - Phase Difference for {target}",
        xaxis_title="Voltage (V)",
        yaxis_title="Phase Difference (rad)",
        width=600,
        height=400,
    )
    return {
        "amplitude": fig_amplitude,
        "iq_plane": fig_iq,
        "phase": fig_phase,
        "phase_difference": fig_phase_diff,
    }


def display_impa_voltage_calibration(
    *,
    result: Result,
    plot: bool = True,
):
    if plot:
        for fig in (result.figures or {}).values():
            fig.show()
    print("IMPA Voltage Calibration Result:")
    print(f"Peak Voltages (V): {result.data['analysis']['peak_voltages']}")
    print(
        f"Peak Shifted Phases (rad): {result.data['analysis']['peak_shifted_phases']}"
    )
    print(f"Best Peak Voltage (V): {result.data['analysis']['best_peak_voltage']}")
    print(
        f"Best Peak Shifted Phase (rad): {result.data['analysis']['best_peak_shifted_phase']}"
    )


def impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    dc_bias_controller,
    voltage_range: ArrayLike,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, voltage_range)
    run_result = run_impa_voltage_calibration(
        exp=exp,
        target=target,
        dc_bias_controller=dc_bias_controller,
        voltage_range=voltage_range,
        shot_interval=shot_interval,
        n_shots=n_shots,
    )
    analysis_result = analyze_impa_voltage_calibration(
        exp=exp,
        target=target,
        voltage_range=voltage_range,
        run_result=run_result,
    )
    figures = plot_impa_voltage_calibration(
        exp=exp,
        target=target,
        voltage_range=voltage_range,
        analysis_result=analysis_result,
    )
    data = {
        "voltage_range": voltage_range,
        "raw": run_result,
        "analysis": analysis_result,
    }
    result = Result(
        data=data,
        figures=figures,
    )
    display_impa_voltage_calibration(result=result, plot=plot)
    return result
