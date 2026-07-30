"""
This module provides functions to perform IMPA voltage calibration.

The voltage sweep is performed in the order of `voltage_range` (forward sweep).
If `enable_backward_sweep` is True, the sweep is repeated in the reversed order
(backward sweep) to check for hysteresis, and the forward and backward sweeps
are plotted together in the same figures.

This module assumes that an external instance `dc_bias_controller` is provided, which has the following methods:
- `validate_voltage(voltage: float)`: Validates if the given voltage is within the acceptable range for the DC bias controller.
- `voltage(mux: str, voltage_in_volts: float = None)`: Gets or sets the voltage for the specified multiplexer (mux).
- `voltage_sweep_rate(mux: str, rate_in_volts_per_second: float = None)`: Gets or sets the voltage sweep rate for the specified multiplexer (mux).
"""

import qubex as qx
from qubex.experiment.models import Result
import qubex.visualization as viz
import numpy as np
from numpy.typing import ArrayLike


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


def run_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    dc_bias_controller,
    voltage_range: ArrayLike,
    enable_backward_sweep: bool,
    shot_interval: float,
    n_shots: int,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, voltage_range)
    voltage_range = np.atleast_1d(np.asarray(voltage_range, dtype=float))
    qubit = exp.ctx.resolve_qubit_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    blank_schedule = qx.PulseSchedule()
    with blank_schedule as ps:
        ps.add(qubit, qx.pulse.Blank(0))
    original_voltage = dc_bias_controller.voltage(mux=mux)
    print(f"Original voltage for mux '{mux}': {original_voltage} V")
    print(
        f"Voltage sweep rate for mux '{mux}': {dc_bias_controller.voltage_sweep_rate(mux=mux)} V/s"
    )

    def measure_at(voltage):
        dc_bias_controller.voltage(mux=mux, voltage_in_volts=voltage)
        return exp.measure(
            sequence=blank_schedule,
            mode="avg",
            time_integration=True,
            shot_interval=shot_interval,
            n_shots=n_shots,
        )

    results_forward = [measure_at(voltage) for voltage in voltage_range]
    results_backward = (
        [measure_at(voltage) for voltage in voltage_range[::-1]]
        if enable_backward_sweep
        else None
    )
    dc_bias_controller.voltage(mux=mux, voltage_in_volts=original_voltage)
    print(
        f"Voltage for mux '{mux}' restored to original value: {dc_bias_controller.voltage(mux=mux)} V"
    )
    return {
        "results_forward": results_forward,
        "results_backward": results_backward,
    }


def analyze_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    voltage_range: ArrayLike,
    run_result,
):
    iq_data_forward = np.array(
        [res.data[target].raw for res in run_result["results_forward"]]
    )
    amplitude_forward = np.abs(iq_data_forward)
    phase_forward = np.angle(iq_data_forward)
    results_backward = run_result["results_backward"]
    if results_backward is not None:
        iq_data_backward = np.array([res.data[target].raw for res in results_backward])
        amplitude_backward = np.abs(iq_data_backward)
        phase_backward = np.angle(iq_data_backward)
        phase_list = [phase_forward, phase_backward]
    else:
        iq_data_backward = None
        amplitude_backward = None
        phase_backward = None
        phase_list = [phase_forward]
    # Find a common branch cut so that both sweeps share the same phase offset.
    min_val = 1e20
    phase_offset = 0.0
    for start_phase in np.arange(-np.pi, np.pi, np.pi / 32):
        buf = sum(
            np.abs(
                np.diff(np.where(phase > start_phase, phase - 2 * np.pi, phase))
            ).sum()
            for phase in phase_list
        )
        if buf < min_val:
            min_val = buf
            phase_offset = start_phase
    shifted_phase_forward = np.where(
        phase_forward > phase_offset, phase_forward - 2 * np.pi, phase_forward
    )
    shifted_phase_backward = (
        np.where(
            phase_backward > phase_offset, phase_backward - 2 * np.pi, phase_backward
        )
        if phase_backward is not None
        else None
    )
    return {
        "iq_data_forward": iq_data_forward,
        "amplitude_forward": amplitude_forward,
        "phase_forward": phase_forward,
        "shifted_phase_forward": shifted_phase_forward,
        "iq_data_backward": iq_data_backward,
        "amplitude_backward": amplitude_backward,
        "phase_backward": phase_backward,
        "shifted_phase_backward": shifted_phase_backward,
        "phase_offset": phase_offset,
    }


def plot_impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    voltage_range: ArrayLike,
    analysis_result,
):
    voltage_range = np.atleast_1d(np.asarray(voltage_range, dtype=float))
    backward_voltage_range = voltage_range[::-1]
    amplitude_forward = analysis_result["amplitude_forward"]
    phase_forward = analysis_result["phase_forward"]
    shifted_phase_forward = analysis_result["shifted_phase_forward"]
    amplitude_backward = analysis_result["amplitude_backward"]
    phase_backward = analysis_result["phase_backward"]
    shifted_phase_backward = analysis_result["shifted_phase_backward"]
    phase_offset = analysis_result["phase_offset"]

    # Plot amplitude
    fig_amplitude = viz.make_figure()
    fig_amplitude.add_scatter(
        x=voltage_range,
        y=amplitude_forward,
        mode="lines+markers",
        name="Forward",
    )
    if amplitude_backward is not None:
        fig_amplitude.add_scatter(
            x=backward_voltage_range,
            y=amplitude_backward,
            mode="lines+markers",
            name="Backward",
        )
    fig_amplitude.update_layout(
        title=f"IMPA Voltage Calibration - Amplitude for {target}",
        xaxis_title="Voltage (V)",
        yaxis_title="Amplitude (a.u.)",
        width=600,
        height=400,
    )
    # Plot IQ-plane
    max_amplitude = np.max(amplitude_forward)
    if amplitude_backward is not None:
        max_amplitude = max(max_amplitude, np.max(amplitude_backward))
    fig_iq = viz.make_figure()
    fig_iq.add_scatter(
        x=amplitude_forward * np.cos(phase_forward),
        y=amplitude_forward * np.sin(phase_forward),
        mode="lines+markers",
        name="Forward",
    )
    if amplitude_backward is not None:
        fig_iq.add_scatter(
            x=amplitude_backward * np.cos(phase_backward),
            y=amplitude_backward * np.sin(phase_backward),
            mode="lines+markers",
            name="Backward",
        )
    fig_iq.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_amplitude * np.cos(phase_offset),
        y1=max_amplitude * np.sin(phase_offset),
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
        y=shifted_phase_forward,
        mode="lines+markers",
        name="Forward",
    )
    if shifted_phase_backward is not None:
        fig_phase.add_scatter(
            x=backward_voltage_range,
            y=shifted_phase_backward,
            mode="lines+markers",
            name="Backward",
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
        y=np.diff(shifted_phase_forward),
        mode="lines+markers",
        name="Forward",
    )
    if shifted_phase_backward is not None:
        fig_phase_diff.add_scatter(
            x=0.5 * (backward_voltage_range[:-1] + backward_voltage_range[1:]),
            y=np.diff(shifted_phase_backward),
            mode="lines+markers",
            name="Backward",
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
    print(f"Phase Offset (rad): {result.data['analysis']['phase_offset']}")


def impa_voltage_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    dc_bias_controller,
    voltage_range: ArrayLike,
    enable_backward_sweep: bool,
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
        enable_backward_sweep=enable_backward_sweep,
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
