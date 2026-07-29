"""
This module provides functions to perform IMPA pump pulse calibration.

This module assumes that an external instance `dc_bias_controller` is provided, which has the following methods:
- `validate_voltage(voltage: float)`: Validates if the given voltage is within the acceptable range for the DC bias controller.
- `voltage(mux: str, voltage_in_volts: float = None)`: Gets or sets the voltage for the specified multiplexer (mux).
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
    readout = exp.ctx.resolve_readout_label(target)
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    schedule = qx.PulseSchedule()
    readout_pulse = exp.readout(target)
    readout_amplitude = None if send_readout_pulse else 0.0
    pump_amplitude = None if send_pump_pulse else 0.0
    with schedule as ps:
        ps.add(
            mux,
            qx.pulse.FlatTop(
                amplitude=pump_amplitude,
                duration=readout_pulse.duration + 2 * pump_buffer_time,
                tau=pump_ramptime,
            ),
        )
        ps.add(readout, qx.pulse.Blank(duration=pump_buffer_time))
        ps.add(readout, readout_pulse)
    if plot_pulse_schedule:
        schedule.plot()
    return schedule


def run_impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    dc_bias_controller,
    voltage: float,
    n_shots: int,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, [voltage])
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    original_voltage = dc_bias_controller.voltage(mux=mux)
    dc_bias_controller.voltage(mux=mux, voltage_in_volts=voltage)
    pass
    dc_bias_controller.voltage(mux=mux, voltage_in_volts=original_voltage)


def analyze_impa_pump_pulse_calibration():
    pass


def plot_impa_pump_pulse_calibration():
    pass


def display_impa_pump_pulse_calibration():
    pass


def impa_pump_pulse_calibration(
    exp: qx.Experiment,
    *,
    target: str,
    pump_frequency_range: ArrayLike,
    pump_amplitude_range: ArrayLike,
    dc_bias_controller,
    voltage: float,
    shot_interval: float,
    n_shots: int,
):
    _validate_dc_bias_controller(dc_bias_controller)
    _validate_voltage_range(dc_bias_controller, [voltage])
