"""
Control–readout timing (skew) check.

Measure the relative timing between a qubit *control* pulse (drive channel
``Qxx``) and a resonator *readout* pulse (read channel ``RQxx``).

Method
------
A fixed readout pulse is played on the read channel while a short control
pulse is swept across it with a range of time offsets. For each offset the
shot-averaged, time-integrated readout IQ value is measured with
``Experiment.execute``. A single **reference** measurement is taken with no
control pulse; the distance ``|z - z_ref|`` in the complex IQ plane is then
plotted against the time offset. The offset at which the distance turns on/off
locates the control-vs-readout timing alignment (the skew).

Sequence layout (one sweep point)::

    Qxx  (control) : [Blank(max(0, +offset))]  control_pulse
    RQxx (readout) : [Blank(max(0, -offset))]  readout_pulse

Both channels start together, so
``control_start - readout_start == time_offset``: a positive offset makes the
control pulse land during/after the readout, a negative offset delays the
readout so the control pulse lands before it.

Unlike ``ac_stark_spectroscopy`` (whose drives must precede the readout), this
experiment deliberately allows the control pulse to overlap the readout — that
overlap is exactly what the skew check probes. For very negative offsets the
readout is delayed, so an excited qubit waits longer before readout and decays
slightly (T1); keep the offsets within tens–low-hundreds of ns to make this
negligible.

This module mirrors the structure and conventions of
``ac_stark_spectroscopy.py`` (measure / analyze / plot / display split,
keyword-only API, Plotly figures via ``qubex.visualization``, and an
off-by-default ``reset_awg_and_capunits`` gate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.pulse import Blank, FlatTop, PulseSchedule, Waveform

TIME_GRID_TOLERANCE = 1e-6


def _sampling_period() -> float:
    return float(Waveform.SAMPLING_PERIOD)


def _validate_finite_scalar(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"`{name}` must be a real number, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value!r}.")


def _validate_time_on_grid(name: str, value: float) -> None:
    _validate_finite_scalar(name, value)
    dt = _sampling_period()
    if abs(value - round(value / dt) * dt) > TIME_GRID_TOLERANCE:
        raise ValueError(
            f"`{name}` ({value} ns) must be a multiple of the {dt} ns sampling period."
        )


def _validate_amplitude(name: str, value: float, *, allow_zero: bool) -> None:
    _validate_finite_scalar(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be within [0, 1], got {value}.")
    if not allow_zero and value == 0.0:
        raise ValueError(f"`{name}` must be positive, got {value}.")


def _validate_flat_top(
    name: str,
    *,
    duration: float,
    ramptime: float,
) -> None:
    if duration <= 0:
        raise ValueError(f"`{name}_duration` must be positive, got {duration}.")
    if ramptime <= 0:
        raise ValueError(f"`{name}_ramptime` must be positive, got {ramptime}.")
    if duration < 2 * ramptime:
        raise ValueError(
            f"`{name}_duration` ({duration} ns) must be at least twice "
            f"`{name}_ramptime` ({ramptime} ns)."
        )


def _validate_axis(name: str, values: float | ArrayLike) -> None:
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a scalar or a 1D sequence.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.unique(array).size != array.size:
        raise ValueError(f"`{name}` must not contain duplicate values.")


def _validate_n_shots(n_shots: int) -> None:
    if isinstance(n_shots, bool) or not isinstance(n_shots, (int, np.integer)):
        raise TypeError(f"`n_shots` must be an integer, got {n_shots!r}.")
    if n_shots <= 0:
        raise ValueError(f"`n_shots` must be positive, got {n_shots}.")


def _extract_readout_iq(
    measure_result: MultipleMeasureResult,
    *,
    read_label: str,
    target: str,
) -> complex:
    """
    Return the time-integrated readout IQ value from a measurement result.

    The skew-check schedule plays a single readout pulse, so exactly one
    capture window is expected; the last capture is used defensively in case
    the compiler emits additional windows.
    """
    available = list(measure_result.data)
    if read_label in available:
        data_target = read_label
    elif target in available:
        data_target = target
    else:
        raise KeyError(
            f"Neither {read_label=} nor {target=} exists in the measurement "
            f"result. Available targets: {available}"
        )
    captures = measure_result.data[data_target]
    if len(captures) == 0:
        raise RuntimeError(f"No capture windows returned for {data_target}.")
    return complex(captures[-1].kerneled)


def skew_check_qr_sequence(
    exp: Experiment,
    *,
    target: str,
    readout_pulse: Waveform,
    control_pulse: Waveform | None,
    time_offset: float,
) -> PulseSchedule:
    """
    Build the skew-check pulse schedule for one sweep point.

    Parameters
    ----------
    exp : Experiment
        Experiment object providing the hardware context.
    target : str
        Target qubit name or label.
    readout_pulse : Waveform
        Resonator readout pulse, played on the read channel.
    control_pulse : Waveform | None
        Qubit control pulse, played on the drive channel. ``None`` builds the
        reference schedule (readout only, no control pulse).
    time_offset : float
        Control pulse start relative to the readout pulse start (ns). Positive
        values delay the control pulse (it lands during/after the readout);
        negative values delay the readout (the control pulse lands before it).
        Ignored for the reference (``control_pulse is None``).

    Returns
    -------
    PulseSchedule
        Schedule with the readout pulse on the read channel and, unless it is
        the reference, the control pulse on the drive channel.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    resonator = exp.ctx.resolve_read_label(target)

    control_delay = max(0.0, time_offset)
    readout_delay = max(0.0, -time_offset)

    with PulseSchedule() as ps:
        if control_pulse is not None and readout_delay > 0:
            ps.add(resonator, Blank(readout_delay))
        ps.add(resonator, readout_pulse)
        if control_pulse is not None:
            if control_delay > 0:
                ps.add(qubit, Blank(control_delay))
            ps.add(qubit, control_pulse)
    return ps


def run_skew_check_qr(
    exp: Experiment,
    *,
    target: str,
    readout_amplitude: float | None = None,
    control_amplitude: float | None = None,
    control_duration: float | None = None,
    control_ramptime: float | None = None,
    using_pi_pulse: bool = False,
    time_offset_range: float | ArrayLike,
    n_shots: int,
    shot_interval: float,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> dict[str, Any]:
    """
    Acquire the reference and the time-offset sweep.

    Builds the readout pulse (the default readout pulse, with the amplitude
    optionally overridden) and the control pulse (the calibrated pi pulse, or a
    FlatTop from the amplitude/duration/ramptime triple), then measures the
    readout IQ with no control pulse (reference) and, for each time offset,
    with the control pulse applied. Each point is a shot-averaged,
    time-integrated measurement via ``Experiment.execute``.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    readout_amplitude : float | None, optional
        Amplitude override for the resonator readout pulse. When ``None``
        (default), the calibrated default readout amplitude is used. All other
        readout-pulse parameters (duration, ramp time, margins) always use the
        default readout pulse.
    control_amplitude : float | None, optional
        Amplitude of the qubit control pulse. Required unless
        ``using_pi_pulse`` is True.
    control_duration : float | None, optional
        Flat-top duration of the control pulse in ns. Required unless
        ``using_pi_pulse`` is True.
    control_ramptime : float | None, optional
        Ramp time (flat-top tau) of the control pulse in ns. Required unless
        ``using_pi_pulse`` is True.
    using_pi_pulse : bool, optional
        If True, use the calibrated pi pulse (``exp.x180(target)``) as the
        control pulse. Mutually exclusive with the ``control_amplitude`` /
        ``control_duration`` / ``control_ramptime`` triple.
    time_offset_range : float | ArrayLike
        Control pulse time offsets in ns (may be negative). Each value must lie
        on the sampling-period grid.
    n_shots : int
        Number of shots per point.
    shot_interval : float
        Interval between shots in ns.
    plot_pulse_schedule : bool, optional
        If True, plot the pulse schedule of the first offset point.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    dict[str, Any]
        Raw acquisition data with keys ``target``, ``qubit_label``,
        ``read_label``, ``time_offset_range``, ``iq`` (one complex value per
        offset), ``reference_iq``, the echoed pulse/shot parameters
        (``readout_amplitude``, ``control_amplitude``, ``control_duration``,
        ``control_ramptime``, ``using_pi_pulse``, ``n_shots``,
        ``shot_interval``).
    """
    if readout_amplitude is not None:
        _validate_amplitude("readout_amplitude", readout_amplitude, allow_zero=False)
    _validate_axis("time_offset_range", time_offset_range)
    offsets = np.atleast_1d(np.asarray(time_offset_range, dtype=float))
    for offset in offsets:
        _validate_time_on_grid("time_offset", float(offset))
    _validate_n_shots(n_shots)
    _validate_finite_scalar("shot_interval", shot_interval)
    if shot_interval <= 0:
        raise ValueError(f"`shot_interval` must be positive, got {shot_interval}.")

    # Resolve the control pulse: either the calibrated pi pulse or a FlatTop
    # built from the amplitude/duration/ramptime triple (mutually exclusive).
    if using_pi_pulse:
        if (
            control_amplitude is not None
            or control_duration is not None
            or control_ramptime is not None
        ):
            raise ValueError(
                "When `using_pi_pulse=True`, do not also provide the control "
                "amplitude/duration/ramptime."
            )
        control_pulse = exp.x180(target)
    else:
        if (
            control_amplitude is None
            or control_duration is None
            or control_ramptime is None
        ):
            raise ValueError(
                "Provide `control_amplitude`, `control_duration` and "
                "`control_ramptime`, or set `using_pi_pulse=True`."
            )
        _validate_amplitude("control_amplitude", control_amplitude, allow_zero=False)
        _validate_time_on_grid("control_duration", control_duration)
        _validate_time_on_grid("control_ramptime", control_ramptime)
        _validate_flat_top(
            "control",
            duration=control_duration,
            ramptime=control_ramptime,
        )
        control_pulse = FlatTop(
            duration=control_duration,
            amplitude=control_amplitude,
            tau=control_ramptime,
        )

    # Use the default readout pulse, overriding only the amplitude when given
    # (``amplitude=None`` resolves to the calibrated default amplitude).
    readout_pulse = exp.pulse.readout(target, amplitude=readout_amplitude)

    reset_first = (
        bool(reset_awg_and_capunits) if reset_awg_and_capunits is not None else False
    )

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)

    # Reference (no control pulse), measured first so that an optional reset
    # applies to the very first execution.
    reference_schedule = skew_check_qr_sequence(
        exp,
        target=target,
        readout_pulse=readout_pulse,
        control_pulse=None,
        time_offset=0.0,
    )
    reference_result = exp.execute(
        schedule=reference_schedule,
        mode="avg",
        time_integration=True,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=reset_first,
    )
    reference_iq = _extract_readout_iq(
        reference_result,
        read_label=read_label,
        target=target,
    )

    iq_values: list[complex] = []
    for index, offset in enumerate(tqdm(offsets, desc="time offset", leave=False)):
        schedule = skew_check_qr_sequence(
            exp,
            target=target,
            readout_pulse=readout_pulse,
            control_pulse=control_pulse,
            time_offset=float(offset),
        )
        if plot_pulse_schedule and index == 0:
            schedule.plot(
                title=f"Skew check : {target} (offset={offset:g} ns)",
            )
        measure_result = exp.execute(
            schedule=schedule,
            mode="avg",
            time_integration=True,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=False,
        )
        iq_values.append(
            _extract_readout_iq(
                measure_result,
                read_label=read_label,
                target=target,
            )
        )

    return {
        "target": target,
        "qubit_label": qubit_label,
        "read_label": read_label,
        "time_offset_range": offsets,
        "iq": np.asarray(iq_values, dtype=complex),
        "reference_iq": complex(reference_iq),
        "readout_amplitude": readout_amplitude,
        "control_amplitude": control_amplitude,
        "control_duration": control_duration,
        "control_ramptime": control_ramptime,
        "using_pi_pulse": using_pi_pulse,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
    }


def analyze_skew_check_qr(
    *,
    time_offset_range: ArrayLike,
    iq: ArrayLike,
    reference_iq: complex,
) -> dict[str, Any]:
    """
    Compute the IQ distance from the reference for each time offset.

    Parameters
    ----------
    time_offset_range : ArrayLike
        Control pulse time offsets in ns.
    iq : ArrayLike
        Complex readout IQ value for each time offset.
    reference_iq : complex
        Reference readout IQ value (no control pulse).

    Returns
    -------
    dict[str, Any]
        Analysis data with keys ``time_offset_range``, ``iq``,
        ``reference_iq``, ``distance`` (``|iq - reference_iq|``),
        ``max_distance`` and ``max_distance_offset``.
    """
    offsets = np.atleast_1d(np.asarray(time_offset_range, dtype=float))
    iq_array = np.atleast_1d(np.asarray(iq, dtype=complex))
    reference = complex(reference_iq)
    distance = np.abs(iq_array - reference)

    if distance.size:
        max_index = int(np.argmax(distance))
        max_distance = float(distance[max_index])
        max_distance_offset = float(offsets[max_index])
    else:
        max_distance = float("nan")
        max_distance_offset = float("nan")

    return {
        "time_offset_range": offsets,
        "iq": iq_array,
        "reference_iq": reference,
        "distance": distance,
        "max_distance": max_distance,
        "max_distance_offset": max_distance_offset,
    }


def skew_check_qr(
    exp: Experiment,
    *,
    target: str,
    readout_amplitude: float | None = None,
    control_amplitude: float | None = None,
    control_duration: float | None = None,
    control_ramptime: float | None = None,
    using_pi_pulse: bool = False,
    time_offset_range: float | ArrayLike,
    n_shots: int,
    shot_interval: float,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
    reset_awg_and_capunits: bool | None = None,
) -> Result:
    """
    Measure the control–readout timing (skew) of a qubit.

    Sweeps the qubit control pulse across the resonator readout pulse and, for
    each time offset, measures the distance of the shot-averaged,
    time-integrated readout IQ value from a no-control reference.

    Parameters
    ----------
    exp : Experiment
        Experiment object.
    target : str
        Target qubit name or label.
    readout_amplitude : float | None, optional
        Amplitude override for the resonator readout pulse. When ``None``
        (default), the calibrated default readout amplitude is used. All other
        readout-pulse parameters (duration, ramp time, margins) always use the
        default readout pulse.
    control_amplitude : float | None, optional
        Amplitude of the qubit control pulse. Required unless
        ``using_pi_pulse`` is True.
    control_duration : float | None, optional
        Flat-top duration of the control pulse in ns. Required unless
        ``using_pi_pulse`` is True.
    control_ramptime : float | None, optional
        Ramp time (flat-top tau) of the control pulse in ns. Required unless
        ``using_pi_pulse`` is True.
    using_pi_pulse : bool, optional
        If True, use the calibrated pi pulse (``exp.x180(target)``) as the
        control pulse. Mutually exclusive with the ``control_amplitude`` /
        ``control_duration`` / ``control_ramptime`` triple: set this flag or
        provide the three scalars, not both.
    time_offset_range : float | ArrayLike
        Control pulse time offsets in ns (may be negative). Each value must lie
        on the sampling-period grid.
    n_shots : int
        Number of shots per point.
    shot_interval : float
        Interval between shots in ns.
    plot : bool, optional
        If True, display the result figures.
    plot_pulse_schedule : bool, optional
        If True, plot the pulse schedule of the first offset point.
    reset_awg_and_capunits : bool | None, optional
        If True, reset the AWGs and capture units once before the first
        execution. Defaults to off (no implicit hardware operation).

    Returns
    -------
    Result
        Result holding the acquisition/analysis data and the figures.
    """
    raw = run_skew_check_qr(
        exp,
        target=target,
        readout_amplitude=readout_amplitude,
        control_amplitude=control_amplitude,
        control_duration=control_duration,
        control_ramptime=control_ramptime,
        using_pi_pulse=using_pi_pulse,
        time_offset_range=time_offset_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot_pulse_schedule=plot_pulse_schedule,
        reset_awg_and_capunits=reset_awg_and_capunits,
    )

    analysis = analyze_skew_check_qr(
        time_offset_range=raw["time_offset_range"],
        iq=raw["iq"],
        reference_iq=raw["reference_iq"],
    )

    data: dict[str, Any] = {
        "target": raw["target"],
        "qubit_label": raw["qubit_label"],
        "read_label": raw["read_label"],
        "time_offset_range": raw["time_offset_range"],
        "iq": raw["iq"],
        "reference_iq": raw["reference_iq"],
        "distance": analysis["distance"],
        "max_distance": analysis["max_distance"],
        "max_distance_offset": analysis["max_distance_offset"],
        "parameters": {
            "readout_amplitude": raw["readout_amplitude"],
            "control_amplitude": raw["control_amplitude"],
            "control_duration": raw["control_duration"],
            "control_ramptime": raw["control_ramptime"],
            "using_pi_pulse": raw["using_pi_pulse"],
            "n_shots": raw["n_shots"],
            "shot_interval": raw["shot_interval"],
        },
    }

    figures = plot_skew_check_qr(data)
    result = Result(data=data, figures=figures)
    display_skew_check_qr(result, plot=plot)
    return result


def plot_skew_check_qr(data: dict[str, Any]) -> dict[str, go.Figure]:
    """
    Build the result figures for a skew-check dataset.

    Parameters
    ----------
    data : dict[str, Any]
        The ``Result.data`` payload produced by ``skew_check_qr``.

    Returns
    -------
    dict[str, go.Figure]
        Figures keyed by ``"distance_vs_offset"`` (the primary result) and
        ``"iq_plane"`` (a diagnostic of the readout IQ trajectory).
    """
    target = data["target"]
    offsets = data["time_offset_range"]
    distance = data["distance"]
    iq = data["iq"]
    reference_iq = data["reference_iq"]

    figures: dict[str, go.Figure] = {}

    distance_figure = viz.make_figure()
    distance_figure.add_scatter(
        x=offsets,
        y=distance,
        mode="lines+markers",
        name="distance",
    )
    distance_figure.update_layout(
        title=f"Control–readout skew check : {target}",
        xaxis_title="Control pulse time offset (ns)",
        yaxis_title="Distance from reference (arb. units)",
        width=600,
        height=400,
    )
    figures["distance_vs_offset"] = distance_figure

    iq_figure = viz.make_figure()
    iq_figure.add_scatter(
        x=iq.real,
        y=iq.imag,
        mode="markers",
        name="offset sweep",
        marker=dict(
            color=offsets,
            colorscale="Viridis",
            colorbar=dict(title=dict(text="offset (ns)", side="right")),
        ),
    )
    iq_figure.add_scatter(
        x=[reference_iq.real],
        y=[reference_iq.imag],
        mode="markers",
        name="reference",
        marker=dict(color="red", size=10, symbol="x"),
    )
    iq_figure.update_layout(
        title=f"Readout IQ : {target}",
        xaxis_title="I (arb. units)",
        yaxis_title="Q (arb. units)",
        width=600,
        height=400,
    )
    figures["iq_plane"] = iq_figure

    return figures


def display_skew_check_qr(result: Result, *, plot: bool = True) -> None:
    """
    Show the figures and print a summary of a skew-check result.

    Parameters
    ----------
    result : Result
        Result returned by ``skew_check_qr``.
    plot : bool, optional
        If True, display the stored figures.
    """
    data = result.data
    if plot and result.figures:
        for fig in result.figures.values():
            fig.show()

    offsets = data["time_offset_range"]
    distance = data["distance"]

    print(f"=== Control–readout skew check : {data['target']} ===")
    print(f"reference IQ : {data['reference_iq']:.6g}")
    if np.isfinite(data["max_distance"]):
        print(
            f"max distance : {data['max_distance']:.6g} "
            f"at offset {data['max_distance_offset']:g} ns"
        )
    print(f"{'offset (ns)':>12} {'distance':>16}")
    for offset, dist in zip(offsets, distance, strict=True):
        print(f"{offset:>12g} {dist:>16.6g}")
