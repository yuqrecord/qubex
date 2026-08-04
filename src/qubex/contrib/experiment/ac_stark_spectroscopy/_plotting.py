"""
Shared figure primitives for the AC Stark spectroscopy variants.

These builders take plain arrays so that each variant can compose them
with its own sweep axes; the composition itself (which figures exist and
what their axes are) stays in the variant module.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

import qubex.visualization as viz
from qubex.pulse import Waveform


def add_slice_dropdown(
    fig: go.Figure,
    *,
    labels: list[str],
    traces_per_slice: int,
) -> None:
    """Add a dropdown that shows one slice of traces at a time."""
    n_slices = len(labels)
    if n_slices <= 1:
        return
    buttons = []
    for slice_index, label in enumerate(labels):
        visible = [False] * (n_slices * traces_per_slice)
        for trace_offset in range(traces_per_slice):
            visible[slice_index * traces_per_slice + trace_offset] = True
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visible}],
            )
        )
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ]
    )


def spectroscopy_heatmap(
    *,
    target: str,
    state: str,
    x_values: NDArray[np.float64],
    x_title: str,
    frequency_range: NDArray[np.float64],
    signal_slices: list[NDArray[np.float64]],
    f0_slices: list[NDArray[np.float64]],
    valid_slices: list[NDArray[np.bool_]],
    slice_labels: list[str],
) -> go.Figure:
    """Build a signal heatmap over (sweep axis, qubit drive frequency)."""
    order = np.argsort(frequency_range)
    fig = viz.make_figure()
    for slice_index, signal in enumerate(signal_slices):
        visible = slice_index == 0
        fig.add_heatmap(
            z=signal[:, order].T,
            x=x_values,
            y=frequency_range[order],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Signal", side="right")),
            visible=visible,
        )
        valid = valid_slices[slice_index]
        fig.add_scatter(
            x=x_values[valid],
            y=f0_slices[slice_index][valid],
            mode="markers",
            name="Lorentzian fit f0",
            marker=dict(color="white", size=6, line=dict(color="black", width=1)),
            visible=visible,
        )
    add_slice_dropdown(fig, labels=slice_labels, traces_per_slice=2)
    fig.update_layout(
        title=f"AC Stark spectroscopy : {target} : |{state}>",
        xaxis_title=x_title,
        yaxis_title="Qubit drive frequency (GHz)",
        width=600,
        height=400,
    )
    return fig


def single_spectrum_figure(
    *,
    target: str,
    state: str,
    frequency_range: NDArray[np.float64],
    signal: NDArray[np.float64],
    f0: float,
    fit_valid: bool,
) -> go.Figure:
    """Build a single spectroscopy trace with the fitted center marked."""
    order = np.argsort(frequency_range)
    fig = viz.make_figure()
    fig.add_scatter(
        x=frequency_range[order],
        y=signal[order],
        mode="lines+markers",
        name="data",
    )
    if fit_valid:
        fig.add_vline(x=f0, line_dash="dash")
    fig.update_layout(
        title=f"AC Stark spectroscopy : {target} : |{state}>",
        xaxis_title="Qubit drive frequency (GHz)",
        yaxis_title="Signal",
        width=600,
        height=400,
    )
    return fig


def summary_figure(
    *,
    title: str,
    x_values: NDArray[np.float64],
    x_title: str,
    shift_traces: list[NDArray[np.float64]],
    shift_error_traces: list[NDArray[np.float64]],
    trace_names: list[str],
    dispersive_shift: float | None,
    secondary_x_shift: float | None = None,
    secondary_x_title: str = "",
) -> go.Figure:
    """
    Build an AC Stark shift summary plot.

    A secondary y axis shows the mean photon number when
    ``dispersive_shift`` is available, and a secondary top x axis is added
    when ``secondary_x_shift`` is given (used to display the probe start
    measured from the resonator drive end).
    """
    fig = viz.make_figure()
    for shift, shift_error, name in zip(
        shift_traces, shift_error_traces, trace_names, strict=True
    ):
        fig.add_scatter(
            x=x_values,
            y=shift,
            error_y=dict(type="data", array=shift_error, visible=True),
            mode="lines+markers",
            name=name,
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="AC Stark shift (GHz)",
        width=600,
        height=400,
        showlegend=True,
    )

    if secondary_x_shift is not None and x_values.size > 0:
        # Top x-axis: probe start measured from the resonator drive end
        # (= offset + secondary_x_shift). Positive values are the ring-down
        # region, where the shift decays to zero as the cavity empties.
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        pad = 0.02 * (x_max - x_min) if x_max > x_min else 1.0
        primary_range = [x_min - pad, x_max + pad]
        fig.update_layout(
            xaxis=dict(range=primary_range),
            xaxis2=dict(
                title=secondary_x_title,
                overlaying="x",
                side="top",
                range=[
                    primary_range[0] + secondary_x_shift,
                    primary_range[1] + secondary_x_shift,
                ],
            ),
        )

    finite_values = []
    for shift, shift_error in zip(shift_traces, shift_error_traces, strict=True):
        error = np.where(np.isfinite(shift_error), shift_error, 0.0)
        finite_values.append(shift - error)
        finite_values.append(shift + error)
    stacked = np.concatenate(finite_values)
    stacked = stacked[np.isfinite(stacked)]
    if dispersive_shift is not None and stacked.size > 0:
        low = float(np.min(stacked))
        high = float(np.max(stacked))
        padding = 0.1 * (high - low) if high > low else max(abs(high), 1e-6) * 0.1
        low -= padding
        high += padding
        fig.add_scatter(
            x=[x_values[0]],
            y=[low / (2 * dispersive_shift)],
            yaxis="y2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.update_layout(
            yaxis=dict(range=[low, high]),
            yaxis2=dict(
                title="Mean photon number",
                overlaying="y",
                side="right",
                range=[low / (2 * dispersive_shift), high / (2 * dispersive_shift)],
                showgrid=False,
            ),
        )
    return fig


def waveform_figure(pulse: Waveform, *, title: str) -> go.Figure:
    """Build an I/Q envelope plot of a waveform."""
    fig = viz.make_figure()
    values = np.asarray(pulse.values)
    times = np.asarray(pulse.times)
    fig.add_scatter(x=times, y=values.real, mode="lines", name="I")
    fig.add_scatter(x=times, y=values.imag, mode="lines", name="Q")
    fig.add_scatter(
        x=times,
        y=np.abs(values),
        mode="lines",
        name="|amplitude|",
        line=dict(dash="dot"),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude",
        width=600,
        height=400,
        showlegend=True,
    )
    return fig
