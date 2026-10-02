import matplotlib.pyplot as plt
import numpy as np
import qubex as qx
from qubex.experiment.models import Result
from qubex.measurement import (
    MeasurementSchedule,
    MeasurementScheduleRunner,
    Quel1MeasurementBackendAdapter,
)
from qubex.measurement.models.capture_schedule import CaptureSchedule, Capture
from qxpulse import Waveform
from scipy.optimize import minimize

from .assignment_fidelity import (
    STATE_TO_COLOR,
    STATE_TO_LABEL,
    gaussian,
    gaussian_fit,
)

__all__ = [
    "mist_benchmarking_experiment",
    "run_mist_benchmarking",
    "analyze_mist_benchmarking",
    "display_mist_benchmarking",
    "plot_mist_benchmarking",
]

SEQUENCES = ["off", "on"]
STATES = ["g", "e"]
# The off block is zero-amplitude, so its capture window holds no readout signal:
# classifying it would measure the background, not an assignment.
ASSIGNMENT_SEQUENCE = "on"
POST_SELECTIONS = ["before_post_selection", "after_post_selection"]
OBSERVED_STATES = ["g", "e", "L"]
OBSERVED_LABELS = ["|g>", "|e>", "L"]
# Keyed by the observed (predicted) class, unlike the sibling scatters which colour by the
# prepared state. "L" is the pooled leakage class, not specifically |f>, and STATE_TO_COLOR
# has no entry for it.
OBSERVED_STATE_TO_COLOR = {
    "g": STATE_TO_COLOR["g"],
    "e": STATE_TO_COLOR["e"],
    "L": "orange",
}
SEQUENCE_TO_LINESTYLE = {"off": "--", "on": "-"}
RATE_NAMES = ["P01", "P0L", "P10", "P1L", "PL0", "PL1"]
INITIAL_RATES = [1e-4, 1e-5, 5e-3, 1e-5, 1e-6, 1e-6]
# The rates span decades, so the fit runs on log10(rate). That keeps them positive
# and conditions the problem: fitted linearly, the gradient is ~1e7 while the rates
# are ~1e-5 and the line search stalls on the first iteration. The upper bound keeps
# every row of the transition matrix stochastic.
LOG_RATE_BOUNDS = (-12.0, np.log10(0.4))
INITIAL_POPULATION = {"g": [1.0, 0.0, 0.0], "e": [0.0, 1.0, 0.0]}


def _transition_matrix(rates):
    p01, p0l, p10, p1l, pl0, pl1 = rates
    return np.array(
        [
            [1 - p01 - p0l, p01, p0l],
            [p10, 1 - p10 - p1l, p1l],
            [pl0, pl1, 1 - pl0 - pl1],
        ]
    )


def _model_population(rates, state, n_repeats):
    matrix = _transition_matrix(rates)
    return np.array(INITIAL_POPULATION[state]) @ np.linalg.matrix_power(
        matrix, n_repeats
    )


def _histogram_fit(signal, n_bins):
    counts, bin_edges = np.histogram(signal, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return gaussian_fit(bin_centers, counts)


def _assignment_summary(*, counts):
    counts = np.asarray(counts, dtype=int)
    shots = counts.sum(axis=1)
    # A row with no retained shots leaves its fidelity undefined rather than aborting:
    # one sparse repetition count must not destroy the whole analysis.
    matrix = np.full(counts.shape, np.nan)
    np.divide(counts, shots[:, None], out=matrix, where=shots[:, None] > 0)
    fidelity = {state: float(matrix[i, i]) for i, state in enumerate(STATES)}
    fidelity["avg"] = float(np.mean([fidelity[state] for state in STATES]))
    return {
        "counts": counts,
        "matrix": matrix,
        "shots": {state: int(shots[i]) for i, state in enumerate(STATES)},
        "fidelity": fidelity,
    }


def _fit_three_class_model(*, counts, n_repeats_list, block_duration):
    # Multinomial likelihood rather than least squares: the leakage class holds
    # only a handful of counts per point, where Gaussian weights break down.
    def negative_log_likelihood(log_rates):
        matrix = _transition_matrix(10.0 ** np.asarray(log_rates))
        total = 0.0
        for state in STATES:
            population = np.array(INITIAL_POPULATION[state])
            for n_repeats in n_repeats_list:
                q = population @ np.linalg.matrix_power(matrix, n_repeats)
                total -= np.sum(
                    counts[state][n_repeats] * np.log(np.clip(q, 1e-15, None))
                )
        return total

    fit = minimize(
        negative_log_likelihood,
        np.log10(INITIAL_RATES),
        method="L-BFGS-B",
        bounds=[LOG_RATE_BOUNDS] * len(RATE_NAMES),
    )
    rates = dict(zip(RATE_NAMES, 10.0**fit.x, strict=True))
    p10 = rates["P10"]
    return {
        "rates": rates,
        # The exact discrete-to-continuous inversion: |1> survives one block with
        # probability 1 - P10, so matching (1 - P10)**m to exp(-m * t_meas / T1) puts
        # a logarithm here.
        "t1_meas": -block_duration / np.log(1 - p10) if 0 < p10 < 1 else np.nan,
        # arXiv:2608.25484 Sec. V reads the rate off linearly instead,
        # T1_meas = t_meas / P10, which is the first-order form of the above. Both are
        # reported so the number can be compared with the paper directly: they are
        # 0.38 % apart at the paper's own P10(on) = 7.55e-3.
        "t1_meas_arxiv2608.25484": block_duration / p10 if 0 < p10 < 1 else np.nan,
        "leakage_per_measurement": (rates["P0L"] + rates["P1L"]) / 2,
        "negative_log_likelihood": float(fit.fun),
        "success": bool(fit.success),
    }


def run_mist_benchmarking(
    exp: qx.Experiment,
    *,
    target: str,
    ge_readout_pulse: Waveform,
    ge_readout_post_margin: float,
    heralding_readout_pulse: Waveform | None = None,
    heralding_readout_post_margin: float | None = None,
    leakage_sensitive_readout_pulse: Waveform,
    leakage_sensitive_readout_post_margin: float,
    leakage_sensitive_readout_detuning: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    n_repeats_list,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
):
    qubit = exp.ctx.qubits[target].label
    resonator = exp.ctx.resonators[target].label
    mux = exp.quantum_system.chip_graph.get_mux_of_qubit(qubit)
    drag_pi_pulse = exp.drag_pi_pulse[qubit]
    # The herald gets its own pulse so that it can be optimised for g/e discrimination
    # independently of the repeated blocks. Its classifier has to travel with it:
    # `Classifier.weight` holds one complex sample per capture sample, so a classifier
    # trained on one window length cannot be applied to the other at all.
    if heralding_readout_pulse is None:
        heralding_readout_pulse = ge_readout_pulse
    if heralding_readout_post_margin is None:
        heralding_readout_post_margin = ge_readout_post_margin
    padded_leakage_sensitive_readout_pulse = leakage_sensitive_readout_pulse.detuned(
        leakage_sensitive_readout_detuning
    ).padded(
        leakage_sensitive_readout_pulse.duration + leakage_sensitive_readout_post_margin
    )
    # The off block is a zero-amplitude pulse rather than a Blank so that it still
    # opens its capture window. That keeps the window layout identical between the
    # two sequences and leaves the first block readable as the assignment
    # measurement in both.
    block_pulse = {
        "on": ge_readout_pulse,
        "off": ge_readout_pulse.scaled(0),
    }
    block_duration = ge_readout_pulse.duration + waittime_after_readout_pulse
    barrier_labels = [qubit, resonator]
    result = dict()
    for n_repeats in n_repeats_list:
        result[n_repeats] = dict()
        pump_duration = (
            2 * pump_buffer_time
            + heralding_readout_pulse.duration
            + heralding_readout_post_margin
            + waittime_after_readout_pulse
            + drag_pi_pulse.duration
            + waittime_after_ctrl_pulse
            + n_repeats * block_duration
            + padded_leakage_sensitive_readout_pulse.duration
        )
        for sequence in SEQUENCES:
            result[n_repeats][sequence] = dict()
            for state in STATES:
                schedule = qx.PulseSchedule()
                with schedule as s:
                    s.add(
                        mux,
                        qx.pulse.FlatTop(
                            duration=pump_duration,
                            tau=pump_ramptime,
                            amplitude=pump_amplitude,
                        ),
                    )
                    s.add(resonator, qx.pulse.Blank(pump_buffer_time))
                    s.barrier(labels=barrier_labels)
                    # The herald. It has to stay a single capture-producing pulse:
                    # a `PulseArray` holding several non-Blank pulses would open extra
                    # windows and shift the capture indices selected below.
                    s.add(resonator, heralding_readout_pulse)
                    s.add(
                        qubit,
                        qx.pulse.Blank(
                            heralding_readout_pulse.duration
                            + waittime_after_readout_pulse
                        ),
                    )
                    if state == "e":
                        s.add(qubit, drag_pi_pulse)
                    else:
                        s.add(qubit, qx.pulse.Blank(drag_pi_pulse.duration))
                    s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
                    s.barrier(labels=barrier_labels)
                    for _ in range(n_repeats):
                        s.add(resonator, block_pulse[sequence])
                        s.add(resonator, qx.pulse.Blank(waittime_after_readout_pulse))
                    s.barrier(labels=barrier_labels)
                    s.add(resonator, padded_leakage_sensitive_readout_pulse)
                    s.add(resonator, qx.pulse.Blank(pump_buffer_time))
                if plot_pulse_schedule and (
                    n_repeats == n_repeats_list[0] or n_repeats == n_repeats_list[1]
                ):
                    schedule.plot(
                        title=f"Pulse Schedule {sequence} m={n_repeats} |{state}>"
                    )
                reference_schedule = exp.build_measurement_schedule(
                    schedule,
                    final_measurement=False,
                    capture_placement="pulse_aligned",
                    readout_amplification=False,
                )
                all_captures = sorted(
                    reference_schedule.capture_schedule.channels[resonator],
                    key=lambda capture: capture.start_time,
                )
                workaround = [
                    capture for capture in all_captures if capture.is_workaround
                ][0]
                real_captures = [
                    capture for capture in all_captures if not capture.is_workaround
                ]
                selected_captures = [
                    workaround,
                    real_captures[0],  # Herald
                    real_captures[1],  # First repeated readout
                    real_captures[-1],  # Final readout
                ]
                for i, post_margin in [
                    (1, heralding_readout_post_margin),
                    (2, ge_readout_post_margin),
                ]:
                    selected_captures[i] = Capture(
                        channels=[resonator],
                        start_time=selected_captures[i].start_time,
                        duration=selected_captures[i].duration + post_margin,
                        is_workaround=False,
                    )
                capture_schedule = CaptureSchedule(
                    captures=selected_captures,
                )
                measurement_schedule = MeasurementSchedule(
                    pulse_schedule=reference_schedule.pulse_schedule,
                    capture_schedule=capture_schedule,
                )

                class ThreeCaptureAdapter(Quel1MeasurementBackendAdapter):
                    def validate_schedule(self, _schedule):
                        super().validate_schedule(reference_schedule)

                adapter = ThreeCaptureAdapter(
                    backend_controller=exp.backend_controller,
                    experiment_system=exp.experiment_system,
                    constraint_profile=exp.measurement.constraint_profile,
                )
                runner = MeasurementScheduleRunner(
                    measurement_backend_adapter=adapter,
                    backend_controller=exp.backend_controller,
                )
                config = exp.measurement.create_measurement_config(
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    shot_averaging=False,
                    time_integration=False,
                    state_classification=False,
                )
                result[n_repeats][sequence][state] = runner.execute_sync(
                    schedule=measurement_schedule,
                    config=config,
                )
    return result


def analyze_mist_benchmarking(
    *,
    target,
    run_result,
    ge_classifier,
    heralding_classifier=None,
    leakage_sensitive_classifier,
    leakage_sensitive_readout_detuning,
    capture_sampling_period,
    block_duration,
    n_bins=100,
):
    if heralding_classifier is None:
        heralding_classifier = ge_classifier
    n_repeats_list = sorted(run_result.keys())
    counts = dict()
    populations = dict()
    post_selection_rate = dict()
    assignment_signal = dict()
    assignment_counts = {
        n_repeats: {
            w: np.zeros((len(STATES), len(STATES)), dtype=int) for w in POST_SELECTIONS
        }
        for n_repeats in n_repeats_list
    }
    # Only the largest repetition count is kept: the whole sweep would be tens of MB of
    # single-shot traces, and the endpoint is the longest exposure to the readout.
    leakage_sensitive_iq = dict()
    for sequence in SEQUENCES:
        counts[sequence] = dict()
        populations[sequence] = dict()
        post_selection_rate[sequence] = dict()
        assignment_signal[sequence] = dict()
        for state in STATES:
            counts[sequence][state] = dict()
            populations[sequence][state] = dict()
            post_selection_rate[sequence][state] = dict()
            signals = []
            for n_repeats in n_repeats_list:
                readouts = run_result[n_repeats][sequence][state].data[target]
                herald = readouts[0].data
                assignment = readouts[1].data
                leakage_sensitive = readouts[-1].data
                selected = heralding_classifier.predict(herald) == 0
                if sequence == ASSIGNMENT_SEQUENCE:
                    # The first repeated readout sits at the same position in every
                    # schedule, so its shots pool across the whole `n_repeats_list`.
                    # What this yields is
                    # P(assigned = s | nominal preparation s, herald classified |g>):
                    # it folds in herald misclassification, preparation error and any
                    # transition between the two windows, so it is not an isolated
                    # detector fidelity.
                    assignment_prediction = ge_classifier.predict(assignment)
                    if int(assignment_prediction.max()) >= len(STATES):
                        raise ValueError(
                            f"`ge_classifier` must classify into the {len(STATES)} "
                            f"states {STATES}, but it returned the label "
                            f"{int(assignment_prediction.max())}."
                        )
                    row = STATES.index(state)
                    for w, predicted in [
                        ("before_post_selection", assignment_prediction),
                        ("after_post_selection", assignment_prediction[selected]),
                    ]:
                        assignment_counts[n_repeats][w][row] = np.bincount(
                            predicted, minlength=len(STATES)
                        )
                # The leakage classifier was trained on de-rotated traces, so the
                # same rotation has to come off here before it is applied.
                derotation = np.exp(
                    -2j
                    * np.pi
                    * leakage_sensitive_readout_detuning
                    * np.arange(leakage_sensitive.shape[1])
                    * capture_sampling_period
                )
                prediction = leakage_sensitive_classifier.predict(
                    leakage_sensitive[selected] * derotation
                )
                if sequence == ASSIGNMENT_SEQUENCE and n_repeats == n_repeats_list[-1]:
                    # The weight is the |g>-|e> difference direction and the three-class
                    # model is fitted on (Re, Im) of this projection, so this complex
                    # plane is the classifier's own decision space. The same `selected`
                    # mask orders these points and `prediction` identically.
                    leakage_sensitive_iq[state] = {
                        "weighted_signal": leakage_sensitive_classifier.weighted_signal(
                            leakage_sensitive[selected] * derotation
                        ),
                        "prediction": prediction,
                    }
                count = np.array(
                    [np.sum(prediction == i) for i in range(len(OBSERVED_STATES))]
                )
                counts[sequence][state][n_repeats] = count
                populations[sequence][state][n_repeats] = count / count.sum()
                post_selection_rate[sequence][state][n_repeats] = sum(selected) / len(
                    selected
                )
                signals.append(
                    np.real(ge_classifier.weighted_signal(assignment[selected]))
                )
            assignment_signal[sequence][state] = np.concatenate(signals)
    fit = {
        sequence: _fit_three_class_model(
            counts=counts[sequence],
            n_repeats_list=n_repeats_list,
            block_duration=block_duration,
        )
        for sequence in SEQUENCES
    }
    model_population = {
        sequence: {
            state: {
                n_repeats: _model_population(
                    list(fit[sequence]["rates"].values()), state, n_repeats
                )
                for n_repeats in n_repeats_list
            }
            for state in STATES
        }
        for sequence in SEQUENCES
    }
    assignment_histogram = {
        sequence: {
            state: _histogram_fit(assignment_signal[sequence][state], n_bins)
            for state in STATES
        }
        for sequence in SEQUENCES
    }
    # Integer counts are summed before they are normalised, so the pooled matrix is
    # shot-weighted: averaging the per-m matrices would misweight unequal retention.
    pooled_assignment_counts = {
        w: np.sum(
            [assignment_counts[n_repeats][w] for n_repeats in n_repeats_list], axis=0
        )
        for w in POST_SELECTIONS
    }
    for w in POST_SELECTIONS:
        empty = [
            state
            for i, state in enumerate(STATES)
            if pooled_assignment_counts[w][i].sum() == 0
        ]
        if empty:
            raise ValueError(
                f"No shots are left to compute the assignment fidelity "
                f"({w.replace('_', ' ')}) for the prepared state(s) {empty}. "
                "Check the herald classifier and `n_shots`."
            )
    assignment_fidelity = {
        w: _assignment_summary(counts=pooled_assignment_counts[w])
        for w in POST_SELECTIONS
    }
    assignment_fidelity_by_n_repeats = {
        n_repeats: {
            w: _assignment_summary(counts=assignment_counts[n_repeats][w])
            for w in POST_SELECTIONS
        }
        for n_repeats in n_repeats_list
    }
    return {
        "n_repeats_list": n_repeats_list,
        "counts": counts,
        "populations": populations,
        "post_selection_rate": post_selection_rate,
        "assignment_signal": assignment_signal,
        "assignment_histogram": assignment_histogram,
        "assignment_fidelity": assignment_fidelity,
        "assignment_fidelity_by_n_repeats": assignment_fidelity_by_n_repeats,
        "leakage_sensitive_iq_scatter": {
            "n_repeats": n_repeats_list[-1],
            "sequence": ASSIGNMENT_SEQUENCE,
            "selection": "herald_g",
            "weighted_signal": {
                state: leakage_sensitive_iq[state]["weighted_signal"]
                for state in STATES
            },
            "prediction": {
                state: leakage_sensitive_iq[state]["prediction"] for state in STATES
            },
        },
        "fit": fit,
        "model_population": model_population,
        "block_duration": block_duration,
    }


def plot_mist_benchmarking(
    *,
    n_repeats_list,
    populations,
    model_population,
    assignment_signal,
    assignment_histogram,
    assignment_fidelity,
    leakage_sensitive_iq_scatter,
    n_bins=100,
    alpha_histogram=0.5,
    alpha_scatter=0.15,
    colors=STATE_TO_COLOR,
    labels=STATE_TO_LABEL,
):
    figures = dict()
    fig, axes = plt.subplots(1, len(SEQUENCES), figsize=(11, 4), sharey=True)
    for ax, sequence in zip(axes, SEQUENCES, strict=True):
        for state in STATES:
            counts, bin_edges = np.histogram(
                assignment_signal[sequence][state], bins=n_bins
            )
            ax.stairs(
                counts,
                bin_edges,
                color=colors[state],
                label=labels[state],
                fill=True,
                alpha=alpha_histogram,
            )
            fit_params = assignment_histogram[sequence][state]
            if fit_params is not None:
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
                ax.plot(
                    x_fit,
                    gaussian(
                        x_fit, fit_params["a"], fit_params["mu"], fit_params["sigma"]
                    ),
                    color="black",
                    linestyle="--",
                )
        ax.set_xlabel("S_I")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        ax.grid()
        ax.set_title(f"Assignment measurement ({sequence})")
        ax.legend()
    axes[0].set_ylabel("Counts")
    plt.close(fig)
    figures["assignment_histogram"] = {"figure": fig, "ax": axes}

    matrix = assignment_fidelity["after_post_selection"]["matrix"]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.6f}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] < 0.5 else "white",
            )
    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels([labels[state] for state in STATES])
    ax.set_yticklabels([labels[state] for state in STATES])
    ax.set_xlabel("Observed State")
    ax.set_ylabel("Prepared State")
    ax.set_title(
        "Assignment Fidelity Matrix after post selection\n"
        f"({ASSIGNMENT_SEQUENCE} sequence, first readout, all m pooled)"
    )
    fig.colorbar(im, ax=ax)
    plt.close(fig)
    figures["assignment_fidelity"] = {"figure": fig, "ax": ax}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, state in zip(axes, STATES, strict=True):
        signal = leakage_sensitive_iq_scatter["weighted_signal"][state]
        prediction = leakage_sensitive_iq_scatter["prediction"][state]
        # Drawn in OBSERVED_STATES order so the rare L class lands on top. It is given a
        # larger, opaque marker to stay visible, which means the ink density no longer
        # reflects the populations: the legend carries the counts instead.
        for i, observed in enumerate(OBSERVED_STATES):
            mask = prediction == i
            leakage = observed == "L"
            ax.scatter(
                np.real(signal[mask]),
                np.imag(signal[mask]),
                color=OBSERVED_STATE_TO_COLOR[observed],
                label=f"{OBSERVED_LABELS[i]} (n={int(mask.sum())})",
                s=12 if leakage else 6,
                alpha=0.8 if leakage else alpha_scatter,
                edgecolors="none",
            )
        ax.set_xlabel("Weighted I (a.u.)")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Prepared {labels[state]}")
        legend = ax.legend(title="Observed class (predicted)")
        # The legend is what keeps the counts honest against the deliberate alpha and
        # size asymmetry, so its markers must be legible whatever the points look like.
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            handle.set_sizes([24])
    axes[0].set_ylabel("Weighted Q (a.u.)")
    fig.suptitle(
        "Leakage-sensitive weighted IQ "
        f"({leakage_sensitive_iq_scatter['sequence']} sequence, final readout, "
        f"m={leakage_sensitive_iq_scatter['n_repeats']}, herald-selected)"
    )
    plt.close(fig)
    figures["leakage_sensitive_iq_scatter"] = {"figure": fig, "ax": axes}

    for name, index, ylabel in [
        ("population_1", 1, "|e> class population"),
        ("population_L", 2, "L class population"),
    ]:
        fig, ax = plt.subplots()
        for sequence in SEQUENCES:
            for state in STATES:
                ax.scatter(
                    n_repeats_list,
                    [populations[sequence][state][m][index] for m in n_repeats_list],
                    color=colors[state],
                    marker="o" if sequence == "on" else "x",
                    label=f"{labels[state]} ({sequence})",
                )
                ax.plot(
                    n_repeats_list,
                    [
                        model_population[sequence][state][m][index]
                        for m in n_repeats_list
                    ],
                    color="black",
                    linestyle=SEQUENCE_TO_LINESTYLE[sequence],
                    linewidth=1,
                )
        ax.set_xlabel("Number of repeated measurements m")
        ax.set_ylabel(ylabel)
        if name == "population_L":
            ax.set_yscale("log")
        ax.grid()
        ax.legend()
        plt.close(fig)
        figures[name] = {"figure": fig, "ax": ax}
    return {"figures": figures}


def display_mist_benchmarking(result: Result, plot: bool = True):
    try:
        from IPython.display import display
    except ImportError:
        display = lambda x: None

    def simple_display(figures):
        for fig in figures:
            display(fig)

    def tiled_display(figures, n_cols=2):
        outputs = []
        for fig in figures:
            output = widgets.Output()
            with output:
                display(fig)
            outputs.append(output)
        grid = widgets.GridBox(
            outputs,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({n_cols}, 1fr)",
                grid_gap="10px",
            ),
        )
        display(grid)

    if plot:
        figures_to_display = [
            result.figures["assignment_histogram"]["figure"],
            result.figures["assignment_fidelity"]["figure"],
            result.figures["leakage_sensitive_iq_scatter"]["figure"],
            result.figures["population_1"]["figure"],
            result.figures["population_L"]["figure"],
        ]
        try:
            import ipywidgets as widgets

            tiled_display(figures_to_display, n_cols=2)
        except ImportError:
            simple_display(figures_to_display)

    fit = result.data["fit"]
    post_selection_rate = result.data["post_selection_rate"]
    n_repeats_list = result.data["n_repeats_list"]
    for sequence in SEQUENCES:
        print(f"{sequence} sequence:")
        for name in RATE_NAMES:
            print(f"  {name}: {fit[sequence]['rates'][name]:.3e}")
        print(f"  T1_meas: {fit[sequence]['t1_meas'] * 1e-3:.2f} us")
        print(
            "  T1_meas (arXiv:2608.25484): "
            f"{fit[sequence]['t1_meas_arxiv2608.25484'] * 1e-3:.2f} us"
        )
        print(f"  P_L: {fit[sequence]['leakage_per_measurement']:.3e}")
        if not fit[sequence]["success"]:
            print("  The fit did not converge.")
        rates = [
            post_selection_rate[sequence][state][m]
            for state in STATES
            for m in n_repeats_list
        ]
        print(
            f"  Post-selection rate: {min(rates):.6f} - {max(rates):.6f} "
            f"over {len(n_repeats_list)} repetition counts"
        )

    assignment_fidelity = result.data["assignment_fidelity"]
    assignment_fidelity_by_n_repeats = result.data["assignment_fidelity_by_n_repeats"]
    print(
        f"Assignment measurement ({ASSIGNMENT_SEQUENCE} sequence, first readout, "
        f"{len(n_repeats_list)} repetition counts pooled):"
    )
    for w in POST_SELECTIONS:
        summary = assignment_fidelity[w]
        print(f"  {w.replace('_', ' ')}:")
        print("    Assignment fidelity")
        for state in STATES:
            print(
                f"      {STATE_TO_LABEL[state]}: {summary['fidelity'][state]:.6f} "
                f"({summary['shots'][state]} shots)"
            )
        print(f"      avg: {summary['fidelity']['avg']:.6f}")
        print("    P(observed | prepared)")
        print(
            "      prepared |"
            + "".join(f"{STATE_TO_LABEL[state]:>9}" for state in STATES)
        )
        for i, state in enumerate(STATES):
            row = "".join(f"{summary['matrix'][i, j]:9.6f}" for j in range(len(STATES)))
            print(f"      {STATE_TO_LABEL[state]:>8} |" + row)
    # The pump tone grows with m, so the pooled number is only trustworthy if the
    # per-repetition-count fidelities do not drift.
    print("  after post selection, per repetition count:")
    print("         m |      F_g      F_e      avg |   N_g   N_e")
    for n_repeats in n_repeats_list:
        summary = assignment_fidelity_by_n_repeats[n_repeats]["after_post_selection"]
        print(
            f"      {n_repeats:>4} |"
            f" {summary['fidelity']['g']:8.6f} {summary['fidelity']['e']:8.6f}"
            f" {summary['fidelity']['avg']:8.6f} |"
            f" {summary['shots']['g']:>5} {summary['shots']['e']:>5}"
        )


def mist_benchmarking_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    leakage_sensitive_readout_classifier: dict,
    leakage_sensitive_readout_frequency: float | None = None,
    ge_readout_classifier: dict,
    heralding_readout_classifier: dict | None = None,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    n_repeats_list,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    missing_keys = {"readout_pulse", "readout_post_margin", "classifier"} - set(
        leakage_sensitive_readout_classifier
    )
    if missing_keys:
        raise ValueError(
            "`leakage_sensitive_readout_classifier` must contain the keys 'readout_pulse', 'readout_post_margin' and "
            f"'classifier' (missing: {sorted(missing_keys)})."
        )
    missing_keys = {"readout_pulse", "readout_post_margin", "classifier"} - set(
        ge_readout_classifier
    )
    if missing_keys:
        raise ValueError(
            "`ge_readout_classifier` must contain the keys 'readout_pulse', 'readout_post_margin', and "
            f"'classifier' (missing: {sorted(missing_keys)})."
        )
    # Omitting the heralding one reproduces the previous behaviour exactly: one pulse,
    # margin and classifier then serve both the herald and the repeated blocks.
    if heralding_readout_classifier is None:
        heralding_readout_classifier = ge_readout_classifier
    missing_keys = {"readout_pulse", "readout_post_margin", "classifier"} - set(
        heralding_readout_classifier
    )
    if missing_keys:
        raise ValueError(
            "`heralding_readout_classifier` must contain the keys 'readout_pulse', 'readout_post_margin', and "
            f"'classifier' (missing: {sorted(missing_keys)})."
        )
    ge_readout_pulse = ge_readout_classifier["readout_pulse"]
    ge_readout_post_margin = ge_readout_classifier["readout_post_margin"]
    ge_classifier = ge_readout_classifier["classifier"]
    heralding_readout_pulse = heralding_readout_classifier["readout_pulse"]
    heralding_readout_post_margin = heralding_readout_classifier["readout_post_margin"]
    heralding_classifier = heralding_readout_classifier["classifier"]
    leakage_sensitive_readout_pulse = leakage_sensitive_readout_classifier[
        "readout_pulse"
    ]
    leakage_sensitive_readout_post_margin = leakage_sensitive_readout_classifier[
        "readout_post_margin"
    ]
    leakage_sensitive_classifier = leakage_sensitive_readout_classifier["classifier"]
    resonator = exp.ctx.resonators[target].label
    if leakage_sensitive_readout_frequency is None:
        leakage_sensitive_readout_detuning = 0.0
    else:
        leakage_sensitive_readout_detuning = (
            leakage_sensitive_readout_frequency - exp.ctx.targets[resonator].frequency
        )
    capture_sampling_period = (
        exp.backend_controller.sampling_period_ns
        * exp.backend_controller.CAPTURE_DECIMATION_FACTOR
    )
    block_duration = ge_readout_pulse.duration + waittime_after_readout_pulse
    run_result = run_mist_benchmarking(
        exp,
        target=target,
        ge_readout_pulse=ge_readout_pulse,
        ge_readout_post_margin=ge_readout_post_margin,
        heralding_readout_pulse=heralding_readout_pulse,
        heralding_readout_post_margin=heralding_readout_post_margin,
        leakage_sensitive_readout_pulse=leakage_sensitive_readout_pulse,
        leakage_sensitive_readout_post_margin=leakage_sensitive_readout_post_margin,
        leakage_sensitive_readout_detuning=leakage_sensitive_readout_detuning,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        n_repeats_list=n_repeats_list,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        pump_amplitude=pump_amplitude,
        shot_interval=shot_interval,
        n_shots=n_shots,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    analyze_result = analyze_mist_benchmarking(
        target=target,
        run_result=run_result,
        ge_classifier=ge_classifier,
        heralding_classifier=heralding_classifier,
        leakage_sensitive_classifier=leakage_sensitive_classifier,
        leakage_sensitive_readout_detuning=leakage_sensitive_readout_detuning,
        capture_sampling_period=capture_sampling_period,
        block_duration=block_duration,
    )
    plot_result = plot_mist_benchmarking(
        n_repeats_list=analyze_result["n_repeats_list"],
        populations=analyze_result["populations"],
        model_population=analyze_result["model_population"],
        assignment_signal=analyze_result["assignment_signal"],
        assignment_histogram=analyze_result["assignment_histogram"],
        assignment_fidelity=analyze_result["assignment_fidelity"],
        leakage_sensitive_iq_scatter=analyze_result["leakage_sensitive_iq_scatter"],
        colors=STATE_TO_COLOR,
        labels=STATE_TO_LABEL,
    )
    result = Result(
        data={
            "target": target,
            "raw": run_result,
            "ge_readout_pulse": ge_readout_pulse,
            "ge_readout_post_margin": ge_readout_post_margin,
            "ge_classifier": ge_classifier,
            "heralding_readout_pulse": heralding_readout_pulse,
            "heralding_readout_post_margin": heralding_readout_post_margin,
            "heralding_classifier": heralding_classifier,
            "leakage_sensitive_readout_pulse": leakage_sensitive_readout_pulse,
            "leakage_sensitive_readout_post_margin": leakage_sensitive_readout_post_margin,
            "leakage_sensitive_classifier": leakage_sensitive_classifier,
            "leakage_sensitive_readout_frequency": leakage_sensitive_readout_frequency,
            "leakage_sensitive_readout_detuning": leakage_sensitive_readout_detuning,
            "n_repeats_list": analyze_result["n_repeats_list"],
            "counts": analyze_result["counts"],
            "populations": analyze_result["populations"],
            "post_selection_rate": analyze_result["post_selection_rate"],
            "assignment_signal": analyze_result["assignment_signal"],
            "assignment_histogram": analyze_result["assignment_histogram"],
            "assignment_fidelity": analyze_result["assignment_fidelity"],
            "assignment_fidelity_by_n_repeats": analyze_result[
                "assignment_fidelity_by_n_repeats"
            ],
            "leakage_sensitive_iq_scatter": analyze_result[
                "leakage_sensitive_iq_scatter"
            ],
            "fit": analyze_result["fit"],
            "model_population": analyze_result["model_population"],
            "block_duration": block_duration,
            "capture_sampling_period": capture_sampling_period,
        },
        figure=None,
        figures=plot_result["figures"],
    )
    display_mist_benchmarking(result, plot)
    return result
