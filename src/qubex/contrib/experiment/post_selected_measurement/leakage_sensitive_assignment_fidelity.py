import numpy as np
import qubex as qx
import matplotlib.pyplot as plt
from qubex.experiment.models import Result
from qubex.system.transition import LEVEL_NAMES, TRANSITION_NAMES
from qxpulse import Waveform
from .assignment_fidelity import Classifier, STATE_TO_COLOR, STATE_TO_LABEL

__all__ = [
    "post_selected_leakage_sensitive_assignment_fidelity_experiment",
    "run_post_selected_leakage_sensitive_assignment_fidelity",
    "analyze_post_selected_leakage_sensitive_assignment_fidelity",
    "display_post_selected_leakage_sensitive_assignment_fidelity",
    "plot_post_selected_leakage_sensitive_assignment_fidelity",
]

# The leakage-sensitive readout resolves three outcomes: |g>, |e>, and a single
# merged leakage class L holding every level from |f> up. The order fixes the
# integer labels the classifier returns.
OBSERVED_STATES = ["g", "e", "L"]
OBSERVED_LABELS = ["|g>", "|e>", "L"]


def _validate_readout_pulse_start_position(
    *,
    ge_readout_duration: float,
    ladder_duration: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    capture_sampling_period: float,
):
    second_readout_start = (
        ge_readout_duration
        + waittime_after_readout_pulse
        + ladder_duration
        + waittime_after_ctrl_pulse
    )
    if second_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The start position of the second readout pulse ({second_readout_start} ns) "
            f"is not aligned with the capture sampling period ({capture_sampling_period} ns)."
        )


def _resolve_ladder(exp: qx.Experiment, *, qubit: str, max_level: int):
    labels = [
        exp.ctx.resolve_transition_label(qubit, name)
        for name in TRANSITION_NAMES[:max_level]
    ]
    pulses = [exp.pulse.x180(label) for label in labels]
    return labels, pulses


def run_post_selected_leakage_sensitive_assignment_fidelity(
    exp: qx.Experiment,
    *,
    target: str,
    max_level: int,
    ge_readout_pulse: Waveform | None = None,
    leakage_sensitive_readout_pulse: Waveform | None = None,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
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
    if ge_readout_pulse is None:
        ge_readout_pulse = exp.readout(target)
    if leakage_sensitive_readout_pulse is None:
        leakage_sensitive_readout_pulse = exp.readout(target)
    padded_ge_readout_pulse = ge_readout_pulse.padded(
        ge_readout_pulse.duration + exp.readout_post_margin
    )
    padded_leakage_sensitive_readout_pulse = leakage_sensitive_readout_pulse.padded(
        leakage_sensitive_readout_pulse.duration + exp.readout_post_margin
    )
    ladder_labels, ladder_pulses = _resolve_ladder(
        exp, qubit=qubit, max_level=max_level
    )
    ladder_duration = sum(pulse.duration for pulse in ladder_pulses)
    # Successive rungs are played on different channels, so they have to be
    # serialized with barriers. The mux stays out of the set, as in the sibling
    # experiment, so the barriers never truncate the pump tone.
    barrier_labels = [resonator, *ladder_labels]
    pump_duration = (
        padded_ge_readout_pulse.duration
        + padded_leakage_sensitive_readout_pulse.duration
        + waittime_after_readout_pulse
        + ladder_duration
        + waittime_after_ctrl_pulse
        + 2 * pump_buffer_time
    )
    result = dict()
    for state_idx in range(max_level + 1):
        state = LEVEL_NAMES[state_idx]
        schedule = qx.PulseSchedule()
        with schedule as s:
            s.add(
                mux,
                qx.pulse.FlatTop(
                    duration=pump_duration, tau=pump_ramptime, amplitude=pump_amplitude
                ),
            )
            s.add(resonator, qx.pulse.Blank(pump_buffer_time))
            s.barrier(labels=barrier_labels)
            s.add(resonator, padded_ge_readout_pulse)
            s.add(
                qubit,
                qx.pulse.Blank(
                    ge_readout_pulse.duration + waittime_after_readout_pulse
                ),
            )
            # The ladder is padded at the front, not the back, so that every
            # prepared state has the same interval between its last pi pulse and
            # the leakage-sensitive readout. Padding afterwards would let the
            # lower states relax for longer and bias the assignment matrix.
            s.add(
                qubit,
                qx.pulse.Blank(
                    ladder_duration
                    - sum(pulse.duration for pulse in ladder_pulses[:state_idx])
                ),
            )
            for rung in range(state_idx):
                if rung > 0:
                    s.barrier(labels=barrier_labels)
                s.add(ladder_labels[rung], ladder_pulses[rung])
            s.barrier(labels=barrier_labels)
            s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
            s.barrier(labels=barrier_labels)
            s.add(resonator, padded_leakage_sensitive_readout_pulse)
            s.add(resonator, qx.pulse.Blank(pump_buffer_time))
        if plot_pulse_schedule:
            schedule.plot(title=f"Pulse Schedule |{state}>")
        result[state] = exp.execute(
            schedule=schedule,
            mode="single",
            time_integration=False,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
    return result


def plot_post_selected_leakage_sensitive_assignment_fidelity(
    *,
    iq_data,
    classifier,
    post_selection,
    posterior_weighted_signal,
    population_matrix,
    capture_sampling_period,
    alpha_scatter=0.01,
    colors=STATE_TO_COLOR,
    labels=STATE_TO_LABEL,
):
    figures = dict()
    posterior_averaged_iq = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        posterior_averaged_iq[w] = dict()
        for s in iq_data.keys():
            posterior_averaged_iq[w][s] = dict()
            for d in ["train", "test"]:
                if w == "before_post_selection":
                    posterior_averaged_iq[w][s][d] = iq_data[s]["posterior"][d].mean(
                        axis=0
                    )
                else:
                    posterior_averaged_iq[w][s][d] = iq_data[s]["posterior"][d][
                        post_selection[s][d], :
                    ].mean(axis=0)
    # TD averaged iq data
    figures["td_iq"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["td_iq"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in iq_data.keys():
                ax.scatter(
                    np.arange(len(posterior_averaged_iq[w][s][d]))
                    * capture_sampling_period,
                    np.real(posterior_averaged_iq[w][s][d]),
                    color=colors[s],
                    label=f"I_{labels[s]}",
                    marker="o",
                )
                ax.scatter(
                    np.arange(len(posterior_averaged_iq[w][s][d]))
                    * capture_sampling_period,
                    np.imag(posterior_averaged_iq[w][s][d]),
                    color=colors[s],
                    label=f"Q_{labels[s]}",
                    marker="x",
                )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.grid()
            title = f"Time-domain averaged IQ data {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            ax.legend()
            plt.close(fig)
            figures["td_iq"][w][d] = {"figure": fig, "ax": ax}
    # TD averaged iq data distance from origin
    figures["td_iq_distance"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["td_iq_distance"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in iq_data.keys():
                ax.scatter(
                    np.arange(len(posterior_averaged_iq[w][s][d]))
                    * capture_sampling_period,
                    np.abs(posterior_averaged_iq[w][s][d]),
                    color=colors[s],
                    label=f"[(I_{labels[s]})^2 + (Q_{labels[s]})^2]^(1/2)",
                    marker="o",
                )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.grid()
            title = f"Time-domain averaged IQ data {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            ax.legend()
            plt.close(fig)
            figures["td_iq_distance"][w][d] = {"figure": fig, "ax": ax}
    # Plot IQ data
    figures["iq_scatter"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["iq_scatter"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in iq_data.keys():
                ax.scatter(
                    np.real(posterior_weighted_signal[w][s][d]),
                    np.imag(posterior_weighted_signal[w][s][d]),
                    color=colors[s],
                    label=labels[s],
                    alpha=alpha_scatter,
                )
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.grid()
            ax.set_aspect("equal", adjustable="box")
            title = f"IQ data {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            ax.legend()
            plt.close(fig)
            figures["iq_scatter"][w][d] = {"figure": fig, "ax": ax}
    # Plot population matrix
    figures["population_matrix"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["population_matrix"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            prepared_states = list(population_matrix[w].keys())
            matrix = np.array(
                [
                    [
                        population_matrix[w][s_prepared][s_observed][d]
                        for s_observed in OBSERVED_STATES
                    ]
                    for s_prepared in prepared_states
                ]
            )
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(len(prepared_states)):
                for j in range(len(OBSERVED_STATES)):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color="black" if matrix[i, j] < 0.5 else "white",
                    )
            ax.set_xticks(range(len(OBSERVED_STATES)))
            ax.set_yticks(range(len(prepared_states)))
            ax.set_xticklabels(OBSERVED_LABELS)
            ax.set_yticklabels([labels[s] for s in prepared_states])
            ax.set_xlabel("Observed State")
            ax.set_ylabel("Prepared State")
            title = f"Assignment matrix {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            plt.close(fig)
            figures["population_matrix"][w][d] = {"figure": fig, "ax": ax}
    return {
        "figures": figures,
    }


def _population_matrix(*, iq_data, classifier, post_selection):
    population_matrix = dict()
    w = "before_post_selection"
    population_matrix[w] = dict()
    for s_prepared in iq_data.keys():
        population_matrix[w][s_prepared] = dict()
        for s_observed in OBSERVED_STATES:
            population_matrix[w][s_prepared][s_observed] = dict()
            for d in ["train", "test"]:
                population_matrix[w][s_prepared][s_observed][d] = sum(
                    classifier.predict(iq_data[s_prepared]["posterior"][d])
                    == OBSERVED_STATES.index(s_observed)
                ) / len(iq_data[s_prepared]["posterior"][d])
    w = "after_post_selection"
    population_matrix[w] = dict()
    for s_prepared in iq_data.keys():
        population_matrix[w][s_prepared] = dict()
        for s_observed in OBSERVED_STATES:
            population_matrix[w][s_prepared][s_observed] = dict()
            for d in ["train", "test"]:
                population_matrix[w][s_prepared][s_observed][d] = sum(
                    classifier.predict(
                        iq_data[s_prepared]["posterior"][d][
                            post_selection[s_prepared][d]
                        ]
                    )
                    == OBSERVED_STATES.index(s_observed)
                ) / sum(post_selection[s_prepared][d])
    return population_matrix


def analyze_post_selected_leakage_sensitive_assignment_fidelity(
    *, target, ge_classifier, run_result, n_train, n_test
):
    iq_data = {
        s: {
            o: {
                d: run_result[s].data[target][i].raw[:n_train]
                if d == "train"
                else run_result[s].data[target][i].raw[n_train : n_train + n_test]
                for d in ["train", "test"]
            }
            for i, o in enumerate(["prior", "posterior"])
        }
        for s in run_result.keys()
    }
    # The herald classifier is pretrained on the ge readout and is applied only to
    # the prior capture; the leakage classifier below is fit here and applied only
    # to the posterior capture. Keeping the two apart avoids both a weight/sample
    # length mismatch and circular heralding.
    post_selection = {
        s: {
            d: ge_classifier.predict(iq_data[s]["prior"][d]) == 0
            for d in ["train", "test"]
        }
        for s in iq_data.keys()
    }
    classifier = Classifier()
    classifier.update_weight(
        iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
        iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
    )
    # Exactly three fit groups, so that `predict` returns 0 for |g>, 1 for |e> and
    # 2 for L. Merging every level from |f> up into L gives that class
    # (max_level - 1) times as many training shots as |g> or |e>, which tilts the
    # logistic-regression boundary towards L. The imbalance is deliberate: the
    # reference experiment trains the same way.
    classifier.fit(
        [
            iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
            iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
            np.concatenate(
                [
                    iq_data[s]["posterior"]["train"][post_selection[s]["train"], :]
                    for s in iq_data.keys()
                    if s not in ["g", "e"]
                ]
            ),
        ]
    )
    post_selection_rate = {
        s: {
            d: sum(post_selection[s][d]) / len(post_selection[s][d])
            for d in ["train", "test"]
        }
        for s in iq_data.keys()
    }
    posterior_weighted_signal = {
        w: {
            s: {
                d: classifier.weighted_signal(iq_data[s]["posterior"][d])
                if w == "before_post_selection"
                else classifier.weighted_signal(
                    iq_data[s]["posterior"][d][post_selection[s][d], :]
                )
                for d in ["train", "test"]
            }
            for s in iq_data.keys()
        }
        for w in ["before_post_selection", "after_post_selection"]
    }
    population_matrix = _population_matrix(
        iq_data=iq_data, classifier=classifier, post_selection=post_selection
    )
    return {
        "iq_data": iq_data,
        "classifier": classifier,
        "post_selection": post_selection,
        "post_selection_rate": post_selection_rate,
        "posterior_weighted_signal": posterior_weighted_signal,
        "population_matrix": population_matrix,
    }


def display_post_selected_leakage_sensitive_assignment_fidelity(
    result: Result, plot: bool = True
):
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

    population_matrix = result.data["population_matrix"]
    post_selection_rate = result.data["post_selection_rate"]
    for w in ["before_post_selection", "after_post_selection"]:
        for d in ["train", "test"]:
            if plot:
                figures_to_display = [
                    result.figures["td_iq"][w][d]["figure"],
                    result.figures["td_iq_distance"][w][d]["figure"],
                    result.figures["iq_scatter"][w][d]["figure"],
                    result.figures["population_matrix"][w][d]["figure"],
                ]
                try:
                    import ipywidgets as widgets

                    tiled_display(figures_to_display, n_cols=2)
                except ImportError:
                    simple_display(figures_to_display)
            print(f"{w.replace('_', ' ')} ({d} set):")
            print("Assignment matrix P(observed | prepared)")
            print("  prepared |" + "".join(f"{o:>9}" for o in OBSERVED_LABELS))
            for s_prepared in population_matrix[w].keys():
                row = "".join(
                    f"{population_matrix[w][s_prepared][s_observed][d]:9.4f}"
                    for s_observed in OBSERVED_STATES
                )
                print(f"  {STATE_TO_LABEL[s_prepared]:>8} |" + row)
            if w == "after_post_selection":
                for s in post_selection_rate.keys():
                    print(
                        f"Post-selection rate {STATE_TO_LABEL[s]}: "
                        f"{post_selection_rate[s][d]:.4f}"
                    )


def post_selected_leakage_sensitive_assignment_fidelity_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    max_level: int,
    leakage_sensitive_readout_pulse: Waveform | None = None,
    ge_readout_classifier: dict,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    n_train: int,
    n_test: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    missing_keys = {"readout_pulse", "classifier"} - set(ge_readout_classifier)
    if missing_keys:
        raise ValueError(
            "`readout_classifier` must contain the keys 'readout_pulse' and "
            f"'classifier' (missing: {sorted(missing_keys)})."
        )
    if max_level < 2:
        raise ValueError(
            f"max_level must be at least 2 for leakage-sensitive assignment fidelity "
            f"(got {max_level})."
        )
    ge_readout_pulse = ge_readout_classifier["readout_pulse"]
    ge_classifier = ge_readout_classifier["classifier"]
    if leakage_sensitive_readout_pulse is None:
        leakage_sensitive_readout_pulse = exp.readout(target)
    capture_sampling_period = (
        exp.backend_controller.sampling_period_ns
        * exp.backend_controller.CAPTURE_DECIMATION_FACTOR
    )
    qubit = exp.ctx.qubits[target].label
    _, ladder_pulses = _resolve_ladder(exp, qubit=qubit, max_level=max_level)
    _validate_readout_pulse_start_position(
        ge_readout_duration=ge_readout_pulse.duration,
        ladder_duration=sum(pulse.duration for pulse in ladder_pulses),
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        capture_sampling_period=capture_sampling_period,
    )
    run_result = run_post_selected_leakage_sensitive_assignment_fidelity(
        exp,
        target=target,
        max_level=max_level,
        ge_readout_pulse=ge_readout_pulse,
        leakage_sensitive_readout_pulse=leakage_sensitive_readout_pulse,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        pump_amplitude=pump_amplitude,
        shot_interval=shot_interval,
        n_shots=n_train + n_test,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    analyze_result = analyze_post_selected_leakage_sensitive_assignment_fidelity(
        target=target,
        ge_classifier=ge_classifier,
        run_result=run_result,
        n_train=n_train,
        n_test=n_test,
    )
    iq_data = analyze_result["iq_data"]
    classifier = analyze_result["classifier"]
    post_selection = analyze_result["post_selection"]
    post_selection_rate = analyze_result["post_selection_rate"]
    posterior_weighted_signal = analyze_result["posterior_weighted_signal"]
    population_matrix = analyze_result["population_matrix"]
    plot_result = plot_post_selected_leakage_sensitive_assignment_fidelity(
        iq_data=iq_data,
        classifier=classifier,
        post_selection=post_selection,
        posterior_weighted_signal=posterior_weighted_signal,
        population_matrix=population_matrix,
        capture_sampling_period=capture_sampling_period,
        alpha_scatter=0.05,
        colors=STATE_TO_COLOR,
        labels=STATE_TO_LABEL,
    )
    figures = plot_result["figures"]
    result = Result(
        data={
            "target": target,
            "raw": run_result,
            "leakage_sensitive_readout_pulse": leakage_sensitive_readout_pulse,
            "ge_readout_pulse": ge_readout_pulse,
            "ge_classifier": ge_classifier,
            "iq_data": iq_data,
            "classifier": classifier,
            "post_selection": post_selection,
            "post_selection_rate": post_selection_rate,
            "posterior_weighted_signal": posterior_weighted_signal,
            "population_matrix": population_matrix,
            "capture_sampling_period": capture_sampling_period,
        },
        figure=None,
        figures=figures,
    )
    display_post_selected_leakage_sensitive_assignment_fidelity(result, plot)
    return result
