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
    leakage_sensitive_readout_detuning: float = 0.0,
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
    padded_leakage_sensitive_readout_pulse = leakage_sensitive_readout_pulse.detuned(
        leakage_sensitive_readout_detuning
    ).padded(leakage_sensitive_readout_pulse.duration + exp.readout_post_margin)
    ladder_labels, ladder_pulses = _resolve_ladder(
        exp, qubit=qubit, max_level=max_level
    )
    ladder_duration = sum(pulse.duration for pulse in ladder_pulses)
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
    integration_start_idx,
    integration_end_idx,
    post_selection,
    posterior_weighted_signal,
    population_matrix,
    capture_sampling_period,
    alpha_scatter=0.01,
    colors=STATE_TO_COLOR,
    labels=STATE_TO_LABEL,
):
    vspan_color = "gray"
    vspan_alpha = 0.2
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
            ax.axvspan(
                integration_start_idx * capture_sampling_period,
                (integration_end_idx - 1) * capture_sampling_period,
                color=vspan_color,
                alpha=vspan_alpha,
                label="Integration window",
            )
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
            ax.axvspan(
                integration_start_idx * capture_sampling_period,
                (integration_end_idx - 1) * capture_sampling_period,
                color=vspan_color,
                alpha=vspan_alpha,
                label="Integration window",
            )
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
                        f"{matrix[i, j]:.6f}",
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


def _leakage_sensitive_assignment_fidelity(*, population_matrix):
    leakage_state_keys = [
        k
        for k in population_matrix["before_post_selection"].keys()
        if k not in ["g", "e"]
    ]
    p_LL = {
        w: {
            d: sum(population_matrix[w][s]["L"][d] for s in leakage_state_keys)
            / len(leakage_state_keys)
            for d in ["train", "test"]
        }
        for w in ["before_post_selection", "after_post_selection"]
    }
    return {
        w: {
            d: (
                sum(
                    population_matrix[w][s_prepared][s_prepared][d]
                    for s_prepared in ["g", "e"]
                )
                + p_LL[w][d]
            )
            / 3
            for d in ["train", "test"]
        }
        for w in ["before_post_selection", "after_post_selection"]
    }


def analyze_post_selected_leakage_sensitive_assignment_fidelity(
    *,
    target,
    ge_classifier,
    run_result,
    n_train,
    n_test,
    leakage_sensitive_readout_detuning=0.0,
    capture_sampling_period=None,
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
    if leakage_sensitive_readout_detuning:
        n_samples = next(iter(iq_data.values()))["posterior"]["train"].shape[1]
        derotation = np.exp(
            -2j
            * np.pi
            * leakage_sensitive_readout_detuning
            * np.arange(n_samples)
            * capture_sampling_period
        )
        for s in iq_data.keys():
            for d in ["train", "test"]:
                iq_data[s]["posterior"][d] = iq_data[s]["posterior"][d] * derotation
    post_selection = {
        s: {
            d: ge_classifier.predict(iq_data[s]["prior"][d]) == 0
            for d in ["train", "test"]
        }
        for s in iq_data.keys()
    }
    num_samples = len(iq_data["g"]["posterior"]["train"][0])
    integration_start_idx = 0
    integration_end_idx = num_samples
    highest_assignment_fidelity = 0.0
    for i_start in reversed(range(num_samples)):
        for i_end in range(i_start + 1, num_samples + 1):
            leakage_sensitive_classifier = Classifier()
            leakage_sensitive_classifier.fit(
                [
                    iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
                    iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
                    np.concatenate(
                        [
                            iq_data[s]["posterior"]["train"][
                                post_selection[s]["train"], :
                            ]
                            for s in iq_data.keys()
                            if s not in ["g", "e"]
                        ]
                    ),
                ],
                integration_start_idx=i_start,
                integration_end_idx=i_end,
            )
            population_matrix = _population_matrix(
                iq_data=iq_data,
                classifier=leakage_sensitive_classifier,
                post_selection=post_selection,
            )
            leakage_sensitive_assignment_fidelity = (
                _leakage_sensitive_assignment_fidelity(
                    population_matrix=population_matrix
                )
            )
            fidelity_temp = leakage_sensitive_assignment_fidelity[
                "after_post_selection"
            ]["test"]
            if fidelity_temp > highest_assignment_fidelity:
                highest_assignment_fidelity = fidelity_temp
                integration_start_idx = i_start
                integration_end_idx = i_end
    leakage_sensitive_classifier = Classifier()
    leakage_sensitive_classifier.fit(
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
        ],
        integration_start_idx=integration_start_idx,
        integration_end_idx=integration_end_idx,
    )
    population_matrix = _population_matrix(
        iq_data=iq_data,
        classifier=leakage_sensitive_classifier,
        post_selection=post_selection,
    )
    leakage_sensitive_assignment_fidelity = _leakage_sensitive_assignment_fidelity(
        population_matrix=population_matrix
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
                d: leakage_sensitive_classifier.weighted_signal(
                    iq_data[s]["posterior"][d]
                )
                if w == "before_post_selection"
                else leakage_sensitive_classifier.weighted_signal(
                    iq_data[s]["posterior"][d][post_selection[s][d], :]
                )
                for d in ["train", "test"]
            }
            for s in iq_data.keys()
        }
        for w in ["before_post_selection", "after_post_selection"]
    }
    return {
        "iq_data": iq_data,
        "leakage_sensitive_classifier": leakage_sensitive_classifier,
        "leakage_sensitive_assignment_fidelity": leakage_sensitive_assignment_fidelity,
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
            print(
                f"Leakage-sensitive assignment fidelity: "
                f"{result.data['leakage_sensitive_assignment_fidelity'][w][d]:.6f}"
            )
            print("Assignment matrix P(observed | prepared)")
            print("  prepared |" + "".join(f"{o:>9}" for o in OBSERVED_LABELS))
            for s_prepared in population_matrix[w].keys():
                row = "".join(
                    f"{population_matrix[w][s_prepared][s_observed][d]:9.6f}"
                    for s_observed in OBSERVED_STATES
                )
                print(f"  {STATE_TO_LABEL[s_prepared]:>8} |" + row)
            if w == "after_post_selection":
                for s in post_selection_rate.keys():
                    print(
                        f"Post-selection rate {STATE_TO_LABEL[s]}: "
                        f"{post_selection_rate[s][d]:.6f}"
                    )


def post_selected_leakage_sensitive_assignment_fidelity_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    max_level: int,
    leakage_sensitive_readout_pulse: Waveform | None = None,
    leakage_sensitive_readout_frequency: float | None = None,
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
    resonator = exp.ctx.resonators[target].label
    if leakage_sensitive_readout_frequency is None:
        leakage_sensitive_readout_detuning = 0.0
    else:
        leakage_sensitive_readout_detuning = (
            leakage_sensitive_readout_frequency - exp.ctx.targets[resonator].frequency
        )
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
        leakage_sensitive_readout_detuning=leakage_sensitive_readout_detuning,
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
        leakage_sensitive_readout_detuning=leakage_sensitive_readout_detuning,
        capture_sampling_period=capture_sampling_period,
    )
    iq_data = analyze_result["iq_data"]
    leakage_sensitive_classifier = analyze_result["leakage_sensitive_classifier"]
    leakage_sensitive_assignment_fidelity = analyze_result[
        "leakage_sensitive_assignment_fidelity"
    ]
    post_selection = analyze_result["post_selection"]
    post_selection_rate = analyze_result["post_selection_rate"]
    posterior_weighted_signal = analyze_result["posterior_weighted_signal"]
    population_matrix = analyze_result["population_matrix"]
    plot_result = plot_post_selected_leakage_sensitive_assignment_fidelity(
        iq_data=iq_data,
        integration_start_idx=leakage_sensitive_classifier.integration_start_idx,
        integration_end_idx=leakage_sensitive_classifier.integration_end_idx,
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
            "leakage_sensitive_classifier": leakage_sensitive_classifier,
            "leakage_sensitive_assignment_fidelity": leakage_sensitive_assignment_fidelity,
            "leakage_sensitive_readout_frequency": leakage_sensitive_readout_frequency,
            "leakage_sensitive_readout_detuning": leakage_sensitive_readout_detuning,
            "ge_readout_pulse": ge_readout_pulse,
            "ge_classifier": ge_classifier,
            "iq_data": iq_data,
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
