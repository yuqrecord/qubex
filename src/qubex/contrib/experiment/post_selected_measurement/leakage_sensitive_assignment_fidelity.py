import numpy as np
import qubex as qx
import matplotlib.pyplot as plt
from qubex.experiment.models import Result
from qxpulse import Waveform
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
from scipy.special import erfc
from .assignment_fidelity import STATE_TO_COLOR, STATE_TO_LABEL
from copy import deepcopy

__all__ = [
    "post_selected_leakage_sensitive_assignment_fidelity_experiment",
    "run_post_selected_leakage_sensitive_assignment_fidelity",
    "analyze_post_selected_leakage_sensitive_assignment_fidelity",
    "display_post_selected_leakage_sensitive_assignment_fidelity",
    "plot_post_selected_leakage_sensitive_assignment_fidelity",
]


def _validate_readout_pulse_start_position(
    exp: qx.Experiment,
    *,
    target: str,
    ge_readout_duration: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    capture_sampling_period: float,
):
    qubit = exp.ctx.qubits[target].label
    drag_pi_pulse_duration = exp.drag_pi_pulse[qubit].duration
    second_readout_start = (
        ge_readout_duration
        + waittime_after_readout_pulse
        + drag_pi_pulse_duration
        + waittime_after_ctrl_pulse
    )
    if second_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The start position of the second readout pulse ({second_readout_start} ns) "
            f"is not aligned with the capture sampling period ({capture_sampling_period} ns)."
        )


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
    padded_ge_readout_pulse = ge_readout_pulse.padded(
        ge_readout_pulse.duration + exp.readout_post_margin
    )
    padded_leakage_sensitive_readout_pulse = leakage_sensitive_readout_pulse.padded(
        leakage_sensitive_readout_pulse.duration + exp.readout_post_margin
    )
    barrier_labels = [qubit, resonator]
    pump_duration = (
        padded_ge_readout_pulse.duration
        + padded_leakage_sensitive_readout_pulse.duration
        + waittime_after_readout_pulse
        + exp.drag_pi_pulse[qubit].duration
        + waittime_after_ctrl_pulse
        + 2 * pump_buffer_time
    )
    result = dict()
    for state_idx in range(max_level + 1):
        state =  # MSG TO AI: 0なら"g", 1なら"e", 2以上なら"f"のように変換するコードを作成してください
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
                qx.pulse.Blank(readout_pulse.duration + waittime_after_readout_pulse),
            )
            # MSG TO AI: このコメントアウトでは2状態しか準備できないので、state_idxに応じて適切な制御パルスを追加するコードを作成してください
            # if state_idx == "e":
            #     s.add(qubit, exp.drag_pi_pulse[qubit])
            # else:
            #     s.add(qubit, qx.pulse.Blank(exp.drag_pi_pulse[qubit].duration))
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


def plot_post_selected_assignment_fidelity(
    *,
    iq_data,
    classifier,
    post_selection,
    posterior_weighted_signal,
    population_matrix,
    capture_sampling_period,
    alpha_scatter=0.01,
    alpa_histogram=0.5,
    colors=STATE_TO_COLOR,
    labels=STATE_TO_LABEL
):
    figures = dict()
    posterior_averaged_iq = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        posterior_averaged_iq[w] = dict()
        for s in ["g", "e"]:
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
            matrix = np.array(
                # MSG TO AI: population_matrix[w][s_prepared][s_observed][d]の値を使って行列を作成してください。
            )
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color="black" if matrix[i, j] < 0.5 else "white",
                    )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["|g>", "|e>"])
            ax.set_yticklabels(["|g>", "|e>"])
            ax.set_xlabel("Observed State")
            ax.set_ylabel("Prepared State")
            title = f"Assignment Fidelity Matrix {w.replace('_', ' ')} ({d} set)"
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
        for s_observed in ["g", "e", "L"]:
            population_matrix[w][s_prepared][s_observed] = dict()
            for d in ["train", "test"]:
                population_matrix[w][s_prepared][s_observed][d] = sum(
                    classifier.predict(iq_data[s_prepared]["posterior"][d])
                    == (0 if s_observed == "g" else 1 if s_observed == "e" else 2)
                ) / len(iq_data[s_prepared]["posterior"][d])
    w = "after_post_selection"
    population_matrix[w] = dict()
    for s_prepared in iq_data.keys():
        population_matrix[w][s_prepared] = dict()
        for s_observed in ["g", "e", "L"]:
            population_matrix[w][s_prepared][s_observed] = dict()
            for d in ["train", "test"]:
                population_matrix[w][s_prepared][d] = sum(
                    classifier.predict(iq_data[s_prepared]["posterior"][d][post_selection[s_prepared][d]])
                    == (0 if s_observed == "g" else 1 if s_observed == "e" else 2)
                ) / sum(post_selection[s_prepared][d])
    return population_matrix


def analyze_post_selected_assignment_fidelity(
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
    post_selection = {
        s: {
            d: ge_classifier.predict(iq_data[s]["prior"][d]) == 0
            for d in ["train", "test"]
        }
        for s in iq_data.keys()
    }
    classifier = deepcopy(ge_classifier)
    classifier.fit(
        [
            iq_data[s]["posterior"]["train"][post_selection[s]["train"]]
            for s in iq_data.keys()
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


def display_post_selected_assignment_fidelity(result: Result, plot: bool = True):
    try:
        from IPython.display import display
    except ImportError:
        display = lambda x: None

    def simple_display(figures):
        for fig in figures:
            display(fig)

    def tiled_display(figures, n_cols=3):
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

    for w in ["before_post_selection", "after_post_selection"]:
        for d in ["train", "test"]:
            if plot:
                figures_to_display = [
                    result.figures["td_iq"][w][d]["figure"],
                    result.figures["td_iq_distance"][w][d]["figure"],
                    result.figures["td_separation"][w][d]["figure"],
                    result.figures["iq_scatter"][w][d]["figure"],
                    result.figures["iq_histogram"][w][d]["figure"],
                    result.figures["assignment_fidelity"][w][d]["figure"],
                ]
                try:
                    import ipywidgets as widgets

                    tiled_display(figures_to_display, n_cols=3)
                except ImportError:
                    simple_display(figures_to_display)
            print(f"{w.replace('_', ' ')} ({d} set):")
            print("Assignment fidelity")
            for s in ["g", "e"]:
                print(f"  |{s}>: {result.data['assignment_fidelity'][w][s][d]:.4f}")
            print(f"  avg: {result.data['assignment_fidelity'][w]['avg'][d]:.4f}")
            if w == "after_post_selection":
                for s in ["g", "e"]:
                    print(
                        f"Post-selection rate |{s}>: {result.data['post_selection_rate'][s][d]:.4f}"
                    )
            snr = result.data["snr"][w][d]
            print(f"SNR: {snr:.4f}")
            print(
                f"1-erfc(SNR/(2*sqrt(2)))/2: {1 - erfc(snr / (2 * np.sqrt(2))) / 2:.4f} "
                "(equal-variance Gaussian, midpoint threshold)"
            )
            gaussian_fit_snr = result.data["weighted_signal_histogram"][
                "gaussian_fit_snr"
            ][w][d]
            if gaussian_fit_snr is None:
                print("Gaussian fit SNR: N/A (fit failed)")
                print("1-erfc(Gaussian fit SNR/(2*sqrt(2)))/2: N/A (fit failed)")
            else:
                print(f"Gaussian fit SNR: {gaussian_fit_snr:.4f}")
                print(
                    "1-erfc(Gaussian fit SNR/(2*sqrt(2)))/2: "
                    f"{1 - erfc(gaussian_fit_snr / (2 * np.sqrt(2))) / 2:.4f}"
                )


def post_selected_assignment_fidelity_experiment(
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
    ge_readout_pulse = ge_readout_classifier["readout_pulse"]
    ge_classifier = ge_readout_classifier["classifier"]
    if leakage_sensitive_readout_pulse is None:
        leakage_sensitive_readout_pulse = exp.readout(target)
    capture_sampling_period = (
        exp.backend_controller.sampling_period_ns
        * exp.backend_controller.CAPTURE_DECIMATION_FACTOR
    )
    _validate_readout_pulse_start_position(
        exp,
        target=target,
        ge_readout_duration=ge_readout_pulse.duration,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        capture_sampling_period=capture_sampling_period,
    )
    if max_level < 2:
        raise ValueError(
            f"max_level must be at least 2 for leakage-sensitive assignment fidelity "
            f"(got {max_level})."
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
    analyze_result = analyze_post_selected_assignment_fidelity(
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
    plot_result = plot_post_selected_assignment_fidelity(
        iq_data=iq_data,
        classifier=classifier,
        post_selection=post_selection,
        posterior_weighted_signal=posterior_weighted_signal,
        population_matrix=population_matrix,
        capture_sampling_period=capture_sampling_period,
        alpha_scatter=0.05,
        alpa_histogram=0.5,
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
            "population_matrix": population_matrix,
            "capture_sampling_period": capture_sampling_period,
        },
        figure=None,
        figures=figures,
    )
    display_post_selected_assignment_fidelity(result, plot)
    return result
