import numpy as np
import qubex as qx
import matplotlib.pyplot as plt
from qubex.experiment.models import Result
from qxpulse import Waveform
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
from scipy.special import erfc

MIN_POST_SELECTED_SHOTS = 10
STATE_TO_COLOR = {
    "g": "blue",
    "e": "red",
    "f": "orange",
    "h": "green",
    "i": "purple",
    "j": "pink",
}
STATE_TO_LABEL = {k: f"|{k}>" for k in STATE_TO_COLOR.keys()}

__all__ = [
    "post_selected_assignment_fidelity_experiment",
    "run_post_selected_assignment_fidelity",
    "analyze_post_selected_assignment_fidelity",
    "display_post_selected_assignment_fidelity",
    "plot_post_selected_assignment_fidelity",
    "Classifier",
]


def _validate_readout_pulse_start_position(
    exp: qx.Experiment,
    *,
    target: str,
    readout_duration: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    capture_sampling_period: float,
):
    qubit = exp.ctx.qubits[target].label
    drag_pi_pulse_duration = exp.drag_pi_pulse[qubit].duration
    second_readout_start = (
        readout_duration
        + waittime_after_readout_pulse
        + drag_pi_pulse_duration
        + waittime_after_ctrl_pulse
    )
    if second_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The start position of the second readout pulse ({second_readout_start} ns) "
            f"is not aligned with the capture sampling period ({capture_sampling_period} ns)."
        )


__all__ = [
    "post_selected_assignment_fidelity_experiment",
    "run_post_selected_assignment_fidelity",
    "analyze_post_selected_assignment_fidelity",
    "display_post_selected_assignment_fidelity",
    "plot_post_selected_assignment_fidelity",
    "Classifier",
]


def _validate_readout_pulse_start_position(
    exp: qx.Experiment,
    *,
    target: str,
    readout_duration: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    capture_sampling_period: float,
):
    qubit = exp.ctx.qubits[target].label
    drag_pi_pulse_duration = exp.drag_pi_pulse[qubit].duration
    second_readout_start = (
        readout_duration
        + waittime_after_readout_pulse
        + drag_pi_pulse_duration
        + waittime_after_ctrl_pulse
    )
    if second_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The start position of the second readout pulse ({second_readout_start} ns) "
            f"is not aligned with the capture sampling period ({capture_sampling_period} ns)."
        )


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_fit(x, y):
    total = np.sum(y)
    if total <= 0:
        return None
    mu0 = np.sum(x * y) / total
    sigma0 = np.sqrt(np.sum(y * (x - mu0) ** 2) / total)
    p0 = [max(y), mu0, sigma0]
    if not np.all(np.isfinite(p0)):
        return None
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(popt)):
        return None
    return {"a": popt[0], "mu": popt[1], "sigma": popt[2]}


def _create_iq_data(*, target, run_result, n_train):
    iq_data = dict()
    for s, res in run_result.items():
        iq_data[s] = dict()
        for o, data in zip(["prior", "posterior"], res.data[target]):
            iq_data[s][o] = dict()
            for d in ["train", "test"]:
                if d == "train":
                    iq_data[s][o][d] = data.raw[:n_train]
                else:
                    iq_data[s][o][d] = data.raw[n_train:]
    return iq_data


class Classifier:
    def __init__(self):
        self.model = LogisticRegression()
        self.weight = None
        self.integration_start_idx = 0
        self.integration_end_idx = None

    @staticmethod
    def expand_complex_feature(c_data):
        return np.column_stack([np.real(c_data), np.imag(c_data)])

    def weighted_signal(self, iq_data):
        return iq_data @ self.weight

    def _update_weight(
        self, iq_data_g, iq_data_e, integration_start_idx=0, integration_end_idx=None
    ):
        self.weight = np.conj(iq_data_e.mean(axis=0) - iq_data_g.mean(axis=0))
        # norm = np.linalg.norm(self.weight)
        # if norm == 0:
        #     raise ValueError(
        #         "Mean IQ data of |g> and |e> are identical. Cannot compute the weight."
        #     )
        # self.weight = self.weight / norm
        self.weight[:integration_start_idx] = 0
        self.integration_start_idx = integration_start_idx
        self.integration_end_idx = integration_end_idx
        if integration_end_idx is not None:
            self.weight[integration_end_idx:] = 0

    def fit(self, iq_data_list, integration_start_idx=0, integration_end_idx=None):
        self._update_weight(
            iq_data_list[0], iq_data_list[1], integration_start_idx, integration_end_idx
        )
        train_data = np.concatenate(
            [self.weighted_signal(iq_data) for iq_data in iq_data_list]
        )
        train_label = np.concatenate(
            [
                np.full(len(iq_data), i, dtype=int)
                for i, iq_data in enumerate(iq_data_list)
            ]
        )
        self.model.fit(self.expand_complex_feature(train_data), train_label)

    def predict(self, iq_data):
        signal = self.weighted_signal(iq_data)
        return self.model.predict(self.expand_complex_feature(signal))


def run_post_selected_assignment_fidelity(
    exp: qx.Experiment,
    *,
    target: str,
    readout_pulse: Waveform | None = None,
    readout_post_margin: float | None = None,
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
    if readout_pulse is None:
        readout_pulse = exp.readout(target)
    if readout_post_margin is None:
        readout_post_margin = exp.readout_post_margin
    padded_readout_pulse = readout_pulse.padded(
        readout_pulse.duration + readout_post_margin
    )
    barrier_labels = [qubit, resonator]
    pump_duration = (
        2 * padded_readout_pulse.duration
        + waittime_after_readout_pulse
        + exp.drag_pi_pulse[qubit].duration
        + waittime_after_ctrl_pulse
        + 2 * pump_buffer_time
    )
    result = dict()
    for state in ["g", "e"]:
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
            s.add(resonator, padded_readout_pulse)
            s.add(
                qubit,
                qx.pulse.Blank(readout_pulse.duration + waittime_after_readout_pulse),
            )
            if state == "e":
                s.add(qubit, exp.drag_pi_pulse[qubit])
            else:
                s.add(qubit, qx.pulse.Blank(exp.drag_pi_pulse[qubit].duration))
            s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
            s.barrier(labels=barrier_labels)
            s.add(resonator, padded_readout_pulse)
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
    integration_start_idx,
    integration_end_idx,
    post_selection,
    posterior_weighted_signal,
    assignment_fidelity,
    weighted_signal_histogram,
    capture_sampling_period,
    alpha_scatter=0.01,
    alpa_histogram=0.5,
    colors=STATE_TO_COLOR,
    labels=STATE_TO_LABEL,
):
    vspan_color = "gray"
    vspan_alpha = 0.2
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
            ax.axvspan(
                integration_start_idx * capture_sampling_period,
                (integration_end_idx - 1) * capture_sampling_period,
                color=vspan_color,
                alpha=vspan_alpha,
                label="Integration Window",
            )
            for s in ["g", "e"]:
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
                label="Integration Window",
            )
            for s in ["g", "e"]:
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
    # IQ averaged separation
    figures["td_separation"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["td_separation"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            ax.axvspan(
                integration_start_idx * capture_sampling_period,
                (integration_end_idx - 1) * capture_sampling_period,
                color=vspan_color,
                alpha=vspan_alpha,
                label="Integration Window",
            )
            separation = (
                posterior_averaged_iq[w]["e"][d] - posterior_averaged_iq[w]["g"][d]
            )
            ax.scatter(
                np.arange(len(separation)) * capture_sampling_period,
                np.abs(separation),
                color="black",
            )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.grid()
            title = (
                f"Time-domain averaged IQ separation {w.replace('_', ' ')} ({d} set)"
            )
            ax.set_title(title)
            ax.legend()
            plt.close(fig)
            figures["td_separation"][w][d] = {"figure": fig, "ax": ax}
    # Plot IQ data
    figures["iq_scatter"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["iq_scatter"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in ["g", "e"]:
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
    # Plot histograms
    figures["iq_histogram"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["iq_histogram"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in ["g", "e"]:
                ax.stairs(
                    weighted_signal_histogram["histogram"][w][s][d]["counts"],
                    weighted_signal_histogram["histogram"][w][s][d]["bin_edges"],
                    color=colors[s],
                    label=labels[s],
                    fill=True,
                    alpha=alpa_histogram,
                )
                fit_params = weighted_signal_histogram["gaussian_fit_params"][w][s][d]
                if fit_params is not None:
                    x_fit = np.linspace(
                        weighted_signal_histogram["histogram"][w][s][d]["bin_edges"][0],
                        weighted_signal_histogram["histogram"][w][s][d]["bin_edges"][
                            -1
                        ],
                        1000,
                    )
                    y_fit = gaussian(
                        x_fit,
                        fit_params["a"],
                        fit_params["mu"],
                        fit_params["sigma"],
                    )
                    ax.plot(x_fit, y_fit, color="black", linestyle="--")
            ax.set_xlabel("I")
            ax.set_ylabel("Counts")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
            ax.grid()
            title = f"Histogram of I {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            ax.legend()
            plt.close(fig)
            figures["iq_histogram"][w][d] = {"figure": fig, "ax": ax}
    # Plot Assignment Fidelity matrix
    figures["assignment_fidelity"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["assignment_fidelity"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            matrix = np.array(
                [
                    [
                        assignment_fidelity[w]["g"][d],
                        1 - assignment_fidelity[w]["g"][d],
                    ],
                    [
                        1 - assignment_fidelity[w]["e"][d],
                        assignment_fidelity[w]["e"][d],
                    ],
                ]
            )
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.6f}",
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
            figures["assignment_fidelity"][w][d] = {"figure": fig, "ax": ax}
    return {
        "figures": figures,
    }


def _build_classifier(
    *,
    iq_data,
    n_train,
    n_test,
    integration_start_idx=0,
    integration_end_idx=None,
    n_max_epoch=10,
    verbose=True,
):
    post_selection = dict()
    post_selection["g"] = dict()
    post_selection["e"] = dict()
    post_selection["g"]["train"] = np.ones(n_train, dtype=bool)
    post_selection["e"]["train"] = np.ones(n_train, dtype=bool)
    post_selection["g"]["test"] = np.ones(n_test, dtype=bool)
    post_selection["e"]["test"] = np.ones(n_test, dtype=bool)
    classifier = Classifier()
    converged = False
    for i_epoch in range(n_max_epoch):
        if verbose:
            print("epoch: ", i_epoch)
            print(
                "# of post-selected shots (g train): ",
                sum(post_selection["g"]["train"]),
            )
            print(
                "# of post-selected shots (e train): ",
                sum(post_selection["e"]["train"]),
            )
        classifier.fit(
            [
                iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
                iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
            ],
            integration_start_idx=integration_start_idx,
            integration_end_idx=integration_end_idx,
        )
        post_selection_prev = {s: post_selection[s]["train"] for s in ["g", "e"]}
        for s in ["g", "e"]:
            post_selection[s]["train"] = (
                classifier.predict(iq_data[s]["prior"]["train"]) == 0
            )
            if sum(post_selection[s]["train"]) < MIN_POST_SELECTED_SHOTS:
                raise ValueError(
                    f"Number of post-selected shots ({s} train) is less than "
                    f"{MIN_POST_SELECTED_SHOTS}. Please check the parameters."
                )
        if all(
            sum(post_selection[s]["train"]) == sum(post_selection_prev[s])
            for s in ["g", "e"]
        ):
            if verbose:
                print("No change in post-selection. Stop iteration.")
            converged = True
            break
    if not converged:
        print(f"Post-selection did not converge within {n_max_epoch} epochs.")
    classifier.fit(
        [
            iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
            iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
        ],
        integration_start_idx=integration_start_idx,
        integration_end_idx=integration_end_idx,
    )
    for s in ["g", "e"]:
        post_selection[s]["test"] = classifier.predict(iq_data[s]["prior"]["test"]) == 0
        if sum(post_selection[s]["test"]) < MIN_POST_SELECTED_SHOTS:
            raise ValueError(
                f"Number of post-selected shots ({s} test) is less than "
                f"{MIN_POST_SELECTED_SHOTS}. Please check the parameters."
            )
    return classifier, post_selection


def _assignment_fidelity(*, iq_data, classifier, post_selection):
    assignment_fidelity = dict()
    w = "before_post_selection"
    assignment_fidelity[w] = dict()
    for s in ["g", "e"]:
        assignment_fidelity[w][s] = dict()
        for d in ["train", "test"]:
            assignment_fidelity[w][s][d] = sum(
                classifier.predict(iq_data[s]["posterior"][d]) == (0 if s == "g" else 1)
            ) / len(iq_data[s]["posterior"][d])
    w = "after_post_selection"
    assignment_fidelity[w] = dict()
    for s in ["g", "e"]:
        assignment_fidelity[w][s] = dict()
        for d in ["train", "test"]:
            assignment_fidelity[w][s][d] = sum(
                classifier.predict(iq_data[s]["posterior"][d][post_selection[s][d]])
                == (0 if s == "g" else 1)
            ) / sum(post_selection[s][d])
    for w in ["before_post_selection", "after_post_selection"]:
        assignment_fidelity[w]["avg"] = dict()
        for d in ["train", "test"]:
            assignment_fidelity[w]["avg"][d] = (
                assignment_fidelity[w]["g"][d] + assignment_fidelity[w]["e"][d]
            ) / 2
    return assignment_fidelity


def _posterior_weighted_signal(*, iq_data, classifier, post_selection):
    posterior_weighted_signal = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        posterior_weighted_signal[w] = dict()
        for s in ["g", "e"]:
            posterior_weighted_signal[w][s] = dict()
            for d in ["train", "test"]:
                if w == "before_post_selection":
                    posterior_weighted_signal[w][s][d] = classifier.weighted_signal(
                        iq_data[s]["posterior"][d]
                    )
                else:
                    posterior_weighted_signal[w][s][d] = classifier.weighted_signal(
                        iq_data[s]["posterior"][d][post_selection[s][d], :]
                    )
    return posterior_weighted_signal


def _signal_to_noise_ratio(*, posterior_weighted_signal):
    snr = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        snr[w] = dict()
        for d in ["train", "test"]:
            mu_g = posterior_weighted_signal[w]["g"][d].real.mean()
            mu_e = posterior_weighted_signal[w]["e"][d].real.mean()
            var_g = posterior_weighted_signal[w]["g"][d].real.var(ddof=1)
            var_e = posterior_weighted_signal[w]["e"][d].real.var(ddof=1)
            snr[w][d] = np.abs(mu_g - mu_e) / np.sqrt((var_g + var_e) / 2)
    return snr


def _weighted_signal_histogram(*, posterior_weighted_signal, n_bins=100):
    histogram = dict()
    gaussian_fit_params = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        histogram[w] = dict()
        gaussian_fit_params[w] = dict()
        for s in ["g", "e"]:
            histogram[w][s] = dict()
            gaussian_fit_params[w][s] = dict()
            for d in ["train", "test"]:
                counts, bin_edges = np.histogram(
                    posterior_weighted_signal[w][s][d].real, bins=n_bins
                )
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                gaussian_fit_params[w][s][d] = gaussian_fit(bin_centers, counts)
                histogram[w][s][d] = {
                    "counts": counts,
                    "bin_edges": bin_edges,
                }
    gaussian_fit_snr = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        gaussian_fit_snr[w] = dict()
        for d in ["train", "test"]:
            params_g = gaussian_fit_params[w]["g"][d]
            params_e = gaussian_fit_params[w]["e"][d]
            if params_g is None or params_e is None:
                gaussian_fit_snr[w][d] = None
            else:
                gaussian_fit_snr[w][d] = np.abs(
                    params_g["mu"] - params_e["mu"]
                ) / np.sqrt((params_g["sigma"] ** 2 + params_e["sigma"] ** 2) / 2)
    return dict(
        histogram=histogram,
        gaussian_fit_params=gaussian_fit_params,
        gaussian_fit_snr=gaussian_fit_snr,
    )


def analyze_post_selected_assignment_fidelity(
    *, target, run_result, n_train, n_test, n_max_epoch=10, verbose=True
):
    iq_data = _create_iq_data(target=target, run_result=run_result, n_train=n_train)
    num_samples = len(iq_data["g"]["posterior"]["train"][0])
    integration_start_idx = 0
    integration_end_idx = num_samples
    highest_assignment_fidelity = 0.0
    for i_start in reversed(range(num_samples)):
        for i_end in range(i_start + 1, num_samples + 1):
            try:
                classifier, post_selection = _build_classifier(
                    iq_data=iq_data,
                    n_train=n_train,
                    n_test=n_test,
                    integration_start_idx=i_start,
                    integration_end_idx=i_end,
                    n_max_epoch=n_max_epoch,
                    verbose=False,
                )
            except Exception:
                continue
            assignment_fidelity = _assignment_fidelity(
                iq_data=iq_data, classifier=classifier, post_selection=post_selection
            )
            fidelity_temp = assignment_fidelity["after_post_selection"]["avg"]["test"]
            if fidelity_temp > highest_assignment_fidelity:
                highest_assignment_fidelity = fidelity_temp
                integration_start_idx = i_start
                integration_end_idx = i_end
    classifier, post_selection = _build_classifier(
        iq_data=iq_data,
        n_train=n_train,
        n_test=n_test,
        integration_start_idx=integration_start_idx,
        integration_end_idx=integration_end_idx,
        n_max_epoch=n_max_epoch,
        verbose=verbose,
    )
    assignment_fidelity = _assignment_fidelity(
        iq_data=iq_data, classifier=classifier, post_selection=post_selection
    )
    posterior_weighted_signal = _posterior_weighted_signal(
        iq_data=iq_data, classifier=classifier, post_selection=post_selection
    )
    weighted_signal_histogram = _weighted_signal_histogram(
        posterior_weighted_signal=posterior_weighted_signal, n_bins=100
    )
    snr = _signal_to_noise_ratio(posterior_weighted_signal=posterior_weighted_signal)
    post_selection_rate = dict()
    for s in ["g", "e"]:
        post_selection_rate[s] = dict()
        for d in ["train", "test"]:
            post_selection_rate[s][d] = sum(post_selection[s][d]) / len(
                post_selection[s][d]
            )
    return {
        "iq_data": iq_data,
        "classifier": classifier,
        "post_selection": post_selection,
        "post_selection_rate": post_selection_rate,
        "posterior_weighted_signal": posterior_weighted_signal,
        "assignment_fidelity": assignment_fidelity,
        "weighted_signal_histogram": weighted_signal_histogram,
        "snr": snr,
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
                print(f"  |{s}>: {result.data['assignment_fidelity'][w][s][d]:.6f}")
            print(f"  avg: {result.data['assignment_fidelity'][w]['avg'][d]:.6f}")
            if w == "after_post_selection":
                for s in ["g", "e"]:
                    print(
                        f"Post-selection rate |{s}>: {result.data['post_selection_rate'][s][d]:.6f}"
                    )
            snr = result.data["snr"][w][d]
            print(f"SNR: {snr:.6f}")
            print(
                f"1-erfc(SNR/(2*sqrt(2)))/2: {1 - erfc(snr / (2 * np.sqrt(2))) / 2:.6f} "
                "(equal-variance Gaussian, midpoint threshold)"
            )
            gaussian_fit_snr = result.data["weighted_signal_histogram"][
                "gaussian_fit_snr"
            ][w][d]
            if gaussian_fit_snr is None:
                print("Gaussian fit SNR: N/A (fit failed)")
                print("1-erfc(Gaussian fit SNR/(2*sqrt(2)))/2: N/A (fit failed)")
            else:
                print(f"Gaussian fit SNR: {gaussian_fit_snr:.6f}")
                print(
                    "1-erfc(Gaussian fit SNR/(2*sqrt(2)))/2: "
                    f"{1 - erfc(gaussian_fit_snr / (2 * np.sqrt(2))) / 2:.6f}"
                )


def post_selected_assignment_fidelity_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    readout_pulse: Waveform | None = None,
    readout_post_margin: float | None = None,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    n_train: int,
    n_test: int,
    n_max_epoch: int = 10,
    verbose: bool = True,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    if readout_pulse is None:
        readout_pulse = exp.readout(target)
    if readout_post_margin is None:
        readout_post_margin = exp.readout_post_margin
    capture_sampling_period = (
        exp.backend_controller.sampling_period_ns
        * exp.backend_controller.CAPTURE_DECIMATION_FACTOR
    )
    _validate_readout_pulse_start_position(
        exp,
        target=target,
        readout_duration=readout_pulse.duration,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        capture_sampling_period=capture_sampling_period,
    )
    run_result = run_post_selected_assignment_fidelity(
        exp,
        target=target,
        readout_pulse=readout_pulse,
        readout_post_margin=readout_post_margin,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        pump_amplitude=pump_amplitude,
        shot_interval=shot_interval,
        n_shots=n_train + n_test,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    print("Finished running the experiment. Analyzing the results...")
    analyze_result = analyze_post_selected_assignment_fidelity(
        target=target,
        run_result=run_result,
        n_train=n_train,
        n_test=n_test,
        n_max_epoch=n_max_epoch,
        verbose=verbose,
    )
    iq_data = analyze_result["iq_data"]
    classifier = analyze_result["classifier"]
    post_selection = analyze_result["post_selection"]
    post_selection_rate = analyze_result["post_selection_rate"]
    posterior_weighted_signal = analyze_result["posterior_weighted_signal"]
    assignment_fidelity = analyze_result["assignment_fidelity"]
    weighted_signal_histogram = analyze_result["weighted_signal_histogram"]
    snr = analyze_result["snr"]
    plot_result = plot_post_selected_assignment_fidelity(
        iq_data=iq_data,
        integration_start_idx=classifier.integration_start_idx,
        integration_end_idx=classifier.integration_end_idx,
        post_selection=post_selection,
        posterior_weighted_signal=posterior_weighted_signal,
        assignment_fidelity=assignment_fidelity,
        weighted_signal_histogram=weighted_signal_histogram,
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
            "readout_pulse": readout_pulse,
            "readout_post_margin": readout_post_margin,
            "iq_data": iq_data,
            "classifier": classifier,
            "post_selection": post_selection,
            "post_selection_rate": post_selection_rate,
            "posterior_weighted_signal": posterior_weighted_signal,
            "assignment_fidelity": assignment_fidelity,
            "weighted_signal_histogram": weighted_signal_histogram,
            "snr": snr,
            "capture_sampling_period": capture_sampling_period,
        },
        figure=None,
        figures=figures,
    )
    display_post_selected_assignment_fidelity(result, plot)
    return result
