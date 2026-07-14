import numpy as np
import qubex as qx
import matplotlib.pyplot as plt
from qubex.experiment.models import Result
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib.pyplot as plt


def two_steps(
    *,
    height1: float,
    height2: float,
    rise_duration: float,
    top1_duration: float,
    fall1_duration: float,
    top2_duration: float,
    fall2_duration: float,
    dt: float = 2.0,
) -> qx.pulse.Arbitrary:
    """
    Create a smooth two-step pulse.

    Shape:
        0 -> height1 -> height2 -> 0

    Parameters
    ----------
    height1:
        First plateau amplitude.
    height2:
        Second plateau amplitude.
    rise_duration:
        Duration of smooth rise from 0 to height1 [ns].
    fall1_duration:
        Duration of smooth fall from height1 to height2 [ns].
    fall2_duration:
        Duration of smooth fall from height2 to 0 [ns].
    top1_duration:
        Duration of first plateau at height1 [ns].
    top2_duration:
        Duration of second plateau at height2 [ns].
    dt:
        Sampling period [ns].
    """

    for name, value in {
        "rise_duration": rise_duration,
        "fall1_duration": fall1_duration,
        "fall2_duration": fall2_duration,
        "top1_duration": top1_duration,
        "top2_duration": top2_duration,
        "dt": dt,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    total_duration = (
        rise_duration + top1_duration + fall1_duration + top2_duration + fall2_duration
    )

    n_samples = round(total_duration / dt)
    if not np.isclose(n_samples * dt, total_duration):
        raise ValueError("total_duration must be a multiple of dt.")

    # サンプル区間の中心で評価
    t = (np.arange(n_samples) + 0.5) * dt
    y = np.zeros_like(t, dtype=float)

    t0 = 0.0

    # 1. 0 -> height1
    t1 = t0 + rise_duration
    m = (t0 <= t) & (t < t1)
    u = t[m] - t0
    y[m] = height1 * 0.5 * (1 - np.cos(np.pi * u / rise_duration))

    # 2. height1 top
    t0 = t1
    t1 = t0 + top1_duration
    m = (t0 <= t) & (t < t1)
    y[m] = height1

    # 3. height1 -> height2
    t0 = t1
    t1 = t0 + fall1_duration
    m = (t0 <= t) & (t < t1)
    u = t[m] - t0
    y[m] = height1 + (height2 - height1) * 0.5 * (
        1 - np.cos(np.pi * u / fall1_duration)
    )

    # 4. height2 top
    t0 = t1
    t1 = t0 + top2_duration
    m = (t0 <= t) & (t < t1)
    y[m] = height2

    # 5. height2 -> 0
    t0 = t1
    t1 = t0 + fall2_duration
    m = (t0 <= t) & (t < t1)
    u = t[m] - t0
    y[m] = height2 * 0.5 * (1 + np.cos(np.pi * u / fall2_duration))

    return qx.pulse.Arbitrary(y.astype(np.complex128), sampling_period=dt)


def next_pow2(x: float | int, nth: int = 1) -> int:
    """Return the next power of 2 greater than or equal to x."""
    if x <= 0:
        raise ValueError("x must be positive.")
    if nth < 1:
        raise ValueError("nth must be a positive integer.")
    return 2 ** (int(np.ceil(np.log2(x))) + nth - 1)


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_fit(x, y):
    p0 = [max(y), np.mean(x), np.std(x)]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    return {"a": popt[0], "mu": popt[1], "sigma": popt[2]}


def create_iq_data(result, target, n_train):
    iq_data = dict()
    for s, res in result.items():
        iq_data[s] = dict()
        for o, data in zip(["prior", "posterior"], res.data[target]):
            iq_data[s][o] = dict()
            for d in ["train", "test"]:
                if d == "train":
                    iq_data[s][o][d] = data.raw[:n_train]
                else:
                    iq_data[s][o][d] = data.raw[n_train:]
    return iq_data


def make_weight(iq_data_g, iq_data_e):
    weight = np.conj(iq_data_g.mean(axis=0) - iq_data_e.mean(axis=0))
    weight = weight / np.linalg.norm(weight)
    return weight


def convert_data(iq_data):
    return np.column_stack([np.real(iq_data), np.imag(iq_data)])


def run_single_shot_measurement_with_post_selection(
    exp: qx.Experiment,
    *,
    target: str,
    height1: float,
    height2: float,
    rise_duration: float,
    top1_duration: float,
    fall1_duration: float,
    top2_duration: float,
    fall2_duration: float,
    waittime: float,
    shot_interval: float,
    n_shots: int,
    plot_pulse_schedule: bool = True,
):
    qubit = exp.ctx.qubits[target].label
    resonator = exp.ctx.resonators[target].label
    readout_pulse_org = two_steps(
        height1=height1,
        height2=height2,
        rise_duration=rise_duration,
        top1_duration=top1_duration,
        fall1_duration=fall1_duration,
        top2_duration=top2_duration,
        fall2_duration=fall2_duration,
        dt=2.0,
    )
    if plot_pulse_schedule:
        readout_pulse_org.plot(title="Original Readout Pulse")
    readout_pulse = readout_pulse_org.padded(
        next_pow2(readout_pulse_org.duration, 1), "right"
    )
    if plot_pulse_schedule:
        readout_pulse.plot(title="Padded Readout Pulse")
    result = dict()
    for state in ["g", "e"]:
        schedule = qx.PulseSchedule()
        with schedule as s:
            s.add(resonator, readout_pulse)
            s.barrier()
            s.add(qubit, qx.pulse.Blank(waittime))
            if state == "e":
                s.add(qubit, exp.drag_pi_pulse[qubit])
            s.barrier()
            s.add(resonator, readout_pulse)
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


def plot_single_shot_measurement_with_post_selection(
    *,
    iq_data,
    signal,
    post_selection,
    assignment_fidelity,
    alpha_scatter=0.01,
    alpha_hist=0.5,
    colors={"g": "blue", "e": "red"},
    labels={"g": "|g>", "e": "|e>"},
    n_bins=100,
):
    figures = dict()
    posterior_averaged_iq = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        posterior_averaged_iq[w] = dict()
        for s in ["g", "e"]:
            posterior_averaged_iq[w][s] = dict()
            for d in ["train", "test"]:
                if w == "before_post_selection":
                    posterior_averaged_iq[w][s][d] = iq_data[s]["posterior"][d].mean(axis=0)
                else:
                    posterior_averaged_iq[w][s][d] = iq_data[s]["posterior"][d][
                            post_selection[s][d], :
                    ].mean(axis=0)
    posterior_signal = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        posterior_signal[w] = dict()
        for s in ["g", "e"]:
            posterior_signal[w][s] = dict()
            for d in ["train", "test"]:
                if w == "before_post_selection":
                    posterior_signal[w][s][d] = signal[s]["posterior"][d]
                else:
                    posterior_signal[w][s][d] = signal[s]["posterior"][d][
                        post_selection[s][d]
                    ]
    # TD averaged iq data
    figures["td_iq"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["td_iq"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in ["g", "e"]:
                ax.scatter(
                    np.arange(len(posterior_averaged_iq[w][s][d]))*8.0,
                    np.real(posterior_averaged_iq[w][s][d]),
                    color=colors[s],
                    label=f"I_{labels[s]}",
                    marker="o",
                )
                ax.scatter(
                    np.arange(len(posterior_averaged_iq[w][s][d]))*8.0,
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
    # IQ averaged separation
    figures["td_separation"] = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["td_separation"][w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            separation = (
                posterior_averaged_iq[w]["e"][d] - posterior_averaged_iq[w]["g"][d]
            )
            ax.scatter(
                np.arange(len(separation))*8.0,
                np.abs(separation),
                color="black",
            )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.grid()
            title = f"Time-domain averaged IQ separation {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
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
                    np.real(posterior_signal[w][s][d]),
                    np.imag(posterior_signal[w][s][d]),
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
    gaussian_params = dict()
    snr = dict()
    for w in ["before_post_selection", "after_post_selection"]:
        figures["iq_histogram"][w] = dict()
        gaussian_params[w] = {"g": dict(), "e": dict()}
        snr[w] = dict()
        for d in ["train", "test"]:
            fig, ax = plt.subplots()
            for s in ["g", "e"]:
                counts, bin_edges, _ = ax.hist(
                    np.real(posterior_signal[w][s][d]),
                    bins=n_bins,
                    color=colors[s],
                    alpha=alpha_hist,
                    label=labels[s],
                )
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                gaussian_params[w][s][d] = gaussian_fit(bin_centers, counts)
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
                y_fit = gaussian(x_fit, **gaussian_params[w][s][d])
                ax.plot(x_fit, y_fit, color=colors[s], linestyle="--")
            snr[w][d] = np.sqrt(
                (gaussian_params[w]["g"][d]["mu"] - gaussian_params[w]["e"][d]["mu"])
                ** 2
                / (
                    gaussian_params[w]["g"][d]["sigma"] ** 2
                    + gaussian_params[w]["e"][d]["sigma"] ** 2
                )
            )
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
                    text = ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color="black",
                    )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["|g>", "|e>"])
            ax.set_yticklabels(["|g>", "|e>"])
            ax.set_xlabel("Prepared State")
            ax.set_ylabel("Observed State")
            title = f"Assignment Fidelity Matrix {w.replace('_', ' ')} ({d} set)"
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            plt.close(fig)
            figures["assignment_fidelity"][w][d] = {"figure": fig, "ax": ax}
    return {
        "figures": figures,
        "gaussian_params": gaussian_params,
        "snr": snr,
    }


def analyze_single_shot_measurement_with_post_selection(
    iq_data, n_train, n_test, n_max_epoch=10
):
    weight = make_weight(
        iq_data["g"]["posterior"]["train"], iq_data["e"]["posterior"]["train"]
    )
    post_selection = dict()
    post_selection["g"] = dict()
    post_selection["e"] = dict()
    post_selection["g"]["train"] = np.ones(n_train, dtype=bool)
    post_selection["e"]["train"] = np.ones(n_train, dtype=bool)
    post_selection["g"]["test"] = np.ones(n_test, dtype=bool)
    post_selection["e"]["test"] = np.ones(n_test, dtype=bool)
    num_post_selection_g = sum(post_selection["g"]["train"])
    num_post_selection_e = sum(post_selection["e"]["train"])
    for i_epoch in range(n_max_epoch):
        print("epoch: ", i_epoch)
        print("# of |e> at the 1st meas. (g train): ", num_post_selection_g)
        print("# of |e> at the 1st meas. (e train): ", num_post_selection_e)
        signal = dict()
        for s in ["g", "e"]:
            signal[s] = dict()
            for o in ["prior", "posterior"]:
                signal[s][o] = dict()
                for d in ["train", "test"]:
                    signal[s][o][d] = iq_data[s][o][d] @ weight
        train_data = np.concatenate(
            [
                signal["g"]["posterior"]["train"][post_selection["g"]["train"]],
                signal["e"]["posterior"]["train"][post_selection["e"]["train"]],
            ]
        )
        train_label = np.concatenate(
            [
                np.zeros(num_post_selection_g, dtype=int),
                np.ones(num_post_selection_e, dtype=int),
            ]
        )
        classifier = LogisticRegression()
        classifier.fit(convert_data(train_data), train_label)
        for s in ["g", "e"]:
            post_selection[s]["train"] = (
                classifier.predict(convert_data(signal[s]["prior"]["train"])) == 0
            )
        if num_post_selection_g == sum(
            post_selection["g"]["train"]
        ) and num_post_selection_e == sum(post_selection["e"]["train"]):
            print("No change in post-selection. Stop iteration.")
            break
        num_post_selection_g = sum(post_selection["g"]["train"])
        num_post_selection_e = sum(post_selection["e"]["train"])
        if num_post_selection_g == 0 or num_post_selection_e == 0:
            raise ValueError(
                "All data are rejected by post-selection. Please check the parameters."
            )
        weight = make_weight(
            iq_data["g"]["posterior"]["train"][post_selection["g"]["train"], :],
            iq_data["e"]["posterior"]["train"][post_selection["e"]["train"], :],
        )
    for s in ["g", "e"]:
        post_selection[s]["test"] = (
            classifier.predict(convert_data(signal[s]["prior"]["test"])) == 0
        )
    assignment_fidelity = dict()
    assignment_fidelity["before_post_selection"] = dict()
    assignment_fidelity["after_post_selection"] = dict()
    for s in ["g", "e"]:
        assignment_fidelity["before_post_selection"][s] = dict()
        assignment_fidelity["after_post_selection"][s] = dict()
        for d in ["train", "test"]:
            assignment_fidelity["before_post_selection"][s][d] = sum(
                classifier.predict(convert_data(signal[s]["posterior"][d]))
                == (0 if s == "g" else 1)
            ) / len(signal[s]["posterior"][d])
            assignment_fidelity["after_post_selection"][s][d] = sum(
                classifier.predict(
                    convert_data(signal[s]["posterior"][d][post_selection[s][d]])
                )
                == (0 if s == "g" else 1)
            ) / sum(post_selection[s][d])
    for w in ["before_post_selection", "after_post_selection"]:
        assignment_fidelity[w]["avg"] = dict()
        for d in ["train", "test"]:
            assignment_fidelity[w]["avg"][d] = (
                assignment_fidelity[w]["g"][d] + assignment_fidelity[w]["e"][d]
            ) / 2
    return {
        "weight": weight,
        "signal": signal,
        "classifier": classifier,
        "post_selection": post_selection,
        "assignment_fidelity": assignment_fidelity,
    }


def display_result(result: Result, plot: bool = True):
    for w in ["before_post_selection", "after_post_selection"]:
        for d in ["train", "test"]:
            if plot:
                try:
                    from IPython.display import display
                except ImportError:
                    display = lambda x: None
                display(result.figures["td_iq"][w][d]["figure"])
                display(result.figures["td_separation"][w][d]["figure"])
                display(result.figures["iq_scatter"][w][d]["figure"])
                display(result.figures["iq_histogram"][w][d]["figure"])
                display(result.figures["assignment_fidelity"][w][d]["figure"])
            print(f"{w.replace('_', ' ')} ({d} set):")
            print(f"Assinment fidelity")
            for s in ["g", "e"]:
                print(f"  |{s}>: {result.data['assignment_fidelity'][w][s][d]:.4f}")
            print(f"  avg: {result.data['assignment_fidelity'][w]['avg'][d]:.4f}")
            print(f"SNR: {result.data['snr'][w][d]:.4f}")
            print(f"1-erfc(SNR/2)/2: {1-erfc(result.data['snr'][w][d]/2)/2:.4f}")


def single_shot_measurement_with_post_selection(
    exp: qx.Experiment,
    *,
    target: str,
    height1: float,
    height2: float,
    rise_duration: float,
    top1_duration: float,
    fall1_duration: float,
    top2_duration: float,
    fall2_duration: float,
    waittime: float,
    shot_interval: float,
    n_train: int,
    n_test: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    run_result = run_single_shot_measurement_with_post_selection(
        exp,
        target=target,
        height1=height1,
        height2=height2,
        rise_duration=rise_duration,
        top1_duration=top1_duration,
        fall1_duration=fall1_duration,
        top2_duration=top2_duration,
        fall2_duration=fall2_duration,
        waittime=waittime,
        shot_interval=shot_interval,
        n_shots=n_train + n_test,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    iq_data = create_iq_data(run_result, target, n_train)
    analyze_result = analyze_single_shot_measurement_with_post_selection(
        iq_data=iq_data, n_train=n_train, n_test=n_test, n_max_epoch=10
    )
    weight = analyze_result["weight"]
    signal = analyze_result["signal"]
    classifier = analyze_result["classifier"]
    post_selection = analyze_result["post_selection"]
    assignment_fidelity = analyze_result["assignment_fidelity"]
    plot_result = plot_single_shot_measurement_with_post_selection(
        iq_data=iq_data,
        signal=signal,
        post_selection=post_selection,
        assignment_fidelity=assignment_fidelity,
        alpha_scatter=0.05,
        alpha_hist=0.5,
        colors={"g": "blue", "e": "red"},
        labels={"g": "|g>", "e": "|e>"},
        n_bins=100,
    )
    figures = plot_result["figures"]
    gaussian_params = plot_result["gaussian_params"]
    snr = plot_result["snr"]
    result = Result(
        data={
            "target": target,
            "iq_data": iq_data,
            "signal": signal,
            "weight": weight,
            "classifier": classifier,
            "post_selection": post_selection,
            "assignment_fidelity": assignment_fidelity,
            "gaussian_params": gaussian_params,
            "snr": snr,
        },
        figure=None,
        figures=figures,
    )
    display_result(result, plot)
    return result
