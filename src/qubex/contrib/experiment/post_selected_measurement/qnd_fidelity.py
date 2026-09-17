"""
QND-fidelity measurement with two consecutive readouts.

For each sequence ("standard" and "mist_sensitive") and each prepared state
(|g> and |e>) the qubit is optionally prepared with the DRAG pi pulse and
then read out twice:

- standard:       [prep] -- readout -- wait -- readout
- mist_sensitive: [prep] -- readout -- wait -- drag_pi_pulse -- readout

Both readouts are classified shot by shot with a pre-trained
:class:`.assignment_fidelity.Classifier`. The readout pulse and the
classifier must come from the same
:func:`.assignment_fidelity.post_selected_assignment_fidelity_experiment`
run and are passed bundled as
``readout_classifier={"readout_pulse": ..., "classifier": ...}`` (the key
names used in that experiment's ``Result.data``).

The first readout acts as the conditioning (post-selection): for each
prepared state s the shots whose first outcome is s are kept and the
conditional distribution of the second outcome forms one row of

    qnd_matrix[i][j] = P(second readout = j | first readout = i)

The QND fidelity is the diagonal entry for the standard sequence (the second
readout should repeat the first) and the anti-diagonal entry for the
mist_sensitive sequence (the pi pulse before the second readout should flip
the outcome). If a measurement-induced state transition (MIST) occurs during
the first readout, the pi pulse of the mist_sensitive sequence cannot flip
the state back, so the shot is correctly counted as non-QND -- the standard
sequence alone would count it as QND.

Caveats:

- The binary g/e classifier cannot flag leakage (e.g. the f state) or other
  out-of-distribution signals; the mist_sensitive sequence is a MIST-sensitive
  proxy, not a direct transition or leakage measurement.
- The inter-readout spacing of the mist_sensitive sequence is longer than that
  of the standard sequence by the pi-pulse duration plus
  ``waittime_after_ctrl_pulse``; the actual spacings are stored in the
  result data (``inter_readout_spacing``).
"""

import numpy as np
import qubex as qx
import matplotlib.pyplot as plt
from qubex.experiment.models import Result
from qxpulse import Waveform

MIN_CONDITIONED_SHOTS = 10

SEQUENCES = ["standard", "mist_sensitive"]
STATES = ["g", "e"]
STATE_INDEX = {"g": 0, "e": 1}

__all__ = [
    "qnd_fidelity_experiment",
    "run_qnd_fidelity",
    "analyze_qnd_fidelity",
    "display_qnd_fidelity",
    "plot_qnd_fidelity",
]


def _validate_readout_pulse_start_position(
    exp: qx.Experiment,
    *,
    target: str,
    readout_duration: float,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    pump_buffer_time: float,
    capture_sampling_period: float,
):
    qubit = exp.ctx.qubits[target].label
    drag_pi_pulse_duration = exp.drag_pi_pulse[qubit].duration
    first_readout_start = (
        pump_buffer_time + drag_pi_pulse_duration + waittime_after_ctrl_pulse
    )
    if first_readout_start % capture_sampling_period != 0:
        raise ValueError(
            f"The start position of the first readout pulse ({first_readout_start} ns) "
            f"is not aligned with the capture sampling period ({capture_sampling_period} ns)."
        )
    for sequence in SEQUENCES:
        inter_readout_spacing = readout_duration + waittime_after_readout_pulse
        if sequence == "mist_sensitive":
            inter_readout_spacing += drag_pi_pulse_duration + waittime_after_ctrl_pulse
        if inter_readout_spacing % capture_sampling_period != 0:
            raise ValueError(
                f"The spacing between the readout pulses ({inter_readout_spacing} ns, "
                f"{sequence} sequence) is not aligned with the capture sampling period "
                f"({capture_sampling_period} ns)."
            )


def run_qnd_fidelity(
    exp: qx.Experiment,
    *,
    target: str,
    readout_pulse: Waveform,
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
    drag_pi_pulse = exp.drag_pi_pulse[qubit]
    if waittime_after_readout_pulse + drag_pi_pulse.duration < exp.readout_post_margin:
        raise ValueError(
            f"`waittime_after_readout_pulse` ({waittime_after_readout_pulse} ns) + `drag_pi_pulse.duration` ({drag_pi_pulse.duration} ns) must "
            f"be >= `exp.readout_post_margin` ({exp.readout_post_margin} ns)."
        )
    padded_readout_pulse = readout_pulse.padded(
        readout_pulse.duration + exp.readout_post_margin
    )
    barrier_labels = [qubit, resonator]
    result = dict()
    for sequence in SEQUENCES:
        pump_duration = (
            2 * drag_pi_pulse.duration
            + 2 * waittime_after_ctrl_pulse
            + 2 * padded_readout_pulse.duration
            + waittime_after_readout_pulse
            + 2 * pump_buffer_time
        )
        result[sequence] = dict()
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
                s.add(qubit, qx.pulse.Blank(pump_buffer_time))
                if state == "e":
                    s.add(qubit, drag_pi_pulse)
                else:
                    s.add(qubit, qx.pulse.Blank(drag_pi_pulse.duration))
                s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
                s.barrier(labels=barrier_labels)
                s.add(resonator, padded_readout_pulse)
                s.add(
                    qubit,
                    qx.pulse.Blank(
                        readout_pulse.duration + waittime_after_readout_pulse
                    ),
                )
                if sequence == "mist_sensitive":
                    s.add(qubit, drag_pi_pulse)
                else:
                    s.add(qubit, qx.pulse.Blank(drag_pi_pulse.duration))
                s.add(qubit, qx.pulse.Blank(waittime_after_ctrl_pulse))
                s.barrier(labels=barrier_labels)
                s.add(resonator, padded_readout_pulse)
                s.add(resonator, qx.pulse.Blank(pump_buffer_time))
            if plot_pulse_schedule:
                schedule.plot(title=f"Pulse Schedule {sequence} |{state}>")
            result[sequence][state] = exp.execute(
                schedule=schedule,
                mode="single",
                time_integration=False,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
    return result


def _extract_iq_data(*, target, run_result):
    iq_data = dict()
    for sequence in SEQUENCES:
        if sequence not in run_result:
            raise ValueError(f"`run_result` does not contain the {sequence} sequence.")
        iq_data[sequence] = dict()
        for state in STATES:
            if state not in run_result[sequence]:
                raise ValueError(
                    f"`run_result` does not contain the {sequence} sequence for "
                    f"|{state}>."
                )
            readouts = run_result[sequence][state].data[target]
            if len(readouts) != 2:
                raise ValueError(
                    f"Expected exactly 2 capture windows but found {len(readouts)} "
                    f"({sequence} sequence, |{state}>)."
                )
            iq_data[sequence][state] = {
                "first": readouts[0].raw,
                "second": readouts[1].raw,
            }
    return iq_data


def _validate_classifier(*, classifier, iq_data):
    if classifier.weight is None or not hasattr(classifier.model, "coef_"):
        raise ValueError(
            "`classifier` must be trained by the post-selected assignment fidelity "
            "experiment before it can be used."
        )
    for sequence in SEQUENCES:
        for state in STATES:
            for order in ["first", "second"]:
                n_samples = iq_data[sequence][state][order].shape[1]
                if n_samples != len(classifier.weight):
                    raise ValueError(
                        f"The number of capture samples ({n_samples}, {sequence} "
                        f"sequence, |{state}>, {order} readout) does not match the "
                        f"length of the classifier weight "
                        f"({len(classifier.weight)}). Use the same readout pulse as "
                        "the one used to train the classifier."
                    )


def analyze_qnd_fidelity(*, target, run_result, classifier):
    iq_data = _extract_iq_data(target=target, run_result=run_result)
    _validate_classifier(classifier=classifier, iq_data=iq_data)
    labels = dict()
    qnd_matrix = dict()
    qnd_fidelity = dict()
    conditioning_rate = dict()
    conditioned_shots = dict()
    joint_probabilities = dict()
    for sequence in SEQUENCES:
        labels[sequence] = dict()
        qnd_fidelity[sequence] = dict()
        conditioning_rate[sequence] = dict()
        conditioned_shots[sequence] = dict()
        joint_probabilities[sequence] = dict()
        matrix = np.zeros((2, 2))
        for state in STATES:
            first_labels = classifier.predict(iq_data[sequence][state]["first"])
            second_labels = classifier.predict(iq_data[sequence][state]["second"])
            labels[sequence][state] = {
                "first": first_labels,
                "second": second_labels,
            }
            joint = np.zeros((2, 2))
            for a in range(2):
                for b in range(2):
                    joint[a, b] = np.mean((first_labels == a) & (second_labels == b))
            joint_probabilities[sequence][state] = joint
            conditioned = first_labels == STATE_INDEX[state]
            n_conditioned = int(np.sum(conditioned))
            if n_conditioned < MIN_CONDITIONED_SHOTS:
                raise ValueError(
                    f"Number of shots with first readout |{state}> "
                    f"({n_conditioned}, {sequence} sequence) is less than "
                    f"{MIN_CONDITIONED_SHOTS}. Please check the parameters."
                )
            conditioning_rate[sequence][state] = float(np.mean(conditioned))
            conditioned_shots[sequence][state] = n_conditioned
            for observed in STATES:
                matrix[STATE_INDEX[state], STATE_INDEX[observed]] = np.mean(
                    second_labels[conditioned] == STATE_INDEX[observed]
                )
        qnd_matrix[sequence] = matrix
        for state in STATES:
            i = STATE_INDEX[state]
            j = i if sequence == "standard" else 1 - i
            qnd_fidelity[sequence][state] = matrix[i, j]
        qnd_fidelity[sequence]["avg"] = (
            qnd_fidelity[sequence]["g"] + qnd_fidelity[sequence]["e"]
        ) / 2
    return {
        "labels": labels,
        "qnd_matrix": qnd_matrix,
        "qnd_fidelity": qnd_fidelity,
        "conditioning_rate": conditioning_rate,
        "conditioned_shots": conditioned_shots,
        "joint_probabilities": joint_probabilities,
    }


def plot_qnd_fidelity(*, qnd_matrix, labels={"g": "|g>", "e": "|e>"}):
    figures = dict()
    figures["qnd_matrix"] = dict()
    for sequence in SEQUENCES:
        matrix = qnd_matrix[sequence]
        fig, ax = plt.subplots()
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
        ax.set_xticklabels([labels["g"], labels["e"]])
        ax.set_yticklabels([labels["g"], labels["e"]])
        ax.set_xlabel("Second Observed State")
        ax.set_ylabel("First Observed State")
        title = f"QND Fidelity Matrix ({sequence} sequence)"
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.close(fig)
        figures["qnd_matrix"][sequence] = {"figure": fig, "ax": ax}
    return {
        "figures": figures,
    }


def display_qnd_fidelity(result: Result, plot: bool = True):
    try:
        from IPython.display import display
    except ImportError:
        display = lambda x: None

    for sequence in SEQUENCES:
        if plot:
            display(result.figures["qnd_matrix"][sequence]["figure"])
        print(f"{sequence} sequence:")
        print("QND fidelity")
        for s in STATES:
            print(f"  |{s}>: {result.data['qnd_fidelity'][sequence][s]:.6f}")
        print(f"  avg: {result.data['qnd_fidelity'][sequence]['avg']:.6f}")
        print("Conditioning on the first readout")
        for s in STATES:
            print(
                f"  rate |{s}>: {result.data['conditioning_rate'][sequence][s]:.6f} "
                f"({result.data['conditioned_shots'][sequence][s]} shots)"
            )
        print("Joint probabilities P(first=a, second=b | prepared)")
        for s in STATES:
            joint = result.data["joint_probabilities"][sequence][s]
            print(
                f"  |{s}>: gg={joint[0, 0]:.6f}, ge={joint[0, 1]:.6f}, "
                f"eg={joint[1, 0]:.6f}, ee={joint[1, 1]:.6f}"
            )


def qnd_fidelity_experiment(
    exp: qx.Experiment,
    *,
    target: str,
    readout_classifier: dict,
    waittime_after_readout_pulse: float,
    waittime_after_ctrl_pulse: float,
    pump_buffer_time: float,
    pump_ramptime: float,
    pump_amplitude: float,
    shot_interval: float,
    n_shots: int,
    plot: bool = True,
    plot_pulse_schedule: bool = True,
) -> Result:
    missing_keys = {"readout_pulse", "classifier"} - set(readout_classifier)
    if missing_keys:
        raise ValueError(
            "`readout_classifier` must contain the keys 'readout_pulse' and "
            f"'classifier' (missing: {sorted(missing_keys)})."
        )
    readout_pulse = readout_classifier["readout_pulse"]
    classifier = readout_classifier["classifier"]
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
        pump_buffer_time=pump_buffer_time,
        capture_sampling_period=capture_sampling_period,
    )
    run_result = run_qnd_fidelity(
        exp,
        target=target,
        readout_pulse=readout_pulse,
        waittime_after_readout_pulse=waittime_after_readout_pulse,
        waittime_after_ctrl_pulse=waittime_after_ctrl_pulse,
        pump_buffer_time=pump_buffer_time,
        pump_ramptime=pump_ramptime,
        pump_amplitude=pump_amplitude,
        shot_interval=shot_interval,
        n_shots=n_shots,
        plot_pulse_schedule=plot_pulse_schedule,
    )
    analyze_result = analyze_qnd_fidelity(
        target=target,
        run_result=run_result,
        classifier=classifier,
    )
    plot_result = plot_qnd_fidelity(qnd_matrix=analyze_result["qnd_matrix"])
    qubit = exp.ctx.qubits[target].label
    drag_pi_pulse_duration = exp.drag_pi_pulse[qubit].duration
    inter_readout_spacing = {
        "standard": readout_pulse.duration + waittime_after_readout_pulse,
        "mist_sensitive": readout_pulse.duration
        + waittime_after_readout_pulse
        + drag_pi_pulse_duration
        + waittime_after_ctrl_pulse,
    }
    result = Result(
        data={
            "target": target,
            "raw": run_result,
            "readout_pulse": readout_pulse,
            "classifier": classifier,
            "labels": analyze_result["labels"],
            "qnd_matrix": analyze_result["qnd_matrix"],
            "qnd_fidelity": analyze_result["qnd_fidelity"],
            "conditioning_rate": analyze_result["conditioning_rate"],
            "conditioned_shots": analyze_result["conditioned_shots"],
            "joint_probabilities": analyze_result["joint_probabilities"],
            "inter_readout_spacing": inter_readout_spacing,
            "capture_sampling_period": capture_sampling_period,
            "parameters": {
                "target": target,
                "waittime_after_readout_pulse": waittime_after_readout_pulse,
                "waittime_after_ctrl_pulse": waittime_after_ctrl_pulse,
                "pump_buffer_time": pump_buffer_time,
                "pump_ramptime": pump_ramptime,
                "pump_amplitude": pump_amplitude,
                "shot_interval": shot_interval,
                "n_shots": n_shots,
            },
        },
        figure=None,
        figures=plot_result["figures"],
    )
    display_qnd_fidelity(result, plot)
    return result
