"""Tests for calibration note helpers."""

from datetime import datetime, timedelta

import pytest

from qubex.experiment.models.calibration_note import CalibrationNote
from qubex.experiment.models.experiment_note import ExperimentNote


def test_inheritance():
    """CalibrationNote should inherit from ExperimentNote."""
    assert issubclass(CalibrationNote, ExperimentNote)


def test_empty_init():
    """CalibrationNote should raise a TypeError if no chip_id is provided."""
    with pytest.raises(TypeError):
        CalibrationNote()  # type: ignore


def test_init(tmp_path):
    """CalibrationNote should be initialized with a chip_id."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    assert note.chip_id == chip_id
    assert note.file_path == calibration_dir / f"{chip_id}.json"
    assert note.rabi_params == {}
    assert note.hpi_params == {}
    assert note.pi_params == {}
    assert note.drag_hpi_params == {}
    assert note.drag_pi_params == {}
    assert note.state_params == {}
    assert note.cr_params == {}


def test_update_rabi_param(tmp_path):
    """CalibrationNote should update the rabi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
            "distance": 0.5,
            "r2": 0.5,
            "reference_phase": 0.0,
        },
    )
    param = note.get_rabi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["frequency"] == 0.5
    assert param["phase"] == 0.5
    assert param["offset"] == 0.5
    assert param["noise"] == 0.5
    assert param["angle"] == 0.5


def test_update_hpi_param(tmp_path):
    """CalibrationNote should update the hpi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "tau": 0.5,
        },
    )
    param = note.get_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["tau"] == 0.5


def test_update_pi_param(tmp_path):
    """CalibrationNote should update the pi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "tau": 0.5,
        },
    )
    param = note.get_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["tau"] == 0.5


def test_update_drag_hpi_param(tmp_path):
    """CalibrationNote should update the drag_hpi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_drag_hpi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
        },
    )
    param = note.get_drag_hpi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["beta"] == 0.5


def test_update_drag_pi_param(tmp_path):
    """CalibrationNote should update the drag_pi_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_drag_pi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
        },
    )
    param = note.get_drag_pi_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["amplitude"] == 0.5
    assert param["duration"] == 0.5
    assert param["beta"] == 0.5


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_a_drag_param_carries_its_envelope_through_a_save_and_load(
    tmp_path, pulse_type
):
    """CalibrationNote should round-trip the DRAG envelope through the file."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    getattr(note, f"update_drag_{pulse_type}_param")(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
            "type": "Blackman",
        },
    )
    note.save()

    reloaded = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    param = getattr(reloaded, f"get_drag_{pulse_type}_param")("Q00") or {}
    assert param["type"] == "Blackman"


@pytest.mark.parametrize("pulse_type", ["hpi", "pi"])
def test_a_drag_param_without_an_envelope_reads_back_as_none(tmp_path, pulse_type):
    """CalibrationNote should leave `type` absent for a calibration that omits it."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    getattr(note, f"update_drag_{pulse_type}_param")(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "duration": 0.5,
            "beta": 0.5,
        },
    )

    param = getattr(note, f"get_drag_{pulse_type}_param")("Q00") or {}
    # `PulseService` reads it with `.get`, so an old note means a Gaussian.
    assert param.get("type") is None


def test_update_state_param(tmp_path):
    """CalibrationNote should update the state_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_state_param(
        "Q00",
        {
            "target": "Q00",
            "centers": {"0": [0.5, 0.5], "1": [0.5, 0.5]},
            "reference_phase": 0.0,
        },
    )
    param = note.get_state_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["centers"]["0"] == [0.5, 0.5]
    assert param["centers"]["1"] == [0.5, 0.5]


def test_update_cr_param(tmp_path):
    """CalibrationNote should update the cr_param."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_cr_param(
        "Q00",
        {
            "target": "Q00",
            "duration": 0.5,
            "ramptime": 0.5,
            "cr_amplitude": 0.5,
            "cr_phase": 0.5,
            "cr_beta": 0.5,
            "cancel_amplitude": 0.5,
            "cancel_phase": 0.5,
            "cancel_beta": 0.5,
            "rotary_amplitude": 0.5,
            "zx_rotation_rate": 0.5,
        },
    )
    param = note.get_cr_param("Q00") or {}
    assert param["target"] == "Q00"
    assert param["duration"] == 0.5
    assert param["ramptime"] == 0.5
    assert param["cr_amplitude"] == 0.5
    assert param["cr_phase"] == 0.5
    assert param["cr_beta"] == 0.5
    assert param["cancel_amplitude"] == 0.5
    assert param["cancel_phase"] == 0.5
    assert param["cancel_beta"] == 0.5
    assert param["rotary_amplitude"] == 0.5
    assert param["zx_rotation_rate"] == 0.5


def test_timestamp(tmp_path):
    """CalibrationNote should return the timestamp of the last update."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
            "distance": 0.5,
            "r2": 0.5,
            "reference_phase": 0.0,
        },
    )
    param = note.get_rabi_param("Q00") or {}
    timestamp_text = param.get("timestamp")
    assert timestamp_text is not None
    timestamp = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 1.0,
            "frequency": 1.0,
            "phase": 1.0,
            "offset": 1.0,
            "noise": 1.0,
            "angle": 1.0,
            "distance": 1.0,
            "r2": 1.0,
            "reference_phase": 0.0,
            "timestamp": datetime.strftime(
                timestamp + timedelta(seconds=1), "%Y-%m-%d %H:%M:%S"
            ),
        },
    )
    updated_param = note.get_rabi_param("Q00") or {}
    updated_timestamp_text = updated_param.get("timestamp")
    assert updated_timestamp_text is not None
    updated_timestamp = datetime.strptime(updated_timestamp_text, "%Y-%m-%d %H:%M:%S")
    assert updated_timestamp > timestamp


def test_get_param_with_valid_days(tmp_path):
    """CalibrationNote should return None if the param is older than valid days."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_rabi_param(
        "Q00",
        {
            "target": "Q00",
            "amplitude": 0.5,
            "frequency": 0.5,
            "phase": 0.5,
            "offset": 0.5,
            "noise": 0.5,
            "angle": 0.5,
            "distance": 0.5,
            "r2": 0.5,
            "reference_phase": 0.0,
            "timestamp": datetime.strftime(
                datetime.now() - timedelta(days=2), "%Y-%m-%d %H:%M:%S"
            ),
        },
    )
    assert note.get_rabi_param("Q00", valid_days=1) is None
    assert note.get_rabi_param("Q00", valid_days=3) is not None


def test_higher_transition_labels_round_trip(tmp_path):
    """CalibrationNote should store parameters keyed by higher transition labels."""
    chip_id = "CHIP_ID"
    calibration_dir = tmp_path / ".calibration"
    note = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    note.update_hpi_param(
        "Q00-hi",
        {
            "target": "Q00-hi",
            "amplitude": 0.25,
            "duration": 30.0,
            "tau": 10.0,
        },
    )
    note.update_pi_param(
        "Q00-mn",
        {
            "target": "Q00-mn",
            "amplitude": 0.125,
            "duration": 30.0,
            "tau": 10.0,
        },
    )
    note.save()

    reloaded = CalibrationNote(chip_id=chip_id, calibration_dir=calibration_dir)
    hpi_param = reloaded.get_hpi_param("Q00-hi") or {}
    pi_param = reloaded.get_pi_param("Q00-mn") or {}

    assert hpi_param["target"] == "Q00-hi"
    assert hpi_param["amplitude"] == 0.25
    assert pi_param["target"] == "Q00-mn"
    assert pi_param["amplitude"] == 0.125
    assert reloaded.get_hpi_param("Q00") is None
