"""Phase 1: mir_eval ruler + normalize_chord_symbol fixes."""
import numpy as np

from eval_chord_utils import normalize_chord_symbol
from eval_mir_utils import (
    chordlift_to_harte,
    evaluate_mir_chords,
    load_harte_lab,
    segments_to_mir,
)


def test_normalize_maj7_not_collapsed_to_7():
    assert normalize_chord_symbol("Cmaj7") == "Cmaj7"
    assert normalize_chord_symbol("Fmaj7") == "Fmaj7"
    assert normalize_chord_symbol("C7") == "C7"


def test_chordlift_to_harte():
    assert chordlift_to_harte("C") == "C:maj"
    assert chordlift_to_harte("Am") == "A:min"
    assert chordlift_to_harte("Cmaj7") == "C:maj7"
    assert chordlift_to_harte("N") == "N"


def test_load_harte_lab(tmp_path):
    lab = tmp_path / "t.lab"
    lab.write_text("0.0 1.0 N\n1.0 2.0 C:maj\n2.0 3.5 G:maj\n")
    intervals, labels = load_harte_lab(lab)
    assert intervals.shape == (3, 2)
    assert labels[0] == "N"
    assert labels[1:] == ["C:maj", "G:maj"]


def test_mir_eval_perfect_match():
    ref_i = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    ref_l = ["C:maj", "G:maj"]
    scores = evaluate_mir_chords(ref_i, ref_l, ref_i, ref_l)
    assert scores["majmin"] == 1.0
    assert scores["sevenths"] == 1.0
    assert scores["root"] == 1.0
    assert scores["seg"] == 1.0


def test_segments_to_mir_harte():
    segs = [
        {"time": 0.0, "end_time": 1.0, "chord": "C"},
        {"time": 1.0, "end_time": 2.0, "chord": "Am"},
    ]
    intervals, labels = segments_to_mir(segs)
    assert labels == ["C:maj", "A:min"]
    assert intervals.shape == (2, 2)


def test_gold_lab_files_committed():
    from pathlib import Path

    root = Path(__file__).resolve().parent / "fixtures" / "gold" / "lab"
    expected = {
        "let-it-be.lab",
        "yellow-submarine.lab",
        "come-together.lab",
        "yesterday.lab",
        "help.lab",
        "twist-and-shout.lab",
        "something.lab",
        "ticket-to-ride.lab",
    }
    assert expected <= {p.name for p in root.glob("*.lab")}
