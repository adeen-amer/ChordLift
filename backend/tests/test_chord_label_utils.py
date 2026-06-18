from chord_label_utils import (
    harte_label_to_internal,
    internal_to_harte_label,
    raw_labels_to_segments,
)
from eval_mir_utils import chordlift_to_harte


def test_harte_maj_min():
    assert harte_label_to_internal("F:maj") == "F"
    assert harte_label_to_internal("Bb:min") == "A#m"
    assert harte_label_to_internal("N") is None


def test_harte_bare_root_and_slash():
    assert harte_label_to_internal("C") == "C"
    assert harte_label_to_internal("Bb") == "A#"
    assert harte_label_to_internal("C/7") == "C"
    assert harte_label_to_internal("A:min/b7") == "Am"


def test_harte_extensions_and_qualities():
    assert harte_label_to_internal("F:maj6") == "Fmaj6"
    assert harte_label_to_internal("E:min7(4)") == "Em7"
    assert harte_label_to_internal("F:maj(*3)") == "F"
    assert harte_label_to_internal("G:hdim7") == "Gm7b5"


def test_internal_roundtrip_common():
    for internal in ("F", "A#m", "Cmaj7", "Gm7", "Gm7b5", "Fmaj6", "Dsus4"):
        harte = chordlift_to_harte(internal)
        back = harte_label_to_internal(harte)
        assert back == internal, f"{internal} -> {harte} -> {back}"


def test_internal_to_harte_label():
    assert internal_to_harte_label("A#m") == "A#:min"
    assert internal_to_harte_label("Gm7b5") == "G:hdim7"
    assert internal_to_harte_label("Fmaj6") == "F:maj6"


def test_raw_labels_autochord_tuples():
    segs = raw_labels_to_segments([(0.0, 2.0, "C:maj"), (2.0, 4.0, "G:min")])
    assert len(segs) == 2
    assert segs[0]["chord"] == "C"
    assert segs[1]["chord"] == "Gm"


def test_raw_labels_chordia_dicts():
    segs = raw_labels_to_segments([
        {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj7"},
        {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
    ])
    assert segs[0]["chord"] == "Cmaj7"
    assert segs[1]["chord"] == "F"
