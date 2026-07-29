"""backend/tests/test_pseudo_label.py

Task 3 adds a module-level `pytest.importorskip("lv_chordia")` once
label_track needs it, same convention as test_chord_training_finetune.py:
the whole file skips together when lv_chordia/torch aren't installed,
rather than gating individual tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "chord_training"))

lv_chordia = pytest.importorskip("lv_chordia")


def _sine_chord(sr=22050, duration=5.0):
    """C major triad -- same recipe as tests/test_chordia_probs.py."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [261.63, 329.63, 392.00]
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return y.astype(np.float32)


def test_frame_confidences_is_max_over_classes():
    from pseudo_label import frame_confidences

    probs = [np.array([[0.1, 0.9], [0.6, 0.4]])]
    assert list(frame_confidences(probs)) == pytest.approx([0.9, 0.6])


def test_segment_mean_confidence_averages_covered_frames():
    from pseudo_label import segment_mean_confidence

    frame_conf = np.array([0.2, 0.8, 0.8, 0.2])
    # hop=100, sr=1000 -> 0.1s/frame; segment [0.1s, 0.3s) covers frames 1,2
    seg = {"start_time": 0.1, "end_time": 0.3}
    assert segment_mean_confidence(seg, frame_conf, sr=1000, hop=100) == pytest.approx(0.8)


def test_filter_low_confidence_segments_drops_below_threshold():
    from pseudo_label import filter_low_confidence_segments

    frame_conf = np.array([0.9, 0.9, 0.1, 0.1])
    segs = [
        {"start_time": 0.0, "end_time": 0.2, "chord": "C:maj"},
        {"start_time": 0.2, "end_time": 0.4, "chord": "X"},
    ]
    kept = filter_low_confidence_segments(segs, frame_conf, sr=1000, hop=100, threshold=0.5)
    assert kept == [segs[0]]


def test_write_lab_round_trips_through_read_lab(tmp_path):
    from dataset import read_lab
    from pseudo_label import write_lab

    segs = [
        {"start_time": 0.0, "end_time": 1.5, "chord": "C:maj"},
        {"start_time": 1.5, "end_time": 3.0, "chord": "G:maj"},
    ]
    path = tmp_path / "out.lab"
    write_lab(str(path), segs)
    assert read_lab(str(path)) == [(0.0, 1.5, "C:maj"), (1.5, 3.0, "G:maj")]


def test_retained_coverage_fraction():
    from pseudo_label import retained_coverage

    segs = [{"start_time": 0.0, "end_time": 3.0}, {"start_time": 5.0, "end_time": 6.0}]
    assert retained_coverage(segs, track_duration=10.0) == pytest.approx(0.4)


def test_retained_coverage_zero_duration_is_zero():
    from pseudo_label import retained_coverage

    assert retained_coverage([], track_duration=0.0) == 0.0


def test_label_track_threshold_zero_keeps_most_of_track(tmp_path):
    import soundfile as sf

    from pseudo_label import label_track

    y = _sine_chord()
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), y, 22050)
    lab = tmp_path / "clip.lab"

    coverage = label_track(str(wav), str(lab), threshold=0.0)

    assert coverage is not None
    assert coverage > 0.9
    assert lab.exists()


def test_ia_download_url():
    from pseudo_label import _ia_download_url

    assert (
        _ia_download_url("001-7976", "Salsif_Bromen_-_01_-_001.mp3")
        == "https://archive.org/download/001-7976/Salsif_Bromen_-_01_-_001.mp3"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1728.35", 1728.35),  # seconds, the common shape
        ("05:32", 332.0),  # mm:ss, what IA returns for some derivatives
        ("1:02:03", 3723.0),  # hh:mm:ss
        (None, None),  # field absent
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_ia_length_handles_every_shape_ia_returns(raw, expected):
    from pseudo_label import _parse_ia_length

    assert _parse_ia_length(raw) == expected


def test_select_audio_file_picks_longest_original_mp3():
    from pseudo_label import select_audio_file

    files = [
        {"name": "cover.jpg", "format": "JPEG", "source": "original"},
        {"name": "short.mp3", "format": "VBR MP3", "source": "original", "length": "40"},
        {"name": "long.mp3", "format": "VBR MP3", "source": "original", "length": "300"},
        {"name": "long.ogg", "format": "Ogg Vorbis", "source": "derivative", "length": "300"},
    ]
    picked = select_audio_file(files, min_duration_sec=60.0)
    assert picked is not None
    assert picked["name"] == "long.mp3"
    assert picked["duration_sec"] == pytest.approx(300.0)


def test_select_audio_file_returns_none_when_all_too_short():
    from pseudo_label import select_audio_file

    files = [{"name": "a.mp3", "format": "VBR MP3", "source": "original", "length": "30"}]
    assert select_audio_file(files, min_duration_sec=60.0) is None


def test_select_audio_file_skips_files_with_unusable_length():
    from pseudo_label import select_audio_file

    files = [
        {"name": "nolen.mp3", "format": "VBR MP3", "source": "original"},
        {"name": "ok.mp3", "format": "VBR MP3", "source": "original", "length": "120"},
    ]
    picked = select_audio_file(files, min_duration_sec=60.0)
    assert picked["name"] == "ok.mp3"


def test_select_random_identifiers_is_deterministic_and_caps_at_pool_size(tmp_path):
    from pseudo_label import select_random_identifiers

    pool = tmp_path / "pool.txt"
    pool.write_text("a\nb\nc\nd\n")

    ids = select_random_identifiers(str(pool), n=2, seed=42)
    assert len(ids) == 2
    assert set(ids) <= {"a", "b", "c", "d"}
    assert select_random_identifiers(str(pool), n=2, seed=42) == ids
    assert set(select_random_identifiers(str(pool), n=99, seed=0)) == {"a", "b", "c", "d"}


def test_label_track_impossible_threshold_returns_none(tmp_path):
    import soundfile as sf

    from pseudo_label import label_track

    y = _sine_chord()
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), y, 22050)
    lab = tmp_path / "clip.lab"

    coverage = label_track(str(wav), str(lab), threshold=1.01, min_coverage=0.5)

    assert coverage is None
    assert not lab.exists()
