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
    """C major triad — same recipe as tests/test_chordia_probs.py."""
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


def test_track_download_url():
    from pseudo_label import _track_download_url

    assert (
        _track_download_url("some/path/track.mp3")
        == "https://files.freemusicarchive.org/some/path/track.mp3"
    )


def test_select_random_track_ids_filters_duration_and_is_deterministic(tmp_path):
    from pseudo_label import select_random_track_ids

    csv_path = tmp_path / "pool.csv"
    csv_path.write_text(
        "track_id,duration_sec\n"
        "1,30\n"
        "2,120\n"
        "3,200\n"
        "4,180\n"
    )

    ids = select_random_track_ids(str(csv_path), n=2, min_duration_sec=60.0, seed=42)

    assert len(ids) == 2
    assert 1 not in ids  # below min_duration_sec
    assert set(ids) <= {2, 3, 4}
    assert select_random_track_ids(str(csv_path), n=2, min_duration_sec=60.0, seed=42) == ids


def test_select_random_track_ids_caps_at_pool_size(tmp_path):
    from pseudo_label import select_random_track_ids

    csv_path = tmp_path / "pool.csv"
    csv_path.write_text("track_id,duration_sec\n1,120\n2,180\n")

    ids = select_random_track_ids(str(csv_path), n=10, min_duration_sec=60.0, seed=0)

    assert set(ids) == {1, 2}


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


def test_merge_manifests_appends_new_and_skips_duplicates(tmp_path):
    from pseudo_label import merge_manifests

    train = tmp_path / "train_manifest.txt"
    train.write_text("a.wav\ta.lab\n")
    pseudo = tmp_path / "pseudo_manifest.txt"
    pseudo.write_text("a.wav\ta.lab\nb.wav\tb.lab\n")

    added = merge_manifests(str(train), str(pseudo))

    assert added == ["b.wav\tb.lab"]
    assert train.read_text().splitlines() == ["a.wav\ta.lab", "b.wav\tb.lab"]


def test_cli_label_stage_writes_manifest(tmp_path):
    import subprocess
    import sys as _sys

    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    y = _sine_chord(duration=5.0)
    sf.write(str(audio_dir / "fma-1.wav"), y, 22050)

    manifest_path = tmp_path / "pseudo_manifest.txt"
    r = subprocess.run(
        [_sys.executable, str(BACKEND / "chord_training" / "pseudo_label.py"),
         "--stage", "label", "--data-dir", str(audio_dir),
         "--confidence-threshold", "0.0", "--pseudo-manifest", str(manifest_path)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert manifest_path.exists()
    lines = manifest_path.read_text().splitlines()
    assert len(lines) == 1
    audio_path, lab_path = lines[0].split("\t")
    assert Path(lab_path).exists()


def test_cli_manifest_stage_merges_and_passes_leakage_check(tmp_path):
    import subprocess
    import sys as _sys

    train = tmp_path / "train_manifest.txt"
    train.write_text("")
    pseudo = tmp_path / "pseudo_manifest.txt"
    (tmp_path / "fma-1.wav").write_text("x")
    (tmp_path / "fma-1.lab").write_text("0.0\t1.0\tC:maj\n")
    pseudo.write_text(f"{tmp_path / 'fma-1.wav'}\t{tmp_path / 'fma-1.lab'}\n")

    r = subprocess.run(
        [_sys.executable, str(BACKEND / "chord_training" / "pseudo_label.py"),
         "--stage", "manifest", "--train-manifest", str(train),
         "--pseudo-manifest", str(pseudo)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "fma-1.wav" in train.read_text()
    assert "PASS" in r.stdout or "OK" in r.stdout
