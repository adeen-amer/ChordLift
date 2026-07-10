"""backend/tests/test_chord_training_data.py"""
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_build_manifest_pairs_and_leakage(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for stem in ("track-a", "track-b"):
        (data / f"{stem}.mp3").write_bytes(b"\x00")
        (data / f"{stem}.lab").write_text("0.0\t1.0\tC:maj\n")
    (data / "orphan.mp3").write_bytes(b"\x00")  # no lab -> excluded
    out = tmp_path / "train_manifest.txt"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "build_manifest.py"),
         "--data-dir", str(data), "--out", str(out), "--min-pairs", "2"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    audio, lab = lines[0].split("\t")
    assert audio.endswith(".mp3") and lab.endswith(".lab")


def test_build_manifest_rejects_gold_track(tmp_path):
    gold_stem = "let-it-be"  # known gold-24 id; must trip the leakage gate
    data = tmp_path / "data"
    data.mkdir()
    (data / f"{gold_stem}.mp3").write_bytes(b"\x00")
    (data / f"{gold_stem}.lab").write_text("0.0\t1.0\tC:maj\n")
    out = tmp_path / "m.txt"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "build_manifest.py"),
         "--data-dir", str(data), "--out", str(out), "--min-pairs", "1"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode != 0


def test_billboard_to_lab_converts_events(tmp_path):
    ann = tmp_path / "ann" / "0003"
    ann.mkdir(parents=True)
    (ann / "salami_chords.txt").write_text(
        "# title: Example\n# artist: Example Artist\n"
        "0.0\tsilence\n"
        "0.5\tA, intro, | C:maj | F:maj |\n"
        "4.5\tB, verse, | G:maj | C:maj |\n"
        "8.5\tend\n"
    )
    out = tmp_path / "labs"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "billboard_to_lab.py"),
         "--annotations-dir", str(tmp_path / "ann"), "--out-labs", str(out)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    lab = (out / "0003.lab").read_text().strip().splitlines()
    # 4 bars over two 4s sections -> 4 chord segments, Harte labels preserved
    assert len(lab) == 4
    first_start, first_end, first_chord = lab[0].split("\t")
    assert first_chord == "C:maj"
    assert float(first_start) == pytest.approx(0.5)
    assert float(first_end) == pytest.approx(2.5)
