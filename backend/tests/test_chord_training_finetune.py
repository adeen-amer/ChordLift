"""backend/tests/test_chord_training_finetune.py

CPU end-to-end: synthetic 20s clips -> storages -> dry-run -> 1-round train.
Slow (~1-3 min). Skipped when torch/lv_chordia missing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lv_chordia")
BACKEND = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(BACKEND / "chord_training"))
from dataset import build_storages, encode_labels, normalize_harte, read_lab  # noqa: E402
from lv_chordia.complex_chord import Chord  # noqa: E402


def _make_pair(d: Path, stem: str):
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 20.0, sr * 20, endpoint=False)
    freqs = [(261.63, 329.63, 392.0), (349.23, 440.0, 523.25)]  # C, F triads
    y = np.concatenate([
        sum(np.sin(2 * np.pi * f * t[: sr * 10]) for f in trio) / 3
        for trio in freqs
    ]).astype(np.float32)
    sf.write(str(d / f"{stem}.wav"), y, sr)
    (d / f"{stem}.lab").write_text("0.0\t10.0\tC:maj\n10.0\t20.0\tF:maj\n")


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    d = tmp_path_factory.mktemp("ft_data")
    for stem in ("clip-a", "clip-b"):
        _make_pair(d, stem)
    train = d / "train_manifest.txt"
    val = d / "val_manifest.txt"
    train.write_text(f"{d}/clip-a.wav\t{d}/clip-a.lab\n")
    val.write_text(f"{d}/clip-b.wav\t{d}/clip-b.lab\n")
    return train, val, d


def test_dry_run_ok(manifests, tmp_path):
    train, val, _ = manifests
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "finetune.py"),
         "--train-manifest", str(train), "--val-manifest", str(val),
         "--work-dir", str(tmp_path), "--seeds", "0", "--sample-length", "800", "--dry-run"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY-RUN OK" in r.stdout


def test_read_lab_whitespace_agnostic(tmp_path):
    """Real Isophonics .lab files are space-separated; our Billboard converter
    writes tabs. read_lab must parse both (bug: it used to split on \\t only,
    which silently broke on every real Isophonics train pair)."""
    space_lab = tmp_path / "space.lab"
    space_lab.write_text("0.0 10.0 C:maj\n10.0 20.0 F:maj\n")
    tab_lab = tmp_path / "tab.lab"
    tab_lab.write_text("0.0\t10.0\tC:maj\n10.0\t20.0\tF:maj\n")

    expected = [(0.0, 10.0, "C:maj"), (10.0, 20.0, "F:maj")]
    assert read_lab(str(space_lab)) == expected
    assert read_lab(str(tab_lab)) == expected


@pytest.mark.parametrize("raw,expected", [
    ("A", "A:maj"),
    ("D/5", "D:maj"),
    ("E:min/5", "E:min"),
    ("C:maj(9)", "C:maj"),
    ("N", "N"),
])
def test_normalize_harte(raw, expected):
    """Regression: Chord() rejects bare roots, slash-bass, and parenthesized
    extensions (~60% of real Isophonics segments, measured) -> silently
    fell back to X. normalize_harte must rewrite these into Chord()'s
    dialect before parsing."""
    assert normalize_harte(raw) == expected
    Chord(normalize_harte(raw))  # must not raise


def test_encode_labels_bare_root_not_x(tmp_path):
    """Regression: a bare-root label ('A', Harte shorthand for a major triad)
    used to silently fall back to X in encode_labels because Chord("A")
    raises. After normalize_harte it must decode to a real major-triad
    frame, not X."""
    segs = [(0.0, 1.0, "A")]
    n_frames = 5  # covers the 1.0s segment at HOP/SR ~= 0.023s/frame
    y = encode_labels(segs, n_frames)
    x_arr = Chord("X").to_numpy()
    a_maj = Chord("A:maj").to_numpy()
    assert not (y[0] == x_arr).all(), "bare-root label must not decode to X"
    assert (y[0] == a_maj).all()


def test_build_storages_rerun_into_same_dir_ok(tmp_path):
    """Regression: re-running build_storages into a work-dir that already has
    storage files from a prior run must not crash. It used to call
    FramedH5DataStorage.delete() on a freshly-constructed object, which
    unconditionally does self.root.close() -> AttributeError on None (root is
    only set by load()/create_and_cache(), never by __init__)."""
    _make_pair(tmp_path, "clip")
    pairs = [(str(tmp_path / "clip.wav"), str(tmp_path / "clip.lab"))]
    out_dir = tmp_path / "storage"
    out_dir.mkdir()

    x1, y1 = build_storages(pairs, str(out_dir))
    x2, y2 = build_storages(pairs, str(out_dir))  # must not raise

    assert x1 == x2 and y1 == y2
    for p in (x1, y1):
        assert Path(f"{p}.h5d").exists()


def test_one_round_train_writes_checkpoint(manifests, tmp_path):
    train, val, _ = manifests
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "finetune.py"),
         "--train-manifest", str(train), "--val-manifest", str(val),
         "--work-dir", str(tmp_path), "--seeds", "0", "--sample-length", "800",
         "--lr", "1e-4", "--epochs-cap", "1", "--batch-size", "2"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "SEED 0 DONE" in r.stdout
    out = list((tmp_path / "cache_data").glob("*_ft1_s0.best*"))
    assert out, "fine-tuned checkpoint not written"
