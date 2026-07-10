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
         "--work-dir", str(tmp_path), "--seeds", "0", "--dry-run"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY-RUN OK" in r.stdout


def test_one_round_train_writes_checkpoint(manifests, tmp_path):
    train, val, _ = manifests
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "finetune.py"),
         "--train-manifest", str(train), "--val-manifest", str(val),
         "--work-dir", str(tmp_path), "--seeds", "0",
         "--lr", "1e-4", "--epochs-cap", "1", "--batch-size", "2"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "SEED 0 DONE" in r.stdout
    out = list((tmp_path / "cache_data").glob("*_ft1_s0.best*"))
    assert out, "fine-tuned checkpoint not written"
