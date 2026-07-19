#!/usr/bin/env python3
"""backend/chord_training/pack_bundle.py

Packs the Phase B desktop training bundle: audio/labs + manifests (paths
rewritten relative to the bundle root), the finetune kit, the five packaged
pretrained checkpoints, RUNBOOK.md and requirements-train.txt.

Run:
  cd backend && .venv/bin/python chord_training/pack_bundle.py \\
      --train-manifest train_manifest.txt --val-manifest val_manifest.txt \\
      --out training_bundle.zip
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from lv_chordia.mir.common import CACHE_DATA_PATH

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
# ponytail: was hardcoded to BACKEND/".venv"/"share"/..., which assumes a venv
# literally named .venv -- broke in CI (no .venv there, just the system
# Python) and on any dev setup with a differently named venv. CACHE_DATA_PATH
# is lv_chordia's own resolution of where it installed its packaged
# checkpoints, correct regardless of venv layout.
CHECKPOINTS_DIR = Path(CACHE_DATA_PATH)
KIT_FILES = ["dataset.py", "finetune.py", "FINDINGS.md"]


def _read_manifest(path: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        audio, lab = line.split("\t")
        pairs.append((Path(audio), Path(lab)))
    return pairs


def _stage_pairs(pairs: list[tuple[Path, Path]], data_dir: Path,
                  copied: dict[str, tuple[str, str]]) -> list[str]:
    """Copy each (audio, lab) into data_dir (deduped across train+val via
    `copied`, keyed by original audio path) and return bundle-relative
    `data/<name>\\tdata/<name>` manifest lines."""
    lines = []
    for audio, lab in pairs:
        key = str(audio)
        if key not in copied:
            idx = len(copied)
            audio_name = f"{idx:04d}_{audio.name}"
            lab_name = f"{idx:04d}_{lab.name}"
            shutil.copy2(audio, data_dir / audio_name)
            shutil.copy2(lab, data_dir / lab_name)
            copied[key] = (audio_name, lab_name)
        audio_name, lab_name = copied[key]
        lines.append(f"data/{audio_name}\tdata/{lab_name}")
    return lines


def build_bundle(train_manifest: Path, val_manifest: Path, out: Path) -> None:
    checkpoints = sorted(CHECKPOINTS_DIR.glob("*.sdict"))
    if not checkpoints:
        raise SystemExit(f"no .sdict checkpoints found under {CHECKPOINTS_DIR}")

    staging = out.parent / f"{out.stem}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    data_dir = staging / "data"
    data_dir.mkdir(parents=True)

    copied: dict[str, tuple[str, str]] = {}
    train_lines = _stage_pairs(_read_manifest(train_manifest), data_dir, copied)
    val_lines = _stage_pairs(_read_manifest(val_manifest), data_dir, copied)
    (staging / "train_manifest.txt").write_text("\n".join(train_lines) + "\n")
    (staging / "val_manifest.txt").write_text("\n".join(val_lines) + "\n")

    kit_dir = staging / "kit"
    kit_dir.mkdir()
    for name in KIT_FILES:
        shutil.copy2(HERE / name, kit_dir / name)

    ckpt_dir = staging / "checkpoints_in"
    ckpt_dir.mkdir()
    for ckpt in checkpoints:
        shutil.copy2(ckpt, ckpt_dir / ckpt.name)

    shutil.copy2(HERE / "RUNBOOK.md", staging / "RUNBOOK.md")
    shutil.copy2(HERE / "requirements-train.txt", staging / "requirements-train.txt")

    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())
    shutil.rmtree(staging)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({size_mb:.1f} MB)")
    print(f"  data/: {len(copied)} audio+lab pairs "
          f"({len(train_lines)} train lines, {len(val_lines)} val lines)")
    print(f"  checkpoints_in/: {len(checkpoints)} .sdict")
    print(f"  kit/: {', '.join(KIT_FILES)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack the Phase B desktop training bundle")
    ap.add_argument("--train-manifest", required=True, type=Path)
    ap.add_argument("--val-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    build_bundle(args.train_manifest.resolve(), args.val_manifest.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
