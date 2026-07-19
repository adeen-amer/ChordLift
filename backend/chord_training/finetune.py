#!/usr/bin/env python3
"""backend/chord_training/finetune.py

Per-seed warm-start fine-tune CLI. Productionizes spike_warmstart.py's warm-start
recipe (FINDINGS.md §6-9) over real manifests instead of two synthetic clips.

Run:
  cd backend && .venv/bin/python chord_training/finetune.py \\
      --train-manifest train_manifest.txt --val-manifest val_manifest.txt \\
      --work-dir /abs/out/dir --seeds 0,1,2,3,4

Manifest format (Task 2): one line per track, `<audio_path>\\t<lab_path>`.
Paths may be relative (resolved against CWD) or absolute.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import lv_chordia
from lv_chordia.chordnet_ismir_naive import ChordNet, LSTM_TRAIN_LENGTH
from lv_chordia.mir.common import CACHE_DATA_PATH
from lv_chordia.mir.nn.train import NetworkInterface

from dataset import build_storages, make_providers

PKG = os.path.dirname(lv_chordia.__file__)
NAME_TMPL = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{seed}.best"
FT_NAME_TMPL = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft1_s{seed}.best"

# ponytail: default is LSTM_TRAIN_LENGTH, the FINDINGS-verified production
# value (FINDINGS.md §1). Short fixture clips (Task 5, ~20s) fall under this and
# must pass --sample-length explicitly (e.g. 800) to clear the train provider's
# floor; tests do so rather than shrinking the production default.
DEFAULT_SAMPLE_LENGTH = LSTM_TRAIN_LENGTH


def _read_manifest(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audio, lab = line.split("\t")
            if not os.path.isabs(audio):
                audio = os.path.join(os.getcwd(), audio)
            if not os.path.isabs(lab):
                lab = os.path.join(os.getcwd(), lab)
            pairs.append((audio, lab))
    return pairs


def _warm_start(seed: int, cache_dir: str, pretrained_dir: str) -> NetworkInterface:
    """Fresh net under the ft1 save name, hard-loaded with the packaged s{seed}
    weights (FINDINGS.md §6): constructing under a NEW name keeps finalized=False
    so train_supervised will accept it; loading via the constructor instead would
    load the ft1 checkpoint (if one already exists) and trip the finalized guard.
    """
    counter = pickle.load(open(os.path.join(PKG, "data", f"cross_subpart_weight{seed}.pkl"), "rb"))
    pretrained = os.path.join(pretrained_dir, NAME_TMPL.format(seed=seed) + ".sdict")
    iface = NetworkInterface(ChordNet(counter), FT_NAME_TMPL.format(seed=seed),
                              load_checkpoint=False, load_path=cache_dir)
    if iface.finalized:
        new_name = FT_NAME_TMPL.format(seed=seed)
        raise RuntimeError(
            f"checkpoint {new_name} already exists in {cache_dir} — delete it or choose a new work dir"
        )
    iface.net.load_state_dict(torch.load(pretrained, map_location="cpu")["net"])
    return iface


def _send_to_gpu(var):
    """Verbatim copy of NetworkInterface.train_supervised's local send_to_gpu
    (lv_chordia/mir/nn/train.py:166-171): it's a closure nested inside the
    training loop, not an importable module-level symbol, so the recursive
    list/tuple-of-tensors walk is reproduced here rather than imported."""
    if isinstance(var, list):
        return [_send_to_gpu(sub_var) for sub_var in var]
    if isinstance(var, tuple):
        return tuple(_send_to_gpu(sub_var) for sub_var in var)
    return var.cuda()


def _dry_run_step(iface: NetworkInterface, train_provider, batch_size: int) -> None:
    """One forward/backward pass, no checkpoint write (train_supervised always
    saves on completion, so it's skipped here entirely)."""
    train_provider.init_worker(-1, True)
    loader = DataLoader(train_provider, batch_size=batch_size,
                         shuffle=train_provider.need_shuffle, num_workers=0,
                         collate_fn=train_provider.collate_fn)
    batch = next(iter(loader))
    if iface.net.use_gpu:  # mirrors train.py:172-173's gate before the loss call
        batch = _send_to_gpu(batch)
    iface.net.init_settings(True)
    iface.optimizer.zero_grad()
    raw_loss = iface.net.loss(*batch)
    loss = torch.sum(torch.stack(raw_loss)) if isinstance(raw_loss, tuple) else raw_loss
    if loss.grad_fn is not None:
        loss.backward()
        iface.optimizer.step()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B per-seed warm-start fine-tune")
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs-cap", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample-length", type=int, default=DEFAULT_SAMPLE_LENGTH)
    ap.add_argument("--pretrained-dir", default=CACHE_DATA_PATH,
                     help="dir holding the packaged pretrained *_s{seed}.best.sdict "
                          "checkpoints (default: the active venv's lv_chordia share "
                          "cache; the training bundle's checkpoints_in/ overrides this "
                          "so the run doesn't depend on what pip happened to install)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    work_dir = os.path.abspath(args.work_dir)
    cache_dir = os.path.join(work_dir, "cache_data")
    storage_dir = os.path.join(work_dir, "storage")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)

    train_pairs = _read_manifest(args.train_manifest)
    val_pairs = _read_manifest(args.val_manifest)
    train_x, train_y = build_storages(train_pairs, os.path.join(storage_dir, "train"))
    val_x, val_y = build_storages(val_pairs, os.path.join(storage_dir, "val"))

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_provider = make_providers(train_x, train_y, train=True,
                                         sample_length=args.sample_length)
        val_provider = make_providers(val_x, val_y, train=False)

        iface = _warm_start(seed, cache_dir, args.pretrained_dir)

        if args.dry_run:
            _dry_run_step(iface, train_provider, args.batch_size)
        else:
            iface.train_supervised(train_provider, val_provider, batch_size=args.batch_size,
                                    learning_rates_dict={args.lr: args.epochs_cap},
                                    round_per_val=-1, early_end_epochs=10, val_batch_size=1)
            print(f"SEED {seed} DONE")

    if args.dry_run:
        print("DRY-RUN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
