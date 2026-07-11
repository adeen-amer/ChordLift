"""backend/chord_training/dataset.py

Storage/provider builder for Phase B fine-tuning. Productionizes the recipe
proven end-to-end by `spike_warmstart.py` — see FINDINGS.md for the full
data contract (dtypes, path redirection, provider wiring quirks). Crib from
the spike freely; this module just wraps it into reusable functions over
arbitrary (audio, lab) pairs instead of two hardcoded synthetic clips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

from lv_chordia.chordnet_ismir_naive import (
    SPEC_DIM, SHIFT_LOW, SHIFT_HIGH, LSTM_TRAIN_LENGTH,
    ComplexChordShifter,
)
from lv_chordia.complex_chord import Chord
from lv_chordia.mir.nn.data_decorator import CQTPitchShifter
from lv_chordia.mir.nn.data_provider import FramedDataProvider
from lv_chordia.mir.nn.data_storage import FramedH5DataStorage
from lv_chordia.settings import DEFAULT_SR as SR, DEFAULT_HOP_LENGTH as HOP

CQT_BINS = 288  # CQTV2 n_bins: 252-bin SPEC_DIM + 36-bin pitch-shift margin (FINDINGS.md §1)


# ---- (wav, lab) -> (CQT array, complex-chord label array) ------------------

def compute_cqt(wav_path: str) -> np.ndarray:
    """Replicate CQTV2 + MusicIO exactly: load @ DEFAULT_SR mono, hybrid_cqt, |.|, f32."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    cqt = librosa.core.hybrid_cqt(
        y, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"),
        n_bins=CQT_BINS, tuning=None, hop_length=HOP).T
    return np.abs(cqt).astype(np.float32)


def read_lab(lab_path: str) -> list[tuple[float, float, str]]:
    """Harte .lab: start end label, whitespace-separated (tab or space).

    Real Isophonics .lab files are space-separated; our Billboard converter
    writes tabs. maxsplit=2 is safe because Harte chord labels never contain
    whitespace.
    """
    segs = []
    with open(lab_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, e, lab = line.split(None, 2)
            segs.append((float(s), float(e), lab))
    return segs


def encode_labels(segs, n_frames: int) -> np.ndarray:
    """Frame-align a .lab to the 6-component complex-chord array via Chord().to_numpy().

    Frame f covers time f*HOP/SR. Frames past the last segment, or whose label
    isn't in the Harte subset `Chord` parses, -> 'X' (ignored by the loss;
    FINDINGS.md §3). Returns int16 (n_frames, 6).
    """
    y = np.empty((n_frames, 6), dtype=np.int16)
    x_arr = Chord("X").to_numpy()
    cache = {}
    for f in range(n_frames):
        t = f * HOP / SR
        lab = next((l for s, e, l in segs if s <= t < e), None)
        if lab is None:
            y[f] = x_arr
        else:
            if lab not in cache:
                try:
                    cache[lab] = Chord(lab).to_numpy()
                except Exception:
                    cache[lab] = x_arr  # unparseable label -> ignored, not fatal
            y[f] = cache[lab]
    return y


# ---- minimal shim to feed create_and_cache without the datasets/io stack ---

class _Proxy:
    def __init__(self, arr):
        self.arr = arr

    def get(self, entry):
        return self.arr


class _Entry:
    """Duck-types the fields FramedH5DataStorage.create_h5 touches."""

    def __init__(self, name, proxy_name, arr):
        self.name = name
        self.n_frame = arr.shape[0]
        self.dict = {proxy_name: _Proxy(arr)}

    def free(self):
        pass


def _build_storage(abs_name, proxy_name, arrays, dtype):
    storage = FramedH5DataStorage(abs_name, dtype=dtype)
    if storage.created:  # deterministic re-runs: clear pre-existing storage files
        # ponytail: storage.delete() calls self.root.close() unconditionally, but
        # root is only ever set by load()/create_and_cache(), never by __init__ -
        # crashes on this freshly-constructed object. Unlink the files it writes
        # (final .h5d + its .h5part rename staging temp) directly instead.
        Path(storage.filename).unlink(missing_ok=True)
        Path(storage.filename + ".h5part").unlink(missing_ok=True)
        storage = FramedH5DataStorage(abs_name, dtype=dtype)
    entries = [_Entry("song%d" % i, proxy_name, a) for i, a in enumerate(arrays)]
    storage.create_and_cache(entries, proxy_name)
    return storage


def build_storages(pairs: list[tuple[str, str]], out_dir: str) -> tuple[str, str]:
    """(audio, lab) pairs -> the FramedH5DataStorage (X, Y) abs paths under out_dir.

    Any pair whose audio/lab can't be turned into arrays (unreadable audio,
    malformed .lab) is skipped and reported to stderr rather than raising —
    the rest of the manifest still builds.
    """
    cqts, ys = [], []
    for audio_path, lab_path in pairs:
        try:
            cqt = compute_cqt(audio_path)
            lab = encode_labels(read_lab(lab_path), cqt.shape[0])
        except Exception as exc:
            print(f"skipping pair (encoding failed): {audio_path} / {lab_path}: {exc}",
                  file=sys.stderr)
            continue
        cqts.append(cqt)
        ys.append(lab)

    if not cqts:
        raise ValueError(f"no encodable (audio, lab) pairs among {len(pairs)} given")

    storage_x_path = f"{out_dir}/cqt"
    storage_y_path = f"{out_dir}/xchord"
    _build_storage(storage_x_path, "cqt", cqts, np.float16)
    _build_storage(storage_y_path, "xchord", ys, np.int16)
    return storage_x_path, storage_y_path


# ---- provider wiring — verbatim from chordnet_ismir_naive.__main__ ---------

def make_providers(storage_x_path: str, storage_y_path: str, train: bool,
                    sample_length: int = LSTM_TRAIN_LENGTH) -> FramedDataProvider:
    """Reopen the built storages and wire a train or val FramedDataProvider.

    train=True: shift augmentation (SHIFT_LOW..SHIFT_HIGH) + shuffle, cropped to
    `sample_length` frames per sample (songs shorter than this are dropped —
    FINDINGS.md §1's `LSTM_TRAIN_LENGTH` floor; `sample_length` is the knob
    finetune.py's `--sample-length` flag drives).
    train=False: whole-song, no shift, no shuffle (val_batch_size=1 at call site).
    """
    storage_x = FramedH5DataStorage(storage_x_path)
    storage_y = FramedH5DataStorage(storage_y_path)
    idx = np.arange(storage_x.get_length())

    if train:
        provider = FramedDataProvider(train_sample_length=sample_length,
                                       shift_low=SHIFT_LOW, shift_high=SHIFT_HIGH,
                                       num_workers=0, average_samples_per_song=1)
    else:
        provider = FramedDataProvider(train_sample_length=-1, shift_low=0, shift_high=0,
                                       num_workers=0, average_samples_per_song=1,
                                       need_shuffle=False)
    provider.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=idx)
    provider.link(storage_y, ComplexChordShifter(), subrange=idx)
    return provider
