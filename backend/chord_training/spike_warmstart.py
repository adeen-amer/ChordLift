"""Phase B warm-start + data-contract spike (Mac, CPU only).

Proves end-to-end, against the packaged lv_chordia serving stack, that:
  1. a packaged pretrained checkpoint can be warm-started under a NEW save_name
     (fresh net + torch.load(...)['net'] -> load_state_dict; finalized stays False);
  2. train/val data can be built from plain (audio.wav, harte.lab) pairs into the
     package's FramedH5DataStorage / FramedDataProvider stack;
  3. train_supervised runs >=1 round, writes a new .sdict OUTSIDE the shared venv
     cache, and that checkpoint reloads and infers with the packaged 6-head contract.

Run:  cd backend && .venv/bin/python chord_training/spike_warmstart.py
Exits printing "SPIKE OK" only if every assertion passes. See FINDINGS.md for the
data contract distilled for the productionizer.

ponytail: single-file spike, no abstractions. dataset.py/finetune.py (Task 3)
generalize this; here everything is inline so the proof is auditable in one read.
"""
import hashlib
import os
import pickle
import shutil
import sys
import tempfile
import time

import librosa
import numpy as np
import soundfile as sf
import torch

import lv_chordia
from lv_chordia.chordnet_ismir_naive import (
    ChordNet, ComplexChordShifter, chord_limit,
    SPEC_DIM, SHIFT_LOW, SHIFT_HIGH, LSTM_TRAIN_LENGTH,
)
from lv_chordia.complex_chord import Chord
from lv_chordia.mir.common import CACHE_DATA_PATH
from lv_chordia.mir.nn.data_decorator import CQTPitchShifter
from lv_chordia.mir.nn.data_provider import FramedDataProvider
from lv_chordia.mir.nn.data_storage import FramedH5DataStorage
from lv_chordia.mir.nn.train import NetworkInterface

SR = 22050          # lv_chordia.settings.DEFAULT_SR
HOP = 512           # lv_chordia.settings.DEFAULT_HOP_LENGTH  == CQTV2 hop assert
CQT_BINS = 288      # CQTV2 n_bins (252 spec + 36 shift margin)
PKG = os.path.dirname(lv_chordia.__file__)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_out")
PRETRAINED = os.path.join(
    CACHE_DATA_PATH, "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best.sdict")
# New save_name: the s0 checkpoint's ft0 (fine-tune) sibling.
NEW_NAME = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft0_s0.best"

# C-F-G triad fundamentals (root, maj third, fifth) in Hz.
TRIAD_HZ = {
    "C:maj": [261.63, 329.63, 392.00],
    "F:maj": [349.23, 440.00, 523.25],
    "G:maj": [392.00, 493.88, 587.33],
    "N":     [],  # silence
}


# ---- synthetic (audio.wav, harte.lab) generation ---------------------------

def make_clip(schedule, seg_dur=2.0):
    """Return (mono float32 waveform, [(start,end,label), ...]) for a chord loop."""
    segs, chunks, t0 = [], [], 0.0
    for label in schedule:
        n = int(SR * seg_dur)
        tt = np.arange(n) / SR
        if TRIAD_HZ[label]:
            wave = sum(0.3 * np.sin(2 * np.pi * f * tt) for f in TRIAD_HZ[label])
        else:
            wave = np.zeros(n)  # N == silence
        chunks.append(wave.astype(np.float32))
        segs.append((t0, t0 + seg_dur, label))
        t0 += seg_dur
    return np.concatenate(chunks), segs


def write_lab(path, segs):
    """Harte .lab: tab-separated  start<TAB>end<TAB>label  (ChordLabIO format)."""
    with open(path, "w") as f:
        for s, e, lab in segs:
            f.write("%.3f\t%.3f\t%s\n" % (s, e, lab))


def read_lab(path):
    segs = []
    for line in open(path):
        line = line.strip()
        if line:
            s, e, lab = line.split("\t")
            segs.append((float(s), float(e), lab))
    return segs


# ---- (wav, lab) -> (CQT array, complex-chord label array) ------------------

def compute_cqt(wav_path):
    """Replicate CQTV2 + MusicIO exactly: load @ DEFAULT_SR mono, hybrid_cqt, |.|, f32."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    cqt = librosa.core.hybrid_cqt(
        y, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"),
        n_bins=CQT_BINS, tuning=None, hop_length=HOP).T
    return np.abs(cqt).astype(np.float32)


def encode_labels(segs, n_frames):
    """Frame-align a .lab to the 6-component complex-chord array via Chord().to_numpy().

    Frame f covers time f*HOP/SR. Frames past the last segment -> 'X' (ignored by loss).
    Returns int16 (n_frames, 6): [triad, bass, seventh, ninth, eleventh, thirteenth].
    """
    y = np.empty((n_frames, 6), dtype=np.int16)
    x_arr = Chord("X").to_numpy()  # frames with no label -> X -> masked out in loss
    cache = {}
    for f in range(n_frames):
        t = f * HOP / SR
        lab = next((l for s, e, l in segs if s <= t < e), None)
        if lab is None:
            y[f] = x_arr
        else:
            if lab not in cache:
                cache[lab] = Chord(lab).to_numpy()
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


def build_storage(abs_name, proxy_name, arrays, dtype):
    """Create a FramedH5DataStorage at an absolute path (bypasses DEFAULT_DATA_STORAGE_PATH).

    os.path.join('E:/dataset/', '/abs/name') -> '/abs/name', so an absolute `name`
    redirects the .h5d out of the package's Windows storage root and into OUT_DIR.
    """
    storage = FramedH5DataStorage(abs_name, dtype=dtype)
    if storage.created:            # deterministic re-runs
        storage.delete()
        storage = FramedH5DataStorage(abs_name, dtype=dtype)
    entries = [_Entry("song%d" % i, proxy_name, a) for i, a in enumerate(arrays)]
    storage.create_and_cache(entries, proxy_name)
    return storage


def hash_dir(d):
    return {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    assert not torch.cuda.is_available(), "spike is CPU-only by design"
    cache_before = hash_dir(CACHE_DATA_PATH)

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    # 1. synth 2 clips + labels in a tempdir; derive CQT (X) and label (Y) arrays.
    schedules = [["C:maj", "F:maj", "G:maj"] * 5,
                 ["G:maj", "C:maj", "F:maj"] * 4 + ["N"]]
    cqts, ys = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, sched in enumerate(schedules):
            wav_path = os.path.join(td, "clip%d.wav" % i)
            lab_path = os.path.join(td, "clip%d.lab" % i)
            y, segs = make_clip(sched)
            sf.write(wav_path, y, SR)
            write_lab(lab_path, segs)
            cqt = compute_cqt(wav_path)
            lab = encode_labels(read_lab(lab_path), cqt.shape[0])
            assert cqt.shape[1] == CQT_BINS, cqt.shape
            assert cqt.shape[0] >= LSTM_TRAIN_LENGTH, \
                "clip too short: %d < %d frames" % (cqt.shape[0], LSTM_TRAIN_LENGTH)
            assert lab.shape == (cqt.shape[0], 6), lab.shape
            cqts.append(cqt)
            ys.append(lab)
    print("clips: frames=%s  cqt_dim=%d" % ([c.shape[0] for c in cqts], CQT_BINS))

    # 2. build storages: X = CQT (float16, upcast to f32 by data_type_fix == serving),
    #    Y = complex-chord ids (int16, upcast to int64 for cross-entropy targets).
    storage_x = build_storage(os.path.join(OUT_DIR, "spike_cqt"), "cqt", cqts, np.float16)
    storage_y = build_storage(os.path.join(OUT_DIR, "spike_xchord"), "xchord", ys, np.int16)

    # 3. provider wiring — cribbed verbatim from chordnet_ismir_naive.__main__.
    train_idx = np.array([0, 1])
    val_idx = np.array([0])
    train_provider = FramedDataProvider(train_sample_length=LSTM_TRAIN_LENGTH,
                                        shift_low=SHIFT_LOW, shift_high=SHIFT_HIGH,
                                        num_workers=0, average_samples_per_song=1)
    train_provider.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=train_idx)
    train_provider.link(storage_y, ComplexChordShifter(), subrange=train_idx)

    val_provider = FramedDataProvider(train_sample_length=-1, shift_low=0, shift_high=0,
                                      num_workers=0, average_samples_per_song=1, need_shuffle=False)
    val_provider.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=val_idx)
    val_provider.link(storage_y, ComplexChordShifter(), subrange=val_idx)
    assert train_provider.valid_song_count == 2, train_provider.valid_song_count
    assert val_provider.valid_song_count == 1, val_provider.valid_song_count

    # 4. warm-start: fresh net under NEW_NAME (no .sdict yet -> not finalized),
    #    then hard-load the pretrained s0 weights.
    counter = pickle.load(open(os.path.join(PKG, "data", "cross_subpart_weight0.pkl"), "rb"))
    iface = NetworkInterface(ChordNet(counter), NEW_NAME, load_checkpoint=False, load_path=OUT_DIR)
    assert iface.finalized is False, "new save_name must not be finalized"
    iface.net.load_state_dict(torch.load(PRETRAINED, map_location="cpu")["net"])
    print("warm-start: loaded pretrained s0 weights into fresh %s" % NEW_NAME)

    # 5. train >=1 round; writes NEW_NAME.sdict into OUT_DIR (NOT the shared cache).
    t0 = time.time()
    iface.train_supervised(train_provider, val_provider, batch_size=2,
                           learning_rates_dict=1e-4, round_per_val=-1,
                           early_end_epochs=1, val_batch_size=1)
    train_secs = time.time() - t0
    sdict_path = os.path.join(OUT_DIR, "%s.sdict" % NEW_NAME)
    assert os.path.exists(sdict_path), "train_supervised did not write %s" % sdict_path
    print("train_supervised: %.1fs -> %s" % (train_secs, os.path.basename(sdict_path)))

    # 6. reload proof: fresh NetworkInterface finds NEW_NAME.sdict and infers.
    reloaded = NetworkInterface(ChordNet(None), NEW_NAME, load_checkpoint=False, load_path=OUT_DIR)
    assert reloaded.finalized is True, "reloaded checkpoint should be finalized"
    out = reloaded.inference(cqts[0])
    assert isinstance(out, tuple) and len(out) == 6, "expected 6-head output"
    dims = [h.shape[1] for h in out]
    frames = [h.shape[0] for h in out]

    # contract == the packaged serving model's contract, on the same input.
    packaged = NetworkInterface(
        ChordNet(None), "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best",
        load_checkpoint=False)  # default load_path='cache_data' -> READ-ONLY
    ref = packaged.inference(cqts[0])
    ref_dims = [h.shape[1] for h in ref]
    expected = [chord_limit.bass_slice_begin, 13, chord_limit.seventh_limit + 1,
                chord_limit.ninth_limit + 1, chord_limit.eleventh_limit + 1,
                chord_limit.thirteenth_limit + 1]
    assert dims == ref_dims == expected, (dims, ref_dims, expected)
    assert all(fr == frames[0] for fr in frames), frames
    print("inference: 6 heads dims=%s frames=%d (== packaged contract)" % (dims, frames[0]))

    # 7. shared cache must be byte-identical (no writes, no new files).
    assert hash_dir(CACHE_DATA_PATH) == cache_before, "SHARED cache_data was modified!"
    print("shared cache_data byte-identical: OK")

    print("SPIKE OK")


if __name__ == "__main__":
    sys.exit(main())
