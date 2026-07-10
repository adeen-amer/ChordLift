# Phase B data contract & warm-start recipe (proven by `spike_warmstart.py`)

This is the recipe Task 3 (`dataset.py` / `finetune.py`) implements. Everything below
**ran on this Mac, CPU only** and is asserted end-to-end by `spike_warmstart.py`
(exits `SPIKE OK`). You do **not** need to read the `lv_chordia` source to use it —
this file is the contract. All package paths are under
`backend/.venv/lib/python3.11/site-packages/lv_chordia/` (read-only; never edit).

## 0. What is proven

- A packaged serving checkpoint (`..._s0.best.sdict`) can be **warm-started** under a
  new `save_name` and fine-tuned without tripping the `finalized` guard.
- Train/val tensors build from plain `(audio.wav, harte.lab)` pairs into the package's
  `FramedH5DataStorage` + `FramedDataProvider` stack.
- `train_supervised` runs >=1 round, writes a new `.sdict` **outside** the shared venv
  cache, and that checkpoint reloads and infers with the exact packaged 6-head contract.
- The shared `cache_data/` is **byte-identical** before and after (asserted in-script by
  SHA-1 of every file).

## 1. Fixed constants (import these; do not hardcode)

| Constant | Value | Source |
|---|---|---|
| `SR` | 22050 | `lv_chordia.settings.DEFAULT_SR` |
| `HOP` | 512 | `lv_chordia.settings.DEFAULT_HOP_LENGTH` (CQTV2 asserts `hop_length==512`) |
| frame rate | 22050/512 ~= **43.07 fps** | derived |
| `SPEC_DIM` | 252 | `chordnet_ismir_naive` |
| CQT storage width | **288** | `CQTV2 n_bins` = 252 + 36-bin shift margin |
| `SHIFT_LOW, SHIFT_HIGH` | -5, 6 | `chordnet_ismir_naive` |
| `SHIFT_STEP` | 3 | 3 CQT bins per semitone (bins_per_octave=36) |
| `LSTM_TRAIN_LENGTH` | 1000 | `chordnet_ismir_naive` |

`min_input_dim` the pitch-shifter demands = `(-SHIFT_LOW+SHIFT_HIGH)*SHIFT_STEP + SPEC_DIM`
= `11*3 + 252 = 285`. The 288-bin storage satisfies it.

**Quirk (audio length floor):** the train provider drops any song with fewer than
`LSTM_TRAIN_LENGTH=1000` frames (`FramedDataProvider.link`, data_provider.py:72/78).
1000 frames = **23.2 s** at 22050/512. Clips shorter than ~24 s are silently discarded
from the *train* set (`valid_song_count` drops). The val provider uses
`train_sample_length=-1`, which disables the filter, so short clips still validate.
The spike uses 30 s and 26 s clips.

## 2. `(wav, lab)` -> CQT array (X) — replicates CQTV2 + MusicIO exactly

Serving (`chord_recognition.py`) feeds `entry.cqt` **straight** to `net.inference` — a
raw float32 hybrid-CQT magnitude, **no log, no normalization, no int quantization.**
Replicate it without the datasets/io framework:

```python
y, _ = librosa.load(wav_path, sr=22050, mono=True)          # == MusicIO.read
cqt = librosa.core.hybrid_cqt(
        y, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"),
        n_bins=288, tuning=None, hop_length=512).T
cqt = np.abs(cqt).astype(np.float32)                        # shape (frames, 288)
```

**Quirk (dtype — the big one):** `storage_creation.py` builds *both* storages with
`dtype=np.int16`. That is wrong/legacy for CQT — int16 would floor the magnitudes to
0/1 and the conv stack would receive a `LongTensor` (via `data_type_fix`, which maps
`int16 -> int64`) and crash. **Store CQT as `np.float16`.** `data_type_fix`
(`mir/nn/data_decorator.py:4`) upcasts `float16 -> float32`, which is exactly what the
serving path feeds. (`float32` storage works too; `float16` halves disk and matches the
`data_type_fix` float16 branch, implying the original storages were float16.)

## 3. `(wav, lab)` -> complex-chord label array (Y)

Harte `.lab` is tab-separated `start<TAB>end<TAB>label` (`io_new/chordlab_io.py`). Encode
each frame's label to the 6-component array with `complex_chord.Chord(name).to_numpy()`:

```python
Chord("C:maj").to_numpy()  # [ 1, 0, 0, 0, 0, 0]  int8
Chord("F:maj").to_numpy()  # [ 6, 5, 0, 0, 0, 0]
Chord("G:maj").to_numpy()  # [ 8, 7, 0, 0, 0, 0]
Chord("N").to_numpy()      # [ 0,-1,-2,-2,-2,-2]
Chord("X").to_numpy()      # [-2,-2,-2,-2,-2,-2]
```

Components are `[triad, bass, seventh, ninth, eleventh, thirteenth]`. Frame `f` covers
time `f*HOP/SR`; look up the covering segment; frames past the last segment -> encode as
`X`. **Store Y as `np.int16`** (`data_type_fix` maps `int16 -> int64`, the dtype
`cross_entropy` targets need). The row count of Y **must equal** the CQT frame count of
the same song (the provider asserts X and Y `start`/`length` arrays match — section 5).

**N / X label behavior in the loss** (`ReweightedLoss.forward`,
`chordnet_ismir_naive.py:46`): each head's loss masks out negative targets
(`a[b>=0]`), and returns 0 if all targets in a head are negative.
- `X` -> all six components `< 0` -> **fully ignored** (no gradient). Use it as the
  "don't care / unknown" fill; but do not build a training batch that is *only* X — the
  loss then has no `grad_fn` and the optimizer step is skipped (train.py prints
  "Warning: the loss of the batch does not have a grad_fn. Ignored.").
- `N` (no-chord/silence) -> triad component `0` is a **valid class** (trained); bass is
  `-1` but the bass head adds `+1` (`tag[:,1]+1`, chordnet_ismir_naive.py:57) so
  `-1 -> 0` (valid); the four extension heads are `-2` (ignored). So `N` **is** a real
  supervised target.
- `Chord(name)` raises on labels outside the Harte subset it parses. Accepted: root +
  `BASIC_TYPES` (`.`, maj, min, sus4, sus2, dim, aug, 5, 1), `EXTENDED_TYPES` (maj6,
  min6, 7, maj7, min7, minmaj7, dim7, hdim7, 9, maj9, min9, 11, min11, 13, maj13,
  min13), optional `(add/omit)` brackets and `/bass`, plus bare `N` and `X`
  (`complex_chord.py`, SuffixDecoder). Anything else (e.g. `C:weird`) raises
  `Unknown chord type`. Wrap real-data ingestion in try/except and map unparseable
  labels to `X` (ignored) rather than crashing the storage build.
- The chop into the model's `chord_limit` `[6,3,3,2,2]` and the pitch-shift both happen
  automatically at sample time via `ComplexChordShifter` — **store the raw
  `to_numpy()` arrays**, do not pre-chop.

## 4. Build the two H5 storages (absolute-path redirect + duck-typed entry)

`FramedH5DataStorage(name)` writes to `os.path.join(DEFAULT_DATA_STORAGE_PATH, name+'.h5d')`
and `DEFAULT_DATA_STORAGE_PATH` is the package default `"E:/dataset/"` (Windows).
**Pass an absolute path as `name`** — `os.path.join("E:/dataset/", "/abs/x")` returns
`/abs/x`, so the storage lands wherever you want (e.g. `spike_out/`), never in `E:/`.

`create_and_cache(entries, proxy_name)` pulls data via `entry.dict[proxy_name].get(entry)`
and reads `entry.n_frame` / `entry.name` / `entry.free()`. You do **not** need the
`datasets` / `io` / extractor machinery — a tiny duck-type is enough (ran verbatim):

```python
class _Proxy:
    def __init__(self, arr): self.arr = arr
    def get(self, entry):    return self.arr

class _Entry:
    def __init__(self, name, proxy_name, arr):
        self.name, self.n_frame = name, arr.shape[0]
        self.dict = {proxy_name: _Proxy(arr)}
    def free(self): pass

storage_x = FramedH5DataStorage("/abs/out/spike_cqt",    dtype=np.float16)
storage_y = FramedH5DataStorage("/abs/out/spike_xchord", dtype=np.int16)
storage_x.create_and_cache([_Entry("song0", "cqt", cqt0), ...],  "cqt")
storage_y.create_and_cache([_Entry("song0", "xchord", y0), ...], "xchord")
```

`create_and_cache` writes `<name>.h5d` — a single HDF5 file holding datasets `feature`
(all songs concatenated), `start`, `length` — sets `created=True`, and calls
`load_meta()`. No separate meta files. To rebuild deterministically, call
`storage.delete()` first if `storage.created` (the constructor auto-detects an existing
file and would otherwise skip the rebuild). `proxy_name` is arbitrary but must match
between the entries and the `create_and_cache` call.

## 5. Provider wiring (verbatim from `chordnet_ismir_naive.__main__`)

```python
train = FramedDataProvider(train_sample_length=LSTM_TRAIN_LENGTH,
                           shift_low=SHIFT_LOW, shift_high=SHIFT_HIGH,
                           num_workers=0, average_samples_per_song=1)
train.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=train_idx)
train.link(storage_y, ComplexChordShifter(),                            subrange=train_idx)

val = FramedDataProvider(train_sample_length=-1, shift_low=0, shift_high=0,
                         num_workers=0, average_samples_per_song=1, need_shuffle=False)
val.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=val_idx)
val.link(storage_y, ComplexChordShifter(),                           subrange=val_idx)
```

- `ComplexChordShifter` (defined in `chordnet_ismir_naive.py`) does chop-to-`chord_limit`
  **and** pitch-shift at sample time; `CQTPitchShifter` slices the 288-bin CQT to the
  252-bin window for the requested shift.
- X **must** be linked before Y (first link sets the reference `start`/`length`; the
  second asserts equality — hence identical per-song frame counts, section 3).
- `subrange` is a numpy int array of song indices. Split train/val by index.
- `need_shuffle=False` on the val provider keeps validation deterministic;
  `train_sample_length=-1` makes it return whole songs (so `val_batch_size=1`).
- **Quirk (`num_workers`):** the package `__main__` uses `num_workers=1`, but its own
  `train.py` flags multiprocess data loading as an unfinished TODO. Use
  **`num_workers=0`** — `train_supervised` then calls `init_worker(-1, ...)` in-process
  (train.py:123/131), which `storage.load()`s everything in the main process.
  Deterministic, no macOS spawn/pickling issues.

## 6. Warm-start recipe

```python
counter = pickle.load(open(f"{PKG}/data/cross_subpart_weight0.pkl", "rb"))  # list, len 6
NEW_NAME = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft0_s0.best"
iface = NetworkInterface(ChordNet(counter), NEW_NAME, load_checkpoint=False,
                         load_path=OUT_DIR)          # OUT_DIR = abs work dir (section 7)
assert iface.finalized is False                      # new name => no .sdict on disk yet
iface.net.load_state_dict(
    torch.load(PRETRAINED_S0_SDICT, map_location="cpu")["net"])
```

- **Why not load via the constructor:** if `<save_name>.sdict` exists, the constructor
  loads it *and sets `finalized=True`* (train.py:86), and `train_supervised` refuses a
  finalized model (train.py:107-109). So you **construct fresh under a new name**
  (nothing on disk -> `finalized=False`) and hard-load the pretrained weights manually.
- **Key match is exact (verified):** a fresh `ChordNet(counter)` state_dict has the same
  keys as the packaged `..._s0.best.sdict['net']` — zero missing, zero extra
  (`ReweightedLoss.weight` is a plain Python list, not registered buffers, so the loss
  contributes no state_dict keys). Strict `load_state_dict` succeeds.
- The packaged `.sdict` top-level keys: `['loss','learning_rate','step','batch_count',
  'net','opt','counter']` — take `['net']` only; do **not** restore `['opt']`/`['counter']`
  (fresh Adam + counter=0 is what you want for fine-tuning, and restoring `counter`
  would make `train_supervised` skip that many batches as "already done").
- `ChordNet(counter)` needs a counter for its loss; `ChordNet(None)` (no loss module) is
  fine for inference-only reload (section 8). Pick the `cross_subpart_weight{slice}.pkl`
  matching the checkpoint slice you warm-start (s0 -> weight0, ... s4 -> weight4);
  they live in the package: `lv_chordia/data/`.

## 7. Checkpoint save-dir redirection (keep the shared cache pristine)

`NetworkInterface.__init__(net, save_name, load_checkpoint, load_path='cache_data')`
resolves (train.py:62):

```python
base_path = CACHE_DATA_PATH if load_path == 'cache_data' \
            else os.path.join(WORKING_PATH, load_path)
```

`train_supervised` writes `<base_path>/<save_name>.sdict` (+ `.cp.sdict`, `.best.sdict`).
Default `load_path='cache_data'` -> **the shared venv
`.venv/share/lv-chordia/cache_data/` — never use it for saving.** `WORKING_PATH` is
`site-packages/` (also inside `.venv`, off-limits). The clean redirect: **pass an
absolute path as `load_path`** — `os.path.join(WORKING_PATH, "/abs")` returns `/abs`,
and `"/abs" != 'cache_data'` so `CACHE_DATA_PATH` is bypassed:

```python
NetworkInterface(net, NEW_NAME, load_checkpoint=False, load_path="/abs/spike_out")
```

Reading a packaged model for reference (section 8) with the default `load_path` is safe —
the constructor and `inference` never write; only `train_supervised` does. The spike
asserts `cache_data/` SHA-1s are unchanged after the whole run.

## 8. Reload + inference + the 6-head contract

```python
reloaded = NetworkInterface(ChordNet(None), NEW_NAME, load_checkpoint=False,
                            load_path=OUT_DIR)
assert reloaded.finalized is True                 # <NEW_NAME>.sdict now exists
triad, bass, s7, s9, s11, s13 = reloaded.inference(cqt)  # cqt = full 288-wide float32
```

`inference` returns a **6-tuple** of per-frame softmax arrays. Head widths (derived from
`chord_limit`; asserted identical to the packaged serving model on the same input):

| head | width | meaning |
|---|---|---|
| triad | **73** | `chord_limit.bass_slice_begin` = 6*12+1 (N + 6 triad types x 12 roots) |
| bass  | **13** | 12 pitch classes + no-bass |
| 7th   | **4**  | `seventh_limit+1` |
| 9th   | **4**  | `ninth_limit+1` |
| 11th  | **3**  | `eleventh_limit+1` |
| 13th  | **3**  | `thirteenth_limit+1` |

Pass the **full 288-bin** CQT to `inference`; `ChordNet.inference` internally slices the
center 252-bin window (`x[:, SHIFT_HIGH*SHIFT_STEP : +SPEC_DIM]` = `x[:, 18:270]`).

## 9. `train_supervised` call & save behavior

```python
iface.train_supervised(train, val, batch_size=2, learning_rates_dict=1e-4,
                       round_per_val=-1, early_end_epochs=1, val_batch_size=1)
```

- `learning_rates_dict` may be a scalar (single LR, 1 epoch) or `{lr: n_epochs}` dict
  (the package uses e.g. `{1e-3:60, 1e-4:30, 1e-5:30, 1e-6:10}`).
- `round_per_val=-1` selects the **per-epoch validation** branch (train.py:240): writes
  `<name>.best.sdict` on val-loss improvement, `<name>.cp.sdict` every epoch, and the
  final `<name>.sdict` when the LR schedule ends. Since `save_name` already ends in
  `.best` (matching the serving naming), files on disk are `<name>.sdict`,
  `<name>.cp.sdict`, `<name>.best.sdict`. Load `<name>.sdict` (final) or
  `<name>.best.sdict` (best-val).
- `batch_count = ceil(len(train)/batch_size)`; `len(train) = valid_song_count *
  average_samples_per_song * (SHIFT_HIGH-SHIFT_LOW+1)`. Spike: `2*1*12 = 24 -> 12`
  batches. `early_end_epochs` is patience in epochs per LR stage.

## 10. Timing (this Mac, CPU)

Full spike wall time **~10 s**; `train_supervised` alone **5.3 s** for 1 epoch / 12
batches (2 songs x 12 pitch shifts, seq len 1000, batch 2) + one full-clip validation.
CQT extraction ~1 s per 30 s clip. Real Phase B runs on the 2060 Super; budget CPU only
for smoke tests.

## Quirks checklist (each hit, each fixed)

1. **CQT dtype** — store `float16`, not the `int16` from `storage_creation.py` (section 2).
2. **Y dtype** — `int16` -> `data_type_fix` -> `int64` targets (section 3).
3. **Storage path** — abs `name` dodges the `E:/dataset/` Windows default (section 4).
4. **`storage.created` short-circuit** — `delete()` before rebuild for determinism (section 4).
5. **`finalized` guard** — construct fresh under a new name, hard-load `['net']` only (section 6).
6. **Save dir** — abs `load_path` keeps writes out of the read-only venv cache (section 7).
7. **`num_workers=0`** — avoids the package's unfinished multiprocess loader (section 5).
8. **Audio >= 24 s** — else the train provider silently filters the song out (section 1).
9. **N vs X** — `N` is supervised (triad class 0, bass 0 after +1); `X`/negatives are
   masked; never train on all-`X` batches (no `grad_fn`) (section 3).
10. **`inference` input** — pass the full 288-bin CQT; the net slices to 252 (section 8).
