# Beat-Transformer contract spike — FINDINGS

**Outcome: SPIKE OK.** The full pipeline runs on CPU with the pretrained
checkpoint. Task 4 implements the `"transformer"` beat engine from the verified
recipe below and does **not** need to re-read the upstream repo.

Upstream: <https://github.com/zhaojw1998/Beat-Transformer> (MIT license),
pinned at commit **`063667fc9e4e11507f9d76dc1154d9db953a85eb`** (default branch
`main`, 2024-04-07). All facts below are read from source at that commit, not
from the README prose.

Verified with `torch 2.5.1+cpu`, `librosa` from the main-repo venv, by running
`beat_transformer/spike_infer.py` -> prints `SPIKE OK`.

---

## 1. Checkpoint acquisition (Step 2) — TRIVIALLY NON-INTERACTIVE

The pretrained weights are **committed directly in the git repo** under
`checkpoint/` (8 folds of 8-fold cross-validation), each a plain (non-LFS) blob:

```
checkpoint/fold_0_trf_param.pt  ...  fold_7_trf_param.pt   (37,235,863 bytes each)
```

No Google Drive, no Colab, no auth. `raw.githubusercontent.com` serves them
directly (37 MB < the 100 MB raw cap). We use **fold 0**:

```bash
curl -sSL -o backend/models/beat_transformer.pth \
  https://raw.githubusercontent.com/zhaojw1998/Beat-Transformer/main/checkpoint/fold_0_trf_param.pt
```

| field | value |
|-------|-------|
| saved path | `backend/models/beat_transformer.pth` (gitignored via `*.pth`) |
| size | **37,235,863 bytes** |
| SHA-256 | **`1940a5034bb2bb7860c1a3219d359906913be88c915f6135a5ce8382c239d738`** |

The Drive links in the README are only the **33 GB training dataset**
(`demix_spectrogram_data.npz`) and the Colab **demo UI** — neither is needed to
run inference. The README's "checkpoint folder" is literally in the repo tree.

`torch.load(path, map_location="cpu")` returns a dict with a single top-level
key `"state_dict"` (181 tensors). Load with `model.load_state_dict(ckpt["state_dict"])`
— strict load succeeds, which itself confirms every architecture dim below.

> Task 4 note: any of the 8 folds works; fold 0 is arbitrary. Upstream's Colab
> uses fold 0 too.

---

## 2. Model construction (`code/DilatedTransformer.py`, `DilatedTransformerLayer.py`)

Class `Demixed_DilatedTransformerModel`. The checkpoint was trained with the
constants in `code/eight_fold_test.py` (NOT the class-signature defaults, which
differ — `dmodel=128, nhead=2` in the signature vs. the trained `256/8`):

```python
from beat_transformer import Demixed_DilatedTransformerModel
model = Demixed_DilatedTransformerModel(
    attn_len=5, instr=5, ntoken=2,
    dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True,
)
```

Confirmed against checkpoint tensor shapes: `conv1.weight (32,1,5,3)`,
`conv3.weight (256,64,3,6)`, `out_linear.weight (2,256)`,
`out_linear_t.weight (300,256)`.

- **`nhead=8` is mandatory** — the dilated attention hardcodes an 8-head split
  (heads `0:4` symmetric + heads `4,5,6,7` skewed by shift +/-1 / +/-2). It will
  not run with any other head count.
- Vendored verbatim into this package as `DilatedTransformer.py` /
  `DilatedTransformerLayer.py`; the only edit is making the intra-package import
  relative (`from .DilatedTransformerLayer import ...`).
- Upstream keeps a **deliberate training-time bug** in `forward()` (layer
  key-roll uses `k[:, 6:7]` where `k[:, 7:8]` was intended); it is preserved on
  purpose "to fit the checkpoints." Do not fix it — it must match the weights.

### forward() I/O contract

```python
pred, tempo = model(x)          # eval(), no_grad
```

- **Input `x`: `(batch, instr=5, time=T, melbin=128)` float32.** In `forward`,
  `batch, instr, time, melbin = x.shape`; the conv stack pools **only the mel
  axis** (`MaxPool2d(kernel=(1,3))` x3), so the **time axis is preserved end to
  end: output T == input T**.
- **Output `pred`: `(batch, T, ntoken=2)`** — raw logits.
  `pred[..., 0]` = beat, `pred[..., 1]` = downbeat. Apply `torch.sigmoid` to get
  activations in [0,1].
- **Output `tempo`: `(batch, 300)`** — a 300-bin tempo head. Unused for
  beat/downbeat tracking; ignore it.

Verified live: input `(1,5,259,128)` -> `pred (1,259,2)`, `tempo (1,300)`.

---

## 3. Spectrogram preprocessing — the ground-truth contract

Source of truth = `preprocessing/demixing.py` (how the stored spectrograms were
built) + `code/spectrogram_dataset.py::__getitem__` (the final per-stem scaling
applied before the model). **This is the exact recipe; reproduce it per stem.**

Constants (all from `demixing.py` / `eight_fold_test.py`):

| param | value | source |
|-------|-------|--------|
| sample rate | **44100** | `demixing.py self.SR = 44100` |
| STFT n_fft (frame) | **4096** | Spleeter default + `mel(n_fft=4096)` |
| STFT hop | **1024** | implied by `FPS = 44100/1024` in `eight_fold_test.py` |
| mel filterbank | `librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000)` | `demixing.py` |
| n_mels (== model melbin) | **128** | same |
| per-stem scaling | `librosa.power_to_db(mel_power, ref=np.max)` | `spectrogram_dataset.__getitem__` |
| **frame rate FPS** | **44100/1024 ~= 43.066 Hz** | `eight_fold_test.py FPS` |

`demixing.py` computes, per Spleeter output stem: `magnitude = |STFT|`,
`mel_spec = magnitude**2 . mel_f` (a **power** mel), stores `(T, 5, 128)` per
song. Then `__getitem__` applies `power_to_db(..., ref=np.max)` **independently
per stem** and hands `(5, T, 128)` (batched -> `(B,5,T,128)`) to the model.

### Verified per-stem construction (runs in `spike_infer.py`)

```python
import librosa, numpy as np
MEL_FB = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000)  # (128, 2049)

def stem_logmel(stem_mono_44100):          # -> (T, 128) float32
    stft = librosa.stft(stem_mono_44100, n_fft=4096, hop_length=1024)  # (2049, T)
    power = np.abs(stft) ** 2                                          # (2049, T)
    mel_power = MEL_FB @ power                                        # (128, T)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)              # (128, T)
    return mel_db.T.astype(np.float32)                              # (T, 128)
```

Stack 5 stems -> `(5, T, 128)` -> `torch.from_numpy(...).unsqueeze(0)` ->
`(1, 5, T, 128)`. **Stems must be at 44100 Hz** — resample the StemBundle (which
is at the pipeline sample rate) to 44100 before STFT, or run `separate_stems` on
44100 audio (the spike does the latter).

> Note: upstream builds the STFT/mel through Spleeter's TF graph, not librosa.
> The librosa reimplementation above matches the documented math (n_fft/hop/mel
> filter/power/power_to_db) and is what Task 4 uses — we do not depend on
> Spleeter/TensorFlow. Numeric parity with Spleeter's STFT is close but not
> bit-exact; acceptable for a mel-power model and confirmed to produce a valid
> forward pass. librosa `stft` defaults (Hann window, center=True) align with
> Spleeter's; tighten only if Task 4 A/B shows it matters.

### Stem mapping (Spleeter 5-stem -> Demucs/HPSS StemBundle)

The model was trained on **`spleeter:5stems`**, channel order
**`[vocals, drums, bass, piano, other]`**. `backend/stem_separation.py`'s
`StemBundle` (Demucs `htdemucs` or HPSS fallback) provides
`{vocals, drums, bass, other}` and **no piano stem**. Mapping used:

| ch | Spleeter stem | StemBundle field |
|----|---------------|------------------|
| 0 | vocals | `stems.vocals` |
| 1 | drums  | `stems.drums`  |
| 2 | bass   | `stems.bass`   |
| 3 | piano  | `stems.other`  *(proxy — no Demucs piano stem)* |
| 4 | other  | `stems.other`  |

This is an **explicit spike assumption**, documented, not silent. It does not
affect the tensor shape/dtype contract (verified). It *may* affect accuracy: the
model's per-instrument attention (`instr_attention` layers 3-5) then `x.mean(1)`
averages across the 5 channels, so it is fairly robust to stem identity, but
Task 4 should A/B this proxy against alternatives (piano <- `other`, piano <-
`full` mix, or duplicating `other`) on real audio before trusting downbeat
accuracy. `StemBundle.full` (the mix) is a plausible piano proxy too.

---

## 4. Activations -> (beat_times, downbeat_times)

**Canonical post-processing (`code/eight_fold_test.py`) is madmom DBN**, at
`FPS = 44100/1024`:

```python
import madmom
FPS = 44100 / 1024
beat_act = torch.sigmoid(pred[0, :, 0]).cpu().numpy()   # (T,)
down_act = torch.sigmoid(pred[0, :, 1]).cpu().numpy()   # (T,)

# beats
beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
    min_bpm=55.0, max_bpm=215.0, fps=FPS,
    transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2)
beat_times = beat_tracker(beat_act)                     # seconds

# downbeats: feed a 2-col activation [max(beat-down,0), down]
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS,
    transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2)
combined = np.stack([np.maximum(beat_act - down_act, 0.0), down_act], axis=-1)  # (T,2)
db = downbeat_tracker(combined)                         # rows: [time, beat_pos]
downbeat_times = db[db[:, 1] == 1][:, 0]                # positions == 1 are downbeats
```

Frame `i` <-> time `i / FPS` seconds.

**Env caveat for the spike:** madmom is **not installed** in the shared venv
(a known, already-documented Task 1 limitation). To still prove the
activation->time conversion runs, `spike_infer.py` uses a trivial
threshold+local-max picker (`times = peak_frames / FPS`) as a stand-in. That
stand-in is **not** the shipping post-processor — **Task 4 must install madmom
and use the DBN block above** (it maps cleanly onto the existing
`_track_beats_madmom` DBN usage in `beat_tracking.py`). The DBN's
`beats_per_bar=[3,4]` also means the transformer engine infers 3/4 vs 4/4 rather
than assuming 4/4 like the librosa heuristic.

---

## 5. What Task 4 wires up

1. Add `models/beat_transformer.pth` acquisition (curl from the pinned raw URL,
   verify SHA-256 above) — or lazy-download on first use into `models/`.
2. New `_track_beats_transformer(stems, sr)` in `beat_tracking.py`:
   resample each of the 5 mapped stems to 44100 -> `stem_logmel` -> `(1,5,T,128)`
   -> `model.eval()` no-grad forward -> sigmoid -> madmom DBN (section 4) ->
   `BeatGrid`. Cache the constructed model (like `model_cache.get_demucs_model`).
3. Extend `track_beats_auto` / `CHORD_BEAT_ENGINE` with a `"transformer"` value
   and try it first in `"auto"` when real Demucs stems are available.
4. Add `madmom` to the ML requirements (already needed by Task 1's madmom engine).
5. Decide the piano-proxy mapping (section 3) empirically on real audio.

Vendored model files (`DilatedTransformer.py`, `DilatedTransformerLayer.py`) are
already in this package and import-clean.
