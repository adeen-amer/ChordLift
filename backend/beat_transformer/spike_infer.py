"""Task 3 contract spike: Beat-Transformer end-to-end on CPU with a synthetic clip.

Proves the verified contract in FINDINGS.md actually runs:
  StemBundle -> (1, 5, T, 128) log-power-mel -> checkpoint forward -> (1, T, 2)
  -> (beat_times, downbeat_times).

Run:  cd backend && <venv>/python.exe beat_transformer/spike_infer.py
Expect the final line to be:  SPIKE OK
"""
from __future__ import annotations

import os
import sys

import numpy as np
import librosa
import torch

# Make backend/ importable (stem_separation) and beat_transformer/ a package,
# whether run as a script or a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from stem_separation import StemBundle, separate_stems  # noqa: E402
from beat_transformer import Demixed_DilatedTransformerModel  # noqa: E402

# --- Contract constants (verified from upstream @063667f) ------------------
SR = 44100          # Spleeter/model sample rate (preprocessing/demixing.py)
N_FFT = 4096         # Spleeter STFT frame length
HOP = 1024           # Spleeter STFT hop -> FPS = SR/HOP
N_MELS = 128         # mel bins == model input "melbin" dim
FMIN, FMAX = 30, 11000
FPS = SR / HOP       # 43.066 Hz, frame rate of both input and output
CKPT = os.path.join(_BACKEND, "models", "beat_transformer.pth")

# Model construction args (code/eight_fold_test.py constants).
MODEL_KW = dict(attn_len=5, instr=5, ntoken=2, dmodel=256,
                nhead=8, d_hid=1024, nlayers=9, norm_first=True)

# Reference Spleeter 5-stem channel order the checkpoint was trained on.
# StemBundle (Demucs/HPSS) has no piano stem -> we proxy piano with `other`.
# This is a documented spike assumption; it does not affect the shape/dtype
# contract this spike verifies. See FINDINGS.md "Stem mapping".
SPLEETER_ORDER = ("vocals", "drums", "bass", "piano", "other")

_MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                              fmin=FMIN, fmax=FMAX)  # (128, 2049)


def _sine_chord(sr=SR, duration=6.0):
    """Same synthetic A-minor-ish triad as tests/test_beat_engine.py."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 110 * t)
    y += 0.25 * np.sin(2 * np.pi * 138 * t)
    y += 0.2 * np.sin(2 * np.pi * 165 * t)
    return y.astype(np.float32)


def _stem_logmel(stem: np.ndarray) -> np.ndarray:
    """One mono stem -> (T, 128) log-power-mel, per demixing.py + dataset.py.

    demixing.py:  power = |STFT|**2 ;  mel = power . mel_f  (power mel)
    dataset.py __getitem__:  librosa.power_to_db(mel, ref=np.max)  (per stem)
    """
    stft = librosa.stft(stem, n_fft=N_FFT, hop_length=HOP)   # (2049, T)
    power = np.abs(stft) ** 2                                 # (2049, T)
    mel_power = _MEL_FB @ power                               # (128, T)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)       # (128, T)
    return mel_db.T.astype(np.float32)                        # (T, 128)


def build_model_input(stems: StemBundle) -> torch.Tensor:
    """StemBundle -> (1, 5, T, 128) float tensor in Spleeter channel order."""
    field_for = {
        "vocals": stems.vocals, "drums": stems.drums, "bass": stems.bass,
        "piano": stems.other,   # no Demucs piano stem -> proxy with `other`
        "other": stems.other,
    }
    mels = [_stem_logmel(field_for[name]) for name in SPLEETER_ORDER]  # 5x(T,128)
    T = min(m.shape[0] for m in mels)
    arr = np.stack([m[:T] for m in mels], axis=0)          # (5, T, 128)
    return torch.from_numpy(arr).unsqueeze(0).float()      # (1, 5, T, 128)


def activations_to_times(pred: torch.Tensor):
    """(1, T, 2) logits -> (beat_times, downbeat_times) seconds.

    Canonical post-processing (code/eight_fold_test.py) feeds the sigmoid
    activations to madmom DBN trackers:
        beat_act = sigmoid(pred[0, :, 0]);  down_act = sigmoid(pred[0, :, 1])
        combined = concat(max(beat_act - down_act, 0), down_act)  # (T, 2)
        DBNBeatTrackingProcessor(fps=FPS)(beat_act)
        DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=FPS)(combined)
    madmom is NOT installed in this env (known Task 1 gap), so the spike uses a
    simple threshold + local-max picker purely to prove the activation->time
    conversion runs and yields sorted second-arrays. Task 4 must swap in the
    madmom DBN above for real accuracy.
    """
    beat_act = torch.sigmoid(pred[0, :, 0]).cpu().numpy()
    down_act = torch.sigmoid(pred[0, :, 1]).cpu().numpy()

    def _peak_times(act, thr=0.5, min_gap_frames=3):
        idx = []
        last = -min_gap_frames
        for i in range(1, len(act) - 1):
            if act[i] >= thr and act[i] >= act[i - 1] and act[i] > act[i + 1]:
                if i - last >= min_gap_frames:
                    idx.append(i)
                    last = i
        return np.asarray(idx, dtype=np.float64) / FPS

    return _peak_times(beat_act), _peak_times(down_act)


def main() -> int:
    assert os.path.exists(CKPT), f"checkpoint missing: {CKPT}"

    # 1. synthetic clip -> stems (HPSS by default; contract is stem-agnostic)
    y = _sine_chord()
    stems = separate_stems(y, SR)
    assert isinstance(stems, StemBundle)

    # 2. traced spectrogram input
    x = build_model_input(stems)
    T_in = x.shape[2]
    assert x.shape == (1, 5, T_in, 128), f"bad input shape {tuple(x.shape)}"
    assert x.dtype == torch.float32

    # 3. construct model + load checkpoint, eval, no-grad CPU forward
    model = Demixed_DilatedTransformerModel(**MODEL_KW)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        pred, tempo = model(x)          # pred: (1, T, 2), tempo: (1, 300)

    # 4. assert output contract, convert to times
    assert pred.shape == (1, T_in, 2), (
        f"output {tuple(pred.shape)} != (1, {T_in}, 2)")
    assert tempo.shape == (1, 300), f"tempo head {tuple(tempo.shape)}"
    beat_times, downbeat_times = activations_to_times(pred)
    assert beat_times.ndim == 1 and downbeat_times.ndim == 1
    assert np.all(np.diff(beat_times) >= 0)
    assert np.all(np.diff(downbeat_times) >= 0)

    print(f"input   : {tuple(x.shape)}  (batch, instr, time, melbin)  "
          f"FPS={FPS:.3f}")
    print(f"output  : pred={tuple(pred.shape)}  tempo={tuple(tempo.shape)}")
    print(f"beats   : {len(beat_times)}  downbeats: {len(downbeat_times)}  "
          f"(synthetic clip -> sparse, expected)")
    print("SPIKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
