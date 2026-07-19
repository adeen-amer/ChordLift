"""Beat-Transformer inference: StemBundle -> BeatGrid. See FINDINGS.md for the contract."""
from __future__ import annotations

import logging

import librosa
import numpy as np

from beat_tracking import BEATS_PER_BAR, BeatGrid

logger = logging.getLogger(__name__)

# Contract constants (verified from upstream @063667f, see FINDINGS.md sec 2-3).
SR = 44100
N_FFT = 4096
HOP = 1024
N_MELS = 128
FMIN, FMAX = 30, 11000
FPS = SR / HOP  # 43.066 Hz, frame rate of both model input and output

# Reference Spleeter 5-stem channel order the checkpoint was trained on.
# StemBundle (Demucs/HPSS) has no piano stem -> proxy with `other` (FINDINGS.md sec 3).
SPLEETER_ORDER = ("vocals", "drums", "bass", "piano", "other")

_MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


def _stem_logmel(stem: np.ndarray) -> np.ndarray:
    """One mono stem -> (T, 128) log-power-mel, per demixing.py + dataset.py."""
    stft = librosa.stft(stem, n_fft=N_FFT, hop_length=HOP)  # (2049, T)
    power = np.abs(stft) ** 2  # (2049, T)
    mel_power = _MEL_FB @ power  # (128, T)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)  # (128, T)
    return mel_db.T.astype(np.float32)  # (T, 128)


def build_model_input(stems, sr: int):
    """StemBundle -> (1, 5, T, 128) float tensor in Spleeter channel order."""
    import torch

    def _resample(y: np.ndarray) -> np.ndarray:
        return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y

    field_for = {
        "vocals": stems.vocals, "drums": stems.drums, "bass": stems.bass,
        "piano": stems.other,  # no Demucs piano stem -> proxy with `other`
        "other": stems.other,
    }
    mels = [_stem_logmel(_resample(field_for[name])) for name in SPLEETER_ORDER]  # 5x(T,128)
    T = min(m.shape[0] for m in mels)
    arr = np.stack([m[:T] for m in mels], axis=0)  # (5, T, 128)
    return torch.from_numpy(arr).unsqueeze(0).float()  # (1, 5, T, 128)


def run(stems, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid:
    """Run Beat-Transformer on a StemBundle's real (Demucs) stems.

    Raises on any failure — callers handle fallback (beat_tracking.track_beats_auto).
    """
    import madmom
    import torch

    from model_cache import get_beat_transformer_model

    model = get_beat_transformer_model()
    x = build_model_input(stems, sr)
    with torch.no_grad():
        pred, _tempo = model(x)  # pred: (1, T, 2); tempo head unused (FINDINGS.md sec 2)

    beat_act = torch.sigmoid(pred[0, :, 0]).cpu().numpy()
    down_act = torch.sigmoid(pred[0, :, 1]).cpu().numpy()

    # Canonical post-processing (code/eight_fold_test.py), see FINDINGS.md sec 4.
    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
        min_bpm=55.0, max_bpm=215.0, fps=FPS,
        transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2,
    )
    beat_times = np.asarray(beat_tracker(beat_act), dtype=np.float64)

    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS,
        transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2,
    )
    combined = np.stack([np.maximum(beat_act - down_act, 0.0), down_act], axis=-1)  # (T, 2)
    db = downbeat_tracker(combined)  # rows: [time, beat_pos]
    downbeat_times = np.asarray(db[db[:, 1] == 1][:, 0], dtype=np.float64)

    if len(beat_times) == 0:
        raise RuntimeError("Beat-Transformer produced no beats")
    if len(downbeat_times) == 0:
        downbeat_times = beat_times[::beats_per_bar]

    tempo_bpm = 60.0 / float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 120.0

    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        tempo_bpm=tempo_bpm,
        beats_per_bar=beats_per_bar,
    )
