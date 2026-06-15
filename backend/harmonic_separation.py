"""Harmonic source extraction for chord analysis."""
from __future__ import annotations

import logging
import os

import librosa
import numpy as np

logger = logging.getLogger(__name__)

USE_DEMUCS = os.getenv("CHORD_USE_DEMUCS", "0").lower() in ("1", "true", "yes")


def _demucs_harmonic(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Optional Demucs 4-stem separation (requires `pip install demucs`).

    Uses the other stem (everything except drums/bass/vocals approximation)
    as a harmonic-heavy signal for chroma.
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs")
    model.eval()

    if y.ndim > 1:
        wav = y.T
    else:
        wav = np.stack([y, y])

    ref_sr = model.samplerate
    if sr != ref_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=ref_sr, axis=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)

    with torch.no_grad():
        sources = apply_model(model, tensor, device=device)[0]

    # stems: drums, bass, other, vocals — other + vocals carry harmony/melody
    other = sources[2].mean(dim=0).cpu().numpy()
    vocals = sources[3].mean(dim=0).cpu().numpy()
    harmonic = 0.65 * other + 0.35 * vocals

    if sr != ref_sr:
        harmonic = librosa.resample(harmonic, orig_sr=ref_sr, target_sr=sr)

    return harmonic.astype(np.float32)


def extract_harmonic(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Return a harmonic-heavy mono signal for chroma extraction.

    Default: two-pass HPSS with widened margin (no extra deps).
    Set CHORD_USE_DEMUCS=1 to try Demucs when installed.
    """
    if USE_DEMUCS:
        try:
            return _demucs_harmonic(y, sr)
        except Exception as exc:
            logger.warning("Demucs separation failed, using HPSS: %s", exc)

    y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
    y_harmonic, _ = librosa.effects.hpss(y_harmonic, margin=(1.0, 3.0))
    return y_harmonic
