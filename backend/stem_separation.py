"""4-stem separation for chord analysis (Chord AI-style harmonic prep)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

STEM_MODE = os.getenv("CHORD_STEM_MODE", "auto").lower().strip()
USE_DEMUCS = os.getenv("CHORD_USE_DEMUCS", "0").lower() in ("1", "true", "yes")
PREFER_DEMUCS = os.getenv("CHORD_STEM_PREFER_DEMUCS", "0").lower() in ("1", "true", "yes")

_demucs_checked = False
_demucs_available = False


def demucs_available() -> bool:
    """True when demucs + torch are importable."""
    global _demucs_checked, _demucs_available
    if _demucs_checked:
        return _demucs_available
    _demucs_checked = True
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        _demucs_available = True
    except Exception:
        _demucs_available = False
    return _demucs_available


@dataclass(frozen=True)
class StemBundle:
    """Mono stems at the input sample rate."""

    full: np.ndarray
    bass: np.ndarray
    drums: np.ndarray
    vocals: np.ndarray
    other: np.ndarray
    method: str

    @property
    def chord_signal(self) -> np.ndarray:
        """Harmony-heavy mix for chord ID (other + bass, Chord AI-style)."""
        if self.method == "demucs":
            return (0.55 * self.other + 0.45 * self.bass).astype(np.float32)
        return (0.65 * self.other + 0.35 * self.bass).astype(np.float32)


def _lowpass_bass(y: np.ndarray, sr: int, cutoff_hz: float = 220.0) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _hpss_stems(y: np.ndarray, sr: int) -> StemBundle:
    """Lightweight pseudo-stems without Demucs (HPSS + low-pass bass)."""
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    y_harmonic, _ = librosa.effects.hpss(y_harmonic, margin=(1.0, 3.0))
    y_bass = _lowpass_bass(y, sr)
    y_other = np.clip(y_harmonic - 0.35 * y_bass, -1.0, 1.0).astype(np.float32)
    y_vocals = np.clip(0.55 * y_harmonic - 0.25 * y_other, -1.0, 1.0).astype(np.float32)
    return StemBundle(
        full=y.astype(np.float32),
        bass=y_bass,
        drums=y_percussive.astype(np.float32),
        vocals=y_vocals,
        other=y_other,
        method="hpss",
    )


def _demucs_stems(y: np.ndarray, sr: int) -> StemBundle:
    import torch
    from demucs.apply import apply_model

    from model_cache import get_demucs_model

    model = get_demucs_model()

    wav = np.stack([y, y]) if y.ndim == 1 else y.T
    ref_sr = model.samplerate
    if sr != ref_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=ref_sr, axis=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(model, tensor, device=device)[0]

    def _stem(idx: int) -> np.ndarray:
        mono = sources[idx].mean(dim=0).cpu().numpy()
        if sr != ref_sr:
            mono = librosa.resample(mono, orig_sr=ref_sr, target_sr=sr)
        return mono.astype(np.float32)

    return StemBundle(
        full=y.astype(np.float32),
        drums=_stem(0),
        bass=_stem(1),
        other=_stem(2),
        vocals=_stem(3),
        method="demucs",
    )


def separate_stems(y: np.ndarray, sr: int) -> StemBundle:
    """
    Separate audio into bass / drums / vocals / other for chord analysis.

    Modes (CHORD_STEM_MODE):
      - demucs: require Demucs (falls back to HPSS)
      - hpss: fast HPSS pseudo-stems (default for CI)
      - best: Demucs when installed, else HPSS
      - auto: demucs when CHORD_USE_DEMUCS=1 or CHORD_STEM_PREFER_DEMUCS=1 and demucs installed
    """
    mode = STEM_MODE
    if mode == "best":
        mode = "demucs" if demucs_available() else "hpss"
    elif mode == "auto":
        use_demucs = USE_DEMUCS or (PREFER_DEMUCS and demucs_available())
        mode = "demucs" if use_demucs else "hpss"

    if mode == "demucs":
        try:
            return _demucs_stems(y, sr)
        except Exception as exc:
            logger.warning("Demucs stem separation failed, using HPSS: %s", exc)

    return _hpss_stems(y, sr)
