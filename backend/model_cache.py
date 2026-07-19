"""Singleton ML models — avoid reloading Demucs / basic-pitch on every request."""
from __future__ import annotations

import hashlib
import logging
import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_demucs_model: Any = None
_basic_pitch_model: Any = None
_beat_transformer_model: Any = None

# Beat-Transformer checkpoint (fold 0 of 8), pinned to the commit FINDINGS.md
# verified against — not `/main/`, to avoid drift if upstream force-pushes.
_BT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "beat_transformer.pth"
_BT_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/zhaojw1998/Beat-Transformer/"
    "063667fc9e4e11507f9d76dc1154d9db953a85eb/checkpoint/fold_0_trf_param.pt"
)
_BT_CHECKPOINT_SHA256 = "1940a5034bb2bb7860c1a3219d359906913be88c915f6135a5ce8382c239d738"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_beat_transformer_checkpoint() -> Path:
    """Download the Beat-Transformer checkpoint on first use, verifying SHA-256."""
    if _BT_CHECKPOINT_PATH.exists() and _sha256(_BT_CHECKPOINT_PATH) == _BT_CHECKPOINT_SHA256:
        return _BT_CHECKPOINT_PATH
    _BT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Beat-Transformer checkpoint (fold 0, ~35MB)")
    tmp_path = _BT_CHECKPOINT_PATH.with_suffix(".tmp")
    urllib.request.urlretrieve(_BT_CHECKPOINT_URL, tmp_path)
    digest = _sha256(tmp_path)
    if digest != _BT_CHECKPOINT_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Beat-Transformer checkpoint SHA-256 mismatch: expected "
            f"{_BT_CHECKPOINT_SHA256}, got {digest}"
        )
    tmp_path.replace(_BT_CHECKPOINT_PATH)
    return _BT_CHECKPOINT_PATH


def get_demucs_model():
    global _demucs_model
    if _demucs_model is not None:
        return _demucs_model
    with _lock:
        if _demucs_model is None:
            from demucs.pretrained import get_model

            logger.info("Loading Demucs htdemucs model (cached for process lifetime)")
            model = get_model("htdemucs")
            model.eval()
            _demucs_model = model
        return _demucs_model


def get_beat_transformer_model():
    global _beat_transformer_model
    if _beat_transformer_model is not None:
        return _beat_transformer_model
    with _lock:
        if _beat_transformer_model is None:
            import torch

            from beat_transformer import Demixed_DilatedTransformerModel

            logger.info("Loading Beat-Transformer model (cached for process lifetime)")
            ckpt_path = _ensure_beat_transformer_checkpoint()
            # Constructor args per FINDINGS.md sec 2 (code/eight_fold_test.py
            # constants — NOT the class-signature defaults, which differ).
            model = Demixed_DilatedTransformerModel(
                attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
                d_hid=1024, nlayers=9, norm_first=True,
            )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            model.eval()
            _beat_transformer_model = model
        return _beat_transformer_model


def get_basic_pitch_model():
    global _basic_pitch_model
    if _basic_pitch_model is not None:
        return _basic_pitch_model
    with _lock:
        if _basic_pitch_model is None:
            from basic_pitch.inference import ICASSP_2022_MODEL_PATH, Model

            logger.info("Loading basic-pitch model (cached for process lifetime)")
            _basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
        return _basic_pitch_model


def preload_ml_models() -> None:
    """Warm models at startup when deps are present (best-effort)."""
    try:
        from stem_separation import demucs_available

        if demucs_available():
            get_demucs_model()
    except Exception as exc:
        logger.debug("Demucs preload skipped: %s", exc)
    try:
        get_basic_pitch_model()
    except Exception as exc:
        logger.debug("basic-pitch preload skipped: %s", exc)
