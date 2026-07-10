"""ISMIR 2019 large-vocabulary chord recognition (lv-chordia / PyTorch)."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

from chord_label_utils import raw_labels_to_segments

logger = logging.getLogger(__name__)

CHORDIA_DICT = os.getenv("CHORD_CHORDIA_DICT", "submission").lower().strip()


def _model_names() -> list[str]:
    """Serving checkpoint list; CHORD_CHORDIA_MODELS overrides (Phase B)."""
    raw = os.getenv("CHORD_CHORDIA_MODELS", "").strip()
    if raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    from lv_chordia.chord_recognition import MODEL_NAMES
    return list(MODEL_NAMES)


def recognize_chordia_probs(y_chord, sr: int):
    """
    Run lv-chordia inference and return frame posteriors instead of labels.

    Returns (probs, hmm, entry):
      probs — list of 6 arrays [n_frames, n_classes] (triad, bass, 7, 9, 11, 13
              softmax heads), averaged over the 5 packaged checkpoints.
      hmm   — XHMMDecoder ready to decode these probs.
      entry — DataEntry carrying sr/hop metadata the decoder needs.

    Mirrors lv_chordia.chord_recognition.chord_recognition() internals so the
    posteriors can be reused (confidence scoring, TTA averaging) before decode.
    """
    import importlib.resources

    from lv_chordia.chordnet_ismir_naive import ChordNet
    from lv_chordia.extractors.cqt import CQTV2
    from lv_chordia.extractors.xhmm_ismir import XHMMDecoder
    from lv_chordia.mir import DataEntry, io
    from lv_chordia.mir.nn.train import NetworkInterface
    from lv_chordia.settings import DEFAULT_HOP_LENGTH, DEFAULT_SR

    with importlib.resources.path(
        "lv_chordia.data", f"{CHORDIA_DICT}_chord_list.txt"
    ) as data_file:
        hmm = XHMMDecoder(template_file=str(data_file))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y_chord, sr)
    try:
        entry = DataEntry()
        entry.prop.set("sr", DEFAULT_SR)
        entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
        entry.append_file(tmp_path, io.MusicIO, "music")
        entry.append_extractor(CQTV2, "cqt")
        cqt = entry.cqt  # materialize before the temp file is unlinked
        model_dir = os.getenv("CHORD_CHORDIA_MODEL_DIR", "").strip()
        net_kwargs = {"load_path": model_dir} if model_dir else {}
        per_model = []
        for model_name in _model_names():
            net = NetworkInterface(
                ChordNet(None), model_name, load_checkpoint=False, **net_kwargs
            )
            logger.info("Inference: %s", model_name)
            per_model.append(net.inference(cqt))
        probs = [
            np.mean([p[i] for p in per_model], axis=0)
            for i in range(len(per_model[0]))
        ]
        return probs, hmm, entry
    finally:
        os.unlink(tmp_path)


def decode_chordia_probs(probs, hmm, entry) -> list[dict]:
    """XHMM-decode posteriors to raw chordia JSON segments."""
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    return [
        {
            "start_time": float(f"{seg[0]:.2f}"),
            "end_time": float(f"{seg[1]:.2f}"),
            "chord": str(seg[2]),
        }
        for seg in chordlab
    ]


def chordia_confidence(probs) -> float:
    """Decoder confidence: mean over frames of the max triad posterior."""
    return float(np.asarray(probs[0]).max(axis=1).mean())


def recognize_chordia(y_chord, sr: int) -> list[dict]:
    """
    Run lv-chordia on a mono waveform (prefer stem chord_signal).

    Returns raw chordia JSON segments: [{start_time, end_time, chord}, ...]
    """
    probs, hmm, entry = recognize_chordia_probs(y_chord, sr)
    return decode_chordia_probs(probs, hmm, entry)


def chordia_to_segments(raw) -> list[dict]:
    return raw_labels_to_segments(raw, default_confidence=0.78)
