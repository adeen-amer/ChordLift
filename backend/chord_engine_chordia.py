"""ISMIR 2019 large-vocabulary chord recognition (lv-chordia / PyTorch)."""
from __future__ import annotations

import logging
import os
import tempfile

import soundfile as sf

from chord_label_utils import raw_labels_to_segments

logger = logging.getLogger(__name__)

CHORDIA_DICT = os.getenv("CHORD_CHORDIA_DICT", "submission").lower().strip()


def recognize_chordia(y_chord, sr: int) -> list[dict]:
    """
    Run lv-chordia on a mono waveform (prefer stem chord_signal).

    Returns raw chordia JSON segments: [{start_time, end_time, chord}, ...]
    """
    from lv_chordia.chord_recognition import chord_recognition

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y_chord, sr)

    try:
        return chord_recognition(audio_path=tmp_path, chord_dict_name=CHORDIA_DICT)
    finally:
        os.unlink(tmp_path)


def chordia_to_segments(raw) -> list[dict]:
    return raw_labels_to_segments(raw, default_confidence=0.78)
