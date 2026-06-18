"""Display-time chord timeline polish (Phase 13 — no model changes)."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from analyzer import (
    DIATONIC_MAJOR_INTERVALS,
    DIATONIC_MAJOR_QUALITIES,
    DIATONIC_MINOR_INTERVALS,
    _chord_root_name,
    _root_to_pitch_class,
)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LOW_CONF = 0.35
MED_CONF = 0.55

# Natural minor triad qualities: i, ii°, III, iv, v, VI, VII
DIATONIC_MINOR_QUALITIES = ["m", "dim", "", "m", "m", "", ""]


def _chord_quality_suffix(chord: str) -> str:
    root = _chord_root_name(chord)
    return chord[len(root):] if chord.startswith(root) else ""


def _diatonic_chord_set(key_root: int, is_major: bool) -> set[str]:
    if is_major:
        intervals, qualities = DIATONIC_MAJOR_INTERVALS, DIATONIC_MAJOR_QUALITIES
    else:
        intervals, qualities = DIATONIC_MINOR_INTERVALS, DIATONIC_MINOR_QUALITIES
    out = set()
    for iv, q in zip(intervals, qualities):
        root = NOTE_NAMES[(key_root + iv) % 12]
        out.add(f"{root}{q}")
    return out


def _scale_pitch_classes(key_root: int, is_major: bool) -> set[int]:
    intervals = DIATONIC_MAJOR_INTERVALS if is_major else DIATONIC_MINOR_INTERVALS
    return {(key_root + iv) % 12 for iv in intervals}


def _is_secondary_dominant(root_pc: int, quality: str, key_root: int, is_major: bool) -> bool:
    """Major (or dom7) chord a P5 above a diatonic degree — likely V/x."""
    if "m" in quality and "maj" not in quality:
        return False
    if "dim" in quality:
        return False
    scale = _scale_pitch_classes(key_root, is_major)
    target = (root_pc - 7) % 12
    return target in scale


def _nearest_diatonic(chord: str, allowed: set[str], key_root: int, is_major: bool) -> str:
    if not chord or chord == "N" or chord in allowed:
        return chord
    root = _chord_root_name(chord)
    quality = _chord_quality_suffix(chord)
    root_pc = _root_to_pitch_class(root)
    if root_pc is None:
        return chord
    if _is_secondary_dominant(root_pc, quality, key_root, is_major):
        return chord

    same_q = [c for c in allowed if _chord_quality_suffix(c) == quality]
    if not same_q and quality in ("", "maj"):
        same_q = [c for c in allowed if _chord_quality_suffix(c) in ("", "maj")]
    if not same_q and quality.startswith("m"):
        same_q = [c for c in allowed if _chord_quality_suffix(c).startswith("m")]

    pool = same_q if same_q else list(allowed)
    best = chord
    best_dist = 99
    for cand in pool:
        cand_pc = _root_to_pitch_class(_chord_root_name(cand))
        if cand_pc is None:
            continue
        dist = min((cand_pc - root_pc) % 12, (root_pc - cand_pc) % 12)
        if dist < best_dist:
            best_dist = dist
            best = cand
    if best_dist <= 2 and _chord_quality_suffix(best) == quality:
        return best
    if best_dist <= 1:
        return best
    return chord


def _snap_time(t: float, beat_times: np.ndarray) -> float:
    if beat_times is None or len(beat_times) == 0:
        return t
    idx = int(np.argmin(np.abs(beat_times - t)))
    return float(beat_times[idx])


def snap_timeline_to_beats(timeline: list[dict], beat_times: np.ndarray) -> list[dict]:
    """Snap chord boundaries to nearest beat — display only."""
    if len(timeline) < 2 or beat_times is None or len(beat_times) == 0:
        return timeline
    out = []
    for i, seg in enumerate(timeline):
        seg = copy.deepcopy(seg)
        seg["time"] = _snap_time(float(seg["time"]), beat_times)
        if i < len(timeline) - 1:
            seg["end_time"] = _snap_time(float(timeline[i + 1]["time"]), beat_times)
        else:
            seg["end_time"] = max(float(seg.get("end_time", seg["time"] + 1)), seg["time"] + 0.25)
        out.append(seg)
    merged = [out[0]]
    for seg in out[1:]:
        if seg["chord"] == merged[-1]["chord"]:
            merged[-1]["end_time"] = seg["end_time"]
            merged[-1]["confidence"] = max(merged[-1].get("confidence", 0), seg.get("confidence", 0))
        else:
            merged.append(seg)
    return merged


def constrain_timeline_to_key(
    timeline: list[dict],
    key_root: int,
    is_major: bool,
    *,
    low_conf_only: bool = True,
) -> list[dict]:
    """Snap low-confidence out-of-key display chords to nearest diatonic (preserve quality)."""
    allowed = _diatonic_chord_set(key_root, is_major)
    out = []
    for seg in timeline:
        seg = copy.deepcopy(seg)
        conf = float(seg.get("confidence", 0.75))
        if seg.get("chord") and (not low_conf_only or conf < MED_CONF or seg.get("is_low_confidence")):
            original = seg["chord"]
            snapped = _nearest_diatonic(original, allowed, key_root, is_major)
            if snapped != original:
                seg["chord"] = snapped
                seg["display_adjusted"] = True
                seg["model_chord"] = original
        out.append(seg)
    return out


def capo_info_default() -> dict[str, Any]:
    """Capo is not inferred from pitch detune — only from future capo detection."""
    return {"capo_fret": 0, "display": None, "transpose_semitones": 0}


def apply_confidence_tiers(timeline: list[dict]) -> list[dict]:
    out = []
    for seg in timeline:
        seg = copy.deepcopy(seg)
        conf = float(seg.get("confidence", 0.75))
        seg["is_low_confidence"] = conf < LOW_CONF
        if conf < LOW_CONF:
            seg["confidence_tier"] = "low"
        elif conf < MED_CONF:
            seg["confidence_tier"] = "medium"
        else:
            seg["confidence_tier"] = "high"
        out.append(seg)
    return out


def build_display_timeline(
    model_timeline: list[dict],
    pipeline,
    key_info: dict,
) -> dict[str, Any]:
    """Full presentation pipeline for API / frontend."""
    key_root = NOTE_NAMES.index(key_info["root"]) if key_info.get("root") in NOTE_NAMES else 0
    is_major = key_info.get("mode", "major") == "major"

    timeline = apply_confidence_tiers(model_timeline)
    timeline = snap_timeline_to_beats(timeline, pipeline.beat_times)
    timeline = constrain_timeline_to_key(timeline, key_root, is_major)

    beats = {
        "tempo_bpm": round(float(pipeline.tempo_bpm), 1),
        "beat_times": [round(float(t), 3) for t in pipeline.beat_times[:500]],
        "downbeat_times": [round(float(t), 3) for t in pipeline.downbeat_times[:200]],
        "beat_duration": round(float(pipeline.beat_duration), 3),
    }

    return {
        "timeline": timeline,
        "model_timeline": model_timeline,
        "beats": beats,
        "capo": capo_info_default(),
        "presentation": "v13",
    }
