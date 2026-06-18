"""Bar-level chord timeline decoding (one chord per bar by default)."""
from __future__ import annotations

import os

import numpy as np

from chord_constants import CHORD_POLISH_DEFAULTS

BAR_DECODE_ENABLED = os.getenv("CHORD_BAR_DECODE", "1").lower() not in ("0", "false", "no")
BAR_DECODE_STRICT = os.getenv("CHORD_BAR_STRICT", "1").lower() not in ("0", "false", "no")


def _bar_index(time_sec: float, downbeat_times: np.ndarray, beat_duration: float, beats_per_bar: int) -> int:
    if len(downbeat_times) == 0:
        return int(time_sec / max(beat_duration * beats_per_bar, 1e-6))
    idx = int(np.searchsorted(downbeat_times, time_sec, side="right") - 1)
    return max(0, idx)


def quantize_segment_starts_to_downbeats(segments, downbeat_times, beat_duration, beats_per_bar=4):
    """Snap segment start times to the nearest preceding downbeat."""
    if len(segments) < 2 or len(downbeat_times) == 0:
        return segments

    out = []
    for seg in segments:
        bar_idx = _bar_index(float(seg["time"]), downbeat_times, beat_duration, beats_per_bar)
        bar_idx = min(bar_idx, len(downbeat_times) - 1)
        entry = dict(seg)
        entry["time"] = float(downbeat_times[bar_idx])
        out.append(entry)

    out.sort(key=lambda s: float(s["time"]))
    for i in range(len(out) - 1):
        out[i]["end_time"] = float(out[i + 1]["time"])
    if out:
        out[-1]["end_time"] = float(segments[-1].get("end_time", out[-1]["time"] + beat_duration * beats_per_bar))
    return out


def merge_segments_within_bar(
    segments,
    downbeat_times,
    beat_duration,
    score_fn,
    beats_per_bar=None,
    match_threshold=None,
):
    """
    Collapse multiple labels in the same bar to one chord.

    score_fn(bar_start, bar_end) -> (best_chord, best_score, candidates)
    Keeps beat-level splits when an alternate root scores strongly on its span.
    """
    beats_per_bar = beats_per_bar or CHORD_POLISH_DEFAULTS["ml_beats_per_bar"]
    match_threshold = match_threshold or CHORD_POLISH_DEFAULTS["ml_bar_merge_match_threshold"]

    if len(segments) < 2 or len(downbeat_times) == 0:
        return segments

    groups: list[tuple[int, list[dict]]] = []
    current_bar: int | None = None
    current: list[dict] = []

    for seg in segments:
        bar_idx = _bar_index(float(seg["time"]), downbeat_times, beat_duration, beats_per_bar)
        if current_bar is None or bar_idx != current_bar:
            if current:
                groups.append((current_bar, current))
            current = [seg]
            current_bar = bar_idx
        else:
            current.append(seg)
    if current:
        groups.append((current_bar, current))

    merged: list[dict] = []
    for bar_idx, group in groups:
        bar_idx = min(int(bar_idx), len(downbeat_times) - 1)
        bar_start = float(downbeat_times[bar_idx])
        if bar_idx + 1 < len(downbeat_times):
            bar_end = float(downbeat_times[bar_idx + 1])
        else:
            bar_end = float(group[-1].get("end_time", bar_start + beat_duration * beats_per_bar))

        candidates = list(dict.fromkeys(seg.get("chord", "N") for seg in group))
        best_chord, best_score, _ = score_fn(bar_start, bar_end, candidates)

        strong_alternates = []
        if not BAR_DECODE_STRICT:
            margin = CHORD_POLISH_DEFAULTS["bar_alternate_min_margin"]
            for seg in group:
                chord = seg.get("chord", "N")
                if chord == best_chord:
                    continue
                seg_start = float(seg["time"])
                seg_end = float(seg.get("end_time", seg_start + beat_duration))
                if seg_end - seg_start < beat_duration * 0.75:
                    continue
                _, alt_score, _ = score_fn(seg_start, seg_end, [chord])
                if alt_score >= match_threshold and alt_score >= best_score + margin:
                    strong_alternates.append(dict(seg))

        if strong_alternates:
            merged.extend(dict(seg) for seg in group)
            continue

        merged.append({
            "time": bar_start,
            "end_time": bar_end,
            "chord": best_chord,
            "confidence": max(float(s.get("confidence", 0)) for s in group),
            "is_low_confidence": False,
            "is_power": str(best_chord).endswith("5"),
            "strumming": group[0].get("strumming", ""),
        })

    for i in range(len(merged) - 1):
        merged[i]["end_time"] = float(merged[i + 1]["time"])
    if merged:
        merged[-1]["end_time"] = float(segments[-1].get("end_time", merged[-1]["time"] + beat_duration * beats_per_bar))
    return merged


def _median_segment_duration(segments):
    if len(segments) < 2:
        return None
    durs = [float(segments[i + 1]["time"]) - float(segments[i]["time"]) for i in range(len(segments) - 1)]
    return float(np.median(durs))


def should_apply_bar_decode(segments, beat_duration, beats_per_bar=4):
    """Skip bar merge when chords change every beat (power-chord eighth notes)."""
    median_dur = _median_segment_duration(segments)
    if median_dur is None:
        return False
    return median_dur > beat_duration * 1.2


def collapse_timeline_to_bars(
    segments,
    downbeat_times,
    beat_duration,
    score_fn,
    beats_per_bar=None,
    alternate_segments=None,
):
    """Hard decode: one chord per downbeat interval (Chord AI-style default)."""
    beats_per_bar = beats_per_bar or CHORD_POLISH_DEFAULTS["ml_beats_per_bar"]
    if not segments or len(downbeat_times) == 0:
        return segments

    song_end = float(
        segments[-1].get("end_time", downbeat_times[-1] + beat_duration * beats_per_bar)
    )
    collapsed: list[dict] = []
    for i, bar_start in enumerate(downbeat_times):
        bar_start = float(bar_start)
        if bar_start >= song_end - 1e-3:
            break
        if i + 1 < len(downbeat_times):
            bar_end = float(downbeat_times[i + 1])
        else:
            bar_end = song_end
        if bar_end <= bar_start + 1e-3:
            continue

        candidates: list[str] = []
        confidences: list[float] = []
        for seg in segments:
            seg_start = float(seg["time"])
            seg_end = float(seg.get("end_time", seg_start + beat_duration))
            if seg_end <= bar_start or seg_start >= bar_end:
                continue
            chord = str(seg.get("chord", "N"))
            if chord and chord != "N":
                candidates.append(chord)
                confidences.append(float(seg.get("confidence", 0.5)))

        if alternate_segments:
            for seg in alternate_segments:
                seg_start = float(seg["time"])
                seg_end = float(seg.get("end_time", seg_start + beat_duration))
                if seg_end <= bar_start or seg_start >= bar_end:
                    continue
                chord = str(seg.get("chord", "N"))
                if chord and chord != "N":
                    candidates.append(chord)

        candidates = list(dict.fromkeys(candidates)) or ["N"]
        best_chord, _, _ = score_fn(bar_start, bar_end, candidates)
        collapsed.append({
            "time": bar_start,
            "end_time": bar_end,
            "chord": best_chord,
            "confidence": max(confidences) if confidences else 0.5,
            "is_low_confidence": False,
            "is_power": str(best_chord).endswith("5"),
            "strumming": "",
        })

    if collapsed:
        collapsed[-1]["end_time"] = song_end
    return collapsed


def should_use_strict_collapse(segments, downbeat_times) -> bool:
    """Only hard-collapse when the timeline is clearly over-segmented."""
    if len(segments) < 3 or len(downbeat_times) < 2:
        return False
    song_end = float(segments[-1].get("end_time", segments[-1]["time"]))
    n_bars = sum(1 for t in downbeat_times if float(t) < song_end - 1e-3)
    if n_bars < 2:
        return False
    ratio = CHORD_POLISH_DEFAULTS["bar_strict_overseg_ratio"]
    return len(segments) > n_bars * ratio


def apply_bar_decode(
    segments,
    downbeat_times,
    beat_duration,
    score_fn,
    enabled: bool | None = None,
    alternate_segments=None,
):
    """Quantize to downbeats and merge within bar when tempo allows."""
    if not segments:
        return segments
    if enabled is None:
        enabled = BAR_DECODE_ENABLED
    if not enabled or not should_apply_bar_decode(segments, beat_duration):
        return segments

    beats_per_bar = CHORD_POLISH_DEFAULTS["ml_beats_per_bar"]
    use_strict = BAR_DECODE_STRICT and should_use_strict_collapse(segments, downbeat_times)
    if use_strict:
        return collapse_timeline_to_bars(
            segments, downbeat_times, beat_duration, score_fn,
            beats_per_bar=beats_per_bar, alternate_segments=alternate_segments,
        )

    segments = quantize_segment_starts_to_downbeats(
        segments, downbeat_times, beat_duration, beats_per_bar,
    )
    segments = merge_segments_within_bar(
        segments, downbeat_times, beat_duration, score_fn,
        beats_per_bar=beats_per_bar,
    )
    return segments
