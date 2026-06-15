"""
Optional ML chord engine (autochord Bi-LSTM-CRF).

Install: pip install autochord tensorflow
Requires VAMP plugins on some platforms — see https://github.com/cjbayron/autochord

Pipeline: autochord boundaries → chroma/NNLS classic re-score per segment
(windowed local key) → power-pattern lock (rock loops) → onset snap.
"""
import os
import tempfile
from collections import Counter

import librosa
import numpy as np

from analyzer import (
    CHORD_POLISH_DEFAULTS,
    _align_segments_to_chroma,
    _bar_aware_min_duration,
    _beat_sync_chroma_stack,
    _build_key_info,
    _chord_quality,
    _chord_root_name,
    _estimate_power_song_ratio,
    _extract_chroma_stack,
    _format_chord_name,
    _key_at_time,
    _match_score_for_chord,
    _merge_adjacent_same_root,
    _polish_chord_timeline,
    _refine_segments_with_vocabulary,
    _resolve_song_key,
    _segment_center_chroma,
    _to_sharp_root,
    _try_pattern_alignment,
    _windowed_keys_from_chroma,
)


def _autochord_label_to_internal(label):
    """Map autochord labels (e.g. F:maj, Bb:min) to ChordLift internal names."""
    label = (label or "").strip()
    if not label or label in ("N", "X", "no_chord"):
        return None

    if ":" in label:
        root, quality = label.split(":", 1)
        root = _to_sharp_root(root.strip())
        quality = quality.lower()
        if quality in ("maj", "major"):
            return root
        if quality in ("min", "minor"):
            return f"{root}m"
        if quality == "7":
            return f"{root}7"
        if quality == "maj7":
            return f"{root}maj7"
        if quality in ("min7", "min7b5"):
            return f"{root}m7"
        if quality in ("dim", "aug", "sus4", "sus2"):
            return f"{root}{quality}"
        return root

    return label


def _autochord_to_segments(chords):
    """Convert autochord [(start, end, label), ...] to ChordLift segment dicts."""
    segments = []
    for start, end, label in chords:
        internal = _autochord_label_to_internal(label)
        if internal is None:
            continue
        segments.append({
            "time": float(start),
            "end_time": float(end),
            "chord": internal,
            "confidence": 0.72,
            "is_low_confidence": False,
            "is_power": internal.endswith("5"),
            "strumming": "D DU UDU",
        })
    return segments


def _merge_short_ml_segments(segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr, beat_duration=0.5):
    """Collapse autochord flicker; keep the chord that best matches the merged audio."""
    if len(segments) < 2:
        return segments

    min_duration = _bar_aware_min_duration(beat_duration) * 0.90
    hop = 512
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        duration = float(seg["time"]) - float(prev["time"])
        if duration < min_duration:
            merge_end = float(seg.get("end_time", seg["time"]))
            if frame_keys is not None and beat_times is not None and len(beat_times) > 0:
                key_r, key_m = _key_at_time(
                    (float(prev["time"]) + merge_end) / 2.0, beat_times, frame_keys,
                )
            else:
                key_r, key_m = 0, True
            prev_root = _chord_root_name(prev.get("chord", "N"))
            seg_root = _chord_root_name(seg.get("chord", "N"))
            prev_c, prev_b, prev_m = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(prev["time"]), merge_end, sr, hop,
            )
            score_prev = _match_score_for_chord(
                prev_c, prev_b, prev_m, prev.get("chord", "N"), key_r, key_m,
            )
            score_new = _match_score_for_chord(
                prev_c, prev_b, prev_m, seg.get("chord", "N"), key_r, key_m,
            )
            seg_c, seg_b, seg_m = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(seg["time"]), merge_end, sr, hop,
            )
            score_seg_self = _match_score_for_chord(
                seg_c, seg_b, seg_m, seg.get("chord", "N"), key_r, key_m,
            )
            distinct_root = (
                seg_root not in ("N", prev_root)
                and score_seg_self >= score_prev * 0.72
            )
            if distinct_root:
                merged.append(dict(seg))
                continue
            prev["end_time"] = merge_end
            if score_new > score_prev:
                prev["chord"] = seg["chord"]
                prev["confidence"] = seg.get("confidence", prev.get("confidence", 0))
        else:
            merged.append(dict(seg))

    for i in range(len(merged) - 1):
        merged[i]["end_time"] = float(merged[i + 1]["time"])
    if merged:
        merged[-1]["end_time"] = float(segments[-1].get("end_time", merged[-1]["time"] + 1.0))
    return merged


def _snap_ml_segments_to_beats(segments, beat_times):
    """
    Snap autochord boundary times to the nearest beat grid (Phase 1 smoothing).

    Collapses sub-beat boundaries before chroma re-labeling.
    """
    if len(segments) < 2 or beat_times is None or len(beat_times) < 2:
        return segments

    beats = np.asarray(beat_times, dtype=np.float64)
    snapped: list[dict] = []

    for seg in segments:
        start = float(seg["time"])
        beat_idx = int(np.argmin(np.abs(beats - start)))
        snapped_start = float(beats[beat_idx])

        if snapped and snapped_start <= snapped[-1]["time"] + 1e-6:
            snapped[-1]["end_time"] = max(
                float(seg.get("end_time", snapped_start)),
                float(snapped[-1].get("end_time", snapped_start)),
            )
            if float(seg.get("confidence", 0)) > float(snapped[-1].get("confidence", 0)):
                snapped[-1]["chord"] = seg["chord"]
                snapped[-1]["confidence"] = seg["confidence"]
            continue

        entry = dict(seg)
        entry["time"] = snapped_start
        snapped.append(entry)

    for i in range(len(snapped) - 1):
        snapped[i]["end_time"] = float(snapped[i + 1]["time"])
    if snapped:
        snapped[-1]["end_time"] = float(
            segments[-1].get("end_time", snapped[-1]["time"] + 1.0),
        )
    return _merge_adjacent_same_root(snapped)


def _bar_start_times(beat_times, beats_per_bar=4):
    """Downbeat times from a beat grid (assumes 4/4)."""
    beats = np.asarray(beat_times, dtype=np.float64)
    if len(beats) == 0:
        return beats
    indices = np.arange(0, len(beats), beats_per_bar)
    return beats[indices]


def _bar_index_for_time(time_sec, beat_times, beats_per_bar=4):
    beats = np.asarray(beat_times, dtype=np.float64)
    if len(beats) == 0:
        return 0
    beat_idx = int(np.clip(np.searchsorted(beats, time_sec, side="right") - 1, 0, len(beats) - 1))
    return beat_idx // beats_per_bar


def _median_segment_duration(segments):
    if len(segments) < 2:
        return None
    durs = [float(segments[i + 1]["time"]) - float(segments[i]["time"]) for i in range(len(segments) - 1)]
    return float(np.median(durs))


def _should_use_bar_quantize(segments, beat_duration, beats_per_bar=4):
    """
    Skip bar-level merge when chords change every beat (e.g. power-chord eighth notes).

    General tempo/segment-duration gate — not song-specific.
    """
    median_dur = _median_segment_duration(segments)
    if median_dur is None:
        return False
    return median_dur > beat_duration * 1.2


def _quantize_ml_segments_to_bars(segments, beat_times, beats_per_bar=4):
    """Snap segment starts to their bar downbeat."""
    if len(segments) < 2 or beat_times is None or len(beat_times) < beats_per_bar:
        return segments

    bar_starts = _bar_start_times(beat_times, beats_per_bar)
    if len(bar_starts) == 0:
        return segments

    out: list[dict] = []
    for seg in segments:
        bar_idx = _bar_index_for_time(float(seg["time"]), beat_times, beats_per_bar)
        bar_idx = min(bar_idx, len(bar_starts) - 1)
        entry = dict(seg)
        entry["time"] = float(bar_starts[bar_idx])
        out.append(entry)

    out.sort(key=lambda s: float(s["time"]))
    for i in range(len(out) - 1):
        out[i]["end_time"] = float(out[i + 1]["time"])
    if out:
        out[-1]["end_time"] = float(segments[-1].get("end_time", out[-1]["time"] + 1.0))
    return out


def _merge_ml_segments_within_bar(
    segments,
    beat_times,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    sr,
    beat_duration=0.5,
    beats_per_bar=4,
    match_threshold=0.28,
):
    """
    Collapse multiple ML labels in the same bar to one chord.

    Keeps beat-level splits when an alternate root scores strongly on its span.
    """
    if len(segments) < 2 or beat_times is None or len(beat_times) < beats_per_bar:
        return segments

    bar_starts = _bar_start_times(beat_times, beats_per_bar)
    if len(bar_starts) == 0:
        return segments

    hop = 512
    groups: list[tuple[int, list[dict]]] = []
    current_bar: int | None = None
    current: list[dict] = []

    for seg in segments:
        bar_idx = _bar_index_for_time(float(seg["time"]), beat_times, beats_per_bar)
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
        bar_idx = min(int(bar_idx), len(bar_starts) - 1)
        bar_start = float(bar_starts[bar_idx])
        if bar_idx + 1 < len(bar_starts):
            bar_end = float(bar_starts[bar_idx + 1])
        else:
            bar_end = float(group[-1].get("end_time", bar_start + beat_duration * beats_per_bar))

        if frame_keys is not None and len(beat_times) > 0:
            key_r, key_m = _key_at_time((bar_start + bar_end) / 2.0, beat_times, frame_keys)
        else:
            key_r, key_m = 0, True

        mc, mb, mm = _segment_center_chroma(
            chroma, chroma_low, chroma_mid, bar_start, bar_end, sr, hop,
        )

        candidates = list(dict.fromkeys(seg.get("chord", "N") for seg in group))
        best_chord = candidates[0]
        best_score = -1.0
        for chord in candidates:
            if not chord or chord == "N":
                continue
            score = _match_score_for_chord(mc, mb, mm, chord, key_r, key_m)
            if score > best_score:
                best_score = score
                best_chord = chord

        strong_alternates = []
        for seg in group:
            chord = seg.get("chord", "N")
            if _chord_root_name(chord) == _chord_root_name(best_chord):
                continue
            seg_start = float(seg["time"])
            seg_end = float(seg.get("end_time", seg_start + beat_duration))
            if seg_end - seg_start < beat_duration * 0.75:
                continue
            sc, sb, sm = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, seg_start, seg_end, sr, hop,
            )
            if _match_score_for_chord(sc, sb, sm, chord, key_r, key_m) >= match_threshold:
                strong_alternates.append(dict(seg))

        if strong_alternates:
            for seg in group:
                merged.append(dict(seg))
            continue

        merged.append({
            "time": bar_start,
            "end_time": bar_end,
            "chord": best_chord,
            "confidence": max(float(s.get("confidence", 0)) for s in group),
            "is_low_confidence": False,
            "is_power": str(best_chord).endswith("5"),
            "strumming": group[0].get("strumming", "D DU UDU"),
        })

    for i in range(len(merged) - 1):
        merged[i]["end_time"] = float(merged[i + 1]["time"])
    if merged:
        merged[-1]["end_time"] = float(segments[-1].get("end_time", merged[-1]["time"] + 1.0))
    return _merge_adjacent_same_root(merged)


def _smooth_ml_segment_boundaries(
    segments,
    beat_times,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    sr,
    beat_duration,
):
    """Beat-snap → bar-quantize → within-bar merge (when tempo allows)."""
    beats_per_bar = CHORD_POLISH_DEFAULTS["ml_beats_per_bar"]
    match_threshold = CHORD_POLISH_DEFAULTS["ml_bar_merge_match_threshold"]

    use_bar = _should_use_bar_quantize(segments, beat_duration, beats_per_bar)
    segments = _snap_ml_segments_to_beats(segments, beat_times)
    if not use_bar:
        return segments

    segments = _quantize_ml_segments_to_bars(segments, beat_times, beats_per_bar)
    segments = _merge_ml_segments_within_bar(
        segments, beat_times, chroma, chroma_low, chroma_mid,
        frame_keys, sr, beat_duration, beats_per_bar, match_threshold,
    )
    return segments


def _ml_power_fraction(segments):
    if not segments:
        return 0.0
    return sum(1 for s in segments if str(s.get("chord", "")).endswith("5")) / len(segments)


def _looks_like_power_rock_ml(segments, chroma_power_ratio):
    """
    True when ML labels look like a 3-root power-chord loop (Baba O'Riley),
    not a triad song with minor harmony (Viva La Vida).
    """
    if chroma_power_ratio < 0.45:
        return False
    if _ml_power_fraction(segments) >= 0.10:
        return True

    total_dur = 0.0
    minor_dur = 0.0
    root_weight = Counter()
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        total_dur += dur
        root = _chord_root_name(chord)
        root_weight[root] += dur
        q = _chord_quality(chord)
        if q in ("m", "m7"):
            minor_dur += dur

    if total_dur <= 0:
        return False
    significant_roots = {
        root for root, weight in root_weight.items()
        if weight >= total_dur * 0.06
    }
    if len(significant_roots) > 4:
        return False
    if minor_dur / total_dur > 0.16:
        return False
    return True


def _maybe_power_pattern_timeline(
    y,
    sr,
    y_harmonic,
    chroma,
    chroma_low,
    chroma_mid,
    key_root,
    is_major,
    onset_times,
    beat_duration,
    ml_segments,
    refined_segments,
):
    """
    For power-chord rock loops, replace ML timeline with sheet-music I–I–V–V–IV pattern.

    Requires autochord to already label mostly power chords — avoids forcing the
    Baba O'Riley pattern onto orchestral or triad songs (e.g. Viva La Vida).
    """
    chroma_segments, bass_segments, mid_segments, beat_times = _beat_sync_chroma_stack(
        y_harmonic, chroma, chroma_low, chroma_mid, sr,
    )
    power_ratio = _estimate_power_song_ratio(chroma_segments)
    if power_ratio < 0.45:
        return None

    if not _looks_like_power_rock_ml(ml_segments, power_ratio):
        return None

    pattern_segments = _try_pattern_alignment(
        beat_times,
        chroma_segments,
        bass_segments,
        key_root,
        is_major,
        prefer_power=True,
        onset_times=onset_times,
    )
    return pattern_segments


def extract_chords_ml(y, sr):
    import autochord
    import soundfile as sf

    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(y, sr)
    chroma_mean = np.mean(chroma, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y, sr)

    try:
        raw = autochord.recognize(tmp_path)
    finally:
        os.unlink(tmp_path)

    segments = _autochord_to_segments(raw)
    if not segments:
        from analyzer import extract_chords as extract_chords_classic
        return extract_chords_classic(y, sr)

    segments = _merge_adjacent_same_root(segments)

    frame_keys, beat_times, chroma_segments, _, _ = _windowed_keys_from_chroma(
        chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )

    try:
        tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
    except AttributeError:
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    beat_duration = 60.0 / max(float(tempo), 1.0)

    segments = _smooth_ml_segment_boundaries(
        segments, beat_times, chroma, chroma_low, chroma_mid,
        frame_keys, sr, beat_duration,
    )
    segments = _merge_short_ml_segments(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr, beat_duration,
    )
    ml_segments = [dict(s) for s in segments]

    segments, _transpose = _align_segments_to_chroma(segments, chroma_mean)

    frame_keys, beat_times, chroma_segments, _, _ = _windowed_keys_from_chroma(
        chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    power_ratio = _estimate_power_song_ratio(chroma_segments)
    prefer_power = power_ratio > 0.48

    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )

    onset_times = librosa.frames_to_time(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True),
        sr=sr,
        hop_length=512,
    )
    pattern_segments = _maybe_power_pattern_timeline(
        y, sr, y_harmonic, chroma, chroma_low, chroma_mid,
        key_root, is_major, onset_times, beat_duration,
        ml_segments, segments,
    )
    if pattern_segments is not None:
        segments = pattern_segments
    else:
        segments = _polish_chord_timeline(
            segments, chroma, chroma_low, chroma_mid,
            frame_keys, beat_times, onset_times, beat_duration, sr,
            key_root=key_root, is_major=is_major,
            prefer_sevenths=(mode == "dorian"),
            prefer_power=prefer_power,
            ml_root_bias=1.02,
        )

    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )

    for seg in segments:
        seg["chord"] = _format_chord_name(seg["chord"], key_root, is_major)

    key_info = _build_key_info(
        key_root,
        is_major,
        mode=mode if mode not in ("major", "minor") else None,
    )

    return segments, key_info
