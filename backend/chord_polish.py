"""Chord timeline polish pipeline (extracted from analyzer)."""
import librosa

from chord_constants import CHORD_POLISH_DEFAULTS, PLACEHOLDER_STRUMMING

_analyzer = None


def _a():
    global _analyzer
    if _analyzer is None:
        import analyzer as _analyzer_mod
        _analyzer = _analyzer_mod
    return _analyzer


def _quality_bonus_for_song(prefer_sevenths=False, prefer_power=False):
    if prefer_sevenths:
        return {
            '': 1.04, 'm': 0.92, '5': 1.0,
            'sus4': 0.84, 'sus2': 0.84,
            '7': 0.90, 'maj7': 0.86, 'm7': 1.18,
        }
    if prefer_power:
        return {
            '': 1.0, 'm': 0.88, '5': 1.14,
            'sus4': 0.90, 'sus2': 0.90,
            '7': 0.94, 'maj7': 0.92, 'm7': 0.90,
        }
    return {
        '': 1.05, 'm': 1.05, '5': 1.02,
        'sus4': 0.86, 'sus2': 0.86,
        '7': 0.92, 'maj7': 0.90, 'm7': 0.92,
    }


def cross_validate_adjacent_segments(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    sr,
    hop=512,
    min_gain_ratio=None,
):
    """Swap neighbor chord labels when each fits the other's region better."""
    a = _a()
    if len(segments) < 2:
        return segments

    if min_gain_ratio is None:
        min_gain_ratio = CHORD_POLISH_DEFAULTS["cross_validate_min_gain"]

    out = [dict(s) for s in segments]
    for i in range(len(out) - 1):
        seg_a = out[i]
        seg_b = out[i + 1]
        if seg_a.get("chord") == seg_b.get("chord"):
            continue

        a_end = float(seg_b["time"])
        b_end = float(seg_b.get("end_time", a_end + 1.0))
        key_a = a._key_at_time((float(seg_a["time"]) + a_end) / 2.0, beat_times, frame_keys)
        key_b = a._key_at_time((a_end + b_end) / 2.0, beat_times, frame_keys)

        ca, ba, ma = a._segment_center_chroma(chroma, chroma_low, chroma_mid, float(seg_a["time"]), a_end, sr, hop)
        cb, cbass, cm = a._segment_center_chroma(chroma, chroma_low, chroma_mid, a_end, b_end, sr, hop)

        score_aa = a._match_score_for_chord(ca, ba, ma, seg_a["chord"], key_a[0], key_a[1], bass_weight=0.10)
        score_bb = a._match_score_for_chord(cb, cbass, cm, seg_b["chord"], key_b[0], key_b[1], bass_weight=0.10)
        score_ab = a._match_score_for_chord(ca, ba, ma, seg_b["chord"], key_a[0], key_a[1], bass_weight=0.10)
        score_ba = a._match_score_for_chord(cb, cbass, cm, seg_a["chord"], key_b[0], key_b[1], bass_weight=0.10)

        current = score_aa + score_bb
        swapped = score_ab + score_ba
        if current <= 0:
            continue
        if swapped > current * (1.0 + min_gain_ratio):
            seg_a["chord"], seg_b["chord"] = seg_b["chord"], seg_a["chord"]
            seg_a["confidence"], seg_b["confidence"] = seg_b.get("confidence", 0), seg_a.get("confidence", 0)

    return out


_cross_validate_adjacent_segments = cross_validate_adjacent_segments


def _bar_aware_merge_short_segments(
    segments,
    beat_duration,
    flux,
    sr,
    hop,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    flux_keep_threshold=None,
    match_keep_threshold=None,
    min_duration_override=None,
):
    a = _a()
    if len(segments) < 2:
        return segments

    if flux_keep_threshold is None:
        flux_keep_threshold = CHORD_POLISH_DEFAULTS["flux_keep_threshold"]
    if match_keep_threshold is None:
        match_keep_threshold = CHORD_POLISH_DEFAULTS["match_keep_threshold"]

    min_dur = (
        min_duration_override
        if min_duration_override is not None
        else a._bar_aware_min_duration(beat_duration)
    )
    merged = [dict(segments[0])]

    for seg in segments[1:]:
        prev = merged[-1]
        duration = float(seg["time"]) - float(prev["time"])
        if duration >= min_dur:
            merged.append(dict(seg))
            continue

        boundary_t = float(seg["time"])
        merge_end = float(seg.get("end_time", boundary_t + beat_duration))
        left_name = prev.get("chord", "N")
        right_name = seg.get("chord", "N")
        left_root = a._chord_root_name(left_name)
        right_root = a._chord_root_name(right_name)

        flux_val = a._flux_at_time(flux, boundary_t, sr, hop) if flux is not None else 0.0
        keep_split = False
        if (
            flux_val >= flux_keep_threshold
            and left_root not in ("N", "")
            and right_root not in ("N", "")
            and left_root != right_root
        ):
            if frame_keys is not None and beat_times is not None:
                key_l = a._key_at_time((float(prev["time"]) + boundary_t) / 2.0, beat_times, frame_keys)
                key_r = a._key_at_time((boundary_t + merge_end) / 2.0, beat_times, frame_keys)
            else:
                key_l = key_r = (0, True)

            lc, lb, lm = a._segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(prev["time"]), boundary_t, sr, hop,
            )
            rc, rb, rm = a._segment_center_chroma(
                chroma, chroma_low, chroma_mid, boundary_t, merge_end, sr, hop,
            )
            score_l = a._match_score_for_chord(lc, lb, lm, left_name, key_l[0], key_l[1])
            score_r = a._match_score_for_chord(rc, rb, rm, right_name, key_r[0], key_r[1])
            keep_split = score_l >= match_keep_threshold and score_r >= match_keep_threshold

        if keep_split:
            merged.append(dict(seg))
            continue

        if frame_keys is not None and beat_times is not None:
            key_m = a._key_at_time((float(prev["time"]) + merge_end) / 2.0, beat_times, frame_keys)
        else:
            key_m = (0, True)

        mc, mb, mm = a._segment_center_chroma(
            chroma, chroma_low, chroma_mid, float(prev["time"]), merge_end, sr, hop,
        )
        score_prev = a._match_score_for_chord(mc, mb, mm, left_name, key_m[0], key_m[1])
        score_next = a._match_score_for_chord(mc, mb, mm, right_name, key_m[0], key_m[1])
        prev["end_time"] = merge_end
        if score_next > score_prev:
            prev["chord"] = right_name
            prev["confidence"] = max(float(prev.get("confidence", 0)), float(seg.get("confidence", 0)))

    for i in range(len(merged) - 1):
        merged[i]["end_time"] = float(merged[i + 1]["time"])
    if merged:
        merged[-1]["end_time"] = float(segments[-1].get("end_time", merged[-1]["time"] + beat_duration))
    return merged


def _polish_segment_boundaries(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    onset_times,
    beat_duration,
    sr,
    hop=512,
):
    a = _a()
    if not segments:
        return segments, None

    flux = a._chroma_flux_series(chroma)
    flux_peak_times = librosa.frames_to_time(
        a._flux_peak_frames(flux, sr, hop, beat_duration),
        sr=sr,
        hop_length=hop,
    ).tolist()

    segments = a._optimize_segment_boundaries(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr,
        flux, flux_peak_times, onset_times, beat_duration, hop,
    )
    segments = a._snap_boundaries_to_harmonic_changes(
        segments, flux, flux_peak_times, onset_times, beat_duration, sr, hop,
    )
    return segments, flux


def _polish_segment_labels(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    key_root,
    is_major,
    prefer_sevenths=False,
    prefer_power=False,
    ml_root_bias=1.0,
    label_inertia=None,
):
    a = _a()
    preserve_ml = ml_root_bias > 1.0
    if label_inertia is None:
        label_inertia = (
            CHORD_POLISH_DEFAULTS["ml_label_inertia"]
            if preserve_ml
            else CHORD_POLISH_DEFAULTS["label_inertia"]
        )
    if not segments:
        return segments

    quality_bonus = _quality_bonus_for_song(prefer_sevenths, prefer_power)
    segments = a._refine_segments_with_vocabulary(
        segments, chroma, chroma_low, key_root, is_major,
        prefer_sevenths=prefer_sevenths,
        prefer_power=prefer_power,
        chroma_mid=chroma_mid,
        frame_keys=frame_keys,
        beat_times=beat_times,
        ml_root_bias=ml_root_bias,
        quality_bonus=quality_bonus,
        label_inertia=label_inertia,
    )
    segments = cross_validate_adjacent_segments(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, 22050, 512,
    )
    if not preserve_ml:
        segments = a._refine_segments_with_vocabulary(
            segments, chroma, chroma_low, key_root, is_major,
            prefer_sevenths=prefer_sevenths,
            prefer_power=prefer_power,
            chroma_mid=chroma_mid,
            frame_keys=frame_keys,
            beat_times=beat_times,
            ml_root_bias=1.0,
            quality_bonus=quality_bonus,
            label_inertia=label_inertia,
        )
    return segments


def polish_chord_timeline(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    onset_times,
    beat_duration,
    sr,
    hop=512,
    key_root=0,
    is_major=True,
    prefer_sevenths=False,
    prefer_power=False,
    ml_root_bias=1.0,
):
    """Audio-driven polish: boundaries first, then labels, then bar-aware merge."""
    a = _a()
    if not segments:
        return segments

    segments, flux = _polish_segment_boundaries(
        segments, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, onset_times, beat_duration, sr, hop,
    )
    segments = _polish_segment_labels(
        segments, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, key_root, is_major,
        prefer_sevenths=prefer_sevenths,
        prefer_power=prefer_power,
        ml_root_bias=ml_root_bias,
    )
    if flux is not None:
        segments = _bar_aware_merge_short_segments(
            segments, beat_duration, flux, sr, hop,
            chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        )
        segments = cross_validate_adjacent_segments(
            segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr, hop,
        )
        segments = _merge_flicker_segments(
            segments, beat_duration, flux, sr, hop,
            chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        )
    else:
        segments = a._merge_adjacent_same_root(segments)
    segments = apply_intro_guard(
        segments, beat_duration, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, sr, hop,
        key_root=key_root, is_major=is_major,
    )
    return a._finalize_timeline(segments, beat_duration)


_polish_chord_timeline = polish_chord_timeline


def _merge_flicker_segments(
    segments,
    beat_duration,
    flux,
    sr,
    hop,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
):
    a = _a()
    if len(segments) < 2:
        return segments

    segments = a._merge_adjacent_same_root(segments)
    min_dur = min(max(float(beat_duration) * CHORD_POLISH_DEFAULTS["flicker_min_beats"], 0.5), 1.35)
    return _bar_aware_merge_short_segments(
        segments, beat_duration, flux, sr, hop,
        chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        flux_keep_threshold=CHORD_POLISH_DEFAULTS["flux_keep_threshold_flicker"],
        match_keep_threshold=CHORD_POLISH_DEFAULTS["match_keep_threshold_flicker"],
        min_duration_override=min_dur,
    )


def apply_intro_guard(
    segments,
    beat_duration,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    sr,
    hop=512,
    key_root=0,
    is_major=True,
    stable_beats=None,
    max_intro_sec=None,
):
    """Collapse unstable early segment flips into one intro chord."""
    a = _a()
    if stable_beats is None:
        stable_beats = CHORD_POLISH_DEFAULTS["intro_stable_beats"]
    if max_intro_sec is None:
        max_intro_sec = CHORD_POLISH_DEFAULTS["intro_max_sec"]
    if len(segments) < 2:
        return segments

    intro_cap = float(segments[0]["time"]) + max_intro_sec
    flicker_end = 0
    for i in range(len(segments) - 1):
        seg_end = float(segments[i + 1]["time"])
        if seg_end > intro_cap:
            break
        duration = seg_end - float(segments[i]["time"])
        if duration >= float(beat_duration) * stable_beats:
            break
        flicker_end = i + 1

    if flicker_end == 0:
        return segments

    merge_start = float(segments[0]["time"])
    if flicker_end + 1 < len(segments):
        merge_end = float(segments[flicker_end + 1]["time"])
    else:
        merge_end = float(
            segments[flicker_end].get("end_time", segments[flicker_end]["time"] + beat_duration),
        )
    if merge_end - merge_start < float(beat_duration) * 0.75:
        return segments

    if frame_keys is not None and beat_times is not None and len(beat_times) > 0:
        seg_key_root, seg_major = a._key_at_time(
            (merge_start + merge_end) / 2.0, beat_times, frame_keys,
        )
    else:
        seg_key_root, seg_major = key_root, is_major

    best_name, confidence = a._pick_chord_for_region(
        chroma, chroma_low, chroma_mid,
        merge_start, merge_end, sr, hop,
        seg_key_root, seg_major,
        hint_root=a._chord_root_name(segments[flicker_end].get("chord", "N")),
    )

    merged = {
        "time": merge_start,
        "end_time": merge_end,
        "chord": a._format_chord_name(best_name, seg_key_root, seg_major),
        "confidence": confidence,
        "is_low_confidence": confidence < 0.35,
        "is_power": str(best_name).endswith("5"),
        "strumming": segments[0].get("strumming", PLACEHOLDER_STRUMMING),
    }
    tail = [dict(s) for s in segments[flicker_end + 1:]]
    return [merged] + tail


_apply_intro_guard = apply_intro_guard
