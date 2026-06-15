"""Shared chord-symbol normalization and eval metrics."""
from __future__ import annotations

import re

FLAT_TO_SHARP = {"Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#"}
SHARP_TO_FLAT = {v: k for k, v in FLAT_TO_SHARP.items()}


def normalize_chord_symbol(chord: str) -> str:
    """Canonical symbol for eval comparison (sharps, no spaces)."""
    if not chord or chord == "N":
        return "N"

    chord = chord.strip()
    match = re.match(r"^([A-G](?:#|b)?)(.*)$", chord)
    if not match:
        return chord

    root = FLAT_TO_SHARP.get(match.group(1), match.group(1))
    suffix = match.group(2) or ""

    suffix = suffix.replace("min", "m").replace("maj", "")
    if suffix == "major":
        suffix = ""
    if suffix == "minor":
        suffix = "m"

    return f"{root}{suffix}"


def _expand_enharmonic_chords(chords: list[str]) -> set[str]:
    expanded = set()
    for chord in chords:
        norm = normalize_chord_symbol(chord)
        expanded.add(norm)
        match = re.match(r"^([A-G]#?)(.*)$", norm)
        if match and match.group(1) in SHARP_TO_FLAT:
            flat_root = SHARP_TO_FLAT[match.group(1)]
            expanded.add(f"{flat_root}{match.group(2)}")
    return expanded


def _chord_match_variants(chord: str, tolerant: bool = True) -> set[str]:
    """Expand a chord into equivalent symbols for tolerant eval matching."""
    norm = normalize_chord_symbol(chord)
    variants = set(_expand_enharmonic_chords([norm]))
    if not tolerant:
        return variants

    match = re.match(r"^([A-G]#?)(.*)$", norm)
    if not match:
        return variants

    root = match.group(1)
    suffix = match.group(2) or ""

    if suffix in ("", "5"):
        variants.add(root)
        variants.add(f"{root}5")
    if suffix in ("m", "m7"):
        variants.add(f"{root}m")
        variants.add(f"{root}m7")
    if suffix == "maj7":
        variants.add(root)
        variants.add(f"{root}maj7")
    if suffix == "7":
        variants.add(f"{root}7")
        variants.add(root)

    flat = SHARP_TO_FLAT.get(root)
    if flat:
        for v in list(variants):
            vm = re.match(r"^([A-G]#?)(.*)$", v)
            if vm and vm.group(1) == root:
                variants.add(f"{flat}{vm.group(2)}")

    return variants


def symbol_recall(
    predicted: list[str],
    expected: list[str],
    *,
    tolerant: bool = True,
) -> float:
    """Fraction of predicted symbols matching the expected set (ignoring N)."""
    predicted = [normalize_chord_symbol(c) for c in predicted if c and c != "N"]
    if not predicted:
        return 0.0

    expected_set: set[str] = set()
    for chord in expected:
        expected_set |= _chord_match_variants(chord, tolerant=tolerant)

    hits = 0
    for pred in predicted:
        pred_variants = _chord_match_variants(pred, tolerant=tolerant)
        if pred_variants & expected_set:
            hits += 1

    return hits / len(predicted)


def root_quality_match(predicted: str, expected: str) -> bool:
    """True when root and quality class match (triad vs m vs 7 vs 5)."""
    pred = normalize_chord_symbol(predicted)
    exp = normalize_chord_symbol(expected)

    def quality_class(sym: str) -> str:
        if sym.endswith("m7"):
            return "m7"
        if sym.endswith("maj7"):
            return "maj7"
        if sym.endswith("7"):
            return "7"
        if sym.endswith("5"):
            return "5"
        if sym.endswith("m"):
            return "m"
        if "sus" in sym:
            return "sus"
        return "triad"

    def root_of(sym: str) -> str:
        match = re.match(r"^([A-G]#?)", sym)
        return match.group(1) if match else sym

    return root_of(pred) == root_of(exp) and quality_class(pred) == quality_class(exp)


def _boundary_match_count(
    predicted_segments: list[dict],
    reference_changes: list[dict],
    tolerance_sec: float,
) -> tuple[int, int]:
    pred_changes = []
    for i, seg in enumerate(predicted_segments):
        if i == 0:
            continue
        prev = predicted_segments[i - 1].get("chord", "N")
        curr = seg.get("chord", "N")
        if prev != curr and curr != "N":
            pred_changes.append({"time": float(seg["time"]), "chord": curr})

    if not reference_changes:
        return 0, 0
    if not pred_changes:
        return 0, len(reference_changes)

    matched = 0
    for ref in reference_changes:
        ref_time = float(ref["time"])
        ref_chord = normalize_chord_symbol(ref.get("chord", ""))
        ref_root = re.match(r"^([A-G]#?)", ref_chord)
        for pc in pred_changes:
            if abs(pc["time"] - ref_time) > tolerance_sec:
                continue
            pred_norm = normalize_chord_symbol(pc["chord"])
            if pred_norm == ref_chord:
                matched += 1
                break
            pred_root = re.match(r"^([A-G]#?)", pred_norm)
            if ref_root and pred_root and pred_root.group(1) == ref_root.group(1):
                matched += 1
                break

    return matched, len(reference_changes)


def boundary_alignment_score(
    predicted_segments: list[dict],
    reference_changes: list[dict],
    tolerance_sec: float = 0.35,
    max_offset_sec: float = 2.0,
    offset_step_sec: float = 0.1,
) -> float | None:
    """
    Fraction of reference chord changes matched by a predicted change within tolerance.

    Searches a small timeline offset on reference changes to absorb beat-track drift.
    """
    if not reference_changes:
        return None

    best = 0.0
    offset = -max_offset_sec
    while offset <= max_offset_sec + 1e-9:
        shifted = [
            {"time": float(ref["time"]) + offset, "chord": ref["chord"]}
            for ref in reference_changes
        ]
        matched, total = _boundary_match_count(
            predicted_segments, shifted, tolerance_sec,
        )
        if total > 0:
            best = max(best, matched / total)
        offset += offset_step_sec

    return best


def chord_at_time(segments: list[dict], time_sec: float) -> str:
    """Return chord label active at time_sec, or N."""
    for seg in segments:
        start = float(seg.get("time", 0))
        end = float(seg.get("end_time", start + 1.0))
        if start <= time_sec < end:
            return seg.get("chord", "N") or "N"
    if segments:
        last = segments[-1]
        if time_sec >= float(last.get("time", 0)):
            return last.get("chord", "N") or "N"
    return "N"


def chords_match(pred: str, ref: str, *, tolerant: bool = True) -> bool:
    """True when predicted and reference symbols match (optionally tolerant)."""
    if not pred or pred == "N" or not ref or ref == "N":
        return False
    pred_variants = _chord_match_variants(normalize_chord_symbol(pred), tolerant=tolerant)
    ref_variants = _chord_match_variants(normalize_chord_symbol(ref), tolerant=tolerant)
    if pred_variants & ref_variants:
        return True
    pred_root = re.match(r"^([A-G]#?)", normalize_chord_symbol(pred))
    ref_root = re.match(r"^([A-G]#?)", normalize_chord_symbol(ref))
    return bool(
        tolerant
        and pred_root
        and ref_root
        and pred_root.group(1) == ref_root.group(1)
    )


def reference_timeline_from_progression(
    y,
    sr: int,
    progression: list[str],
    *,
    beats_per_chord: int = 4,
    cycles: int = 8,
    hop_length: int = 512,
    skip_beats: int = 0,
) -> list[dict]:
    """
    Build a full reference segment timeline from a looped progression + beat track.

    Used for duration-weighted timeline agreement (stronger than change-point-only).
    """
    changes = reference_changes_from_progression(
        y,
        sr,
        progression,
        beats_per_chord=beats_per_chord,
        cycles=cycles,
        hop_length=hop_length,
        skip_beats=skip_beats,
    )
    if not changes:
        return []

    duration = float(len(y) / sr)
    segments: list[dict] = []
    for i, change in enumerate(changes):
        start = float(change["time"])
        end = float(changes[i + 1]["time"]) if i + 1 < len(changes) else duration
        segments.append({
            "time": start,
            "end_time": end,
            "chord": change["chord"],
        })
    return segments


def timeline_agreement_score(
    predicted_segments: list[dict],
    reference_segments: list[dict],
    *,
    tolerant: bool = True,
    step_sec: float = 0.1,
    max_offset_sec: float = 2.0,
    offset_step_sec: float = 0.1,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> float | None:
    """
    Duration-weighted fraction of time where predicted chord matches reference.

    Searches ±max_offset_sec on the reference timeline to absorb beat-track drift.
    """
    if not predicted_segments or not reference_segments:
        return None

    pred_start = float(predicted_segments[0]["time"])
    pred_end = float(predicted_segments[-1].get(
        "end_time", predicted_segments[-1]["time"] + 1.0,
    ))
    ref_start = float(reference_segments[0]["time"])
    ref_end = float(reference_segments[-1].get(
        "end_time", reference_segments[-1]["time"] + 1.0,
    ))

    t_start = eval_start if eval_start is not None else max(pred_start, ref_start)
    t_end = eval_end if eval_end is not None else min(pred_end, ref_end)
    if t_end <= t_start + step_sec:
        return None

    best = 0.0
    offset = -max_offset_sec
    while offset <= max_offset_sec + 1e-9:
        matched = 0.0
        total = 0.0
        t = t_start
        while t < t_end:
            pred = chord_at_time(predicted_segments, t)
            ref = chord_at_time(reference_segments, t + offset)
            total += step_sec
            if chords_match(pred, ref, tolerant=tolerant):
                matched += step_sec
            t += step_sec
        if total > 0:
            best = max(best, matched / total)
        offset += offset_step_sec

    return best


def switch_timing_mae(
    predicted_segments: list[dict],
    reference_changes: list[dict],
    *,
    tolerance_sec: float = 0.5,
    max_offset_sec: float = 2.0,
    offset_step_sec: float = 0.1,
) -> float | None:
    """
    Mean absolute error (seconds) for matched chord-change times after offset search.

    Lower is better; None when no reference changes or no predicted changes.
    """
    if not reference_changes:
        return None

    pred_changes = []
    for i, seg in enumerate(predicted_segments):
        if i == 0:
            continue
        prev = predicted_segments[i - 1].get("chord", "N")
        curr = seg.get("chord", "N")
        if prev != curr and curr != "N":
            pred_changes.append({"time": float(seg["time"]), "chord": curr})

    if not pred_changes:
        return None

    best_mae = None
    offset = -max_offset_sec
    while offset <= max_offset_sec + 1e-9:
        errors: list[float] = []
        used_pred: set[int] = set()
        for ref in reference_changes:
            ref_time = float(ref["time"]) + offset
            ref_chord = ref.get("chord", "")
            best_idx = None
            best_dt = None
            for j, pc in enumerate(pred_changes):
                if j in used_pred:
                    continue
                dt = abs(pc["time"] - ref_time)
                if dt > tolerance_sec:
                    continue
                if not chords_match(pc["chord"], ref_chord, tolerant=True):
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = j
            if best_idx is not None and best_dt is not None:
                errors.append(best_dt)
                used_pred.add(best_idx)
        if errors:
            mae = sum(errors) / len(errors)
            if best_mae is None or mae < best_mae:
                best_mae = mae
        offset += offset_step_sec

    return best_mae


def segments_with_end_times(
    segments: list[dict],
    duration: float | None = None,
) -> list[dict]:
    """Return segments with end_time filled from the next segment or duration."""
    if not segments:
        return []

    out: list[dict] = []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        start = float(seg.get("time", 0))
        if seg.get("end_time") is not None:
            seg["end_time"] = float(seg["end_time"])
        elif i + 1 < len(segments):
            seg["end_time"] = float(segments[i + 1]["time"])
        elif duration is not None:
            seg["end_time"] = float(duration)
        else:
            seg["end_time"] = start + 1.0
        out.append(seg)
    return out


def _chord_root(chord: str) -> str | None:
    match = re.match(r"^([A-G]#?)", normalize_chord_symbol(chord))
    return match.group(1) if match else None


def _error_kind(predicted: str, expected: str, *, tolerant: bool) -> str:
    if chords_match(predicted, expected, tolerant=tolerant):
        return "match"
    pred_root = _chord_root(predicted)
    exp_root = _chord_root(expected)
    if pred_root and exp_root and pred_root == exp_root:
        return "quality_mismatch"
    return "root_mismatch"


def timeline_mismatch_windows(
    predicted_segments: list[dict],
    reference_segments: list[dict],
    *,
    tolerant: bool = True,
    step_sec: float = 0.25,
    min_window_sec: float = 0.75,
    eval_start: float | None = None,
    eval_end: float | None = None,
    max_offset_sec: float = 2.0,
    offset_step_sec: float = 0.1,
) -> tuple[list[dict], float]:
    """
    Find contiguous windows where predicted chord != reference.

    Returns (windows, best_timeline_offset_sec).
    """
    if not predicted_segments or not reference_segments:
        return [], 0.0

    pred_start = float(predicted_segments[0]["time"])
    pred_end = float(predicted_segments[-1]["end_time"])
    ref_start = float(reference_segments[0]["time"])
    ref_end = float(reference_segments[-1]["end_time"])
    t_start = eval_start if eval_start is not None else max(pred_start, ref_start)
    t_end = eval_end if eval_end is not None else min(pred_end, ref_end)
    if t_end <= t_start + step_sec:
        return [], 0.0

    best_offset = 0.0
    best_score = -1.0
    offset = -max_offset_sec
    while offset <= max_offset_sec + 1e-9:
        matched = 0.0
        total = 0.0
        t = t_start
        while t < t_end:
            total += step_sec
            if chords_match(
                chord_at_time(predicted_segments, t),
                chord_at_time(reference_segments, t + offset),
                tolerant=tolerant,
            ):
                matched += step_sec
            t += step_sec
        score = matched / total if total else 0.0
        if score > best_score:
            best_score = score
            best_offset = offset
        offset += offset_step_sec

    windows: list[dict] = []
    current: dict | None = None
    t = t_start
    while t < t_end:
        pred = chord_at_time(predicted_segments, t)
        ref = chord_at_time(reference_segments, t + best_offset)
        kind = _error_kind(pred, ref, tolerant=tolerant)
        if kind != "match":
            if current and current["error_kind"] == kind and abs(t - current["end"]) < step_sec * 1.5:
                current["end"] = t + step_sec
                current["duration"] = round(current["end"] - current["start"], 2)
            else:
                if current and current["duration"] >= min_window_sec:
                    windows.append(current)
                current = {
                    "start": round(t, 2),
                    "end": round(t + step_sec, 2),
                    "duration": round(step_sec, 2),
                    "predicted": pred,
                    "expected": ref,
                    "error_kind": kind,
                }
        elif current:
            if current["duration"] >= min_window_sec:
                windows.append(current)
            current = None
        t += step_sec

    if current and current["duration"] >= min_window_sec:
        windows.append(current)

    windows.sort(key=lambda w: w["duration"], reverse=True)
    return windows, best_offset


def _collect_changes(segments: list[dict]) -> list[dict]:
    changes: list[dict] = []
    for i, seg in enumerate(segments):
        if i == 0:
            continue
        prev = segments[i - 1].get("chord", "N")
        curr = seg.get("chord", "N")
        if prev != curr and curr != "N":
            changes.append({"time": float(seg["time"]), "chord": curr, "from": prev})
    return changes


def unmatched_reference_changes(
    predicted_segments: list[dict],
    reference_changes: list[dict],
    *,
    tolerance_sec: float = 0.5,
    offset_sec: float = 0.0,
    tolerant: bool = True,
) -> list[dict]:
    """Reference changes with no matching predicted change within tolerance."""
    pred_changes = _collect_changes(predicted_segments)
    missed: list[dict] = []
    for ref in reference_changes:
        ref_time = float(ref["time"]) + offset_sec
        ref_chord = ref.get("chord", "")
        found = False
        for pc in pred_changes:
            if abs(pc["time"] - ref_time) > tolerance_sec:
                continue
            if chords_match(pc["chord"], ref_chord, tolerant=tolerant):
                found = True
                break
        if not found:
            missed.append({
                "time": round(float(ref["time"]), 2),
                "chord": ref_chord,
            })
    return missed


def spurious_predicted_changes(
    predicted_segments: list[dict],
    reference_changes: list[dict],
    *,
    tolerance_sec: float = 0.5,
    offset_sec: float = 0.0,
    tolerant: bool = True,
) -> list[dict]:
    """Predicted changes with no matching reference change within tolerance."""
    pred_changes = _collect_changes(predicted_segments)
    extra: list[dict] = []
    for pc in pred_changes:
        found = False
        for ref in reference_changes:
            ref_time = float(ref["time"]) + offset_sec
            if abs(pc["time"] - ref_time) > tolerance_sec:
                continue
            if chords_match(pc["chord"], ref.get("chord", ""), tolerant=tolerant):
                found = True
                break
        if not found:
            extra.append({
                "time": round(pc["time"], 2),
                "chord": pc["chord"],
                "from": pc.get("from"),
            })
    return extra


def analyze_chord_prediction(
    predicted_segments: list[dict],
    reference_timeline: list[dict],
    reference_changes: list[dict],
    *,
    duration: float | None = None,
    expected_symbols: list[str] | None = None,
    tolerant: bool = True,
    tolerance_sec: float = 0.5,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> dict:
    """
    Structured diff between engine output and reference chord stamps.

    Used by scripts/analyze_chord_engine.py for the benchmark-improve loop.
    """
    predicted = segments_with_end_times(predicted_segments, duration)
    reference = segments_with_end_times(reference_timeline, duration)

    timeline = timeline_agreement_score(
        predicted,
        reference,
        tolerant=tolerant,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    boundary = boundary_alignment_score(
        predicted,
        reference_changes,
        tolerance_sec=tolerance_sec,
    )
    switch_mae = switch_timing_mae(
        predicted,
        reference_changes,
        tolerance_sec=tolerance_sec,
    )

    windows, ref_offset = timeline_mismatch_windows(
        predicted,
        reference,
        tolerant=tolerant,
        eval_start=eval_start,
        eval_end=eval_end,
    )

    missed = unmatched_reference_changes(
        predicted,
        reference_changes,
        tolerance_sec=tolerance_sec,
        offset_sec=ref_offset,
        tolerant=tolerant,
    )
    spurious = spurious_predicted_changes(
        predicted,
        reference_changes,
        tolerance_sec=tolerance_sec,
        offset_sec=ref_offset,
        tolerant=tolerant,
    )

    error_seconds = {"match": 0.0, "quality_mismatch": 0.0, "root_mismatch": 0.0}
    step = 0.25
    t_start = eval_start if eval_start is not None else max(
        float(predicted[0]["time"]) if predicted else 0.0,
        float(reference[0]["time"]) if reference else 0.0,
    )
    t_end = eval_end if eval_end is not None else min(
        float(predicted[-1]["end_time"]) if predicted else 0.0,
        float(reference[-1]["end_time"]) if reference else 0.0,
    )
    t = t_start
    while t < t_end:
        kind = _error_kind(
            chord_at_time(predicted, t),
            chord_at_time(reference, t + ref_offset),
            tolerant=tolerant,
        )
        error_seconds[kind] += step
        t += step
    total_sec = sum(error_seconds.values()) or 1.0

    sym_recall = None
    if expected_symbols:
        symbols = [
            normalize_chord_symbol(seg.get("chord", ""))
            for seg in predicted
            if seg.get("chord") and seg.get("chord") != "N"
        ]
        sym_recall = symbol_recall(symbols, expected_symbols, tolerant=tolerant)

    quality_windows = [w for w in windows if w["error_kind"] == "quality_mismatch"]
    root_windows = [w for w in windows if w["error_kind"] == "root_mismatch"]

    return {
        "timeline_score": round(timeline, 3) if timeline is not None else None,
        "boundary_score": round(boundary, 3) if boundary is not None else None,
        "switch_mae_sec": round(switch_mae, 3) if switch_mae is not None else None,
        "symbol_recall": round(sym_recall, 3) if sym_recall is not None else None,
        "reference_offset_sec": round(ref_offset, 2),
        "eval_window": {"start": eval_start, "end": eval_end},
        "segment_counts": {
            "predicted": len(predicted),
            "reference": len(reference),
        },
        "change_counts": {
            "predicted": len(_collect_changes(predicted)),
            "reference": len(reference_changes),
            "missed_reference": len(missed),
            "spurious_predicted": len(spurious),
        },
        "error_breakdown_sec": {
            k: round(v, 1) for k, v in error_seconds.items()
        },
        "error_breakdown_pct": {
            k: round(100.0 * v / total_sec, 1) for k, v in error_seconds.items()
        },
        "top_mismatches": windows[:8],
        "top_quality_errors": quality_windows[:5],
        "top_root_errors": root_windows[:5],
        "missed_changes": missed[:12],
        "spurious_changes": spurious[:12],
    }


def reference_changes_from_progression(
    y,
    sr: int,
    progression: list[str],
    *,
    beats_per_chord: int = 4,
    cycles: int = 8,
    hop_length: int = 512,
    skip_beats: int = 0,
) -> list[dict]:
    """
    Build approximate reference chord-change times from a known loop + beat track.

    Used when Hooktheory gives a progression but not absolute timestamps.
    """
    if not progression or len(y) < sr // 8:
        return []

    import librosa

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames",
    )
    beat_frames = librosa.util.fix_frames(
        beat_frames, x_min=0, x_max=max(1, len(y) // hop_length - 1),
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    if len(beat_times) <= skip_beats:
        return []

    changes: list[dict] = []
    total_changes = cycles * len(progression)
    for change_idx in range(total_changes):
        beat_idx = skip_beats + change_idx * beats_per_chord
        if beat_idx >= len(beat_times):
            break
        chord = progression[change_idx % len(progression)]
        changes.append({"time": float(beat_times[beat_idx]), "chord": chord})

    return changes


def filter_reference_changes(
    reference_changes: list[dict],
    *,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> list[dict]:
    """Keep reference change points inside an eval window."""
    if not reference_changes:
        return reference_changes
    out = []
    for ref in reference_changes:
        t = float(ref["time"])
        if eval_start is not None and t < eval_start:
            continue
        if eval_end is not None and t > eval_end:
            continue
        out.append(ref)
    return out


def filter_reference_timeline(
    reference_timeline: list[dict],
    *,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> list[dict]:
    """Clip reference segments to an eval window."""
    if not reference_timeline:
        return reference_timeline

    duration = float(reference_timeline[-1].get(
        "end_time", reference_timeline[-1]["time"] + 1.0,
    ))
    start = eval_start if eval_start is not None else float(reference_timeline[0]["time"])
    end = eval_end if eval_end is not None else duration

    out: list[dict] = []
    for seg in reference_timeline:
        t0 = float(seg["time"])
        t1 = float(seg.get("end_time", t0 + 1.0))
        if t1 <= start or t0 >= end:
            continue
        out.append({
            "time": max(t0, start),
            "end_time": min(t1, end),
            "chord": seg["chord"],
        })
    return out


def reference_changes_from_aligned_progression(
    y,
    sr: int,
    progression: list[str],
    *,
    beats_per_chord: int = 4,
    cycles: int = 8,
    skip_beats: int = 0,
    max_skip_beats: int | None = None,
    hop_length: int = 512,
    snap_flux: bool = True,
    flux_search_sec: float = 0.28,
) -> tuple[list[dict], list[dict], dict]:
    """
    Build reference change times by phase-aligning a known progression to audio.

    Searches start beat + cycle phase via chroma template scores, then snaps
    each internal boundary to the nearest chroma-flux peak. Returns
    (reference_changes, reference_timeline, alignment_meta).
    """
    if not progression or len(y) < sr // 8:
        return [], [], {}

    import librosa
    import numpy as np

    from analyzer import (
        _chroma_flux_series,
        _extract_chroma_stack,
        _flux_at_time,
        _flux_peak_frames,
        _match_score_for_chord,
        _segment_center_chroma,
    )

    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(y, sr, hop_length)
    duration = float(len(y) / sr)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y_harmonic, sr=sr, hop_length=hop_length, units="frames",
    )
    beat_frames = librosa.util.fix_frames(
        beat_frames, x_min=0, x_max=max(1, chroma.shape[1] - 1),
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if len(beat_times) < beats_per_chord + 2:
        return [], [], {}

    beat_duration = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    slot_duration = beat_duration * beats_per_chord

    if max_skip_beats is None:
        max_skip_beats = min(len(beat_times) - beats_per_chord, max(64, beats_per_chord * 8))

    best_score = -1.0
    best_skip = skip_beats
    best_phase = 0

    for try_skip in range(skip_beats, max_skip_beats + 1, max(1, beats_per_chord)):
        for phase in range(len(progression)):
            total = 0.0
            count = 0
            for i in range(cycles * len(progression)):
                beat_idx = try_skip + i * beats_per_chord
                if beat_idx >= len(beat_times):
                    break
                t0 = float(beat_times[beat_idx])
                if i + 1 < cycles * len(progression):
                    next_idx = try_skip + (i + 1) * beats_per_chord
                    t1 = float(beat_times[next_idx]) if next_idx < len(beat_times) else duration
                else:
                    t1 = min(duration, t0 + slot_duration)
                if t1 <= t0 + 0.05:
                    continue
                chord = progression[(i + phase) % len(progression)]
                c, b, m = _segment_center_chroma(
                    chroma, chroma_low, chroma_mid, t0, t1, sr, hop_length, trim=0.12,
                )
                total += _match_score_for_chord(c, b, m, chord, 0, True)
                count += 1
            if count == 0:
                continue
            avg = total / count
            if avg > best_score:
                best_score = avg
                best_skip = try_skip
                best_phase = phase

    flux = _chroma_flux_series(chroma) if snap_flux else None
    flux_peak_times = []
    if flux is not None:
        flux_peak_times = librosa.frames_to_time(
            _flux_peak_frames(flux, sr, hop_length, beat_duration),
            sr=sr,
            hop_length=hop_length,
        ).tolist()

    changes: list[dict] = []
    for i in range(cycles * len(progression)):
        beat_idx = best_skip + i * beats_per_chord
        if beat_idx >= len(beat_times):
            break
        t = float(beat_times[beat_idx])
        if i > 0 and snap_flux and flux is not None:
            search = min(max(flux_search_sec, beat_duration * 0.25), 0.42)
            candidates = [t]
            for pt in flux_peak_times:
                if abs(pt - t) <= search:
                    candidates.append(float(pt))
            best_t = t
            best_flux = _flux_at_time(flux, t, sr, hop_length)
            for cand in candidates:
                val = _flux_at_time(flux, cand, sr, hop_length)
                if val > best_flux:
                    best_flux = val
                    best_t = cand
            t = best_t
        changes.append({
            "time": t,
            "chord": progression[(i + best_phase) % len(progression)],
        })

    if not changes:
        return [], [], {}

    timeline: list[dict] = []
    for i, change in enumerate(changes):
        start = float(change["time"])
        end = float(changes[i + 1]["time"]) if i + 1 < len(changes) else duration
        timeline.append({
            "time": start,
            "end_time": end,
            "chord": change["chord"],
        })

    meta = {
        "alignment_score": round(best_score, 4),
        "skip_beats": best_skip,
        "phase": best_phase,
        "beats_per_chord": beats_per_chord,
        "slot_duration_sec": round(slot_duration, 3),
    }
    return changes, timeline, meta


def resolve_boundary_references(
    y,
    sr: int,
    track: dict,
) -> tuple[list[dict], list[dict], str | None]:
    """
    Build boundary/timeline references for eval from audio + track config.

    Uses phase-aligned progression by default (chroma + flux on the recording).
    Set ``boundary_method: beat_loop`` to use the legacy beat-index fallback.
    """
    progression = track.get("progression")
    if not progression:
        return [], [], None

    method = track.get("boundary_method", "phase_aligned")
    beats = track.get("beats_per_chord", 4)
    cycles = track.get("boundary_cycles", 8)
    skip = track.get("boundary_skip_beats", 0)

    if method == "beat_loop":
        changes = reference_changes_from_progression(
            y, sr, progression,
            beats_per_chord=beats,
            cycles=cycles,
            skip_beats=skip,
        )
        timeline = reference_timeline_from_progression(
            y, sr, progression,
            beats_per_chord=beats,
            cycles=cycles,
            skip_beats=skip,
        )
        return changes, timeline, "beat_loop"

    changes, timeline, _meta = reference_changes_from_aligned_progression(
        y, sr, progression,
        beats_per_chord=beats,
        cycles=cycles,
        skip_beats=skip,
    )
    return changes, timeline, "phase_aligned"
