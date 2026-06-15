"""Merge eval fixture overlays into track definitions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ENRICHMENT = (
    Path(__file__).parent / "tests" / "fixtures" / "eval_enrichment.json"
)
DEFAULT_STAMP_REFS = (
    Path(__file__).parent / "tests" / "fixtures" / "chord_stamp_refs.json"
)
DEFAULT_GOLD_LABELS = (
    Path(__file__).parent / "tests" / "fixtures" / "chord_gold_labels.json"
)
DEFAULT_BENCHMARK = (
    Path(__file__).parent / "tests" / "fixtures" / "chord_benchmark.json"
)

STAMP_OVERLAY_KEYS = (
    "reference_changes",
    "reference_timeline",
    "boundary_method",
    "alignment",
)


def load_enrichment(path: Path | None = None) -> dict[str, dict[str, Any]]:
    enrich_path = path or DEFAULT_ENRICHMENT
    if not enrich_path.exists():
        return {}
    data = json.loads(enrich_path.read_text())
    return data.get("overlays", {})


def load_all_overlays(enrichment_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return merged eval enrichment overlays."""
    return load_enrichment(enrichment_path)


def merge_track_enrichment(
    track: dict[str, Any],
    overlays: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return track copy with enrichment fields filled in where missing."""
    overlays = overlays if overlays is not None else load_all_overlays()
    overlay = overlays.get(track.get("id", ""))
    if not overlay:
        return track

    merged = dict(track)
    for key, value in overlay.items():
        if key == "expected_symbols_power_ok":
            continue
        if key not in merged or merged[key] in (None, [], ""):
            merged[key] = value
        elif key == "expected_symbols":
            existing = set(merged.get("expected_symbols") or [])
            merged["expected_symbols"] = sorted(existing | set(value))
    if overlay.get("expected_symbols_power_ok"):
        merged["symbol_match_power_as_triad"] = True
    return merged


def merge_all_tracks(tracks: list[dict[str, Any]], overlays: dict | None = None) -> list[dict[str, Any]]:
    overlays = overlays if overlays is not None else load_all_overlays()
    return [merge_track_enrichment(t, overlays) for t in tracks]


def load_chord_stamps(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load precomputed chord change stamps (reference_changes + timeline)."""
    stamp_path = path or DEFAULT_STAMP_REFS
    if not stamp_path.exists():
        return {}
    data = json.loads(stamp_path.read_text())
    return data.get("tracks", {})


def load_gold_labels(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load hand-verified full-song chord timelines (override generated stamps)."""
    gold_path = path or DEFAULT_GOLD_LABELS
    if not gold_path.exists():
        return {}
    data = json.loads(gold_path.read_text())
    return data.get("tracks", {})


def load_benchmark_ids(path: Path | None = None) -> list[str]:
    """Return anchor track ids for the analyze-improve loop."""
    from eval_split_utils import load_benchmark_ids as _ids

    return sorted(_ids(path))


def apply_gold_labels(
    track: dict[str, Any],
    gold: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replace reference timeline with hand-verified gold labels when present."""
    if not gold:
        return track
    segments = gold.get("segments") or gold.get("reference_timeline")
    if not segments:
        return track

    merged = dict(track)
    merged["reference_timeline"] = segments
    changes: list[dict] = []
    for i, seg in enumerate(segments):
        if i == 0:
            continue
        prev = segments[i - 1].get("chord", "N")
        curr = seg.get("chord", "N")
        if prev != curr and curr != "N":
            changes.append({"time": float(seg["time"]), "chord": curr})
    if changes:
        merged["reference_changes"] = changes
    merged["boundary_method"] = gold.get("source", "gold")
    merged["reference_source"] = "gold"
    return merged


def apply_chord_stamps(
    track: dict[str, Any],
    stamp: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach cached reference_changes / reference_timeline from stamp overlay."""
    if not stamp:
        return track
    merged = dict(track)
    for key in STAMP_OVERLAY_KEYS:
        if key in stamp:
            merged[key] = stamp[key]
    return merged


def prepare_eval_track(
    track: dict[str, Any],
    overlays: dict[str, dict[str, Any]] | None = None,
    stamps: dict[str, dict[str, Any]] | None = None,
    gold: dict[str, dict[str, Any]] | None = None,
    *,
    live_boundaries: bool = False,
) -> dict[str, Any]:
    """
    Merge enrichment overlays, then gold labels, then chord stamp refs.
    """
    merged = merge_track_enrichment(track, overlays)
    if live_boundaries:
        for key in STAMP_OVERLAY_KEYS:
            merged.pop(key, None)
        merged.pop("reference_source", None)
        return merged

    gold_map = gold if gold is not None else load_gold_labels()
    merged = apply_gold_labels(merged, gold_map.get(merged.get("id", "")))
    if merged.get("reference_source") == "gold":
        return merged

    stamp_map = stamps if stamps is not None else load_chord_stamps()
    merged = apply_chord_stamps(merged, stamp_map.get(merged.get("id", "")))
    if merged.get("reference_changes"):
        merged.setdefault("reference_source", "stamp")
    return merged
