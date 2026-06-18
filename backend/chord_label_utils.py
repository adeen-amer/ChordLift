"""Harte / autochord-style chord label ↔ ChordLift internal symbols."""
import re

from analyzer import _to_sharp_root

_HARTE_PAREN = re.compile(r"\([^)]*\)")
_HARTE_INV = re.compile(r"/[^/]+$")


def _harte_base_quality(quality: str) -> str:
    """Strip inversion bass and parenthetical extensions (Harte v1.2)."""
    q = (quality or "").strip().lower()
    q = _HARTE_PAREN.sub("", q)
    q = _HARTE_INV.sub("", q)
    return q.strip()


def harte_label_to_internal(label: str) -> str | None:
    """Map labels like F:maj, Bb:min7/b7, N to internal chord names."""
    label = (label or "").strip()
    if not label or label in ("N", "X", "no_chord", "None"):
        return None

    if ":" in label:
        root, quality = label.split(":", 1)
        root = _to_sharp_root(root.strip())
        quality = _harte_base_quality(quality)
        if quality in ("maj", "major", "", "1"):
            return root
        if quality in ("min", "minor"):
            return f"{root}m"
        if quality == "7":
            return f"{root}7"
        if quality == "maj7":
            return f"{root}maj7"
        if quality in ("min7", "min7b5"):
            return f"{root}m7"
        if quality == "hdim7":
            return f"{root}m7b5"
        if quality in ("dim",):
            return f"{root}dim"
        if quality in ("aug",):
            return f"{root}aug"
        if quality in ("sus2", "sus4"):
            return f"{root}{quality}"
        if quality in ("maj6", "min6", "min9", "maj9", "9", "11", "13"):
            if quality.startswith("min"):
                return f"{root}m{quality[3:]}" if len(quality) > 3 else f"{root}m"
            if quality.startswith("maj"):
                return f"{root}{quality}"
            return f"{root}{quality}"
        if quality == "5":
            return f"{root}5"
        return root

    # Bare root or slash bass, e.g. C, Bb, C/7, F/7
    if "/" in label:
        root = _to_sharp_root(label.split("/", 1)[0].strip())
        return root if root else None
    root = _to_sharp_root(label.strip())
    return root if root else None


def internal_to_harte_label(chord: str) -> str:
    """Map internal symbol back to Harte (mir_eval) label."""
    if not chord or chord == "N":
        return "N"

    from eval_chord_utils import normalize_chord_symbol

    norm = normalize_chord_symbol(chord)
    match = re.match(r"^([A-G](?:#)?)(.*)$", norm)
    if not match:
        return "N"
    root = match.group(1)
    suffix = match.group(2) or ""

    if suffix in ("", "maj"):
        return f"{root}:maj"
    if suffix == "m":
        return f"{root}:min"
    if suffix == "maj7":
        return f"{root}:maj7"
    if suffix == "m7":
        return f"{root}:min7"
    if suffix == "m7b5":
        return f"{root}:hdim7"
    if suffix == "7":
        return f"{root}:7"
    if suffix == "5":
        return f"{root}:5"
    if suffix in ("sus2", "sus4"):
        return f"{root}:{suffix}"
    if suffix in ("dim", "aug"):
        return f"{root}:{suffix}"
    if suffix in ("maj6", "maj9", "9", "11", "13"):
        return f"{root}:{suffix}"
    if suffix.startswith("m") and suffix[1:].isdigit():
        return f"{root}:min{suffix[1:]}"
    return f"{root}:maj"


def raw_labels_to_segments(entries, default_confidence=0.75):
    """
    Convert recognition output to ChordLift segment dicts.

    Accepts autochord tuples (start, end, label) or chordia dicts
    {start_time, end_time, chord}.
    """
    from chord_constants import PLACEHOLDER_STRUMMING

    segments = []
    for entry in entries:
        if isinstance(entry, dict):
            start = float(entry.get("start_time", entry.get("time", 0)))
            end = float(entry.get("end_time", start))
            label = entry.get("chord", "")
            confidence = float(entry.get("confidence", default_confidence))
        else:
            start, end, label = entry
            confidence = default_confidence

        internal = harte_label_to_internal(str(label))
        if internal is None:
            continue
        segments.append({
            "time": float(start),
            "end_time": float(end),
            "chord": internal,
            "confidence": confidence,
            "is_low_confidence": False,
            "is_power": internal.endswith("5"),
            "strumming": PLACEHOLDER_STRUMMING,
        })
    return segments
