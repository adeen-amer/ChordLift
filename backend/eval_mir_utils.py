"""mir_eval chord + segmentation metrics (Phase 1 honest ruler)."""
from __future__ import annotations

import re
from pathlib import Path

import mir_eval.chord as mir_chord
import numpy as np

from eval_chord_utils import FLAT_TO_SHARP, normalize_chord_symbol

HARTE_ROOT = re.compile(r"^([A-G](?:#|b)?)")


def load_harte_lab(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load Isophonics/Harte .lab (start end label)."""
    intervals: list[list[float]] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        start, end, label = float(parts[0]), float(parts[1]), " ".join(parts[2:])
        if end <= start:
            continue
        intervals.append([start, end])
        labels.append(label)
    if not intervals:
        return np.zeros((0, 2), dtype=float), []
    return np.array(intervals, dtype=float), labels


def _sharp_root(root: str) -> str:
    return FLAT_TO_SHARP.get(root, root)


def chordlift_to_harte(chord: str) -> str:
    """Map ChordLift segment symbol to Harte/mir_eval label."""
    if not chord or chord == "N":
        return "N"
    if ":" in chord and HARTE_ROOT.match(chord):
        return chord

    from chord_label_utils import internal_to_harte_label

    return internal_to_harte_label(chord)


def segments_to_mir(
    segments: list[dict],
    *,
    duration: float | None = None,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert ChordLift segments to mir_eval intervals + Harte labels."""
    rows: list[list[float]] = []
    labels: list[str] = []
    for seg in segments:
        chord = seg.get("chord", "N")
        if not chord:
            continue
        start = float(seg.get("time", seg.get("start", 0)))
        end = float(seg.get("end_time", seg.get("end", start)))
        if eval_start is not None and end <= eval_start:
            continue
        if eval_end is not None and start >= eval_end:
            continue
        start = max(start, eval_start or start)
        end = min(end, eval_end if eval_end is not None else end)
        if end <= start:
            continue
        rows.append([start, end])
        labels.append(chordlift_to_harte(chord))

    if not rows:
        return np.zeros((0, 2), dtype=float), []

    intervals = np.array(rows, dtype=float)
    if duration is not None and intervals[-1, 1] < duration:
        if labels[-1] != "N":
            intervals = np.vstack([intervals, [[intervals[-1, 1], duration]]])
            labels.append(labels[-1])
    return intervals, labels


def clip_lab_to_window(
    intervals: np.ndarray,
    labels: list[str],
    *,
    eval_start: float | None = None,
    eval_end: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Clip reference lab intervals to an eval window."""
    if intervals.size == 0:
        return intervals, []
    out_i: list[list[float]] = []
    out_l: list[str] = []
    for (start, end), label in zip(intervals, labels):
        if eval_start is not None and end <= eval_start:
            continue
        if eval_end is not None and start >= eval_end:
            continue
        s = max(start, eval_start or start)
        e = min(end, eval_end if eval_end is not None else end)
        if e <= s:
            continue
        out_i.append([s, e])
        out_l.append(label)
    if not out_i:
        return np.zeros((0, 2), dtype=float), []
    return np.array(out_i, dtype=float), out_l


def evaluate_mir_chords(
    ref_intervals: np.ndarray,
    ref_labels: list[str],
    est_intervals: np.ndarray,
    est_labels: list[str],
) -> dict[str, float]:
    """
    mir_eval weighted CSR + segmentation (no offset search, no tolerant matching).

    Returns root, majmin, sevenths, seg scores in [0, 1].
    """
    if ref_intervals.size == 0 or est_intervals.size == 0:
        return {
            "root": 0.0,
            "majmin": 0.0,
            "sevenths": 0.0,
            "seg": 0.0,
        }

    scores = mir_chord.evaluate(
        ref_intervals,
        ref_labels,
        est_intervals,
        est_labels,
    )
    return {
        "root": float(scores["root"]),
        "majmin": float(scores["majmin"]),
        "sevenths": float(scores["sevenths"]),
        "seg": float(scores["seg"]),
    }
