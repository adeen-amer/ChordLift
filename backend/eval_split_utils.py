"""Train / benchmark / held-out splits for chord eval (avoid tuning on held-out)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

DEFAULT_BENCHMARK = (
    Path(__file__).parent / "tests" / "fixtures" / "chord_benchmark.json"
)
DEFAULT_HELD_OUT = (
    Path(__file__).parent / "tests" / "fixtures" / "chord_held_out.json"
)

EvalSplit = Literal["benchmark", "held_out", "tune", "gold"]


def load_split_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {t["id"] for t in data.get("tracks", []) if t.get("id")}


def load_benchmark_ids(path: Path | None = None) -> set[str]:
    return load_split_ids(path or DEFAULT_BENCHMARK)


def load_held_out_ids(path: Path | None = None) -> set[str]:
    return load_split_ids(path or DEFAULT_HELD_OUT)


def load_gold_ids(gold_path: Path | None = None) -> set[str]:
    from eval_fixture_utils import DEFAULT_GOLD_LABELS, load_gold_labels

    gold = load_gold_labels(gold_path or DEFAULT_GOLD_LABELS)
    return set(gold.keys())


def classify_track(
    track_id: str,
    *,
    benchmark_ids: set[str] | None = None,
    held_out_ids: set[str] | None = None,
    gold_ids: set[str] | None = None,
) -> EvalSplit:
    benchmark_ids = benchmark_ids if benchmark_ids is not None else load_benchmark_ids()
    held_out_ids = held_out_ids if held_out_ids is not None else load_held_out_ids()
    gold_ids = gold_ids if gold_ids is not None else load_gold_ids()

    if track_id in gold_ids:
        return "gold"
    if track_id in held_out_ids:
        return "held_out"
    if track_id in benchmark_ids:
        return "benchmark"
    return "tune"


def split_summary(results: list[dict], *, id_key: str = "id") -> dict[str, dict]:
    """Aggregate metric means by split for analyze/eval JSON rows."""
    buckets: dict[str, list[dict]] = {
        "benchmark": [],
        "held_out": [],
        "tune": [],
        "gold": [],
    }
    benchmark_ids = load_benchmark_ids()
    held_out_ids = load_held_out_ids()
    gold_ids = load_gold_ids()

    for row in results:
        if row.get("skipped"):
            continue
        tid = row.get(id_key, "")
        split = classify_track(
            tid,
            benchmark_ids=benchmark_ids,
            held_out_ids=held_out_ids,
            gold_ids=gold_ids,
        )
        buckets[split].append(row)

    summary: dict[str, dict] = {}
    for split, rows in buckets.items():
        if not rows:
            continue
        summary[split] = {
            "track_count": len(rows),
            "avg_timeline_score": _avg(rows, "timeline_score"),
            "avg_boundary_score": _avg(rows, "boundary_score"),
            "avg_symbol_recall": _avg(rows, "symbol_recall"),
            "avg_root_recall": _avg(rows, "root_recall"),
        }
    return summary


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None
