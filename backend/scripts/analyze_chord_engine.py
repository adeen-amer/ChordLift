#!/usr/bin/env python3
"""
Benchmark the chord engine against stamped reference timelines.

Workflow:
  1. Reference stamps in chord_stamp_refs.json (or hand gold in chord_gold_labels.json)
  2. Run this script → per-track diff + aggregate error breakdown
  3. Fix the worst failure modes in analyzer / chord_engine_ml
  4. Re-run until metrics improve

Usage:
  python scripts/analyze_chord_engine.py
  python scripts/analyze_chord_engine.py --ids let-it-be,creep
  python scripts/analyze_chord_engine.py --write analysis/chord_engine_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from analyzer import ANALYZER_VERSION  # noqa: E402
from chord_engine import CHORD_ENGINE, extract_chords  # noqa: E402
from eval_chord_utils import (  # noqa: E402
    analyze_chord_prediction,
    filter_reference_changes,
    filter_reference_timeline,
)
from eval_fixture_utils import (  # noqa: E402
    DEFAULT_BENCHMARK,
    load_all_overlays,
    load_chord_stamps,
    load_gold_labels,
    prepare_eval_track,
)
from eval_split_utils import classify_track, load_benchmark_ids, load_held_out_ids, split_summary  # noqa: E402

FIXTURES = BACKEND / "tests" / "fixtures" / "chord_refs.json"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def _print_track_report(track_id: str, title: str, analysis: dict, ref_source: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{track_id} — {title or track_id}")
    print(f"Reference: {ref_source}")
    print(
        f"Timeline {_fmt_pct(analysis.get('timeline_score'))} | "
        f"Boundary {_fmt_pct(analysis.get('boundary_score'))} | "
        f"Symbol {_fmt_pct(analysis.get('symbol_recall'))} | "
        f"Switch MAE {analysis.get('switch_mae_sec', '—')}s"
    )
    bd = analysis.get("error_breakdown_pct", {})
    print(
        f"Time in song: match {bd.get('match', 0):.0f}% | "
        f"wrong quality {bd.get('quality_mismatch', 0):.0f}% | "
        f"wrong root {bd.get('root_mismatch', 0):.0f}%"
    )
    cc = analysis.get("change_counts", {})
    print(
        f"Changes: pred {cc.get('predicted')} vs ref {cc.get('reference')} | "
        f"missed {cc.get('missed_reference')} | spurious {cc.get('spurious_predicted')}"
    )

    if analysis.get("top_mismatches"):
        print("\nTop mismatch windows:")
        for w in analysis["top_mismatches"][:5]:
            print(
                f"  {w['start']:.1f}s–{w['end']:.1f}s ({w['duration']:.1f}s) "
                f"pred={w['predicted']} expected={w['expected']} [{w['error_kind']}]"
            )

    if analysis.get("missed_changes"):
        print("\nMissed reference changes (timing):")
        for m in analysis["missed_changes"][:5]:
            print(f"  @{m['time']:.1f}s → {m['chord']}")

    if analysis.get("spurious_changes"):
        print("\nSpurious predicted changes:")
        for s in analysis["spurious_changes"][:5]:
            print(f"  @{s['time']:.1f}s {s.get('from')} → {s['chord']}")


def analyze_track(track: dict, base_dir: Path) -> dict | None:
    audio_path = base_dir / track["file"]
    if not audio_path.exists():
        return {"skipped": True, "reason": f"missing {audio_path}"}

    ref_timeline = track.get("reference_timeline")
    ref_changes = track.get("reference_changes")
    if not ref_timeline or not ref_changes:
        return {"skipped": True, "reason": "no reference stamps"}

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr)
    segments, _key = extract_chords(y, sr)

    eval_start = track.get("boundary_eval_start")
    eval_end = track.get("boundary_eval_end")
    ref_timeline = filter_reference_timeline(
        ref_timeline, eval_start=eval_start, eval_end=eval_end,
    )
    ref_changes = filter_reference_changes(
        ref_changes, eval_start=eval_start, eval_end=eval_end,
    )

    analysis = analyze_chord_prediction(
        segments,
        ref_timeline,
        ref_changes,
        duration=duration,
        expected_symbols=track.get("expected_symbols"),
        tolerant=track.get("symbol_match_power_as_triad", True),
        tolerance_sec=track.get("boundary_tolerance", 0.5),
        eval_start=eval_start,
        eval_end=eval_end,
    )
    analysis["skipped"] = False
    analysis["predicted_segment_count"] = len(segments)
    analysis["reference_source"] = track.get("reference_source", "stamp")
    return analysis


def aggregate_reports(reports: list[dict]) -> dict:
    scored = [r for r in reports if not r.get("skipped")]
    if not scored:
        return {}

    def avg(key: str) -> float | None:
        vals = [r[key] for r in scored if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    root_pct = sum(
        r.get("error_breakdown_pct", {}).get("root_mismatch", 0) for r in scored
    ) / len(scored)
    quality_pct = sum(
        r.get("error_breakdown_pct", {}).get("quality_mismatch", 0) for r in scored
    ) / len(scored)

    by_timeline = sorted(scored, key=lambda r: r.get("timeline_score") or 0)
    return {
        "track_count": len(scored),
        "avg_timeline_score": avg("timeline_score"),
        "avg_boundary_score": avg("boundary_score"),
        "avg_symbol_recall": avg("symbol_recall"),
        "avg_root_mismatch_pct": round(root_pct, 1),
        "avg_quality_mismatch_pct": round(quality_pct, 1),
        "worst_timeline": [
            {"id": r["id"], "score": r.get("timeline_score")}
            for r in by_timeline[:5]
        ],
        "best_timeline": [
            {"id": r["id"], "score": r.get("timeline_score")}
            for r in by_timeline[-5:][::-1]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze chord engine vs reference stamps")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ids", help="Comma-separated track ids (default: chord_benchmark.json)")
    parser.add_argument("--write", type=Path, metavar="PATH", help="Write JSON report")
    parser.add_argument(
        "--split",
        choices=("benchmark", "held_out", "gold", "all"),
        default="all",
        help="Eval split to run (default: benchmark + held_out + gold)",
    )
    args = parser.parse_args()

    refs = json.loads(args.fixtures.read_text())
    overlays = load_all_overlays()
    stamps = load_chord_stamps()
    gold = load_gold_labels()

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
    elif args.split == "benchmark":
        wanted = load_benchmark_ids(args.benchmark)
    elif args.split == "held_out":
        wanted = load_held_out_ids()
    elif args.split == "gold":
        wanted = set(gold.keys())
    else:
        wanted = load_benchmark_ids(args.benchmark) | load_held_out_ids() | set(gold.keys())
        if not wanted:
            wanted = set(stamps.keys())

    tracks = [
        prepare_eval_track(t, overlays, stamps, gold)
        for t in refs["tracks"]
        if t["id"] in wanted
    ]

    print(f"Chord engine analysis | analyzer v{ANALYZER_VERSION} | engine={CHORD_ENGINE}")
    print(f"Tracks: {len(tracks)}\n")

    reports: list[dict] = []
    for track in tracks:
        tid = track["id"]
        result = analyze_track(track, BACKEND)
        if not result:
            continue
        result["id"] = tid
        result["title"] = track.get("title")
        result["eval_split"] = classify_track(tid)
        reports.append(result)

        if result.get("skipped"):
            print(f"SKIP {tid}: {result.get('reason')}")
            continue

        split_tag = result["eval_split"]
        _print_track_report(tid, track.get("title"), result, f"{result.get('reference_source', '?')} [{split_tag}]")

    split_stats = split_summary(reports)
    if split_stats:
        print(f"\n{'=' * 60}")
        print("SPLIT SUMMARY")
        for split, stats in split_stats.items():
            print(
                f"  {split:10s} n={stats['track_count']} "
                f"timeline={_fmt_pct(stats.get('avg_timeline_score'))} "
                f"boundary={_fmt_pct(stats.get('avg_boundary_score'))} "
                f"symbol={_fmt_pct(stats.get('avg_symbol_recall'))}"
            )

    summary = aggregate_reports(reports)
    if summary:
        print(f"\n{'=' * 60}")
        print("AGGREGATE (all analyzed tracks)")
        print(
            f"Timeline {_fmt_pct(summary.get('avg_timeline_score'))} | "
            f"Boundary {_fmt_pct(summary.get('avg_boundary_score'))} | "
            f"Symbol {_fmt_pct(summary.get('avg_symbol_recall'))}"
        )
        print(
            f"Avg time wrong root: {summary.get('avg_root_mismatch_pct', 0):.0f}% | "
            f"wrong quality: {summary.get('avg_quality_mismatch_pct', 0):.0f}%"
        )
        print("\nWorst timeline:")
        for row in summary.get("worst_timeline", []):
            print(f"  {row['id']}: {_fmt_pct(row.get('score'))}")

        print("\nSuggested focus:")
        if (summary.get("avg_quality_mismatch_pct") or 0) > (summary.get("avg_root_mismatch_pct") or 0):
            print("  → Quality/symbol polish (_third_quality_bias, label inertia, maj/min)")
        else:
            print("  → Root detection + boundaries (flux merge, cross_validate, ML root bias)")
        if summary.get("worst_timeline"):
            print(f"  → Start with: {summary['worst_timeline'][0]['id']}")

    if args.write:
        payload = {
            "analyzer_version": ANALYZER_VERSION,
            "chord_engine": CHORD_ENGINE,
            "summary": summary,
            "split_summary": split_stats,
            "tracks": reports,
        }
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nReport written to {args.write}")

    skipped = sum(1 for r in reports if r.get("skipped"))
    return 1 if skipped == len(reports) and reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
