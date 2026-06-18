#!/usr/bin/env python3
"""Phase 1 gate: mir_eval majmin + sevenths WCSR vs Isophonics gold labs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent
FIXTURES = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
SPLIT_FILE = BACKEND / "tests" / "fixtures" / "gold_split_v2.json"
BASELINES = {
    "dev": BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v49.json",
    "test": BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v49.json",
}
MAJMIN_TOLERANCE = 0.015

sys.path.insert(0, str(BACKEND))

from analyzer import ANALYZER_VERSION  # noqa: E402
from chord_engine import CHORD_ENGINE, extract_chords  # noqa: E402
from engine_verify import assert_engine_honest, prepare_eval_environment  # noqa: E402
from eval_mir_utils import (  # noqa: E402
    clip_lab_to_window,
    evaluate_mir_chords,
    load_harte_lab,
    segments_to_mir,
)
from gold_audio_verify import approved_track_ids  # noqa: E402


def _avg(reports: list[dict], key: str) -> float | None:
    vals = [r[key] for r in reports if r.get(key) is not None and not r.get("skipped")]
    return round(sum(vals) / len(vals), 4) if vals else None


def evaluate_track(track: dict, *, eval_start: float | None, eval_end: float | None) -> dict:
    tid = track["id"]
    lab_path = BACKEND / track["lab"]
    audio_path = BACKEND / track["file"]
    if not lab_path.is_file():
        return {"id": tid, "skipped": True, "reason": f"missing lab {lab_path}"}
    if not audio_path.is_file():
        return {"id": tid, "skipped": True, "reason": f"missing audio {audio_path}"}

    ref_i, ref_l = load_harte_lab(lab_path)
    ref_i, ref_l = clip_lab_to_window(ref_i, ref_l, eval_start=eval_start, eval_end=eval_end)

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr)
    segments, key_info = extract_chords(y, sr)
    assert_engine_honest(key_info)

    est_i, est_l = segments_to_mir(
        segments, duration=duration, eval_start=eval_start, eval_end=eval_end,
    )
    scores = evaluate_mir_chords(ref_i, ref_l, est_i, est_l)
    return {
        "id": tid,
        "title": track.get("title"),
        "skipped": False,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "ref_segments": len(ref_l),
        "est_segments": len(est_l),
        **scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="mir_eval gold chord eval (Phase 1)")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--split", choices=("dev", "test"), help="DEV/TEST ids from gold_split_v2.json")
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--ids", help="Comma-separated track ids")
    parser.add_argument(
        "--require-audio-identity",
        action="store_true",
        help="Skip tracks not approved in gold_audio_identity.json",
    )
    parser.add_argument("--eval-start", type=float, default=None, help="Eval window start (sec)")
    parser.add_argument("--eval-end", type=float, default=None, help="Eval window end (sec)")
    parser.add_argument("--require-all", action="store_true", help="Fail if any track missing audio")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--write", type=Path, metavar="PATH")
    parser.add_argument("--check-baseline", action="store_true", help="Fail if below v49 split baseline")
    parser.add_argument("--update-baseline", action="store_true", help="Write split baseline JSON (v49)")
    args = parser.parse_args()

    prepare_eval_environment(no_cache=args.no_cache)
    catalog = json.loads(args.fixtures.read_text())
    tracks = catalog["tracks"]
    if args.split:
        split = json.loads(args.split_file.read_text())
        wanted = set(split[args.split])
        tracks = [t for t in tracks if t["id"] in wanted]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        tracks = [t for t in tracks if t["id"] in wanted]
    if args.require_audio_identity:
        approved = approved_track_ids()
        tracks = [t for t in tracks if t["id"] in approved]

    print(
        f"mir_eval gold | analyzer v{ANALYZER_VERSION} | engine={CHORD_ENGINE} | "
        f"model={os.getenv('CHORD_ML_MODEL', 'chordia')} | "
    )
    split_label = f" | split={args.split}" if args.split else ""
    print(
        f"Tracks: {len(tracks)} (Isophonics Harte labs, strict — no offset search{split_label})\n"
    )

    reports: list[dict] = []
    for track in tracks:
        result = evaluate_track(track, eval_start=args.eval_start, eval_end=args.eval_end)
        reports.append(result)
        if result.get("skipped"):
            print(f"SKIP {result['id']}: {result.get('reason')}")
            continue
        print(
            f"{result['id']:18s} root={result['root']:.3f} majmin={result['majmin']:.3f} "
            f"sevenths={result['sevenths']:.3f} seg={result['seg']:.3f} "
            f"(ref={result['ref_segments']} est={result['est_segments']})"
        )

    scored = [r for r in reports if not r.get("skipped")]
    skipped = [r for r in reports if r.get("skipped")]
    if args.require_all and skipped:
        print(f"\nERROR: {len(skipped)} track(s) missing lab/audio", file=sys.stderr)
        return 1
    if not scored:
        print("\nERROR: no tracks scored", file=sys.stderr)
        return 1

    summary = {
        "track_count": len(scored),
        "avg_root_wcsr": _avg(reports, "root"),
        "avg_majmin_wcsr": _avg(reports, "majmin"),
        "avg_sevenths_wcsr": _avg(reports, "sevenths"),
        "avg_seg_score": _avg(reports, "seg"),
    }
    print(f"\n{'=' * 60}")
    print(
        f"GOLD AGGREGATE n={summary['track_count']} | "
        f"root WCSR {summary['avg_root_wcsr']:.3f} | "
        f"majmin {summary['avg_majmin_wcsr']:.3f} | "
        f"sevenths {summary['avg_sevenths_wcsr']:.3f} | "
        f"seg {summary['avg_seg_score']:.3f}"
    )

    payload = {
        "analyzer_version": ANALYZER_VERSION,
        "chord_engine": CHORD_ENGINE,
        "chord_ml_model": os.getenv("CHORD_ML_MODEL", "chordia"),
        "summary": summary,
        "tracks": reports,
    }
    out = args.write or (BACKEND / "analysis" / "mir_gold_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nReport: {out}")

    if args.check_baseline:
        if not args.split:
            print("ERROR: --check-baseline requires --split dev|test", file=sys.stderr)
            return 1
        baseline_path = BASELINES[args.split]
        if not baseline_path.is_file():
            print(f"ERROR: missing baseline {baseline_path}", file=sys.stderr)
            return 1
        baseline = json.loads(baseline_path.read_text())
        b_mm = baseline["summary"]["avg_majmin_wcsr"]
        got_mm = summary["avg_majmin_wcsr"]
        if got_mm < b_mm - MAJMIN_TOLERANCE:
            print(
                f"FAIL: majmin {got_mm:.3f} < baseline {b_mm:.3f} - {MAJMIN_TOLERANCE}",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: majmin {got_mm:.3f} >= baseline {b_mm:.3f} - {MAJMIN_TOLERANCE}")

    if args.update_baseline and args.split:
        baseline_path = BASELINES[args.split]
        baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Baseline: {baseline_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
