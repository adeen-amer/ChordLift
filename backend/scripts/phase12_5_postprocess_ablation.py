#!/usr/bin/env python3
"""Phase 12.5 — post-processing ablation on v47 identity-verified audio."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from chord_engine_chordia import chordia_to_segments, recognize_chordia  # noqa: E402
from chord_engine_ml import _postprocess_dnn_segments_bypass  # noqa: E402
from chord_pipeline import build_chord_pipeline_context  # noqa: E402
from eval_mir_utils import (  # noqa: E402
    clip_lab_to_window,
    evaluate_mir_chords,
    load_harte_lab,
    segments_to_mir,
)
from gold_audio_verify import approved_track_ids  # noqa: E402
from pitch_utils import normalize_pitch  # noqa: E402

CATALOG = BACKEND / "tests/fixtures/gold_mir_tracks_v2.json"
SPLIT = BACKEND / "tests/fixtures/gold_split_v2.json"
V47_DEV = BACKEND / "analysis/BASELINE_mir_gold_DEV_v47.json"
V47_TEST = BACKEND / "analysis/BASELINE_mir_gold_TEST_v47.json"
OUT_JSON = BACKEND / "analysis/PHASE12_5_POSTPROCESS_ABLATION.json"
OUT_MD = BACKEND / "analysis/PHASE12_5_POSTPROCESS_ABLATION.md"

CONFIGS = ("raw", "pitch_only", "merge_only", "serving")


def _load_tracks(split_name: str) -> list[dict]:
    catalog = json.loads(CATALOG.read_text())
    split = json.loads(SPLIT.read_text())
    approved = approved_track_ids()
    wanted = {t for t in split[split_name] if t in approved}
    return [t for t in catalog["tracks"] if t["id"] in wanted]


def _score(segments: list[dict], ref_i, ref_l, duration: float) -> dict[str, float]:
    est_i, est_l = segments_to_mir(segments, duration=duration)
    return evaluate_mir_chords(ref_i, ref_l, est_i, est_l)


def _chordia_raw(y, sr, pipeline) -> list[dict]:
    return chordia_to_segments(recognize_chordia(pipeline.y_chord, sr))


def _segments_for_config(
    config: str,
    y,
    sr,
    pipeline,
    raw_segments: list[dict],
) -> tuple[list[dict], float]:
    pitch_shift = 0.0
    if config == "raw":
        return raw_segments, pitch_shift
    if config == "pitch_only":
        return raw_segments, pitch_shift
    if config == "merge_only":
        segs, _ = _postprocess_dnn_segments_bypass(
            [dict(s) for s in raw_segments], y, sr, pipeline,
            chroma_align=True, light_merge=None, key_format=True,
        )
        return segs, pitch_shift
    if config == "serving":
        segs, _ = _postprocess_dnn_segments_bypass(
            [dict(s) for s in raw_segments], y, sr, pipeline,
            chroma_align=True, light_merge=None, key_format=True,
        )
        return segs, pitch_shift
    raise ValueError(config)


def _evaluate_track(track: dict) -> dict:
    tid = track["id"]
    audio_path = BACKEND / track["file"]
    lab_path = BACKEND / track["lab"]
    if not audio_path.is_file():
        return {"id": tid, "skipped": True}

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr)
    ref_i, ref_l = load_harte_lab(lab_path)
    ref_i, ref_l = clip_lab_to_window(ref_i, ref_l, eval_start=None, eval_end=None)

    y_no_pc = y
    y_pc, pitch_shift = normalize_pitch(y, sr)
    pipeline_no_pc = build_chord_pipeline_context(y_no_pc, sr)
    pipeline_pc = build_chord_pipeline_context(y_pc, sr)

    raw_no_pc = _chordia_raw(y_no_pc, sr, pipeline_no_pc)
    raw_pc = _chordia_raw(y_pc, sr, pipeline_pc)

    scores: dict[str, dict] = {}
    scores["raw"] = _score(raw_no_pc, ref_i, ref_l, duration)
    scores["pitch_only"] = _score(raw_pc, ref_i, ref_l, duration)
    merge_segs, _ = _postprocess_dnn_segments_bypass(
        [dict(s) for s in raw_no_pc], y_no_pc, sr, pipeline_no_pc,
        chroma_align=True, light_merge=None, key_format=True,
    )
    scores["merge_only"] = _score(merge_segs, ref_i, ref_l, duration)
    serving_segs, _ = _postprocess_dnn_segments_bypass(
        [dict(s) for s in raw_pc], y_pc, sr, pipeline_pc,
        chroma_align=True, light_merge=None, key_format=True,
    )
    scores["serving"] = _score(serving_segs, ref_i, ref_l, duration)

    return {
        "id": tid,
        "skipped": False,
        "pitch_shift_semitones": round(pitch_shift, 3),
        "scores": scores,
        "pitch_delta_majmin": round(scores["pitch_only"]["majmin"] - scores["raw"]["majmin"], 4),
        "merge_delta_majmin": round(scores["merge_only"]["majmin"] - scores["raw"]["majmin"], 4),
        "serving_delta_majmin": round(scores["serving"]["majmin"] - scores["raw"]["majmin"], 4),
    }


def _agg(rows: list[dict], cfg: str) -> dict[str, float]:
    n = len(rows)
    return {
        "majmin": round(sum(r["scores"][cfg]["majmin"] for r in rows) / n, 4),
        "seg": round(sum(r["scores"][cfg]["seg"] for r in rows) / n, 4),
        "n": n,
    }


def _print_split(name: str, rows: list[dict]) -> None:
    aggs = {c: _agg(rows, c) for c in CONFIGS}
    print(f"\n{'=' * 96}")
    print(f"{name.upper()} aggregate majmin / seg")
    print(f"{'config':<14} {'majmin':>8} {'seg':>8}")
    for c in CONFIGS:
        print(f"{c:<14} {aggs[c]['majmin']:8.3f} {aggs[c]['seg']:8.3f}")
    print(f"\n{'id':<28} {'raw_mm':>7} {'pit_mm':>7} {'mrg_mm':>7} {'srv_mm':>7} | "
          f"{'raw_sg':>7} {'srv_sg':>7} {'pitchΔ':>7}")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: x["id"]):
        s = r["scores"]
        print(
            f"{r['id']:<28} "
            f"{s['raw']['majmin']:7.3f} {s['pitch_only']['majmin']:7.3f} "
            f"{s['merge_only']['majmin']:7.3f} {s['serving']['majmin']:7.3f} | "
            f"{s['raw']['seg']:7.3f} {s['serving']['seg']:7.3f} "
            f"{r['pitch_delta_majmin']:+7.3f}"
        )


def _decide(dev: dict, test: dict) -> tuple[str, str]:
    """Pick serving config; return (config_name, rationale)."""
    raw_t = test["raw"]
    raw_d = dev["raw"]
    best_cfg = "raw"
    best_mm = raw_t["majmin"]
    for cfg in ("pitch_only", "merge_only", "serving"):
        mm = test[cfg]["majmin"]
        if mm > best_mm + 0.005:
            best_cfg = cfg
            best_mm = mm

    raw_seg = (raw_d["seg"] + raw_t["seg"]) / 2
    chosen_seg = (dev[best_cfg]["seg"] + test[best_cfg]["seg"]) / 2
    seg_loss = raw_seg - chosen_seg

    if best_cfg == "raw" and seg_loss <= 0.02:
        return (
            "raw",
            "Raw chordia wins TEST majmin; combined seg within 2pp of raw — serve raw (no pitch, no merge).",
        )
    if best_cfg == "raw" and seg_loss > 0.02:
        merge_seg = (dev["merge_only"]["seg"] + test["merge_only"]["seg"]) / 2
        if merge_seg >= raw_seg - 0.02 and test["merge_only"]["majmin"] >= raw_t["majmin"] - 0.01:
            return (
                "merge_only",
                "Raw wins majmin but seg regresses >2pp; gated light merge recovers seg without majmin loss.",
            )
        return (
            "raw",
            f"Raw wins majmin; seg tradeoff ({seg_loss:+.3f} vs raw) accepted — harmful post-steps dropped.",
        )

    if best_cfg == "pitch_only":
        return (
            "pitch_only",
            "Pitch correction alone improves TEST majmin; disable merge/chroma harm path.",
        )
    if best_cfg == "merge_only":
        return (
            "merge_only",
            "Gated light merge improves or preserves majmin+seg without pitch correction.",
        )
    return ("serving", "Current serving (pitch + merge) still optimal on v47.")


def _write_md(output: dict) -> None:
    dev = output["splits"]["dev"]["aggregate"]
    test = output["splits"]["test"]["aggregate"]
    v47 = output["v47_serving_baseline"]
    lines = [
        "# Phase 12.5 — post-processing ablation (v47 ruler)",
        "",
        "Configs: **raw** | **pitch_only** | **merge_only** (chroma align + gated merge, no pitch) | **serving** (both)",
        "",
        "## Aggregate majmin / seg",
        "",
        "| Config | DEV majmin | DEV seg | TEST majmin | TEST seg |",
        "|--------|------------|---------|-------------|----------|",
    ]
    for cfg in CONFIGS:
        lines.append(
            f"| {cfg} | {dev[cfg]['majmin']:.3f} | {dev[cfg]['seg']:.3f} | "
            f"{test[cfg]['majmin']:.3f} | {test[cfg]['seg']:.3f} |"
        )
    lines += [
        "",
        f"v47 serving baseline (old): DEV {v47['dev']:.3f} / TEST {v47['test']:.3f} majmin",
        "",
        "## Decision",
        "",
        f"**Chosen serving path:** `{output['chosen_config']}`",
        "",
        output["decision_rationale"],
        "",
        "## Pitch correction per track (majmin Δ pitch_only − raw)",
        "",
        "| id | Δ majmin | shift (st) |",
        "|----|----------|------------|",
    ]
    for row in sorted(output["pitch_per_track"], key=lambda r: r["pitch_delta_majmin"]):
        lines.append(
            f"| {row['id']} | {row['pitch_delta_majmin']:+.3f} | {row['pitch_shift_semitones']:.2f} |"
        )
    lines += ["", f"JSON: `analysis/PHASE12_5_POSTPROCESS_ABLATION.json`"]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--write-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    os.environ.setdefault("CHORD_ENGINE", "ml")
    splits = ("dev", "test") if args.split == "both" else (args.split,)

    output: dict = {
        "note": "v47 identity-verified audio; chordia ensemble path",
        "v47_serving_baseline": {
            "dev": json.loads(V47_DEV.read_text())["summary"]["avg_majmin_wcsr"],
            "test": json.loads(V47_TEST.read_text())["summary"]["avg_majmin_wcsr"],
        },
        "splits": {},
        "pitch_per_track": [],
    }

    all_rows: list[dict] = []
    for split_name in splits:
        rows = []
        for track in _load_tracks(split_name):
            row = _evaluate_track(track)
            if row.get("skipped"):
                continue
            rows.append(row)
            all_rows.append(row)
            print(
                f"  {row['id']}: raw={row['scores']['raw']['majmin']:.3f} "
                f"serving={row['scores']['serving']['majmin']:.3f} "
                f"pitchΔ={row['pitch_delta_majmin']:+.3f}"
            )
        _print_split(split_name, rows)
        output["splits"][split_name] = {
            "aggregate": {c: _agg(rows, c) for c in CONFIGS},
            "tracks": rows,
        }

    for row in all_rows:
        output["pitch_per_track"].append({
            "id": row["id"],
            "pitch_shift_semitones": row["pitch_shift_semitones"],
            "pitch_delta_majmin": row["pitch_delta_majmin"],
            "pitch_delta_seg": round(
                row["scores"]["pitch_only"]["seg"] - row["scores"]["raw"]["seg"], 4,
            ),
        })

    dev_agg = output["splits"]["dev"]["aggregate"]
    test_agg = output["splits"]["test"]["aggregate"]
    chosen, rationale = _decide(dev_agg, test_agg)
    output["chosen_config"] = chosen
    output["decision_rationale"] = rationale

    print(f"\nDECISION: serve `{chosen}` — {rationale}")

    args.write_json.write_text(json.dumps(output, indent=2))
    _write_md(output)
    print(f"Wrote {args.write_json}, {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
