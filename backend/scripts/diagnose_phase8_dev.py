#!/usr/bin/env python3
"""Phase 8 — DEV accuracy leak diagnosis (measurement only, no engine edits)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from analyzer import (  # noqa: E402
    _align_segments_to_chroma,
    _extract_chroma_stack,
    _format_chord_name,
    _merge_adjacent_same_root,
    _resolve_song_key,
    _windowed_keys_from_chroma,
    extract_chords as extract_chords_classic,
)
from chord_engine_chordia import chordia_to_segments, recognize_chordia  # noqa: E402
from chord_engine_ml import (  # noqa: E402
    _merge_short_ml_segments,
    _maybe_power_pattern_timeline,
    _snap_ml_segments_to_beats,
)
from chord_label_utils import harte_label_to_internal  # noqa: E402
from chord_pipeline import build_chord_pipeline_context, finalize_bar_timeline  # noqa: E402
from chord_polish import polish_chord_timeline  # noqa: E402
from eval_mir_utils import (  # noqa: E402
    chordlift_to_harte,
    clip_lab_to_window,
    evaluate_mir_chords,
    load_harte_lab,
    segments_to_mir,
)
from gold_audio_verify import approved_track_ids  # noqa: E402
from stem_separation import _demucs_stems, separate_stems  # noqa: E402

CATALOG = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
SPLIT = BACKEND / "tests" / "fixtures" / "gold_split_v2.json"


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _score(segments: list[dict], ref_i, ref_l, duration: float) -> dict[str, float]:
    est_i, est_l = segments_to_mir(segments, duration=duration)
    return evaluate_mir_chords(ref_i, ref_l, est_i, est_l)


def _chordia_raw(y_in: np.ndarray, sr: int) -> list[dict]:
    return chordia_to_segments(recognize_chordia(y_in, sr))


def _prep_context(y, sr, *, demucs: bool = False):
    if demucs:
        stems = _demucs_stems(y, sr)
        from beat_tracking import track_beats
        from chord_pipeline import ChordPipelineContext

        beats = track_beats(stems.chord_signal, stems.bass, sr)
        ctx = ChordPipelineContext(stems=stems, beats=beats, sr=sr)
    else:
        ctx = build_chord_pipeline_context(y, sr)
    return ctx


def _postprocess_stages(
    raw_segments: list[dict],
    y,
    sr,
    pipeline,
    alternate_segments,
) -> dict[str, list[dict]]:
    """Return segment timelines at each diagnostic checkpoint."""
    y_chord = pipeline.y_chord
    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(
        y, sr, y_harmonic=y_chord,
    )
    chroma_mean = np.mean(chroma, axis=1)
    beat_duration = pipeline.beat_duration

    stages: dict[str, list[dict]] = {
        "0_raw": [dict(s) for s in raw_segments],
    }

    segments = [dict(s) for s in raw_segments]
    segments = _merge_adjacent_same_root(segments)
    frame_keys, beat_times, chroma_segments, _, _ = _windowed_keys_from_chroma(
        chroma, chroma_low, chroma_mid, y_harmonic, sr, beat_times=pipeline.beat_times,
    )
    segments = _snap_ml_segments_to_beats(segments, beat_times)
    segments = _merge_short_ml_segments(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr, beat_duration,
    )
    ml_segments = [dict(s) for s in segments]

    segments, _transpose = _align_segments_to_chroma(segments, chroma_mean)
    stages["1_align"] = [dict(s) for s in segments]

    frame_keys, beat_times, chroma_segments, _, _ = _windowed_keys_from_chroma(
        chroma, chroma_low, chroma_mid, y_harmonic, sr, beat_times=pipeline.beat_times,
    )
    from analyzer import _estimate_power_song_ratio

    power_ratio = _estimate_power_song_ratio(chroma_segments)
    prefer_power = power_ratio > 0.48
    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    onset_times = librosa.frames_to_time(
        librosa.onset.onset_detect(y=y_chord, sr=sr, hop_length=512, backtrack=True),
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
        segments = polish_chord_timeline(
            segments, chroma, chroma_low, chroma_mid,
            frame_keys, beat_times, onset_times, beat_duration, sr,
            key_root=key_root, is_major=is_major,
            prefer_sevenths=(mode == "dorian"),
            prefer_power=prefer_power,
            ml_root_bias=1.02,
        )
    stages["2_polish"] = [dict(s) for s in segments]

    segments = finalize_bar_timeline(
        segments, pipeline, chroma, chroma_low, chroma_mid, frame_keys, sr,
        alternate_segments=alternate_segments,
    )
    segments = _merge_adjacent_same_root(segments)
    stages["3_bar_finalize"] = [dict(s) for s in segments]

    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    for seg in segments:
        seg["chord"] = _format_chord_name(seg["chord"], key_root, is_major)
    stages["4_final"] = [dict(s) for s in segments]
    return stages


def experiment1(tracks: list[dict]) -> dict:
    rows = []
    for track in tracks:
        tid = track["id"]
        audio_path = BACKEND / track["file"]
        lab_path = BACKEND / track["lab"]
        if not audio_path.is_file() or not lab_path.is_file():
            continue

        ref_i, ref_l = load_harte_lab(lab_path)
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration = float(len(y) / sr)

        ctx_hpss = _prep_context(y, sr, demucs=False)
        try:
            ctx_demucs = _prep_context(y, sr, demucs=True)
            y_demucs_ob = ctx_demucs.stems.chord_signal
            demucs_ok = True
            demucs_err = None
        except Exception as exc:
            y_demucs_ob = None
            demucs_ok = False
            demucs_err = str(exc)

        inputs = {
            "a_raw_mix": y,
            "b_hpss_chord_stem": ctx_hpss.y_chord,
        }
        if demucs_ok and y_demucs_ob is not None:
            inputs["c_demucs_other_bass"] = y_demucs_ob

        row = {"id": tid, "demucs_available": demucs_ok}
        if not demucs_ok:
            row["demucs_error"] = demucs_err

        for key, y_in in inputs.items():
            raw = _chordia_raw(y_in, sr)
            scores = _score(raw, ref_i, ref_l, duration)
            row[key] = {**scores, "n_segments": len(raw)}

        rows.append(row)
        print(
            f"  exp1 {tid}: "
            f"mix majmin={row['a_raw_mix']['majmin']:.3f} "
            f"hpss={row['b_hpss_chord_stem']['majmin']:.3f}"
            + (
                f" demucs={row['c_demucs_other_bass']['majmin']:.3f}"
                if "c_demucs_other_bass" in row
                else " demucs=n/a"
            )
        )

    summary = {}
    for key in ("a_raw_mix", "b_hpss_chord_stem", "c_demucs_other_bass"):
        subset = [r[key] for r in rows if key in r]
        if subset:
            summary[key] = {
                "majmin": _avg(subset, "majmin"),
                "sevenths": _avg(subset, "sevenths"),
                "seg": _avg(subset, "seg"),
                "root": _avg(subset, "root"),
                "n_tracks": len(subset),
            }
    return {"tracks": rows, "summary": summary}


def experiment2(tracks: list[dict]) -> dict:
    stage_keys = ("0_raw", "1_align", "2_polish", "3_bar_finalize", "4_final")
    per_track = []
    stage_scores: dict[str, list[dict]] = {k: [] for k in stage_keys}

    for track in tracks:
        tid = track["id"]
        audio_path = BACKEND / track["file"]
        lab_path = BACKEND / track["lab"]
        if not audio_path.is_file() or not lab_path.is_file():
            continue

        ref_i, ref_l = load_harte_lab(lab_path)
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration = float(len(y) / sr)

        pipeline = build_chord_pipeline_context(y, sr)
        stem_alt, _ = extract_chords_classic(y, sr, pipeline, bar_finalize=False)
        raw = _chordia_raw(pipeline.y_chord, sr)
        stages = _postprocess_stages(raw, y, sr, pipeline, stem_alt)

        row = {"id": tid}
        for sk in stage_keys:
            sc = _score(stages[sk], ref_i, ref_l, duration)
            row[sk] = sc
            stage_scores[sk].append(sc)
        per_track.append(row)

        deltas = []
        prev_m = row["0_raw"]["majmin"]
        for sk in stage_keys[1:]:
            m = row[sk]["majmin"]
            deltas.append(f"{sk}:{m - prev_m:+.3f}")
            prev_m = m
        print(f"  exp2 {tid}: majmin " + " ".join(
            f"{sk}={row[sk]['majmin']:.3f}" for sk in stage_keys
        ))

    summary = {}
    for sk in stage_keys:
        subset = stage_scores[sk]
        summary[sk] = {
            "majmin": _avg(subset, "majmin"),
            "sevenths": _avg(subset, "sevenths"),
            "seg": _avg(subset, "seg"),
            "root": _avg(subset, "root"),
        }
    return {"tracks": per_track, "summary": summary}


def experiment3(tracks: list[dict]) -> dict:
    label_counts: Counter = Counter()
    mapping_pairs: list[dict] = []
    quality_loss: Counter = Counter()

    per_track_mapping_loss = []

    for track in tracks:
        lab_path = BACKEND / track["lab"]
        if not lab_path.is_file():
            continue
        ref_i, ref_l = load_harte_lab(lab_path)
        mapped = []
        for lab in ref_l:
            label_counts[lab] += 1
            internal = harte_label_to_internal(lab)
            if internal is None:
                roundtrip = "N"
                quality_loss["dropped_to_N"] += 1
            else:
                roundtrip = chordlift_to_harte(internal)
                if roundtrip != lab:
                    quality_loss[f"{lab} -> {roundtrip}"] += 1
            mapped.append(roundtrip)
            if lab not in {x["ref"] for x in mapping_pairs}:
                mapping_pairs.append({
                    "ref": lab,
                    "internal": internal,
                    "roundtrip": roundtrip,
                    "quality_preserved": roundtrip == lab,
                })

        if ref_i.size == 0:
            continue
        perfect = evaluate_mir_chords(ref_i, ref_l, ref_i, ref_l)
        after_map = evaluate_mir_chords(ref_i, ref_l, ref_i, mapped)
        per_track_mapping_loss.append({
            "id": track["id"],
            "perfect": perfect,
            "after_mapping": after_map,
            "majmin_loss": round(perfect["majmin"] - after_map["majmin"], 4),
            "sevenths_loss": round(perfect["sevenths"] - after_map["sevenths"], 4),
        })

    unique_refs = len(mapping_pairs)
    preserved = sum(1 for p in mapping_pairs if p["quality_preserved"])
    return {
        "unique_ref_labels": unique_refs,
        "quality_preserved_count": preserved,
        "quality_preserved_pct": round(100 * preserved / unique_refs, 1) if unique_refs else 0,
        "quality_loss_examples": dict(quality_loss.most_common(30)),
        "mapping_table": sorted(mapping_pairs, key=lambda x: x["ref"]),
        "per_track_oracle_loss": per_track_mapping_loss,
        "summary": {
            "avg_majmin_loss_from_mapping": _avg(per_track_mapping_loss, "majmin_loss"),
            "avg_sevenths_loss_from_mapping": _avg(per_track_mapping_loss, "sevenths_loss"),
        },
    }


def write_report(payload: dict, path: Path) -> None:
    exp1 = payload["experiment1"]["summary"]
    exp2 = payload["experiment2"]["summary"]
    exp3 = payload["experiment3"]["summary"]
    baseline = payload.get("baseline_dev", {})

    lines = [
        "# Phase 8 — DEV accuracy leak diagnosis",
        "",
        "**Measurement only** — no engine changes.",
        "",
        f"DEV baseline (v45): majmin **{baseline.get('majmin', '?')}** | "
        f"sevenths {baseline.get('sevenths', '?')} | seg {baseline.get('seg', '?')}",
        "",
        "## Experiment 1 — lv-chordia input signal (raw output, no post-processing)",
        "",
        "| Input | majmin | sevenths | seg | n |",
        "|-------|--------|----------|-----|---|",
    ]
    labels = {
        "a_raw_mix": "(a) raw full mix",
        "b_hpss_chord_stem": "(b) HPSS chord stem (today)",
        "c_demucs_other_bass": "(c) Demucs other+bass",
    }
    for key, label in labels.items():
        if key in exp1:
            s = exp1[key]
            lines.append(
                f"| {label} | {s['majmin']:.3f} | {s['sevenths']:.3f} | "
                f"{s['seg']:.3f} | {s['n_tracks']} |"
            )

    lines.extend([
        "",
        "## Experiment 2 — per-stage post-processing (HPSS chordia → pipeline)",
        "",
        "| Stage | majmin | sevenths | seg | Δ majmin vs prev |",
        "|-------|--------|----------|-----|------------------|",
    ])
    prev_m = None
    stage_labels = {
        "0_raw": "(0) raw chordia",
        "1_align": "(1) + chroma align",
        "2_polish": "(2) + polish",
        "3_bar_finalize": "(3) + bar finalize",
        "4_final": "(4) final format",
    }
    for sk, label in stage_labels.items():
        s = exp2[sk]
        delta = "" if prev_m is None else f"{s['majmin'] - prev_m:+.3f}"
        lines.append(
            f"| {label} | {s['majmin']:.3f} | {s['sevenths']:.3f} | "
            f"{s['seg']:.3f} | {delta} |"
        )
        prev_m = s["majmin"]

    lines.extend([
        "",
        "## Experiment 3 — label mapping loss (oracle self-score)",
        "",
        f"- Unique Harte ref labels: {payload['experiment3']['unique_ref_labels']}",
        f"- Quality preserved roundtrip: {payload['experiment3']['quality_preserved_pct']}%",
        f"- Avg majmin lost to mapping alone: **{exp3.get('avg_majmin_loss_from_mapping', 0):.4f}**",
        f"- Avg sevenths lost to mapping alone: **{exp3.get('avg_sevenths_loss_from_mapping', 0):.4f}**",
        "",
        "## Gate verdict",
        "",
        payload.get("verdict", "(see JSON for full breakdown)"),
        "",
    ])
    path.write_text("\n".join(lines) + "\n")


def build_verdict(payload: dict) -> str:
    exp1 = payload["experiment1"]["summary"]
    exp2 = payload["experiment2"]["summary"]
    exp3 = payload["experiment3"]["summary"]
    baseline_m = payload.get("baseline_dev", {}).get("majmin", 0.488)

    raw_mix = exp1.get("a_raw_mix", {}).get("majmin", 0)
    hpss = exp1.get("b_hpss_chord_stem", {}).get("majmin", 0)
    chordia_ceiling = exp2.get("0_raw", {}).get("majmin", 0)
    final = exp2.get("4_final", {}).get("majmin", 0)
    map_loss = exp3.get("avg_majmin_loss_from_mapping", 0) or 0

    post_loss = chordia_ceiling - final
    stem_penalty = hpss - raw_mix if raw_mix > hpss else 0

    parts = [
        f"**DEV gap vs chordia raw ceiling:** baseline final {baseline_m:.3f} vs raw chordia {chordia_ceiling:.3f} "
        f"→ pipeline throws away ~{max(0, chordia_ceiling - baseline_m):.3f} majmin.",
        f"**Input signal:** raw mix {raw_mix:.3f} vs HPSS stem {hpss:.3f} "
        f"({'HPSS hurts' if hpss < raw_mix - 0.01 else 'HPSS neutral/helpful'}; Δ={hpss - raw_mix:+.3f}).",
    ]
    if "c_demucs_other_bass" in exp1:
        demucs_m = exp1["c_demucs_other_bass"]["majmin"]
        parts.append(f"**Demucs other+bass:** {demucs_m:.3f} (Δ vs HPSS {demucs_m - hpss:+.3f}).")

    # Find worst postprocess stage drop (most negative majmin delta)
    stage_keys = ("0_raw", "1_align", "2_polish", "3_bar_finalize", "4_final")
    worst = ("", 0.0)
    prev = exp2["0_raw"]["majmin"]
    for sk in stage_keys[1:]:
        drop = exp2[sk]["majmin"] - prev  # negative = loss
        if drop < worst[1]:
            worst = (sk, drop)
        prev = exp2[sk]["majmin"]
    parts.append(
        f"**Largest post-process drop:** stage `{worst[0]}` Δ majmin {worst[1]:+.3f} "
        f"(raw→final net {exp2['4_final']['majmin'] - exp2['0_raw']['majmin']:+.3f}; "
        f"align net {exp2['1_align']['majmin'] - exp2['0_raw']['majmin']:+.3f})."
    )
    parts.append(
        f"**Label mapping:** oracle majmin loss {map_loss:.4f} "
        f"({exp3.get('avg_sevenths_loss_from_mapping', 0):.4f} sevenths) — "
        f"~{100 * map_loss / max(0.01, chordia_ceiling - final):.0f}% of raw→final gap."
    )

    # Primary attribution
    align_delta = exp2["1_align"]["majmin"] - exp2["0_raw"]["majmin"]
    polish_drop = exp2["2_polish"]["majmin"] - exp2["1_align"]["majmin"]
    bar_drop = exp2["3_bar_finalize"]["majmin"] - exp2["2_polish"]["majmin"]

    if hpss >= raw_mix - 0.01:
        parts.append("**Input signal verdict:** HPSS stem does *not* hurt lv-chordia (mix ≤ HPSS on avg). Hypothesis rejected.")
    else:
        parts.append(f"**Input signal verdict:** HPSS hurts vs full mix (Δ {hpss - raw_mix:+.3f}).")

    if polish_drop < -0.05 and bar_drop < -0.05:
        attr = (
            "**Phase 10 target:** Post-processing is the main leak — "
            f"`polish_chord_timeline` ({polish_drop:+.3f}) and `finalize_bar_timeline` ({bar_drop:+.3f}). "
            "Chroma align helps on average (+{:.3f}); not the primary bug.".format(align_delta)
        )
    elif worst[0] == "1_align" and worst[1] < -0.03:
        attr = "**Phase 10 target:** Global chroma transpose (_align_segments_to_chroma)."
    elif chordia_ceiling < 0.52:
        attr = "**Phase 10 target:** lv-chordia raw ceiling (~{:.3f}) plus post-processing.".format(chordia_ceiling)
    else:
        attr = "**Phase 10 target:** Mixed — tune post-processing first; raw chordia ceiling ~{:.3f}.".format(chordia_ceiling)

    parts.append(attr)
    return "\n\n".join(f"- {p}" for p in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 DEV diagnosis")
    parser.add_argument("--write-json", type=Path, default=BACKEND / "analysis" / "PHASE8_DIAGNOSIS_DEV.json")
    parser.add_argument("--write-md", type=Path, default=BACKEND / "analysis" / "PHASE8_DIAGNOSIS_DEV.md")
    args = parser.parse_args()

    os.environ.setdefault("CHORD_ENGINE", "ml")
    catalog = json.loads(CATALOG.read_text())
    split = json.loads(SPLIT.read_text())
    approved = approved_track_ids()
    dev_ids = [t for t in split["dev"] if t in approved]
    tracks = [t for t in catalog["tracks"] if t["id"] in dev_ids]

    baseline_path = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v45.json"
    baseline_dev = {}
    if baseline_path.is_file():
        b = json.loads(baseline_path.read_text()).get("summary", {})
        baseline_dev = {
            "majmin": b.get("avg_majmin_wcsr"),
            "sevenths": b.get("avg_sevenths_wcsr"),
            "seg": b.get("avg_seg_score"),
            "root": b.get("avg_root_wcsr"),
        }

    print(f"Phase 8 diagnosis — {len(tracks)} DEV tracks\n")
    print("Experiment 1: chordia input signal...")
    e1 = experiment1(tracks)
    print("\nExperiment 2: post-processing stages...")
    e2 = experiment2(tracks)
    print("\nExperiment 3: label mapping...")
    e3 = experiment3(tracks)

    payload = {
        "description": "Phase 8 DEV accuracy leak diagnosis (measurement only)",
        "n_tracks": len(tracks),
        "baseline_dev": baseline_dev,
        "experiment1": e1,
        "experiment2": e2,
        "experiment3": e3,
    }
    payload["verdict"] = build_verdict(payload)

    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload, args.write_md)
    print(f"\nWrote {args.write_json}")
    print(f"Wrote {args.write_md}")
    print("\n" + payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
