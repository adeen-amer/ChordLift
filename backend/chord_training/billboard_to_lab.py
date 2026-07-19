#!/usr/bin/env python3
"""Convert McGill Billboard `salami_chords.txt` annotations to Harte `.lab`
files (`<start>\\t<end>\\t<chord>` per line).

McGill format: each line is `<time>\\t<content>`. Content is either a
no-chord marker (silence/end/noise/&pause/section-only) or a section label
followed by `|`-delimited bars, e.g. `A, intro, | C:maj | F:maj |`, with an
optional trailing `x<N>` repeat-count for the whole bar group. A bar may
hold multiple space-separated chords, splitting that bar's time evenly.
`N` (no-chord) passes through as a normal token. A line's time span runs
until the next line's timestamp.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPEAT_RE = re.compile(r"x(\d+)\s*$")


def _parse_line(line: str) -> tuple[float, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None
    return float(parts[0]), parts[1].strip()


def _split_bars(content: str) -> list[list[str]]:
    """`A, intro, | C:maj | F:maj | x2` -> repeated [[chords-per-bar], ...]."""
    repeat = 1
    m = _REPEAT_RE.search(content)
    if m:
        repeat = int(m.group(1))
        content = content[: m.start()].rstrip()
    # segments[0] is the section-label prefix before the first '|'; the
    # final split segment (after the last '|') is empty and is dropped too.
    segments = content.split("|")[1:]
    bars = [seg.split() for seg in segments if seg.strip()]
    return bars * repeat


def parse_salami(text: str) -> list[tuple[float, float, str]]:
    events = [e for e in (_parse_line(ln) for ln in text.splitlines()) if e is not None]

    segments = []
    for (time, content), (end_time, _next_content) in zip(events, events[1:]):
        if "|" not in content or end_time <= time:
            continue  # no-chord marker (silence/end/noise/&pause/section-only)
        bars = _split_bars(content)
        if not bars:
            continue
        bar_dur = (end_time - time) / len(bars)
        t = time
        for bar in bars:
            chords = bar or ["N"]
            chord_dur = bar_dur / len(chords)
            for chord in chords:
                segments.append((t, t + chord_dur, chord))
                t += chord_dur
    return segments


def main() -> int:
    ap = argparse.ArgumentParser(description="McGill Billboard -> Harte .lab converter")
    ap.add_argument("--annotations-dir", required=True, type=Path)
    ap.add_argument("--out-labs", required=True, type=Path)
    args = ap.parse_args()

    if not args.annotations_dir.is_dir():
        print(f"error: {args.annotations_dir} is not a directory", file=sys.stderr)
        return 1
    args.out_labs.mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    for sub in sorted(args.annotations_dir.iterdir()):
        ann_file = sub / "salami_chords.txt"
        if not ann_file.is_file():
            continue
        stem = sub.name
        try:
            text = ann_file.read_text(encoding="utf-8", errors="ignore")
            segments = parse_salami(text)
            if not segments:
                raise ValueError("no chord segments parsed")
        except Exception as exc:  # noqa: BLE001 - skip unparseable files, don't crash
            print(f"{stem}: skip ({exc})")
            skip += 1
            continue
        lab_lines = "".join(f"{s:.6f}\t{e:.6f}\t{c}\n" for s, e, c in segments)
        (args.out_labs / f"{stem}.lab").write_text(lab_lines)
        print(f"{stem}: ok")
        ok += 1

    print(f"converted={ok} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
