#!/usr/bin/env python3
"""Verify gold audio — hardened Phase 11.5 identity gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from gold_audio_verify import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_MANIFEST,
    verify_catalog,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold audio identity gate (Phase 11.5)")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-ear", action="store_true", help="Fail if ear_check not approved")
    parser.add_argument(
        "--approve-ear",
        action="store_true",
        help="Approve ear-check only for pass tracks without manual-review flag",
    )
    parser.add_argument(
        "--approve-manual",
        metavar="IDS",
        help="Approve ear-check for scrutiny tracks (comma-separated ids, or 'all')",
    )
    args = parser.parse_args()

    results, failed = verify_catalog(args.catalog, BACKEND, require_ear_approved=args.require_ear)

    manual_ids: set[str] | None = None
    if args.approve_manual:
        if args.approve_manual.strip().lower() == "all":
            manual_ids = {r["id"] for r in results if r.get("needs_manual_review")}
        else:
            manual_ids = {x.strip() for x in args.approve_manual.split(",") if x.strip()}

    if args.approve_ear:
        for r in results:
            if r.get("status") == "pass" and not r.get("needs_manual_review"):
                r["ear_check_status"] = "approved"
    if manual_ids:
        for r in results:
            if r.get("status") == "pass" and r["id"] in manual_ids:
                r["ear_check_status"] = "approved"
                r["needs_manual_review"] = False
                r["manual_review_reason"] = ""

    write_manifest(results, args.manifest)

    scrutiny = []
    for r in results:
        status = r.get("status", "?")
        lab_d = r.get("duration_delta_sec")
        sp_d = r.get("spotify_delta_sec")
        lab_s = f" labΔ={lab_d:.2f}s" if lab_d is not None else ""
        sp_s = f" spotifyΔ={sp_d:.2f}s" if sp_d is not None else ""
        flag = " [MANUAL]" if r.get("needs_manual_review") else ""
        if status == "fail":
            print(f"FAIL {r['id']}: {r.get('reason')}{lab_s}{sp_s}{flag}")
        elif status == "no_audio":
            print(f"NO_AUDIO {r['id']}")
        elif status == "pass_with_warning":
            print(f"WARN {r['id']}: {r.get('reason')}{lab_s}{sp_s}{flag}")
            scrutiny.append(r["id"])
        else:
            print(f"PASS {r['id']}{lab_s}{sp_s}{flag}")
        if r.get("needs_manual_review"):
            scrutiny.append(r["id"])

    passed = sum(1 for r in results if r.get("status") == "pass")
    print(f"\n{passed}/{len(results)} strict pass (lab+Spotify)")
    if scrutiny:
        print(f"Manual review required: {', '.join(sorted(set(scrutiny)))}")

    if failed:
        return 1
    if args.require_ear:
        manifest = json.loads(args.manifest.read_text())
        pending = [
            tid for tid, m in manifest.get("tracks", {}).items()
            if m.get("duration_status") == "pass"
            and m.get("ear_check_status") != "approved"
        ]
        if pending:
            print(f"Ear-check pending: {', '.join(pending)}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
