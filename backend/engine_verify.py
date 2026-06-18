"""Phase 0: verify the chord engine that ran matches what was requested."""
from __future__ import annotations

import os
from pathlib import Path

from chord_model_registry import get_chord_model


class EngineMismatchError(RuntimeError):
    """Raised when ML was requested but classic (or another engine) ran."""


def strict_engine_checks_enabled() -> bool:
    return os.getenv("CHORD_ENGINE_STRICT", "0").lower() in ("1", "true", "yes")


def assert_engine_honest(key_info: dict | None) -> None:
    """
    Hard-fail eval/CI when CHORD_ENGINE=ml but the ML path did not run.

    Set CHORD_ENGINE_STRICT=1 (eval scripts set this automatically).
    """
    if not strict_engine_checks_enabled():
        return

    requested = os.getenv("CHORD_ENGINE", "ml").lower().strip()
    if requested != "ml":
        return

    info = key_info or {}
    actual = info.get("chord_engine_actual")
    if actual is None:
        raise EngineMismatchError(
            "Requested CHORD_ENGINE=ml but key_info has no chord_engine_actual "
            "(engine provenance not recorded)."
        )
    if actual != "ml":
        model = get_chord_model()
        reason = info.get("ml_fallback_reason", "unknown")
        raise EngineMismatchError(
            f"Requested CHORD_ENGINE=ml (CHORD_MODEL={model}) "
            f"but ran {actual}. ML fallback reason: {reason}"
        )

    ran_model = info.get("ml_model")
    wanted_model = get_chord_model()
    if ran_model and ran_model.lower().strip() != wanted_model:
        raise EngineMismatchError(
            f"Requested CHORD_MODEL={wanted_model} but ml_model={ran_model}"
        )


def clear_analysis_cache(cache_dir: str = "cache") -> int:
    """Remove cached analysis JSON files. Returns count removed."""
    root = Path(cache_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def prepare_eval_environment(*, no_cache: bool = False) -> None:
    """Eval harness: strict engine checks + optional cache wipe."""
    os.environ.setdefault("CHORD_ENGINE_STRICT", "1")
    if no_cache:
        clear_analysis_cache()
