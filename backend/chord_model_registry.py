"""Chord ML model id — serving engine is lv-chordia only."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ML_MODEL_CHORDIA = "chordia"
ML_MODEL_DEFAULT = ML_MODEL_CHORDIA

# `ensemble` was a no-op alias; `btc` removed in cleanup sprint.
_LEGACY_ALIASES = {"ensemble": ML_MODEL_CHORDIA}
_REMOVED = frozenset({"btc", "conformer", "autochord", "stem", "chordia-only"})


def get_chord_model() -> str:
    """Resolve active chord model id from CHORD_MODEL or CHORD_ML_MODEL."""
    for key in ("CHORD_MODEL", "CHORD_ML_MODEL"):
        raw = os.getenv(key)
        if raw:
            return raw.lower().strip()
    return ML_MODEL_DEFAULT


def validate_chord_model(model_id: str) -> str:
    """Return canonical model id (chordia) or raise/warn per strict mode."""
    from engine_verify import strict_engine_checks_enabled

    if model_id == ML_MODEL_CHORDIA:
        return ML_MODEL_CHORDIA

    if model_id in _LEGACY_ALIASES:
        logger.warning(
            "CHORD_MODEL=%r is deprecated; using %r.",
            model_id,
            _LEGACY_ALIASES[model_id],
        )
        return _LEGACY_ALIASES[model_id]

    if model_id in _REMOVED:
        msg = f"CHORD_MODEL={model_id!r} is removed. Use {ML_MODEL_CHORDIA!r}."
        if strict_engine_checks_enabled():
            raise ValueError(msg)
        logger.warning("%s Using %s.", msg, ML_MODEL_DEFAULT)
        return ML_MODEL_DEFAULT

    msg = f"CHORD_MODEL={model_id!r} invalid; use {ML_MODEL_CHORDIA!r}."
    if strict_engine_checks_enabled():
        raise ValueError(msg)
    logger.warning("%s Using %s.", msg, ML_MODEL_DEFAULT)
    return ML_MODEL_DEFAULT
