"""ML chord-engine dependency checks."""
import os

from chord_model_registry import get_chord_model


def check_chordia_deps():
    missing = []
    try:
        import lv_chordia  # noqa: F401
    except Exception as exc:
        missing.append(f"lv-chordia ({exc.__class__.__name__})")
    try:
        import torch  # noqa: F401
    except Exception as exc:
        missing.append(f"torch ({exc.__class__.__name__})")
    return missing


def check_ml_chord_deps():
    """Return list of missing pieces for CHORD_ENGINE=ml."""
    get_chord_model()
    return check_chordia_deps()


def check_demucs_deps():
    """Return list of missing pieces for Demucs stem separation."""
    missing = []
    try:
        import demucs  # noqa: F401
    except Exception as exc:
        missing.append(f"demucs ({exc.__class__.__name__})")
    try:
        import torch  # noqa: F401
    except Exception as exc:
        missing.append(f"torch ({exc.__class__.__name__})")
    return missing


def ml_setup_hint():
    return "See backend/ML_SETUP.md — pip install -r requirements-ml.txt (lv-chordia + torch)"


def summarize_ml_readiness():
    """Status dict for logging and /api/health."""
    missing = check_ml_chord_deps()
    demucs_missing = check_demucs_deps()
    model = get_chord_model()
    return {
        "engine_requested": os.getenv("CHORD_ENGINE", "ml").lower().strip(),
        "ml_model": model,
        "ml_deps_ok": not missing,
        "missing_packages": missing,
        "demucs_available": not demucs_missing,
        "demucs_missing_packages": demucs_missing,
        "setup_hint": ml_setup_hint() if missing else None,
    }
