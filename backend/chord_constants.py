"""Shared chord-engine constants (no analyzer imports)."""

# Frozen polish defaults (v25) — change only with held-out eval, not per-song tuning.
CHORD_POLISH_DEFAULTS = {
    "label_inertia": 0.15,
    "ml_label_inertia": 0.45,
    "chroma_align_min_gain_ratio": 0.15,
    "third_quality_mid_threshold": 1.15,
    "flux_keep_threshold": 0.50,
    "flux_keep_threshold_flicker": 0.56,
    "match_keep_threshold": 0.30,
    "match_keep_threshold_flicker": 0.32,
    "cross_validate_min_gain": 0.14,
    "intro_stable_beats": 4,
    "intro_max_sec": 22.0,
    "flicker_min_beats": 1.5,
    "ml_beats_per_bar": 4,
    "ml_bar_merge_match_threshold": 0.28,
    "bar_alternate_min_margin": 0.12,
    "bar_strict_overseg_ratio": 1.25,
}

# Shown when no rhythm-density heuristic has run yet.
PLACEHOLDER_STRUMMING = ""
