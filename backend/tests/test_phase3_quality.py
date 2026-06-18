"""Phase 3: symmetric quality matching."""
import numpy as np

from analyzer import _match_score_for_chord, _quality_match_variants


def test_quality_match_variants_symmetric():
    assert set(_quality_match_variants("")) == {"", "5"}
    assert set(_quality_match_variants("m")) == {"m", "m7"}
    assert set(_quality_match_variants("m7")) == {"m7", "m"}


def test_match_score_minor_uses_m7_template():
    """Am7 template can score an Am label (symmetric with major + 5)."""
    chroma = np.zeros(12, dtype=np.float64)
    chroma[9] = 1.0   # A
    chroma[0] = 0.8    # C (min third)
    chroma[7] = 0.5    # G (min7)
    bass = chroma.copy()
    mid = chroma.copy()

    score_am = _match_score_for_chord(chroma, bass, mid, "Am", 9, False)
    assert score_am > 0.0


def test_third_quality_bias_uses_mid_register():
    from analyzer import _third_quality_bias

    full = np.zeros(12, dtype=np.float64)
    full[0] = 1.0
    full[4] = 0.9

    mid = np.zeros(12, dtype=np.float64)
    mid[0] = 1.0
    mid[3] = 0.85

    bias_mid = _third_quality_bias(full, "C", chroma_mid=mid)
    assert bias_mid.get("m", 1.0) > 1.0
