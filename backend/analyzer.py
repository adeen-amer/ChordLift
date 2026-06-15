import os
import json
import librosa
import numpy as np
import logging
from collections import Counter
from scipy.ndimage import median_filter
from scipy.optimize import nnls
from scipy.signal import find_peaks

previous_level = logging.getLogger().getEffectiveLevel()
logging.getLogger().setLevel(logging.ERROR)
from basic_pitch.inference import predict
logging.getLogger().setLevel(previous_level)

CACHE_DIR = "cache"
ANALYZER_VERSION = "27"

# Frozen polish defaults (v25) — change only with held-out eval, not per-song tuning.
CHORD_POLISH_DEFAULTS = {
    "label_inertia": 0.15,
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
}
os.makedirs(CACHE_DIR, exist_ok=True)


def get_chord_engine_name():
    return os.getenv("CHORD_ENGINE", "ml").lower().strip()


def cache_metadata():
    return {
        "analyzer_version": ANALYZER_VERSION,
        "chord_engine": get_chord_engine_name(),
    }


def is_cache_valid(cached):
    if not cached:
        return False
    meta = cache_metadata()
    return (
        cached.get("analyzer_version") == meta["analyzer_version"]
        and cached.get("chord_engine") == meta["chord_engine"]
    )

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_KEYS = {1, 3, 5, 6, 8, 10}  # legacy — display always uses sharps now

MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Major-key diatonic triads: I, ii, iii, IV, V, vi (skip vii°)
DIATONIC_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9]
DIATONIC_MAJOR_QUALITIES = ['', 'm', 'm', '', '', 'm']

ENHARMONIC = {
    'A#': 'Bb', 'Bb': 'Bb',
    'C#': 'Db', 'Db': 'Db',
    'D#': 'Eb', 'Eb': 'Eb',
    'F#': 'Gb', 'Gb': 'Gb',
    'G#': 'Ab', 'Ab': 'Ab',
}

SHARP_TO_INTERNAL = {'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#'}


def _to_sharp_root(root):
    """Normalize any root spelling to sharp notation (C, C#, D, …)."""
    if not root:
        return root
    return SHARP_TO_INTERNAL.get(root, root)


def _chord_to_internal(name):
    root = name.rstrip('m')
    suffix = 'm' if name.endswith('m') else ''
    root = SHARP_TO_INTERNAL.get(root, root)
    return f"{root}{suffix}"


def _rotate_template(intervals, root_idx):
    template = np.zeros(12)
    for interval in intervals:
        template[(root_idx + interval) % 12] = 1
    return template


INTERVAL_WEIGHTS = {
    0: 1.0, 2: 0.35, 3: 0.90, 4: 0.90, 5: 0.40, 7: 0.80, 10: 0.65, 11: 0.58,
}


def _rotate_weighted_template(intervals, root_idx):
    """Harmonic-weighted template for NNLS decomposition."""
    template = np.zeros(12)
    for interval in intervals:
        weight = INTERVAL_WEIGHTS.get(interval, 0.45)
        idx = (root_idx + interval) % 12
        template[idx] = max(template[idx], weight)
    return template


def _build_triad_templates():
    templates = {}
    for root_idx, root in enumerate(ROOT_NAMES):
        templates[root] = _rotate_template([0, 4, 7], root_idx)
        templates[f"{root}m"] = _rotate_template([0, 3, 7], root_idx)
        templates[f"{root}5"] = _rotate_template([0, 7], root_idx)
    return templates


TRIAD_TEMPLATES = _build_triad_templates()
TRIAD_NAMES = list(TRIAD_TEMPLATES.keys())
TRIAD_MATRIX = np.array(list(TRIAD_TEMPLATES.values())).T

CHORD_QUALITIES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'sus4': [0, 5, 7],
    'sus2': [0, 2, 7],
    '5': [0, 7],
}

DECODE_QUALITIES = {'', 'm', '7', 'm7', '5'}


def _build_chord_vocabulary(qualities):
    names = ['N']
    templates = [np.zeros(12)]
    weighted_templates = [np.zeros(12)]
    for root_idx, root in enumerate(ROOT_NAMES):
        for suffix, intervals in CHORD_QUALITIES.items():
            if suffix not in qualities:
                continue
            names.append(f"{root}{suffix}")
            templates.append(_rotate_template(intervals, root_idx))
            weighted_templates.append(_rotate_weighted_template(intervals, root_idx))
    matrix = np.array(templates).T
    weighted_matrix = np.array(weighted_templates).T
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix_n = matrix / (norms + 1e-6)
    return names, matrix, matrix_n, weighted_matrix


CHORD_VOCAB, CHORD_MATRIX, CHORD_MATRIX_N, CHORD_NNLS_MATRIX = _build_chord_vocabulary(set(CHORD_QUALITIES))
DECODE_VOCAB, DECODE_MATRIX, DECODE_MATRIX_N, DECODE_NNLS_MATRIX = _build_chord_vocabulary(DECODE_QUALITIES)
N_DECODE = len(DECODE_VOCAB)
NO_CHORD_IDX = DECODE_VOCAB.index('N')


def _chord_quality(name):
    if name == 'N' or not name:
        return ''
    for suffix in ('maj7', 'sus4', 'sus2', 'm7', 'm', '7', '5'):
        if name.endswith(suffix):
            return suffix
    return ''


def _diatonic_chord_names(key_root, major=True):
    if major:
        return [f"{ROOT_NAMES[(key_root + iv) % 12]}{q}"
                for iv, q in zip(DIATONIC_MAJOR_INTERVALS, DIATONIC_MAJOR_QUALITIES)]
    intervals = [0, 2, 3, 5, 7, 8, 10]
    qualities = ['m', 'dim', 'M', 'm', 'm', 'M', 'M']
    names = []
    for iv, q in zip(intervals, qualities):
        root = ROOT_NAMES[(key_root + iv) % 12]
        if q == 'M':
            names.append(root)
        elif q == 'm':
            names.append(f"{root}m")
        elif q == 'dim':
            names.append(f"{root}dim")
    return names


def _apply_simplicity_prior(scores, vocab, extension_threshold=1.12):
    """Prefer triads unless 7th/power evidence is clearly stronger."""
    by_root = {}
    for i, name in enumerate(vocab):
        if name == 'N':
            continue
        root = _chord_root_name(name)
        quality = _chord_quality(name)
        by_root.setdefault(root, {})[quality] = i

    for root, indices in by_root.items():
        base_idx = indices.get('')
        if base_idx is None:
            base_idx = indices.get('m')
        if base_idx is None:
            continue
        base_score = scores[base_idx]
        minor_idx = indices.get('m')
        if minor_idx is not None and scores[minor_idx] < base_score * 0.92:
            scores[minor_idx] *= 0.88
        for ext in ('7', 'm7', '5'):
            ext_idx = indices.get(ext)
            if ext_idx is not None and scores[ext_idx] < base_score * extension_threshold:
                scores[ext_idx] *= 0.82

    return scores


def _internal_root(root_name):
    """Map display flats to internal sharp roots used in CHORD_VOCAB."""
    return SHARP_TO_INTERNAL.get(root_name, root_name)


def _pick_best_chord_index(scores, vocab, sus_margin=1.16, ext_margin=1.12):
    """
    Pick the best chord index, preferring plain triads over sus/7th
    unless the extension score clearly wins.
    """
    best_idx = int(np.argmax(scores))
    best_name = vocab[best_idx]
    quality = _chord_quality(best_name)

    if quality not in ('sus2', 'sus4', '7', 'maj7', 'm7'):
        return best_idx

    root = _chord_root_name(best_name)
    margin = sus_margin if quality in ('sus2', 'sus4') else ext_margin
    best_score = scores[best_idx]

    for triad_q in ('', 'm'):
        triad_name = f"{root}{triad_q}"
        if triad_name not in vocab:
            continue
        triad_idx = vocab.index(triad_name)
        if best_score <= 0 or scores[triad_idx] * margin >= best_score:
            return triad_idx

    return best_idx


def _chroma_beat_flux(chroma_segments):
    """Harmonic change strength at each beat boundary."""
    n_frames = chroma_segments.shape[1]
    flux = np.zeros(n_frames)
    if n_frames < 2:
        return flux
    for i in range(1, n_frames):
        flux[i] = float(np.linalg.norm(chroma_segments[:, i] - chroma_segments[:, i - 1]))
    return flux


def _decode_chord_sequence(raw_emissions, flux, smooth_window=5, min_run=2):
    """
    Frame argmax + median smoothing + flux-gated flicker removal.
    Full adaptive Viterbi over 60+ states tends to collapse songs to 1–2 chords;
    this keeps local evidence while suppressing beat-level jitter.
    """
    n_frames = raw_emissions.shape[0]
    if n_frames == 0:
        return []

    path = np.array([int(np.argmax(raw_emissions[t])) for t in range(n_frames)], dtype=np.int32)

    if smooth_window > 1 and n_frames >= 3:
        half = smooth_window // 2
        smoothed = path.copy()
        for i in range(n_frames):
            window = path[max(0, i - half):min(n_frames, i + half + 1)]
            counts = Counter(window.tolist())
            smoothed[i] = counts.most_common(1)[0][0]
        path = smoothed

    return _stabilize_path_with_flux(path, flux, min_run=min_run).tolist()


def _viterbi_decode_chords(log_emissions, frame_keys, flux, stay_log=0.45, min_run=2):
    """
    Key-aware Viterbi over decode vocabulary.

    Favors staying on the same chord and moving to diatonic chords in the
    local key window; penalizes wild root jumps.
    """
    n_frames, n_states = log_emissions.shape
    if n_frames == 0:
        return []

    diatonic_by_frame = []
    for t in range(n_frames):
        key_root, is_major = frame_keys[t]
        diatonic_roots = {_chord_root_name(c) for c in _diatonic_chord_names(key_root, is_major)}
        diatonic_by_frame.append({
            i for i, name in enumerate(DECODE_VOCAB)
            if name == 'N' or _chord_root_name(name) in diatonic_roots
        })

    dp = np.full((n_frames, n_states), -np.inf)
    back = np.zeros((n_frames, n_states), dtype=np.int32)
    dp[0] = log_emissions[0]

    for t in range(1, n_frames):
        diatonic = diatonic_by_frame[t]
        for s in range(n_states):
            best_prev = 0
            best_score = -np.inf
            curr_name = DECODE_VOCAB[s]
            for p in range(n_states):
                if p == s:
                    trans = stay_log
                else:
                    prev_name = DECODE_VOCAB[p]
                    if prev_name == 'N' or curr_name == 'N':
                        trans = -0.15
                    elif s in diatonic:
                        trans = -0.12
                    else:
                        trans = -0.55
                score = dp[t - 1, p] + trans
                if score > best_score:
                    best_score = score
                    best_prev = p
            dp[t, s] = best_score + log_emissions[t, s]
            back[t, s] = best_prev

    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return _stabilize_path_with_flux(path, flux, min_run=min_run).tolist()


def _stabilize_path_with_flux(path, flux, min_run=2):
    """Suppress brief flickers unless chroma flux supports a chord change."""
    if len(path) <= 1:
        return path

    out = path.copy()
    threshold = float(np.percentile(flux[flux > 0], 62)) if np.any(flux > 0) else 0.0
    i = 1
    while i < len(out):
        if out[i] == out[i - 1]:
            i += 1
            continue

        new_chord = out[i]
        run = 1
        j = i + 1
        while j < len(out) and out[j] == new_chord:
            run += 1
            j += 1

        if run < min_run and flux[i] < threshold:
            out[i:j] = out[i - 1]
        i = j

    return out


def _key_at_time(time_sec, beat_times, frame_keys):
    """Map a timestamp to the window-local key estimate on the beat grid."""
    if not frame_keys:
        return (0, True)
    if beat_times is None or len(beat_times) == 0:
        return frame_keys[0]

    idx = int(np.searchsorted(beat_times, time_sec, side='right')) - 1
    idx = max(0, min(idx, len(frame_keys) - 1))
    return frame_keys[idx]


def _beat_sync_chroma_stack(y_harmonic, chroma, chroma_low, chroma_mid, sr, hop_length=512):
    """Sync chroma layers to beat grid; returns segments and beat times in seconds."""
    _, beat_frames = librosa.beat.beat_track(
        y=y_harmonic, sr=sr, hop_length=hop_length, units='frames',
    )
    beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1] - 1)
    if len(beat_frames) < 4:
        beat_frames = np.arange(0, chroma.shape[1], max(1, chroma.shape[1] // 64))

    chroma_segments = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    bass_segments = librosa.util.sync(chroma_low, beat_frames, aggregate=np.median)
    mid_segments = librosa.util.sync(chroma_mid, beat_frames, aggregate=np.median)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return chroma_segments, bass_segments, mid_segments, beat_times


def _resolve_key_from_chroma_stack(chroma, chroma_low, chroma_mid, y_harmonic, sr, hop_length=512):
    """Classic-style key vote from beat-synced chroma (shared by ML + classic paths)."""
    chroma_mean = np.mean(chroma, axis=1)
    chroma_segments, bass_segments, mid_segments, _ = _beat_sync_chroma_stack(
        y_harmonic, chroma, chroma_low, chroma_mid, sr, hop_length,
    )
    _, raw_em = _frame_emission_matrix(
        chroma_segments, bass_segments, 0, True, mid_segments,
    )
    return _resolve_key(chroma_mean, raw_em, bass_segments, chroma_segments)


def _windowed_keys_from_chroma(chroma, chroma_low, chroma_mid, y_harmonic, sr, hop_length=512, step=12):
    """Beat-synced chroma + per-window key estimates for segment refinement."""
    chroma_mean = np.mean(chroma, axis=1)
    chroma_segments, bass_segments, mid_segments, beat_times = _beat_sync_chroma_stack(
        y_harmonic, chroma, chroma_low, chroma_mid, sr, hop_length,
    )
    frame_keys = _windowed_key_for_frames(
        chroma_segments, bass_segments, chroma_mean, mid_segments, step=step,
    )
    return frame_keys, beat_times, chroma_segments, bass_segments, mid_segments


def _chroma_profile_score(chroma_mean, key_root, is_major):
    cm = np.asarray(chroma_mean, dtype=np.float64)
    cm = cm / (np.linalg.norm(cm) + 1e-6)
    rolled = np.roll(cm, -key_root)
    profile = MAJOR_KEY_PROFILE if is_major else MINOR_KEY_PROFILE
    return float(np.dot(rolled, profile))


def _segment_diatonic_coverage(segments, key_root, is_major):
    """Fraction of segment duration whose roots are diatonic in the given key."""
    diatonic_pcs = {
        _root_to_pitch_class(_chord_root_name(c))
        for c in _diatonic_chord_names(key_root, is_major)
    }
    diatonic_pcs.discard(None)
    total = 0.0
    covered = 0.0
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        pc = _root_to_pitch_class(_chord_root_name(chord))
        if pc is None:
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        total += dur
        if pc in diatonic_pcs:
            covered += dur
        elif is_major and any(
            (pc + delta) % 12 in diatonic_pcs for delta in (-1, 1)
        ):
            covered += dur * 0.5
    return covered / total if total > 0 else 0.0


def _transpose_chord(chord_name, semitones):
    """Transpose a chord symbol by semitones (negative = down)."""
    if not chord_name or chord_name == "N" or semitones == 0:
        return chord_name
    quality = _chord_quality(chord_name)
    root = chord_name[: -len(quality)] if quality else chord_name
    pc = _root_to_pitch_class(root)
    if pc is None:
        return chord_name
    new_pc = (pc + semitones) % 12
    return f"{ROOT_NAMES[new_pc]}{quality}"


def _chroma_alignment_score(segments, chroma_mean, shift):
    """Weighted chroma energy at segment roots after transposing labels by -shift."""
    cm = np.asarray(chroma_mean, dtype=np.float64)
    score = 0.0
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        pc = _root_to_pitch_class(_chord_root_name(_transpose_chord(chord, -shift)))
        if pc is None:
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        score += float(cm[pc]) * dur
    return score


def _align_segments_to_chroma(segments, chroma_mean, min_gain=0.06):
    """
    Globally shift ML chord labels so roots match the chroma profile.

    Fixes systematic autochord transposition (e.g. Viva La Vida +8 semitones).
    """
    if not segments:
        return segments, 0

    base = _chroma_alignment_score(segments, chroma_mean, 0)
    best_shift = 0
    best_score = base
    for shift in range(1, 12):
        score = _chroma_alignment_score(segments, chroma_mean, shift)
        if score > best_score:
            best_score = score
            best_shift = shift

    if best_shift == 0 or (best_score - base) < min_gain * max(base, 1e-6):
        return segments, 0

    out = []
    for seg in segments:
        row = dict(seg)
        chord = row.get("chord", "")
        if chord and chord != "N":
            row["chord"] = _transpose_chord(chord, -best_shift)
        out.append(row)
    return out, best_shift


def _score_pop_loop_fit(segments, key_root, is_major):
    """Score fit to I–V–vi–IV (major) loop roots with ±1 semitone tolerance."""
    if not is_major:
        return 0.0
    loop_pcs = {(key_root + iv) % 12 for iv in (0, 7, 9, 5)}
    total = 0.0
    matched = 0.0
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        pc = _root_to_pitch_class(_chord_root_name(chord))
        if pc is None:
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        total += dur
        if pc in loop_pcs:
            matched += dur
        elif any((pc + delta) % 12 in loop_pcs for delta in (-1, 1)):
            matched += dur * 0.55
    return matched / total if total > 0 else 0.0


def _detect_major_loop_tonic(segments, min_coverage=0.72):
    """
    If segments fit a I–V–vi–IV loop, return the tonic pitch class.

    Uses ±1 semitone tolerance so a mis-spelled Fm still supports C# over G#.
    """
    if not segments:
        return None

    root_weight = Counter()
    total = 0.0
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        pc = _root_to_pitch_class(_chord_root_name(chord))
        if pc is None:
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        root_weight[pc] += dur
        total += dur

    if total <= 0:
        return None

    best_tonic = None
    best_cov = 0.0
    for tonic in range(12):
        loop_pcs = {(tonic + iv) % 12 for iv in (0, 7, 9, 5)}
        covered = 0.0
        for pc, weight in root_weight.items():
            if pc in loop_pcs:
                covered += weight
            elif any((pc + delta) % 12 in loop_pcs for delta in (-1, 1)):
                covered += weight * 0.55
        cov = covered / total
        if cov > best_cov:
            best_cov = cov
            best_tonic = tonic

    if best_tonic is None or best_cov < min_coverage:
        return None
    return best_tonic


def _key_candidate_score(
    root,
    major,
    segments,
    chroma_mean,
    classic_root,
    classic_major,
    profile_root,
    profile_major,
):
    diatonic = _segment_diatonic_coverage(segments, root, major)
    chroma_sc = _chroma_profile_score(chroma_mean, root, major)
    score = diatonic * 3.0 + chroma_sc * 0.12
    if major:
        score += _score_pop_loop_fit(segments, root, True) * 1.8
    if (root, major) == (classic_root, classic_major):
        score += 0.3
    if (root, major) == (profile_root, profile_major):
        score += 0.15
    return score


def _resolve_song_key(chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr):
    """
    Pick global key/mode from chroma profile, chord votes, and segment fit.

    Evaluates relative major/minor pairs so a strong C-major chroma profile
    is not overridden by spurious A-minor segment votes (and vice versa).
    """
    dorian = _detect_dorian_from_segments(segments)
    if dorian is not None:
        return dorian[0], False, "dorian"

    classic_root, classic_major = _resolve_key_from_chroma_stack(
        chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    profile_root, profile_major = _estimate_key(chroma_mean, major_only=False)

    chords = [s["chord"] for s in segments if s.get("chord") and s["chord"] != "N"]
    if chords:
        vote_root, vote_major = _estimate_key_from_chord_votes(chords)
    else:
        vote_root, vote_major = classic_root, classic_major

    seeds = [
        (classic_root, classic_major),
        (vote_root, vote_major),
        (profile_root, profile_major),
    ]
    loop_tonic = _detect_major_loop_tonic(segments)
    if loop_tonic is not None:
        seeds.insert(0, (loop_tonic, True))

    candidates = []
    seen = set()
    for root, major in seeds:
        for r, m in ((root, major), ((root + 9) % 12, False), ((root + 3) % 12, True)):
            if (r, m) not in seen:
                seen.add((r, m))
                candidates.append((r, m))

    best = (classic_root, classic_major)
    best_score = -1.0
    for root, major in candidates:
        score = _key_candidate_score(
            root, major, segments, chroma_mean,
            classic_root, classic_major, profile_root, profile_major,
        )
        if score > best_score:
            best_score = score
            best = (root, major)

    if loop_tonic is not None and best[1]:
        loop_score = _key_candidate_score(
            loop_tonic, True, segments, chroma_mean,
            classic_root, classic_major, profile_root, profile_major,
        )
        if loop_score >= best_score * 0.82:
            best = (loop_tonic, True)
            best_score = loop_score

    # Prefer I over V when the winning root is the dominant, not the tonic (e.g. C# over G#).
    if best[1] and best[0] not in (classic_root, profile_root):
        tonic_pc = (best[0] - 7) % 12
        tonic_score = _key_candidate_score(
            tonic_pc, True, segments, chroma_mean,
            classic_root, classic_major, profile_root, profile_major,
        )
        if tonic_score >= best_score * 0.85:
            pop_tonic = _score_pop_loop_fit(segments, tonic_pc, True)
            pop_dominant = _score_pop_loop_fit(segments, best[0], True)
            if pop_tonic > pop_dominant * 0.92 and tonic_score >= best_score * 0.88:
                best = (tonic_pc, True)
                best_score = tonic_score

    # Same-root major/minor: follow sustained chord quality (e.g. Bad Guy → G minor).
    if best[1]:
        minor_score = _key_candidate_score(
            best[0], False, segments, chroma_mean,
            classic_root, classic_major, profile_root, profile_major,
        )
        minor_w = 0.0
        major_w = 0.0
        for seg in segments:
            chord = seg.get("chord", "")
            if not chord or chord == "N":
                continue
            if _root_to_pitch_class(_chord_root_name(chord)) != best[0]:
                continue
            dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
            q = _chord_quality(chord)
            if q in ("m", "m7"):
                minor_w += dur
            elif q in ("", "5", "7", "maj7", "sus2", "sus4"):
                major_w += dur
        if minor_w > major_w * 1.2 and minor_w > major_w + 1.0 and minor_score >= best_score * 0.82:
            best = (best[0], False)
            best_score = minor_score

    # Prefer chroma-stack key when segment vote wins but chroma profile disagrees.
    if (
        chords
        and best == (vote_root, vote_major)
        and (vote_root, vote_major) != (classic_root, classic_major)
    ):
        sc_classic = _chroma_profile_score(chroma_mean, classic_root, classic_major)
        sc_vote = _chroma_profile_score(chroma_mean, vote_root, vote_major)
        if sc_classic >= sc_vote * 1.05:
            classic_total = _key_candidate_score(
                classic_root, classic_major, segments, chroma_mean,
                classic_root, classic_major, profile_root, profile_major,
            )
            if classic_total >= best_score * 0.84:
                best = (classic_root, classic_major)
                best_score = classic_total

    mode = "major" if best[1] else "minor"
    return best[0], best[1], mode


def _dedupe_adjacent_segments(segments):
    """Merge consecutive segments with the same chord symbol."""
    if not segments:
        return segments

    out = [dict(segments[0])]
    for seg in segments[1:]:
        if seg.get("chord") == out[-1].get("chord"):
            out[-1]["end_time"] = float(seg.get("end_time", seg["time"]))
            out[-1]["confidence"] = max(
                float(out[-1].get("confidence", 0)),
                float(seg.get("confidence", 0)),
            )
            out[-1]["is_low_confidence"] = (
                out[-1]["confidence"] < 0.35 and seg.get("is_low_confidence", False)
            )
        else:
            out.append(dict(seg))

    for i in range(len(out) - 1):
        out[i]["end_time"] = float(out[i + 1]["time"])
    return out


def _finalize_timeline(segments, beat_duration=0.5, min_duration=None):
    """Dedupe adjacent chords and drop very brief flicker segments."""
    segments = _dedupe_adjacent_segments(segments)
    if len(segments) <= 1:
        return segments

    if min_duration is None:
        min_duration = _bar_aware_min_duration(beat_duration)
    min_duration = min(max(min_duration, beat_duration * 0.5), 1.4)
    pruned = [segments[0]]
    for seg in segments[1:]:
        prev = pruned[-1]
        duration = float(seg["time"]) - float(prev["time"])
        if duration < min_duration and len(pruned) > 1:
            prev["end_time"] = float(seg.get("end_time", seg["time"]))
            prev["confidence"] = max(float(prev.get("confidence", 0)), float(seg.get("confidence", 0)))
            if seg.get("chord") == prev.get("chord"):
                continue
            pruned[-1] = seg
        else:
            pruned.append(seg)

    for i in range(len(pruned) - 1):
        pruned[i]["end_time"] = float(pruned[i + 1]["time"])
    return pruned


def _windowed_key_for_frames(chroma_segments, bass_segments, chroma_mean, mid_segments=None, step=12):
    """Local key estimates over beat windows to tolerate brief modulations."""
    n_frames = chroma_segments.shape[1]
    frame_keys = [(0, True)] * n_frames
    if n_frames == 0:
        return frame_keys

    for start in range(0, n_frames, step):
        end = min(n_frames, start + step)
        window_chroma = np.mean(chroma_segments[:, start:end], axis=1)
        window_mid = None
        if mid_segments is not None:
            window_mid = mid_segments[:, start:end]
        _, raw_em = _frame_emission_matrix(
            chroma_segments[:, start:end],
            bass_segments[:, start:end],
            0, True,
            window_mid,
        )
        key_root, is_major = _resolve_key(
            window_chroma if end - start > 2 else chroma_mean,
            raw_em,
            bass_segments[:, start:end],
            chroma_segments[:, start:end],
        )
        for idx in range(start, end):
            frame_keys[idx] = (key_root, is_major)

    return frame_keys


def _nnls_frame_scores(chroma, nnls_matrix):
    """Non-negative least-squares fit of chroma to weighted harmonic templates."""
    import warnings

    chroma = np.clip(np.asarray(chroma, dtype=np.float64), 0, None)
    norm = np.linalg.norm(chroma)
    if norm < 1e-6:
        return None
    chroma = chroma / norm
    matrix = nnls_matrix / (np.linalg.norm(nnls_matrix, axis=0, keepdims=True) + 1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coeffs, _ = nnls(matrix, chroma)

    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(coeffs.sum())
    if total < 1e-8:
        return np.zeros(len(coeffs))
    return coeffs / total


def _estimate_bar_duration(beat_duration):
    """Assume 4/4 time for bar-aligned minimum segment length."""
    return beat_duration * 4


def _bar_aware_min_duration(beat_duration):
    """~5/8 bar minimum in 4/4 — reduces segment flicker (v25)."""
    target = float(beat_duration) * 2.5
    return min(max(target, 0.55), 1.4)


def _merge_adjacent_same_root(segments):
    """Merge consecutive segments that share a chord root (ML pre-polish)."""
    if len(segments) < 2:
        return segments

    out = [dict(segments[0])]
    for seg in segments[1:]:
        prev = out[-1]
        prev_root = _chord_root_name(prev.get("chord", "N"))
        seg_root = _chord_root_name(seg.get("chord", "N"))
        if (
            prev_root not in ("N", "")
            and seg_root == prev_root
        ):
            prev["end_time"] = float(seg.get("end_time", seg["time"]))
            prev["confidence"] = max(
                float(prev.get("confidence", 0)),
                float(seg.get("confidence", 0)),
            )
            if not str(prev.get("chord", "")).endswith("5") and str(seg.get("chord", "")).endswith("5"):
                prev["chord"] = prev.get("chord", seg.get("chord"))
            elif str(prev.get("chord", "")).endswith("5") and not str(seg.get("chord", "")).endswith("5"):
                prev["chord"] = seg.get("chord", prev.get("chord"))
            continue
        out.append(dict(seg))

    for i in range(len(out) - 1):
        out[i]["end_time"] = float(out[i + 1]["time"])
    if out:
        out[-1]["end_time"] = float(segments[-1].get("end_time", out[-1]["time"] + 1.0))
    return out


def _root_pitch_class(root_name):
    root = _internal_root(_chord_root_name(root_name))
    try:
        return ROOT_NAMES.index(root)
    except ValueError:
        return None


def _third_quality_bias(chroma, root_name):
    """
    Boost maj vs min (and extensions) from relative third strength in chroma.

    Audio-driven quality hint for a fixed root — not song-specific tuning.
    """
    root_pc = _root_pitch_class(root_name)
    if root_pc is None:
        return {}

    chroma = np.clip(np.asarray(chroma, dtype=np.float64), 0, None)
    total = float(chroma.sum())
    if total < 1e-6:
        return {}

    c = chroma / total
    min3 = float(c[(root_pc + 3) % 12])
    maj3 = float(c[(root_pc + 4) % 12])

    if min3 > maj3 * 1.10:
        return {"m": 1.14, "m7": 1.10, "": 0.92, "7": 0.96, "5": 0.98}
    if maj3 > min3 * 1.10:
        return {"": 1.12, "7": 1.06, "maj7": 1.04, "m": 0.90, "m7": 0.88, "5": 1.02}
    return {}


def _apply_third_quality_bias(vote, vocab, chroma, root_hint=None):
    """Apply third-interval bias to vote vector for likely roots in the region."""
    if root_hint and root_hint not in ("N", ""):
        roots = {_chord_root_name(root_hint)}
    else:
        roots = set()
        top = np.argsort(vote)[::-1][:6]
        for idx in top:
            name = vocab[int(idx)]
            if name != "N":
                roots.add(_chord_root_name(name))

    for root in roots:
        bias = _third_quality_bias(chroma, root)
        if not bias:
            continue
        for j, name in enumerate(vocab):
            if name == "N" or _chord_root_name(name) != root:
                continue
            q = _chord_quality(name)
            if q in bias:
                vote[j] *= bias[q]
    return vote


def _downbeat_indices(beat_frames, y_harmonic, sr, hop_length):
    """Approximate downbeat positions on the beat grid (4/4)."""
    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=hop_length)
    fixed = librosa.util.fix_frames(beat_frames, x_min=0, x_max=len(onset_env) - 1)
    if len(fixed) == 0:
        return set()
    strengths = onset_env[fixed]
    anchor = int(np.argmax(strengths[:min(32, len(strengths))]))
    return {i for i in range(len(beat_frames)) if (i - anchor) % 4 == 0}


def _derive_key_from_segments(segments, duration_weighted=False):
    """Re-estimate key from the final decoded chord sequence."""
    if not segments:
        return None

    weighted = []
    for seg in segments:
        chord = seg.get("chord")
        if not chord or chord == "N":
            continue
        if duration_weighted:
            duration = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
            repeat = max(1, int(duration * 2))
            weighted.extend([chord] * repeat)
        else:
            weighted.append(chord)

    if not weighted:
        return None
    return _estimate_key_from_chord_votes(weighted)


def _score_frame(chroma, bass, matrix_n, vocab, key_root, is_major, chroma_mid=None, nnls_matrix=None, bass_weight=0.30):
    chroma = np.clip(np.asarray(chroma, dtype=np.float64), 0, None)
    norm = np.linalg.norm(chroma)
    if norm < 1e-6:
        scores = np.zeros(len(vocab))
        scores[NO_CHORD_IDX] = 1.0
        return scores

    chroma_n = chroma / norm
    chroma_n = np.nan_to_num(chroma_n, nan=0.0, posinf=0.0, neginf=0.0)
    scores = chroma_n @ matrix_n
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if nnls_matrix is not None:
        nnls_scores = _nnls_frame_scores(chroma, nnls_matrix)
        if nnls_scores is not None:
            scores = 0.58 * nnls_scores + 0.42 * scores

    if chroma_mid is not None:
        mid = np.clip(np.asarray(chroma_mid, dtype=np.float64), 0, None)
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-6:
            mid_n = mid / mid_norm
            mid_n = np.nan_to_num(mid_n, nan=0.0, posinf=0.0, neginf=0.0)
            scores += 0.14 * (mid_n @ matrix_n)

    if bass is not None:
        bass_norm = np.linalg.norm(bass)
        if bass_norm > 1e-6:
            bass_n = bass / bass_norm
            for i, name in enumerate(vocab):
                if name == 'N':
                    continue
                root = _chord_root_name(name)
                scores[i] += bass_weight * float(bass_n[ROOT_NAMES.index(root)])

    diatonic = set(_diatonic_chord_names(key_root, is_major))
    for i, name in enumerate(vocab):
        if name in diatonic:
            scores[i] *= 1.04
        elif name != 'N':
            scores[i] *= 0.97

    if vocab is DECODE_VOCAB:
        scores = _apply_simplicity_prior(scores, vocab)

    return scores


def _frame_emission_matrix(chroma_frames, bass_frames, key_root, is_major, mid_frames=None):
    """Emission scores for each beat × decode vocabulary entry."""
    n_frames = chroma_frames.shape[1]
    emissions = np.zeros((n_frames, N_DECODE))

    for t in range(n_frames):
        bass = bass_frames[:, t] if bass_frames is not None else None
        mid = mid_frames[:, t] if mid_frames is not None else None
        emissions[t] = _score_frame(
            chroma_frames[:, t], bass, DECODE_MATRIX_N, DECODE_VOCAB,
            key_root, is_major, chroma_mid=mid, nnls_matrix=DECODE_NNLS_MATRIX,
        )

    row_max = emissions.max(axis=1, keepdims=True)
    log_em = np.log(emissions + 1e-8) - np.log(row_max + 1e-8)
    return log_em, emissions


def _frame_emission_matrix_windowed(chroma_frames, bass_frames, mid_frames, frame_keys):
    """Per-beat emissions using window-local key bias."""
    n_frames = chroma_frames.shape[1]
    emissions = np.zeros((n_frames, N_DECODE))

    for t in range(n_frames):
        key_root, is_major = frame_keys[t]
        bass = bass_frames[:, t] if bass_frames is not None else None
        mid = mid_frames[:, t] if mid_frames is not None else None
        emissions[t] = _score_frame(
            chroma_frames[:, t], bass, DECODE_MATRIX_N, DECODE_VOCAB,
            key_root, is_major, chroma_mid=mid, nnls_matrix=DECODE_NNLS_MATRIX,
        )

    row_max = emissions.max(axis=1, keepdims=True)
    log_em = np.log(emissions + 1e-8) - np.log(row_max + 1e-8)
    return log_em, emissions


def _estimate_key_from_emissions(raw_emissions):
    """Estimate key from per-frame chord scores (major and minor)."""
    root_votes = Counter()
    for t in range(raw_emissions.shape[0]):
        top_idx = int(np.argmax(raw_emissions[t]))
        name = DECODE_VOCAB[top_idx]
        if name != 'N':
            root_votes[_chord_root_name(name)] += 1

    if not root_votes:
        return 0, True

    best_key, best_major, best_score = 0, True, -1
    for key_root in range(12):
        for major in (True, False):
            diatonic = _diatonic_chord_names(key_root, major)
            diatonic_roots = {_chord_root_name(c) for c in diatonic}
            score = sum(root_votes.get(r, 0) for r in diatonic_roots)
            if score > best_score:
                best_score = score
                best_key = key_root
                best_major = major

    return best_key, best_major


def _estimate_key_from_root_histogram(bass_segments, chroma_segments):
    """Major-key vote from recurring bass/chroma roots (works for power-chord rock)."""
    root_counts = Counter()
    for i in range(bass_segments.shape[1]):
        bass = bass_segments[:, i]
        chroma = chroma_segments[:, i]
        if np.max(bass) > 0:
            root_counts[int(np.argmax(bass))] += 1.0
        if np.max(chroma) > 0:
            root_counts[int(np.argmax(chroma))] += 0.35

    if not root_counts:
        return 0, True

    best_key, best_score = 0, -1.0
    for key_root in range(12):
        iv_roots = [(key_root + x) % 12 for x in (0, 5, 7)]
        score = sum(root_counts.get(r, 0) for r in iv_roots)
        tonic_count = root_counts.get(key_root, 0)
        score += tonic_count * 0.5
        if score > best_score:
            best_score = score
            best_key = key_root

    return best_key, True


def _resolve_key(chroma_mean, raw_emissions, bass_segments, chroma_segments):
    vote_key, vote_major = _estimate_key_from_emissions(raw_emissions)
    profile_key, profile_major = _estimate_key(chroma_mean, major_only=False)
    hist_key, hist_major = _estimate_key_from_root_histogram(bass_segments, chroma_segments)

    candidates = [
        (hist_key, True, 1.2),
        (vote_key, vote_major, 1.0),
        (profile_key, profile_major, 0.9),
    ]

    best = candidates[0][:2]
    best_score = -1.0
    for key_root, is_major, weight in candidates:
        diatonic_roots = {_chord_root_name(c) for c in _diatonic_chord_names(key_root, is_major)}
        frame_score = 0
        for t in range(raw_emissions.shape[0]):
            top_idx = int(np.argmax(raw_emissions[t]))
            name = DECODE_VOCAB[top_idx]
            if name != 'N' and _chord_root_name(name) in diatonic_roots:
                frame_score += 1
        frame_score /= max(raw_emissions.shape[0], 1)
        total = frame_score * weight
        if total > best_score:
            best_score = total
            best = (key_root, is_major)

    return best


def _build_key_info(key_root, is_major, mode=None):
    root = ROOT_NAMES[key_root]
    if mode == "dorian":
        return {"root": root, "mode": "dorian", "display": f"{root} Dorian"}
    if mode == "mixolydian":
        return {"root": root, "mode": "mixolydian", "display": f"{root} Mixolydian"}
    mode_name = "major" if is_major else "minor"
    display = f"{root} {'Major' if is_major else 'Minor'}"
    return {"root": root, "mode": mode_name, "display": display}


def _detect_dorian_from_segments(segments):
    """
    Detect B-Dorian-style loops: minor i, major III, major IV, minor v
    (e.g. Get Lucky: Bm7–D–F#m7–E).
    """
    if not segments:
        return None

    root_weight = Counter()
    root_qualities = {}
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        root = _chord_root_name(chord)
        pc = _root_to_pitch_class(root)
        if pc is None:
            continue
        dur = max(float(seg.get("end_time", 0)) - float(seg.get("time", 0)), 0.25)
        root_weight[pc] += dur
        root_qualities.setdefault(pc, Counter())
        root_qualities[pc][_chord_quality(chord)] += dur

    if len(root_weight) < 4:
        return None

    top_pcs = {pc for pc, _ in root_weight.most_common(6)}
    total = sum(root_weight.values())
    min_share = total * 0.10
    best_tonic = None
    best_score = 0.0

    for tonic in top_pcs:
        third = (tonic + 3) % 12
        fourth = (tonic + 5) % 12
        fifth = (tonic + 7) % 12
        loop = [tonic, third, fourth, fifth]
        if not all(pc in top_pcs for pc in loop):
            continue
        if any(root_weight.get(pc, 0) < min_share for pc in loop):
            continue

        # Major IV is the Dorian hallmark (not iv minor).
        iv_major = sum(
            w for q, w in root_qualities.get(fourth, Counter()).items()
            if _chord_quality_matches(q, "")
        )
        if iv_major < min_share * 0.5:
            continue

        score = (
            root_weight.get(tonic, 0) * 1.5
            + root_weight.get(third, 0)
            + root_weight.get(fourth, 0) * 1.4
            + root_weight.get(fifth, 0)
        )
        if score > best_score:
            best_score = score
            best_tonic = tonic

    if best_tonic is None or best_score < total * 0.55:
        return None

    # Prefer Dorian only when it beats a straight major-key reading.
    major_vote = _estimate_key_from_chord_votes(
        [s["chord"] for s in segments if s.get("chord") and s["chord"] != "N"]
    )
    if major_vote[1]:
        major_root = major_vote[0]
        major_diatonic = {(major_root + iv) % 12 for iv in DIATONIC_MAJOR_INTERVALS}
        major_score = sum(root_weight.get(pc, 0) for pc in major_diatonic)
        if major_score >= best_score * 0.92:
            return None

    return best_tonic, "dorian"


def _extract_chroma_stack(y, sr, hop_length=512):
    from harmonic_separation import extract_harmonic

    y_harmonic = extract_harmonic(y, sr)
    chroma_low = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=hop_length,
        fmin=librosa.note_to_hz('C2'), n_octaves=4,
    )
    chroma_mid = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=hop_length,
        fmin=librosa.note_to_hz('C3'), n_octaves=3,
    )
    min_len = min(chroma_low.shape[1], chroma_mid.shape[1])
    chroma = 0.55 * chroma_low[:, :min_len] + 0.45 * chroma_mid[:, :min_len]
    chroma = np.clip(chroma, 0, None)
    chroma = librosa.util.normalize(chroma, axis=0)
    chroma = median_filter(chroma, size=(1, 5))
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    chroma = librosa.util.normalize(chroma, axis=0)
    return y_harmonic, chroma, chroma_low[:, :min_len], chroma_mid[:, :min_len]


def _estimate_key(chroma_mean, major_only=True):
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)
    best_corr = -np.inf
    best_key = (0, True)

    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        major_corr = float(np.dot(rolled, MAJOR_KEY_PROFILE))
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = (shift, True)
        if not major_only:
            minor_corr = float(np.dot(rolled, MINOR_KEY_PROFILE))
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = (shift, False)

    return best_key


def _diatonic_triad_map(key_root, is_major):
    """Map pitch class -> expected triad quality ('' major, 'm' minor)."""
    if is_major:
        pairs = zip(DIATONIC_MAJOR_INTERVALS, DIATONIC_MAJOR_QUALITIES)
    else:
        pairs = zip(DIATONIC_MINOR_INTERVALS, ['m', '', '', 'm', 'm', '', ''])
    return {((key_root + iv) % 12): q for iv, q in pairs}


def _chord_quality_matches(chord_quality, expected):
    if expected == 'm':
        return chord_quality == 'm'
    return chord_quality in ('', 'maj', '5', '7', 'maj7', 'sus2', 'sus4')


def _key_vote_score(parsed, key_root, is_major):
    """Score how well chord roots/qualities fit a key candidate."""
    triads = _diatonic_triad_map(key_root, is_major)
    score = 0.0
    for pc, quality in parsed:
        expected = triads.get(pc)
        if expected is None:
            continue
        score += 1.0
        if _chord_quality_matches(quality, expected):
            score += 0.35
    return score


def _estimate_key_from_chord_votes(bar_chords):
    """Find the key whose diatonic triads best explain observed chord roots."""
    if not bar_chords:
        return 5, True

    parsed = []
    for chord in bar_chords:
        root = _chord_root_name(chord)
        pc = _root_to_pitch_class(root)
        if pc is not None:
            parsed.append((pc, _chord_quality(chord)))

    if not parsed:
        return 5, True

    root_counts = Counter(pc for pc, _ in parsed)
    top_pc, top_count = root_counts.most_common(1)[0]
    best_key = 5
    best_is_major = True
    best_score = -1.0

    for key_root in range(12):
        for is_major in (True, False):
            score = _key_vote_score(parsed, key_root, is_major)

            if top_pc == key_root:
                score += top_count * 0.5

            if score > best_score:
                best_score = score
                best_key = key_root
                best_is_major = is_major

    # Prefer minor tonic when vi-of-major dominates as a minor chord (Wonderwall: F#m vs A).
    if best_is_major:
        vi_pc = (best_key + 9) % 12
        vi_minor_w = sum(
            1.0 + (0.35 if q in ('m', 'm7') else 0)
            for pc, q in parsed if pc == vi_pc
        )
        i_major_w = sum(
            1.0 + (0.35 if _chord_quality_matches(q, '') else 0)
            for pc, q in parsed if pc == best_key
        )
        if (
            vi_pc == top_pc
            and vi_minor_w > max(i_major_w * 1.35, 2.0)
        ):
            minor_score = _key_vote_score(parsed, vi_pc, False)
            minor_score += root_counts.get(vi_pc, 0) * 0.65
            if minor_score >= best_score * 0.88:
                best_key, best_is_major = vi_pc, False

    # Prefer natural minor when the most-used minor root should be tonic (Hotel California: Bm).
    minor_root_w = Counter()
    for pc, q in parsed:
        if q in ('m', 'm7'):
            minor_root_w[pc] += 1
    if minor_root_w:
        top_minor_pc, top_minor_count = minor_root_w.most_common(1)[0]
        if top_minor_count >= 3:
            minor_score = _key_vote_score(parsed, top_minor_pc, False)
            minor_score += top_minor_count * 0.55
            current = _key_vote_score(parsed, best_key, best_is_major)
            if best_is_major:
                current += root_counts.get(best_key, 0) * 0.5
            if minor_score >= current * 0.92 and top_minor_pc != best_key:
                best_key, best_is_major = top_minor_pc, False

    return best_key, best_is_major


def _diatonic_major_roots(key_root):
    """I, IV, V roots for a major key — common rock/power-chord vocabulary."""
    return [ROOT_NAMES[(key_root + iv) % 12] for iv in (0, 5, 7)]


def _two_bar_iiiv_pattern(key_root):
    """
    Standard 2-bar I–V–IV loop (Baba O'Riley / sheet music):
    Bar 1: I (beats 1–2), V (beats 3–4)  |  Bar 2: IV (beats 1–4)
    """
    tonic, subdominant, dominant = _diatonic_major_roots(key_root)
    return [tonic, tonic, dominant, dominant,
            subdominant, subdominant, subdominant, subdominant]


def _beat_root_scores(chroma_segments, bass_segments, roots, prefer_power):
    beat_scores = []
    for i in range(chroma_segments.shape[1]):
        scores = {}
        for root in roots:
            _, score, _ = _score_chord_frame(
                chroma_segments[:, i], [root], prefer_power=prefer_power,
                bass_chroma=bass_segments[:, i],
            )
            scores[root] = score
        beat_scores.append(scores)
    return beat_scores


def _find_pattern_phase(beat_scores, pattern):
    cycle = len(pattern)
    tonic = pattern[0]
    subdominant = pattern[4]

    valid_phases = [
        phase for phase in range(cycle)
        if pattern[phase % cycle] == tonic and pattern[(4 + phase) % cycle] == subdominant
    ]
    if not valid_phases:
        valid_phases = [0]

    skip = max(len(beat_scores) // 5, 0)
    search = range(skip, len(beat_scores))

    best_phase = valid_phases[0]
    best_score = -1.0
    for phase in valid_phases:
        score = sum(
            beat_scores[i].get(pattern[(i + phase) % cycle], 0.0)
            for i in search
        )
        if score > best_score:
            best_score = score
            best_phase = phase

    return best_phase, best_score


def _iv_ratio_in_votes(beat_scores, roots):
    if not beat_scores:
        return 0.0
    hits = 0
    for scores in beat_scores:
        if not scores:
            continue
        if max(scores, key=scores.get) in roots:
            hits += 1
    return hits / len(beat_scores)


def _pattern_agreement(beat_scores, pattern, phase):
    if not beat_scores:
        return 0.0
    hits = 0
    for i, scores in enumerate(beat_scores):
        expected = pattern[(i + phase) % len(pattern)]
        if scores and max(scores, key=scores.get) == expected:
            hits += 1
    return hits / len(beat_scores)


def _chord_display_name(internal_root, prefer_power, key_root, is_major):
    name = f"{internal_root}5" if prefer_power else internal_root
    return _format_chord_name(name, key_root, is_major)


def _strum_for_duration(beats, prefer_power=True):
    """Sheet rhythm: 2-beat I/V get two downstrokes; 4-beat IV gets four."""
    if beats >= 3.5:
        return "D D D D"
    return "D D"


def _chroma_flux_series(chroma):
    """Normalized per-frame harmonic change strength."""
    n_frames = chroma.shape[1]
    flux = np.zeros(n_frames, dtype=np.float64)
    if n_frames < 2:
        return flux
    for i in range(1, n_frames):
        flux[i] = float(np.linalg.norm(chroma[:, i] - chroma[:, i - 1]))
    positive = flux[flux > 0]
    scale = float(np.percentile(positive, 90)) if positive.size else 1.0
    return flux / max(scale, 1e-6)


def _flux_peak_frames(flux, sr, hop, beat_duration):
    """Frame indices where chroma changes sharply — candidate chord boundaries."""
    if flux.size < 3:
        return np.array([], dtype=int)
    min_distance = max(1, int((beat_duration * 0.32) * sr / hop))
    height = float(np.percentile(flux[flux > 0], 52)) if np.any(flux > 0) else 0.05
    peaks, _ = find_peaks(flux, height=height, distance=min_distance)
    return peaks


def _flux_at_time(flux, time_sec, sr, hop):
    frame = int(librosa.time_to_frames(time_sec, sr=sr, hop_length=hop))
    frame = max(0, min(frame, len(flux) - 1))
    return float(flux[frame])


def _boundary_candidate_times(approx, flux_peak_times, onset_times, beat_duration, search_sec):
    """Times near approx where a chord change is likely (flux peak and/or onset)."""
    candidates = {float(approx)}
    for t in flux_peak_times:
        if abs(t - approx) <= search_sec:
            candidates.add(float(t))
    onset_window = min(search_sec, max(beat_duration * 0.28, 0.08))
    for t in onset_times:
        if abs(t - approx) <= onset_window:
            candidates.add(float(t))
    return sorted(candidates)


def _pick_chord_for_region(
    chroma,
    chroma_low,
    chroma_mid,
    start_sec,
    end_sec,
    sr,
    hop,
    key_root,
    is_major,
    quality_bonus=None,
    frame_keys=None,
    beat_times=None,
    hint_root=None,
    ml_root_bias=1.0,
    edge_trim=0.18,
):
    """
    Label a time region by summing per-frame template scores (robust to boundary bleed).
    """
    start_f = int(librosa.time_to_frames(start_sec, sr=sr, hop_length=hop))
    end_f = min(int(librosa.time_to_frames(end_sec, sr=sr, hop_length=hop)), chroma.shape[1])
    if end_f <= start_f + 1:
        start_f = max(0, start_f)
        end_f = min(chroma.shape[1], start_f + 2)

    span = end_f - start_f
    trim_f = int(span * edge_trim)
    start_f = min(start_f + trim_f, end_f - 1)
    end_f = max(end_f - trim_f, start_f + 1)

    vote = np.zeros(len(CHORD_VOCAB), dtype=np.float64)
    n_frames = 0
    for f in range(start_f, end_f):
        bass = chroma_low[:, f] if f < chroma_low.shape[1] else None
        mid = chroma_mid[:, f] if chroma_mid is not None and f < chroma_mid.shape[1] else None
        scores = _score_frame(
            chroma[:, f], bass, CHORD_MATRIX_N, CHORD_VOCAB,
            key_root, is_major, chroma_mid=mid, nnls_matrix=CHORD_NNLS_MATRIX,
            bass_weight=0.38,
        )
        vote += scores
        n_frames += 1

    mean_chroma = np.mean(chroma[:, start_f:end_f], axis=1)
    mean_bass = np.mean(chroma_low[:, start_f:end_f], axis=1)
    mean_mid = np.mean(chroma_mid[:, start_f:end_f], axis=1) if chroma_mid is not None else None
    mean_scores = _score_frame(
        mean_chroma, mean_bass, CHORD_MATRIX_N, CHORD_VOCAB,
        key_root, is_major, chroma_mid=mean_mid, nnls_matrix=CHORD_NNLS_MATRIX,
        bass_weight=0.38,
    )
    if n_frames > 0:
        vote = 0.62 * vote + 0.38 * (mean_scores * n_frames)

    vote = _apply_third_quality_bias(vote, CHORD_VOCAB, mean_chroma, root_hint=hint_root)

    if quality_bonus:
        for j, name in enumerate(CHORD_VOCAB):
            if name == "N":
                continue
            vote[j] *= quality_bonus.get(_chord_quality(name), 1.0)

    best_idx = _pick_best_chord_index(vote, CHORD_VOCAB)
    best_score = float(vote[best_idx])
    if ml_root_bias > 1.0 and hint_root and hint_root != "N":
        hint = _internal_root(hint_root)
        ml_indices = [
            j for j, name in enumerate(CHORD_VOCAB)
            if name != "N" and _internal_root(_chord_root_name(name)) == hint
        ]
        if ml_indices:
            ml_best = max(float(vote[j]) for j in ml_indices)
            if ml_best >= best_score * 0.88:
                for j in ml_indices:
                    vote[j] *= ml_root_bias
                best_idx = _pick_best_chord_index(vote, CHORD_VOCAB)
                best_score = float(vote[best_idx])

    confidence = best_score / max(n_frames, 1)
    return CHORD_VOCAB[best_idx], confidence


def _snap_segments_to_onsets(segments, onset_times, beat_duration, downbeat_times=None):
    """
    Shift segment starts to the nearest strum onset within a small window.

    Beat tracking and ML boundaries often sit slightly before or after the heard
    attack; symmetric snapping avoids systematically early chord cards.
    """
    if not segments:
        return segments

    onsets = sorted(float(t) for t in onset_times)
    backward_window = min(max(beat_duration * 0.12, 0.04), 0.10)
    forward_window = min(max(beat_duration * 0.30, 0.06), 0.20)
    min_gap = min(max(beat_duration * 0.15, 0.04), 0.12)
    bar_window = beat_duration * 0.45
    downbeats = sorted(float(t) for t in (downbeat_times or []))

    snapped = []
    for seg in segments:
        seg = dict(seg)
        t = float(seg["time"])
        nearby = [o for o in onsets if t - backward_window <= o <= t + forward_window]
        if nearby:
            t = min(nearby, key=lambda o: abs(o - t))
        if downbeats:
            bar_candidates = [d for d in downbeats if abs(d - t) <= bar_window]
            if bar_candidates:
                db = min(bar_candidates, key=lambda d: abs(d - t))
                if abs(db - t) <= bar_window * 0.65:
                    t = db
        if snapped and t < snapped[-1]["time"] + min_gap:
            t = snapped[-1]["time"] + min_gap
        seg["time"] = t
        snapped.append(seg)

    for i in range(len(snapped) - 1):
        snapped[i]["end_time"] = float(snapped[i + 1]["time"])
    if snapped:
        last = snapped[-1]
        prev_end = float(last.get("end_time", last["time"] + beat_duration))
        last["end_time"] = max(prev_end, last["time"] + beat_duration)

    return snapped


def _segment_local_key(chroma, bass, chroma_mid=None):
    """Key estimate from a single segment's chroma (avoids global key mismatch)."""
    chroma_mean = np.asarray(chroma, dtype=np.float64)
    if chroma_mean.ndim > 1:
        chroma_mean = np.mean(chroma_mean, axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)
    return _estimate_key(chroma_mean, major_only=False)


def _segment_center_chroma(chroma, chroma_low, chroma_mid, start_sec, end_sec, sr, hop=512, trim=0.22):
    """Mean chroma over the stable center of a segment (excludes transition bleed)."""
    start_f = int(librosa.time_to_frames(start_sec, sr=sr, hop_length=hop))
    end_f = int(librosa.time_to_frames(end_sec, sr=sr, hop_length=hop))
    end_f = min(end_f, chroma.shape[1])
    if end_f <= start_f + 2:
        start_f = max(0, start_f)
        end_f = min(chroma.shape[1], start_f + 2)
    span = end_f - start_f
    trim_f = int(span * trim)
    start_f = min(start_f + trim_f, end_f - 1)
    end_f = max(end_f - trim_f, start_f + 1)

    c = np.mean(chroma[:, start_f:end_f], axis=1)
    b = np.mean(chroma_low[:, start_f:end_f], axis=1)
    m = np.mean(chroma_mid[:, start_f:end_f], axis=1) if chroma_mid is not None else None
    return c, b, m


def _match_score_for_chord(chroma, bass, chroma_mid, chord_name, key_root, is_major, bass_weight=0.0):
    """How well a chroma vector matches a chord label."""
    if not chord_name or chord_name == "N":
        return 0.0

    scores = _score_frame(
        chroma, bass, CHORD_MATRIX_N, CHORD_VOCAB,
        key_root, is_major, chroma_mid=chroma_mid, nnls_matrix=CHORD_NNLS_MATRIX,
        bass_weight=0.36 if bass_weight > 0 else 0.0,
    )
    target_root = _internal_root(_chord_root_name(chord_name))
    target_q = _chord_quality(chord_name)
    best = 0.0
    for j, name in enumerate(CHORD_VOCAB):
        if name == "N":
            continue
        if _internal_root(_chord_root_name(name)) != target_root:
            continue
        q = _chord_quality(name)
        if target_q and q != target_q:
            continue
        if not target_q and q not in ("", "5"):
            continue
        best = max(best, float(scores[j]))

    if bass_weight > 0 and bass is not None and target_root is not None:
        bass_arr = np.clip(np.asarray(bass, dtype=np.float64), 0, None)
        bt = float(bass_arr.sum())
        if bt > 1e-6:
            bass_peak = int(np.argmax(bass_arr))
            if bass_peak == target_root:
                best *= 1.0 + bass_weight

    return best


def _optimize_segment_boundaries(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    sr,
    flux,
    flux_peak_times,
    onset_times,
    beat_duration,
    hop=512,
    search_sec=None,
):
    """
    Place each internal boundary at a nearby harmonic-change peak that best
    separates the existing chord labels on either side.
    """
    if len(segments) < 2:
        return segments

    if search_sec is None:
        search_sec = min(max(beat_duration * 0.42, 0.14), 0.38)

    out = [dict(segments[0])]
    track_end = chroma.shape[1] * hop / sr

    for i in range(1, len(segments)):
        seg = dict(segments[i])
        prev = out[-1]
        approx = float(seg["time"])
        seg_end = float(seg.get("end_time", approx + 1.0))
        left_name = prev.get("chord", "N")
        right_name = seg.get("chord", "N")

        if not left_name or not right_name or left_name == right_name:
            seg["time"] = approx
            out.append(seg)
            continue

        t_lo = max(float(prev["time"]) + 0.08, approx - search_sec)
        t_hi = min(track_end, approx + search_sec)
        candidates = [
            t for t in _boundary_candidate_times(
                approx, flux_peak_times, onset_times, beat_duration, search_sec,
            )
            if t_lo <= t <= t_hi
        ]
        if not candidates:
            candidates = [approx]

        best_t = approx
        best_score = -1.0

        for t in candidates:
            left_c, left_b, left_m = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(prev["time"]), t, sr, hop, trim=0.08,
            )
            right_c, right_b, right_m = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, t, seg_end, sr, hop, trim=0.08,
            )
            key_l = _key_at_time((float(prev["time"]) + t) / 2.0, beat_times, frame_keys)
            key_r = _key_at_time((t + seg_end) / 2.0, beat_times, frame_keys)
            label_score = (
                _match_score_for_chord(left_c, left_b, left_m, left_name, key_l[0], key_l[1])
                + _match_score_for_chord(right_c, right_b, right_m, right_name, key_r[0], key_r[1])
            )
            flux_bonus = _flux_at_time(flux, t, sr, hop)
            score = label_score * (1.0 + 0.35 * flux_bonus)
            if score > best_score:
                best_score = score
                best_t = t

        prev["end_time"] = best_t
        seg["time"] = best_t
        out.append(seg)

    if out:
        out[-1]["end_time"] = float(segments[-1].get("end_time", out[-1]["time"] + 1.0))
    return out


def _snap_boundaries_to_harmonic_changes(
    segments,
    flux,
    flux_peak_times,
    onset_times,
    beat_duration,
    sr,
    hop=512,
):
    """Nudge internal boundaries onto the strongest nearby harmonic-change peak."""
    if len(segments) < 2:
        return segments

    search_sec = min(max(beat_duration * 0.36, 0.12), 0.32)
    out = [dict(segments[0])]

    for i in range(1, len(segments)):
        seg = dict(segments[i])
        approx = float(seg["time"])
        candidates = _boundary_candidate_times(
            approx, flux_peak_times, onset_times, beat_duration, search_sec,
        )
        best_t = approx
        best_flux = _flux_at_time(flux, approx, sr, hop)
        for t in candidates:
            val = _flux_at_time(flux, t, sr, hop)
            if any(abs(t - o) <= 0.045 for o in onset_times):
                val *= 1.12
            if val > best_flux:
                best_flux = val
                best_t = t
        out[-1]["end_time"] = best_t
        seg["time"] = best_t
        out.append(seg)

    if out:
        out[-1]["end_time"] = float(segments[-1].get("end_time", out[-1]["time"] + 1.0))
    return out


def _quality_bonus_for_song(prefer_sevenths=False, prefer_power=False):
    if prefer_sevenths:
        return {
            '': 1.04, 'm': 0.76, '5': 1.0,
            'sus4': 0.84, 'sus2': 0.84,
            '7': 0.90, 'maj7': 0.86, 'm7': 1.18,
        }
    if prefer_power:
        return {
            '': 1.0, 'm': 0.88, '5': 1.14,
            'sus4': 0.90, 'sus2': 0.90,
            '7': 0.94, 'maj7': 0.92, 'm7': 0.90,
        }
    return {
        '': 1.05, 'm': 1.05, '5': 1.02,
        'sus4': 0.86, 'sus2': 0.86,
        '7': 0.92, 'maj7': 0.90, 'm7': 0.92,
    }


def _bar_aware_merge_short_segments(
    segments,
    beat_duration,
    flux,
    sr,
    hop,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    flux_keep_threshold=None,
    match_keep_threshold=None,
    min_duration_override=None,
):
    """
    Merge sub-half-bar flicker unless flux and chroma support a real change.

    Keeps short segments when harmonic flux is strong and both neighbors
    match their labels in the merged audio.
    """
    if len(segments) < 2:
        return segments

    if flux_keep_threshold is None:
        flux_keep_threshold = CHORD_POLISH_DEFAULTS["flux_keep_threshold"]
    if match_keep_threshold is None:
        match_keep_threshold = CHORD_POLISH_DEFAULTS["match_keep_threshold"]

    min_dur = (
        min_duration_override
        if min_duration_override is not None
        else _bar_aware_min_duration(beat_duration)
    )
    merged = [dict(segments[0])]

    for seg in segments[1:]:
        prev = merged[-1]
        duration = float(seg["time"]) - float(prev["time"])
        if duration >= min_dur:
            merged.append(dict(seg))
            continue

        boundary_t = float(seg["time"])
        merge_end = float(seg.get("end_time", boundary_t + beat_duration))
        left_name = prev.get("chord", "N")
        right_name = seg.get("chord", "N")
        left_root = _chord_root_name(left_name)
        right_root = _chord_root_name(right_name)

        flux_val = _flux_at_time(flux, boundary_t, sr, hop) if flux is not None else 0.0
        keep_split = False
        if (
            flux_val >= flux_keep_threshold
            and left_root not in ("N", "")
            and right_root not in ("N", "")
            and left_root != right_root
        ):
            if frame_keys is not None and beat_times is not None:
                key_l = _key_at_time((float(prev["time"]) + boundary_t) / 2.0, beat_times, frame_keys)
                key_r = _key_at_time((boundary_t + merge_end) / 2.0, beat_times, frame_keys)
            else:
                key_l = key_r = (0, True)

            lc, lb, lm = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(prev["time"]), boundary_t, sr, hop,
            )
            rc, rb, rm = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, boundary_t, merge_end, sr, hop,
            )
            score_l = _match_score_for_chord(lc, lb, lm, left_name, key_l[0], key_l[1])
            score_r = _match_score_for_chord(rc, rb, rm, right_name, key_r[0], key_r[1])
            keep_split = score_l >= match_keep_threshold and score_r >= match_keep_threshold

        if keep_split:
            merged.append(dict(seg))
            continue

        if frame_keys is not None and beat_times is not None:
            key_m = _key_at_time((float(prev["time"]) + merge_end) / 2.0, beat_times, frame_keys)
        else:
            key_m = (0, True)

        mc, mb, mm = _segment_center_chroma(
            chroma, chroma_low, chroma_mid, float(prev["time"]), merge_end, sr, hop,
        )
        score_prev = _match_score_for_chord(mc, mb, mm, left_name, key_m[0], key_m[1])
        score_next = _match_score_for_chord(mc, mb, mm, right_name, key_m[0], key_m[1])
        prev["end_time"] = merge_end
        if score_next > score_prev:
            prev["chord"] = right_name
            prev["confidence"] = max(float(prev.get("confidence", 0)), float(seg.get("confidence", 0)))

    for i in range(len(merged) - 1):
        merged[i]["end_time"] = float(merged[i + 1]["time"])
    if merged:
        merged[-1]["end_time"] = float(segments[-1].get("end_time", merged[-1]["time"] + beat_duration))
    return merged


def _polish_segment_boundaries(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    onset_times,
    beat_duration,
    sr,
    hop=512,
):
    """Move segment boundaries to flux/onset peaks; labels stay fixed."""
    if not segments:
        return segments, None

    flux = _chroma_flux_series(chroma)
    flux_peak_times = librosa.frames_to_time(
        _flux_peak_frames(flux, sr, hop, beat_duration),
        sr=sr,
        hop_length=hop,
    ).tolist()

    segments = _optimize_segment_boundaries(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr,
        flux, flux_peak_times, onset_times, beat_duration, hop,
    )
    segments = _snap_boundaries_to_harmonic_changes(
        segments, flux, flux_peak_times, onset_times, beat_duration, sr, hop,
    )
    return segments, flux


def _polish_segment_labels(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    key_root,
    is_major,
    prefer_sevenths=False,
    prefer_power=False,
    ml_root_bias=1.0,
    label_inertia=None,
):
    """Re-score chord symbols for fixed segment windows."""
    if label_inertia is None:
        label_inertia = CHORD_POLISH_DEFAULTS["label_inertia"]
    if not segments:
        return segments

    quality_bonus = _quality_bonus_for_song(prefer_sevenths, prefer_power)
    segments = _refine_segments_with_vocabulary(
        segments, chroma, chroma_low, key_root, is_major,
        prefer_sevenths=prefer_sevenths,
        prefer_power=prefer_power,
        chroma_mid=chroma_mid,
        frame_keys=frame_keys,
        beat_times=beat_times,
        ml_root_bias=ml_root_bias,
        quality_bonus=quality_bonus,
        label_inertia=label_inertia,
    )
    segments = _cross_validate_adjacent_segments(
        segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, 22050, 512,
    )
    segments = _refine_segments_with_vocabulary(
        segments, chroma, chroma_low, key_root, is_major,
        prefer_sevenths=prefer_sevenths,
        prefer_power=prefer_power,
        chroma_mid=chroma_mid,
        frame_keys=frame_keys,
        beat_times=beat_times,
        ml_root_bias=1.0,
        quality_bonus=quality_bonus,
        label_inertia=label_inertia,
    )
    return segments


def _polish_chord_timeline(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    onset_times,
    beat_duration,
    sr,
    hop=512,
    key_root=0,
    is_major=True,
    prefer_sevenths=False,
    prefer_power=False,
    ml_root_bias=1.0,
):
    """
    Audio-driven polish: boundaries first, then labels, then bar-aware merge.
    """
    if not segments:
        return segments

    segments, flux = _polish_segment_boundaries(
        segments, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, onset_times, beat_duration, sr, hop,
    )
    segments = _polish_segment_labels(
        segments, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, key_root, is_major,
        prefer_sevenths=prefer_sevenths,
        prefer_power=prefer_power,
        ml_root_bias=ml_root_bias,
    )
    if flux is not None:
        segments = _bar_aware_merge_short_segments(
            segments, beat_duration, flux, sr, hop,
            chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        )
        segments = _cross_validate_adjacent_segments(
            segments, chroma, chroma_low, chroma_mid, frame_keys, beat_times, sr, hop,
        )
        segments = _merge_flicker_segments(
            segments, beat_duration, flux, sr, hop,
            chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        )
    else:
        segments = _merge_adjacent_same_root(segments)
    segments = _apply_intro_guard(
        segments, beat_duration, chroma, chroma_low, chroma_mid,
        frame_keys, beat_times, sr, hop,
        key_root=key_root, is_major=is_major,
    )
    return _finalize_timeline(segments, beat_duration)


def _merge_flicker_segments(
    segments,
    beat_duration,
    flux,
    sr,
    hop,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
):
    """Extra anti-flicker pass (~1.5 beats) after label polish."""
    if len(segments) < 2:
        return segments

    segments = _merge_adjacent_same_root(segments)
    min_dur = min(max(float(beat_duration) * CHORD_POLISH_DEFAULTS["flicker_min_beats"], 0.5), 1.35)
    return _bar_aware_merge_short_segments(
        segments, beat_duration, flux, sr, hop,
        chroma, chroma_low, chroma_mid, frame_keys, beat_times,
        flux_keep_threshold=CHORD_POLISH_DEFAULTS["flux_keep_threshold_flicker"],
        match_keep_threshold=CHORD_POLISH_DEFAULTS["match_keep_threshold_flicker"],
        min_duration_override=min_dur,
    )


def _apply_intro_guard(
    segments,
    beat_duration,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    sr,
    hop=512,
    key_root=0,
    is_major=True,
    stable_beats=None,
    max_intro_sec=None,
):
    """
    Collapse unstable early segment flips into one intro chord.

    Waits until a segment spans stable_beats before allowing changes.
    """
    if stable_beats is None:
        stable_beats = CHORD_POLISH_DEFAULTS["intro_stable_beats"]
    if max_intro_sec is None:
        max_intro_sec = CHORD_POLISH_DEFAULTS["intro_max_sec"]
    if len(segments) < 2:
        return segments

    intro_cap = float(segments[0]["time"]) + max_intro_sec
    flicker_end = 0
    for i in range(len(segments) - 1):
        seg_end = float(segments[i + 1]["time"])
        if seg_end > intro_cap:
            break
        duration = seg_end - float(segments[i]["time"])
        if duration >= float(beat_duration) * stable_beats:
            break
        flicker_end = i + 1

    if flicker_end == 0:
        return segments

    merge_start = float(segments[0]["time"])
    if flicker_end + 1 < len(segments):
        merge_end = float(segments[flicker_end + 1]["time"])
    else:
        merge_end = float(
            segments[flicker_end].get("end_time", segments[flicker_end]["time"] + beat_duration),
        )
    if merge_end - merge_start < float(beat_duration) * 0.75:
        return segments

    if frame_keys is not None and beat_times is not None and len(beat_times) > 0:
        seg_key_root, seg_major = _key_at_time(
            (merge_start + merge_end) / 2.0, beat_times, frame_keys,
        )
    else:
        seg_key_root, seg_major = key_root, is_major

    best_name, confidence = _pick_chord_for_region(
        chroma, chroma_low, chroma_mid,
        merge_start, merge_end, sr, hop,
        seg_key_root, seg_major,
        hint_root=_chord_root_name(segments[flicker_end].get("chord", "N")),
    )

    merged = {
        "time": merge_start,
        "end_time": merge_end,
        "chord": _format_chord_name(best_name, seg_key_root, seg_major),
        "confidence": confidence,
        "is_low_confidence": confidence < 0.35,
        "is_power": str(best_name).endswith("5"),
        "strumming": segments[0].get("strumming", "D DU UDU"),
    }
    tail = [dict(s) for s in segments[flicker_end + 1:]]
    return [merged] + tail


def _cross_validate_adjacent_segments(
    segments,
    chroma,
    chroma_low,
    chroma_mid,
    frame_keys,
    beat_times,
    sr,
    hop=512,
    min_gain_ratio=None,
):
    """Swap neighbor chord labels when each fits the other's region better."""
    if len(segments) < 2:
        return segments

    if min_gain_ratio is None:
        min_gain_ratio = CHORD_POLISH_DEFAULTS["cross_validate_min_gain"]

    out = [dict(s) for s in segments]
    for i in range(len(out) - 1):
        a = out[i]
        b = out[i + 1]
        if a.get("chord") == b.get("chord"):
            continue

        a_end = float(b["time"])
        b_end = float(b.get("end_time", a_end + 1.0))
        key_a = _key_at_time((float(a["time"]) + a_end) / 2.0, beat_times, frame_keys)
        key_b = _key_at_time((a_end + b_end) / 2.0, beat_times, frame_keys)

        ca, ba, ma = _segment_center_chroma(chroma, chroma_low, chroma_mid, float(a["time"]), a_end, sr, hop)
        cb, cbass, cm = _segment_center_chroma(chroma, chroma_low, chroma_mid, a_end, b_end, sr, hop)

        score_aa = _match_score_for_chord(ca, ba, ma, a["chord"], key_a[0], key_a[1], bass_weight=0.10)
        score_bb = _match_score_for_chord(cb, cbass, cm, b["chord"], key_b[0], key_b[1], bass_weight=0.10)
        score_ab = _match_score_for_chord(ca, ba, ma, b["chord"], key_a[0], key_a[1], bass_weight=0.10)
        score_ba = _match_score_for_chord(cb, cbass, cm, a["chord"], key_b[0], key_b[1], bass_weight=0.10)

        current = score_aa + score_bb
        swapped = score_ab + score_ba
        if current <= 0:
            continue
        if swapped > current * (1.0 + min_gain_ratio):
            a["chord"], b["chord"] = b["chord"], a["chord"]
            a["confidence"], b["confidence"] = b.get("confidence", 0), a.get("confidence", 0)

    return out


def _segments_from_pattern(beat_times, pattern, phase, prefer_power, key_root, is_major, onset_times=None):
    """Build timeline segments locked to the 2-bar sheet pattern."""
    if len(beat_times) == 0:
        return []

    beat_duration = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    cycle = len(pattern)

    beat_labels = [
        _chord_display_name(pattern[(i + phase) % cycle], prefer_power, key_root, is_major)
        for i in range(len(beat_times))
    ]

    segments = []
    current = beat_labels[0]
    start_idx = 0

    for i in range(1, len(beat_labels)):
        if beat_labels[i] != current:
            n_beats = i - start_idx
            segments.append({
                "time": float(beat_times[start_idx]),
                "chord": current,
                "confidence": 0.92,
                "is_low_confidence": False,
                "is_power": prefer_power,
                "strumming": _strum_for_duration(n_beats, prefer_power),
            })
            current = beat_labels[i]
            start_idx = i

    n_beats = len(beat_labels) - start_idx
    segments.append({
        "time": float(beat_times[start_idx]),
        "chord": current,
        "confidence": 0.92,
        "is_low_confidence": False,
        "is_power": prefer_power,
        "strumming": _strum_for_duration(n_beats, prefer_power),
    })

    for i in range(len(segments)):
        if i + 1 < len(segments):
            segments[i]["end_time"] = float(segments[i + 1]["time"])
        else:
            segments[i]["end_time"] = float(beat_times[-1] + beat_duration)

    if onset_times is not None and len(onset_times) > 0:
        segments = _snap_segments_to_onsets(segments, onset_times, beat_duration)

    return segments


def _try_pattern_alignment(beat_times, chroma_segments, bass_segments, key_root, is_major, prefer_power, onset_times):
    """
    Align to 2-bar I–I–V–V–IV–IV–IV–IV when audio matches (e.g. Baba O'Riley).
    Returns segments if fit is good, else None.
    """
    roots = _diatonic_major_roots(key_root)
    if len(set(roots)) < 3:
        return None

    pattern = _two_bar_iiiv_pattern(key_root)
    beat_scores = _beat_root_scores(chroma_segments, bass_segments, roots, prefer_power)

    phase, _ = _find_pattern_phase(beat_scores, pattern)
    agreement = _pattern_agreement(beat_scores, pattern, phase)
    iv_ratio = _iv_ratio_in_votes(beat_scores, roots)

    if iv_ratio < 0.45 and agreement < 0.38:
        return None

    # Pattern lock only for power-chord rock loops (e.g. Baba O'Riley).
    # Full triad songs must use HMM decoding — loose gates were forcing I–V–IV on everything.
    if not prefer_power:
        return None
    if agreement < 0.32:
        return None

    return _segments_from_pattern(
        beat_times, pattern, phase, prefer_power, key_root, is_major, onset_times=onset_times,
    )


def _diatonic_triads(key_root, major=True):
    if major:
        chords = []
        for interval, quality in zip(DIATONIC_MAJOR_INTERVALS, DIATONIC_MAJOR_QUALITIES):
            root = ROOT_NAMES[(key_root + interval) % 12]
            chords.append(f"{root}{quality}")
        return chords
    chords = []
    for interval, quality in zip(DIATONIC_MAJOR_INTERVALS, DIATONIC_MAJOR_QUALITIES):
        root = ROOT_NAMES[(key_root + interval - 3) % 12]
        chords.append(f"{root}{quality}")
    return chords


def _chord_root_name(chord_name):
    if not chord_name or chord_name == 'N':
        return chord_name
    quality = _chord_quality(chord_name)
    return chord_name[: -len(quality)] if quality else chord_name


def _root_to_pitch_class(root):
    """Map a chord root spelling to pitch-class index 0–11."""
    if not root:
        return None
    root = ENHARMONIC.get(root, root)
    root = SHARP_TO_INTERNAL.get(root, root)
    try:
        return ROOT_NAMES.index(root)
    except ValueError:
        return None


DIATONIC_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _format_chord_name(chord_name, key_root, major=True):
    if not chord_name or chord_name == 'N':
        return chord_name

    quality = _chord_quality(chord_name)
    root = chord_name[: -len(quality)] if quality else chord_name
    root = _to_sharp_root(root)

    return f"{root}{quality}"


def _third_strength(normalized_chroma, root_idx):
    root = float(normalized_chroma[root_idx])
    fifth = float(normalized_chroma[(root_idx + 7) % 12])
    major_third = float(normalized_chroma[(root_idx + 4) % 12])
    minor_third = float(normalized_chroma[(root_idx + 3) % 12])
    return (major_third + minor_third) / (root + fifth + 1e-6)


def _score_chord_frame(chroma_vector, allowed_roots, prefer_power=False, bass_chroma=None):
    """Score a frame against allowed roots, optionally as power chords (root + 5th)."""
    chroma = chroma_vector.astype(np.float64)
    chroma = np.clip(chroma, 0, None)
    norm = np.linalg.norm(chroma)
    if norm < 1e-6:
        return None, 0.0, False

    normalized = chroma_vector / norm
    bass_hint = None
    if bass_chroma is not None:
        bass_norm = np.linalg.norm(bass_chroma)
        if bass_norm > 1e-6:
            bass_n = bass_chroma / bass_norm
            bass_scores = {r: float(bass_n[ROOT_NAMES.index(r)]) for r in allowed_roots}
            bass_hint = max(bass_scores, key=bass_scores.get)

    best_name = None
    best_score = -1.0
    best_is_power = False

    for root in allowed_roots:
        root_idx = ROOT_NAMES.index(root)

        triad_vec = TRIAD_MATRIX[:, TRIAD_NAMES.index(root)]
        triad_score = float(normalized.dot(triad_vec))

        power_name = f"{root}5"
        power_vec = TRIAD_MATRIX[:, TRIAD_NAMES.index(power_name)]
        power_score = float(normalized.dot(power_vec))

        if bass_hint == root:
            triad_score += 0.15
            power_score += 0.15

        third = _third_strength(normalized, root_idx)
        use_power = prefer_power or third < 0.42 or power_score >= triad_score * 0.88

        if use_power and power_score >= triad_score * 0.85:
            if power_score > best_score:
                best_score = power_score
                best_name = power_name
                best_is_power = True
        else:
            if triad_score > best_score:
                best_score = triad_score
                best_name = root
                best_is_power = False

    if best_name is None:
        return None, 0.0, False

    if prefer_power and not best_name.endswith('5'):
        best_name = f"{_chord_root_name(best_name)}5"
        best_is_power = True

    return best_name, best_score, best_is_power


def _estimate_power_song_ratio(chroma_bars, allowed_roots=None):
    power_votes = 0
    total = chroma_bars.shape[1]
    roots = allowed_roots if allowed_roots is not None else ROOT_NAMES
    for i in range(total):
        _, _, is_power = _score_chord_frame(chroma_bars[:, i], roots, prefer_power=False)
        if is_power:
            power_votes += 1
    return power_votes / max(total, 1)


def _smooth_chord_labels(labels, window=5):
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        window_labels = labels[start:end]
        counts = {}
        for label in window_labels:
            counts[label] = counts.get(label, 0) + 1
        smoothed.append(max(counts, key=counts.get))
    return smoothed


def _segments_from_labels(labels, times, confidences):
    if not labels:
        return []

    segments = []
    current = labels[0]
    start_idx = 0
    conf_acc = [confidences[0]]

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append({
                "time": float(times[start_idx]),
                "chord": current,
                "confidence": float(np.mean(conf_acc)),
                "is_low_confidence": bool(np.mean(conf_acc) < 0.35),
            })
            current = labels[i]
            start_idx = i
            conf_acc = [confidences[i]]
        else:
            conf_acc.append(confidences[i])

    segments.append({
        "time": float(times[start_idx]),
        "chord": current,
        "confidence": float(np.mean(conf_acc)),
        "is_low_confidence": bool(np.mean(conf_acc) < 0.35),
    })
    return segments


def _prune_short_segments(segments, min_duration=1.0):
    if len(segments) <= 1:
        return segments

    pruned = [segments[0]]
    for seg in segments[1:]:
        prev = pruned[-1]
        duration = seg["time"] - prev["time"]
        if duration < min_duration and len(pruned) > 1:
            prev_duration = prev["time"] - pruned[-2]["time"]
            merge_target = pruned[-2] if prev_duration >= duration else prev
            merge_target["confidence"] = max(merge_target["confidence"], seg["confidence"])
            if merge_target is prev:
                pruned[-1] = seg
        else:
            pruned.append(seg)
    return pruned


def _refine_segments_with_vocabulary(
    segments,
    chroma,
    chroma_low,
    key_root,
    major=True,
    prefer_sevenths=False,
    prefer_power=False,
    chroma_mid=None,
    frame_keys=None,
    beat_times=None,
    ml_root_bias=1.08,
    quality_bonus=None,
    label_inertia=None,
):
    """Re-score each segment with per-frame voting over the chord vocabulary."""
    if label_inertia is None:
        label_inertia = CHORD_POLISH_DEFAULTS["label_inertia"]
    if len(segments) < 1:
        return segments

    hop = 512
    sr = 22050
    if quality_bonus is None:
        quality_bonus = _quality_bonus_for_song(prefer_sevenths, prefer_power)

    refined = []
    for i, seg in enumerate(segments):
        end = segments[i + 1]["time"] if i + 1 < len(segments) else seg.get("end_time", seg["time"] + 4.0)

        if frame_keys is not None and beat_times is not None:
            seg_key_root, seg_major = _key_at_time(
                (float(seg["time"]) + float(end)) / 2.0,
                beat_times,
                frame_keys,
            )
        else:
            seg_key_root, seg_major = key_root, major

        oc, ob, om = _segment_center_chroma(
            chroma, chroma_low, chroma_mid, float(seg["time"]), float(end), sr, hop,
        )
        score_kr, score_maj = _segment_local_key(oc, ob, om)

        hint = _chord_root_name(seg.get("chord", "N"))
        orig_chord = seg.get("chord", "N")
        best_name, confidence = _pick_chord_for_region(
            chroma, chroma_low, chroma_mid,
            float(seg["time"]), float(end), sr, hop,
            score_kr, score_maj,
            quality_bonus=quality_bonus,
            frame_keys=frame_keys,
            beat_times=beat_times,
            hint_root=hint,
            ml_root_bias=ml_root_bias,
        )

        if orig_chord and orig_chord not in ("N", "") and _format_chord_name(best_name, seg_key_root, seg_major) != _format_chord_name(orig_chord, seg_key_root, seg_major):
            oc, ob, om = _segment_center_chroma(
                chroma, chroma_low, chroma_mid, float(seg["time"]), float(end), sr, hop,
            )
            orig_score = _match_score_for_chord(
                oc, ob, om, orig_chord, seg_key_root, seg_major,
            )
            if orig_score >= confidence * (1.0 - label_inertia):
                best_name = orig_chord if orig_chord in CHORD_VOCAB else best_name
                confidence = orig_score

        seg = dict(seg)
        seg["chord"] = _format_chord_name(best_name, seg_key_root, seg_major)
        seg["confidence"] = confidence
        seg["is_power"] = best_name.endswith('5')
        seg["is_low_confidence"] = confidence < 0.35
        refined.append(seg)

    return refined


def extract_chords(y, sr):
    hop_length = 512
    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(y, sr, hop_length)

    _, beat_frames = librosa.beat.beat_track(
        y=y_harmonic, sr=sr, hop_length=hop_length, units='frames',
    )
    beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1] - 1)
    if len(beat_frames) < 4:
        beat_frames = np.arange(0, chroma.shape[1], max(1, chroma.shape[1] // 64))

    segment_frames = beat_frames
    if len(segment_frames) < 2:
        segment_frames = np.arange(0, chroma.shape[1], max(1, chroma.shape[1] // 48))

    chroma_segments = librosa.util.sync(chroma, segment_frames, aggregate=np.median)
    bass_segments = librosa.util.sync(chroma_low, segment_frames, aggregate=np.median)
    mid_segments = librosa.util.sync(chroma_mid, segment_frames, aggregate=np.median)
    segment_times = librosa.frames_to_time(segment_frames, sr=sr, hop_length=hop_length)
    beat_duration = float(np.median(np.diff(segment_times))) if len(segment_times) > 1 else 0.5
    bar_duration = _estimate_bar_duration(beat_duration)
    downbeats = _downbeat_indices(segment_frames, y_harmonic, sr, hop_length)

    log_em, raw_em = _frame_emission_matrix(
        chroma_segments, bass_segments, 0, True, mid_segments,
    )
    key_root, is_major = _resolve_key(np.mean(chroma, axis=1), raw_em, bass_segments, chroma_segments)
    key_info = _build_key_info(key_root, is_major)

    power_ratio = _estimate_power_song_ratio(chroma_segments)
    prefer_power = power_ratio > 0.42

    onset_frames = librosa.onset.onset_detect(
        y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    pattern_segments = _try_pattern_alignment(
        segment_times, chroma_segments, bass_segments,
        key_root, is_major, prefer_power, onset_times,
    )
    if pattern_segments is not None:
        return pattern_segments, key_info

    frame_keys = _windowed_key_for_frames(
        chroma_segments, bass_segments, np.mean(chroma, axis=1), mid_segments, step=12,
    )
    log_em, raw_em = _frame_emission_matrix_windowed(
        chroma_segments, bass_segments, mid_segments, frame_keys,
    )
    flux = _chroma_beat_flux(chroma_segments)
    path = _viterbi_decode_chords(log_em, frame_keys, flux, min_run=2)

    labels = []
    confidences = []
    for i, idx in enumerate(path):
        kr, maj = frame_keys[i]
        name = DECODE_VOCAB[idx]
        labels.append(_format_chord_name(name, kr, maj))
        confidences.append(float(raw_em[i, idx]))

    segments = _segments_from_labels(labels, segment_times, confidences)
    segments = _prune_short_segments(segments, min_duration=_bar_aware_min_duration(beat_duration) * 0.75)
    segments = _polish_chord_timeline(
        segments, chroma, chroma_low, chroma_mid,
        frame_keys, segment_times, onset_times, beat_duration, sr, hop_length,
        key_root=key_root, is_major=is_major,
        prefer_power=prefer_power,
    )

    chroma_mean = np.mean(chroma, axis=1)
    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    key_info = _build_key_info(
        key_root,
        is_major,
        mode=mode if mode not in ("major", "minor") else None,
    )

    track_duration = float(len(y) / sr)
    for i in range(len(segments)):
        start_time = segments[i]["time"]
        if i + 1 < len(segments):
            end_time = segments[i + 1]["time"]
        else:
            end_time = max(track_duration, start_time + beat_duration * 2)
        segments[i]["end_time"] = float(end_time)

        onsets_in_chord = [t for t in onset_times if start_time <= t < end_time]
        duration = end_time - start_time
        density = len(onsets_in_chord) / duration if duration > 0 else 0

        if density < 0.5:
            strum = "D"
        elif density < 1.0:
            strum = "D  D"
        elif density < 1.5:
            strum = "D DU"
        elif density < 2.5:
            strum = "D DU UDU"
        else:
            strum = "D DU UDUD"

        segments[i]["strumming"] = strum

    return segments, key_info


def midi_to_string_fret(midi_note, prev_string=None, prev_fret=None):
    tuning = [64, 59, 55, 50, 45, 40]
    candidates = []
    for i, string_base in enumerate(tuning):
        fret = midi_note - string_base
        if 0 <= fret <= 15:
            candidates.append((i + 1, fret))

    if not candidates:
        return 1, max(0, min(24, midi_note - 64))

    if prev_string is not None and prev_fret is not None:
        return min(candidates, key=lambda c: abs(c[0] - prev_string) * 4 + abs(c[1] - prev_fret))

    return min(candidates, key=lambda c: c[1])


def _map_notes_to_fretboard(note_events):
    raw_notes = []
    prev_string, prev_fret = None, None

    for note in note_events:
        start_time, end_time, pitch, velocity, _ = note
        if 64 <= pitch <= 100 and velocity >= 20:
            string, fret = midi_to_string_fret(pitch, prev_string, prev_fret)
            prev_string, prev_fret = string, fret
            raw_notes.append({
                "time": float(start_time),
                "end": float(end_time),
                "pitch": int(pitch),
                "note": librosa.midi_to_note(pitch),
                "string": int(string),
                "fret": int(fret),
            })

    raw_notes.sort(key=lambda x: x["time"])
    return raw_notes


def _cluster_solo_sections(raw_notes):
    solo_sections = []
    if not raw_notes:
        return solo_sections

    current_section_start = raw_notes[0]["time"]
    current_section_notes = [raw_notes[0]]

    def flush_section(notes, start, end):
        duration = end - start
        density = len(notes) / (duration if duration > 0 else 1)
        if duration >= 3.5 and density >= 1.8:
            solo_sections.append({
                "start": float(start),
                "end": float(end),
                "type": "solo",
                "confidence": "high" if density > 3.5 else "borderline",
                "notes": notes,
            })

    for i in range(1, len(raw_notes)):
        note = raw_notes[i]
        prev_note = raw_notes[i - 1]

        if note["time"] - prev_note["end"] > 1.5:
            flush_section(current_section_notes, current_section_start, prev_note["end"])
            current_section_start = note["time"]
            current_section_notes = [note]
        else:
            current_section_notes.append(note)

    last_note = current_section_notes[-1]
    flush_section(current_section_notes, current_section_start, last_note["end"])
    return solo_sections


def analyze_audio(audio_path, video_id, progress_callback=None, force_reanalyze=False, song_metadata=None):
    cache_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    if force_reanalyze and os.path.exists(cache_file):
        os.remove(cache_file)

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
        if is_cache_valid(cached):
            if song_metadata and not cached.get("song"):
                cached["song"] = song_metadata
            if progress_callback:
                progress_callback("Loaded from cache")
            return cached

    y, sr = librosa.load(audio_path, sr=22050)

    if progress_callback:
        progress_callback("Analyzing chords...")
    from chord_engine import extract_chords as run_chord_extraction
    chords, key_info = run_chord_extraction(y, sr)

    if progress_callback:
        progress_callback("Fetching lyrics...")

    from downloader import extract_youtube_id
    from lyrics import attach_lyrics_to_timeline

    youtube_id = None
    if song_metadata and song_metadata.get("youtube_id"):
        youtube_id = song_metadata["youtube_id"]

    chords, lyrics_meta = attach_lyrics_to_timeline(
        chords,
        song_metadata,
        len(y) / sr,
        youtube_id=youtube_id,
    )

    if progress_callback:
        progress_callback("Detecting solos...")

    _, _, note_events = predict(audio_path)
    raw_notes = _map_notes_to_fretboard(note_events)
    solo_sections = _cluster_solo_sections(raw_notes)

    meta = cache_metadata()
    result = {
        "video_id": video_id,
        "analyzer_version": meta["analyzer_version"],
        "chord_engine": meta["chord_engine"],
        "timeline": chords,
        "solos": solo_sections,
        "key": key_info,
        "song": song_metadata or {},
        "lyrics": lyrics_meta,
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result
