"""Key-label matching helpers for eval (no ML imports)."""

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#"}


def normalize_root(root):
    return FLAT_TO_SHARP.get(root, root)


def root_to_pc(root):
    root = normalize_root(root)
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    try:
        return PITCH_CLASSES.index(root)
    except ValueError:
        return None


def parse_key_label(label):
    if not label:
        return None, None
    label = label.strip()
    if " " not in label:
        return normalize_root(label), None
    root, mode = label.rsplit(" ", 1)
    return normalize_root(root.strip()), mode.strip().lower()


def roots_match(expected_pc, predicted_pc, transpose_invariant=False):
    if expected_pc is None or predicted_pc is None:
        return False
    if expected_pc == predicted_pc:
        return True
    if transpose_invariant:
        for shift in (-1, 1):
            if (expected_pc + shift) % 12 == predicted_pc:
                return True
    return False


def key_labels_match(
    expected_label,
    predicted_label,
    transpose_invariant=False,
    allow_relative=True,
):
    expected_root, expected_mode = parse_key_label(expected_label)
    predicted_root, predicted_mode = parse_key_label(predicted_label)
    if expected_root is None or predicted_root is None:
        return False
    if expected_mode is None or predicted_mode is None:
        return False

    expected_pc = root_to_pc(expected_root)
    predicted_pc = root_to_pc(predicted_root)

    if expected_mode == predicted_mode:
        if roots_match(expected_pc, predicted_pc, transpose_invariant):
            return True

    if allow_relative:
        if expected_mode == "major" and predicted_mode == "minor":
            relative_minor_pc = (expected_pc + 9) % 12 if expected_pc is not None else None
            if roots_match(relative_minor_pc, predicted_pc, transpose_invariant):
                return True
        if expected_mode == "minor" and predicted_mode == "major":
            relative_major_pc = (predicted_pc + 9) % 12 if predicted_pc is not None else None
            if roots_match(expected_pc, relative_major_pc, transpose_invariant):
                return True

    minor_family = {"minor", "dorian"}
    major_family = {"major", "mixolydian"}
    if expected_mode in minor_family and predicted_mode in minor_family:
        if roots_match(expected_pc, predicted_pc, transpose_invariant):
            return True
    if expected_mode in major_family and predicted_mode in major_family:
        if roots_match(expected_pc, predicted_pc, transpose_invariant):
            return True

    return False
