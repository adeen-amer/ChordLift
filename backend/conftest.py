"""Session-wide pytest setup."""
from madmom_compat import ensure_collections_compat

# Several tests do `pytest.importorskip("madmom")`, which is a bare `import
# madmom` that bypasses the compat call inside beat_tracking.py/infer.py.
# Patch it once here so any `import madmom` anywhere in the suite is safe.
ensure_collections_compat()
