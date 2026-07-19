"""
madmom 0.16.1 (2018, still the latest PyPI release) imports
`collections.MutableSequence` directly. Python moved that alias to
`collections.abc` in 3.3 and removed the `collections`-level alias in 3.10,
so `import madmom` raises `ImportError: cannot import name 'MutableSequence'
from 'collections'` on Python 3.10+. Upstream fixed this on GitHub main
(https://github.com/CPJKU/madmom) but has never cut a new PyPI release.

Call ensure_collections_compat() before any `import madmom` / `from madmom
import ...`.
"""
from __future__ import annotations

import collections
import collections.abc


def ensure_collections_compat() -> None:
    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence
