"""Tests for the collections.MutableSequence shim madmom 0.16.1 needs on Python 3.10+."""
import collections

from madmom_compat import ensure_collections_compat


def test_patches_missing_mutablesequence(monkeypatch):
    monkeypatch.delattr(collections, "MutableSequence", raising=False)
    ensure_collections_compat()
    assert collections.MutableSequence is collections.abc.MutableSequence


def test_noop_when_already_present(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(collections, "MutableSequence", sentinel, raising=False)
    ensure_collections_compat()
    assert collections.MutableSequence is sentinel
