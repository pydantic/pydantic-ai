"""Tests for the deprecated `pydantic_ai_harness.peers` names."""

from __future__ import annotations

import re

import pytest

import pydantic_ai_harness.peers as package
from pydantic_ai_harness import HarnessDeprecationWarning


def test_renamed_class_alias_warns_and_resolves() -> None:
    message = '`pydantic_ai_harness.peers.PeerNetwork` has been renamed to `pydantic_ai_harness.peers.PeerConnection`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(package, 'PeerNetwork')

    assert alias is package.PeerConnection
    assert 'PeerConnection' in package.__all__
    assert 'PeerNetwork' not in package.__all__


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(package, 'DoesNotExist')
