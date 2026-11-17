"""Tests for the deprecated `pydantic_ai_harness.experimental.acp` names."""

from __future__ import annotations

import re

import pytest

import pydantic_ai_harness.experimental.acp as package
from pydantic_ai_harness import HarnessDeprecationWarning


@pytest.mark.parametrize(
    ('old', 'new'),
    [
        ('AcpFileSystemToolset', 'ACPFileSystemToolset'),
        ('AcpSession', 'ACPSession'),
        ('AcpSessionConfig', 'ACPSessionConfig'),
        ('AcpTerminalToolset', 'ACPTerminalToolset'),
    ],
)
def test_renamed_class_alias_warns_and_resolves(old: str, new: str) -> None:
    message = f'`pydantic_ai_harness.experimental.acp.{old}` has been renamed to `pydantic_ai_harness.experimental.acp.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(package, old)

    assert alias is getattr(package, new)
    assert new in package.__all__
    assert old not in package.__all__


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(package, 'DoesNotExist')
