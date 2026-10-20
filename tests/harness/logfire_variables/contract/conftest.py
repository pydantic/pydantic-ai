"""Fixtures for the Agent Control contract suite: the cross-language vectors in `spec/`."""

from __future__ import annotations

import json
import pathlib
from typing import Any, cast

import pytest

SPEC = pathlib.Path(__file__).parent / 'spec'
"""The cross-language vectors both cores are tested against; see `spec/README.md` beside them."""


def _vectors(name: str) -> list[dict[str, Any]]:
    return cast('list[dict[str, Any]]', json.loads((SPEC / name).read_text()))


def expand_repeats(value: Any) -> Any:
    """Expand the vector files' `{repeat, times}` shorthand, which keeps the 64 KiB cases readable."""
    if isinstance(value, dict):
        typed = cast(dict[str, Any], value)
        if set(typed) == {'repeat', 'times'}:
            return cast(str, typed['repeat']) * cast(int, typed['times'])
        return {key: expand_repeats(item) for key, item in typed.items()}
    if isinstance(value, list):
        return [expand_repeats(item) for item in cast(list[Any], value)]
    return value


@pytest.fixture(scope='session')
def agent_name_vectors() -> list[dict[str, Any]]:
    return _vectors('agent-name.json')


@pytest.fixture(scope='session')
def baseline_vectors() -> list[dict[str, Any]]:
    return _vectors('baseline.json')


@pytest.fixture(scope='session')
def config_parsing_vectors() -> list[dict[str, Any]]:
    return _vectors('config-parsing.json')


@pytest.fixture(scope='session')
def instructions_apply_vectors() -> list[dict[str, Any]]:
    return _vectors('instructions-apply.json')


@pytest.fixture(scope='session')
def tools_apply_vectors() -> list[dict[str, Any]]:
    return _vectors('tools-apply.json')


@pytest.fixture(scope='session')
def settings_apply_vectors() -> list[dict[str, Any]]:
    return _vectors('settings-apply.json')
