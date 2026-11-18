"""Fixtures for the Agent Control contract suite: the cross-language vectors in `spec/`."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from pydantic import TypeAdapter

SPEC = pathlib.Path(__file__).parent / 'spec'
"""The cross-language vectors both cores are tested against; see `spec/README.md` beside them."""


_OBJECT = TypeAdapter(dict[str, Any])
_ARRAY = TypeAdapter(list[Any])


def _vectors(name: str) -> list[dict[str, Any]]:
    return TypeAdapter(list[dict[str, Any]]).validate_json((SPEC / name).read_text())


def expand_repeats(value: Any) -> Any:
    """Expand the vector files' `{repeat, times}` shorthand, which keeps the 64 KiB cases readable."""
    match value:
        case {'repeat': str(text), 'times': int(times)} if len(value) == 2:
            return text * times
        case dict():
            return {key: expand_repeats(item) for key, item in _OBJECT.validate_python(value).items()}
        case list():
            return [expand_repeats(item) for item in _ARRAY.validate_python(value)]
        case _:
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
