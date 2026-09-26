"""Test the shared hosted MCP helpers: credential lookup and read-only selection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness._mcp import credential, is_read_only, one_connection


def test_explicit_credential_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', 'deployment-token')
    assert credential('user-token', env='WHOAMI_TOKEN', service='whoami') == 'user-token'


@pytest.mark.parametrize('auth', [None, ''])
def test_environment_credential(auth: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', 'deployment-token')
    assert credential(auth, env='WHOAMI_TOKEN', service='whoami') == 'deployment-token'


@pytest.mark.parametrize('env', ['WHOAMI_TOKEN', None])
@pytest.mark.parametrize('auth', [None, ''])
def test_missing_credential_raises(auth: str | None, env: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', '')
    with pytest.raises(UserError, match='to connect to whoami'):
        credential(auth, env=env, service='whoami')


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'annotations': {'readOnlyHint': True}}, True),
        ({'annotations': {'readOnlyHint': False}}, False),
        ({'annotations': {}}, False),
        (None, False),
    ],
)
def test_is_read_only_requires_an_explicit_hint(metadata: dict[str, object] | None, expected: bool) -> None:
    assert is_read_only(ToolDefinition(name='tool', metadata=metadata)) is expected


@dataclass
class Connection(AbstractCapability[None]):
    token: str = ''


def test_same_configuration_is_one_connection() -> None:
    first = Connection(token='a', id='whoami')
    assert one_connection([first, Connection(token='a', id='whoami')]) is first


def test_different_configurations_raise() -> None:
    with pytest.raises(
        UserError, match="Capability id 'whoami' is used by multiple Connection capabilities that disagree on 'token'"
    ):
        one_connection([Connection(token='a', id='whoami'), Connection(token='b', id='whoami')])
