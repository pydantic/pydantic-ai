"""Test the shared hosted MCP helpers: credential lookup, per-run connections, and read-only selection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._mcp import credential, is_read_only, per_run_connection


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


def run_context(deps: str | None) -> RunContext[str | None]:
    return RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())


def token_from_deps(ctx: RunContext[str | None]) -> str | None:
    return ctx.deps


async def token_from_deps_async(ctx: RunContext[str | None]) -> str | None:
    return ctx.deps


def must_not_connect(token: str) -> FunctionToolset[str | None]:
    raise AssertionError('`oauth` must not connect')  # pragma: no cover


@pytest.mark.parametrize('auth', [token_from_deps, token_from_deps_async], ids=['sync', 'async'])
async def test_per_run_connection_connects_with_each_run_token(
    auth: Callable[[RunContext[str | None]], str | None | Awaitable[str | None]],
) -> None:
    tokens: list[str] = []

    def connect(token: str) -> FunctionToolset[str | None]:
        tokens.append(token)
        return FunctionToolset[str | None]()

    toolset = per_run_connection(auth, connect, id='whoami')
    for deps in ['alice', None, '', 'bob']:
        await toolset.for_run(run_context(deps))
    assert (toolset.id, tokens) == ('whoami', ['alice', 'bob'])


async def test_per_run_connection_rejects_oauth() -> None:
    toolset = per_run_connection(token_from_deps_async, must_not_connect, id=None)
    with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
        await toolset.for_run(run_context('oauth'))


async def test_per_run_connection_names_the_expected_token() -> None:
    toolset = per_run_connection(token_from_deps, must_not_connect, id=None, expects='an access token')
    with pytest.raises(UserError, match="must return an access token, not 'oauth'"):
        await toolset.for_run(run_context('oauth'))


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
    assert one_per_id([first, Connection(token='a', id='whoami')]) is first


def test_different_configurations_raise() -> None:
    with pytest.raises(
        UserError, match="Capability id 'whoami' is used by multiple Connection capabilities that disagree on 'token'"
    ):
        one_per_id([Connection(token='a', id='whoami'), Connection(token='b', id='whoami')])
