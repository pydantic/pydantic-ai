"""Test the shared hosted MCP helpers: credential lookup, per-run connection, and read-only selection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness._mcp import connect_with_auth, credential, is_read_only, one_connection


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


def _connect(auth: str | None) -> AbstractToolset[str | None]:
    """Stand in for an integration's `_connect`: a toolset that records the token it was given."""
    return FunctionToolset[str | None](id=f'connected-as-{auth}')


async def _resolved_for(toolset: AbstractToolset[str | None], deps: str | None) -> str | None:
    """The id of the toolset a run with `deps` connects, or `None` when it has no tools."""
    assert isinstance(toolset, DynamicToolset)
    run_toolset = await toolset.for_run(RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage()))
    assert isinstance(run_toolset, DynamicToolset)
    inner = run_toolset._toolset  # pyright: ignore[reportPrivateUsage]
    return None if inner is None else inner.id


def test_fixed_auth_connects_once() -> None:
    assert connect_with_auth('user-token', _connect, id='whoami').id == 'connected-as-user-token'


async def test_sync_and_async_auth_functions_connect_each_run() -> None:
    def sync_token(ctx: RunContext[str | None]) -> str | None:
        return ctx.deps

    async def async_token(ctx: RunContext[str | None]) -> str | None:
        return ctx.deps

    for auth in (sync_token, async_token):
        toolset = connect_with_auth(auth, _connect, id='whoami')
        assert toolset.id == 'whoami'
        assert await _resolved_for(toolset, 'alice') == 'connected-as-alice'
        assert await _resolved_for(toolset, 'bob') == 'connected-as-bob'


@pytest.mark.parametrize('missing', [None, ''])
async def test_auth_function_returning_nothing_means_no_tools(missing: str | None) -> None:
    toolset = connect_with_auth(lambda ctx: ctx.deps, _connect, id='whoami')
    assert await _resolved_for(toolset, missing) is None


async def test_auth_function_returning_oauth_raises() -> None:
    toolset = connect_with_auth(lambda ctx: ctx.deps, _connect, id='whoami')
    with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
        await _resolved_for(toolset, 'oauth')


async def test_reject_oauth_false_leaves_oauth_to_connect() -> None:
    toolset = connect_with_auth(lambda ctx: ctx.deps, _connect, id='whoami', reject_oauth=False)
    assert await _resolved_for(toolset, 'oauth') == 'connected-as-oauth'


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
