"""Tests for the `StackOne` capability through the public `Agent(capabilities=[...])` surface."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

import pytest
from fastmcp.client.transports import StreamableHttpTransport

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.stackone import StackOne, StackOneToolset

if TYPE_CHECKING:
    from .._mcp_server import MCPServer

DepsT = TypeVar('DepsT')


INDIVIDUAL_PHRASE = '{connector}_{action}_{entity}'


def instructions(messages: list[ModelMessage]) -> str:
    return '\n'.join(message.instructions or '' for message in messages if isinstance(message, ModelRequest))


def tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def http_transport(toolset: AbstractToolset[DepsT]) -> StreamableHttpTransport:
    assert isinstance(toolset, StackOneToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def per_user_key(ctx: RunContext[str | None]) -> str | None:
    """Read the run's API key from its deps, as an app serving many users would."""
    return ctx.deps


async def per_user_key_async(ctx: RunContext[str | None]) -> str | None:
    return ctx.deps


async def connections_for(capability: StackOne[str | None], deps: str | None) -> list[StackOneToolset[str | None]]:
    """The StackOne connections a run with `deps` would open."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)
    connections: list[StackOneToolset[str | None]] = []

    def collect(leaf: AbstractToolset[str | None]) -> None:
        if isinstance(leaf, StackOneToolset):
            connections.append(leaf)

    toolset.apply(collect)
    return connections


class TestStackOne:
    def test_settings_reach_the_connection(self):
        capability = StackOne[None](account_id='45320', auth='key', base_url='https://api.eu1.stackone.com')
        transport = http_transport(capability.get_toolset())
        assert (transport.url, transport.headers['x-account-id']) == (
            'https://api.eu1.stackone.com/mcp?tool-mode=search_execute',
            '45320',
        )

    @pytest.mark.parametrize(
        ('capability', 'toolset_id'),
        [
            (StackOne[None](account_id='45320', auth='key'), 'stackone-45320'),
            (StackOne[None](account_id='45320', auth='key', id='hr'), 'hr'),
        ],
        ids=['derived', 'custom'],
    )
    def test_toolset_id(self, capability: StackOne[None], toolset_id: str):
        assert capability.get_toolset().id == toolset_id

    def test_two_accounts_stay_two_capabilities(self):
        """One account is one provider connection, so the account is what names the capability.

        A fixed `'stackone'` id would make these two merge into one, and one linked account would
        silently drop out of the agent -- the failure mode a shared id is supposed to prevent.
        """
        first = StackOne(account_id='45320', auth='key')
        second = StackOne(account_id='99811', auth='key')
        assert (first.id, second.id) == ('stackone-45320', 'stackone-99811')

    @pytest.mark.parametrize(
        'capability',
        [
            StackOne[None](account_id='45320', auth='secret'),
            StackOne[None](account_id='45320', auth='key', client='https://user:secret@example.com/mcp'),
        ],
        ids=['api-key', 'client'],
    )
    def test_secrets_are_hidden_from_repr(self, capability: StackOne[None]):
        assert 'secret' not in repr(capability)

    def test_agent_spec_schema_excludes_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([StackOne])
        assert '"client"' not in json.dumps(schema)

    def test_agent_spec_reads_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        capability = StackOne[None].from_spec(account_id='45320')
        assert http_transport(capability.get_toolset()).headers['Authorization'] == basic('env-key')

    def test_agent_spec_api_key_is_deprecated(self, monkeypatch: pytest.MonkeyPatch):
        # With no `STACKONE_API_KEY` to fall back on, construction only succeeds if the spec's key arrives.
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        spec = {'capabilities': [{'StackOne': {'account_id': '45320', 'api_key': 'key'}}]}
        with pytest.warns(HarnessDeprecationWarning, match='set the `STACKONE_API_KEY` environment variable'):
            Agent.from_spec(spec, custom_capability_types=[StackOne], model=TestModel())

    def test_agent_spec_schema_excludes_auth(self):
        schema = AgentSpec.model_json_schema_with_capabilities([StackOne])
        assert '"auth"' not in json.dumps(schema)

    @pytest.mark.parametrize(
        ('arguments', 'match'),
        [
            ({'tool_mode': 'search-execute'}, '`tool_mode` must be'),
            ({'tool_mode': 'search_execute', 'actions': '*_list_*'}, 'cannot apply in `search_execute` mode'),
        ],
    )
    def test_agent_spec_rejects_invalid_configuration(
        self, arguments: dict[str, object], match: str, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv('STACKONE_API_KEY', 'key')
        spec = {'capabilities': [{'StackOne': {'account_id': '45320', **arguments}}]}
        with pytest.raises(ValueError, match=match):
            Agent.from_spec(spec, custom_capability_types=[StackOne], model=TestModel())

    @pytest.mark.parametrize('actions', [['*_list_*'], '*_LIST_*'])
    async def test_agent_calls_only_matching_actions(self, stackone_server: MCPServer, actions: list[str] | str):
        capability = StackOne(account_id='45320', auth='key', client=stackone_server, actions=actions)
        result = await Agent(TestModel(), capabilities=[capability]).run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_list_employees'}

    async def test_metadata_overrides_server_metadata(self, stackone_server: MCPServer, run_context: RunContext[None]):
        # `task` collides with a server-provided key: user metadata must win, matching `.with_metadata()`.
        toolset = StackOne(
            account_id='45320', auth='key', client=stackone_server, metadata={'task': 'hr'}
        ).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert [(tool.tool_def.metadata or {}).get('task') for tool in tools.values()] == ['hr', 'hr']

    @pytest.mark.parametrize(
        ('capability', 'phrase'),
        [
            (StackOne[None](account_id='45320', auth='key'), 'must never be guessed'),
            (
                StackOne[None](account_id='45320', auth='key', tool_mode='individual'),
                '{connector}_{action}_{entity}',
            ),
            (StackOne[None](account_id='45320', auth='key', actions='*_list_*'), '{connector}_{action}_{entity}'),
        ],
        ids=['default', 'individual', 'actions'],
    )
    def test_instructions_follow_the_tool_mode(self, capability: StackOne[None], phrase: str):
        assert phrase in (capability.get_instructions() or '')

    def test_instructions_can_be_disabled(self):
        assert StackOne(account_id='45320', auth='key', include_instructions=False).get_instructions() is None


def basic(api_key: str) -> str:
    return 'Basic ' + base64.b64encode(f'{api_key}:'.encode()).decode()


class TestStackOneAuth:
    def test_api_key_is_a_deprecated_alias_of_auth(self):
        with pytest.warns(
            HarnessDeprecationWarning, match=r'`StackOne\(api_key=\.\.\.\)` has been renamed to `StackOne\(auth='
        ):
            capability = StackOne[None](account_id='45320', api_key='key')
        assert (capability.auth, capability.api_key) == ('key', None)
        assert http_transport(capability.get_toolset()).headers['Authorization'] == basic('key')

    def test_auth_and_api_key_together_raise(self):
        with pytest.raises(UserError, match='Pass `auth` only'):
            StackOne(account_id='45320', auth='key', api_key='other')

    def test_missing_key_error_names_auth(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='pass `auth` to `StackOne`'):
            StackOne[None](account_id='45320').get_toolset()

    def test_auth_function_is_hidden_from_repr(self):
        assert 'per_user_key' not in repr(StackOne[str | None](account_id='45320', auth=per_user_key))

    @pytest.mark.parametrize('auth', [per_user_key, per_user_key_async], ids=['sync', 'async'])
    async def test_each_run_connects_with_its_own_key(
        self, auth: Callable[[RunContext[str | None]], str | None | Awaitable[str | None]]
    ):
        capability = StackOne[str | None](account_id='45320', auth=auth)
        [alice] = await connections_for(capability, 'alice-key')
        [bob] = await connections_for(capability, 'bob-key')
        assert (http_transport(alice).headers, http_transport(bob).headers) == (
            {'Authorization': basic('alice-key'), 'x-account-id': '45320'},
            {'Authorization': basic('bob-key'), 'x-account-id': '45320'},
        )

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_key_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch):
        # The environment key is set to show a function never falls back to it.
        monkeypatch.setenv('STACKONE_API_KEY', 'deployment-key')
        capability = StackOne[str | None](account_id='45320', auth=per_user_key)
        assert await connections_for(capability, missing) == []

    async def test_auth_function_returning_oauth_raises(self):
        capability = StackOne[str | None](account_id='45320', auth=per_user_key)
        with pytest.raises(UserError, match="must return a StackOne API key, not 'oauth'"):
            await connections_for(capability, 'oauth')

    def test_per_run_toolset_keeps_the_derived_id(self):
        assert StackOne[str | None](account_id='45320', auth=per_user_key).get_toolset().id == 'stackone-45320'

    async def test_agent_runs_with_per_run_key(self, stackone_server: MCPServer):
        capability = StackOne[str | None](
            account_id='45320', auth=per_user_key, client=stackone_server, actions=['*_list_*']
        )
        agent = Agent(TestModel(), deps_type=str | None, capabilities=[capability])
        with_key = (await agent.run('list employees', deps='alice-key')).all_messages()
        without_key = (await agent.run('list employees', deps=None)).all_messages()
        assert (tool_call_names(with_key), tool_call_names(without_key)) == ({'bamboohr_list_employees'}, set())
        assert (INDIVIDUAL_PHRASE in instructions(with_key), INDIVIDUAL_PHRASE in instructions(without_key)) == (
            True,
            False,
        )

    async def test_per_run_instructions_can_be_disabled(self, stackone_server: MCPServer):
        capability = StackOne[str | None](
            account_id='45320',
            auth=per_user_key,
            client=stackone_server,
            actions=['*_list_*'],
            include_instructions=False,
        )
        agent = Agent(TestModel(), deps_type=str | None, capabilities=[capability])
        messages = (await agent.run('list employees', deps='alice-key')).all_messages()
        assert (tool_call_names(messages), INDIVIDUAL_PHRASE in instructions(messages)) == (
            {'bamboohr_list_employees'},
            False,
        )
