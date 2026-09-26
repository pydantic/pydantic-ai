"""Tests for the `StackOne` capability through the public `Agent(capabilities=[...])` surface."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.stackone import StackOne

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def http_transport(capability: StackOne[None]) -> StreamableHttpTransport:
    transport = capability.get_toolset().client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestStackOne:
    def test_settings_reach_the_connection(self):
        capability = StackOne[None](account_id='45320', api_key='key', base_url='https://api.eu1.stackone.com')
        transport = http_transport(capability)
        assert (transport.url, transport.headers['x-account-id']) == (
            'https://api.eu1.stackone.com/mcp?tool-mode=search_execute',
            '45320',
        )

    @pytest.mark.parametrize(
        ('capability', 'toolset_id'),
        [
            (StackOne[None](account_id='45320', api_key='key'), 'stackone-45320'),
            (StackOne[None](account_id='45320', api_key='key', id='hr'), 'hr'),
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
        first = StackOne(account_id='45320', api_key='key')
        second = StackOne(account_id='99811', api_key='key')
        assert (first.id, second.id) == ('stackone-45320', 'stackone-99811')

    @pytest.mark.parametrize(
        'capability',
        [
            StackOne[None](account_id='45320', api_key='secret'),
            StackOne[None](account_id='45320', api_key='key', client='https://user:secret@example.com/mcp'),
        ],
        ids=['api-key', 'client'],
    )
    def test_secrets_are_hidden_from_repr(self, capability: StackOne[None]):
        assert 'secret' not in repr(capability)

    def test_agent_spec_schema_excludes_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([StackOne])
        assert '"client"' not in json.dumps(schema)

    def test_agent_spec_forwards_api_key(self, monkeypatch: pytest.MonkeyPatch):
        # With no `STACKONE_API_KEY` to fall back on, construction only succeeds if the spec's key arrives.
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        spec = {'capabilities': [{'StackOne': {'account_id': '45320', 'api_key': 'key'}}]}
        Agent.from_spec(spec, custom_capability_types=[StackOne], model=TestModel())

    @pytest.mark.parametrize(
        ('arguments', 'match'),
        [
            ({'tool_mode': 'search-execute'}, '`tool_mode` must be'),
            ({'tool_mode': 'search_execute', 'actions': '*_list_*'}, 'cannot apply in `search_execute` mode'),
        ],
    )
    def test_agent_spec_rejects_invalid_configuration(self, arguments: dict[str, object], match: str):
        spec = {'capabilities': [{'StackOne': {'account_id': '45320', 'api_key': 'key', **arguments}}]}
        with pytest.raises(ValueError, match=match):
            Agent.from_spec(spec, custom_capability_types=[StackOne], model=TestModel())

    @pytest.mark.anyio
    @pytest.mark.parametrize('actions', [['*_list_*'], '*_LIST_*'])
    async def test_agent_calls_only_matching_actions(self, stackone_server: FastMCP, actions: list[str] | str):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, actions=actions)
        result = await Agent(TestModel(), capabilities=[capability]).run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_list_employees'}

    @pytest.mark.anyio
    async def test_metadata_overrides_server_metadata(self, stackone_server: FastMCP, run_context: RunContext[None]):
        # `task` collides with a server-provided key: user metadata must win, matching `.with_metadata()`.
        toolset = StackOne(
            account_id='45320', api_key='key', client=stackone_server, metadata={'task': 'hr'}
        ).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert [(tool.tool_def.metadata or {}).get('task') for tool in tools.values()] == ['hr', 'hr']

    @pytest.mark.parametrize(
        ('capability', 'phrase'),
        [
            (StackOne[None](account_id='45320', api_key='key'), 'must never be guessed'),
            (
                StackOne[None](account_id='45320', api_key='key', tool_mode='individual'),
                '{connector}_{action}_{entity}',
            ),
            (StackOne[None](account_id='45320', api_key='key', actions='*_list_*'), '{connector}_{action}_{entity}'),
        ],
        ids=['default', 'individual', 'actions'],
    )
    def test_instructions_follow_the_tool_mode(self, capability: StackOne[None], phrase: str):
        assert phrase in (capability.get_instructions() or '')

    def test_instructions_can_be_disabled(self):
        assert StackOne(account_id='45320', api_key='key', include_instructions=False).get_instructions() is None
