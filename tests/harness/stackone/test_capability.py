"""Tests for the `StackOne` capability through the public `Agent(capabilities=[...])` surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.stackone import StackOne
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


def tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def request_instructions(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    return first.instructions or ''


class TestStackOne:
    def test_serialization_name(self):
        assert StackOne.get_serialization_name() == 'StackOne'

    def test_agent_spec_schema_excludes_runtime_client_and_accepts_scalar_actions(self):
        schema = AgentSpec.model_json_schema_with_capabilities([StackOne])
        schema_text = json.dumps(schema, sort_keys=True)
        assert '"client"' not in schema_text
        assert (
            '"actions": {"anyOf": [{"type": "string"}, {"items": {"type": "string"}, "type": "array"}]' in schema_text
        )

    def test_missing_api_key_fails_at_agent_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            Agent(TestModel(), capabilities=[StackOne(account_id='45320')])

    def test_two_accounts_stay_two_capabilities(self):
        """One account is one provider connection, so the account is what names the capability.

        A fixed `'stackone'` id would make these two merge into one, and one linked account would
        silently drop out of the agent -- the failure mode a shared id is supposed to prevent.
        """
        first = StackOne(account_id='45320', api_key='key')
        second = StackOne(account_id='99811', api_key='key')
        assert (first.id, second.id) == ('stackone-45320', 'stackone-99811')

        agent = Agent(TestModel(), capabilities=[first, second])
        ids = [capability.id for capability in agent._root_capability.capabilities]  # pyright: ignore[reportPrivateUsage]
        assert 'stackone-45320' in ids and 'stackone-99811' in ids

    def test_two_of_one_account_are_a_collision(self):
        """Two capabilities on the same connection are a mistake, not a configuration to merge."""
        with pytest.raises(UserError, match="Capability id 'stackone-45320' is used by multiple capabilities"):
            Agent(
                TestModel(),
                capabilities=[
                    StackOne(account_id='45320', api_key='key'),
                    StackOne(account_id='45320', api_key='key'),
                ],
            )

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        assert StackOne(account_id='45320').get_toolset() is not None

    def test_secrets_are_hidden_from_repr(self):
        capability = StackOne(
            account_id='45320',
            api_key='api-secret',
            client='https://user:url-secret@example.com/mcp?signature=query-secret',
        )
        representation = repr(capability)
        assert 'api-secret' not in representation
        assert 'url-secret' not in representation
        assert 'query-secret' not in representation

    def test_construction_does_not_warn(self, recwarn: pytest.WarningsRecorder):
        StackOne(account_id='45320', api_key='key', actions=['*_list_*'])
        StackOne(account_id='45320', api_key='key', defer_loading=True)
        assert not recwarn.list

    def test_agent_spec_loads_stackone(self, tmp_path: Path):
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            """\
capabilities:
  - StackOne:
      account_id: '45320'
      api_key: 'key'
      actions: '*_list_*'
""",
            encoding='utf-8',
        )
        agent = Agent.from_file(spec, custom_capability_types=[StackOne], model=TestModel())
        assert isinstance(agent, Agent)

    @pytest.mark.parametrize(
        ('arguments', 'match'),
        [
            ({'tool_mode': 'search-execute'}, '`tool_mode` must be'),
        ],
    )
    def test_agent_spec_rejects_invalid_configuration(self, arguments: dict[str, object], match: str):
        spec = {
            'capabilities': [
                {'StackOne': {'account_id': '45320', 'api_key': 'key', **arguments}},
            ]
        }
        with pytest.raises(ValueError, match=match):
            Agent.from_spec(spec, custom_capability_types=[StackOne], model=TestModel())

    def test_rejects_actions_in_explicit_search_execute(self):
        with pytest.raises(UserError, match='cannot apply in `search_execute` mode'):
            StackOne(account_id='45320', api_key='key', tool_mode='search_execute', actions=['*_list_*'])

    async def test_agent_run_calls_stackone_tools(self, stackone_server: FastMCP):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, tool_mode='individual')
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert 'bamboohr_list_employees' in tool_call_names(result.all_messages())
        assert 'Ada' in result.output

    async def test_tool_output_limits_reduces_large_stackone_result(self, stackone_server: FastMCP):
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart(tool_name='bamboohr_export_employees', args={'lines': 500})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[
                StackOne(
                    account_id='45320',
                    api_key='key',
                    client=stackone_server,
                    tool_mode='individual',
                ),
                ToolOutputLimits(bands=[Band(over=64, action=Truncate())]),
            ],
        )
        result = await agent.run('export employees')
        returns = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'bamboohr_export_employees'
        ]
        assert len(returns) == 1
        content = returns[0].content
        assert isinstance(content, str)
        assert 'truncated' in content
        assert len(content) < len('\n'.join(f'employee-{index}' for index in range(500)))

    @pytest.mark.parametrize('actions', [['*_list_*'], '*_list_*'])
    async def test_actions_glob_filter(self, stackone_server: FastMCP, actions: list[str] | str):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, actions=actions)
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_list_employees'}

    async def test_deferred_loading_uses_stable_default_id(self, stackone_server: FastMCP):
        # The default id names the linked account, so an agent can reach two of them.
        capability_id = 'stackone-45320'
        seen_revealed: list[bool] = []
        seen_instructions: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # `defer_loading` records what the capability asked for and stays `True` across a
            # reveal; `revealed_tool_names` is what the model can actually call this step.
            revealed = 'bamboohr_list_employees' in info.model_request_parameters.revealed_tool_names
            seen_revealed.append(revealed)
            seen_instructions.append(info.instructions or '')
            if not revealed:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': capability_id})])
            return ModelResponse(parts=[TextPart('done')])

        capability = StackOne(
            account_id='45320',
            api_key='key',
            client=stackone_server,
            actions=['*_list_*'],
            defer_loading=True,
        )
        assert capability.id == capability_id
        agent = Agent(FunctionModel(model_fn), capabilities=[capability])
        result = await agent.run('list employees')

        assert result.output == 'done'
        assert seen_revealed == [False, True]
        assert '{connector}_{action}_{entity}' not in seen_instructions[0]
        returns = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, LoadCapabilityReturnPart)
        ]
        assert '{connector}_{action}_{entity}' in returns[-1].content.get('instructions', '')

    async def test_instructions_follow_the_resolved_mode(self, stackone_server: FastMCP):
        individual = StackOne(account_id='45320', api_key='key', client=stackone_server, actions=['*_list_*'])
        agent = Agent(TestModel(), capabilities=[individual])
        result = await agent.run('list employees')
        assert '{connector}_{action}_{entity}' in request_instructions(result.all_messages())
        default = StackOne(account_id='45320', api_key='key', client=stackone_server)
        agent = Agent(TestModel(), capabilities=[default])
        result = await agent.run('list employees')
        assert 'must never be guessed' in request_instructions(result.all_messages())

    async def test_instructions_disabled(self, stackone_server: FastMCP):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, include_instructions=False)
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert '{connector}_{action}_{entity}' not in request_instructions(result.all_messages())

    async def test_search_execute_mode(self, search_execute_server: FastMCP):
        capability = StackOne(
            account_id='45320', api_key='key', tool_mode='search_execute', client=search_execute_server
        )
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_search_actions', 'bamboohr_execute_action'}
        assert 'must never be guessed' in request_instructions(result.all_messages())

    async def test_metadata_merged_onto_tools(self, stackone_server: FastMCP, run_context: RunContext[None]):
        capability = StackOne(
            account_id='45320', api_key='key', client=stackone_server, metadata={'code_mode': True, 'team': 'hr'}
        )
        toolset = capability.get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert tools
        for tool in tools.values():
            assert tool.tool_def.metadata is not None
            assert tool.tool_def.metadata.get('code_mode') is True
            assert tool.tool_def.metadata.get('team') == 'hr'
