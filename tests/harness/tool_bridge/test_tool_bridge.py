from __future__ import annotations

from collections.abc import Sequence

import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from pydantic_ai import Agent, FunctionToolset, RunContext, Tool, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, ToolSearch
from pydantic_ai.capabilities.abstract import ValidatedToolArgs
from pydantic_ai.exceptions import ApprovalRequired, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import ToolBridge
from pydantic_ai_harness.tool_bridge import ToolBridgeToolset


def _get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'Sunny in {city}'


def _get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f'12:00 in {timezone}'


def _scripted(calls: Sequence[ToolCallPart | None], seen: list[list[str]] | None = None) -> FunctionModel:
    """A model that makes one call per step, then answers `done`; `seen` gets the tool names of each request."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append([tool.name for tool in info.function_tools])
        step = sum(isinstance(message, ModelResponse) for message in messages)
        call = calls[step] if step < len(calls) else None
        return ModelResponse(parts=[call if call is not None else TextPart('done')])

    return FunctionModel(respond)


def _call(tool_name: str, /, **args: object) -> ToolCallPart:
    return ToolCallPart(tool_name, args)


def _returns(messages: list[ModelMessage]) -> list[object]:
    return [part.content for part in _return_parts(messages)]


def _return_parts(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


class TestToolBridge:
    async def test_search_describe_call(self):
        seen: list[list[str]] = []
        model = _scripted(
            [
                _call('tool_search', query='weather'),
                _call('tool_describe', name='_get_weather'),
                _call('tool_call', name='_get_weather', args_json='{"city": "Paris"}'),
            ],
            seen,
        )
        agent = Agent(model, tools=[_get_weather, _get_time], capabilities=[ToolBridge()])

        result = await agent.run('weather in Paris?')

        assert seen == [['tool_search', 'tool_describe', 'tool_call']] * 4
        assert _returns(result.all_messages()) == snapshot(
            [
                [{'name': '_get_weather', 'description': 'Get the current weather for a city.'}],
                {
                    'name': '_get_weather',
                    'description': 'Get the current weather for a city.',
                    'parameters_json_schema': {
                        'additionalProperties': False,
                        'properties': {'city': {'type': 'string'}},
                        'required': ['city'],
                        'type': 'object',
                    },
                },
                'Sunny in Paris',
            ]
        )

    async def test_tool_list_is_fixed_while_tools_change(self):
        seen: list[list[str]] = []
        model = _scripted(
            [_call('tool_search', query='time'), _call('tool_call', name='_get_time', args_json='{"timezone": "UTC"}')],
            seen,
        )
        agent = Agent(model, capabilities=[ToolBridge()])

        @agent.toolset
        def growing(ctx: RunContext[object]) -> FunctionToolset[object]:
            return FunctionToolset([_get_weather] if ctx.run_step <= 1 else [_get_weather, _get_time])

        result = await agent.run('time?')

        assert seen == [['tool_search', 'tool_describe', 'tool_call']] * 3
        assert _returns(result.all_messages()) == snapshot([[], '12:00 in UTC'])

    async def test_selected_tools_only(self):
        seen: list[list[str]] = []
        agent = Agent(_scripted([], seen), tools=[_get_weather, _get_time], capabilities=[ToolBridge(['_get_time'])])

        await agent.run('hi')

        assert seen == [['_get_weather', 'tool_search', 'tool_describe', 'tool_call']]

    async def test_native_tool_passes_through(self):
        model = _scripted([_call('_get_weather', city='Oslo')])
        agent = Agent(model, tools=[_get_weather, _get_time], capabilities=[ToolBridge(['_get_time'])])

        result = await agent.run('weather?')

        assert _returns(result.all_messages()) == ['Sunny in Oslo']

    async def test_mistakes_are_retried(self):
        model = _scripted(
            [
                _call('tool_describe', name='nope'),
                _call('tool_call', name='_get_weather', args_json='{"town": "Paris"}'),
                _call('tool_call', name='_get_weather', args_json='not json'),
            ]
        )
        agent = Agent(model, tools=[_get_weather], capabilities=[ToolBridge()])

        result = await agent.run('weather?')

        assert _returns(result.all_messages()) == snapshot(
            [
                "No tool named 'nope' is available through `tool_call`. Use `tool_search` to find one.",
                "Invalid arguments for tool '_get_weather': [{'type': 'missing', 'loc': ('city',), 'msg': 'Field required', 'input': {'town': 'Paris'}}, {'type': 'extra_forbidden', 'loc': ('town',), 'msg': 'Extra inputs are not permitted', 'input': 'Paris'}]",
                "Invalid arguments for tool '_get_weather': [{'type': 'json_invalid', 'loc': (), 'msg': 'Invalid JSON: expected ident at line 1 column 2', 'input': 'not json'}]",
            ]
        )

    async def test_empty_args_json(self):
        def ping() -> str:
            """Check the service."""
            return 'pong'

        agent = Agent(
            _scripted([_call('tool_call', name='ping', args_json='')]), tools=[ping], capabilities=[ToolBridge()]
        )

        result = await agent.run('ping')

        assert _returns(result.all_messages()) == ['pong']

    async def test_custom_search_and_max_results(self):
        async def everything(
            ctx: RunContext[object], queries: Sequence[str], tools: Sequence[ToolDefinition]
        ) -> list[str]:
            return ['missing', *(tool.name for tool in tools)]

        model = _scripted([_call('tool_search', query='anything')])
        agent = Agent(
            model, tools=[_get_weather, _get_time], capabilities=[ToolBridge(search=everything, max_results=1)]
        )

        result = await agent.run('find')

        assert _returns(result.all_messages()) == snapshot(
            [[{'name': '_get_weather', 'description': 'Get the current weather for a city.'}]]
        )

    async def test_reserved_name_collides(self):
        def tool_call() -> str:
            return 'mine'  # pragma: no cover

        agent = Agent(_scripted([]), tools=[tool_call, _get_weather], capabilities=[ToolBridge(['_get_weather'])])

        with pytest.raises(UserError, match="Tool name 'tool_call' is reserved for `ToolBridge`"):
            await agent.run('hi')

    async def test_hooks_see_the_bridged_tool(self):
        executed: list[str] = []

        class Record(AbstractCapability[object]):
            async def before_tool_execute(
                self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
            ) -> ValidatedToolArgs:
                executed.append(call.tool_name)
                return args

        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Rome"}')])
        agent = Agent(model, tools=[_get_weather], capabilities=[Record(), ToolBridge()])

        await agent.run('weather?')

        assert executed == ['tool_call', '_get_weather']

    async def test_bridged_call_gets_its_own_tool_span(self, capfire: CaptureLogfire):
        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Rome"}')])
        agent = Agent(model, tools=[_get_weather], capabilities=[ToolBridge()])
        agent.instrument = InstrumentationSettings()

        await agent.run('weather?')

        spans = capfire.exporter.exported_spans_as_dict()
        assert [
            span['attributes'].get('gen_ai.tool.name') for span in spans if 'gen_ai.tool.name' in span['attributes']
        ] == [
            '_get_weather',
            'tool_call',
        ]

    async def test_approval_required_tools_stay_native(self):
        seen: list[list[str]] = []
        agent = Agent(
            _scripted([], seen),
            tools=[Tool(_get_weather, requires_approval=True), _get_time],
            capabilities=[ToolBridge()],
        )

        await agent.run('hi')

        assert seen == [['_get_weather', 'tool_search', 'tool_describe', 'tool_call']]

    async def test_approval_raised_by_a_bridged_tool(self):
        def delete_file(ctx: RunContext[None], path: str) -> str:
            """Delete a file."""
            if not ctx.tool_call_approved:
                raise ApprovalRequired
            return f'deleted {path}'

        model = _scripted([_call('tool_call', name='delete_file', args_json='{"path": "a.txt"}')])
        agent = Agent(model, tools=[delete_file], capabilities=[ToolBridge()], output_type=[str, DeferredToolRequests])

        first = await agent.run('delete a.txt')
        assert isinstance(first.output, DeferredToolRequests)
        assert [call.tool_name for call in first.output.approvals] == ['tool_call']

        results = first.output.build_results(approve_all=True)
        second = await agent.run(message_history=first.all_messages(), deferred_tool_results=results)

        assert _returns(second.new_messages()) == ['deleted a.txt']

    async def test_denied_by_a_handler(self):
        def delete_file(ctx: RunContext[None], path: str) -> str:
            """Delete a file."""
            raise ApprovalRequired

        def deny(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            return requests.build_results(approvals={call.tool_call_id: False for call in requests.approvals})

        model = _scripted([_call('tool_call', name='delete_file', args_json='{"path": "a.txt"}')])
        agent = Agent(model, tools=[delete_file], capabilities=[HandleDeferredToolCalls(deny), ToolBridge()])

        result = await agent.run('delete a.txt')

        [denied] = _return_parts(result.all_messages())
        assert (denied.tool_name, denied.outcome, denied.content) == snapshot(
            ('tool_call', 'denied', 'The tool call was denied.')
        )

    async def test_deferred_loading_tools_are_bridged(self):
        seen: list[list[str]] = []
        model = _scripted(
            [
                _call('tool_search', query='weather'),
                _call('tool_call', name='_get_weather', args_json='{"city": "Oslo"}'),
            ],
            seen,
        )
        agent = Agent(model, tools=[Tool(_get_weather, defer_loading=True)], capabilities=[ToolSearch(), ToolBridge()])

        result = await agent.run('weather?')

        assert seen == [['tool_search', 'tool_describe', 'tool_call']] * 3
        assert _returns(result.all_messages())[-1] == 'Sunny in Oslo'

    async def test_capability_tools_are_bridged(self):
        class Weather(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                return FunctionToolset([_get_weather])

        seen: list[list[str]] = []
        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Lima"}')], seen)
        agent = Agent(model, capabilities=[Weather(), ToolBridge()])

        result = await agent.run('weather?')

        assert seen[0] == ['tool_search', 'tool_describe', 'tool_call']
        assert _returns(result.all_messages()) == ['Sunny in Lima']

    async def test_bridged_call_counts_toward_tool_calls_limit(self):
        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Oslo"}')])
        agent = Agent(model, tools=[_get_weather], capabilities=[ToolBridge()])

        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            await agent.run('weather?', usage_limits=UsageLimits(tool_calls_limit=1))

        result = await agent.run('weather?', usage_limits=UsageLimits(tool_calls_limit=2))
        assert result.usage.tool_calls == 2

    async def test_toolset_needs_the_tool_manager(self):
        toolset = ToolBridgeToolset[None](FunctionToolset([_get_weather]))
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = await toolset.get_tools(ctx)

        with pytest.raises(UserError, match="needs the run's tool manager"):
            await toolset.call_tool('tool_call', {'name': '_get_weather', 'args_json': '{}'}, ctx, tools['tool_call'])
