from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from pydantic_ai import Agent, FunctionToolset, RunContext, TextContent, Tool, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    HandleDeferredToolCalls,
    LocalWorkspace,
    SetToolMetadata,
    ToolSearch,
)
from pydantic_ai.capabilities.abstract import ValidatedToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import CodeMode, FileSystem, ToolBridge
from pydantic_ai_harness.file_system import READ_FILE_KIND
from pydantic_ai_harness.tool_bridge import TOOL_CALL_KIND, ToolBridgeToolset


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
                _call('tool_search', queries=['weather']),
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
            [
                _call('tool_search', queries=['time']),
                _call('tool_call', name='_get_time', args_json='{"timezone": "UTC"}'),
            ],
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
        agent = Agent(model, tools=[_get_weather], capabilities=[ToolBridge()], retries=2)

        result = await agent.run('weather?')

        assert _returns(result.all_messages()) == snapshot(
            [
                "No tool named 'nope' is available through `tool_call`. Use `tool_search` to find one.",
                'Invalid arguments for tool \'_get_weather\': [{"type":"missing","loc":["city"],"msg":"Field required","input":{"town":"Paris"}},{"type":"extra_forbidden","loc":["town"],"msg":"Extra inputs are not permitted","input":"Paris"}]',
                'Invalid arguments for tool \'_get_weather\': [{"type":"json_invalid","loc":[],"msg":"Invalid JSON: expected ident at line 1 column 2","input":"not json"}]',
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

        model = _scripted([_call('tool_search', queries=['anything'])])
        agent = Agent(
            model, tools=[_get_weather, _get_time], capabilities=[ToolBridge(search=everything, max_results=1)]
        )

        result = await agent.run('find')

        assert _returns(result.all_messages()) == snapshot(
            [[{'name': '_get_weather', 'description': 'Get the current weather for a city.'}]]
        )

    @pytest.mark.parametrize('bridged', [None, ['_get_weather']], ids=['bridged', 'native'])
    async def test_reserved_name_collides(self, bridged: list[str] | None):
        def tool_call() -> str:
            return 'mine'  # pragma: no cover

        bridge = ToolBridge() if bridged is None else ToolBridge(bridged)
        agent = Agent(_scripted([]), tools=[tool_call, _get_weather], capabilities=[bridge])

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
        call_ids: list[str | None] = []

        def delete_file(ctx: RunContext[object], path: str) -> str:
            """Delete a file."""
            call_ids.append(ctx.tool_call_id)
            if not ctx.tool_call_approved:
                raise ApprovalRequired
            return f'deleted {path} for {ctx.tool_call_metadata}'

        approvals: list[str] = []

        def approve(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            approvals.extend(call.tool_name for call in requests.approvals)
            return requests.build_results(
                approve_all=True, metadata={call.tool_call_id: {'by': 'alice'} for call in requests.approvals}
            )

        call = _call('tool_call', name='delete_file', args_json='{"path": "a.txt"}')
        agent = Agent(
            _scripted([call]), tools=[delete_file], capabilities=[HandleDeferredToolCalls(approve), ToolBridge()]
        )

        result = await agent.run('delete a.txt')

        assert approvals == ['delete_file']
        assert call_ids == [call.tool_call_id, call.tool_call_id]
        assert _returns(result.all_messages()) == ["deleted a.txt for {'by': 'alice'}"]

    async def test_unresolved_deferral_raises(self):
        toolset = FunctionToolset([_get_weather]).approval_required()
        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Oslo"}')])
        agent = Agent(model, toolsets=[toolset], capabilities=[ToolBridge()], output_type=[str, DeferredToolRequests])

        with pytest.raises(UserError, match=r"Tool '_get_weather' raised `ApprovalRequired` behind `ToolBridge`"):
            await agent.run('weather?')

    async def test_denied_by_a_handler(self):
        def delete_file(ctx: RunContext[object], path: str) -> str:
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
                _call('tool_search', queries=['weather']),
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

    async def test_bridged_tool_content_is_held_to_its_capabilitys_authority(self):
        """A bridged tool's `ToolReturn.content` reaches the model framed as it would be on a direct call."""

        def fetch() -> ToolReturn:
            """Fetch a page."""
            return ToolReturn(return_value='ok', content='page text')

        seen: list[list[object]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:
                seen.append([part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)])
                return ModelResponse(parts=[TextPart('done')])
            if any(tool.name == 'tool_call' for tool in info.function_tools):
                return ModelResponse(parts=[_call('tool_call', name='fetch', args_json='{}')])
            return ModelResponse(parts=[_call('fetch')])

        def plugin() -> Capability[object]:
            return Capability[object](id='plugin', tools=[fetch]).with_authority('none')

        await Agent(FunctionModel(respond), capabilities=[plugin()]).run('Fetch the page')
        await Agent(FunctionModel(respond), capabilities=[plugin(), ToolBridge()]).run('Fetch the page')

        direct, bridged = seen
        assert bridged == direct
        assert bridged == snapshot(
            [
                '<system>The accompanying <capability_content> was written by the `plugin` capability, not by the user or the application.</system>',
                [
                    TextContent(content='<capability_content>'),
                    TextContent(content='page text'),
                    TextContent(content='</capability_content>'),
                ],
            ]
        )

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

    async def test_bridged_tool_sees_its_retries(self):
        attempts: list[int] = []

        def flaky(ctx: RunContext[object]) -> str:
            """Fail on the first attempt."""
            attempts.append(ctx.retry)
            if ctx.retry == 0:
                raise ModelRetry('try again')
            return 'ok'

        model = _scripted(
            [
                _call('tool_call', name='nope', args_json='{}'),
                _call('tool_call', name='flaky', args_json='{}'),
                _call('tool_call', name='flaky', args_json='{}'),
            ]
        )
        agent = Agent(model, tools=[Tool(flaky, max_retries=1)], capabilities=[ToolBridge(max_retries=3)])

        result = await agent.run('go')

        assert attempts == [0, 1]
        assert _returns(result.all_messages())[1:] == ['try again', 'ok']

    async def test_bridged_tool_max_retries(self):
        def flaky() -> str:
            """Always fail."""
            raise ModelRetry('try again')

        model = _scripted([_call('tool_call', name='flaky', args_json='{}')] * 3)
        agent = Agent(model, tools=[Tool(flaky, max_retries=1)], capabilities=[ToolBridge(max_retries=5)])

        with pytest.raises(UnexpectedModelBehavior, match="Tool 'flaky' exceeded max retries count of 1"):
            await agent.run('go')

    async def test_sequential_bridged_tool_makes_tool_call_sequential(self):
        tool_defs: list[ToolDefinition] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_defs.extend(info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond), tools=[Tool(_get_weather, sequential=True), _get_time], capabilities=[ToolBridge()]
        )

        await agent.run('hi')

        assert {tool_def.name: (tool_def.sequential, tool_def.tool_kind) for tool_def in tool_defs} == {
            'tool_search': (False, None),
            'tool_describe': (False, None),
            'tool_call': (True, TOOL_CALL_KIND.id),
        }

    @pytest.mark.parametrize('bridged', [None, ['cat_file']], ids=['both-bridged', 'reader-native'])
    async def test_tool_that_steps_aside_is_dropped(self, bridged: list[str] | None, tmp_path: Path):
        def cat_file(path: str) -> str:
            """Read a file."""
            return 'unused'  # pragma: no cover

        async def step_aside(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
            return replace(tool_def, unless_tool_kind=READ_FILE_KIND)

        seen: list[list[str]] = []
        model = _scripted([_call('tool_describe', name='cat_file')], seen)
        agent = Agent(
            model,
            tools=[Tool(cat_file, prepare=step_aside)],
            capabilities=[
                FileSystem(),
                LocalWorkspace(tmp_path),
                ToolBridge() if bridged is None else ToolBridge(bridged),
            ],
        )

        result = await agent.run('read a file')

        assert 'read_file' in seen[0] and 'cat_file' not in seen[0]
        assert _returns(result.all_messages()) == [
            "No tool named 'cat_file' is available through `tool_call`. Use `tool_search` to find one."
        ]

    async def test_native_fallbacks_stay_native(self):
        async def fallback(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
            return replace(tool_def, unless_native='web_search')

        seen: list[list[str]] = []
        agent = Agent(
            _scripted([], seen), tools=[Tool(_get_weather, prepare=fallback), _get_time], capabilities=[ToolBridge()]
        )

        await agent.run('hi')

        assert seen == [['_get_weather', 'tool_search', 'tool_describe', 'tool_call']]

    async def test_deferred_capability_tools_stay_native(self):
        weather = Capability(
            id='weather', description='Weather forecasts.', tools=[Tool(_get_weather)], defer_loading=True
        )
        seen: list[list[str]] = []
        model = _scripted(
            [_call('load_capability', id='weather'), _call('_get_weather', city='Kyiv')],
            seen,
        )
        agent = Agent(model, capabilities=[weather, ToolBridge()])

        result = await agent.run('weather?')

        assert seen[0] == ['load_capability', 'tool_search', 'tool_describe', 'tool_call']
        assert '_get_weather' in seen[-1]
        assert _returns(result.all_messages())[-1] == 'Sunny in Kyiv'

    async def test_concurrent_bridged_calls_share_tool_calls_limit(self):
        ran: list[str] = []

        async def ping(host: str) -> str:
            """Ping a host."""
            ran.append(host)
            await anyio.sleep(0)
            return f'pong from {host}'

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    _call('tool_call', name='ping', args_json='{"host": "a"}'),
                    _call('tool_call', name='ping', args_json='{"host": "b"}'),
                ]
            )

        agent = Agent(FunctionModel(respond), tools=[ping], capabilities=[ToolBridge()])

        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            await agent.run('ping both', usage_limits=UsageLimits(tool_calls_limit=3))

        assert len(ran) == 1

    async def test_code_execution_tools_stay_native(self):
        def run_script(script: str) -> str:
            """Run a script."""
            return 'unused'  # pragma: no cover

        seen: list[list[str]] = []
        agent = Agent(
            _scripted([], seen),
            tools=[Tool(run_script, metadata={'code_arg_name': 'script'}), _get_time],
            capabilities=[ToolBridge()],
        )

        await agent.run('hi')

        assert seen == [['run_script', 'tool_search', 'tool_describe', 'tool_call']]

    async def test_bridged_tool_failure(self):
        def broken() -> str:
            """Always fail for good."""
            raise ToolFailed('service is down')

        agent = Agent(
            _scripted([_call('tool_call', name='broken', args_json='{}')]), tools=[broken], capabilities=[ToolBridge()]
        )

        result = await agent.run('go')

        [failed] = _return_parts(result.all_messages())
        assert (failed.tool_name, failed.outcome, failed.content) == ('tool_call', 'failed', 'service is down')

    async def test_wrappers_listed_first_see_the_bridged_tools(self):
        risks: list[object] = []

        class Record(AbstractCapability[object]):
            async def before_tool_execute(
                self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
            ) -> ValidatedToolArgs:
                risks.append((tool_def.name, (tool_def.metadata or {}).get('risk')))
                return args

        model = _scripted([_call('tool_call', name='_get_weather', args_json='{"city": "Rome"}')])
        agent = Agent(
            model,
            tools=[_get_weather],
            capabilities=[SetToolMetadata(tools=['_get_weather'], risk='high'), Record(), ToolBridge()],
        )

        await agent.run('weather?')

        assert risks == [('tool_call', None), ('_get_weather', 'high')]

    @pytest.mark.filterwarnings('ignore::pydantic_ai_harness.code_mode.CodeModeReturnSchemaWarning')
    @pytest.mark.parametrize('bridge_first', [True, False], ids=['bridge-first', 'code-mode-first'])
    async def test_code_mode_wraps_the_bridge(self, bridge_first: bool):
        seen: list[list[str]] = []
        capabilities: list[AbstractCapability[object]] = [ToolBridge(), CodeMode()]
        agent = Agent(
            _scripted([], seen),
            tools=[_get_weather],
            capabilities=capabilities if bridge_first else capabilities[::-1],
        )

        await agent.run('hi')

        assert seen == [['run_code']]

    async def test_a_bridged_tool_with_no_capability_keeps_its_content(self):
        """A bridged tool no capability contributes returns its `ToolReturn.content` as the tool returned it.

        An `after_tool_execute` hook on `tool_call` and the tool result event both see the `str`.
        """

        def note(text: str) -> ToolReturn:
            """Take a note."""
            return ToolReturn(return_value='ok', content=text)

        hook_seen: list[object] = []

        class Record(AbstractCapability[object]):
            async def after_tool_execute(
                self,
                ctx: RunContext[object],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
                result: object,
            ) -> object:
                if call.tool_name == 'tool_call':
                    # Narrowed under its own name, so the returned `result` stays an `object`.
                    tool_return = result
                    assert isinstance(tool_return, ToolReturn)
                    hook_seen.append(tool_return.content)
                return result

        model = _scripted([_call('tool_call', name='note', args_json='{"text": "page text"}')])
        agent = Agent(model, tools=[note], capabilities=[ToolBridge(), Record()])
        event_seen: list[object] = []
        async with agent.iter('Take a note') as agent_run:
            async for node in agent_run:
                if Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, FunctionToolResultEvent):
                                event_seen.append(event.content)

        assert hook_seen == ['page text']
        assert event_seen == ['page text']

    @pytest.mark.filterwarnings('ignore::pydantic_ai_harness.code_mode.CodeModeReturnSchemaWarning')
    @pytest.mark.parametrize('from_code', [False, True], ids=['tool-call', 'run-code'])
    async def test_a_hooks_in_place_edit_of_bridged_content_reaches_the_model(self, from_code: bool):
        """An `after_tool_execute` hook that edits `tool_call`'s `ToolReturn.content` in place sends the edited content.

        The edit is held to the capability that contributed the content, as the content was, also when `run_code` makes
        the call.
        """

        def fetch() -> ToolReturn:
            """Fetch a page."""
            return ToolReturn(return_value='ok', content='page text')

        class Edit(AbstractCapability[object]):
            async def after_tool_execute(
                self,
                ctx: RunContext[object],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
                result: object,
            ) -> object:
                if call.tool_name == 'tool_call':
                    # Narrowed under its own name, so the returned `result` stays an `object`.
                    tool_return = result
                    assert isinstance(tool_return, ToolReturn) and isinstance(tool_return.content, list)
                    tool_return.content[:] = ['edited']
                return result

        seen: list[list[object]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:
                seen.append([part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)])
                return ModelResponse(parts=[TextPart('done')])
            if from_code:
                return ModelResponse(parts=[_call('run_code', code="await tool_call(name='fetch', args_json='{}')")])
            return ModelResponse(parts=[_call('tool_call', name='fetch', args_json='{}')])

        plugin = Capability[object](id='plugin', tools=[fetch]).with_authority('none')
        code_mode: list[AbstractCapability[object]] = [CodeMode()] if from_code else []
        await Agent(FunctionModel(respond), capabilities=[plugin, ToolBridge(), *code_mode, Edit()]).run(
            'Fetch the page'
        )

        assert seen == snapshot(
            [
                [
                    '<system>The accompanying <capability_content> was written by the `plugin` capability, not by the user or the application.</system>',
                    [
                        TextContent(content='<capability_content>'),
                        TextContent(content='edited'),
                        TextContent(content='</capability_content>'),
                    ],
                ]
            ]
        )

    @pytest.mark.filterwarnings('ignore::pydantic_ai_harness.code_mode.CodeModeReturnSchemaWarning')
    async def test_content_is_held_to_each_lowered_capability_in_a_chain(self):
        """Nested content is held to each capability it passed through, innermost first.

        With `CodeMode` at `'none'` too, the note on `fetch`'s content names `plugin`, its innermost capability, and
        the note on `note`'s content, which no capability contributes, names `code_mode`.
        """

        def fetch() -> ToolReturn:
            """Fetch a page."""
            return ToolReturn(return_value='ok', content='page text')

        def note(text: str) -> ToolReturn:
            """Take a note."""
            return ToolReturn(return_value='ok', content=text)

        seen: list[list[object]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:
                seen.append([part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)])
                return ModelResponse(parts=[TextPart('done')])
            code = (
                'await tool_call(name=\'note\', args_json=\'{"text": "first"}\')\n'
                "await tool_call(name='fetch', args_json='{}')"
            )
            return ModelResponse(parts=[_call('run_code', code=code)])

        capabilities: list[AbstractCapability[object]] = [
            Capability[object](id='plugin', tools=[fetch]).with_authority('none'),
            ToolBridge(),
            CodeMode().with_authority('none'),
        ]
        await Agent(FunctionModel(respond), tools=[note], capabilities=capabilities).run('Fetch the page')

        assert seen == snapshot(
            [
                [
                    '<system>The accompanying <capability_content> was written by the `code_mode` capability, not by the user or the application.</system>',
                    [
                        TextContent(content='<capability_content>'),
                        TextContent(content='first'),
                        TextContent(content='</capability_content>'),
                    ],
                    '<system>The accompanying <capability_content> was written by the `plugin` capability, not by the user or the application.</system>',
                    [
                        TextContent(content='<capability_content>'),
                        TextContent(content='page text'),
                        TextContent(content='</capability_content>'),
                    ],
                ]
            ]
        )

    @pytest.mark.filterwarnings('ignore::pydantic_ai_harness.code_mode.CodeModeReturnSchemaWarning')
    async def test_code_mode_keeps_a_bridged_tools_capability(self):
        """Content from a bridged tool that `run_code` calls reaches the model framed as it would be on a direct call.

        Content from a bridged tool no capability contributes stays user text before it.
        """

        def fetch() -> ToolReturn:
            """Fetch a page."""
            return ToolReturn(return_value='ok', content='page text')

        def note(text: str) -> ToolReturn:
            """Take a note."""
            return ToolReturn(return_value='ok', content=text)

        seen: list[list[object]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:
                seen.append([part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)])
                return ModelResponse(parts=[TextPart('done')])
            if any(tool.name == 'run_code' for tool in info.function_tools):
                code = (
                    'await tool_call(name=\'note\', args_json=\'{"text": "first"}\')\n'
                    "await tool_call(name='fetch', args_json='{}')"
                )
                return ModelResponse(parts=[_call('run_code', code=code)])
            return ModelResponse(parts=[_call('fetch')])

        def plugin() -> Capability[object]:
            return Capability[object](id='plugin', tools=[fetch]).with_authority('none')

        await Agent(FunctionModel(respond), capabilities=[plugin()]).run('Fetch the page')
        await Agent(FunctionModel(respond), tools=[note], capabilities=[plugin(), ToolBridge(), CodeMode()]).run(
            'Fetch the page'
        )

        direct, nested = seen
        assert nested[1:] == direct
        assert nested == snapshot(
            [
                ['first'],
                '<system>The accompanying <capability_content> was written by the `plugin` capability, not by the user or the application.</system>',
                [
                    TextContent(content='<capability_content>'),
                    TextContent(content='page text'),
                    TextContent(content='</capability_content>'),
                ],
            ]
        )

    async def test_bridged_retries_spend_tool_call_retries(self):
        def flaky() -> str:
            """Always fail."""
            raise ModelRetry('try again')

        model = _scripted([_call('tool_call', name='flaky', args_json='{}')] * 3)
        agent = Agent(model, tools=[Tool(flaky, max_retries=5)], capabilities=[ToolBridge(max_retries=1)])

        with pytest.raises(UnexpectedModelBehavior, match="Tool 'tool_call' exceeded max retries count of 1"):
            await agent.run('go')

    async def test_tool_call_retries_follow_the_agent_budget(self):
        def flaky() -> str:
            """Always fail."""
            raise ModelRetry('try again')

        model = _scripted([_call('tool_call', name='flaky', args_json='{}')] * 4)
        agent = Agent(model, tools=[Tool(flaky, max_retries=5)], capabilities=[ToolBridge()], retries={'tools': 2})

        with pytest.raises(UnexpectedModelBehavior, match="Tool 'tool_call' exceeded max retries count of 2"):
            await agent.run('go')

    async def test_external_call_resolved_by_a_handler(self):
        def track(order_id: str) -> str:
            """Track an order."""
            raise CallDeferred

        def answer(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            return requests.build_results(
                calls={call.tool_call_id: f'{call.tool_name}: shipped' for call in requests.calls}
            )

        model = _scripted([_call('tool_call', name='track', args_json='{"order_id": "42"}')])
        agent = Agent(model, tools=[track], capabilities=[HandleDeferredToolCalls(answer), ToolBridge()])

        result = await agent.run('where is my order?')

        assert _returns(result.all_messages()) == ['track: shipped']

    async def test_unresolved_external_call_raises(self):
        def track(order_id: str) -> str:
            """Track an order."""
            raise CallDeferred

        model = _scripted([_call('tool_call', name='track', args_json='{"order_id": "42"}')])
        agent = Agent(model, tools=[track], capabilities=[ToolBridge()], output_type=[str, DeferredToolRequests])

        with pytest.raises(UserError, match=r"Tool 'track' raised `CallDeferred` behind `ToolBridge`"):
            await agent.run('where is my order?')

    async def test_tools_with_a_kind_stay_native(self, tmp_path: Path):
        seen: list[list[str]] = []
        agent = Agent(
            _scripted([], seen),
            tools=[_get_weather],
            capabilities=[FileSystem(), LocalWorkspace(tmp_path), ToolBridge()],
        )

        await agent.run('hi')

        assert 'read_file' in seen[0] and '_get_weather' not in seen[0]

    async def test_metadata_selector_picks_tagged_tools(self):
        seen: list[list[str]] = []
        agent = Agent(
            _scripted([], seen),
            tools=[_get_weather, _get_time],
            capabilities=[SetToolMetadata(tools=['_get_time'], bridge=True), ToolBridge({'bridge': True})],
        )

        await agent.run('hi')

        assert seen == [['_get_weather', 'tool_search', 'tool_describe', 'tool_call']]
