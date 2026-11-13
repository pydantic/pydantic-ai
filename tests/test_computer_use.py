"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError

from pydantic_ai import Agent, RunContext, Tool, capture_run_messages
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    HoldKeys,
    SafetyCheck,
    Screenshot,
    Wait,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.native_tools import AbstractNativeTool, ComputerTool, MemoryTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import ToolDefinition

SCREEN = BinaryImage(data=b'png', media_type='image/png')


def test_a_computer_call_is_promoted_and_reads_its_actions() -> None:
    response = ModelResponse(
        parts=[
            ToolCallPart(
                'my_computer',
                {'actions': [{'type': 'click', 'x': 1, 'y': 2}, {'type': 'screenshot'}]},
                tool_kind=COMPUTER_KIND.id,
                provider_name='openai',
                provider_details={'safety_checks': [{'id': 'sc_1', 'code': 'malicious', 'message': 'Check it.'}]},
            )
        ]
    )
    call = response.parts[0]
    assert isinstance(call, ComputerCallPart)
    assert call.actions == [Click(x=1, y=2), Screenshot()]
    assert call.safety_checks == [SafetyCheck(code='malicious', message='Check it.', id='sc_1')]
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))[0]
    assert isinstance(restored, ModelResponse) and restored.parts == response.parts


def test_a_computer_call_without_complete_actions_or_safety_checks() -> None:
    assert ComputerCallPart('computer', '{"actions": [').actions is None
    assert ComputerCallPart('computer', {'actions': [{'type': 'screenshot'}]}).safety_checks == []


@pytest.mark.parametrize(
    'checks',
    ['flagged', ['flagged'], [{'id': 5}], {}, '', 0, False, None],
    ids=['text', 'list-of-text', 'bad-id', 'empty-dict', 'empty-text', 'zero', 'false', 'null'],
)
def test_safety_checks_a_history_garbled_still_flag_the_call(checks: object) -> None:
    # A history can come back from storage or a UI: what it kept for the checks can't be read, but the call was flagged.
    call = ComputerCallPart(
        'computer', {'actions': [{'type': 'screenshot'}]}, provider_details={'safety_checks': checks}
    )
    assert call.safety_checks == [SafetyCheck()]


def test_a_computer_result_reads_its_outcome_and_screenshot() -> None:
    request = ModelRequest(
        parts=[
            ToolReturnPart(
                'my_computer',
                ['Clicked.', SCREEN],
                metadata={'performed': 1, 'screenshot_size': (1280, 800)},
                tool_kind=COMPUTER_KIND.id,
            )
        ]
    )
    result = request.parts[0]
    assert isinstance(result, ComputerReturnPart)
    assert result.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert result.screenshot == SCREEN


def test_a_computer_result_without_metadata_or_a_final_screenshot() -> None:
    # A result a UI rebuilt has no metadata; one whose screenshot was dropped from the history ends with text.
    result = ComputerReturnPart('computer', [SCREEN, 'An earlier screenshot was removed.'])
    assert result.result is None
    assert result.screenshot is None


def test_a_failed_computer_result_stays_a_base_part() -> None:
    request = ModelRequest(parts=[ToolReturnPart('computer', 'boom', outcome='failed', tool_kind=COMPUTER_KIND.id)])
    assert not isinstance(request.parts[0], ComputerReturnPart)


def _computer_tool_def(*, defer_loading: bool = False) -> ToolDefinition:
    return ToolDefinition(name='my_computer', tool_kind=COMPUTER_KIND, defer_loading=defer_loading)


def _prepare(params: ModelRequestParameters) -> ModelRequestParameters:
    model = FunctionModel(lambda m, i: ModelResponse(parts=[]))  # supports every native tool
    return model.prepare_request(None, params)[1]


def test_computer_tool_is_sent_with_a_computer_tool_to_run_its_calls() -> None:
    computer = ComputerTool(display_size=(1280, 800))
    # A deferred tool of the kind can't run the calls, so the one that isn't deferred does.
    deferred = ToolDefinition(name='deferred_computer', tool_kind=COMPUTER_KIND, defer_loading=True)
    params = _prepare(ModelRequestParameters(function_tools=[deferred, _computer_tool_def()], native_tools=[computer]))
    assert params.native_tools == [computer]
    assert params.tool_name_for_kind(COMPUTER_KIND) == 'my_computer'
    assert params.tool_name_for_kind('computer_use.other') is None


def test_computer_tool_is_sent_once_history_reveals_its_deferred_computer_tool() -> None:
    # A deferred capability's tools stay deferred after the model loads it; history revealing them makes them callable.
    computer = ComputerTool(display_size=(1280, 800), optional=True)
    params = _prepare(
        ModelRequestParameters(
            function_tools=[_computer_tool_def(defer_loading=True)],
            native_tools=[computer],
            revealed_tool_names={'my_computer'},
        )
    )
    assert params.native_tools == [computer]
    assert params.tool_name_for_kind(COMPUTER_KIND) == 'my_computer'


@pytest.mark.parametrize(
    'function_tools',
    [
        [],
        [_computer_tool_def(defer_loading=True)],
        # A fallback for the native tool itself is dropped once the native tool is sent.
        [ToolDefinition(name='my_computer', tool_kind=COMPUTER_KIND, unless_native='computer')],
    ],
    ids=['none', 'deferred', 'fallback'],
)
def test_computer_tool_needs_a_computer_tool_that_is_not_deferred(function_tools: list[ToolDefinition]) -> None:
    optional = ComputerTool(display_size=(1280, 800), optional=True)
    assert _prepare(ModelRequestParameters(function_tools=function_tools, native_tools=[optional])).native_tools == []
    with pytest.raises(
        UserError, match=r"`ComputerTool` needs a tool of each of the kinds \['computer_use.computer'\]"
    ):
        _prepare(
            ModelRequestParameters(function_tools=function_tools, native_tools=[ComputerTool(display_size=(1, 1))])
        )


def test_computer_tool_refuses_two_computer_tools_that_could_run_its_calls() -> None:
    other = ToolDefinition(name='other_computer', tool_kind=COMPUTER_KIND)
    with pytest.raises(UserError, match=r'needs one tool of each of the kinds .* more than one the model can call'):
        _prepare(
            ModelRequestParameters(
                function_tools=[_computer_tool_def(), other],
                native_tools=[ComputerTool(display_size=(1, 1), optional=True)],
            )
        )


def test_computer_tool_does_not_count_a_computer_tool_another_native_tool_replaces() -> None:
    # `MemoryTool` is sent, so the tool standing in for it is dropped and can't run the computer's calls.
    replaced = ToolDefinition(name='memory_computer', tool_kind=COMPUTER_KIND, unless_native='memory')
    natives: list[AbstractNativeTool] = [ComputerTool(display_size=(1, 1)), MemoryTool()]
    params = _prepare(ModelRequestParameters(function_tools=[replaced, _computer_tool_def()], native_tools=natives))
    assert params.tool_name_for_kind(COMPUTER_KIND) == 'my_computer'
    with pytest.raises(UserError, match=r'needs a tool of each of the kinds .* has none the model can call'):
        _prepare(ModelRequestParameters(function_tools=[replaced], native_tools=natives))


def test_computer_tool_counts_a_computer_tool_kept_because_an_empty_tool_search_is_dropped() -> None:
    # The optional tool search has nothing to search, so it isn't sent and the tool standing in for it stays.
    stand_in = ToolDefinition(name='search_computer', tool_kind=COMPUTER_KIND, unless_native='tool_search')
    natives: list[AbstractNativeTool] = [ToolSearchTool(optional=True), ComputerTool(display_size=(1, 1))]
    params = _prepare(ModelRequestParameters(function_tools=[stand_in], native_tools=natives))
    assert params.tool_name_for_kind(COMPUTER_KIND) == 'search_computer'
    with pytest.raises(UserError, match=r'needs one tool of each of the kinds .* more than one the model can call'):
        _prepare(ModelRequestParameters(function_tools=[stand_in, _computer_tool_def()], native_tools=natives))


async def test_a_grid_model_is_told_its_grid_in_every_computer_tool() -> None:
    descriptions: list[list[str | None]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        descriptions.append([tool.description for tool in info.function_tools])
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('ok')])
        return ModelResponse(parts=[ToolCallPart(name, {'actions': []}) for name in ('left', 'right')])

    tools = [Tool(lambda actions: ['Done.', SCREEN], name=name, tool_kind=COMPUTER_KIND) for name in ('left', 'right')]
    agent = Agent(FunctionModel(respond, profile=GRID_PROFILE), tools=tools)
    messages = (await agent.run('Click.')).all_messages()
    assert descriptions == [[GRID_SENTENCE, GRID_SENTENCE]] * 2
    assert [call.coordinate_grid for call in _computer_calls(messages)] == [1000, 1000]


@pytest.mark.parametrize('size', [(0, 800), (1280, -1)])
def test_computer_tool_refuses_a_display_size_that_is_not_positive(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match='must be two positive numbers'):
        ComputerTool(display_size=size)


@pytest.mark.parametrize('hook_asks', [pytest.param(False, id='alone'), pytest.param(True, id='a-hook-asks-too')])
async def test_a_computer_call_with_safety_checks_waits_for_approval_whichever_tool_runs_it(hook_asks: bool) -> None:
    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, Tool
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.exceptions import ApprovalRequired

    # A hook asking for approval as well leads to the same single approval.
    hooks = Hooks[Any]()

    @hooks.on.after_tool_validate
    async def ask(
        ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return args

    ran: list[object] = []

    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
        ran.append(actions)
        return ['Done.', SCREEN]

    checks = [{'id': 'sc_1', 'code': 'malicious_instructions', 'message': 'Check it.'}]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('ok')])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'computer',
                    {'actions': [{'type': 'click', 'x': 1, 'y': 2}]},
                    tool_call_id='call_1',
                    provider_details={'safety_checks': checks},
                )
            ]
        )

    # A plain computer-kind tool with no approval configured: the provider's checks still gate it.
    agent = Agent(
        FunctionModel(respond),
        tools=[Tool(computer, tool_kind=COMPUTER_KIND)],
        capabilities=[hooks] if hook_asks else [],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run('Click.')
    assert isinstance(paused.output, DeferredToolRequests)
    assert [call.tool_call_id for call in paused.output.approvals] == ['call_1']
    assert ran == []

    resumed = await agent.run(
        message_history=paused.all_messages(), deferred_tool_results=DeferredToolResults(approvals={'call_1': True})
    )
    assert resumed.output == 'ok'
    assert ran == [[{'type': 'click', 'x': 1, 'y': 2}]]


@pytest.mark.parametrize('route', ['skipped', 'recovered'])
async def test_a_computer_call_with_safety_checks_waits_for_approval_however_its_arguments_are_accepted(
    route: str,
) -> None:
    from dataclasses import dataclass
    from typing import Any

    from pydantic_ai import Agent, DeferredToolRequests, Tool
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.exceptions import SkipToolValidation

    click = {'actions': [{'type': 'click', 'x': 1, 'y': 2}]}
    ran: list[object] = []

    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
        ran.append(actions)  # pragma: no cover
        return ['Done.', SCREEN]  # pragma: no cover

    @dataclass
    class AcceptArgs(AbstractCapability[Any]):
        async def before_tool_validate(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            if route == 'skipped':
                raise SkipToolValidation(click)
            return args

        async def on_tool_validate_error(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            return click

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = {'wrong': 1} if route == 'recovered' else click
        checks = [{'id': 'sc_1', 'code': 'malicious_instructions', 'message': 'Check it.'}]
        return ModelResponse(
            parts=[ToolCallPart('computer', args, tool_call_id='call_1', provider_details={'safety_checks': checks})]
        )

    agent = Agent(
        FunctionModel(respond),
        tools=[Tool(computer, tool_kind=COMPUTER_KIND)],
        capabilities=[AcceptArgs()],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run('Click.')
    assert isinstance(paused.output, DeferredToolRequests)
    assert [call.tool_call_id for call in paused.output.approvals] == ['call_1']
    assert ran == []


async def test_a_computer_call_with_safety_checks_waits_for_approval_whatever_kind_history_stored() -> None:
    from pydantic_ai import Agent, DeferredToolRequests, Tool

    ran: list[object] = []

    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
        ran.append(actions)  # pragma: no cover
        return ['Done.', SCREEN]  # pragma: no cover

    # The stored kind names some other tool, but the call still runs the computer tool.
    call = ToolCallPart(
        'computer',
        {'actions': [{'type': 'click', 'x': 1, 'y': 2}]},
        tool_call_id='call_1',
        tool_kind='other',
        provider_details={'safety_checks': [{'id': 'sc_1', 'message': 'Check it.'}]},
    )
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('ok')])),
        tools=[Tool(computer, tool_kind=COMPUTER_KIND)],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run(message_history=[ModelResponse(parts=[call])])
    assert isinstance(paused.output, DeferredToolRequests)
    assert [call.tool_call_id for call in paused.output.approvals] == ['call_1']
    assert ran == []


@pytest.mark.parametrize('hook_asks', [pytest.param(False, id='alone'), pytest.param(True, id='a-hook-asks-too')])
async def test_a_computer_call_with_safety_checks_refuses_an_external_executor(hook_asks: bool) -> None:
    from pydantic_ai import Agent, DeferredToolRequests
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.exceptions import ApprovalRequired
    from pydantic_ai.toolsets import ExternalToolset

    # A hook asking for approval as well doesn't let the call through to the external executor.
    hooks = Hooks[Any]()

    @hooks.on.after_tool_validate
    async def ask(
        ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        raise ApprovalRequired

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        checks = [{'id': 'sc_1', 'code': 'malicious_instructions', 'message': 'Check it.'}]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'computer',
                    {'actions': [{'type': 'click', 'x': 1, 'y': 2}]},
                    tool_call_id='call_1',
                    provider_details={'safety_checks': checks},
                )
            ]
        )

    external = ExternalToolset(
        [
            ToolDefinition(
                name='computer',
                parameters_json_schema={
                    'type': 'object',
                    'properties': {'actions': {'type': 'array'}},
                    'required': ['actions'],
                },
                tool_kind=COMPUTER_KIND.id,
            )
        ]
    )
    agent = Agent(
        FunctionModel(respond),
        toolsets=[external],
        capabilities=[hooks] if hook_asks else [],
        output_type=[str, DeferredToolRequests],
    )
    # With a hook asking, core refuses first: no external call can wait for approval in the run.
    message = 'so the run cannot approve it' if hook_asks else 'approval cannot reach'
    with pytest.raises(UserError, match=message):
        await agent.run('Click.')


@pytest.mark.parametrize(
    'unfinished',
    [
        pytest.param({'finish_reason': 'length'}, id='cut-off'),
        pytest.param({'finish_reason': 'content_filter'}, id='refused'),
        pytest.param({'finish_reason': 'error'}, id='failed'),
        pytest.param({'state': 'incomplete'}, id='stream-ended-early'),
    ],
)
async def test_a_computer_call_in_a_response_that_did_not_finish_normally_does_nothing(
    unfinished: dict[str, Any],
) -> None:
    ran: list[object] = []

    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:  # pragma: no cover
        ran.append(actions)
        return ['Done.', SCREEN]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('ok')])
        return ModelResponse(
            parts=[ToolCallPart('computer', {'actions': [{'type': 'click', 'x': 1, 'y': 2}]})], **unfinished
        )

    agent = Agent(FunctionModel(respond), tools=[Tool(computer, tool_kind=COMPUTER_KIND)])
    result = await agent.run('Click.')
    assert result.output == 'ok'
    assert ran == []
    [retry] = result.all_messages()[2].parts
    assert isinstance(retry, ToolReturnPart) and retry.outcome == 'retried'
    assert retry.content == 'The response this call came in did not finish normally, so the computer did nothing.'


GRID_PROFILE = ModelProfile(computer_use_coordinate_grid=1000)
GRID_SENTENCE = (
    'Positions are on a 0-999 grid over the latest screenshot: (0, 0) is the top left, (999, 999) the bottom right.'
)


def _grid_agent(model: Model) -> Agent[None, str]:
    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
        """Act on the screen."""
        return ['Done.', SCREEN]

    return Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)])


def _click_then_stop(descriptions: list[str | None]):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        descriptions.append(info.function_tools[0].description)
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('ok')])
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'click', 'x': 500, 'y': 500}]})])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        descriptions.append(info.function_tools[0].description)
        if len(messages) > 1:
            yield 'ok'
        else:
            yield {0: DeltaToolCall(name='computer', json_args='{"actions": [{"type": "click", "x": 500, "y": 500}]}')}

    return respond, stream


def _computer_calls(messages: list[ModelMessage]) -> list[ComputerCallPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ComputerCallPart)
    ]


@pytest.mark.parametrize('streamed', [False, True])
async def test_a_grid_model_is_told_its_grid_and_its_calls_record_it(streamed: bool) -> None:
    descriptions: list[str | None] = []
    respond, stream = _click_then_stop(descriptions)
    agent = _grid_agent(FunctionModel(respond, stream_function=stream, profile=GRID_PROFILE))
    if streamed:
        async with agent.run_stream('Click.') as result:
            await result.get_output()
        messages = result.all_messages()
    else:
        messages = (await agent.run('Click.')).all_messages()
    assert descriptions == [f'Act on the screen. {GRID_SENTENCE}'] * 2
    [call] = _computer_calls(messages)
    assert call.coordinate_grid == 1000
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert [part.coordinate_grid for part in _computer_calls(restored)] == [1000]


async def test_a_computer_call_a_stream_ended_on_records_its_grid_for_the_run_that_resumes_it() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        yield {0: DeltaToolCall(name='computer', json_args='{"actions": [{"type": "click", "x": 500, "y": 500}]}')}
        raise RuntimeError('Connection dropped.')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('ok')])

    agent = _grid_agent(FunctionModel(respond, stream_function=stream, profile=GRID_PROFILE))
    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='Connection dropped'):
            async with agent.run_stream('Click.'):
                pass  # pragma: no cover
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    # A run resuming the interrupted response executes its call, so the call must say it points on the grid.
    [call] = _computer_calls(history)
    assert call.coordinate_grid == 1000
    assert (await agent.run(message_history=history)).output == 'ok'


@pytest.mark.parametrize(
    'grids',
    [pytest.param([None], id='pixels'), pytest.param([1000, None], id='mixed-fallback')],
)
async def test_a_model_not_known_to_point_on_a_grid_is_left_alone(grids: list[int | None]) -> None:
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    models = [
        FunctionModel(respond, profile=ModelProfile(computer_use_coordinate_grid=grid) if grid else None)
        for grid in grids
    ]
    agent = _grid_agent(models[0] if len(models) == 1 else FallbackModel(*models))
    messages = (await agent.run('Click.')).all_messages()
    assert descriptions == ['Act on the screen.'] * 2
    [call] = _computer_calls(messages)
    assert call.provider_details is None


async def test_a_wrapped_fallback_of_grid_models_is_told_its_grid() -> None:
    # A wrapper around a `FallbackModel` has no profile of its own, so the grid comes from the models it wraps.
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    model = WrapperModel(FallbackModel(FunctionModel(respond, profile=GRID_PROFILE)))
    messages = (await _grid_agent(model).run('Click.')).all_messages()
    assert descriptions == [f'Act on the screen. {GRID_SENTENCE}'] * 2
    [call] = _computer_calls(messages)
    assert call.coordinate_grid == 1000


def _limited(max_side: int | None, max_pixels: int | None) -> ModelProfile:
    profile = ModelProfile()
    if max_side:
        profile['computer_use_screenshot_max_side'] = max_side
    if max_pixels:
        profile['computer_use_screenshot_max_pixels'] = max_pixels
    return profile


@pytest.mark.parametrize(
    'limits, side, pixels',
    [
        pytest.param([(1568, 1_150_000)], 1568, 1_150_000, id='one-model'),
        pytest.param([(2576, 3_600_000), (1568, None), (None, None)], 1568, 3_600_000, id='smallest-in-a-fallback'),
    ],
)
async def test_a_computer_call_records_the_screenshots_its_model_takes(
    limits: list[tuple[int | None, int | None]], side: int, pixels: int
) -> None:
    respond, _ = _click_then_stop([])
    models = [FunctionModel(respond, profile=_limited(max_side, max_pixels)) for max_side, max_pixels in limits]
    agent = _grid_agent(models[0] if len(models) == 1 else FallbackModel(*models))
    [call] = _computer_calls((await agent.run('Click.')).all_messages())
    assert (call.screenshot_max_side, call.screenshot_max_pixels, call.coordinate_grid) == (side, pixels, None)


async def test_a_grid_model_with_a_tool_without_a_description_gets_the_grid_alone() -> None:
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    agent = Agent(
        FunctionModel(respond, profile=GRID_PROFILE),
        tools=[Tool(lambda actions: ['Done.', SCREEN], name='computer', tool_kind=COMPUTER_KIND)],
    )
    await agent.run('Click.')
    assert descriptions == [GRID_SENTENCE] * 2


@pytest.mark.parametrize('value', [True, 0, '1000', None])
def test_a_computer_call_reads_only_real_labels(value: object) -> None:
    keys = ('coordinate_grid', 'screenshot_max_side', 'screenshot_max_pixels')
    call = ComputerCallPart('computer', {'actions': []}, provider_details=dict.fromkeys(keys, value))
    assert (call.coordinate_grid, call.screenshot_max_side, call.screenshot_max_pixels) == (None, None, None)
    # A grid of one point says nothing about where.
    assert (
        ComputerCallPart('computer', {'actions': []}, provider_details={'coordinate_grid': 1}).coordinate_grid is None
    )


@pytest.mark.parametrize('seconds', [float('inf'), float('nan')])
def test_a_duration_must_be_finite(seconds: float) -> None:
    # JSON can't hold it, so a saved history would lose the action.
    with pytest.raises(ValidationError, match='finite number'):
        Wait(seconds=seconds)
    with pytest.raises(ValidationError, match='finite number'):
        HoldKeys(keys=('shift',), seconds=seconds)
