"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    SafetyCheck,
    Screenshot,
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
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.native_tools import ComputerTool
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
    with pytest.raises(UserError, match=r'needs one tool of each of the kinds .* more than one that is not deferred'):
        _prepare(
            ModelRequestParameters(
                function_tools=[_computer_tool_def(), other],
                native_tools=[ComputerTool(display_size=(1, 1), optional=True)],
            )
        )


@pytest.mark.parametrize('size', [(0, 800), (1280, -1)])
def test_computer_tool_refuses_a_display_size_that_is_not_positive(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match='must be two positive numbers'):
        ComputerTool(display_size=size)


async def test_a_computer_call_with_safety_checks_waits_for_approval_whichever_tool_runs_it() -> None:
    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, Tool

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

    from pydantic_ai import Agent, DeferredToolRequests, RunContext, Tool
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


async def test_a_computer_call_with_safety_checks_refuses_an_external_executor() -> None:
    from pydantic_ai import Agent, DeferredToolRequests
    from pydantic_ai.toolsets import ExternalToolset

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
    agent = Agent(FunctionModel(respond), toolsets=[external], output_type=[str, DeferredToolRequests])
    with pytest.raises(UserError, match='approval cannot reach'):
        await agent.run('Click.')


GRID_PROFILE = ModelProfile(computer_use_coordinate_grid=1000)
GRID_SENTENCE = (
    'Positions are on a 0-999 grid over the latest screenshot: (0, 0) is the top left, (999, 999) the bottom right.'
)


def _grid_agent(model: FunctionModel | FallbackModel) -> Agent[None, str]:
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
