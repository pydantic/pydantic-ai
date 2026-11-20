"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool.

These are not recorded-HTTP (VCR) tests. The parts, the request preparation and the labels the agent adds are all
built before or after the provider request. `FunctionModel` reaches all of them, so a recording would add nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, Tool, capture_run_messages
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    Screenshot,
)
from pydantic_ai.exceptions import ModelAPIError, UserError
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
from pydantic_ai.settings import ModelSettings
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
            )
        ]
    )
    call = response.parts[0]
    assert isinstance(call, ComputerCallPart)
    assert call.actions == [Click(x=1, y=2), Screenshot()]
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))[0]
    assert isinstance(restored, ModelResponse) and restored.parts == response.parts


def test_a_computer_call_with_incomplete_arguments_has_no_actions() -> None:
    assert ComputerCallPart('computer', '{"actions": [').actions is None


def test_a_computer_result_reads_its_outcome_and_screenshot() -> None:
    request = ModelRequest(
        parts=[
            ToolReturnPart(
                'my_computer',
                ['Clicked.', SCREEN],
                metadata={'performed': 1},
                tool_kind=COMPUTER_KIND.id,
            )
        ]
    )
    result = request.parts[0]
    assert isinstance(result, ComputerReturnPart)
    assert result.result == {'performed': 1}
    assert result.screenshot == SCREEN


def test_a_computer_result_without_metadata_or_a_final_screenshot() -> None:
    # A result a UI rebuilt has no metadata. A result whose screenshot was dropped from the history ends with text.
    result = ComputerReturnPart('computer', [SCREEN, 'An earlier screenshot was removed.'])
    assert result.result is None
    assert result.screenshot is None


def _computer_tool_def(*, defer_loading: bool = False) -> ToolDefinition:
    return ToolDefinition(name='my_computer', tool_kind=COMPUTER_KIND, defer_loading=defer_loading)


def _prepare(params: ModelRequestParameters) -> ModelRequestParameters:
    model = FunctionModel(lambda m, i: ModelResponse(parts=[]))  # supports every native tool
    return model.prepare_request(None, params)[1]


@pytest.mark.parametrize('display_size', [(0, 800), (1280, -1)])
def test_computer_tool_refuses_a_display_size_that_is_not_positive(display_size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match='display_size` must be positive'):
        ComputerTool(display_size=display_size)


def test_computer_tool_is_sent_with_a_computer_tool_to_run_its_calls() -> None:
    computer = ComputerTool(display_size=(1280, 800))
    # A deferred tool of the kind can't run the calls, so the one that isn't deferred does.
    deferred = ToolDefinition(name='deferred_computer', tool_kind=COMPUTER_KIND, defer_loading=True)
    params = _prepare(ModelRequestParameters(function_tools=[deferred, _computer_tool_def()], native_tools=[computer]))
    assert params.native_tools == [computer]


def test_computer_tool_is_sent_once_history_reveals_its_deferred_computer_tool() -> None:
    # A deferred capability's tools stay deferred after the model loads it. They become callable once history
    # reveals them.
    computer = ComputerTool(display_size=(1280, 800), optional=True)
    params = _prepare(
        ModelRequestParameters(
            function_tools=[_computer_tool_def(defer_loading=True)],
            native_tools=[computer],
            revealed_tool_names={'my_computer'},
        )
    )
    assert params.native_tools == [computer]


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
def test_computer_tool_needs_a_computer_tool_the_model_can_call(function_tools: list[ToolDefinition]) -> None:
    optional = ComputerTool(display_size=(1280, 800), optional=True)
    assert _prepare(ModelRequestParameters(function_tools=function_tools, native_tools=[optional])).native_tools == []
    with pytest.raises(UserError, match='needs a `computer` tool the model can call, and this request has none'):
        _prepare(
            ModelRequestParameters(function_tools=function_tools, native_tools=[ComputerTool(display_size=(1, 1))])
        )


def test_computer_tool_refuses_two_computer_tools_that_could_run_its_calls() -> None:
    other = ToolDefinition(name='other_computer', tool_kind=COMPUTER_KIND)
    with pytest.raises(
        UserError, match='needs a `computer` tool the model can call, and this request has more than one'
    ):
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
    assert params.native_tools == natives
    assert [t.name for t in params.function_tools] == ['my_computer']
    with pytest.raises(UserError, match='needs a `computer` tool the model can call, and this request has none'):
        _prepare(ModelRequestParameters(function_tools=[replaced], native_tools=natives))


def test_computer_tool_counts_a_computer_tool_kept_because_an_empty_tool_search_is_dropped() -> None:
    # The optional tool search has nothing to search, so it isn't sent and the tool standing in for it stays.
    stand_in = ToolDefinition(name='search_computer', tool_kind=COMPUTER_KIND, unless_native='tool_search')
    natives: list[AbstractNativeTool] = [ToolSearchTool(optional=True), ComputerTool(display_size=(1, 1))]
    params = _prepare(ModelRequestParameters(function_tools=[stand_in], native_tools=natives))
    assert params.native_tools == [ComputerTool(display_size=(1, 1))]
    assert [t.name for t in params.function_tools] == ['search_computer']
    with pytest.raises(
        UserError, match='needs a `computer` tool the model can call, and this request has more than one'
    ):
        _prepare(ModelRequestParameters(function_tools=[stand_in, _computer_tool_def()], native_tools=natives))


def _lookup() -> str:
    """Look something up."""
    return 'found'


async def test_a_grid_model_is_told_its_grid_in_every_computer_tool_and_no_other() -> None:
    descriptions: list[list[str | None]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        descriptions.append([tool.description for tool in info.function_tools])
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('ok')])
        calls = [ToolCallPart(name, {'actions': []}) for name in ('left', 'right')]
        return ModelResponse(parts=[*calls, ToolCallPart('_lookup', {})])

    tools = [Tool(lambda actions: ['Done.', SCREEN], name=name, tool_kind=COMPUTER_KIND) for name in ('left', 'right')]
    agent = Agent(FunctionModel(respond, profile=GRID_PROFILE), tools=[*tools, Tool(_lookup)])
    messages = (await agent.run('Click.')).all_messages()
    assert descriptions == [[GRID_SENTENCE, GRID_SENTENCE, 'Look something up.']] * 2
    assert [call.coordinate_grid for call in _computer_calls(messages)] == [1000, 1000]
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.parts[2].provider_details is None


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


def test_a_request_prepared_again_is_told_its_grid_once() -> None:
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[]), profile=GRID_PROFILE)
    _, prepared = model.prepare_request(None, ModelRequestParameters(function_tools=[_computer_tool_def()]))
    _, again = model.prepare_request(None, prepared)
    assert [tool.description for tool in again.function_tools] == [tool.description for tool in prepared.function_tools]
    assert GRID_SENTENCE in (prepared.function_tools[0].description or '')


async def test_a_computer_call_a_stream_ended_on_records_its_grid_for_the_run_that_resumes_it() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        yield {0: DeltaToolCall(name='computer', json_args='{"actions": [{"type": "click", "x": 500, "y": 500}]}')}
        raise RuntimeError('Connection dropped.')

    agent = _grid_agent(FunctionModel(stream_function=stream, profile=GRID_PROFILE))
    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='Connection dropped'):
            async with agent.run_stream('Click.'):
                pass  # pragma: no cover
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    # A run resuming the interrupted response executes its call, so the call must say it points on the grid.
    [call] = _computer_calls(history)
    assert call.coordinate_grid == 1000


async def test_a_model_not_known_to_point_on_a_grid_is_left_alone() -> None:
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    messages = (await _grid_agent(FunctionModel(respond)).run('Click.')).all_messages()
    assert descriptions == ['Act on the screen.'] * 2
    [call] = _computer_calls(messages)
    assert call.provider_details is None


@pytest.mark.parametrize(
    'grids', [pytest.param((1000, None), id='grid-down'), pytest.param((None, 1000), id='pixels-down')]
)
async def test_a_fallback_call_has_the_grid_of_the_model_that_made_it(grids: tuple[int | None, int | None]) -> None:
    # Each model is told its own grid and labels its own calls, so a call made after a fallback points on the grid of
    # the model that made it.
    down_saw: list[str | None] = []
    up_saw: list[str | None] = []

    def down(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        down_saw.append(info.function_tools[0].description)
        raise ModelAPIError('down', 'Unavailable.')

    respond, _ = _click_then_stop(up_saw)
    down_grid, up_grid = grids
    model = FallbackModel(
        FunctionModel(down, profile=ModelProfile(computer_use_coordinate_grid=down_grid) if down_grid else None),
        FunctionModel(respond, profile=ModelProfile(computer_use_coordinate_grid=up_grid) if up_grid else None),
    )
    messages = (await _grid_agent(model).run('Click.')).all_messages()
    told = {None: 'Act on the screen.', 1000: f'Act on the screen. {GRID_SENTENCE}'}
    assert (down_saw, up_saw) == ([told[down_grid]] * 2, [told[up_grid]] * 2)
    [call] = _computer_calls(messages)
    assert call.coordinate_grid == up_grid


async def test_a_wrapped_fallback_of_grid_models_is_told_its_grid() -> None:
    # A wrapper around a `FallbackModel` has no profile of its own, so the grid comes from the models it wraps.
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    model = WrapperModel(FallbackModel(FunctionModel(respond, profile=GRID_PROFILE)))
    messages = (await _grid_agent(model).run('Click.')).all_messages()
    assert descriptions == [f'Act on the screen. {GRID_SENTENCE}'] * 2
    [call] = _computer_calls(messages)
    assert call.coordinate_grid == 1000


async def test_a_wrapper_that_picks_its_own_model_is_not_told_the_grid_of_the_one_it_wraps() -> None:
    # A wrapper with its own `profile` may answer with another model than `wrapped`, as `TemporalModel.using_model()`
    # does, so the grid of the model it wraps may be the wrong one.
    descriptions: list[str | None] = []
    respond, _ = _click_then_stop(descriptions)
    pixels = FunctionModel(respond)

    class PicksAnotherModel(WrapperModel):
        @property
        def profile(self) -> ModelProfile:
            raise NotImplementedError('The model it picks has the profile.')

        async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await pixels.request(messages, model_settings, model_request_parameters)

    model = PicksAnotherModel(FallbackModel(FunctionModel(respond, profile=GRID_PROFILE)))
    messages = (await _grid_agent(model).run('Click.')).all_messages()
    assert descriptions == ['Act on the screen.'] * 2
    [call] = _computer_calls(messages)
    assert call.coordinate_grid is None


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
        pytest.param([(2576, 3_600_000), (1568, None)], 2576, 3_600_000, id='the-model-that-answered-a-fallback'),
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


def _screenshot(width: int, height: int) -> BinaryImage:
    header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
    return BinaryImage(data=header + b'\x08\x06\x00\x00\x00', media_type='image/png')


async def test_a_model_that_cant_see_images_is_refused_the_computer_tool() -> None:
    respond, _ = _click_then_stop([])
    agent = _grid_agent(FunctionModel(respond, model_name='blind', profile=ModelProfile(supports_image_input=False)))
    with pytest.raises(UserError, match="needs a model that accepts images, and 'blind' does not"):
        await agent.run('Click.')


@pytest.mark.parametrize(
    'size, refusal',
    [
        pytest.param((1000, 800), None, id='fits'),
        pytest.param((1280, 800), 'Start the display at 1000x625 or smaller', id='too-long'),
        pytest.param((1000, 1000), 'Start the display at 948x948 or smaller', id='too-many-pixels'),
    ],
)
async def test_a_model_is_refused_a_screenshot_larger_than_it_can_see(
    size: tuple[int, int], refusal: str | None
) -> None:
    respond, _ = _click_then_stop([])
    model = FunctionModel(respond, model_name='small', profile=_limited(1000, 900_000))

    def computer(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
        """Act on the screen."""
        return ['Done.', _screenshot(*size)]

    agent = Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)])
    if refusal is None:
        assert (await agent.run('Click.')).output == 'ok'
    else:
        with pytest.raises(UserError, match=f"larger than 'small' can see. {refusal}"):
            await agent.run('Click.')


@pytest.mark.parametrize('value', [True, 0, '1000'])
def test_a_computer_call_reads_only_real_labels(value: object) -> None:
    keys = ('coordinate_grid', 'screenshot_max_side', 'screenshot_max_pixels')
    call = ComputerCallPart('computer', {'actions': []}, provider_details=dict.fromkeys(keys, value))
    assert (call.coordinate_grid, call.screenshot_max_side, call.screenshot_max_pixels) == (None, None, None)
    # A grid of one point can't say where anything is, so it reads as no grid.
    assert (
        ComputerCallPart('computer', {'actions': []}, provider_details={'coordinate_grid': 1}).coordinate_grid is None
    )
