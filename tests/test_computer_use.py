"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool.

Not VCR tests: the parts, the request preparation and the labels are all built before or after the provider request,
which `FunctionModel` reaches, so a recording would add nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from pydantic_ai import Agent, Tool, capture_run_messages
from pydantic_ai._computer_use import UnservableRequestError
from pydantic_ai.capabilities import BatchModelRequests, ModelFallback
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    Screenshot,
)
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UserError
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

from .batch_utils import FAST_BATCH_SETTINGS, SlowBatchTestModel

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
    # A deferred tool becomes callable once history reveals it.
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


async def _run(agent: Agent[None, str], *, streamed: bool) -> list[ModelMessage]:
    if streamed:
        async with agent.run_stream('Click.') as result:
            await result.get_output()
        return result.all_messages()
    return (await agent.run('Click.')).all_messages()


def _direct(model: Model) -> Model:
    return model


# A model that hands over another model's response, like `FallbackModel` or `WrapperModel`, keeps that model's labels.
WRAPS = [
    pytest.param(_direct, id='direct'),
    pytest.param(FallbackModel, id='fallback'),
    pytest.param(WrapperModel, id='wrapper'),
]


@pytest.mark.parametrize('wrap', WRAPS)
@pytest.mark.parametrize('streamed', [False, True])
async def test_a_grid_model_is_told_its_grid_and_its_calls_record_it(
    streamed: bool, wrap: Callable[[Model], Model]
) -> None:
    descriptions: list[str | None] = []
    respond, stream = _click_then_stop(descriptions)
    agent = _grid_agent(wrap(FunctionModel(respond, stream_function=stream, profile=GRID_PROFILE)))
    messages = await _run(agent, streamed=streamed)
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


@pytest.mark.parametrize('wrapped', [pytest.param(False, id='fallback'), pytest.param(True, id='wrapped-fallback')])
@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize(
    'grids', [pytest.param((1000, None), id='grid-down'), pytest.param((None, 1000), id='pixels-down')]
)
async def test_a_fallback_call_has_the_grid_of_the_model_that_made_it(
    grids: tuple[int | None, int | None], streamed: bool, wrapped: bool
) -> None:
    down_saw: list[str | None] = []
    up_saw: list[str | None] = []

    def down(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        down_saw.append(info.function_tools[0].description)
        raise ModelAPIError('down', 'Unavailable.')

    async def down_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        down(messages, info)
        yield 'unreachable'  # pragma: no cover

    respond, stream = _click_then_stop(up_saw)
    down_grid, up_grid = grids
    model: Model = FallbackModel(
        FunctionModel(
            down,
            stream_function=down_stream,
            profile=ModelProfile(computer_use_coordinate_grid=down_grid) if down_grid else None,
        ),
        FunctionModel(
            respond,
            stream_function=stream,
            profile=ModelProfile(computer_use_coordinate_grid=up_grid) if up_grid else None,
        ),
    )
    messages = await _run(_grid_agent(WrapperModel(model) if wrapped else model), streamed=streamed)
    told = {None: 'Act on the screen.', 1000: f'Act on the screen. {GRID_SENTENCE}'}
    assert (down_saw, up_saw) == ([told[down_grid]] * 2, [told[up_grid]] * 2)
    [call] = _computer_calls(messages)
    assert call.coordinate_grid == up_grid


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


def _look_at_a_big_screen(answered: list[str], name: str, max_side: int) -> FunctionModel:
    """A model that can see screenshots up to `max_side` long, which clicks once on a 1280x800 screen, then stops."""
    respond, stream = _click_then_stop([])

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered.append(name)
        return respond(messages, info)

    async def answer_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        answered.append(name)
        async for chunk in stream(messages, info):
            yield chunk

    return FunctionModel(answer, stream_function=answer_stream, model_name=name, profile=_limited(max_side, None))


def _big_screen(actions: list[dict[str, object]]) -> list[str | BinaryImage]:
    """Act on the screen."""
    return ['Done.', _screenshot(1280, 800)]


def _fallback_agent(
    fallback: str, models: list[FunctionModel], fallback_on: tuple[type[Exception], ...] = (ValueError,)
) -> Agent[None, str]:
    # A refusal is passed over whatever `fallback_on` says, so by default it's set to an error no model raises.
    if fallback == 'FallbackModel':
        return Agent(
            FallbackModel(*models, fallback_on=fallback_on),
            tools=[Tool(_big_screen, name='computer', tool_kind=COMPUTER_KIND)],
        )
    return Agent(
        models[0],
        tools=[Tool(_big_screen, name='computer', tool_kind=COMPUTER_KIND)],
        capabilities=[ModelFallback(*models[1:], fallback_on=fallback_on)],
    )


FALLBACKS = pytest.mark.parametrize('fallback', ['FallbackModel', 'ModelFallback'])


@pytest.mark.parametrize('streamed', [False, True])
@FALLBACKS
@pytest.mark.parametrize(
    'limits, answered',
    [
        pytest.param([1000, 2000], ['small', 'big'], id='small-first'),
        pytest.param([2000, 1000], ['big', 'big'], id='big-first'),
    ],
)
async def test_a_fallback_moves_on_from_a_model_that_cant_see_the_screenshot(
    limits: list[int], answered: list[str], fallback: str, streamed: bool
) -> None:
    # The first request has no screenshot yet, so `small` only refuses the second, after the click.
    calls: list[str] = []
    names = {1000: 'small', 2000: 'big'}
    agent = _fallback_agent(fallback, [_look_at_a_big_screen(calls, names[limit], limit) for limit in limits])
    messages = await _run(agent, streamed=streamed)
    assert messages[-1].parts == [TextPart('ok')]
    assert calls == answered


@pytest.mark.parametrize('streamed', [False, True])
@FALLBACKS
async def test_a_fallback_whose_models_all_cant_see_the_screenshot_raises_the_last_refusal(
    fallback: str, streamed: bool
) -> None:
    calls: list[str] = []
    agent = _fallback_agent(
        fallback, [_look_at_a_big_screen(calls, 'first', 1000), _look_at_a_big_screen(calls, 'second', 1200)]
    )
    with pytest.raises(UnservableRequestError, match=r"larger than 'second' can see\. Start the display at 1200x750"):
        await _run(agent, streamed=streamed)
    # `first` made the click. Both then refused the next request without being called.
    assert calls == ['first']


@FALLBACKS
@pytest.mark.parametrize('answered', [True, False], ids=['a-later-model-answers', 'no-model-answers'])
async def test_a_fallback_moves_on_from_a_refusal_after_a_provider_error(fallback: str, answered: bool) -> None:
    # The primary looks, then fails; the first fallback can't see the screenshot, so the second answers, if any.
    calls: list[str] = []
    looked = False

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal looked
        calls.append('primary')
        if looked:
            raise ModelAPIError('primary', 'Unavailable.')
        looked = True
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]})])

    models = [FunctionModel(primary), _look_at_a_big_screen(calls, 'small', 1000)]
    if answered:
        models.append(_look_at_a_big_screen(calls, 'big', 2000))
    agent = _fallback_agent(fallback, models, fallback_on=(ModelAPIError,))
    if answered:
        assert (await agent.run('Look.')).output == 'ok'
        assert calls == ['primary', 'primary', 'big']
        return
    # Not every model refused the request, so the run ends with all their errors.
    with pytest.raises(FallbackExceptionGroup) as error:
        await agent.run('Look.')
    assert [type(failure) for failure in error.value.exceptions] == [ModelAPIError, UnservableRequestError]
    assert calls == ['primary', 'primary']


class _BatchModel(SlowBatchTestModel):
    """A model whose requests go through the batch API, with `profile`."""

    def __init__(self, model_name: str, profile: ModelProfile) -> None:
        super().__init__(model_name=model_name)
        self._profile = profile


async def test_a_batched_call_records_the_grid_of_its_model() -> None:
    agent = Agent(
        _BatchModel('grid', GRID_PROFILE),
        tools=[Tool(_big_screen, name='computer', tool_kind=COMPUTER_KIND)],
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    result = await agent.run('Look.')
    # `TestModel`'s made-up actions aren't valid ones, so the call stays a plain `ToolCallPart`, with the labels.
    calls = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assert [call.provider_details for call in calls] == [{'coordinate_grid': 1000}]


async def test_a_batched_request_the_model_cant_serve_is_refused_before_it_is_submitted() -> None:
    model = _BatchModel('blind', ModelProfile(supports_image_input=False))
    agent = Agent(
        model,
        tools=[Tool(_big_screen, name='computer', tool_kind=COMPUTER_KIND)],
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    with pytest.raises(UnservableRequestError, match="'blind' does not"):
        await agent.run('Look.')
    assert (model.encoded, model.created) == (0, [])


@pytest.mark.parametrize('value', [True, 0, '1000'])
def test_a_computer_call_reads_only_real_labels(value: object) -> None:
    call = ComputerCallPart('computer', {'actions': []}, provider_details={'coordinate_grid': value})
    assert call.coordinate_grid is None
    # A grid of one point can't say where anything is, so it reads as no grid.
    assert (
        ComputerCallPart('computer', {'actions': []}, provider_details={'coordinate_grid': 1}).coordinate_grid is None
    )
