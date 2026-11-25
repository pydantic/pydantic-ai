"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool.

Not VCR tests: the parts and the request preparation are all built before or after the provider request,
which `FunctionModel` reaches, so a recording would add nothing.
"""

from __future__ import annotations

import pytest

from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    Screenshot,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.native_tools import AbstractNativeTool, ComputerTool, MemoryTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool
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


def test_a_computer_call_reads_the_confirmation_its_provider_asked_for() -> None:
    flagged = ComputerCallPart('computer', {'actions': []}, provider_details={'confirmation': 'It sends a message.'})
    assert flagged.confirmation == 'It sends a message.'
    assert ComputerCallPart('computer', {'actions': []}).confirmation is None
    assert ComputerCallPart('computer', {'actions': []}, provider_details={'confirmation': True}).confirmation is None


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
