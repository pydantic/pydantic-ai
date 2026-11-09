"""The `computer` tool's typed parts, and how a request sends `ComputerTool` in place of that tool."""

from __future__ import annotations

import pytest

from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    ComputerCallPart,
    ComputerReturnPart,
    SafetyCheck,
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
from pydantic_ai.native_tools import ComputerTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import Click

SCREEN = BinaryImage(data=b'png', media_type='image/png')


def test_a_computer_call_is_promoted_and_reads_its_actions() -> None:
    response = ModelResponse(
        parts=[
            ToolCallPart(
                'my_computer',
                {'actions': [{'type': 'click', 'x': 1, 'y': 2}]},
                tool_kind=COMPUTER_KIND.id,
                provider_name='openai',
                provider_details={'safety_checks': [{'id': 'sc_1', 'code': 'malicious', 'message': 'Check it.'}]},
            )
        ]
    )
    call = response.parts[0]
    assert isinstance(call, ComputerCallPart)
    assert call.actions == [Click(x=1, y=2)]
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
    model = FunctionModel(lambda m, i: None)  # pyright: ignore[reportArgumentType]  # supports every native tool
    return model.prepare_request(None, params)[1]


def test_computer_tool_is_sent_with_a_computer_tool_to_run_its_calls() -> None:
    computer = ComputerTool(display_size=(1280, 800))
    # A deferred tool of the kind can't run the calls, so the one that isn't deferred does.
    deferred = ToolDefinition(name='deferred_computer', tool_kind=COMPUTER_KIND, defer_loading=True)
    params = _prepare(ModelRequestParameters(function_tools=[deferred, _computer_tool_def()], native_tools=[computer]))
    assert params.native_tools == [computer]
    assert params.tool_name_for_kind(COMPUTER_KIND) == 'my_computer'
    assert params.tool_name_for_kind('computer_use.other') is None


@pytest.mark.parametrize('function_tools', [[], [_computer_tool_def(defer_loading=True)]], ids=['none', 'deferred'])
def test_computer_tool_needs_a_computer_tool_that_is_not_deferred(function_tools: list[ToolDefinition]) -> None:
    optional = ComputerTool(display_size=(1280, 800), optional=True)
    assert _prepare(ModelRequestParameters(function_tools=function_tools, native_tools=[optional])).native_tools == []
    with pytest.raises(
        UserError, match=r"`ComputerTool` needs a tool of each of the kinds \['computer_use.computer'\]"
    ):
        _prepare(
            ModelRequestParameters(function_tools=function_tools, native_tools=[ComputerTool(display_size=(1, 1))])
        )


def test_screenshot_is_a_computer_action() -> None:
    assert ComputerCallPart('computer', {'actions': [{'type': 'screenshot'}]}).actions == [Screenshot()]
