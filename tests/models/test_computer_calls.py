"""Tests for the helpers the model adapters' computer tools share."""

from __future__ import annotations

import pytest

from pydantic_ai.computer_use import COMPUTER_KIND
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.models._computer_calls import UNFINISHED_COMPUTER_ARGS, block_computer_calls, x11_key


@pytest.mark.parametrize(
    'name, key',
    [('Return', 'enter'), ('Control_L', 'ctrl'), ('KP_5', '5'), ('Page_Down', 'pagedown'), ('Escape', 'escape')],
)
def test_x11_key_names_become_the_computer_tools_names(name: str, key: str) -> None:
    # Claude and Gemini name keys as X11 does. A name the tool spells the same way is only lower-cased.
    assert x11_key(name) == key


def test_blocked_computer_calls_keep_their_place_and_other_parts_are_left_alone() -> None:
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='computer', tool_kind=COMPUTER_KIND.id), ToolDefinition(name='search')]
    )
    text = TextPart('Clicking the button.')
    click = ToolCallPart('computer', {'actions': [{'type': 'click', 'x': 1, 'y': 2}]}, tool_call_id='call_1')
    search = ToolCallPart('search', {'query': 'docs'}, tool_call_id='call_2')

    assert block_computer_calls([text, click, search], params) == [
        text,
        ToolCallPart('computer', UNFINISHED_COMPUTER_ARGS, tool_call_id='call_1'),
        search,
    ]
