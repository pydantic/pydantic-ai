"""Round-trip fidelity for extra keys on typed tool-kind parts.

`ModelMessagesTypeAdapter` is the documented boundary for persisting and reloading
message history, so unknown keys in the `args`/`content` of typed tool-kind parts
must survive a dump/validate round trip instead of being silently stripped.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, FunctionToolset, ModelMessagesTypeAdapter, ToolCallPart
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart
from pydantic_ai._tool_search import (
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.mark.parametrize(
    ('build_msgs', 'part_index'),
    [
        # 1. NativeToolSearchCallPart.args extras
        (
            lambda: [
                ModelRequest(parts=[]),
                ModelResponse(
                    parts=[
                        NativeToolSearchCallPart(
                            tool_name='tool_search',
                            tool_call_id='c1',
                            args={'queries': ['x'], 'extra': 'KEEP_ME'},
                        )
                    ]
                ),
            ],
            (1, 0),
        ),
        # 2. NativeToolSearchReturnPart.content extras
        (
            lambda: [
                ModelResponse(
                    parts=[
                        NativeToolSearchReturnPart(
                            tool_name='tool_search',
                            tool_call_id='c2',
                            content={'discovered_tools': [{'name': 't1'}], 'extra': 'KEEP_ME'},
                        )
                    ]
                ),
            ],
            (0, 0),
        ),
        # 3. ToolSearchCallPart.args extras
        (
            lambda: [
                ModelResponse(
                    parts=[
                        ToolSearchCallPart(
                            tool_name='search_tools',
                            tool_call_id='c3',
                            args={'queries': ['x'], 'extra': 'KEEP_ME'},
                        )
                    ]
                ),
            ],
            (0, 0),
        ),
        # 4. ToolSearchReturnPart.content extras
        (
            lambda: [
                ModelRequest(
                    parts=[
                        ToolSearchReturnPart(
                            tool_name='search_tools',
                            tool_call_id='c4',
                            content={'discovered_tools': [{'name': 't1'}], 'extra': 'KEEP_ME'},
                        )
                    ]
                ),
            ],
            (0, 0),
        ),
        # 5. LoadCapabilityCallPart.args extras (return side is #7805)
        (
            lambda: [
                ModelRequest(parts=[]),
                ModelResponse(
                    parts=[
                        LoadCapabilityCallPart(
                            tool_name='load_capability',
                            tool_call_id='L1',
                            args={'id': 'cap1', 'extra': 'KEEP_ME'},
                        )
                    ]
                ),
            ],
            (1, 0),
        ),
    ],
    ids=[
        'native_tool_search_call_args',
        'native_tool_search_return_content',
        'tool_search_call_args',
        'tool_search_return_content',
        'load_capability_call_args',
    ],
)
def test_typed_tool_kind_round_trip_preserves_extras(build_msgs, part_index):
    """Extra `args`/`content` keys survive a `ModelMessagesTypeAdapter` round trip (#7929)."""
    msgs = build_msgs()
    back = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(msgs))
    msg_idx, part_idx = part_index
    original_part = msgs[msg_idx].parts[part_idx]
    round_tripped_part = back[msg_idx].parts[part_idx]
    assert isinstance(round_tripped_part, type(original_part))
    assert round_tripped_part.tool_kind == original_part.tool_kind
    if hasattr(original_part, 'args') and isinstance(original_part.args, dict):
        assert round_tripped_part.args == original_part.args, (
            f'args extras lost: in={original_part.args!r} out={round_tripped_part.args!r}'
        )
    if hasattr(original_part, 'content') and isinstance(original_part.content, dict):
        assert round_tripped_part.content == original_part.content, (
            f'content extras lost: in={original_part.content!r} out={round_tripped_part.content!r}'
        )


@pytest.mark.anyio
async def test_agent_emitted_extra_args_survive_live_history_and_round_trip() -> None:
    """A model-emitted extra args key survives emission-time promotion and a round trip.

    The same TypeAdapters that strip at the history boundary also validate
    model-emitted tool-search args when `ToolCallPart.narrow_type` promotes the call,
    so the extra key must already be present in `all_messages()` before any dump.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def get_payment(amount: int) -> int:
        return amount * 12

    async def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not [m for m in messages if m.kind == 'response']:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_tools',
                        args={'queries': ['mortgage'], 'extra': 'KEEP_ME'},
                    )
                ]
            )
        else:
            return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_function), toolsets=[toolset], capabilities=[ToolSearch()])
    result = await agent.run('find a mortgage tool')

    live_call_parts = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == 'search_tools'
    ]
    assert live_call_parts, 'expected the emitted search_tools call to be in the live history'
    live_part = live_call_parts[0]
    assert isinstance(live_part, ToolSearchCallPart)
    assert live_part.args == {'queries': ['mortgage'], 'extra': 'KEEP_ME'}

    round_tripped = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(result.all_messages()))
    back_call_parts = [
        part
        for message in round_tripped
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == 'search_tools'
    ]
    assert back_call_parts, 'expected the search_tools call to survive the round trip'
    assert isinstance(back_call_parts[0], ToolSearchCallPart)
    assert back_call_parts[0].args == {'queries': ['mortgage'], 'extra': 'KEEP_ME'}
