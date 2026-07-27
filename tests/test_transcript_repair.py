"""Tests for out-of-the-box repair that makes a message history provider-valid.

An interrupted, hand-built, or context-evicted history can have broken tool-call/tool-result
pairing that strict providers reject. Before each model request, `_clean_message_history` runs an
ordered pipeline that ADDs synthesized results for dangling tool calls and REMOVEs fundamentally
unsendable parts (orphaned results), then merges consecutive messages — so
interrupted and hand-built histories can be reused directly. Native/builtin parts are left
untouched. Synthesized returns carry the `pydantic_ai_synthesized_tool_return` metadata marker so
repairs are inspectable in the history.

These tests capture the exact message list the model receives via `FunctionModel` instead of VCR:
the repair happens in the pre-request history cleaning, and cassette matchers aren't reliably
sensitive to request bodies, so a VCR test could pass green without pinning the repaired shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai._agent_graph import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    _clean_message_history,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsSameStr, IsStr, iter_message_parts

pytestmark = pytest.mark.anyio

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def capture_agent() -> tuple[Agent, list[list[ModelMessage]]]:
    """An agent whose model records the exact message history it receives and replies with text."""
    received: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart('All done.')])

    return Agent(FunctionModel(model_function)), received


async def test_dangling_tool_call_gets_synthesized_return():
    """A dangling tool call mid-history gets a synthesized return in the following request."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(parts=[UserPromptPart('Never mind, tell me a joke.', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Here is a joke.')], timestamp=TS),
    ]

    result = await agent.run('Explain?', message_history=message_history)

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Mexico City'}, tool_call_id='call_1')],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Never mind, tell me a joke.', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(parts=[TextPart(content='Here is a joke.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Explain?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # The synthesized return is persisted in the run's message history, not just sent to the model.
    synthesized = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
    ]
    assert len(synthesized) == 1


async def test_partially_answered_parallel_tool_calls():
    """When a run is interrupted during tool execution, the still-unanswered calls are closed out.

    The synthesized returns are inserted after the tool returns that did complete.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Calculate.', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[
                ToolCallPart('get_volume', '{"size": 6}', tool_call_id='call_volume'),
                ToolCallPart('get_mass', '{"size": 6}', tool_call_id='call_mass'),
                ToolCallPart('get_density', tool_call_id='call_density'),
            ],
            timestamp=TS,
        ),
        ModelRequest(
            parts=[ToolReturnPart('get_volume', 216, tool_call_id='call_volume', timestamp=TS)],
            timestamp=TS,
            state='interrupted',
        ),
    ]

    await agent.run(message_history=message_history)

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Calculate.', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_volume', args='{"size": 6}', tool_call_id='call_volume'),
                    ToolCallPart(tool_name='get_mass', args='{"size": 6}', tool_call_id='call_mass'),
                    ToolCallPart(tool_name='get_density', tool_call_id='call_density'),
                ],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_volume', content=216, tool_call_id='call_volume', timestamp=TS),
                    ToolReturnPart(
                        tool_name='get_mass',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_mass',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    ToolReturnPart(
                        tool_name='get_density',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_density',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_incomplete_tool_call_args_synthesized():
    """A dangling tool call whose args were cut off mid-stream is kept and synthesized a return.

    The call is replayed verbatim — serializers degrade malformed args gracefully, and removing it
    would disturb the response's shape (e.g. leave a thinking-only response) — and is closed out
    like its sibling dangling call whose args did stream completely.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[
                TextPart('Let me look that up.'),
                ToolCallPart('get_time', '{"tz": "UTC"}', tool_call_id='call_1'),
                ToolCallPart('get_weather', '{"city": "Mex', tool_call_id='call_2'),
            ],
            timestamp=TS,
        ),
        ModelRequest(parts=[UserPromptPart('Never mind.', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('OK.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[
                    TextPart(content='Let me look that up.'),
                    ToolCallPart(tool_name='get_time', args='{"tz": "UTC"}', tool_call_id='call_1'),
                    ToolCallPart(tool_name='get_weather', args='{"city": "Mex', tool_call_id='call_2'),
                ],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_time',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_2',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Never mind.', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(parts=[TextPart(content='OK.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Thanks.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_response_with_only_incomplete_tool_call_synthesized():
    """A response whose only part is an incomplete-args tool call is kept and synthesized a return."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', '{"city": "Mex', tool_call_id='call_1')], timestamp=TS),
        ModelRequest(parts=[UserPromptPart('Are you there?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Yes.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Mex', tool_call_id='call_1')],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Are you there?', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(parts=[TextPart(content='Yes.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Thanks.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_trailing_incomplete_tool_call_resumed_with_retry():
    """Promptless resume of a history ending on an unparsable-args call keeps the local retry flow.

    The trailing response is the live frontier: instead of dropping the call, resumption executes
    it, local args validation fails, and the model gets a retry prompt alongside its own call.
    """
    agent, received = capture_agent()

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return 'Sunny'  # pragma: no cover

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', '{"city": "Mex', tool_call_id='call_1')], timestamp=TS),
    ]

    result = await agent.run(message_history=message_history)
    assert result.output == 'All done.'

    # The call is preserved and answered with a retry prompt, not dropped or synthesized.
    assert received[0][1:] == snapshot(
        [
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Mex', tool_call_id='call_1')],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'json_invalid',
                                'loc': (),
                                'msg': 'Invalid JSON: EOF while parsing a string at line 1 column 13',
                                'input': '{"city": "Mex',
                            }
                        ],
                        tool_name='get_weather',
                        tool_call_id='call_1',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deferred_run_history_not_silently_repaired():
    """Resuming a deferred run's history without `deferred_tool_results` keeps the pending call open.

    A run that ends in `DeferredToolRequests` persists a response with an unanswered call followed
    by a 'complete' request with the executed returns. That call may still receive its result via
    `deferred_tool_results`, so it must not be closed out at run start; only the copy sent to the
    model is repaired.
    """
    received: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('get_data', {}, tool_call_id='call_data'),
                    ToolCallPart('create_file', {'path': 'x'}, tool_call_id='call_file'),
                ]
            )
        return ModelResponse(parts=[TextPart('All done.')])

    agent = Agent(FunctionModel(model_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def get_data() -> str:
        return 'data'

    @agent.tool_plain(requires_approval=True)
    def create_file(path: str) -> str:
        return 'created'  # pragma: no cover

    result = await agent.run('Do it.')
    assert isinstance(result.output, DeferredToolRequests)
    message_history = result.all_messages()
    trailing_request = message_history[-1]
    assert isinstance(trailing_request, ModelRequest)
    assert trailing_request.state == 'complete'

    result2 = await agent.run('Never mind.', message_history=message_history)
    assert result2.output == 'All done.'

    # The pending call stays open in the persisted history, so it can still be answered or audited...
    assert not any(
        isinstance(part, ToolReturnPart) and part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
        for message in result2.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    # ...while the copy sent to the model got a synthesized return so the provider accepts it.
    assert any(
        isinstance(part, ToolReturnPart) and part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
        for message in received[1]
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_result_before_call_dropped_as_orphan_and_call_synthesized():
    """A tool result that precedes its call is orphaned: it's dropped, and the now-unanswered call
    gets a synthesized return.

    The result can't be reordered across message boundaries into the call's turn, so the
    fundamentally-invalid early result is removed instead. The interior request it emptied is dropped.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Sunny', tool_call_id='call_1', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(parts=[UserPromptPart('And tomorrow?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('No idea.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # The orphaned 'Sunny' result and its emptied request are gone; the call is synthesized a return.
    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Mexico City'}, tool_call_id='call_1')],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='And tomorrow?', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(parts=[TextPart(content='No idea.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Thanks.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_orphaned_tool_result_dropped():
    """A tool result whose call is nowhere in the history is dropped (no provider accepts it)."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('What is the weather?', timestamp=TS),
                ToolReturnPart('get_weather', 'Sunny', tool_call_id='ghost', timestamp=TS),
            ],
            timestamp=TS,
        ),
        ModelResponse(parts=[TextPart('It is sunny.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # The orphaned result is gone; the surrounding user prompt is preserved.
    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(parts=[TextPart(content='It is sunny.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Thanks.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_orphaned_result_emptying_last_request_keeps_placeholder():
    """When dropping an orphaned result empties the last message, an empty request is kept.

    The history must end on a `ModelRequest`, so the emptied trailing request is kept (empty) rather
    than dropped. Asserted directly on `_clean_message_history` — feeding this into a run would add a
    new prompt that fills the placeholder, hiding whether it was retained.
    """
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Hello.')], timestamp=TS),
        ModelRequest(parts=[ToolReturnPart('ghost_tool', 'x', tool_call_id='ghost', timestamp=TS)], timestamp=TS),
    ]

    cleaned = _clean_message_history(message_history)

    # The orphaned result is dropped, but its now-empty trailing request is retained as a placeholder
    # so the history still ends on a `ModelRequest`.
    assert cleaned == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hi', timestamp=TS)], timestamp=TS),
            ModelResponse(parts=[TextPart(content='Hello.')], timestamp=TS),
            ModelRequest(parts=[], timestamp=TS),
        ]
    )


async def test_orphaned_result_emptying_last_request_merges_with_new_prompt():
    """The kept empty placeholder merges cleanly with a new prompt on the next run."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Hello.')], timestamp=TS),
        ModelRequest(parts=[ToolReturnPart('ghost_tool', 'x', tool_call_id='ghost', timestamp=TS)], timestamp=TS),
    ]

    await agent.run('Continue.', message_history=message_history)

    # The orphaned trailing result is dropped; the new prompt fills the placeholder's slot.
    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hi', timestamp=TS)], timestamp=TS),
            ModelResponse(parts=[TextPart(content='Hello.')], timestamp=TS),
            ModelRequest(parts=[UserPromptPart(content='Continue.', timestamp=IsDatetime())], timestamp=IsDatetime()),
        ]
    )


async def test_native_tool_calls_left_untouched():
    """The pipeline never touches native/builtin parts — even a dangling native call is preserved.

    Native calls and their results are co-located in one `ModelResponse` (or the result can arrive
    in a later response, e.g. Anthropic tool search) and are shaped by each model's own serializer,
    which handles dangling/empty-id cases on the wire; xAI's serializer, for one, skips an empty-id
    native call while the persisted history keeps it. So the history pipeline leaves native parts
    exactly as given. Exercised directly against `_clean_message_history` since the behavior under
    test is that a dangling native call is *not* repaired.
    """
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Compute 2+2.', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[
                TextPart('Let me run that.'),
                # A dangling native call (no co-located result) and an empty-id native call: preserved.
                NativeToolCallPart('code_execution', {'code': '2+2'}, tool_call_id='srv_1', provider_name='anthropic'),
                NativeToolCallPart('code_execution', {}, tool_call_id='', provider_name='anthropic'),
            ],
            timestamp=TS,
        ),
        ModelRequest(parts=[UserPromptPart('What happened?', timestamp=TS)], timestamp=TS),
    ]
    original = ModelMessagesTypeAdapter.dump_json(message_history)

    cleaned = _clean_message_history(message_history, repair_last_response=True)

    assert cleaned == message_history
    assert ModelMessagesTypeAdapter.dump_json(cleaned) == original


async def test_orphaned_result_and_dangling_call_in_one_history():
    """A history with both an orphaned result and a dangling call: each is repaired independently."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('Do two things.', timestamp=TS),
                ToolReturnPart('ghost_tool', 'x', tool_call_id='ghost', timestamp=TS),
            ],
            timestamp=TS,
        ),
        ModelResponse(parts=[ToolCallPart('real_tool', {'a': 1}, tool_call_id='real_1')], timestamp=TS),
        ModelRequest(parts=[UserPromptPart('Actually stop.', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Stopped.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Do two things.', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='real_tool', args={'a': 1}, tool_call_id='real_1')], timestamp=TS
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='real_tool',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='real_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Actually stop.', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(parts=[TextPart(content='Stopped.')], timestamp=TS),
            ModelRequest(
                parts=[UserPromptPart(content='Thanks.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_full_pipeline_idempotent_and_deterministic():
    """The whole pipeline (orphan-drop + synthesize + merge) is idempotent.

    A history exercising every repair — plus a co-located native call left untouched — is
    byte-identical when repaired again, so reusing a repaired history across turns never churns the
    provider prompt-cache prefix.
    """

    def build_history() -> list[ModelMessage]:
        return [
            ModelRequest(
                parts=[
                    UserPromptPart('Start.', timestamp=TS),
                    ToolReturnPart('ghost', 'x', tool_call_id='ghost', timestamp=TS),
                ],
                timestamp=TS,
            ),
            ModelResponse(
                parts=[
                    ToolCallPart('real_tool', {'a': 1}, tool_call_id='real_1'),
                    NativeToolCallPart(
                        'code_execution', {'code': 'x'}, tool_call_id='srv_1', provider_name='anthropic'
                    ),
                    NativeToolReturnPart('code_execution', 'y', tool_call_id='srv_1', provider_name='anthropic'),
                ],
                timestamp=TS,
            ),
            ModelRequest(parts=[UserPromptPart('Never mind.', timestamp=TS)], timestamp=TS),
            ModelResponse(parts=[TextPart('OK.')], timestamp=TS),
        ]

    agent_a, _ = capture_agent()
    result_a = await agent_a.run('Explain?', message_history=build_history())

    once = result_a.all_messages()
    agent_b, received_b = capture_agent()
    await agent_b.run('Explain?', message_history=once)

    # Repairing the already-repaired history is a no-op: the wire-sent prefix is byte-identical.
    prefix = len(once)
    assert ModelMessagesTypeAdapter.dump_json(received_b[0][:prefix]) == ModelMessagesTypeAdapter.dump_json(once)


async def test_duplicate_result_ignored():
    """A duplicate result for an already-answered call is an orphan and triggers no repair."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('get_weather', 'Sunny', tool_call_id='call_1', timestamp=TS),
                ToolReturnPart('get_weather', 'Sunny again', tool_call_id='call_1', timestamp=TS),
            ],
            timestamp=TS,
        ),
        ModelResponse(parts=[TextPart('Sunny!')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    assert received[0][: len(message_history)] == message_history


async def test_retry_prompt_answers_tool_call():
    """A tool-bound `RetryPromptPart` answers its call, so the call is not repaired."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Atlantis'}, tool_call_id='call_1')], timestamp=TS),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    'Unknown city, try again.', tool_name='get_weather', tool_call_id='call_1', timestamp=TS
                )
            ],
            timestamp=TS,
        ),
        ModelResponse(parts=[TextPart('I could not find that city.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    assert received[0][: len(message_history)] == message_history


async def test_plain_retry_prompt_does_not_answer_tool_call():
    """A `RetryPromptPart` with no `tool_name` is validation feedback, not a tool result.

    Even when its `tool_call_id` collides with an open call (a hand-built history), the call is
    still dangling: it gets a synthesized return, inserted ahead of the user-facing feedback.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Atlantis'}, tool_call_id='call_1')], timestamp=TS),
        ModelRequest(
            parts=[RetryPromptPart('Response was not valid, try again.', tool_call_id='call_1', timestamp=TS)],
            timestamp=TS,
        ),
        ModelResponse(parts=[TextPart('Let me try again.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    request = received[0][2]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='get_weather',
                content='The tool call was interrupted before a result was produced.',
                tool_call_id='call_1',
                metadata={'pydantic_ai_synthesized_tool_return': True},
                timestamp=TS,
                outcome='interrupted',
            ),
            RetryPromptPart(content='Response was not valid, try again.', tool_call_id='call_1', timestamp=TS),
        ]
    )


async def test_reused_tool_call_id_dangling_call_repaired():
    """A `tool_call_id` reused across responses doesn't mask the later call being dangling."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Sunny', tool_call_id='call_1', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Amsterdam'}, tool_call_id='call_1')], timestamp=TS),
        ModelRequest(parts=[UserPromptPart('Never mind.', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('OK.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # The earlier answered call is untouched; the later dangling reuse gets a synthesized return.
    request = received[0][4]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='get_weather',
                content='The tool call was interrupted before a result was produced.',
                tool_call_id='call_1',
                metadata={'pydantic_ai_synthesized_tool_return': True},
                timestamp=TS,
                outcome='interrupted',
            ),
            UserPromptPart(content='Never mind.', timestamp=TS),
        ]
    )


async def test_reused_tool_call_id_shadowed_open_call_repaired():
    """When an open call's ID is reused, a later result answers the new call, not the earlier one.

    The earlier call can no longer be answered and gets a synthesized return.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(parts=[UserPromptPart('In Amsterdam, I mean.', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Amsterdam'}, tool_call_id='call_1')], timestamp=TS),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Rainy', tool_call_id='call_1', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Rainy!')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # The earlier shadowed call gets the synthesized return; the later call keeps its real result.
    request = received[0][2]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='get_weather',
                content='The tool call was interrupted before a result was produced.',
                tool_call_id='call_1',
                metadata={'pydantic_ai_synthesized_tool_return': True},
                timestamp=TS,
                outcome='interrupted',
            ),
            UserPromptPart(content='In Amsterdam, I mean.', timestamp=TS),
        ]
    )
    later_result = received[0][4]
    assert isinstance(later_result, ModelRequest)
    assert later_result.parts == [ToolReturnPart('get_weather', 'Rainy', tool_call_id='call_1', timestamp=TS)]


async def test_dangling_tool_call_followed_by_response():
    """A dangling call directly followed by another response gets a new request in between."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')],
            timestamp=TS,
            provider_response_id='resp_1',
        ),
        ModelResponse(parts=[TextPart('I could not check the weather.')], timestamp=TS, provider_response_id='resp_2'),
    ]

    result = await agent.run('Try again?', message_history=message_history)
    assert result.output == 'All done.'

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Mexico City'}, tool_call_id='call_1')],
                timestamp=TS,
                provider_response_id='resp_1',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='I could not check the weather.')], timestamp=TS, provider_response_id='resp_2'
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Try again?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_tool_return_answering_call_across_intervening_response():
    """A return may answer a call across a non-answering intervening response; left unrepaired by design.

    The ordered walk in `_dangling_tool_calls_by_response` opens the call, skips the intervening
    text-only response, and closes the call when the later `ToolReturnPart` reuses its id — so the
    call reads as answered, no synthesized return is inserted, and the real return isn't orphaned.
    The cross-turn ordering (a result separated from its call by another response) is a documented
    out-of-scope boundary of `_clean_message_history` — providers with ordering rules beyond
    call/result pairing may reject it — and this pins that the pipeline leaves the shape untouched so
    a future change can't silently "repair" (and thereby alter) it.
    """
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')],
            timestamp=TS,
            provider_response_id='resp_1',
        ),
        ModelResponse(parts=[TextPart('I could not check the weather.')], timestamp=TS, provider_response_id='resp_2'),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Sunny', tool_call_id='call_1', timestamp=TS)], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # The call and its across-response return both survive verbatim; no synthesized return appears.
    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Mexico City'}, tool_call_id='call_1')],
                timestamp=TS,
                provider_response_id='resp_1',
            ),
            ModelResponse(
                parts=[TextPart(content='I could not check the weather.')], timestamp=TS, provider_response_id='resp_2'
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='Sunny', tool_call_id='call_1', timestamp=TS),
                    UserPromptPart(content='Thanks.', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_valid_history_untouched():
    """A history without dangling tool calls passes through byte-identical."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(
            parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
        ),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Sunny', tool_call_id='call_1', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('It is sunny.')], timestamp=TS),
    ]
    original = ModelMessagesTypeAdapter.dump_json(message_history)

    await agent.run('Thanks!', message_history=message_history)

    assert received[0][: len(message_history)] == message_history
    assert ModelMessagesTypeAdapter.dump_json(received[0][: len(message_history)]) == original


async def test_repair_is_idempotent_and_deterministic():
    """Repairing an already-repaired history is a no-op, and repair of the same input is byte-stable.

    This pins the prompt-cache-friendliness of the repair: the synthesized parts derive entirely
    from the input history, so reusing a repaired history never churns the serialized prefix.
    """

    def build_history() -> list[ModelMessage]:
        return [
            ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
            ),
            ModelRequest(parts=[UserPromptPart('Never mind.', timestamp=TS)], timestamp=TS),
            ModelResponse(parts=[TextPart('OK.')], timestamp=TS),
        ]

    # Determinism: two separate runs over equal inputs repair to byte-identical histories.
    agent_a, received_a = capture_agent()
    agent_b, received_b = capture_agent()
    result_a = await agent_a.run('Explain?', message_history=build_history())
    await agent_b.run('Explain?', message_history=build_history())
    repaired_len = len(build_history())
    assert ModelMessagesTypeAdapter.dump_json(received_a[0][:repaired_len]) == ModelMessagesTypeAdapter.dump_json(
        received_b[0][:repaired_len]
    )

    # Idempotency: feeding the repaired history into a new run leaves it untouched (verified by
    # output-equality, since repair is a no-op on an already-repaired history).
    repaired = result_a.all_messages()
    agent_c, received_c = capture_agent()
    await agent_c.run('Once more?', message_history=repaired)
    assert received_c[0][: len(repaired)] == repaired
    synthesized = [
        part
        for part in iter_message_parts(received_c[0], ModelRequest, ToolReturnPart)
        if part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
    ]
    assert len(synthesized) == 1


async def test_cancelled_stream_with_incomplete_tool_call_round_trips():
    """A stream cancelled mid-tool-call-args can be reused directly with a new user prompt.

    The partial tool call's args are unparsable; the call is kept verbatim in the interrupted
    response and closed out with a synthesized return.
    """

    received: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        received.append(messages)
        yield {0: DeltaToolCall(name='get_weather')}
        yield {0: DeltaToolCall(json_args='{"city": "Mex')}
        yield 'Let me ch'
        # Never reached: the consumer cancels after the first text chunk.
        yield {0: DeltaToolCall(json_args='ico City"}')}  # pragma: no cover
        yield 'eck.'  # pragma: no cover

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart('No problem!')])

    agent = Agent(FunctionModel(model_function, stream_function=stream_function))

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        raise AssertionError('The interrupted tool call should never execute')

    async with agent.run_stream('What is the weather?') as result:
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break
        await result.cancel()

    messages = result.all_messages()
    interrupted = messages[-1]
    assert isinstance(interrupted, ModelResponse)
    assert interrupted.state == 'interrupted'
    assert interrupted.tool_calls[0].args == '{"city": "Mex'

    result2 = await agent.run('Never mind, just say hi.', message_history=messages)
    assert result2.output == 'No problem!'
    # The incomplete tool call survives verbatim, closed out by a synthesized return.
    assert received[1] == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_weather',
                        args='{"city": "Mex',
                        tool_call_id=(tool_call_id := IsSameStr()),
                    ),
                    TextPart(content='Let me ch'),
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=6),
                model_name='function:model_function:stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id=tool_call_id,
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=IsDatetime(),
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Never mind, just say hi.', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_cancelled_stream_with_complete_tool_call_round_trips():
    """An interrupted response whose dangling tool call has complete args gets a synthesized return."""
    received: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        received.append(messages)
        yield {0: DeltaToolCall(name='get_weather', json_args='{"city": "Mexico City"}')}
        yield 'Let me check.'

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart('No problem!')])

    agent = Agent(FunctionModel(model_function, stream_function=stream_function))

    async with agent.run_stream('What is the weather?') as result:
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break
        await result.cancel()

    messages = result.all_messages()

    result2 = await agent.run('Never mind, just say hi.', message_history=messages)
    assert result2.output == 'No problem!'

    # The dangling call is kept and answered with a synthesized return ahead of the new prompt.
    request = received[1][-1]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='get_weather',
                content='The tool call was interrupted before a result was produced.',
                tool_call_id=IsStr(),
                metadata={'pydantic_ai_synthesized_tool_return': True},
                timestamp=IsDatetime(),
                outcome='interrupted',
            ),
            UserPromptPart(content='Never mind, just say hi.', timestamp=IsDatetime()),
        ]
    )


async def test_interrupted_tool_execution_round_trips():
    """A run that crashes mid-tool-execution leaves a partial request that can be resumed directly."""
    received: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('get_volume', {'size': 6}, tool_call_id='call_volume'),
                    ToolCallPart('get_mass', {'size': 6}, tool_call_id='call_mass'),
                ]
            )
        return ModelResponse(parts=[TextPart('The volume is 216.')])

    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain(sequential=True)
    def get_volume(size: int) -> int:
        return size**3

    @agent.tool_plain(sequential=True)
    def get_mass(size: int) -> int:
        raise RuntimeError('missing density')

    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='missing density'):
            await agent.run('Calculate volume and mass.')

    result = await agent.run(message_history=messages)
    assert result.output == 'The volume is 216.'

    # The completed tool return is preserved, and the crashed call is closed out after it.
    request = received[1][-1]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='get_volume',
                content=216,
                tool_call_id='call_volume',
                timestamp=IsDatetime(),
            ),
            ToolReturnPart(
                tool_name='get_mass',
                content='The tool call was interrupted before a result was produced.',
                tool_call_id='call_mass',
                metadata={'pydantic_ai_synthesized_tool_return': True},
                timestamp=IsDatetime(),
                outcome='interrupted',
            ),
        ]
    )


async def test_history_processor_output_repaired():
    """Dangling tool calls introduced by a history processor are repaired before the request is sent."""
    received: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart('All done.')])

    def truncating_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart('get_weather', {'city': 'Mexico City'}, tool_call_id='call_1')], timestamp=TS
            ),
            *messages[-1:],
        ]

    agent = Agent(FunctionModel(model_function), capabilities=[ProcessHistory(truncating_processor)])

    await agent.run('Explain?')

    assert received[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=TS)], timestamp=TS),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Mexico City'}, tool_call_id='call_1')],
                timestamp=TS,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_1',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='interrupted',
                    ),
                    UserPromptPart(content='Explain?', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
