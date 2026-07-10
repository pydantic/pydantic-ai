"""Tests for out-of-the-box repair of message histories with dangling tool calls.

A run that is cancelled or crashes mid-tool leaves a `ModelResponse` containing `ToolCallPart`s
with no matching `ToolReturnPart`/`RetryPromptPart` in a following `ModelRequest`, which strict
providers reject. Before each model request, the framework synthesizes deterministic tool returns
for such dangling calls and drops calls whose args were cut off mid-stream, so interrupted and
hand-built histories can be reused directly. Synthesized returns carry the
`pydantic_ai_synthesized_tool_return` metadata marker so repairs are inspectable in the history.

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
from pydantic_ai._agent_graph import SYNTHESIZED_TOOL_RETURN_METADATA_KEY
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsStr

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
                        outcome='failed',
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
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
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
                        outcome='failed',
                    ),
                    ToolReturnPart(
                        tool_name='get_density',
                        content='The tool call was interrupted before a result was produced.',
                        tool_call_id='call_density',
                        metadata={'pydantic_ai_synthesized_tool_return': True},
                        timestamp=TS,
                        outcome='failed',
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_incomplete_tool_call_args_dropped():
    """A dangling tool call whose args were cut off mid-stream is dropped, not synthesized.

    A sibling dangling call whose args did stream completely still gets a synthesized return.
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
                        outcome='failed',
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


async def test_response_with_only_incomplete_tool_call_dropped():
    """A response left with no parts after dropping an incomplete tool call is dropped entirely."""
    agent, received = capture_agent()

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[ToolCallPart('get_weather', '{"city": "Mex', tool_call_id='call_1')], timestamp=TS),
        ModelRequest(parts=[UserPromptPart('Are you there?', timestamp=TS)], timestamp=TS),
        ModelResponse(parts=[TextPart('Yes.')], timestamp=TS),
    ]

    await agent.run('Thanks.', message_history=message_history)

    # With the response dropped, the surrounding requests are merged into one.
    assert received[0] == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What is the weather?', timestamp=TS),
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


async def test_result_before_call_does_not_mask_dangling_call():
    """A tool result that precedes its call is an orphan and doesn't answer the call."""
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

    # The call after the orphaned result is genuinely dangling and gets a synthesized return.
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
                outcome='failed',
            ),
            UserPromptPart(content='And tomorrow?', timestamp=TS),
        ]
    )


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
                outcome='failed',
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
                outcome='failed',
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
                        outcome='failed',
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
        for message in received_c[0]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}
    ]
    assert len(synthesized) == 1


async def test_cancelled_stream_with_incomplete_tool_call_round_trips():
    """A stream cancelled mid-tool-call-args can be reused directly with a new user prompt.

    The partial tool call's args are unparsable, so it is dropped from the interrupted response.
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
    # The incomplete tool call was dropped; the streamed text survives.
    assert received[1] == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Let me ch')],
                usage=RequestUsage(input_tokens=50, output_tokens=6),
                model_name='function:model_function:stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Never mind, just say hi.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
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
                outcome='failed',
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
                outcome='failed',
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
                        outcome='failed',
                    ),
                    UserPromptPart(content='Explain?', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
