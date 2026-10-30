"""Instructions re-resolved during a realtime session, and their changes delivered as appends."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Callable, Sequence

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._framing import SYSTEM_TAG, wrap
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentInstructionSource,
    CapabilityInstructionSource,
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionDigest,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.realtime._utils import InstructionChangeContext
from pydantic_ai.realtime.codec import (
    OutputTranscript,
    RealtimeInput,
    ResponseDone,
    TextContext,
    ToolCall,
    ToolResult,
)

from ..conftest import IsDatetime
from .test_session import BlockingRealtimeConnection, FakeRealtimeConnection, FakeRealtimeModel, _profile  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio

_TURN = [OutputTranscript(text='Hello.', is_final=True), ResponseDone()]


def _texts(sent: Sequence[RealtimeInput]) -> list[str]:
    return [content.text for content in sent if isinstance(content, TextContext)]


def _changing_agent() -> tuple[Agent[None, str], dict[str, str | None]]:
    """An agent with a named dynamic instruction that reads `state`, and a literal one."""
    state: dict[str, str | None] = {'mood': 'cheerful'}
    agent: Agent[None, str] = Agent(instructions='You are a voice assistant.')

    @agent.instructions(name='mood')
    def mood() -> str | None:
        return f'Be {state["mood"]}.' if state['mood'] is not None else None

    return agent, state


async def test_named_dynamic_instruction_is_tagged_at_connect() -> None:
    """A dynamic block with a name goes out as a `<context>` element, as a standard run's append-mode prefix does."""
    agent, _ = _changing_agent()

    @agent.instructions
    def unnamed() -> str:
        return 'Unnamed text.'

    model = FakeRealtimeModel(FakeRealtimeConnection([ResponseDone()]))
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]
    assert model.last_instructions == snapshot("""\
You are a voice assistant.

<context id="agent:mood">
Be cheerful.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.

Unnamed text.\
""")


async def test_static_and_unnamed_instructions_are_sent_as_before() -> None:
    """Instructions a session can't name or that can't change go out untagged, as before."""
    agent: Agent[None, str] = Agent(instructions='Base')

    @agent.instructions
    def extra() -> str:
        return 'Dynamic'

    model = FakeRealtimeModel(FakeRealtimeConnection([ResponseDone()]))
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]
    assert model.last_instructions == 'Base\n\nDynamic'


async def test_change_after_a_response_is_delivered_as_a_system_message() -> None:
    agent, state = _changing_agent()
    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        state['mood'] = 'calm'
        _ = [event async for event in session]

    (change,) = connection.sent
    assert isinstance(change, InstructionChangeContext)
    assert change.system_prompt == snapshot("""\
<context id="agent:mood">
Be calm.
</context>\
""")
    request = session.all_messages()[-1]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            InstructionBaselinePart(
                instructions=[
                    InstructionDigest(
                        id=InstructionId(source=AgentInstructionSource()),
                        content_hash='4cba3792916a99503cb36a73e828b11a420284576051a904e6b6291c97828b87',
                    ),
                    InstructionPart(
                        content='Be cheerful.',
                        dynamic=True,
                        on_change='append',
                        name='mood',
                        id=InstructionId(source=AgentInstructionSource(), name='mood'),
                    ),
                ]
            ),
            InstructionDeltaPart(
                id=InstructionId(source=AgentInstructionSource(), name='mood'),
                content='Be calm.',
                base_hash='4b81db3acae2b5acffd8d616b38c6defd0f148658ccc83c6d74c35b3d335ea5f',
            ),
        ]
    )


async def test_change_without_a_system_channel_is_system_tagged_text() -> None:
    agent, state = _changing_agent()
    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection, profile=_profile(supports_system_messages=False))
    async with agent.realtime(model).session() as session:
        state['mood'] = 'calm'
        _ = [event async for event in session]

    (change,) = connection.sent
    assert type(change) is TextContext
    assert change.text == snapshot("""\
<system><context id="agent:mood">
Be calm.
</context></system>\
""")


async def test_unchanged_instructions_send_nothing() -> None:
    agent, _ = _changing_agent()
    connection = FakeRealtimeConnection([*_TURN, *_TURN])
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]
    assert connection.sent == []
    assert not any(
        isinstance(part, (InstructionDeltaPart, InstructionBaselinePart))
        for message in session.all_messages()
        for part in message.parts
    )


async def test_later_changes_fold_onto_earlier_ones() -> None:
    """Each change is computed against the last one delivered, and only the first opens the session's window."""
    calls = 0
    agent: Agent[None, str] = Agent()

    @agent.instructions(name='counter')
    def counter() -> str:
        nonlocal calls
        calls += 1
        return f'Turn {calls}.'

    connection = FakeRealtimeConnection([*_TURN, *_TURN])
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]

    assert [change.system_prompt for change in connection.sent if isinstance(change, InstructionChangeContext)] == (
        snapshot(
            [
                """\
<context id="agent:counter">
Turn 2.
</context>\
""",
                """\
<context id="agent:counter">
Turn 3.
</context>\
""",
            ]
        )
    )
    changes = [
        message
        for message in session.all_messages()
        if isinstance(message, ModelRequest) and any(isinstance(part, InstructionDeltaPart) for part in message.parts)
    ]
    assert [[type(part).__name__ for part in message.parts] for message in changes] == snapshot(
        [['InstructionBaselinePart', 'InstructionDeltaPart'], ['InstructionDeltaPart']]
    )


async def test_withdrawn_instruction() -> None:
    agent, state = _changing_agent()
    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        state['mood'] = None
        _ = [event async for event in session]
    (change,) = connection.sent
    assert isinstance(change, InstructionChangeContext)
    assert change.system_prompt == snapshot("""\
<context id="agent:mood">
This context has been withdrawn.
</context>\
""")


async def test_change_made_by_a_tool_goes_out_before_its_result() -> None:
    agent, state = _changing_agent()

    @agent.tool_plain
    def calm_down() -> str:
        state['mood'] = 'calm'
        return 'Done.'

    connection = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='calm_down', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]
    assert [type(content).__name__ for content in connection.sent] == ['InstructionChangeContext', 'ToolResult']
    result = connection.sent[1]
    assert isinstance(result, ToolResult) and result.output == 'Done.'


async def test_unaddressable_change_warns_once_and_sends_nothing() -> None:
    calls = 0
    agent: Agent[None, str] = Agent()

    @agent.instructions
    def unnamed() -> str:
        nonlocal calls
        calls += 1
        return f'Turn {calls}.'

    connection = FakeRealtimeConnection([*_TURN, *_TURN])
    model = FakeRealtimeModel(connection)
    with pytest.warns(UserWarning, match='An instruction without a name of its own changed') as record:
        async with agent.realtime(model).session() as session:
            _ = [event async for event in session]
    assert len(record) == 1
    assert connection.sent == []
    assert calls == 3


async def test_appending_system_prompt_is_tagged_at_connect_and_restated_when_it_changes() -> None:
    state = {'caller': 'unknown'}
    agent: Agent[None, str] = Agent()

    @agent.system_prompt(dynamic=True, on_change='append')
    def caller() -> str:
        return f'The caller is {state["caller"]}.'

    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        state['caller'] = 'Jamie'
        _ = [event async for event in session]

    assert model.last_instructions == snapshot("""\
<context id="system_prompt:caller">
The caller is unknown.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""")
    (change,) = connection.sent
    assert isinstance(change, InstructionChangeContext)
    assert change.system_prompt == snapshot("""\
<context id="system_prompt:caller">
The caller is Jamie.
</context>\
""")
    request = session.all_messages()[-1]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            SystemPromptPart(
                content='The caller is Jamie.',
                timestamp=IsDatetime(),
                dynamic_ref=caller.__qualname__,
                context_id='system_prompt:caller',
            )
        ]
    )


async def test_seeded_appending_system_prompt_is_tagged_at_connect() -> None:
    agent: Agent[None, str] = Agent()

    @agent.system_prompt(dynamic=True, on_change='append')
    def dynamic_prompt() -> str:
        return 'Current value.'

    seed: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Stale value.',
                    dynamic_ref=dynamic_prompt.__qualname__,
                    context_id='system_prompt:dynamic_prompt',
                ),
                UserPromptPart(content='earlier question'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='earlier answer')]),
    ]
    connection = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model, message_history=seed).session() as session:
        _ = [event async for event in session]
    # The seeded prompt is the standing one the session opens with, and its current value is what it's sent.
    assert model.last_instructions == snapshot("""\
<context id="system_prompt:dynamic_prompt">
Current value.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""")
    assert connection.sent == []


async def test_failing_instruction_function_ends_the_session() -> None:
    calls = 0
    agent: Agent[None, str] = Agent()

    @agent.instructions(name='flaky')
    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError('lookup failed')
        return 'Fine.'

    model = FakeRealtimeModel(FakeRealtimeConnection(_TURN))
    with pytest.raises(RuntimeError, match='lookup failed'):
        async with agent.realtime(model).session() as session:
            _ = [event async for event in session]


async def test_reconnect_replays_changes_as_sent() -> None:
    agent, state = _changing_agent()
    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:
        state['mood'] = 'calm'
        _ = [event async for event in session]
        replayed = session._replayable_messages()  # pyright: ignore[reportPrivateUsage]

    (change,) = connection.sent
    assert isinstance(change, InstructionChangeContext)
    last = replayed[-1]
    assert isinstance(last, ModelRequest)
    # Replayed where it was sent: as the first request of a session that opened with none of its own, it's
    # `<system>`-tagged text, as a system prompt sent there with `send()` is.
    assert last.parts == [UserPromptPart(wrap(SYSTEM_TAG, change.system_prompt), timestamp=IsDatetime())]


class _SideChannelConnection(FakeRealtimeConnection):
    @property
    def _replays_instruction_changes(self) -> bool:
        return False


async def test_reconnect_leaves_out_changes_a_connection_carries_itself() -> None:
    agent, state = _changing_agent()
    model = FakeRealtimeModel(_SideChannelConnection(_TURN))
    async with agent.realtime(model).session() as session:
        state['mood'] = 'calm'
        _ = [event async for event in session]
        replayed = session._replayable_messages()  # pyright: ignore[reportPrivateUsage]
    assert not any(isinstance(part, SystemPromptPart) for message in replayed for part in message.parts)
    assert any(isinstance(part, InstructionDeltaPart) for message in session.all_messages() for part in message.parts)


async def test_standard_run_continuing_the_session_sends_its_own_prefix() -> None:
    """A standard run continuing a session's history sends its current instructions, not the session's stale changes."""
    agent, state = _changing_agent()
    model = FakeRealtimeModel(FakeRealtimeConnection(_TURN))
    async with agent.realtime(model).session() as session:
        state['mood'] = 'calm'
        _ = [event async for event in session]

    received: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    state['mood'] = 'focused'
    result = await agent.run('Next?', message_history=session.all_messages(), model=FunctionModel(respond))
    assert result.output == 'ok'
    (messages,) = received
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    assert last.instructions == 'You are a voice assistant.\n\nBe focused.'
    rendered = [part.content for message in messages for part in message.parts if isinstance(part, SystemPromptPart)]
    assert rendered == []


async def test_text_turn_waits_for_the_instructions_resolved_after_the_last_response() -> None:
    """A text turn asks for a response, so it goes out after the change the last response's refresh delivers."""
    resolving = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    agent: Agent[None, str] = Agent()

    @agent.instructions(name='counter')
    async def counter() -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            resolving.set()
            await release.wait()
        return f'Turn {calls}.'

    connection = BlockingRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    async with agent.realtime(model).session() as session:

        async def consume() -> None:
            async for _ in session:
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(resolving.wait(), 5)
        sending = asyncio.create_task(session.send('Next question.'))
        await asyncio.sleep(0)
        assert connection.sent == []
        release.set()
        await asyncio.wait_for(sending, 5)
        consumer.cancel()
    assert [type(content).__name__ for content in connection.sent] == ['InstructionChangeContext', 'str']


async def test_loaded_capability_block_changes_are_delivered_and_left_out_of_the_window() -> None:
    """A capability a seeded run loaded joins the instructions as its block, whose changes a session delivers too.

    The block isn't recorded in the window the session's first change opens: a standard run continuing the
    conversation delivers it through history rather than its prefix, as it did before the session.
    """
    state = {'accent': 'pirate'}

    class Accent(AbstractCapability[None]):
        id = 'accent'
        defer_loading = True

        def get_description(self) -> str:
            return 'Speak with an accent.'

        def get_instructions(self) -> Callable[[], str]:
            return lambda: f'Speak like a {state["accent"]}.'

    def load_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'accent'})])
        return ModelResponse(parts=[TextPart('Arr.')])

    history = (await Agent(FunctionModel(load_then_answer), capabilities=[Accent()]).run('Hello.')).all_messages()

    connection = FakeRealtimeConnection(_TURN)
    model = FakeRealtimeModel(connection)
    agent: Agent[None, str] = Agent(instructions='Be helpful.', capabilities=[Accent()])
    async with agent.realtime(model, message_history=history).session() as session:
        state['accent'] = 'cowboy'
        _ = [event async for event in session]

    assert model.last_instructions == snapshot("""\
Be helpful.

<context id="capability:deferred_capabilities">
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- accent: Speak with an accent.
</context>

<context id="capability:accent">
Speak like a pirate.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""")
    (change,) = connection.sent
    assert isinstance(change, InstructionChangeContext)
    assert change.system_prompt == snapshot("""\
<context id="capability:accent">
Speak like a cowboy.
</context>\
""")
    request = session.all_messages()[-1]
    assert isinstance(request, ModelRequest)
    assert request.parts == snapshot(
        [
            InstructionBaselinePart(
                instructions=[
                    InstructionDigest(
                        id=InstructionId(source=AgentInstructionSource()),
                        content_hash='5ad131c7cb35b34d24e0110e754475cd7d369fc9b9f3dd0c967aeb35e6fbcb4f',
                    )
                ]
            ),
            InstructionDeltaPart(
                id=InstructionId(source=CapabilityInstructionSource(id='accent')),
                content='Speak like a cowboy.',
                base_hash='099d8fe8641adc60bb870214d1b90f754c631f354e450fad727a9e4495329d96',
            ),
        ]
    )
