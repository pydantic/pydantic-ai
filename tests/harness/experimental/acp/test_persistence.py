"""Tests for ACP session persistence (`session/load` via a `SessionStore`)."""

from __future__ import annotations

import asyncio
import logging

import acp
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import InMemorySessionStore, PydanticAIACPAgent, StoredSession
from tests.experimental.acp._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _adapter(store: InMemorySessionStore | None) -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text='hello')), session_store=store)


def test_stored_session_round_trips_through_pydantic() -> None:
    # A durable store serializes `StoredSession` with Pydantic; the whole `SessionUpdate` union
    # (not just the variants a turn happens to produce) must survive a JSON round-trip.
    original = StoredSession(
        messages=[
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='yo')]),
        ],
        updates=[
            acp.update_user_message_text('hi'),
            acp.update_agent_message_text('yo'),
            acp.update_agent_thought_text('thinking'),
        ],
        model='openai:gpt-4o',
    )
    adapter = TypeAdapter(StoredSession)
    assert adapter.validate_json(adapter.dump_json(original)) == original


async def test_initialize_advertises_load_session_only_with_a_store() -> None:
    with_store = (await _adapter(InMemorySessionStore()).initialize(protocol_version=1)).agent_capabilities
    without_store = (await _adapter(None).initialize(protocol_version=1)).agent_capabilities
    assert with_store is not None and with_store.load_session is True
    assert without_store is not None and without_store.load_session is False


async def test_new_session_persists_an_empty_session() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    # The session is stored on creation, so it can be reopened before its first turn.
    stored = await store.load(session.session_id)
    assert stored == StoredSession(messages=[], updates=[])


async def test_turn_persists_history_and_transcript() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    client = RecordingClient()
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)

    stored = await store.load(session.session_id)
    assert stored is not None
    assert len(stored.messages) > 0  # the model exchange was persisted
    # The transcript is the user's prompt (recorded for replay, never sent live -- the client
    # renders its own prompt) followed by exactly what the client was shown this turn.
    assert stored.updates == [acp.update_user_message_text('hi'), *client.updates]


async def test_load_session_restores_history_and_replays_transcript() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    first = RecordingClient()
    adapter.on_connect(first)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)
    shown = list(first.updates)

    # Reopen the session on a fresh connection, as an editor would after a restart.
    reopened = RecordingClient()
    adapter.on_connect(reopened)
    await adapter.load_session(cwd='/ws', session_id=session.session_id)

    # The whole conversation is replayed to the new client: the user's turn (which the live
    # client rendered itself, so it was never sent as an update) followed by what was shown.
    assert reopened.updates == [acp.update_user_message_text('hi'), *shown]
    # ...and the model history is restored so the next turn continues the conversation.
    stored = await store.load(session.session_id)
    assert stored is not None
    assert adapter._sessions[session.session_id].history == stored.messages  # pyright: ignore[reportPrivateUsage]


async def test_load_over_an_active_session_cancels_the_in_flight_turn() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    unwound = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        try:
            await release.wait()  # released only by cancellation
            return 'done'  # pragma: no cover - cancelled by the load
        finally:
            unwound.set()

    store = InMemorySessionStore()
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, session_store=store)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session.session_id))
    await asyncio.wait_for(started.wait(), timeout=5)

    # Reopen the still-open session; its in-flight turn must be torn down, not orphaned to later
    # persist stale state over the restored transcript.
    await adapter.load_session(cwd='/ws', session_id=session.session_id)

    assert unwound.is_set()
    response = await asyncio.wait_for(turn, timeout=5)
    assert response.stop_reason == 'cancelled'
    # The freshly loaded state replaced the old one and carries no leftover in-flight turn.
    assert adapter._sessions[session.session_id].active_turn is None  # pyright: ignore[reportPrivateUsage]


async def test_queued_prompt_does_not_clobber_a_reloaded_session() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await release.wait()
        return 'done'  # pragma: no cover - cancelled by the load

    store = InMemorySessionStore()
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, session_store=store)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    snapshot = await store.load(session.session_id)

    first = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('one')], session_id=session.session_id))
    await asyncio.wait_for(started.wait(), timeout=5)
    # A second prompt queues on the old state's turn lock; the load then replaces that state.
    queued = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('two')], session_id=session.session_id))
    await asyncio.sleep(0)
    await adapter.load_session(cwd='/ws', session_id=session.session_id)

    assert (await asyncio.wait_for(first, timeout=5)).stop_reason == 'cancelled'
    # The queued prompt held the *replaced* state: letting it run would commit and persist that
    # orphaned history over the session just restored from the store.
    with pytest.raises(acp.RequestError):
        await asyncio.wait_for(queued, timeout=5)
    assert await store.load(session.session_id) == snapshot


async def test_load_session_dropped_from_memory_restores_from_the_store() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)

    # Drop the live session (as a close, or a process restart, would) so only the stored copy
    # remains -- exercising the load path where no in-memory session is being replaced.
    await adapter.close_session(session_id=session.session_id)
    assert session.session_id not in adapter._sessions  # pyright: ignore[reportPrivateUsage]

    adapter.on_connect(RecordingClient())
    await adapter.load_session(cwd='/ws', session_id=session.session_id)

    assert session.session_id in adapter._sessions  # pyright: ignore[reportPrivateUsage]


async def test_load_unknown_session_is_rejected() -> None:
    adapter = _adapter(InMemorySessionStore())
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with pytest.raises(acp.RequestError):
        await adapter.load_session(cwd='/ws', session_id='does-not-exist')


async def test_load_session_without_a_store_is_method_not_found() -> None:
    adapter = _adapter(None)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with pytest.raises(acp.RequestError):
        await adapter.load_session(cwd='/ws', session_id='whatever')


class _BlockingSaveStore(InMemorySessionStore):
    """A store whose next save suspends until released, modeling a durable backend mid-write."""

    def __init__(self) -> None:
        super().__init__()
        self.saving = asyncio.Event()
        self.block_next_save = False

    async def save(self, session_id: str, session: StoredSession) -> None:
        if self.block_next_save:
            self.block_next_save = False
            self.saving.set()
            await asyncio.Event().wait()  # suspend until cancelled
        await super().save(session_id, session)


async def test_cancel_landing_in_the_post_commit_save_commits_but_answers_cancelled() -> None:
    store = _BlockingSaveStore()
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
        Agent(TestModel(custom_output_text='done')), session_store=store
    )
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    store.block_next_save = True

    turn = asyncio.ensure_future(
        adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id, message_id='m1')
    )
    await asyncio.wait_for(store.saving.wait(), timeout=5)
    # The turn is fully committed in memory and is suspended inside the store's save: a cancel
    # arriving now came too late to roll anything back. The spec still requires the prompt to
    # answer `cancelled`, but the committed signals must survive: usage is reported, and the
    # session history keeps the turn.
    await adapter.cancel(session_id=session.session_id)
    response = await asyncio.wait_for(turn, timeout=5)

    assert response.stop_reason == 'cancelled'
    assert response.usage is not None
    assert len(adapter._sessions[session.session_id].history) >= 2  # pyright: ignore[reportPrivateUsage]


class _FailingStore:
    """A SessionStore whose operations always raise, standing in for a broken durable backend."""

    async def save(self, session_id: str, session: StoredSession) -> None:
        raise OSError('disk full')

    async def load(self, session_id: str) -> StoredSession | None:
        raise OSError('unreadable')


async def test_save_failure_is_logged_and_does_not_fail_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
        Agent(TestModel(custom_output_text='ok')), session_store=_FailingStore()
    )
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with caplog.at_level(logging.ERROR):
        session = await adapter.new_session(cwd='/ws')  # persisting the empty session fails...
        response = await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)

    # ...as does persisting the turn, yet the turn the user already saw stream still succeeds; the
    # durable-write failure is only logged.
    assert response.stop_reason == 'end_turn'
    assert 'failed to persist' in caplog.text


async def test_set_model_config_save_failure_is_logged_and_does_not_fail_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
        Agent(TestModel()), models=['test'], session_store=_FailingStore()
    )
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    with caplog.at_level(logging.ERROR):
        response = await adapter.set_config_option(config_id='model', value='test', session_id=session.session_id)
    # The selection took effect in memory; only the durable copy is behind, which is logged.
    assert response is not None
    assert adapter._sessions[session.session_id].model == 'test'  # pyright: ignore[reportPrivateUsage]
    assert 'failed to persist' in caplog.text


async def test_load_read_failure_is_reported_as_a_clean_error() -> None:
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()), session_store=_FailingStore())
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with pytest.raises(acp.RequestError) as excinfo:
        await adapter.load_session(cwd='/ws', session_id='whatever')
    # A read/deserialize failure surfaces as a purpose-built internal error, not a leaked exception.
    assert excinfo.value.code == -32603
    assert 'could not be read' in str(excinfo.value.data)
