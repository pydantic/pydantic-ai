"""Tests for the Pydantic AI -> ACP adapter.

Organized by behavior: lifecycle, streaming/event ordering, tool calls, permission/HITL,
cancellation, multi-turn history & isolation, errors, unsupported methods, and entry points.
Most tests drive the adapter directly with an in-memory `FakeClient`; the entry-point tests
spawn a real subprocess over stdio, as a TUI client (Zed/Toad) would.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import acp
import pytest
from acp import RequestError, schema
from pydantic import BaseModel
from pydantic_ai import Agent, DeferredToolRequests, RunContext, UsageLimitExceeded
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    FinishReason,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.experimental.acp import (
    AcpSession,
    AcpSessionConfig,
    AcpTerminalToolset,
    InMemorySessionStore,
    PydanticAIACPAgent,
    ToolCallPresentation,
    chain_presenters,
    default_coding_presenter,
    run_acp_stdio,
    run_acp_stdio_sync,
)
from pydantic_ai_harness.experimental.acp._adapter import (
    _finish_reason_to_stop_reason,
    _TurnState,
    _usage_limit_stop_reason,
)
from pydantic_ai_harness.experimental.acp._content import PromptContentBlock
from pydantic_ai_harness.experimental.acp._presentation import _HANDLERS, absolutize
from pydantic_ai_harness.experimental.acp._serialize import (
    MAX_RAW_FIELD_CHARS,
    MAX_TEXT_UPDATE_BYTES,
    _escaped_len,
    chunk_text,
)
from pydantic_ai_harness.experimental.acp._session import SessionState
from tests.experimental.acp._acp_clients import (  # pyright: ignore[reportMissingTypeStubs]
    RecordingClient,
    RecordingClientBase,
)

pytestmark = pytest.mark.anyio

# A decider maps a permission request (the tool call) to the option_id the client "clicks",
# or to None to signal a cancelled permission outcome (the user dismissed the dialog).
PermissionDecider = Callable[[schema.ToolCallUpdate], str | None]


class FakeClient(RecordingClientBase):
    """In-memory ACP client recording `session/update`s and answering `session/request_permission`.

    `decider` chooses the permission option per request (default: allow once). Returning `None`
    from the decider produces a cancelled outcome. Filesystem and terminal callbacks stay stubbed
    by `RecordingClientBase`.
    """

    def __init__(self, decider: PermissionDecider | None = None) -> None:
        super().__init__()
        self.permission_requests: list[schema.ToolCallUpdate] = []
        self._decider: PermissionDecider = decider or (lambda _call: 'allow_once')

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **kwargs: object,
    ) -> schema.RequestPermissionResponse:
        self.permission_requests.append(tool_call)
        option_id = self._decider(tool_call)
        if option_id is None:
            return schema.RequestPermissionResponse(outcome=schema.DeniedOutcome(outcome='cancelled'))
        return schema.RequestPermissionResponse(outcome=schema.AllowedOutcome(outcome='selected', option_id=option_id))

    # --- assertion helpers -------------------------------------------------------------

    def events(self) -> list[tuple[str, object]]:
        """The ordered stream as `(kind, payload)` pairs, where payload is the text or status.

        Raises on an unexpected update kind so schema drift surfaces loudly rather than silently.
        """
        out: list[tuple[str, object]] = []
        for update in self.updates:
            kind = getattr(update, 'session_update', '')
            if kind in ('agent_message_chunk', 'agent_thought_chunk'):
                out.append((kind, getattr(getattr(update, 'content', None), 'text', '')))
            elif kind == 'tool_call':
                out.append((kind, getattr(update, 'title', '')))
            elif kind == 'tool_call_update':
                out.append((kind, getattr(update, 'status', '')))
            else:  # pragma: no cover - guards against schema drift, not exercised
                raise AssertionError(f'unexpected session/update kind: {kind!r}')
        return out

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events()]

    def text(self) -> str:
        return ''.join(str(payload) for kind, payload in self.events() if kind == 'agent_message_chunk')

    def tool_starts(self) -> list[tuple[object, object, object]]:
        """`(tool_call_id, title, status)` for each `tool_call` (start) update."""
        return [
            (getattr(u, 'tool_call_id', None), getattr(u, 'title', None), getattr(u, 'status', None))
            for u in self.updates
            if getattr(u, 'session_update', '') == 'tool_call'
        ]

    def tool_completions(self) -> list[tuple[object, object, object]]:
        """`(tool_call_id, status, raw_output)` for each `tool_call_update` update."""
        return [
            (getattr(u, 'tool_call_id', None), getattr(u, 'status', None), getattr(u, 'raw_output', None))
            for u in self.updates
            if getattr(u, 'session_update', '') == 'tool_call_update'
        ]


async def _start(adapter: PydanticAIACPAgent[None, object], client: FakeClient) -> str:
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='.')
    return session.session_id


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    last = messages[-1]
    return isinstance(last, ModelRequest) and any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in last.parts)


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    """Every user-prompt text in history, flattening both bare-string and list `content` forms."""
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            if isinstance(part.content, str):  # pragma: no cover - adapter always sends list content
                texts.append(part.content)
            else:
                texts.extend(item for item in part.content if isinstance(item, str))
    return texts


def _calls_tool_each_turn(*deltas: DeltaToolCall) -> FunctionModel:
    """A model that calls the given tool(s) on every user turn, then replies `done`.

    Unlike `TestModel` (which calls each tool only once per conversation), this re-invokes the
    tools on each turn, which is needed to exercise multi-turn approval behavior.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if _has_tool_return(messages):
            yield 'done'
        else:
            yield {i: delta for i, delta in enumerate(deltas)}

    return FunctionModel(stream_function=stream)


def _approval_agent(executed: list[str]) -> Agent[None, str]:
    agent = Agent(_calls_tool_each_turn(DeltaToolCall(name='delete_file', json_args='{"path": "x"}')))

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        executed.append(path)
        return f'deleted {path}'

    return agent


class TestLifecycle:
    async def test_initialize_advertises_capabilities(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()), name='demo', version='9.9')
        response = await adapter.initialize(protocol_version=1)
        assert response.protocol_version == 1
        assert response.agent_info is not None
        assert (response.agent_info.name, response.agent_info.version) == ('demo', '9.9')
        capabilities = response.agent_capabilities
        assert capabilities is not None
        assert capabilities.load_session is False
        assert capabilities.prompt_capabilities is not None
        # Text-only by default; rich content is opt-in via `prompt_capabilities`.
        assert capabilities.prompt_capabilities.image is False
        assert capabilities.prompt_capabilities.audio is False
        assert capabilities.prompt_capabilities.embedded_context is False
        # Session close is implemented, so it is advertised.
        assert capabilities.session_capabilities is not None
        assert capabilities.session_capabilities.close is not None

    async def test_initialize_negotiates_down_to_supported_version(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        response = await adapter.initialize(protocol_version=acp.PROTOCOL_VERSION + 1)
        assert response.protocol_version == acp.PROTOCOL_VERSION

    async def test_initialize_rejects_unsupported_lower_version(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        response = await adapter.initialize(protocol_version=0)
        assert response.protocol_version == acp.PROTOCOL_VERSION

    async def test_prompt_capabilities_can_be_restricted(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            Agent(TestModel()),
            prompt_capabilities=schema.PromptCapabilities(image=False, audio=False, embedded_context=True),
        )
        response = await adapter.initialize(protocol_version=1)
        capabilities = response.agent_capabilities
        assert capabilities is not None and capabilities.prompt_capabilities is not None
        # The conservative values the embedder chose are advertised verbatim.
        assert capabilities.prompt_capabilities.image is False
        assert capabilities.prompt_capabilities.audio is False
        assert capabilities.prompt_capabilities.embedded_context is True

    async def test_image_block_reaches_the_model_as_binary_content(self) -> None:
        # End-to-end through prompt(): an opted-in image block must arrive in the model's user
        # prompt as the decoded bytes, not be dropped or stringified on the way.
        raw = b'\x89PNG\r\n'
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            Agent(TestModel()), prompt_capabilities=schema.PromptCapabilities(image=True)
        )
        client = FakeClient()
        session_id = await _start(adapter, client)

        blocks: list[PromptContentBlock] = [
            acp.text_block('look'),
            acp.image_block(base64.b64encode(raw).decode(), 'image/png'),
        ]
        await adapter.prompt(prompt=blocks, session_id=session_id)

        history = adapter._sessions[session_id].history  # pyright: ignore[reportPrivateUsage]
        prompt_parts = [
            part
            for message in history
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        [user_prompt] = prompt_parts
        assert not isinstance(user_prompt.content, str)
        [image] = [item for item in user_prompt.content if isinstance(item, BinaryContent)]
        assert image.data == raw
        assert image.media_type == 'image/png'

    async def test_name_defaults_to_agent_name(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel(), name='my_agent'))
        response = await adapter.initialize(protocol_version=1)
        assert response.agent_info is not None
        assert response.agent_info.name == 'my_agent'

    async def test_new_session_ids_are_unique_with_empty_history(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        first = await adapter.new_session(cwd='.')
        second = await adapter.new_session(cwd='.')
        assert first.session_id != second.session_id


@dataclass
class _Workspace:
    cwd: str


def _workspace_agent() -> Agent[_Workspace, str]:
    """An agent whose only tool reports the workspace cwd carried in its deps."""
    agent = Agent(TestModel(), deps_type=_Workspace)

    @agent.tool
    def where(ctx: RunContext[_Workspace]) -> str:
        return ctx.deps.cwd

    return agent


class TestSessionConfig:
    def test_config_is_an_immutable_value(self) -> None:
        first = AcpSessionConfig(deps=_Workspace(cwd='/x'))
        second = AcpSessionConfig(deps=_Workspace(cwd='/x'))
        assert first == second  # dataclass value equality
        assert first.toolsets is None  # documented default
        with pytest.raises(FrozenInstanceError):
            first.deps = _Workspace(cwd='/y')  # pyright: ignore[reportAttributeAccessIssue]  # frozen

    async def test_factory_receives_setup_and_supplies_deps(self) -> None:
        seen: list[AcpSession] = []

        def make(session: AcpSession) -> AcpSessionConfig[_Workspace]:
            seen.append(session)
            return AcpSessionConfig(deps=_Workspace(cwd=session.cwd))

        # The factory's per-session deps must override this constructor fallback.
        adapter = PydanticAIACPAgent(_workspace_agent(), deps=_Workspace(cwd='/fallback'), session_config=make)
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1, client_capabilities=schema.ClientCapabilities())
        mcp = schema.McpServerStdio(name='fs', command='echo', args=[], env=[])
        session = await adapter.new_session(cwd='/projects/app', mcp_servers=[mcp])

        await adapter.prompt(prompt=[acp.text_block('where')], session_id=session.session_id)

        # The factory saw the client's full session setup, nothing silently dropped.
        assert seen[0].cwd == '/projects/app'
        assert [server.name for server in seen[0].mcp_servers] == ['fs']
        assert seen[0].client_capabilities is not None
        # ...and the deps it returned reached the run.
        [(_id, _status, raw_output)] = client.tool_completions()
        assert raw_output == '/projects/app'

    async def test_mcp_servers_without_a_session_config_are_rejected(self) -> None:
        # This adapter does not connect MCP servers itself; a session_config must turn them into
        # toolsets. Without one, a request carrying MCP servers is rejected rather than dropped.
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        adapter.on_connect(FakeClient())
        await adapter.initialize(protocol_version=1)
        mcp = schema.McpServerStdio(name='fs', command='echo', args=[], env=[])

        with pytest.raises(RequestError) as exc:
            await adapter.new_session(cwd='/projects/app', mcp_servers=[mcp])

        assert exc.value.data is not None
        assert 'mcp_servers' in str(exc.value.data)

    async def test_initialize_advertises_configured_mcp_capabilities(self) -> None:
        # A spec-following client only sends HTTP/SSE MCP servers when these are advertised
        # during initialize, so an embedder whose session_config connects them must say so.
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            Agent(TestModel()),
            session_config=lambda _session: AcpSessionConfig(deps=None),
            mcp_capabilities=schema.McpCapabilities(http=True, sse=True),
        )
        capabilities = (await adapter.initialize(protocol_version=1)).agent_capabilities
        assert capabilities is not None
        assert capabilities.mcp_capabilities == schema.McpCapabilities(http=True, sse=True)

    async def test_initialize_advertises_no_mcp_capabilities_by_default(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        capabilities = (await adapter.initialize(protocol_version=1)).agent_capabilities
        assert capabilities is not None
        assert capabilities.mcp_capabilities == schema.McpCapabilities(http=False, sse=False)

    def test_mcp_capabilities_without_a_session_config_are_rejected(self) -> None:
        # Advertising HTTP/SSE support without a session_config would invite server
        # definitions that `session/new` then rejects; fail at construction instead.
        with pytest.raises(ValueError, match='session_config'):
            PydanticAIACPAgent(Agent(TestModel()), mcp_capabilities=schema.McpCapabilities(http=True))

    async def test_two_sessions_get_independent_deps(self) -> None:
        adapter = PydanticAIACPAgent(
            _workspace_agent(),
            deps=_Workspace(cwd='/fallback'),
            session_config=lambda s: AcpSessionConfig(deps=_Workspace(cwd=s.cwd)),
        )
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        first = (await adapter.new_session(cwd='/a')).session_id
        second = (await adapter.new_session(cwd='/b')).session_id

        await adapter.prompt(prompt=[acp.text_block('where')], session_id=first)
        await adapter.prompt(prompt=[acp.text_block('where')], session_id=second)

        assert [raw_output for _id, _status, raw_output in client.tool_completions()] == ['/a', '/b']

    async def test_async_factory_is_awaited(self) -> None:
        async def make(session: AcpSession) -> AcpSessionConfig[_Workspace]:
            return AcpSessionConfig(deps=_Workspace(cwd=session.cwd))

        adapter = PydanticAIACPAgent(_workspace_agent(), deps=_Workspace(cwd='/fallback'), session_config=make)
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd='/async')

        await adapter.prompt(prompt=[acp.text_block('where')], session_id=session.session_id)

        [(_id, _status, raw_output)] = client.tool_completions()
        assert raw_output == '/async'

    async def test_session_toolsets_are_added_to_the_run(self) -> None:
        extra = FunctionToolset[None]()

        @extra.tool
        def extra_tool(ctx: RunContext[None]) -> str:
            return 'from-extra'

        adapter = PydanticAIACPAgent(
            Agent(TestModel()), session_config=lambda _s: AcpSessionConfig(deps=None, toolsets=[extra])
        )
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd='.')

        await adapter.prompt(prompt=[acp.text_block('go')], session_id=session.session_id)

        assert 'extra_tool' in {title for _id, title, _status in client.tool_starts()}

    async def test_without_factory_falls_back_to_constructor_deps(self) -> None:
        adapter = PydanticAIACPAgent(_workspace_agent(), deps=_Workspace(cwd='/fixed'))
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session_id = (await adapter.new_session(cwd='.')).session_id

        await adapter.prompt(prompt=[acp.text_block('where')], session_id=session_id)

        [(_id, _status, raw_output)] = client.tool_completions()
        assert raw_output == '/fixed'


class TestStreaming:
    async def test_text_is_streamed_then_turn_ends(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel(custom_output_text='hi there')))
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('hello')], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        assert set(client.kinds()) == {'agent_message_chunk'}  # streamed as one or more text chunks
        assert client.text() == 'hi there'  # chunks concatenate in order

    async def test_structured_output_is_delivered_as_final_message(self) -> None:
        class Weather(BaseModel):
            city: str
            sunny: bool

        adapter: PydanticAIACPAgent[None, Weather] = PydanticAIACPAgent(Agent(TestModel(), output_type=Weather))
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('weather?')], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        # The structured result is serialized to JSON and sent as a single final assistant message.
        assert client.kinds() == ['agent_message_chunk']
        payload = json.loads(client.text())
        assert set(payload) == {'city', 'sunny'}
        assert isinstance(payload['city'], str) and isinstance(payload['sunny'], bool)

    async def test_emit_event_covers_every_part_and_delta_branch(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())

        async def emit(event: AgentStreamEvent) -> None:
            await adapter._emit_event(turn, event)  # pyright: ignore[reportPrivateUsage]

        await emit(PartStartEvent(index=0, part=TextPart(content='hi')))
        await emit(PartStartEvent(index=0, part=TextPart(content='')))  # empty -> no update
        await emit(PartStartEvent(index=0, part=ThinkingPart(content='hmm')))
        await emit(PartStartEvent(index=0, part=ToolCallPart(tool_name='t', tool_call_id='c')))  # no update
        await emit(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='more')))
        await emit(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='think')))
        await emit(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{}')))  # no update

        assert client.events() == [
            ('agent_message_chunk', 'hi'),
            ('agent_thought_chunk', 'hmm'),
            ('agent_message_chunk', 'more'),
            ('agent_thought_chunk', 'think'),
        ]

    async def test_tool_result_failure_is_marked_failed(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        event = FunctionToolResultEvent(part=RetryPromptPart(content='bad args', tool_name='t', tool_call_id='c1'))

        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())
        await adapter._emit_event(turn, event)  # pyright: ignore[reportPrivateUsage]

        [update] = client.updates
        assert (getattr(update, 'session_update', ''), getattr(update, 'status', None)) == (
            'tool_call_update',
            'failed',
        )

    async def test_non_json_tool_output_does_not_crash_the_stream(self) -> None:
        class NotJson:
            def __str__(self) -> str:
                return 'stringified'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        event = FunctionToolResultEvent(part=ToolReturnPart(tool_name='t', content=NotJson(), tool_call_id='c1'))

        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())
        await adapter._emit_event(turn, event)  # pyright: ignore[reportPrivateUsage]

        [update] = client.updates
        assert getattr(update, 'status', None) == 'completed'
        assert getattr(update, 'raw_output', None) == 'stringified'

    async def test_non_utf8_bytes_tool_output_is_base64_encoded(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        event = FunctionToolResultEvent(part=ToolReturnPart(tool_name='t', content=b'\x80\x81\xff', tool_call_id='c1'))

        # Raw non-UTF-8 bytes must not crash the update; they are base64-encoded.
        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())
        await adapter._emit_event(turn, event)  # pyright: ignore[reportPrivateUsage]

        [update] = client.updates
        assert getattr(update, 'status', None) == 'completed'
        assert isinstance(getattr(update, 'raw_output', None), str)

    async def test_large_text_output_is_chunked_under_the_buffer_limit(self) -> None:
        big = 'x' * (MAX_TEXT_UPDATE_BYTES * 3 + 17)
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel(custom_output_text=big)))
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        chunks = [payload for kind, payload in client.events() if kind == 'agent_message_chunk']
        # Split into several updates, each within the byte budget, and losslessly reassembled.
        assert len(chunks) >= 4
        assert all(isinstance(chunk, str) and len(json.dumps(chunk)) - 2 <= MAX_TEXT_UPDATE_BYTES for chunk in chunks)
        assert ''.join(str(chunk) for chunk in chunks) == big

    async def test_prompt_response_reports_token_usage(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel(custom_output_text='hi')))
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        assert response.usage is not None
        assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0
        assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens

    async def test_usage_sums_across_approval_passes(self) -> None:
        # An approval pause splits the turn into two model passes; the reported usage must cover
        # both, not just the resume.
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient()  # default decider approves
        session_id = await _start(adapter, client)
        approval = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        # Baseline: the same final pass alone (a model that answers 'done' with no tool call).
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'done'

        baseline_adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            Agent(FunctionModel(stream_function=stream))
        )
        baseline_session = await _start(baseline_adapter, FakeClient())
        baseline = await baseline_adapter.prompt(prompt=[acp.text_block('go')], session_id=baseline_session)

        assert approval.usage is not None and baseline.usage is not None
        # Reporting only the resume pass would equal the baseline's output; the sum must also
        # include the first pass's tool-call output.
        assert approval.usage.output_tokens > baseline.usage.output_tokens


class TestStopReason:
    """`_finish_reason_to_stop_reason` maps a completed run's finish reason to the ACP stop reason."""

    @pytest.mark.parametrize(
        ('finish_reason', 'expected'),
        [
            ('length', 'max_tokens'),
            ('content_filter', 'refusal'),
            ('stop', 'end_turn'),
            ('tool_call', 'end_turn'),
            (None, 'end_turn'),
        ],
    )
    def test_finish_reason_maps_to_stop_reason(
        self, finish_reason: FinishReason | None, expected: schema.StopReason
    ) -> None:
        assert _finish_reason_to_stop_reason(finish_reason) == expected

    @pytest.mark.parametrize(
        ('message', 'expected'),
        [
            # The real wordings from pydantic-ai's UsageLimits checks.
            ('The next request would exceed the request_limit of 50', 'max_turn_requests'),
            ('The next tool call(s) would exceed the tool_calls_limit of 3 (tool_calls=4).', 'max_turn_requests'),
            ('Exceeded the output_tokens_limit of 5 (output_tokens=10)', 'max_tokens'),
            ('The next request would exceed the total_tokens_limit of 9 (total_tokens=10)', 'max_tokens'),
        ],
    )
    def test_usage_limit_maps_to_stop_reason(self, message: str, expected: schema.StopReason) -> None:
        assert _usage_limit_stop_reason(UsageLimitExceeded(message)) == expected

    async def test_request_limit_ends_the_turn_with_max_turn_requests(self) -> None:
        # A model that calls a tool on every request never finishes the turn, so pydantic-ai's
        # default request_limit (50) trips. ACP defines `max_turn_requests` for exactly this; it
        # must end the turn with that stop reason, not surface as a JSON-RPC error.
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
            yield {0: DeltaToolCall(name='spin', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain
        def spin() -> str:
            return 'again'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id, message_id='m1')

        assert response.stop_reason == 'max_turn_requests'
        # The raising run's messages are not retrievable, so the turn rolls back like a
        # cancellation: no committed history or usage.
        assert response.usage is None
        assert adapter._sessions[session_id].history == []  # pyright: ignore[reportPrivateUsage]

    async def test_configured_usage_limits_end_the_turn_with_max_turn_requests(self) -> None:
        request_count = 0

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
            nonlocal request_count
            request_count += 1
            yield {0: DeltaToolCall(name='spin', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain
        def spin() -> str:
            return 'again'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, usage_limits=UsageLimits(request_limit=2))
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id, message_id='m1')

        assert response.stop_reason == 'max_turn_requests'
        assert response.usage is None
        assert request_count == 2
        assert adapter._sessions[session_id].history == []  # pyright: ignore[reportPrivateUsage]

    async def test_default_usage_limits_allow_normal_tool_resume(self) -> None:
        request_count = 0

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            nonlocal request_count
            request_count += 1
            if _has_tool_return(messages):
                yield 'done'
            else:
                yield {0: DeltaToolCall(name='spin', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain
        def spin() -> str:
            return 'again'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id, message_id='m1')

        assert response.stop_reason == 'end_turn'
        assert response.usage is not None
        assert request_count == 2
        assert len(adapter._sessions[session_id].history) >= 2  # pyright: ignore[reportPrivateUsage]


class TestChunkText:
    """`chunk_text` bounds the JSON-escaped byte size of each streamed update (see `_serialize`)."""

    def test_ascii_splits_by_byte_budget(self) -> None:
        assert list(chunk_text('x' * 25, budget=10)) == ['x' * 10, 'x' * 10, 'x' * 5]

    def test_empty_text_yields_nothing(self) -> None:
        assert list(chunk_text('')) == []

    def test_non_ascii_stays_within_byte_budget_where_a_char_cap_would_not(self) -> None:
        # Each emoji escapes to a 12-byte surrogate pair under `ensure_ascii=True`, so a char-count
        # cap would let a chunk balloon ~12x and overrun the client's read buffer.
        text = '😀' * 1000
        chunks = list(chunk_text(text, budget=120))
        assert ''.join(chunks) == text
        assert all(len(json.dumps(chunk)) - 2 <= 120 for chunk in chunks)
        assert all(len(chunk) <= 10 for chunk in chunks)  # 120 budget / 12 bytes per emoji

    def test_exact_budget_fit_yields_a_single_chunk(self) -> None:
        # The boundary case for `if chunk and size + char_size > budget`: an exact fit must not
        # split, and one char over must spill into a second chunk -- never an empty trailing one.
        assert list(chunk_text('abcde', budget=5)) == ['abcde']
        assert list(chunk_text('abcdef', budget=5)) == ['abcde', 'f']

    def test_escaped_len_covers_every_character_class(self) -> None:
        assert _escaped_len('"') == 2 and _escaped_len('\n') == 2  # short escapes
        assert _escaped_len('\x00') == 6  # other control char -> \u00XX
        assert _escaped_len('a') == 1  # printable ASCII
        assert _escaped_len('é') == 6  # BMP non-ASCII -> \uXXXX
        assert _escaped_len('😀') == 12  # astral -> surrogate pair


class TestToolCalls:
    async def test_single_tool_call_emits_start_and_completion(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'Sunny in {city}'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('weather?')], session_id=session_id)

        # tool_call (start) precedes its tool_call_update (completion), then the final text.
        kinds = client.kinds()
        assert kinds.index('tool_call') < kinds.index('tool_call_update') < kinds.index('agent_message_chunk')
        # The start carries in_progress status and the raw input; completion carries the result, paired by id.
        [(start_id, title, start_status)] = client.tool_starts()
        [(done_id, done_status, raw_output)] = client.tool_completions()
        assert (title, start_status) == ('get_weather', 'in_progress')
        assert start_id == done_id and done_status == 'completed'
        assert raw_output == 'Sunny in a'  # TestModel fills string args with 'a'

    async def test_multiple_tools_are_all_surfaced(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def tool_a() -> str:
            return 'a'

        @agent.tool_plain
        def tool_b() -> str:
            return 'b'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        starts = client.tool_starts()
        completions = client.tool_completions()
        assert {title for _id, title, _status in starts} == {'tool_a', 'tool_b'}
        # Every start has exactly one matching completion, paired by tool_call_id, all completed.
        start_ids = {sid for sid, _t, _s in starts}
        done = {did: status for did, status, _out in completions}
        assert set(done) == start_ids
        assert all(status == 'completed' for status in done.values())


def _path_deleting_agent(deleted: list[str]) -> Agent[None, str]:
    """An agent that calls `delete_file(path=<latest user text>)`, requiring approval each turn."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if _has_tool_return(messages):
            yield 'done'
            return
        latest = _user_texts(messages)[-1]
        yield {0: DeltaToolCall(name='delete_file', json_args=json.dumps({'path': latest}))}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        deleted.append(path)
        return 'deleted'

    return agent


class TestPermission:
    async def test_allow_once_runs_the_tool(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'allow_once')
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        assert len(client.permission_requests) == 1
        assert client.permission_requests[0].title == 'delete_file'
        assert executed == ['x']

    async def test_reject_once_skips_the_tool(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'reject_once')
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        assert executed == []

    async def test_rejected_tool_starts_pending_then_failed(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'reject_once')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)

        # An approval-gated call starts as `pending` (never shown as running before approval) and,
        # when rejected, goes straight to `failed` without an intervening `in_progress`. One start,
        # no duplicate on resume, and the tool never ran.
        [(_id, title, start_status)] = client.tool_starts()
        assert (title, start_status) == ('delete_file', 'pending')
        tool_events = [event for event in client.events() if event[0] in ('tool_call', 'tool_call_update')]
        assert tool_events == [
            ('tool_call', 'delete_file'),
            ('tool_call_update', 'failed'),
        ]
        assert executed == []
        # The failed update carries the denial as its output, so the client can show why.
        [(_cid, _status, raw_output)] = client.tool_completions()
        assert 'Rejected by the client.' in str(raw_output)

    async def test_approved_tool_starts_pending_then_in_progress_then_completed(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'allow_once')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)

        # The lifecycle moves forward only: `pending` while awaiting approval, `in_progress`
        # once the client approves, then `completed`.
        [(_id, title, start_status)] = client.tool_starts()
        assert (title, start_status) == ('delete_file', 'pending')
        tool_events = [event for event in client.events() if event[0] in ('tool_call', 'tool_call_update')]
        assert tool_events == [
            ('tool_call', 'delete_file'),
            ('tool_call_update', 'in_progress'),
            ('tool_call_update', 'completed'),
        ]
        assert executed == ['x']

    async def test_approval_turn_with_a_store_persists_each_update_once(self) -> None:
        # The turn pauses for approval and resumes, accumulating updates across passes. The
        # persisted transcript must be the user's prompt plus what the client saw, with no
        # duplicated tool-call start.
        store = InMemorySessionStore()
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent([]), session_store=store)
        client = FakeClient(decider=lambda _call: 'allow_once')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)

        stored = await store.load(session_id)
        assert stored is not None
        assert stored.updates == [acp.update_user_message_text('delete'), *client.updates]

    async def test_allow_always_skips_the_prompt_on_later_turns(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'allow_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('delete again')], session_id=session_id)

        # The client is asked exactly once; the second turn auto-approves.
        assert len(client.permission_requests) == 1
        assert executed == ['x', 'x']

    async def test_reject_always_skips_the_prompt_on_later_turns(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'reject_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('delete again')], session_id=session_id)

        assert len(client.permission_requests) == 1
        assert executed == []

    async def test_always_allow_is_scoped_to_the_tool(self) -> None:
        ran: list[str] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if _has_tool_return(messages):
                yield 'done'
                return
            latest = _user_texts(messages)[-1]
            yield {0: DeltaToolCall(name='keep' if 'keep' in latest else 'other', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain(requires_approval=True)
        def keep() -> str:
            ran.append('keep')
            return 'kept'

        @agent.tool_plain(requires_approval=True)
        def other() -> str:
            ran.append('other')
            return 'done'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient(decider=lambda _call: 'allow_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('keep it')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('other now')], session_id=session_id)

        # Always-allow for `keep` must not auto-approve a different tool: `other` is still asked.
        assert len(client.permission_requests) == 2
        assert ran == ['keep', 'other']

    async def test_always_allow_is_scoped_to_the_exact_call_by_default(self) -> None:
        deleted: list[str] = []
        adapter = PydanticAIACPAgent(_path_deleting_agent(deleted))
        client = FakeClient(decider=lambda _call: 'allow_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('tmp.txt')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('.env')], session_id=session_id)

        # "Always allow delete_file(path='tmp.txt')" must NOT auto-approve delete_file(path='.env'):
        # the second call has a different scope, so the client is asked again.
        assert len(client.permission_requests) == 2
        assert deleted == ['tmp.txt', '.env']

    async def test_always_allow_survives_reordered_json_string_args(self) -> None:
        deleted: list[str] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if _has_tool_return(messages):
                yield 'done'
                return
            # The same logical call each turn, but the model serializes the argument keys in a
            # different order (as streaming providers do). The remembered "always allow" must still
            # match: the scope is keyed on the canonicalized arguments, not their raw text.
            first_turn = len(_user_texts(messages)) == 1
            args = '{"path": "x", "force": true}' if first_turn else '{"force": true, "path": "x"}'
            yield {0: DeltaToolCall(name='delete_file', json_args=args)}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain(requires_approval=True)
        def delete_file(path: str, force: bool) -> str:
            deleted.append(path)
            return 'deleted'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient(decider=lambda _call: 'allow_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('again')], session_id=session_id)

        # One prompt only: the reordered second call resolves to the same scope and auto-approves.
        assert len(client.permission_requests) == 1
        assert deleted == ['x', 'x']

    async def test_permission_policy_can_widen_scope_to_the_tool_name(self) -> None:
        deleted: list[str] = []
        adapter = PydanticAIACPAgent(_path_deleting_agent(deleted), permission_policy=lambda call: call.tool_name)
        client = FakeClient(decider=lambda _call: 'allow_always')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('tmp.txt')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('.env')], session_id=session_id)

        # Widening the scope to the tool name makes one "always allow" cover any later arguments.
        assert len(client.permission_requests) == 1
        assert deleted == ['tmp.txt', '.env']

    async def test_oversized_tool_output_is_truncated_in_the_update(self) -> None:
        big = 'y' * (MAX_RAW_FIELD_CHARS * 2)
        agent = Agent(TestModel())

        @agent.tool_plain
        def big_tool() -> str:
            return big

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        [(_id, status, raw_output)] = client.tool_completions()
        assert status == 'completed'
        # The whole payload would overrun one notification, so it is replaced with a bounded marker.
        serialized = json.dumps(raw_output)
        assert '"truncated": true' in serialized
        assert '"original_length"' in serialized
        # The marker is far smaller than the original payload (which was twice the cap).
        assert len(serialized) <= MAX_RAW_FIELD_CHARS + 1024

    async def test_always_decisions_are_isolated_per_session(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: 'allow_always')
        first = await _start(adapter, client)
        second = (await adapter.new_session(cwd='.')).session_id

        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=first)
        await adapter.prompt(prompt=[acp.text_block('delete')], session_id=second)

        # "Always allow" in the first session does not carry into the second.
        assert len(client.permission_requests) == 2

    async def test_mixed_approvals_in_one_turn(self) -> None:
        ran: list[str] = []
        model = _calls_tool_each_turn(
            DeltaToolCall(name='keep', json_args='{}'),
            DeltaToolCall(name='drop', json_args='{}'),
        )
        agent = Agent(model)

        @agent.tool_plain(requires_approval=True)
        def keep() -> str:
            ran.append('keep')
            return 'kept'

        @agent.tool_plain(requires_approval=True)
        def drop() -> str:  # pragma: no cover - rejected by the client, so never executed
            ran.append('drop')
            return 'dropped'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient(decider=lambda call: 'allow_once' if call.title == 'keep' else 'reject_once')
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        assert len(client.permission_requests) == 2
        assert ran == ['keep']  # only the approved tool executed

    async def test_cancel_while_a_permission_request_is_in_flight(self) -> None:
        # The spec's normal cancellation interleaving: the user hits stop while a permission
        # dialog is open, so session/cancel races the unanswered request. The turn must end
        # `cancelled` and the pending tool call must be driven to a terminal status.
        requested = asyncio.Event()

        class _BlockedPermissionClient(FakeClient):
            async def request_permission(
                self,
                session_id: str,
                tool_call: schema.ToolCallUpdate,
                options: list[schema.PermissionOption],
                **kwargs: object,
            ) -> schema.RequestPermissionResponse:
                requested.set()
                await asyncio.Event().wait()  # the dialog is never answered; cancel unwinds the turn
                raise AssertionError('unreachable')  # pragma: no cover

        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = _BlockedPermissionClient()
        session_id = await _start(adapter, client)

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(requested.wait(), timeout=5)
        await adapter.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'
        assert executed == []  # the tool never ran
        [(start_id, _title, start_status)] = client.tool_starts()
        assert start_status == 'pending'  # announced as awaiting approval
        assert (start_id, 'failed') in [(cid, status) for cid, status, _out in client.tool_completions()]

    async def test_cancel_during_permission_cancels_the_turn(self) -> None:
        executed: list[str] = []
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent(executed))
        client = FakeClient(decider=lambda _call: None)  # dismissed dialog -> cancelled outcome
        session_id = await _start(adapter, client)

        response = await adapter.prompt(prompt=[acp.text_block('delete')], session_id=session_id, message_id='msg-1')

        assert response.stop_reason == 'cancelled'
        assert executed == []
        # The tool the dismissed dialog was for is closed out as failed, not left pending forever.
        assert 'failed' in [status for _id, status, _out in client.tool_completions()]

    async def test_external_tool_calls_are_rejected(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        requests = DeferredToolRequests(calls=[ToolCallPart(tool_name='run_remote', tool_call_id='c1')])
        state = SessionState(session_id='sid', config=AcpSessionConfig(deps=None), cwd='.')
        turn = _TurnState(conn=FakeClient(), session_id='sid', cwd='.', approval_names=frozenset())
        with pytest.raises(RequestError) as exc:
            await adapter._resolve_approvals(turn, state, requests)  # pyright: ignore[reportPrivateUsage]
        assert exc.value.data is not None
        assert 'external tool execution' in str(exc.value.data)
        assert 'run_remote' in str(exc.value.data)

    def test_approval_names_come_from_function_toolsets_only(self) -> None:
        # A non-`FunctionToolset` session toolset cannot expose `requires_approval` without a live
        # run context, so it contributes nothing and its calls start `in_progress`.
        from pydantic_ai.toolsets import CombinedToolset

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_approval_agent([]))
        config: AcpSessionConfig[None] = AcpSessionConfig(deps=None, toolsets=[CombinedToolset([])])

        names = adapter._approval_tool_names(config)  # pyright: ignore[reportPrivateUsage]

        assert names == frozenset({'delete_file'})


class TestCancellation:
    async def test_cancel_stops_an_in_flight_turn(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        turn = asyncio.ensure_future(
            adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id, message_id='msg-1')
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        await adapter.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'

    async def test_cancel_closes_out_in_flight_tool_calls(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        await adapter.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'
        # The tool was announced `in_progress`; cancelling drives it to a terminal `failed` so the
        # client stops rendering it as running.
        [(start_id, _title, start_status)] = client.tool_starts()
        assert start_status == 'in_progress'
        assert (start_id, 'failed') in [(cid, status) for cid, status, _out in client.tool_completions()]

    async def test_fail_outstanding_tool_calls_closes_only_unfinished(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())
        turn.started.update({'c1', 'c2'})
        turn.resulted.add('c2')  # c2 already has a terminal result; only c1 is still outstanding

        await adapter._fail_outstanding_tool_calls(turn)  # pyright: ignore[reportPrivateUsage]

        assert [(cid, status) for cid, status, _out in client.tool_completions()] == [('c1', 'failed')]

    async def test_fail_outstanding_tool_calls_with_none_outstanding_is_a_noop(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        turn = _TurnState(conn=client, session_id='sid', cwd='.', approval_names=frozenset())

        await adapter._fail_outstanding_tool_calls(turn)  # pyright: ignore[reportPrivateUsage]

        assert client.tool_completions() == []

    async def test_cancelled_turn_with_a_store_persists_nothing(self) -> None:
        store = InMemorySessionStore()
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, session_store=store)
        client = FakeClient()
        session_id = await _start(adapter, client)
        before = await store.load(session_id)

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        await adapter.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'
        # The store still holds the pre-turn (empty) snapshot: a cancelled turn commits nothing.
        assert before is not None and before.messages == [] and before.updates == []
        assert await store.load(session_id) == before

    async def test_cancel_during_terminal_create_kills_the_terminal_end_to_end(self) -> None:
        # The full leak path: session/cancel raw-cancels the turn (which pierces anyio shields)
        # while the editor-native terminal create is in flight. The client started the command
        # regardless, so the late-learned terminal must still be killed and released.
        client = RecordingClient(block_create=True)
        agent = Agent(TestModel(call_tools=['run_command']))
        toolset = AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws')
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            agent, session_config=lambda _session: AcpSessionConfig(deps=None, toolsets=[toolset])
        )
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd='/ws')

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('run')], session_id=session.session_id))
        await asyncio.wait_for(client.create_event.wait(), timeout=5)
        await adapter.cancel(session_id=session.session_id)
        client.release_create.set()  # the client answers the create only after the cancellation
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'
        assert client.killed == ['term-1']
        assert client.released == ['term-1']

    async def test_double_cancel_is_idempotent(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        await adapter.cancel(session_id=session_id)
        await adapter.cancel(session_id=session_id)  # a repeated cancel must not raise or wedge the turn
        response = await asyncio.wait_for(turn, timeout=5)
        assert response.stop_reason == 'cancelled'

    async def test_cancel_unknown_session_is_a_noop(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        adapter.on_connect(FakeClient())
        await adapter.cancel(session_id='no-such-session')  # must not raise

    async def test_cancel_idle_session_is_a_noop(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        session_id = await _start(adapter, client)
        # An existing session with no in-flight turn: cancel must not raise and leaves no turn.
        await adapter.cancel(session_id=session_id)
        assert adapter._sessions[session_id].active_turn is None  # pyright: ignore[reportPrivateUsage]

    async def test_teardown_cancellation_during_a_session_cancel_still_propagates(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        prompt_task = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        # A session/cancel and connection teardown race: the prompt handler itself is cancelled
        # while the turn it just cancelled is still unwinding. Its own cancellation must win --
        # answering a request the connection already abandoned would deadlock its shutdown.
        await adapter.cancel(session_id=session_id)
        prompt_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(prompt_task, timeout=5)

    async def test_teardown_cancellation_racing_a_cancelled_permission_dialog_propagates(self) -> None:
        executed: list[str] = []
        agent = _approval_agent(executed)
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient(decider=lambda _call: None)  # the user dismisses the dialog
        session_id = await _start(adapter, client)

        prompt_task = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        # One tick suffices: prompt() sets active_turn before its first suspension point (an
        # uncontended lock acquire does not yield).
        await asyncio.sleep(0)
        turn = adapter._sessions[session_id].active_turn  # pyright: ignore[reportPrivateUsage]
        assert turn is not None, 'the prompt never started its turn'
        # Tear the prompt handler down in the same tick the dismissed dialog ends the turn: the
        # turn's internal rollback signal must not escape (or replace) the handler's cancellation.
        turn.add_done_callback(lambda _turn: prompt_task.cancel())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(prompt_task, timeout=5)
        assert executed == []

    async def test_outer_cancellation_propagates_and_stops_the_inner_turn(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        tool_finished = False
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            nonlocal tool_finished
            started.set()
            await release.wait()
            tool_finished = True  # pragma: no cover - cancelled before returning
            return 'done'  # pragma: no cover - cancelled before returning

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        # Cancel the prompt coroutine itself (not via adapter.cancel) - e.g. connection teardown.
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        # The inner turn was stopped rather than orphaned, so the tool never completed.
        await asyncio.sleep(0)
        assert tool_finished is False


class TestMultiTurn:
    async def test_session_history_returns_committed_history_only(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        session_id = await _start(adapter, client)

        assert adapter.session_history(session_id) == []
        assert adapter.session_history('no-such-session') is None

        await adapter.prompt(prompt=[acp.text_block('first')], session_id=session_id)
        first_history = adapter.session_history(session_id)
        assert first_history is not None
        assert _user_texts(first_history) == ['first']

        await adapter.prompt(prompt=[acp.text_block('second')], session_id=session_id)
        second_history = adapter.session_history(session_id)
        assert second_history is not None
        assert len(second_history) > len(first_history)
        assert _user_texts(second_history) == ['first', 'second']

        started = asyncio.Event()
        release = asyncio.Event()
        cancel_agent = Agent(TestModel())

        @cancel_agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled before returning

        cancel_adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(cancel_agent)
        cancel_session_id = await _start(cancel_adapter, FakeClient())
        before_cancel = cancel_adapter.session_history(cancel_session_id)
        turn = asyncio.ensure_future(
            cancel_adapter.prompt(prompt=[acp.text_block('cancel me')], session_id=cancel_session_id)
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        await cancel_adapter.cancel(session_id=cancel_session_id)
        response = await asyncio.wait_for(turn, timeout=5)

        assert response.stop_reason == 'cancelled'
        assert cancel_adapter.session_history(cancel_session_id) == before_cancel

    async def test_history_persists_across_turns(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('first')], session_id=session_id)
        first_len = len(adapter._sessions[session_id].history)  # pyright: ignore[reportPrivateUsage]
        await adapter.prompt(prompt=[acp.text_block('second')], session_id=session_id)
        second_len = len(adapter._sessions[session_id].history)  # pyright: ignore[reportPrivateUsage]

        assert first_len >= 2
        assert second_len > first_len  # the second turn appended to, not replaced, history

    async def test_sessions_are_isolated(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        first = await _start(adapter, client)
        second = (await adapter.new_session(cwd='.')).session_id

        await adapter.prompt(prompt=[acp.text_block('hi')], session_id=first)

        assert len(adapter._sessions[first].history) >= 2  # pyright: ignore[reportPrivateUsage]
        assert adapter._sessions[second].history == []  # pyright: ignore[reportPrivateUsage]

    async def test_concurrent_turns_on_one_session_are_serialized(self) -> None:
        # A tool that records how many turns are running at once; the lock should keep it at 1.
        active = 0
        max_active = 0
        agent = Agent(_calls_tool_each_turn(DeltaToolCall(name='work', json_args='{}')))

        @agent.tool_plain
        async def work() -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return 'ok'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        responses = await asyncio.gather(
            adapter.prompt(prompt=[acp.text_block('one')], session_id=session_id),
            adapter.prompt(prompt=[acp.text_block('two')], session_id=session_id),
        )

        assert [r.stop_reason for r in responses] == ['end_turn', 'end_turn']
        assert max_active == 1  # the two turns never overlapped
        assert _user_texts(adapter._sessions[session_id].history) == ['one', 'two']  # pyright: ignore[reportPrivateUsage]

    async def test_second_turn_receives_prior_history(self) -> None:
        seen_user_prompts: list[list[str]] = []

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            seen_user_prompts.append(_user_texts(messages))
            yield 'ok'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(FunctionModel(stream_function=stream)))
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('first')], session_id=session_id)
        await adapter.prompt(prompt=[acp.text_block('second')], session_id=session_id)

        # The model's first turn saw only 'first'; the second turn saw the full history.
        assert seen_user_prompts[0] == ['first']
        assert seen_user_prompts[1] == ['first', 'second']

    async def test_concurrent_turns_on_distinct_sessions(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        sessions = [(await adapter.new_session(cwd='.')).session_id for _ in range(4)]

        await asyncio.gather(*(adapter.prompt(prompt=[acp.text_block('hi')], session_id=s) for s in sessions))

        for s in sessions:
            assert len(adapter._sessions[s].history) >= 2  # pyright: ignore[reportPrivateUsage]


class TestSessionClose:
    async def test_close_discards_session_state(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        client = FakeClient()
        session_id = await _start(adapter, client)
        await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session_id)

        await adapter.close_session(session_id=session_id)

        assert session_id not in adapter._sessions  # pyright: ignore[reportPrivateUsage]
        # A prompt after close is rejected like any unknown session.
        with pytest.raises(RequestError):
            await adapter.prompt(prompt=[acp.text_block('again')], session_id=session_id)

    async def test_close_unknown_session_is_a_noop(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        adapter.on_connect(FakeClient())
        assert await adapter.close_session(session_id='missing') is not None  # does not raise

    async def test_teardown_cancellation_of_close_while_the_turn_unwinds_propagates(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        hold_unwind = asyncio.Event()
        unwinding = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            try:
                await release.wait()
                return 'done'  # pragma: no cover - cancelled by close
            finally:
                unwinding.set()
                await hold_unwind.wait()  # keep the turn's unwind in flight

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)
        prompt_task = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)

        close_task = asyncio.ensure_future(adapter.close_session(session_id=session_id))
        await asyncio.wait_for(unwinding.wait(), timeout=5)
        # The connection tears the close handler down while it waits for the turn's unwind; the
        # handler's own cancellation must propagate, not be mistaken for the turn's.
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=5)

        hold_unwind.set()
        response = await asyncio.wait_for(prompt_task, timeout=5)
        assert response.stop_reason == 'cancelled'

    async def test_queued_prompt_is_rejected_when_the_session_closes_mid_wait(self) -> None:
        store = InMemorySessionStore()
        started = asyncio.Event()
        release = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return 'done'  # pragma: no cover - cancelled by close

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, session_store=store)
        client = FakeClient()
        session_id = await _start(adapter, client)
        snapshot = await store.load(session_id)

        first = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('one')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        # A second prompt queues on the session's turn lock behind the in-flight first turn.
        queued = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('two')], session_id=session_id))
        await asyncio.sleep(0)
        await adapter.close_session(session_id=session_id)

        assert (await asyncio.wait_for(first, timeout=5)).stop_reason == 'cancelled'
        # The queued prompt must not run against the discarded session state: it would be
        # uncancellable (cancel no longer finds it) and would persist over the closed session.
        with pytest.raises(RequestError) as excinfo:
            await asyncio.wait_for(queued, timeout=5)
        assert excinfo.value.code == -32602
        assert await store.load(session_id) == snapshot

    async def test_close_awaits_the_in_flight_turn_before_returning(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        unwound = asyncio.Event()
        agent = Agent(TestModel())

        @agent.tool_plain
        async def slow_tool() -> str:
            started.set()
            try:
                await release.wait()  # released only by cancellation
                return 'done'  # pragma: no cover - cancelled by close
            finally:
                unwound.set()

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)
        turn = asyncio.ensure_future(adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id))
        await asyncio.wait_for(started.wait(), timeout=5)

        await adapter.close_session(session_id=session_id)

        # close awaited the cancelled turn: the tool's cancellation has fully unwound (its `finally`
        # ran) before close returned, rather than racing the client's close response.
        assert unwound.is_set()
        response = await asyncio.wait_for(turn, timeout=5)

        # Closing mid-turn ends the prompt gracefully as cancelled and drops the session.
        assert response.stop_reason == 'cancelled'
        assert session_id not in adapter._sessions  # pyright: ignore[reportPrivateUsage]


class TestErrors:
    async def test_prompt_for_unknown_session_raises_invalid_params(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        adapter.on_connect(FakeClient())
        with pytest.raises(RequestError) as excinfo:
            await adapter.prompt(prompt=[acp.text_block('hi')], session_id='missing')
        assert excinfo.value.code == -32602  # invalid_params

    async def test_empty_prompt_is_handled(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel(custom_output_text='ok')))
        client = FakeClient()
        session_id = await _start(adapter, client)

        # A prompt with no content blocks runs the agent with empty user input rather than erroring.
        response = await adapter.prompt(prompt=[], session_id=session_id)

        assert response.stop_reason == 'end_turn'
        assert client.text() == 'ok'

    async def test_agent_error_propagates_and_cleans_up(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def boom() -> str:
            raise RuntimeError('tool exploded')

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        with pytest.raises(RuntimeError, match='tool exploded'):
            await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)
        # The failed turn is cleared so the session is usable again.
        assert adapter._sessions[session_id].active_turn is None  # pyright: ignore[reportPrivateUsage]
        # The announced tool call was closed out, so the client does not keep rendering it as
        # running alongside the turn's error.
        [(start_id, _title, _status)] = client.tool_starts()
        assert (start_id, 'failed') in [(cid, status) for cid, status, _out in client.tool_completions()]


class TestUnsupportedMethods:
    async def test_authenticate_and_ext_notification_are_noops(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        assert await adapter.authenticate(method_id='x') is None
        assert await adapter.ext_notification(method='x', params={}) is None

    async def test_optional_methods_raise_method_not_found(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        calls = [
            adapter.load_session(cwd='.', session_id='s'),
            adapter.list_sessions(),
            adapter.fork_session(cwd='.', session_id='s'),
            adapter.resume_session(cwd='.', session_id='s'),
            adapter.set_session_mode(mode_id='m', session_id='s'),
            adapter.ext_method(method='x', params={}),
        ]
        for call in calls:
            with pytest.raises(RequestError) as excinfo:
                await call
            assert excinfo.value.code == -32601  # method_not_found for an unsupported method


class TestEntryPoints:
    async def test_run_acp_stdio_serves_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[object] = []
        unstable: list[object] = []

        async def fake_run_agent(agent: object, **kwargs: object) -> None:
            served.append(agent)
            unstable.append(kwargs.get('use_unstable_protocol'))

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        await run_acp_stdio(Agent(TestModel()))
        assert isinstance(served[0], PydanticAIACPAgent)
        # Unstable routing must be on, or `session/close` is rejected by the SDK router before
        # reaching the adapter.
        assert unstable == [True]

    async def test_run_acp_stdio_forwards_session_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[PydanticAIACPAgent[_Workspace, str]] = []

        async def fake_run_agent(agent: PydanticAIACPAgent[_Workspace, str], **kwargs: object) -> None:
            served.append(agent)

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        await run_acp_stdio(
            _workspace_agent(),
            deps=_Workspace(cwd='/fallback'),
            session_config=lambda s: AcpSessionConfig(deps=_Workspace(cwd=s.cwd)),
            permission_policy=lambda call: call.tool_name,
        )

        # The helper handed the workspace factory to the adapter: a run uses the per-session deps.
        adapter = served[0]
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd='/wired')
        await adapter.prompt(prompt=[acp.text_block('where')], session_id=session.session_id)

        assert [raw_output for _id, _status, raw_output in client.tool_completions()] == ['/wired']

    async def test_run_acp_stdio_forwards_usage_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[PydanticAIACPAgent[None, str]] = []

        async def fake_run_agent(agent: PydanticAIACPAgent[None, str], **kwargs: object) -> None:
            served.append(agent)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
            yield {0: DeltaToolCall(name='spin', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream))

        @agent.tool_plain
        def spin() -> str:
            return 'again'

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        await run_acp_stdio(agent, usage_limits=UsageLimits(request_limit=2))

        # The helper handed the limits to the adapter: an endlessly tool-calling run is cut off.
        adapter = served[0]
        session_id = await _start(adapter, FakeClient())
        response = await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id, message_id='m1')
        assert response.stop_reason == 'max_turn_requests'

    async def test_run_acp_stdio_forwards_model_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[PydanticAIACPAgent[None, str]] = []

        async def fake_run_agent(agent: PydanticAIACPAgent[None, str], **kwargs: object) -> None:
            served.append(agent)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'resolved model'

        seen: list[str] = []

        def resolve(model_id: str) -> FunctionModel:
            seen.append(model_id)
            return FunctionModel(stream_function=stream)

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        await run_acp_stdio(Agent(TestModel()), models=['test', 'host:custom'], model_resolver=resolve)

        # The helper handed the resolver to the adapter: a selected model id goes through it.
        adapter = served[0]
        session_id = await _start(adapter, FakeClient())
        await adapter.set_config_option(config_id='model', value='host:custom', session_id=session_id)
        response = await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session_id)
        assert response.stop_reason == 'end_turn'
        assert seen == ['host:custom']

    def test_run_acp_stdio_sync_serves_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[object] = []

        async def fake_run_agent(agent: object, **kwargs: object) -> None:
            served.append(agent)

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        run_acp_stdio_sync(Agent(TestModel()))
        assert isinstance(served[0], PydanticAIACPAgent)

    def test_run_acp_stdio_sync_forwards_model_resolver_and_usage_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[PydanticAIACPAgent[None, str]] = []

        async def fake_run_agent(agent: PydanticAIACPAgent[None, str], **kwargs: object) -> None:
            served.append(agent)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
            yield {0: DeltaToolCall(name='spin', json_args='{}')}

        seen: list[str] = []

        def resolve(model_id: str) -> FunctionModel:
            seen.append(model_id)
            return FunctionModel(stream_function=stream)

        agent = Agent(TestModel())

        @agent.tool_plain
        def spin() -> str:
            return 'again'

        monkeypatch.setattr(acp, 'run_agent', fake_run_agent)
        run_acp_stdio_sync(
            agent,
            models=['test', 'host:custom'],
            model_resolver=resolve,
            usage_limits=UsageLimits(request_limit=2),
        )

        # One drive proves both: the selected id goes through the resolver, whose endlessly
        # tool-calling model is then cut off by the forwarded limits.
        async def drive() -> schema.PromptResponse:
            adapter = served[0]
            session_id = await _start(adapter, FakeClient())
            await adapter.set_config_option(config_id='model', value='host:custom', session_id=session_id)
            return await adapter.prompt(prompt=[acp.text_block('go')], session_id=session_id)

        response = asyncio.run(drive())
        assert seen == ['host:custom']
        assert response.stop_reason == 'max_turn_requests'

    async def test_end_to_end_over_stdio(self) -> None:
        """Drive the adapter as a real subprocess over ACP stdio, as a TUI client would."""
        client = FakeClient()
        script = Path(__file__).parent / '_demo_agent.py'
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            response = await conn.prompt(
                session_id=session.session_id, prompt=[acp.text_block('weather?')], message_id=uuid4().hex
            )

        assert response.stop_reason == 'end_turn'
        kinds = client.kinds()
        # Ordered over the real wire: the tool starts and completes before the final text.
        assert kinds[0] == 'tool_call'
        assert kinds.index('tool_call_update') < kinds.index('agent_message_chunk')
        # The streamed text is the model's JSON-encoded tool result, delivered intact.
        assert 'Sunny in a' in client.text()

    async def test_end_to_end_permission_over_stdio(self) -> None:
        """A tool requiring approval round-trips a real permission request over stdio."""
        client = FakeClient(decider=lambda _call: 'allow_once')
        script = Path(__file__).parent / '_demo_approval_agent.py'
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            response = await conn.prompt(
                session_id=session.session_id, prompt=[acp.text_block('delete')], message_id=uuid4().hex
            )

        assert response.stop_reason == 'end_turn'
        assert len(client.permission_requests) == 1
        assert client.permission_requests[0].title == 'delete_file'

    async def test_cancel_notification_over_stdio(self) -> None:
        """A real `session/cancel` notification mid-prompt yields a cancelled stop reason over stdio."""
        client = FakeClient()
        script = Path(__file__).parent / '_demo_slow_agent.py'
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            prompt = asyncio.ensure_future(
                conn.prompt(session_id=session.session_id, prompt=[acp.text_block('go')], message_id=uuid4().hex)
            )
            await asyncio.sleep(0.5)  # let the slow tool start
            await conn.cancel(session_id=session.session_id)
            response = await asyncio.wait_for(prompt, timeout=10)

        assert response.stop_reason == 'cancelled'

    async def test_large_output_over_stdio(self) -> None:
        """A large streamed message survives the default stdio buffer intact (chunked, not truncated)."""
        client = FakeClient()
        script = Path(__file__).parent / '_demo_large_agent.py'
        # No custom transport buffer: the adapter must chunk so the default client reader (64 KiB)
        # is never overrun. Raising the buffer here would mask the real-client failure mode.
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block('go')], message_id=uuid4().hex)

        assert client.text() == 'x' * 200_000

    async def test_model_config_and_unstable_close_route_over_stdio(self) -> None:
        """Model config updates and unstable `session/close` reach the handler over a real wire."""
        client = FakeClient()
        script = Path(__file__).parent / '_demo_models_agent.py'
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            model_result = await conn.set_config_option(
                config_id='model', value='openai:gpt-4o', session_id=session.session_id
            )
            await conn.close_session(session_id=session.session_id)

        [option] = model_result.config_options
        assert isinstance(option, schema.SessionConfigOptionSelect)
        assert option.current_value == 'openai:gpt-4o'

    async def test_native_toolsets_route_over_stdio(self) -> None:
        """Client-backed fs/terminal tools reach the real client over stdio when mounted per session."""
        client = RecordingClient(files={'notes.txt': 'hello'}, output='hi')
        capabilities = schema.ClientCapabilities(
            fs=schema.FileSystemCapabilities(read_text_file=True, write_text_file=True), terminal=True
        )
        script = Path(__file__).parent / '_demo_native_agent.py'
        async with acp.spawn_agent_process(client, sys.executable, str(script)) as (conn, _proc):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION, client_capabilities=capabilities)
            session = await conn.new_session(cwd='.', mcp_servers=[])
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block('go')], message_id=uuid4().hex)

        # The agent's read_file/run_command calls were served by the client over the wire.
        assert ('notes.txt', session.session_id) in client.reads
        assert any(command == 'echo hi' for command, _cwd in client.created)


class TestExperimentalStatus:
    def test_importing_acp_warns_experimental(self) -> None:
        # Lives here (not test_packaging.py) so base installs without the `acp` SDK never import it.
        module = importlib.import_module('pydantic_ai_harness.experimental.acp')
        with pytest.warns(HarnessExperimentalWarning, match='acp'):
            importlib.reload(module)


def _tool_call(name: str, args: str | dict[str, Any] | None) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args, tool_call_id='call-1')


class TestDefaultCodingPresenter:
    """Unit tests for the mapping from a recognized tool call to its ACP presentation."""

    def test_handler_names_match_the_filesystem_and_shell_tools(self) -> None:
        # Recognition couples to tool names, so a rename in those capabilities would silently
        # degrade rich rendering to generic JSON. This fails loudly instead.
        fs_tools = set(FileSystem[None](root_dir='.').get_toolset().tools)
        shell_tools = set(Shell[None](cwd='.').get_toolset().tools)
        assert set(_HANDLERS) <= fs_tools | shell_tools

    def test_edit_file_yields_edit_kind_location_and_diff(self) -> None:
        presentation = default_coding_presenter(
            _tool_call('edit_file', {'path': 'a.py', 'old_text': 'x', 'new_text': 'y'})
        )
        assert presentation is not None
        assert presentation.kind == 'edit'
        assert [loc.path for loc in presentation.locations] == ['a.py']
        [diff] = presentation.content
        assert (diff.path, diff.old_text, diff.new_text) == ('a.py', 'x', 'y')

    def test_write_file_yields_addition_diff_without_old_text(self) -> None:
        presentation = default_coding_presenter(_tool_call('write_file', {'path': 'a.py', 'content': 'hello'}))
        assert presentation is not None
        assert presentation.kind == 'edit'
        [diff] = presentation.content
        assert (diff.path, diff.old_text, diff.new_text) == ('a.py', None, 'hello')

    def test_read_file_yields_read_kind_and_location(self) -> None:
        presentation = default_coding_presenter(_tool_call('read_file', {'path': 'a.py'}))
        assert presentation is not None
        assert presentation.kind == 'read'
        assert [loc.path for loc in presentation.locations] == ['a.py']
        assert presentation.content == ()

    def test_search_with_path_locates_it(self) -> None:
        presentation = default_coding_presenter(_tool_call('list_directory', {'path': 'src'}))
        assert presentation is not None
        assert presentation.kind == 'search'
        assert [loc.path for loc in presentation.locations] == ['src']

    def test_search_without_path_has_no_location(self) -> None:
        presentation = default_coding_presenter(_tool_call('search_files', {'pattern': 'TODO'}))
        assert presentation is not None
        assert presentation.kind == 'search'
        assert presentation.locations == ()

    def test_create_directory_yields_other_kind(self) -> None:
        presentation = default_coding_presenter(_tool_call('create_directory', {'path': 'pkg'}))
        assert presentation is not None
        assert presentation.kind == 'other'
        assert [loc.path for loc in presentation.locations] == ['pkg']

    def test_run_command_yields_execute_kind_without_location(self) -> None:
        presentation = default_coding_presenter(_tool_call('run_command', {'command': 'ls'}))
        assert presentation is not None
        assert presentation.kind == 'execute'
        assert presentation.locations == ()

    def test_unknown_tool_returns_none(self) -> None:
        assert default_coding_presenter(_tool_call('get_weather', {'city': 'a'})) is None

    def test_recognized_name_with_missing_args_returns_none(self) -> None:
        # `edit_file` shape requires path/old_text/new_text; a partial call falls back to generic.
        assert default_coding_presenter(_tool_call('edit_file', {'path': 'a.py'})) is None

    def test_recognized_name_with_missing_path_returns_none(self) -> None:
        assert default_coding_presenter(_tool_call('read_file', {})) is None
        assert default_coding_presenter(_tool_call('create_directory', {})) is None
        assert default_coding_presenter(_tool_call('write_file', {'path': 'a.py'})) is None

    def test_non_string_argument_returns_none(self) -> None:
        # A recognized name whose argument is the wrong type is not mis-rendered.
        assert default_coding_presenter(_tool_call('read_file', {'path': 123})) is None

    def test_unparseable_args_return_none(self) -> None:
        # Malformed string args become a sentinel dict with no `path`, so the call falls back.
        assert default_coding_presenter(_tool_call('read_file', 'not json')) is None

    def test_presentation_is_immutable(self) -> None:
        presentation = ToolCallPresentation(kind='read')
        with pytest.raises(FrozenInstanceError):
            presentation.kind = 'edit'  # pyright: ignore[reportAttributeAccessIssue]


def _starts(client: FakeClient) -> list[object]:
    return [u for u in client.updates if getattr(u, 'session_update', '') == 'tool_call']


def _edit_agent_each_turn() -> Agent[None, str]:
    """An agent that calls `edit_file(path, old_text, new_text)` once per turn, requiring approval."""
    delta = DeltaToolCall(name='edit_file', json_args=json.dumps({'path': 'a.py', 'old_text': 'x', 'new_text': 'y'}))
    agent = Agent(_calls_tool_each_turn(delta))

    @agent.tool_plain(requires_approval=True)
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        return 'edited'

    return agent


class TestToolCallPresentationIntegration:
    """The adapter applies the presenter to tool-call starts and permission requests."""

    async def test_edit_call_start_carries_rich_presentation(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def edit_file(path: str, old_text: str, new_text: str) -> str:
            return 'edited'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('edit it')], session_id=session_id)

        [start] = _starts(client)
        assert getattr(start, 'kind') == 'edit'
        assert [loc.path for loc in getattr(start, 'locations')] == ['a']
        assert getattr(start, 'content') is not None and len(getattr(start, 'content')) == 1

    async def test_unrecognized_tool_start_has_no_kind(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return 'sunny'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('weather')], session_id=session_id)

        [start] = _starts(client)
        # Falls back to generic rendering: title is the tool name, no kind/locations/content.
        assert getattr(start, 'title') == 'get_weather'
        assert getattr(start, 'kind') is None
        assert getattr(start, 'locations') is None
        assert getattr(start, 'content') is None

    async def test_custom_presenter_overrides_the_default(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return 'sunny'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
            agent, tool_presenter=lambda _call: ToolCallPresentation(kind='fetch', title='Weather')
        )
        client = FakeClient()
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('weather')], session_id=session_id)

        [start] = _starts(client)
        assert getattr(start, 'kind') == 'fetch'
        assert getattr(start, 'title') == 'Weather'

    async def test_permission_request_uses_presenter_kind_not_hardcoded_execute(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(_edit_agent_each_turn())
        client = FakeClient(decider=lambda _call: 'allow_once')
        session_id = await _start(adapter, client)

        await adapter.prompt(prompt=[acp.text_block('edit')], session_id=session_id)

        [request] = client.permission_requests
        assert request.kind == 'edit'
        assert request.content is not None and len(request.content) == 1


class TestPresenterArgValidation:
    """Recognized names still fall back to generic rendering when the argument shape is wrong."""

    def test_execute_requires_a_command(self) -> None:
        assert default_coding_presenter(_tool_call('run_command', {'foo': 'bar'})) is None
        assert default_coding_presenter(_tool_call('start_command', {})) is None

    def test_command_ref_requires_a_command_id(self) -> None:
        assert default_coding_presenter(_tool_call('check_command', {'command_id': 'abc'})) is not None
        assert default_coding_presenter(_tool_call('check_command', {})) is None
        assert default_coding_presenter(_tool_call('stop_command', {'command': 'ls'})) is None

    def test_grep_requires_a_pattern(self) -> None:
        assert default_coding_presenter(_tool_call('search_files', {'junk': 'x'})) is None
        assert default_coding_presenter(_tool_call('find_files', {})) is None

    def test_list_directory_matches_by_name_alone(self) -> None:
        # Its only argument is optional, so it cannot be arg-validated -- documented exception.
        presentation = default_coding_presenter(_tool_call('list_directory', {'unrelated': 1}))
        assert presentation is not None and presentation.kind == 'search'

    def test_empty_path_falls_back(self) -> None:
        assert default_coding_presenter(_tool_call('read_file', {'path': ''})) is None
        assert default_coding_presenter(_tool_call('create_directory', {'path': ''})) is None

    def test_empty_old_text_edit_falls_back(self) -> None:
        assert (
            default_coding_presenter(_tool_call('edit_file', {'path': 'a.py', 'old_text': '', 'new_text': 'y'})) is None
        )

    def test_empty_replacement_and_empty_write_are_still_recognized(self) -> None:
        # An empty new_text (delete a snippet) and empty write content (empty file) are legitimate.
        edit = default_coding_presenter(_tool_call('edit_file', {'path': 'a.py', 'old_text': 'x', 'new_text': ''}))
        assert edit is not None and edit.kind == 'edit'
        write = default_coding_presenter(_tool_call('write_file', {'path': 'a.py', 'content': ''}))
        assert write is not None and write.kind == 'edit'


class TestPresenterComposition:
    def test_chain_uses_the_first_presenter_that_matches(self) -> None:
        def weather_presenter(call: ToolCallPart) -> ToolCallPresentation | None:
            return ToolCallPresentation(kind='fetch') if call.tool_name == 'get_weather' else None

        presenter = chain_presenters(weather_presenter, default_coding_presenter)
        # Custom presenter handles its tool; the default still handles FileSystem/Shell tools.
        weather = presenter(_tool_call('get_weather', {'city': 'a'}))
        assert weather is not None and weather.kind == 'fetch'
        edit = presenter(_tool_call('read_file', {'path': 'a.py'}))
        assert edit is not None and edit.kind == 'read'

    def test_chain_returns_none_when_nothing_matches(self) -> None:
        presenter = chain_presenters(default_coding_presenter)
        assert presenter(_tool_call('mystery_tool', {})) is None


class TestPathAbsolutization:
    def test_relative_paths_resolve_against_cwd(self) -> None:
        presentation = default_coding_presenter(
            _tool_call('edit_file', {'path': 'src/a.py', 'old_text': 'x', 'new_text': 'y'})
        )
        assert presentation is not None
        resolved = absolutize(presentation, '/work')
        assert [loc.path for loc in resolved.locations] == ['/work/src/a.py']
        [diff] = resolved.content
        assert diff.path == '/work/src/a.py'

    def test_absolute_paths_are_left_unchanged(self) -> None:
        presentation = default_coding_presenter(_tool_call('read_file', {'path': '/etc/hosts'}))
        assert presentation is not None
        resolved = absolutize(presentation, '/work')
        assert [loc.path for loc in resolved.locations] == ['/etc/hosts']

    def test_non_file_content_is_left_untouched(self) -> None:
        presentation = ToolCallPresentation(kind='other', content=(acp.tool_content(acp.text_block('note')),))
        resolved = absolutize(presentation, '/work')
        assert resolved.content == presentation.content

    def test_relative_traversal_escaping_the_workspace_drops_the_location(self) -> None:
        # A `..` path that normalizes outside cwd is not shown as a location: the tool sandbox
        # rejects it, and an editor should never get a click-to-file link outside the workspace.
        presentation = default_coding_presenter(_tool_call('read_file', {'path': '../../etc/passwd'}))
        assert presentation is not None
        resolved = absolutize(presentation, '/work/project')
        assert resolved.locations == ()

    def test_escaping_edit_drops_both_location_and_diff(self) -> None:
        presentation = default_coding_presenter(
            _tool_call('edit_file', {'path': '../secret.py', 'old_text': 'x', 'new_text': 'y'})
        )
        assert presentation is not None
        resolved = absolutize(presentation, '/work/project')
        assert resolved.locations == ()
        assert resolved.content == ()


class TestAbsolutizationIntegration:
    async def test_tool_call_paths_are_absolutized_against_session_cwd(self) -> None:
        agent = Agent(TestModel())

        @agent.tool_plain
        def edit_file(path: str, old_text: str, new_text: str) -> str:
            return 'edited'

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent)
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd='/work')

        await adapter.prompt(prompt=[acp.text_block('edit it')], session_id=session.session_id)

        [start] = _starts(client)
        assert [loc.path for loc in getattr(start, 'locations')] == ['/work/a']
        [diff] = getattr(start, 'content')
        assert diff.path == '/work/a'


class TestWorkspaceRooting:
    """A `session_config` factory roots `FileSystem` at the client's `cwd`, with absolute locations."""

    async def test_session_config_roots_filesystem_at_client_cwd(self, tmp_path: Path) -> None:
        from pydantic_ai_harness.filesystem import FileSystem

        write = DeltaToolCall(name='write_file', json_args=json.dumps({'path': 'note.txt', 'content': 'hi'}))
        agent = Agent(_calls_tool_each_turn(write))  # the agent itself has no filesystem tools

        def session_config(session: AcpSession) -> AcpSessionConfig[None]:
            return AcpSessionConfig(deps=None, toolsets=[FileSystem[None](root_dir=session.cwd).get_toolset()])

        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(agent, session_config=session_config)
        client = FakeClient()
        adapter.on_connect(client)
        await adapter.initialize(protocol_version=1)
        session = await adapter.new_session(cwd=str(tmp_path))

        await adapter.prompt(prompt=[acp.text_block('write the note')], session_id=session.session_id)

        # The edit landed in the client's workspace...
        assert (tmp_path / 'note.txt').read_text() == 'hi'
        # ...and the tool call reported the absolute path under that workspace.
        [start] = _starts(client)
        assert [loc.path for loc in getattr(start, 'locations')] == [str(tmp_path / 'note.txt')]
