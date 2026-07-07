"""Adapt a Pydantic AI agent to the Agent Client Protocol (ACP) agent interface.

ACP (https://agentclientprotocol.com) lets a code editor or terminal UI (the *client*)
drive a coding agent (the *server*) over stdio JSON-RPC. This module exposes a Pydantic AI
[`Agent`][pydantic_ai.Agent] as such a server: it streams the agent's output to the client
as `session/update` notifications and bridges ACP permission requests to Pydantic AI's
human-in-the-loop tool approval.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Generic, Literal
from uuid import uuid4

import acp
import anyio
from acp import schema
from acp.interfaces import Client
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied, UsageLimitExceeded
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserContent,
)
from pydantic_ai.models import KnownModelName, Model, known_model_names
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.experimental.acp._content import PromptContentBlock, prompt_blocks_to_user_content
from pydantic_ai_harness.experimental.acp._permission import (
    PermissionPolicy,
    ToolCallPermission,
    default_permission_scope,
)
from pydantic_ai_harness.experimental.acp._presentation import (
    ToolCallContent,
    ToolCallPresentation,
    ToolCallPresenter,
    absolutize,
    default_coding_presenter,
)
from pydantic_ai_harness.experimental.acp._serialize import bounded_jsonable, chunk_text, jsonable
from pydantic_ai_harness.experimental.acp._session import (
    AcpSession,
    AcpSessionConfig,
    McpServers,
    SessionConfigFunc,
    SessionState,
    SessionUpdate,
)
from pydantic_ai_harness.experimental.acp._store import SessionStore, StoredSession

# Version advertised to the client when the caller does not supply one.
DEFAULT_VERSION = '0.1.0'

_logger = logging.getLogger(__name__)
_MODEL_CONFIG_ID = 'model'


def _all_known_model_names() -> tuple[str, ...]:
    """Every model name Pydantic AI exposes through the public `known_model_names()` API."""
    return known_model_names()


def _finish_reason_to_stop_reason(finish_reason: FinishReason | None) -> schema.StopReason:
    """Map the model's terminal `finish_reason` to the ACP stop reason for a completed turn.

    Only `length` and `content_filter` carry a distinct ACP meaning; every other reason (a normal
    stop, a final tool call, an unknown/missing reason) is reported as a plain `end_turn`.
    Cancellation is handled by the caller and never reaches here.
    """
    if finish_reason == 'length':
        return 'max_tokens'
    if finish_reason == 'content_filter':
        return 'refusal'
    return 'end_turn'


def _usage_limit_stop_reason(exc: UsageLimitExceeded) -> schema.StopReason:
    """Map a run's exceeded usage limit to the ACP stop reason ending the turn.

    Token-based limits report `max_tokens`; the request/tool-call count limits report
    `max_turn_requests`. The exception carries no structured detail, so this reads its message; an
    unrecognized wording falls back to `max_turn_requests` (the limit pydantic-ai's default
    configuration can hit).
    """
    return 'max_tokens' if 'tokens_limit' in str(exc) else 'max_turn_requests'


def _to_acp_usage(usage: RunUsage) -> schema.Usage:
    """Map a Pydantic AI `RunUsage` to ACP's per-turn `Usage` (an UNSTABLE field on `PromptResponse`)."""
    return schema.Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_read_tokens=usage.cache_read_tokens,
        cached_write_tokens=usage.cache_write_tokens,
    )


class _TurnCancelled(Exception):
    """Raised internally to unwind a turn the client cancelled mid-flight (e.g. via a permission dialog)."""


@dataclass(kw_only=True)
class _TurnState:
    """Per-turn state shared by the stream handlers, created once per turn in `_run_turn`."""

    conn: Client
    session_id: str
    cwd: str
    approval_names: frozenset[str]
    # Tool-call ids accumulated across the turn's approval-resume passes: `started` so a call
    # paused for approval is announced only once, `denied` so a rejected call's result is failed,
    # `resulted` so a cancelled turn can fail only the calls still left without a terminal status.
    started: set[str] = field(default_factory=set[str])
    denied: set[str] = field(default_factory=set[str])
    resulted: set[str] = field(default_factory=set[str])
    # The `session/update`s sent this turn; appended to the session transcript only on commit.
    updates: list[SessionUpdate] = field(default_factory=list[SessionUpdate])


@dataclass(frozen=True, kw_only=True)
class _ToolCallFields:
    """The ACP tool-call fields derived from the presenter, shared by the start and permission paths."""

    title: str
    kind: schema.ToolKind | None
    content: list[ToolCallContent] | None
    locations: list[schema.ToolCallLocation] | None


class PydanticAIACPAgent(acp.Agent, Generic[AgentDepsT, OutputDataT]):
    """An ACP agent backed by a Pydantic AI [`Agent`][pydantic_ai.Agent].

    Each ACP session is an independent conversation, keyed by session ID. Pass the instance to
    [`run_acp_stdio`][pydantic_ai_harness.experimental.acp.run_acp_stdio] (or the lower-level `acp.run_agent`)
    to serve it over stdio. When calling `acp.run_agent` yourself, pass `use_unstable_protocol=True`
    as `run_acp_stdio` does, or the SDK router rejects `session/close` (still UNSTABLE in the SDK)
    with `method_not_found`.

    A tool that [requires approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/)
    pauses the run and asks the client via `session/request_permission`; "always allow"/"always
    reject" decisions are remembered for the rest of the session. Per-session workspace setup
    (the client's `cwd` and MCP servers) is surfaced through the optional `session_config` factory.

    `session/close` cancels any in-flight turn and discards the session. `session/load` (with a
    `session_store`) and model selection via `session/set_config_option` (with `models`) are
    supported when configured; fork, resume, and modes are advertised as unsupported. As with any
    `output_type` override, this adapter is incompatible with agents that define output validators.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        deps: AgentDepsT = None,
        name: str | None = None,
        version: str = DEFAULT_VERSION,
        session_config: SessionConfigFunc[AgentDepsT] | None = None,
        permission_policy: PermissionPolicy | None = None,
        prompt_capabilities: schema.PromptCapabilities | None = None,
        mcp_capabilities: schema.McpCapabilities | None = None,
        tool_presenter: ToolCallPresenter | None = None,
        session_store: SessionStore | None = None,
        models: Sequence[KnownModelName | str] | Literal['all'] | None = None,
        model_resolver: Callable[[str], Model | str] | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        """Build the adapter.

        Args:
            agent: The Pydantic AI agent to expose. Its model, tools, and instructions are used as-is.
            deps: Dependencies passed to every agent run, equivalent to `Agent.run(..., deps=deps)`.
                Used for sessions that `session_config` does not configure (or when it is not given).
            name: Name advertised to the client. Defaults to the agent's name, then `'pydantic-ai-agent'`.
            version: Version advertised to the client.
            session_config: Optional factory called once per session with the client's
                [`AcpSession`][pydantic_ai_harness.experimental.acp.AcpSession] setup (its `cwd`, MCP servers,
                and capabilities). It returns an
                [`AcpSessionConfig`][pydantic_ai_harness.experimental.acp.AcpSessionConfig] whose `deps` and
                `toolsets` are applied to every run in that session. May be sync or async.
            permission_policy: Optional function deciding the scope under which an "always
                allow"/"always reject" decision is remembered. Defaults to the exact call (tool
                name plus arguments).
            prompt_capabilities: Prompt content the agent advertises support for (image, audio,
                embedded context). Defaults to text-only; enable what the backing model handles,
                e.g. `schema.PromptCapabilities(image=True, embedded_context=True)`.
            mcp_capabilities: MCP transports the agent advertises support for, e.g.
                `schema.McpCapabilities(http=True, sse=True)`. A spec-following client only sends
                HTTP/SSE MCP servers when these are advertised (stdio servers are not gated), so
                set this when the `session_config` connects them. Defaults to neither; requires a
                `session_config`, since without one MCP servers are rejected at `session/new`.
            tool_presenter: Maps a tool call to its rich ACP presentation (`kind`, file
                `locations`, and diff `content`). Defaults to
                [`default_coding_presenter`][pydantic_ai_harness.experimental.acp.default_coding_presenter],
                which recognizes the `FileSystem`/`Shell` tools by name (a custom tool sharing a
                name is also matched). Pass your own presenter (optionally chained ahead of the
                default with [`chain_presenters`][pydantic_ai_harness.experimental.acp.chain_presenters]), or
                `lambda _call: None` to disable rich rendering.
            session_store: Optional [`SessionStore`][pydantic_ai_harness.experimental.acp.SessionStore] enabling
                `session/load`: each committed turn is persisted (model history plus client
                transcript) and a stored session can be reopened. Defaults to `None`, advertising
                `session/load` as unsupported.
            models: Optional models the client may switch between with the stable ACP session
                config option `model`. The first is each session's default. A selection applies as
                a per-run override (the shared agent is never mutated) and is persisted with the
                session. Pass `'all'` to offer every model Pydantic AI knows (its default is then
                the first known model, so the user should pick one). Defaults to `None`,
                advertising no model config option.
            model_resolver: Optional hook mapping an advertised model id to the `Model` (or model
                string) used for that run, applied just before each run's per-run override. Lets a
                host advertise ids Pydantic AI's `infer_model` does not understand (e.g. OAuth/
                subscription models) and supply a pre-built `Model` for them. Returning the id
                unchanged falls back to `infer_model`. Defaults to identity.
            usage_limits: Optional per-run ceilings (`request_limit`, `tool_calls_limit`, token
                limits) applied to every agent run, equivalent to `Agent.run(..., usage_limits=...)`.
                A turn that resumes after a tool approval starts a fresh run, so the limits bound
                each approval-to-approval segment, not the whole turn. An exceeded limit ends the
                turn with a `max_tokens`/`max_turn_requests` stop reason, never a request error.
                Defaults to Pydantic AI's own defaults (50 requests per run).
        """
        self._agent = agent
        self._deps = deps
        self._name = name or agent.name or 'pydantic-ai-agent'
        self._version = version
        self._session_config = session_config
        self._permission_policy = permission_policy or default_permission_scope
        # Text-only by default, so a client is not invited to send image/audio/resource blocks
        # the backing model may not accept.
        self._prompt_capabilities = prompt_capabilities or schema.PromptCapabilities()
        if mcp_capabilities is not None and session_config is None:
            # Advertising HTTP/SSE MCP support would invite server definitions that
            # `_build_config` then rejects; fail at construction instead of per session.
            raise ValueError('mcp_capabilities requires a session_config that connects the MCP servers')
        self._mcp_capabilities = mcp_capabilities or schema.McpCapabilities()
        self._tool_presenter = tool_presenter or default_coding_presenter
        self._session_store = session_store
        self._models: tuple[str, ...] = _all_known_model_names() if models == 'all' else tuple(models) if models else ()
        self._model_resolver = model_resolver
        self._usage_limits = usage_limits
        self._client_capabilities: schema.ClientCapabilities | None = None
        self._conn: Client | None = None
        # All live sessions, keyed by session id. Each owns its own turn lock (the dispatcher
        # delivers requests concurrently), so turns within a session are serialized.
        self._sessions: dict[str, SessionState[AgentDepsT]] = {}

    def on_connect(self, conn: Client) -> None:
        """Capture the outbound connection used to stream updates and request permission."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: schema.ClientCapabilities | None = None,
        client_info: schema.Implementation | None = None,
        **kwargs: object,
    ) -> schema.InitializeResponse:
        """Negotiate the protocol version and advertise the agent's capabilities."""
        self._client_capabilities = client_capabilities
        # Echo a supported client version; otherwise answer with the latest version we speak so
        # an out-of-range client does not negotiate an unsupported one.
        if 1 <= protocol_version <= acp.PROTOCOL_VERSION:
            negotiated_version = protocol_version
        else:
            negotiated_version = acp.PROTOCOL_VERSION
        return schema.InitializeResponse(
            protocol_version=negotiated_version,
            agent_capabilities=schema.AgentCapabilities(
                load_session=self._session_store is not None,
                prompt_capabilities=self._prompt_capabilities,
                mcp_capabilities=self._mcp_capabilities,
                session_capabilities=schema.SessionCapabilities(close=schema.SessionCloseCapabilities()),
            ),
            agent_info=schema.Implementation(name=self._name, version=self._version),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: object,
    ) -> schema.NewSessionResponse:
        """Start a new session with an empty conversation history.

        The session's `cwd` is the workspace the client opened: root the agent's tools at it via
        `session_config` (see the package README) so file edits and tool-call locations line up
        with that workspace. A request with MCP servers but no `session_config` is rejected (see
        `_build_config`).
        """
        # The spec requires `cwd` to be absolute, a MUST on the client; neither the SDK router
        # nor this adapter re-validates it. `additional_directories` is
        # part of the `acp.Agent` interface but not consumed: the capability is not advertised, so
        # a conformant client never sends extra roots, and they are not surfaced to `session_config`.
        session_id = uuid4().hex
        config = await self._build_config(session_id, cwd, mcp_servers)
        # The first configured model is the session default; `None` (no models) uses the agent's own.
        default_model = self._models[0] if self._models else None
        state = SessionState(session_id=session_id, config=config, cwd=cwd, model=default_model)
        self._sessions[session_id] = state
        # Persist the empty session so the client can reopen it even before its first turn.
        await self._persist(state)
        return schema.NewSessionResponse(session_id=session_id, config_options=self._model_config_options(state.model))

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: McpServers = None,
        additional_directories: list[str] | None = None,
        **kwargs: object,
    ) -> schema.LoadSessionResponse | None:
        """Reopen a stored session: restore its model history and replay its transcript to the client.

        Only called when a `session_store` was given (`session/load` is otherwise advertised as
        unsupported). The run configuration is rebuilt from `cwd`/`mcp_servers` via
        `session_config`, as in `new_session`.

        Raises:
            acp.RequestError: if no session with `session_id` is stored, or the stored session
                cannot be read.
        """
        if self._session_store is None:
            # Advertised off, but the SDK router routes `session/load` regardless; a client
            # calling it anyway gets the rejection the advertisement implies.
            raise acp.RequestError.method_not_found('session/load')
        if self._conn is None:  # pragma: no cover - on_connect always runs before load_session
            raise RuntimeError('load_session called before on_connect()')
        try:
            stored = await self._session_store.load(session_id)
        except Exception as exc:
            # A read or deserialization failure is a server-side durability problem, not a bad
            # client request: surface a clear, purpose-built error rather than leaking the store's
            # low-level exception (a corrupt payload would otherwise reach the client as raw
            # pydantic validation detail).
            raise acp.RequestError.internal_error(
                {'session_id': session_id, 'reason': 'stored session could not be read'}
            ) from exc
        if stored is None:
            raise acp.RequestError.invalid_params(
                {'session_id': session_id, 'reason': 'no stored session with this id'}
            )
        # A client may load a session id that is still open (a reconnecting editor, or a double
        # load). Tear down any live turn first so the orphaned turn cannot later persist its stale
        # state over the transcript and history we are about to restore.
        prior = self._sessions.pop(session_id, None)
        if prior is not None:
            await self._cancel_active_turn(prior)
        config = await self._build_config(session_id, cwd, mcp_servers)
        state = SessionState(
            session_id=session_id,
            config=config,
            cwd=cwd,
            history=list(stored.messages),
            transcript=list(stored.updates),
            model=stored.model,
        )
        self._sessions[session_id] = state
        for update in stored.updates:
            await self._conn.session_update(session_id=session_id, update=update)
        return schema.LoadSessionResponse(config_options=self._model_config_options(state.model))

    async def _persist(self, state: SessionState[AgentDepsT]) -> None:
        """Save a session's committed state (history, transcript, selected model), if a store is configured.

        A store failure is logged and swallowed rather than failing the operation that triggered the
        save: the turn (or session) has already streamed and committed in memory, so a durable-write
        error must not surface as a failure for work the user already saw succeed. The session stays
        usable and the next successful save catches the store back up.
        """
        if self._session_store is None:
            return
        try:
            await self._session_store.save(
                state.session_id,
                StoredSession(messages=list(state.history), updates=list(state.transcript), model=state.model),
            )
        except Exception:
            _logger.exception('failed to persist ACP session %s; durable state is now behind', state.session_id)

    def _model_option(self, current_model_id: str | None) -> schema.SessionConfigOptionSelect | None:
        """The model config option advertised to the client, or `None` when none is configured."""
        if current_model_id is None or not self._models:
            return None
        return schema.SessionConfigOptionSelect(
            id=_MODEL_CONFIG_ID,
            name='Model',
            type='select',
            current_value=current_model_id,
            options=[schema.SessionConfigSelectOption(value=model, name=model) for model in self._models],
        )

    def _model_config_options(
        self, current_model_id: str | None
    ) -> list[schema.SessionConfigOptionSelect | schema.SessionConfigOptionBoolean] | None:
        """The full session config option list, or `None` when model switching is not configured."""
        option = self._model_option(current_model_id)
        return None if option is None else [option]

    def session_history(self, session_id: str) -> list[ModelMessage] | None:
        """A snapshot of a resident session's committed model history (what the next turn will send).

        Returns a shallow copy: the list is the caller's to keep, but the `ModelMessage` objects are
        shared with the live session -- treat them as read-only. `None` when the session is not
        resident in this adapter.
        """
        state = self._sessions.get(session_id)
        return list(state.history) if state is not None else None

    def session_model_option(self, session_id: str) -> schema.SessionConfigOptionSelect | None:
        """The model config option for a resident session -- its option set + current selection.

        The public counterpart to the `config_options` entry `new_session`/`load_session` return,
        for callers that re-surface model selection on a later interaction (e.g. an attach/resume
        reply) without re-creating or reloading the session. `None` when the session is unknown to
        this adapter or no switch-set was configured.
        """
        state = self._sessions.get(session_id)
        return self._model_option(state.model) if state is not None else None

    async def _build_config(self, session_id: str, cwd: str, mcp_servers: McpServers) -> AcpSessionConfig[AgentDepsT]:
        """Derive a session's run configuration, calling the `session_config` factory if present.

        Raises:
            acp.RequestError: if the client provides MCP servers but no `session_config` is
                installed to connect them.
        """
        if self._session_config is None:
            if mcp_servers:
                raise acp.RequestError.invalid_params(
                    {
                        'reason': 'this agent does not connect MCP servers; provide a session_config '
                        'that turns mcp_servers into toolsets to use them',
                        'mcp_server_count': len(mcp_servers),
                    }
                )
            return AcpSessionConfig(deps=self._deps)
        if self._conn is None:  # pragma: no cover - on_connect always runs before session setup
            raise RuntimeError('_build_config called before on_connect()')
        session = AcpSession(
            cwd=cwd,
            mcp_servers=mcp_servers or [],
            client_capabilities=self._client_capabilities,
            client=self._conn,
            session_id=session_id,
        )
        result = self._session_config(session)
        return await result if isawaitable(result) else result

    async def prompt(
        self,
        session_id: str,
        prompt: list[PromptContentBlock],
        **kwargs: object,
    ) -> schema.PromptResponse:
        """Run one user turn, streaming the agent's output to the client.

        Turns for a session are serialized. The turn runs in a cancellable task so a concurrent
        `session/cancel` (or a cancelled permission dialog) stops it promptly with a `cancelled`
        stop reason; a cancelled turn's prompt and partial output are not committed to the
        session history.
        """
        if self._conn is None:  # pragma: no cover - on_connect always runs first in practice
            raise RuntimeError('prompt() called before on_connect()')
        state = self._sessions.get(session_id)
        if state is None:
            raise acp.RequestError.invalid_params({'session_id': session_id})

        user_content = prompt_blocks_to_user_content(prompt)
        async with state.lock:
            if self._sessions.get(session_id) is not state:
                # The session was closed or reloaded while this prompt waited its turn on the
                # lock. Running now would resurrect the discarded state: `cancel` could no longer
                # reach the turn, and its commit would persist the orphaned history over the
                # live session.
                raise acp.RequestError.invalid_params(
                    {'session_id': session_id, 'reason': 'session was closed while the prompt was queued'}
                )
            state.cancel_requested = False
            turn = asyncio.ensure_future(self._run_turn(state, prompt, user_content))
            state.active_turn = turn
            try:
                # Shielded so this coroutine's *own* cancellation (connection teardown) surfaces
                # here instead of propagating into the turn: with a bare `await turn` the two are
                # indistinguishable, and answering a request whose handler the connection already
                # cancelled would deadlock its shutdown (the response future is never resolved).
                return await asyncio.shield(turn)
            except _TurnCancelled:
                return schema.PromptResponse(stop_reason='cancelled')
            except asyncio.CancelledError:
                # `turn.done()` cannot fully prove the cancellation came from the turn: in the one
                # tick between the turn completing and this coroutine resuming, a teardown cancel
                # aimed at *this* handler is indistinguishable on 3.10 (3.11's `Task.cancelling()`
                # would settle it). Accepted residual; every wider window is covered below.
                if turn.done() and state.cancel_requested:
                    return schema.PromptResponse(stop_reason='cancelled')
                if state.cancel_requested:
                    # `cancel()` already delivered cancellation to the turn. Do not inject a second
                    # one while Pydantic AI's stream context is closing, or its internal event stream
                    # can be interrupted before its receive side closes.
                    with contextlib.suppress(asyncio.CancelledError, _TurnCancelled):
                        await asyncio.shield(turn)
                    raise
                # The prompt coroutine itself was cancelled (e.g. connection teardown): stop the
                # child turn before propagating, so it is not left running on a closing connection.
                turn.cancel()
                with contextlib.suppress(asyncio.CancelledError, _TurnCancelled):
                    await turn
                raise
            finally:
                state.active_turn = None
                state.cancel_requested = False

    async def cancel(self, session_id: str, **kwargs: object) -> None:
        """Cancel the in-flight turn for a session, if any."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.cancel_requested = True
        if state.active_turn is not None and not state.active_turn.done():
            state.active_turn.cancel()

    async def close_session(self, session_id: str, **kwargs: object) -> schema.CloseSessionResponse | None:
        """Cancel any in-flight turn and discard all state for the session.

        Closing an unknown (or already-closed) session is a no-op.
        """
        state = self._sessions.pop(session_id, None)
        if state is not None:
            await self._cancel_active_turn(state)
        return schema.CloseSessionResponse()

    async def _cancel_active_turn(self, state: SessionState[AgentDepsT]) -> None:
        """End a session's in-flight turn (if any) with a `cancelled` stop reason and await its unwind.

        Awaiting the turn keeps the caller from racing the turn's teardown -- the prompt has fully
        returned `cancelled` and released the session's resources before this returns.
        """
        turn = state.active_turn
        if turn is not None and not turn.done():
            state.cancel_requested = True
            turn.cancel()
            try:
                # Shielded for the same reason as in `prompt`: only the turn's own unwind may be
                # swallowed here, never this coroutine's cancellation (see the comment there).
                await asyncio.shield(turn)
            except (asyncio.CancelledError, _TurnCancelled):
                # As in `prompt`, `turn.done()` is a heuristic with a one-tick residual: a
                # teardown cancel landing between the turn completing and this resuming is
                # swallowed here on 3.10. Accepted.
                if not turn.done():
                    raise

    async def _run_turn(
        self,
        state: SessionState[AgentDepsT],
        prompt: list[PromptContentBlock],
        user_content: list[UserContent],
    ) -> schema.PromptResponse:
        """Drive the agent (resuming across tool-approval pauses), stream updates, and build the response.

        The response carries the ACP stop reason mapped from the final model response, plus the
        turn's total token usage (summed across every run pass, one per approval pause). A turn
        ended by a usage limit rolls back like a cancellation, so its response omits usage while
        still answering with the limit's `max_tokens`/`max_turn_requests` stop reason.
        """
        conn = self._conn
        assert conn is not None
        history = state.history
        config = state.config
        usage = RunUsage()
        stop_reason: schema.StopReason = 'end_turn'
        deferred_results: DeferredToolResults | None = None
        run_input: list[UserContent] | None = user_content
        output_type = [self._agent.output_type, DeferredToolRequests]
        turn = _TurnState(
            conn=conn, session_id=state.session_id, cwd=state.cwd, approval_names=self._approval_tool_names(config)
        )
        # Recorded for replay only, never sent live: the client renders its own prompt during
        # the turn, but `session/load` must rebuild the user side of the conversation too.
        # Going through `turn.updates` keeps the cancel semantics: a rolled-back turn does not
        # persist its user message either.
        turn.updates.extend(acp.update_user_message(block) for block in prompt)

        try:
            while True:
                result = None
                async with self._agent.run_stream_events(
                    run_input,
                    message_history=history,
                    deferred_tool_results=deferred_results,
                    output_type=output_type,
                    deps=config.deps,
                    toolsets=config.toolsets,
                    # Per-run override for the client's model config choice; `None` uses the
                    # agent's own model, never mutating the shared agent. A `model_resolver` (if
                    # given) maps the advertised id to a pre-built `Model` for ids `infer_model`
                    # can't parse.
                    model=self._resolve_run_model(state.model),
                    usage_limits=self._usage_limits,
                ) as stream:
                    event: AgentStreamEvent | AgentRunResultEvent[object]
                    async for event in stream:
                        if isinstance(event, AgentRunResultEvent):
                            result = event.result
                        else:
                            await self._emit_event(turn, event)

                assert result is not None, 'run_stream_events always yields a final result event'
                history = result.all_messages()
                usage += result.usage  # pydantic-ai 2.0: `usage` is a property, not a method
                output = result.output
                if isinstance(output, DeferredToolRequests):
                    if not output.approvals and not output.calls:  # pragma: no cover
                        break  # defensive: the run never yields an empty DeferredToolRequests
                    deferred_results = await self._resolve_approvals(turn, state, output)
                    run_input = None
                    continue
                if output is not None and not isinstance(output, str):
                    # Structured output isn't streamed as text parts, so deliver it as a final message.
                    # (`str` output already streamed via text deltas; `None` has nothing to show.)
                    await self._emit_text(turn, json.dumps(jsonable(output)), thought=False)
                stop_reason = _finish_reason_to_stop_reason(result.response.finish_reason)
                break
        except (asyncio.CancelledError, _TurnCancelled):
            # The turn is ending without finishing its tool calls; close out any still shown as
            # pending/in_progress so the client stops rendering them as running. Shielded because an
            # asyncio cancellation would otherwise abort the sends, and live-only: a cancelled turn
            # never commits its transcript.
            with anyio.CancelScope(shield=True):
                await self._fail_outstanding_tool_calls(turn)
            raise
        except UsageLimitExceeded as exc:
            # ACP models a turn ending at a limit as a normal stop reason, not a request error.
            # The raising run's partial messages are not retrievable, so nothing is committed --
            # the turn rolls back to the prior state, like a cancellation, and the response must
            # not report uncommitted usage.
            await self._fail_outstanding_tool_calls(turn)
            return schema.PromptResponse(stop_reason=_usage_limit_stop_reason(exc))
        except Exception:
            # The turn is failing with the error the client receives as the prompt's response;
            # close out its announced tool calls so they are not left rendering as running.
            await self._fail_outstanding_tool_calls(turn)
            raise

        state.history = history
        # Commit and persist this turn's updates alongside the history; a cancelled turn never
        # reaches here, so only committed turns are persisted.
        if self._session_store is not None:
            state.transcript.extend(turn.updates)
            try:
                await self._persist(state)
            except asyncio.CancelledError:
                # The turn is already committed: a cancel landing inside the store's save came
                # too late to roll anything back. The spec still requires the prompt to answer
                # `cancelled`, so that stop reason is reported alongside committed usage. The
                # interrupted save is the same benign failure `_persist` swallows; the next
                # successful save catches the store up.
                _logger.warning(
                    'persisting ACP session %s was cancelled; durable state is now behind', state.session_id
                )
                stop_reason = 'cancelled'
        return schema.PromptResponse(stop_reason=stop_reason, usage=_to_acp_usage(usage))

    async def _fail_outstanding_tool_calls(self, turn: _TurnState) -> None:
        """Drive every announced-but-unfinished tool call to a terminal `failed` status.

        Called when a turn is cancelled, so a client does not keep rendering a `pending`/`in_progress`
        tool call as running. Sent directly (not recorded for the transcript) and best-effort: the
        connection may already be going away.
        """
        for tool_call_id in turn.started - turn.resulted:
            with contextlib.suppress(Exception):
                await turn.conn.session_update(
                    session_id=turn.session_id,
                    update=acp.update_tool_call(tool_call_id=tool_call_id, status='failed'),
                )

    async def _send_update(self, turn: _TurnState, update: SessionUpdate) -> None:
        """Send one `session/update` to the client, recording it for the transcript."""
        turn.updates.append(update)
        await turn.conn.session_update(session_id=turn.session_id, update=update)

    async def _emit_text(self, turn: _TurnState, text: str, *, thought: bool) -> None:
        """Stream text to the client as one or more chunked `session/update` notifications."""
        for chunk in chunk_text(text):
            update = acp.update_agent_thought_text(chunk) if thought else acp.update_agent_message_text(chunk)
            await self._send_update(turn, update)

    def _tool_call_fields(
        self, call: ToolCallPart, cwd: str, *, default_kind: schema.ToolKind | None
    ) -> _ToolCallFields:
        """Derive the ACP tool-call fields shared by the start and permission paths via the presenter.

        Falls back to the tool name for the title, and to `default_kind` when the presenter
        offers no `kind`.
        """
        presentation = self._tool_presenter(call)
        presentation = absolutize(presentation, cwd) if presentation is not None else ToolCallPresentation()
        return _ToolCallFields(
            title=presentation.title or call.tool_name,
            kind=presentation.kind or default_kind,
            content=list(presentation.content) or None,
            locations=list(presentation.locations) or None,
        )

    async def _emit_event(self, turn: _TurnState, event: AgentStreamEvent) -> None:
        """Translate a single Pydantic AI stream event into an ACP `session/update`."""
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart) and part.content:
                await self._emit_text(turn, part.content, thought=False)
            elif isinstance(part, ThinkingPart) and part.content:
                await self._emit_text(turn, part.content, thought=True)
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta) and delta.content_delta:
                await self._emit_text(turn, delta.content_delta, thought=False)
            elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                await self._emit_text(turn, delta.content_delta, thought=True)
        elif isinstance(event, FunctionToolCallEvent):
            call = event.part
            if call.tool_call_id in turn.started:
                return  # the model re-emits the call event when resuming after approval
            turn.started.add(call.tool_call_id)
            fields = self._tool_call_fields(call, turn.cwd, default_kind=None)
            # A call awaiting approval is not running yet: it starts `pending` and
            # `_resolve_approvals` promotes it once approved. Any other call is already executing.
            status: schema.ToolCallStatus = 'pending' if call.tool_name in turn.approval_names else 'in_progress'
            await self._send_update(
                turn,
                acp.start_tool_call(
                    tool_call_id=call.tool_call_id,
                    title=fields.title,
                    kind=fields.kind,
                    status=status,
                    content=fields.content,
                    locations=fields.locations,
                    raw_input=bounded_jsonable(call.args),
                ),
            )
        elif isinstance(event, FunctionToolResultEvent):
            part = event.part
            if isinstance(part, RetryPromptPart):
                status, raw_output = 'failed', part.model_response()
            elif part.tool_call_id in turn.denied:
                status, raw_output = 'failed', part.content
            else:
                status, raw_output = 'completed', part.content
            turn.resulted.add(part.tool_call_id)
            await self._send_update(
                turn,
                acp.update_tool_call(
                    tool_call_id=part.tool_call_id,
                    status=status,
                    raw_output=bounded_jsonable(raw_output),
                ),
            )

    async def _resolve_approvals(
        self, turn: _TurnState, state: SessionState[AgentDepsT], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        """Ask the client to approve each pending tool call and collect the decisions.

        Records rejected call IDs in `turn.denied` so the eventual result event is reported as failed.

        Raises:
            _TurnCancelled: if the client cancels a permission request.
            acp.RequestError: if the agent requests external tool execution, which is unsupported.
        """
        if requests.calls:
            names = sorted({call.tool_name for call in requests.calls})
            raise acp.RequestError.internal_error(
                {'reason': 'external tool execution is not supported by the ACP adapter', 'tools': names}
            )

        results = DeferredToolResults()
        for call in requests.approvals:
            # `args_as_dict()` canonicalizes the call's arguments to a dict: a model may deliver them
            # as a JSON string (the OpenAI default, and how streamed calls accumulate), and a raw
            # string would make the scope key sensitive to key order, defeating a remembered "always"
            # decision for what is the same logical call.
            scope = self._permission_policy(
                ToolCallPermission(tool_name=call.tool_name, tool_call_id=call.tool_call_id, args=call.args_as_dict())
            )
            if scope in state.always_allow:
                results.approvals[call.tool_call_id] = True
                await self._mark_running(turn, call.tool_call_id)
                continue
            if scope in state.always_reject:
                results.approvals[call.tool_call_id] = ToolDenied('Rejected by the client.')
                turn.denied.add(call.tool_call_id)
                continue
            # Only an approved call is promoted to `in_progress`; a rejected one stays `pending`
            # until its `failed` result, never shown as running.
            decision = await self._request_permission(turn, state, call, scope)
            results.approvals[call.tool_call_id] = decision
            if isinstance(decision, ToolDenied):
                turn.denied.add(call.tool_call_id)
            else:
                await self._mark_running(turn, call.tool_call_id)
        return results

    async def _mark_running(self, turn: _TurnState, tool_call_id: str) -> None:
        """Promote an approved tool call from `pending` to `in_progress`, just before it executes."""
        await self._send_update(turn, acp.update_tool_call(tool_call_id=tool_call_id, status='in_progress'))

    def _approval_tool_names(self, config: AcpSessionConfig[AgentDepsT]) -> frozenset[str]:
        """Names of the tools that pause for the client's approval, announced `pending` in `_emit_event`.

        Only `FunctionToolset`-held tools expose `requires_approval` without a live run context;
        tools from other toolset types are treated as not requiring approval.
        """
        toolsets: list[AbstractToolset[AgentDepsT]] = [*self._agent.toolsets, *(config.toolsets or [])]
        names: set[str] = set()
        for toolset in toolsets:
            if isinstance(toolset, FunctionToolset):
                names.update(name for name, tool in toolset.tools.items() if tool.requires_approval)
        return frozenset(names)

    async def _request_permission(
        self, turn: _TurnState, state: SessionState[AgentDepsT], call: ToolCallPart, scope: Hashable
    ) -> bool | ToolDenied:
        """Ask the client to approve a single tool call, remembering "always" decisions by `scope`."""
        fields = self._tool_call_fields(call, turn.cwd, default_kind='execute')
        response = await turn.conn.request_permission(
            session_id=turn.session_id,
            tool_call=schema.ToolCallUpdate(
                tool_call_id=call.tool_call_id,
                title=fields.title,
                kind=fields.kind,
                status='pending',
                content=fields.content,
                locations=fields.locations,
                raw_input=bounded_jsonable(call.args),
            ),
            options=[
                schema.PermissionOption(kind='allow_once', name='Allow', option_id='allow_once'),
                schema.PermissionOption(kind='allow_always', name='Always allow', option_id='allow_always'),
                schema.PermissionOption(kind='reject_once', name='Reject', option_id='reject_once'),
                schema.PermissionOption(kind='reject_always', name='Always reject', option_id='reject_always'),
            ],
        )
        outcome = response.outcome
        if isinstance(outcome, schema.DeniedOutcome):
            # ACP signals a cancelled turn via a "cancelled" permission outcome.
            raise _TurnCancelled
        if outcome.option_id == 'allow_always':
            state.always_allow.add(scope)
        elif outcome.option_id == 'reject_always':
            state.always_reject.add(scope)
        if outcome.option_id in ('allow_once', 'allow_always'):
            return True
        return ToolDenied('Rejected by the client.')

    # --- Optional ACP methods not supported by this adapter ------------------------------
    # The capabilities for these are advertised as off in `initialize`, but the SDK router still
    # routes the methods here (`acp.Agent` is a Protocol whose inherited stub bodies would answer
    # success-`null`), so these raises are what a client calling them anyway actually receives.

    async def authenticate(self, method_id: str, **kwargs: object) -> schema.AuthenticateResponse | None:
        """No authentication is required, so this is a no-op."""
        return None

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: object,
    ) -> schema.ListSessionsResponse:
        raise acp.RequestError.method_not_found('session/list')

    def _resolve_run_model(self, model_id: str | None) -> Model | str | None:
        """Map an advertised model id to the per-run override, applying `model_resolver` if set.

        `None` (no advertised models) is passed through so the run uses the agent's own model.
        """
        if model_id is None or self._model_resolver is None:
            return model_id
        return self._model_resolver(model_id)

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: object
    ) -> schema.SetSessionModeResponse | None:
        raise acp.RequestError.method_not_found('session/set_mode')

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: object
    ) -> schema.SetSessionConfigOptionResponse | None:
        """Switch the session's model through ACP's stable session config option surface."""
        state = self._sessions.get(session_id)
        if state is None:
            raise acp.RequestError.invalid_params({'session_id': session_id})
        if config_id != _MODEL_CONFIG_ID:
            raise acp.RequestError.invalid_params({'config_id': config_id, 'reason': 'unknown config option'})
        if not isinstance(value, str) or value not in self._models:
            raise acp.RequestError.invalid_params({'model_id': value, 'reason': 'not an advertised model'})
        state.model = value
        await self._persist(state)
        options = self._model_config_options(state.model)
        assert options is not None
        return schema.SetSessionConfigOptionResponse(config_options=options)

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: object,
    ) -> schema.ForkSessionResponse:
        raise acp.RequestError.method_not_found('session/fork')

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: object,
    ) -> schema.ResumeSessionResponse:
        raise acp.RequestError.method_not_found('session/resume')

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        return None
