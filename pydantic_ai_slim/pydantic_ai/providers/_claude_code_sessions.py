"""The live Claude Code sessions behind `ClaudeCodeProvider`'s in-memory Messages API transport.

`AnthropicModel` builds each request as usual and sends it to an `httpx2` transport that never opens a socket. The
transport hands the request to a Claude Code process, started through the Claude Agent SDK with the user's own,
unmodified `claude` login, and turns the raw Messages API stream events the SDK reports back into the
server-sent events `AnthropicModel` decodes.

Each conversation keeps one session while a lease is held (an agent run, or an outer `async with` on the model,
provider or agent), so Claude Code's prompt cache stays warm: a request that continues the session's history only
sends what's new. A request with any other history, instructions, tools or settings starts a new session and replays
the history into it, so correctness never depends on the session that's kept.

The model's function tools reach Claude Code as tools of an in-process MCP server. Claude Code calls them one at a
time after the model's response ends; each call waits until the next request carries its result, so the turn
continues in the same process. Claude Code's own tools, settings files, skills, MCP servers and connectors are off.
"""

from __future__ import annotations as _annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import httpx2
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from .. import __version__

try:
    from anthropic.types.beta import (
        BetaInputJSONDelta,
        BetaRawContentBlockDelta,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStreamEvent,
        BetaSignatureDelta,
        BetaTextDelta,
        BetaThinkingDelta,
    )
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ClaudeSDKError,
        HookContext,
        HookInput,
        HookJSONOutput,
        HookMatcher,
        Message,
        RateLimitEvent,
        ResultMessage,
        SdkMcpTool,
        StreamEvent,
        Transport,
        create_sdk_mcp_server,
    )
    from claude_agent_sdk.types import EffortLevel, ThinkingConfig
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `claude-agent-sdk` to use the Claude Code provider, '
        'you can use the `claude-code` optional group — `pip install "pydantic-ai-slim[claude-code]"`'
    ) from _import_error

__all__ = ('ClaudeCodeSessions', 'ClaudeCodeTransport', 'CONVERSATION_HEADER')

CONVERSATION_HEADER = 'x-pydantic-ai-conversation-id'
"""Carries the request's `conversation_id` from `ClaudeCodeModel` to the transport. It never leaves the process."""

_MCP_SERVER = 'pydantic_ai'
_TOOL_PREFIX = f'mcp__{_MCP_SERVER}__'

_MAX_SESSIONS = 8
"""Live sessions kept per provider; the least recently used idle one is closed to make room for another."""

_ENV = {
    # Claude Code prefers an API key, proxy or cloud provider from the environment over the user's login. Those are
    # usually set for other providers, and would bill the API, or send the login's token to the proxy, instead.
    **dict.fromkeys(
        (
            'ANTHROPIC_API_KEY',
            'ANTHROPIC_AUTH_TOKEN',
            'ANTHROPIC_BASE_URL',
            'CLAUDE_CODE_USE_BEDROCK',
            'CLAUDE_CODE_USE_VERTEX',
            'CLAUDE_CODE_USE_FOUNDRY',
            'CLAUDE_CODE_USE_GATEWAY',
            'CLAUDE_CODE_USE_MANTLE',
            'CLAUDE_CODE_USE_ANTHROPIC_AWS',
            'CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD',
        ),
        '',
    ),
    # No session titles or other side requests that draw on the user's plan.
    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
    # No `@file` expansion, turn-start attachments, CLAUDE.md files, memory, or compaction of the history we sent.
    'CLAUDE_CODE_DISABLE_ATTACHMENTS': '1',
    'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',
    'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
    'DISABLE_AUTO_COMPACT': '1',
    'DISABLE_COMPACT': '1',
    # Never switch to another model or a non-streaming request behind the model's back.
    'CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK': '1',
    'CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK': '1',
    # A tool call waits for the next request, which can take as long as the tool runs; results aren't truncated.
    'MCP_TOOL_TIMEOUT': str(24 * 60 * 60 * 1000),
    'MAX_MCP_OUTPUT_TOKENS': str(10_000_000),
}

_ERROR_STATUS: Mapping[str, tuple[int, str]] = {
    'authentication_failed': (401, 'authentication_error'),
    'billing_error': (402, 'billing_error'),
    'rate_limit': (429, 'rate_limit_error'),
    'invalid_request': (400, 'invalid_request_error'),
    'server_error': (500, 'api_error'),
    'unknown': (500, 'api_error'),
}


class ClaudeCodeExited(Exception):
    """Claude Code's process ended while a session was waiting on it."""


class _Tool(BaseModel):
    model_config = ConfigDict(extra='ignore')

    name: str
    description: str = ''
    input_schema: dict[str, JsonValue]


class _TextBlock(BaseModel):
    model_config = ConfigDict(extra='ignore')

    text: str


class _OutputConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    effort: EffortLevel | None = None


class _Message(BaseModel):
    model_config = ConfigDict(extra='ignore')

    role: Literal['user', 'assistant']
    content: list[dict[str, JsonValue]]

    def blocks(self) -> list[dict[str, JsonValue]]:
        """The content blocks, without the cache breakpoints Claude Code places itself."""
        return [_without_cache_control(block) for block in self.content]


class _MessagesRequest(BaseModel):
    """The parts of a Messages API request body that a Claude Code session can honor."""

    model_config = ConfigDict(extra='ignore')

    model: str
    max_tokens: int
    messages: list[_Message]
    system: str | list[_TextBlock] | None = None
    tools: list[_Tool] = []
    thinking: dict[str, JsonValue] | None = None
    output_config: _OutputConfig | None = None
    stream: bool = False


_thinking_ta: TypeAdapter[ThinkingConfig] = TypeAdapter(ThinkingConfig)
_stream_event_ta: TypeAdapter[BetaRawMessageStreamEvent] = TypeAdapter(BetaRawMessageStreamEvent)
_tool_result_content_ta: TypeAdapter[str | list[dict[str, JsonValue]]] = TypeAdapter(str | list[dict[str, JsonValue]])


def _without_cache_control(block: dict[str, JsonValue]) -> dict[str, JsonValue]:
    block = {key: value for key, value in block.items() if key != 'cache_control'}
    content = block.get('content')
    if isinstance(content, list):
        block['content'] = [
            _without_cache_control(cast(dict[str, JsonValue], item)) if isinstance(item, dict) else item
            for item in content
        ]
    return block


@dataclass(frozen=True)
class _SessionConfig:
    """What a Claude Code process fixes when it starts: a request that differs needs a new session."""

    model: str
    system_prompt: str
    tools: str
    thinking: str
    effort: EffortLevel | None
    max_tokens: int

    @classmethod
    def of(cls, request: _MessagesRequest) -> _SessionConfig:
        system = request.system
        system_prompt = system if isinstance(system, str) else '\n\n'.join(block.text for block in system or [])
        return cls(
            model=request.model,
            system_prompt=system_prompt,
            tools=json.dumps([tool.model_dump() for tool in request.tools], sort_keys=True),
            thinking=json.dumps(request.thinking, sort_keys=True),
            effort=request.output_config.effort if request.output_config else None,
            max_tokens=request.max_tokens,
        )


@dataclass(frozen=True)
class ClaudeCodeOptions:
    """How `ClaudeCodeProvider` starts Claude Code."""

    cli_path: str | Path | None = None
    cwd: str | Path | None = None


_PROJECTED_FIELDS = frozenset({'type', 'text', 'thinking', 'data', 'id', 'name', 'input'})


def _response_projection(blocks: Sequence[Mapping[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    """What has to match for a response in the request's history to be the one the session produced.

    Signatures and citations are left out, and so are empty text blocks, which a stored response may drop.
    """
    return [
        {key: value for key, value in block.items() if key in _PROJECTED_FIELDS}
        for block in blocks
        if block.get('type') != 'text' or block.get('text')
    ]


def _mcp_content(content: JsonValue) -> list[JsonValue] | None:
    """A `tool_result` block's content as MCP tool content, or `None` when MCP can't carry it."""
    items = _tool_result_content_ta.validate_python(content)
    if isinstance(items, str):
        return [{'type': 'text', 'text': items}]
    result: list[JsonValue] = []
    for item in items:
        source = item.get('source')
        if item.get('type') == 'text':
            result.append({'type': 'text', 'text': item.get('text')})
        elif item.get('type') == 'image' and isinstance(source, dict) and source.get('type') == 'base64':
            result.append({'type': 'image', 'data': source.get('data'), 'mimeType': source.get('media_type')})
        else:
            # Such as an image by URL, which MCP has no form for.
            return None
    return result


def _sse(event: Mapping[str, JsonValue]) -> bytes:
    return f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode()


def _stream_error(message: str) -> bytes:
    return _sse({'type': 'error', 'error': {'type': 'api_error', 'message': message}})


def _error_response(status: int, error_type: str, message: str, retry_after: float | None = None) -> httpx2.Response:
    headers = {'retry-after': str(math.ceil(retry_after))} if retry_after is not None else {}
    body = {'type': 'error', 'error': {'type': error_type, 'message': message}}
    return httpx2.Response(status, json=body, headers=headers)


@dataclass(eq=False)
class _Turn:
    """What the session saw of the response it's streaming, for the next request to continue from."""

    blocks: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    partial_json: dict[int, str] = field(default_factory=dict[int, str])
    stop_reason: str | None = None
    complete: bool = True
    """Whether every event was understood, so the blocks are what the next request continues from."""

    def record(self, raw_event: dict[str, JsonValue]) -> None:
        if not self.complete:
            # The block indexes no longer line up with what was recorded.
            return
        try:
            event = _stream_event_ta.validate_python(raw_event)
        except ValidationError:
            # An event from a newer Claude Code than the `anthropic` package knows; `AnthropicModel` still reads it,
            # but the next request is replayed into a new session rather than continued from a partial record.
            self.complete = False
            return
        if isinstance(event, BetaRawContentBlockStartEvent):
            self.blocks.append(event.content_block.model_dump(mode='json', exclude_none=True))
        elif isinstance(event, BetaRawContentBlockDeltaEvent):
            self._apply_delta(self.blocks[event.index], event.index, event.delta)
        elif isinstance(event, BetaRawContentBlockStopEvent):
            if (raw := self.partial_json.pop(event.index, None)) is not None:
                self.blocks[event.index]['input'] = json.loads(raw) if raw else {}
        elif isinstance(event, BetaRawMessageDeltaEvent):
            self.stop_reason = event.delta.stop_reason

    def _apply_delta(self, block: dict[str, JsonValue], index: int, delta: BetaRawContentBlockDelta) -> None:
        if isinstance(delta, BetaTextDelta):
            block['text'] = f'{block.get("text", "")}{delta.text}'
        elif isinstance(delta, BetaThinkingDelta):
            block['thinking'] = f'{block.get("thinking", "")}{delta.thinking}'
        elif isinstance(delta, BetaSignatureDelta):
            block['signature'] = delta.signature
        elif isinstance(delta, BetaInputJSONDelta):  # pragma: no branch
            # Citation deltas don't reach the blocks the next request continues from: Claude Code's tools are off.
            self.partial_json[index] = f'{self.partial_json.get(index, "")}{delta.partial_json}'

    @property
    def tool_use_ids(self) -> list[str]:
        return [str(block['id']) for block in self.blocks if block.get('type') == 'tool_use']


@dataclass(eq=False)
class _Session:
    """One Claude Code process and what it has seen of one conversation."""

    config: _SessionConfig
    loop: asyncio.AbstractEventLoop
    client: ClaudeSDKClient = field(init=False)
    messages: AsyncIterator[Message] = field(init=False)
    seen: list[_Message] = field(default_factory=list[_Message])
    """The history of the request the session last answered, followed by its answer once that has ended."""
    awaiting_results: list[str] = field(default_factory=list[str])
    """The tool calls of the last response, which wait for their results in the next request."""
    results: dict[str, asyncio.Future[dict[str, JsonValue]]] = field(
        default_factory=dict[str, 'asyncio.Future[dict[str, JsonValue]]']
    )
    claims: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    """The IDs of the tool calls Claude Code is about to make, by tool name and arguments."""
    busy: bool = False
    broken: bool = False
    last_used: float = field(default_factory=time.monotonic)

    def result(self, tool_use_id: str) -> asyncio.Future[dict[str, JsonValue]]:
        if (future := self.results.get(tool_use_id)) is None:
            future = self.results[tool_use_id] = self.loop.create_future()
        return future

    async def pre_tool_use(
        self, input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Remember which call the MCP handler about to run belongs to, as MCP doesn't tell it."""
        # The SDK calls this hook for `PreToolUse` only, and Claude Code's `PreToolUse` input always has the call's ID.
        assert tool_use_id is not None and input_data['hook_event_name'] == 'PreToolUse'
        key = _claim_key(input_data['tool_name'], input_data['tool_input'])
        self.claims.setdefault(key, []).append(tool_use_id)
        return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow'}}

    def tool(self, tool: _Tool) -> SdkMcpTool[dict[str, JsonValue]]:
        async def call(arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
            tool_use_id = self.claims[_claim_key(f'{_TOOL_PREFIX}{tool.name}', arguments)].pop(0)
            return await self.result(tool_use_id)

        return SdkMcpTool(name=tool.name, description=tool.description, input_schema=tool.input_schema, handler=call)

    def continuation(self, messages: list[_Message]) -> _Message | None:
        """The new user message, when `messages` is the session's history followed by one; otherwise `None`."""
        seen = self.seen
        if len(messages) != len(seen) + 1 or messages[-1].role != 'user':
            return None
        for sent, now in zip(seen[:-1], messages[: len(seen) - 1]):
            if sent.role != now.role or sent.blocks() != now.blocks():
                return None
        answer, now = seen[-1], messages[len(seen) - 1]
        if now.role != 'assistant' or _response_projection(answer.blocks()) != _response_projection(now.blocks()):
            return None
        return messages[-1]

    async def send(self, message: _Message) -> bool:
        """Continue the session with a new user message. Returns `False` when it can't be sent incrementally."""
        blocks = message.blocks()
        if self.awaiting_results:
            results = {str(block.get('tool_use_id')): block for block in blocks if block.get('type') == 'tool_result'}
            if len(results) != len(blocks) or set(results) != set(self.awaiting_results):
                return False
            mcp_results: dict[str, dict[str, JsonValue]] = {}
            for tool_use_id, block in results.items():
                if (content := _mcp_content(block.get('content', ''))) is None:
                    return False
                mcp_results[tool_use_id] = {'content': content, 'is_error': bool(block.get('is_error'))}
            for tool_use_id, mcp_result in mcp_results.items():
                self.result(tool_use_id).set_result(mcp_result)
            self.awaiting_results = []
            return True
        # Claude Code reports the end of a turn after its response ends.
        await self.read_result()
        await self.client.query(_one_frame(_frame('user', blocks)))
        return True

    async def replay(self, messages: list[_Message]) -> None:
        """Send a new session the request's history, all of it but the last message without querying the model."""
        for index, message in enumerate(messages):
            frame = _frame(message.role, [_with_mcp_tool_name(block) for block in message.blocks()])
            last = index == len(messages) - 1
            if message.role == 'user' and not last:
                frame['shouldQuery'] = False
            await self.client.query(_one_frame(frame))
            if message.role == 'user' and not last:
                # Claude Code acknowledges a message it doesn't query the model for with a result of zero turns.
                await self.read_result()

    async def read_result(self) -> None:
        async for message in self.messages:
            if isinstance(message, ResultMessage):
                return
        raise ClaudeCodeExited()

    async def close(self) -> None:
        self.broken = True
        for future in self.results.values():
            future.cancel()
        await self.client.disconnect()


def _claim_key(tool_name: str, arguments: Mapping[str, object]) -> str:
    return f'{tool_name}\0{json.dumps(arguments, sort_keys=True)}'


def _with_mcp_tool_name(block: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if block.get('type') == 'tool_use':
        return {**block, 'name': f'{_TOOL_PREFIX}{block.get("name")}'}
    return block


def _frame(role: Literal['user', 'assistant'], blocks: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    content: list[JsonValue] = list(blocks)
    message: dict[str, JsonValue] = {'role': role, 'content': content}
    return {'type': role, 'message': message, 'parent_tool_use_id': None}


async def _one_frame(frame: dict[str, JsonValue]) -> AsyncIterator[dict[str, JsonValue]]:
    yield frame


@dataclass(eq=False)
class ClaudeCodeSessions:
    """The live Claude Code sessions of one provider, by conversation."""

    options: ClaudeCodeOptions
    sessions: dict[str, _Session] = field(default_factory=dict[str, _Session])
    leases: int = 0
    transport_factory: Callable[[ClaudeAgentOptions], Transport] | None = None
    """Replaces the Claude Code process, for tests."""

    def acquire(self) -> None:
        """Keep each conversation's session alive until the matching `release_lease`."""
        self.leases += 1

    async def release_lease(self) -> None:
        """End a lease. The last one closes every idle session; a busy one closes once its response ends."""
        self.leases -= 1
        if self.leases:
            return
        idle = [session for session in self.sessions.values() if not session.busy]
        self.sessions = {key: session for key, session in self.sessions.items() if session.busy}
        for session in idle:
            await session.close()

    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[None]:
        self.acquire()
        try:
            yield
        finally:
            await self.release_lease()

    def _client_options(self, session: _Session, request: _MessagesRequest) -> ClaudeAgentOptions:
        config = session.config
        server = create_sdk_mcp_server(_MCP_SERVER, tools=[session.tool(tool) for tool in request.tools])
        return ClaudeAgentOptions(
            model=config.model,
            system_prompt=config.system_prompt,
            tools=[],
            mcp_servers={_MCP_SERVER: server},
            strict_mcp_config=True,
            setting_sources=[],
            skills=[],
            verbatim_prompts=True,
            include_partial_messages=True,
            hooks={'PreToolUse': [HookMatcher(matcher=f'^{_TOOL_PREFIX}', hooks=[session.pre_tool_use])]},
            thinking=_thinking_ta.validate_python(request.thinking) if request.thinking else None,
            effort=config.effort,
            cli_path=self.options.cli_path,
            cwd=self.options.cwd,
            client_app=f'pydantic-ai/{__version__}',
            env={**_ENV, 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(config.max_tokens)},
        )

    async def _start(self, request: _MessagesRequest) -> _Session:
        session = _Session(config=_SessionConfig.of(request), loop=asyncio.get_running_loop())
        options = self._client_options(session, request)
        transport = self.transport_factory(options) if self.transport_factory is not None else None
        session.client = ClaudeSDKClient(options=options, transport=transport)
        try:
            await session.client.connect()
            session.messages = aiter(session.client.receive_messages())
            await session.replay(request.messages)
        except BaseException:
            await session.close()
            raise
        return session

    async def _reusable(self, conversation_id: str | None, request: _MessagesRequest) -> _Session | None:
        if conversation_id is None or (session := self.sessions.get(conversation_id)) is None:
            return None
        if session.busy:
            # Another request of the conversation is still running, so this one gets a session of its own.
            return None
        # Claimed before the first `await`, so a concurrent request of the conversation can't take it too.
        session.busy = True
        reused = False
        try:
            reused = (
                session.config == _SessionConfig.of(request)
                and session.loop is asyncio.get_running_loop()
                and (message := session.continuation(request.messages)) is not None
                and await session.send(message)
            )
        except (ClaudeCodeExited, ClaudeSDKError):
            # The process ended or failed between requests; a new one takes over the conversation.
            pass
        finally:
            if not reused:
                del self.sessions[conversation_id]
                await session.close()
        return session if reused else None

    async def open(self, conversation_id: str | None, request: _MessagesRequest) -> tuple[_Session, bool]:
        """A session that has just been sent the request. Returns it, and whether it's kept after the response."""
        session = await self._reusable(conversation_id, request)
        if session is None:
            session = await self._start(request)
        # Claimed before the session is kept, so a concurrent request of the conversation can't take it.
        session.busy = True
        session.seen = list(request.messages)
        if conversation_id is not None and self.leases and conversation_id not in self.sessions:
            await self._keep(conversation_id, session)
        return session, session in self.sessions.values()

    async def _keep(self, conversation_id: str, session: _Session) -> None:
        """Keep the session, closing the least recently used idle one if that's needed to make room."""
        evicted: _Session | None = None
        if len(self.sessions) >= _MAX_SESSIONS:
            idle = [other for other in self.sessions.values() if not other.busy]
            if not idle:
                return
            evicted = min(idle, key=lambda other: other.last_used)
            self.sessions = {key: other for key, other in self.sessions.items() if other is not evicted}
        # Stored before the first `await`, so a concurrent request can't take the room too.
        self.sessions[conversation_id] = session
        if evicted is not None:
            await evicted.close()

    async def release(self, session: _Session, *, kept: bool) -> None:
        session.busy = False
        session.last_used = time.monotonic()
        if kept and self.leases and not session.broken and session in self.sessions.values():
            return
        self.sessions = {key: other for key, other in self.sessions.items() if other is not session}
        await session.close()


class _TurnStream(httpx2.AsyncByteStream):
    """The server-sent events of one response, read from the session as `AnthropicModel` decodes them."""

    def __init__(
        self, sessions: ClaudeCodeSessions, session: _Session, kept: bool, first: list[dict[str, JsonValue]]
    ) -> None:
        self._sessions = sessions
        self._session = session
        self._kept = kept
        self._first = first
        self._turn = _Turn()
        self._done = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._first:
            yield self._event(event)
        try:
            async for message in self._session.messages:
                if isinstance(message, StreamEvent) and message.parent_tool_use_id is None:
                    yield self._event(message.event)
                    if self._done:
                        return
                elif isinstance(message, ResultMessage):
                    # The turn ended before the response did, as when the API request fails partway.
                    yield _stream_error(message.result or f'Claude Code ended with `{message.subtype}`.')
                    return
        except ClaudeSDKError as e:
            yield _stream_error(f'Claude Code failed: {e}')
        else:
            yield _stream_error('Claude Code exited before the response ended.')

    def _event(self, event: dict[str, JsonValue]) -> bytes:
        event = _with_pydantic_ai_tool_name(event)
        self._turn.record(event)
        if event.get('type') == 'message_stop':
            self._finish()
        return _sse(event)

    def _finish(self) -> None:
        self._done = True
        session, turn = self._session, self._turn
        session.seen = [*session.seen, _Message(role='assistant', content=turn.blocks)]
        session.awaiting_results = turn.tool_use_ids if turn.stop_reason == 'tool_use' else []
        session.broken |= not turn.complete

    async def aclose(self) -> None:
        # A response that didn't end leaves the session mid-turn.
        self._session.broken |= not self._done
        await self._sessions.release(self._session, kept=self._kept)


def _with_pydantic_ai_tool_name(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    block = event.get('content_block')
    if (
        event.get('type') != 'content_block_start'
        or not isinstance(block, dict)
        or block.get('type') != 'tool_use'
        or not isinstance(name := block.get('name'), str)
    ):
        return event
    return {**event, 'content_block': {**block, 'name': name.removeprefix(_TOOL_PREFIX)}}


class ClaudeCodeTransport(httpx2.AsyncBaseTransport):
    """Serves `AnthropicModel`'s Messages API requests from Claude Code sessions, without opening a socket."""

    def __init__(self, sessions: ClaudeCodeSessions) -> None:
        self.sessions = sessions

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path != '/v1/messages':
            return _error_response(404, 'not_found_error', f'Claude Code serves no `{request.url.path}` endpoint.')
        body = _MessagesRequest.model_validate_json(await request.aread())
        # `ClaudeCodeModel` streams every request, and a request's history ends with the user's turn.
        assert body.stream and body.messages[-1].role == 'user'

        try:
            session, kept = await self.sessions.open(request.headers.get(CONVERSATION_HEADER), body)
        except ClaudeCodeExited:
            return _error_response(500, 'api_error', 'Claude Code exited while it was sent the history.')
        except ClaudeSDKError as e:
            return _error_response(500, 'api_error', f'Claude Code failed: {e}')
        try:
            try:
                first, error = await _first_events(session)
            except ClaudeSDKError as e:
                first, error = [], _error_response(500, 'api_error', f'Claude Code failed: {e}')
            if error is not None:
                session.broken = True
                await self.sessions.release(session, kept=kept)
                return error
        except BaseException:
            session.broken = True
            await self.sessions.release(session, kept=kept)
            raise
        stream = _TurnStream(self.sessions, session, kept, first)
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)


async def _first_events(session: _Session) -> tuple[list[dict[str, JsonValue]], httpx2.Response | None]:
    """The response's events up to its first, so an error that comes instead becomes an HTTP error status."""
    error_kind: str | None = None
    rate_limit: RateLimitEvent | None = None
    async for message in session.messages:
        if isinstance(message, StreamEvent) and message.parent_tool_use_id is None:
            return [message.event], None
        if isinstance(message, AssistantMessage) and message.error is not None:
            error_kind = message.error
        elif isinstance(message, RateLimitEvent) and message.rate_limit_info.status == 'rejected':
            rate_limit = message
        elif isinstance(message, ResultMessage):
            return [], _result_error(message, error_kind, rate_limit)
    return [], _error_response(500, 'api_error', 'Claude Code exited without a response.')


def _result_error(result: ResultMessage, error_kind: str | None, rate_limit: RateLimitEvent | None) -> httpx2.Response:
    if rate_limit is not None:
        error_kind = 'rate_limit'
    status, error_type = _ERROR_STATUS.get(error_kind or 'unknown', _ERROR_STATUS['unknown'])
    status = result.api_error_status or status
    retry_after = None
    if rate_limit is not None and (resets_at := rate_limit.rate_limit_info.resets_at) is not None:
        retry_after = max(resets_at - time.time(), 0)
    message = result.result or ', '.join(result.errors or ()) or f'Claude Code ended with `{result.subtype}`.'
    return _error_response(status, error_type, message, retry_after)
