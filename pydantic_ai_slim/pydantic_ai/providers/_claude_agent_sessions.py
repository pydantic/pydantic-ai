"""The live Claude Code sessions behind `ClaudeAgentProvider`'s in-memory Messages API transport.

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
import hashlib
import json
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import anyio
import httpx2
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from .. import __version__
from ._claude_agent_mcp import MCP_PREFIX, ResponseEvents, replay_blocks, server_and_tool, tool_name_server

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
        ResultError,
        ResultMessage,
        SdkMcpTool,
        StreamEvent,
        ToolResultBlock,
        Transport,
        UserMessage,
        create_sdk_mcp_server,
    )
    from claude_agent_sdk.types import EffortLevel, ThinkingConfig
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `claude-agent-sdk` to use the Claude Agent provider, '
        'you can use the `claude-agent` optional group — `pip install "pydantic-ai-slim[claude-agent]"`'
    ) from _import_error

__all__ = ('ClaudeCodeSessions', 'ClaudeCodeTransport', 'CONVERSATION_HEADER', 'INHERIT_MCP_SERVERS_HEADER')

CONVERSATION_HEADER = 'x-pydantic-ai-conversation-id'
"""Carries the request's `conversation_id` from `ClaudeAgentModel` to the transport. It never leaves the process."""

INHERIT_MCP_SERVERS_HEADER = 'x-pydantic-ai-claude-agent-inherit-mcp-servers'
"""Carries `claude_agent_inherit_mcp_servers`, as JSON, from `ClaudeAgentModel` to the transport, like the above."""

_MCP_SERVER = 'pydantic_ai'
_TOOL_PREFIX = f'mcp__{_MCP_SERVER}__'

_MAX_SESSIONS = 8
"""Live sessions kept per provider; the least recently used idle one is closed to make room for another."""

_MAX_FORK_POINTS = 256
"""Finished answers a provider remembers the transcript entry of, so a new session can fork from one."""

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
    # Pydantic AI retries a failed request itself, as `max_retries` says (verified: with this, a 529 is sent once).
    'CLAUDE_CODE_MAX_RETRIES': '0',
    # Claude Code writes 1-hour cache entries on a plan by default, which cost more than they save when an agent's
    # requests come minutes apart; 5 minutes matches what the `anthropic` provider asks for.
    'CLAUDE_CODE_PROMPT_CACHE_TTL': '5m',
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


class ClaudeCodeMcpServerNotFound(Exception):
    """`claude_agent_inherit_mcp_servers` names an MCP server Claude Code doesn't have."""

    def __init__(self, unknown: list[str], known: list[str]) -> None:
        self.unknown, self.known = unknown, known


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


class _ToolChoice(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str


class _MessagesRequest(BaseModel):
    """The parts of a Messages API request body that a Claude Code session can honor."""

    model_config = ConfigDict(extra='ignore')

    model: str
    max_tokens: int
    messages: list[_Message]
    system: str | list[_TextBlock] | None = None
    tools: list[_Tool] = []
    tool_choice: _ToolChoice | None = None
    thinking: ThinkingConfig | None = None
    """Claude Code takes only the thinking configs it has a CLI flag for: a request with another can't be served."""
    output_config: _OutputConfig | None = None
    stream: bool = False
    inherit_mcp_servers: bool | list[str] = False
    """`claude_agent_inherit_mcp_servers`, from its header: the user's own MCP servers Claude Code may bring."""

    @property
    def callable_tools(self) -> list[_Tool]:
        """The tools the model may call: none when `tool_choice` says so, since Claude Code has no such setting."""
        return [] if self.tool_choice is not None and self.tool_choice.type == 'none' else self.tools


_stream_event_ta: TypeAdapter[BetaRawMessageStreamEvent] = TypeAdapter(BetaRawMessageStreamEvent)
_inherit_ta: TypeAdapter[bool | list[str]] = TypeAdapter(bool | list[str])
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
    inherit_mcp_servers: str

    @classmethod
    def of(cls, request: _MessagesRequest) -> _SessionConfig:
        system = request.system
        system_prompt = system if isinstance(system, str) else '\n\n'.join(block.text for block in system or [])
        return cls(
            model=request.model,
            system_prompt=system_prompt,
            tools=json.dumps([tool.model_dump() for tool in request.callable_tools], sort_keys=True),
            thinking=json.dumps(request.thinking, sort_keys=True),
            effort=request.output_config.effort if request.output_config else None,
            max_tokens=request.max_tokens,
            inherit_mcp_servers=json.dumps(request.inherit_mcp_servers),
        )


@dataclass(frozen=True)
class ClaudeCodeOptions:
    """How `ClaudeAgentProvider` starts Claude Code."""

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


def _error_response(
    status: int, error_type: str, message: str, retry_after: float | None = None, *, retry: bool = True
) -> httpx2.Response:
    headers = {'retry-after': str(math.ceil(retry_after))} if retry_after is not None else {}
    if not retry:
        headers['x-should-retry'] = 'false'
    body = {'type': 'error', 'error': {'type': error_type, 'message': message}}
    return httpx2.Response(status, json=body, headers=headers)


@dataclass(eq=False)
class _Turn:
    """What the session saw of the response it's streaming, for the next request to continue from."""

    blocks: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    pieces: dict[tuple[int, str], list[str]] = field(default_factory=dict[tuple[int, str], list[str]])
    """The streamed text of each block's field, joined once the block stops."""
    stop_reason: str | None = None
    fork_point: _ForkPoint | None = None
    """The transcript entry of the turn's last message, which Claude Code reports as each message ends."""
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
            self._finish_block(event.index)
        elif isinstance(event, BetaRawMessageDeltaEvent):
            self.stop_reason = event.delta.stop_reason

    def _finish_block(self, index: int) -> None:
        block = self.blocks[index]
        for key in ('text', 'thinking'):
            if (pieces := self.pieces.pop((index, key), None)) is not None:
                block[key] = f'{block.get(key, "")}{"".join(pieces)}'
        if (raw := self.pieces.pop((index, 'input'), None)) is not None:
            joined = ''.join(raw)
            block['input'] = json.loads(joined) if joined else {}

    def _apply_delta(self, block: dict[str, JsonValue], index: int, delta: BetaRawContentBlockDelta) -> None:
        if isinstance(delta, BetaTextDelta):
            self.pieces.setdefault((index, 'text'), []).append(delta.text)
        elif isinstance(delta, BetaThinkingDelta):
            self.pieces.setdefault((index, 'thinking'), []).append(delta.thinking)
        elif isinstance(delta, BetaSignatureDelta):
            block['signature'] = delta.signature
        elif isinstance(delta, BetaInputJSONDelta):  # pragma: no branch
            # Citation deltas don't reach the blocks the next request continues from: Claude Code's tools are off.
            self.pieces.setdefault((index, 'input'), []).append(delta.partial_json)

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
    allowed_servers: frozenset[str] | None = None
    """The user's own MCP servers whose tools Claude Code may run, by their names in its tool names; `None` for all."""
    server_names: dict[str, str] = field(default_factory=dict[str, str])
    """The configured name of each of the user's own MCP servers, by its name in Claude Code's tool names."""
    external_calls: set[str] = field(default_factory=set[str])
    """The IDs of the tool calls Claude Code ran itself."""
    offered_servers: list[str] = field(default_factory=list[str])
    """The configured names of the user's own MCP servers the session offers."""
    pending_results: list[ToolResultBlock] = field(default_factory=list[ToolResultBlock])
    """Results of Claude Code's own tool calls that came after the response with the call had ended."""

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

    async def pre_external_tool_use(
        self, input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Let Claude Code run a tool of one of the user's own MCP servers that the model may use."""
        assert input_data['hook_event_name'] == 'PreToolUse'
        if (external := server_and_tool(input_data['tool_name'], _TOOL_PREFIX)) is None:
            return {}
        if self.allowed_servers is not None and external[0] not in self.allowed_servers:
            reason = 'This MCP server is not one of `claude_agent_inherit_mcp_servers`.'
            return {
                'hookSpecificOutput': {
                    'hookEventName': 'PreToolUse',
                    'permissionDecision': 'deny',
                    'permissionDecisionReason': reason,
                }
            }
        return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow'}}

    def external_result(self, message: Message) -> list[ToolResultBlock]:
        """The results in `message` of tool calls Claude Code ran itself."""
        if not isinstance(message, UserMessage) or not isinstance(message.content, list):
            return []
        return [
            block
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.tool_use_id in self.external_calls
        ]

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
        frames = [
            frame for message in messages for frame in replay_blocks(message.role, message.blocks(), _TOOL_PREFIX)
        ]
        for index, (role, blocks) in enumerate(frames):
            frame = _frame(role, blocks)
            last = index == len(frames) - 1
            if role == 'user' and not last:
                frame['shouldQuery'] = False
            await self.client.query(_one_frame(frame))
            if role == 'user' and not last:
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


def _frame(role: Literal['user', 'assistant'], blocks: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    content: list[JsonValue] = list(blocks)
    message: dict[str, JsonValue] = {'role': role, 'content': content}
    return {'type': role, 'message': message, 'parent_tool_use_id': None}


async def _one_frame(frame: dict[str, JsonValue]) -> AsyncIterator[dict[str, JsonValue]]:
    yield frame


@dataclass(frozen=True)
class _ForkPoint:
    """Where a session's transcript ends with an answer: a new session forked there starts with its history cached."""

    session_id: str
    message_uuid: str


def _answer_digests(messages: Sequence[_Message]) -> list[tuple[int, str]]:
    """Identifies each prefix of `messages` that ends with an answer, with its length, the way `continuation` compares.

    The messages before the answer must match exactly; the answer itself, only in what a stored response keeps.
    """
    prefix = hashlib.sha256()
    digests: list[tuple[int, str]] = []
    for length, message in enumerate(messages, start=1):
        if message.role == 'assistant':
            answer = json.dumps(_response_projection(message.blocks()), sort_keys=True).encode()
            digests.append((length, hashlib.sha256(prefix.digest() + answer).hexdigest()))
        prefix.update(json.dumps([message.role, message.blocks()], sort_keys=True).encode() + b'\n')
    return digests


@dataclass(eq=False)
class ClaudeCodeSessions:
    """The live Claude Code sessions of one provider, by conversation."""

    options: ClaudeCodeOptions
    sessions: dict[str, _Session] = field(default_factory=dict[str, _Session])
    leases: int = 0
    transport_factory: Callable[[ClaudeAgentOptions], Transport] | None = None
    """Replaces the Claude Code process, for tests."""
    user_mcp_servers: dict[str, str] | None = None
    """The user's own MCP servers and claude.ai connectors, by their names in Claude Code's tool names, once found."""
    fork_points: dict[str, _ForkPoint] = field(default_factory=dict[str, _ForkPoint])
    """Where this provider's sessions ended an answer, by the digest of the history up to it, oldest first.

    Only sessions this provider started are forked: a fork point read from the history itself could name any of the
    user's Claude Code sessions.
    """

    def remember_fork_point(self, history: Sequence[_Message], point: _ForkPoint) -> None:
        """Remember where a session's transcript ends with `history`'s last message, an answer."""
        [*_, (_, digest)] = _answer_digests(history)
        self.fork_points.pop(digest, None)
        self.fork_points[digest] = point
        while len(self.fork_points) > _MAX_FORK_POINTS:
            del self.fork_points[next(iter(self.fork_points))]

    def _fork_point(self, messages: Sequence[_Message]) -> tuple[_ForkPoint, int] | None:
        """The latest answer of the history one of this provider's sessions ended, and the history's length to it."""
        for length, digest in reversed(_answer_digests(messages)):
            if (point := self.fork_points.get(digest)) is not None:
                return point, length
        return None

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

    def _client_options(
        self,
        session: _Session,
        request: _MessagesRequest,
        disallowed_tools: list[str],
        fork: _ForkPoint | None = None,
    ) -> ClaudeAgentOptions:
        config = session.config
        server = create_sdk_mcp_server(_MCP_SERVER, tools=[session.tool(tool) for tool in request.callable_tools])
        hooks = [HookMatcher(matcher=f'^{_TOOL_PREFIX}', hooks=[session.pre_tool_use])]
        if request.inherit_mcp_servers:
            hooks.append(HookMatcher(matcher=f'^{MCP_PREFIX}', hooks=[session.pre_external_tool_use]))
        return ClaudeAgentOptions(
            model=config.model,
            system_prompt=config.system_prompt,
            tools=[],
            mcp_servers={_MCP_SERVER: server},
            # Only `claude_agent_inherit_mcp_servers` lets Claude Code bring the user's own MCP servers and connectors.
            strict_mcp_config=not request.inherit_mcp_servers,
            disallowed_tools=disallowed_tools,
            setting_sources=[],
            skills=[],
            verbatim_prompts=True,
            include_partial_messages=True,
            hooks={'PreToolUse': hooks},
            thinking=request.thinking,
            effort=config.effort,
            cli_path=self.options.cli_path,
            cwd=self.options.cwd,
            client_app=f'pydantic-ai/{__version__}',
            env={**_ENV, 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(config.max_tokens)},
            resume=fork.session_id if fork is not None else None,
            fork_session=fork is not None,
            resume_session_at=fork.message_uuid if fork is not None else None,
        )

    async def _find_user_mcp_servers(self) -> dict[str, str]:
        """The user's own MCP servers, found once from a Claude Code process that's asked for nothing else."""
        if self.user_mcp_servers is None:
            options = ClaudeAgentOptions(
                tools=[],
                strict_mcp_config=False,
                setting_sources=[],
                skills=[],
                cli_path=self.options.cli_path,
                cwd=self.options.cwd,
                client_app=f'pydantic-ai/{__version__}',
                env=_ENV,
            )
            transport = self.transport_factory(options) if self.transport_factory is not None else None
            client = ClaudeSDKClient(options=options, transport=transport)
            await client.connect()
            try:
                servers = await _settled_mcp_servers(client)
            finally:
                await client.disconnect()
            self.user_mcp_servers = {tool_name_server(name): name for name in servers}
        return self.user_mcp_servers

    async def _inherited_servers(self, session: _Session, inherit: bool | list[str]) -> list[str]:
        """Set which of the user's own MCP servers the session offers. Returns the tools to disallow for the rest."""
        if not inherit:
            return []
        session.server_names = servers = await self._find_user_mcp_servers()
        if inherit is True:
            session.offered_servers = list(servers.values())
            return []
        allowed = {tool_name_server(name) for name in inherit}
        if unknown := sorted(name for name in inherit if tool_name_server(name) not in servers):
            raise ClaudeCodeMcpServerNotFound(unknown, sorted(servers.values()))
        session.allowed_servers = frozenset(allowed)
        session.offered_servers = [servers[server] for server in allowed]
        return [f'{MCP_PREFIX}{server}' for server in servers if server not in allowed]

    async def _start(self, request: _MessagesRequest) -> _Session:
        """A new session that has been sent the request: forked from an earlier one's transcript when it can be."""
        if (found := self._fork_point(request.messages)) is not None:
            point, length = found
            try:
                return await self._start_session(request, request.messages[length:], fork=point)
            except ResultError:
                # Claude Code no longer has the transcript, as when it was deleted: the history is replayed instead.
                self.fork_points = {key: other for key, other in self.fork_points.items() if other != point}
            except (ClaudeCodeExited, ClaudeSDKError):
                # The forked process failed for another reason; a new one is sent the whole history.
                pass
        return await self._start_session(request, request.messages)

    async def _start_session(
        self, request: _MessagesRequest, messages: list[_Message], fork: _ForkPoint | None = None
    ) -> _Session:
        session = _Session(config=_SessionConfig.of(request), loop=asyncio.get_running_loop())
        disallowed_tools = await self._inherited_servers(session, request.inherit_mcp_servers)
        options = self._client_options(session, request, disallowed_tools, fork)
        transport = self.transport_factory(options) if self.transport_factory is not None else None
        session.client = ClaudeSDKClient(options=options, transport=transport)
        try:
            await session.client.connect()
            if request.inherit_mcp_servers:
                # Claude Code loads the user's servers and connectors in the background: without them, its first
                # request would offer the model none of their tools.
                await _settled_mcp_servers(session.client, expected=session.offered_servers)
            session.messages = aiter(session.client.receive_messages())
            await session.replay(messages)
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


_MCP_STATUS_POLL = 0.25
_MCP_STATUS_WAIT = 10.0


async def _settled_mcp_servers(client: ClaudeSDKClient, expected: Collection[str] = ()) -> list[str]:
    """The names of the user's own MCP servers, once they've loaded: claude.ai connectors take a few seconds.

    Waits for at least one server, and for each of `expected`, to be listed and no longer pending.
    """
    servers: list[str] = []
    with anyio.move_on_after(_MCP_STATUS_WAIT):
        while True:
            status = [
                server for server in (await client.get_mcp_status())['mcpServers'] if server['name'] != _MCP_SERVER
            ]
            servers = [server['name'] for server in status]
            if (
                servers
                and set(expected) <= set(servers)
                and not any(server['status'] == 'pending' for server in status)
            ):
                break
            await anyio.sleep(_MCP_STATUS_POLL)
    return servers


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
        self._ended = False
        self._events = ResponseEvents(
            own_prefix=_TOOL_PREFIX, server_names=session.server_names, external_calls=session.external_calls
        )

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._opening():
            yield chunk
        try:
            async for message in self._session.messages:
                for chunk in self._message(message):
                    yield chunk
                if self._done or self._ended:
                    return
        except ClaudeSDKError as e:
            yield _stream_error(f'Claude Code failed: {e}')
        else:
            yield _stream_error('Claude Code exited before the response ended.')

    def _opening(self) -> Iterator[bytes]:
        """The events read before the response was returned, then the results left over from an earlier response."""
        session = self._session
        for event in self._first:
            yield from self._event(event)
        # Results of Claude Code's own calls from an earlier response, which ended with a call of ours.
        pending, session.pending_results = session.pending_results, []
        for block in pending:
            yield from self._result(block)

    def _message(self, message: Message) -> Iterator[bytes]:
        if isinstance(message, StreamEvent) and message.parent_tool_use_id is None:
            yield from self._event(message.event)
        elif isinstance(message, AssistantMessage):
            # Claude Code reports each message, with its transcript entry, before the response's `message_stop`, which
            # ends this stream: the last one before it is where the turn's answer ends.
            # A message without its entry, or a subagent's (none come with Claude Code's tools off), leaves none.
            top_level = message.parent_tool_use_id is None
            entry = top_level and message.session_id and message.uuid and _ForkPoint(message.session_id, message.uuid)
            self._turn.fork_point = entry or None
        elif results := self._session.external_result(message):
            for block in results:
                yield from self._result(block)
        elif isinstance(message, ResultMessage):
            # The turn ended before the response did, as when the API request fails partway.
            self._ended = True
            yield _stream_error(message.result or f'Claude Code ended with `{message.subtype}`.')

    def _event(self, event: dict[str, JsonValue]) -> Iterator[bytes]:
        for out in self._events.stream_event(_with_pydantic_ai_tool_name(event)):
            self._turn.record(out)
            if out.get('type') == 'message_stop':
                self._finish()
            yield _sse(out)

    def _result(self, block: ToolResultBlock) -> Iterator[bytes]:
        for out in self._events.tool_result(block.tool_use_id, block.content, block.is_error):
            self._turn.record(out)
            yield _sse(out)

    def _finish(self) -> None:
        self._done = True
        session, turn = self._session, self._turn
        session.seen = [*session.seen, _Message(role='assistant', content=turn.blocks)]
        session.awaiting_results = turn.tool_use_ids if turn.stop_reason == 'tool_use' else []
        session.broken |= not turn.complete
        if turn.complete and turn.fork_point is not None and not session.awaiting_results:
            # A later request that goes on from this answer can fork from it, after this session has gone.
            self._sessions.remember_fork_point(session.seen, turn.fork_point)

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
        try:
            body = _MessagesRequest.model_validate_json(await request.aread())
        except ValidationError as e:
            return _error_response(400, 'invalid_request_error', f"Claude Code can't serve this request: {e}")
        # `ClaudeAgentModel` streams every request, and a request's history ends with the user's turn.
        assert body.stream and body.messages[-1].role == 'user'
        if (inherit := request.headers.get(INHERIT_MCP_SERVERS_HEADER)) is not None:
            body.inherit_mcp_servers = _inherit_ta.validate_json(inherit)

        try:
            session, kept = await self.sessions.open(request.headers.get(CONVERSATION_HEADER), body)
        except ClaudeCodeExited:
            return _error_response(500, 'api_error', 'Claude Code exited while it was sent the history.')
        except ClaudeCodeMcpServerNotFound as e:
            message = f'Claude Code has no MCP server named {", ".join(map(repr, e.unknown))}; it has: {", ".join(map(repr, e.known)) or "none"}.'
            return _error_response(400, 'invalid_request_error', message, retry=False)
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
        session.pending_results.extend(session.external_result(message))
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
    # A plan's usage limit resets hours later, so the Anthropic client shouldn't wait for it.
    return _error_response(status, error_type, message, retry_after, retry=rate_limit is None)
