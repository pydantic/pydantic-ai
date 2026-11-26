"""A scripted stand-in for the Claude Code process, speaking its stream-json protocol to the real Claude Agent SDK.

Claude Code's traffic is a subprocess's stdin and stdout, not HTTP, so it can't be recorded as a cassette. This
transport plays the CLI's side of the protocol instead: it answers the SDK's `initialize` control request, acknowledges
history frames sent with `shouldQuery: false`, and answers each queried user message with the next scripted response,
as raw Messages API stream events. For a response that ends in tool calls it does what Claude Code does: for each call
in turn it runs the `PreToolUse` hook the SDK registered, calls the tool on the SDK's in-process MCP server, and
continues the turn with the next response once the tool returns. The event and message shapes follow the frames
Claude Code 2.1.332 produced in live runs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from claude_agent_sdk import ClaudeAgentOptions, CLIConnectionError, CLINotFoundError, ProcessError, Transport


@dataclass
class FakeResponse:
    """One Messages API response: text and `tool_use` blocks, or an error the CLI reports instead."""

    blocks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    stop_reason: str = 'end_turn'
    usage: dict[str, int] = field(default_factory=lambda: {'input_tokens': 10, 'output_tokens': 5})
    error: str | None = None
    """An `AssistantMessage.error` kind, such as `'authentication_failed'`, ending the turn without a response."""
    error_text: str = 'error'
    rate_limit_resets_at: int | None = None
    cut: Literal['error', 'exit', 'crash'] | None = None
    """After the response has started, the turn fails (as when the API request breaks off), the process exits, or
    reading from it fails."""
    exit: bool = False
    """The process exits instead of responding."""
    crash: BaseException | None = None
    """Reading from the process fails with this error instead of responding."""


def text(content: str) -> dict[str, Any]:
    return {'type': 'text', 'text': content}


def thinking(content: str, signature: str) -> dict[str, Any]:
    return {'type': 'thinking', 'thinking': content, 'signature': signature}


def tool_use(tool_use_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {'type': 'tool_use', 'id': tool_use_id, 'name': f'mcp__pydantic_ai__{name}', 'input': arguments}


@dataclass
class FakeClaudeCode:
    """The scripted responses, shared by every session a test starts, and what each session was sent."""

    responses: list[FakeResponse]
    sessions: list[FakeSession] = field(default_factory=list['FakeSession'])
    exit_on_history: bool = False
    """Each process exits when it's sent history to replay."""
    missing: bool = False
    """There's no Claude Code executable to start."""

    def __call__(self, options: ClaudeAgentOptions) -> FakeSession:
        session = FakeSession(self, options)
        self.sessions.append(session)
        return session


class FakeSession(Transport):
    def __init__(self, script: FakeClaudeCode, options: ClaudeAgentOptions) -> None:
        self.script = script
        self.options = options
        self.frames: list[dict[str, Any]] = []
        """The user and assistant frames the SDK sent, history replay included."""
        self.tool_results: list[dict[str, Any]] = []
        self.closed = False
        self.write_error: BaseException | None = None
        """Writing to the process fails with this error, as when it fails between requests."""
        self._outbox: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._hook_ids: list[str] = []
        self._counter = 0
        self._tasks: set[asyncio.Task[None]] = set()

    async def connect(self) -> None:
        if self.script.missing:
            raise CLINotFoundError('Claude Code not found')

    def is_ready(self) -> bool:  # pragma: no cover
        return not self.closed

    async def end_input(self) -> None:  # pragma: no cover
        pass

    async def close(self) -> None:
        self.closed = True
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._outbox.put_nowait(None)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while (message := await self._outbox.get()) is not None:
            if isinstance(message, BaseException):
                raise message
            yield message

    async def write(self, data: str) -> None:
        if self.closed:
            raise CLIConnectionError('ProcessTransport is not ready for writing')
        if self.write_error is not None:
            raise self.write_error
        message = json.loads(data)
        match message['type']:
            case 'control_request':
                request = message['request']
                assert request['subtype'] == 'initialize'
                hooks: dict[str, list[dict[str, Any]]] = request.get('hooks') or {}
                for matcher in hooks.get('PreToolUse', []):
                    self._hook_ids.extend(matcher['hookCallbackIds'])
                self._send({'type': 'control_response', 'response': self._success(message['request_id'], {})})
            case 'control_response':
                response = message['response']
                self._pending.pop(response['request_id']).set_result(response)
            case 'user' | 'assistant':
                self.frames.append(message)
                if message['type'] == 'user':
                    if message.get('shouldQuery') is False:
                        if self.script.exit_on_history:
                            self._outbox.put_nowait(None)
                            return
                        self._send(self._init())
                        self._send(self._result(num_turns=0))
                    else:
                        task = asyncio.create_task(self._turn())
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
            case other:  # pragma: no cover
                raise AssertionError(f'unexpected frame {other!r}')

    def _send(self, message: dict[str, Any]) -> None:
        self._outbox.put_nowait(message)

    @staticmethod
    def _init() -> dict[str, Any]:
        return {'type': 'system', 'subtype': 'init', 'session_id': 'fake-session', 'apiKeySource': 'none'}

    @staticmethod
    def _success(request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        return {'subtype': 'success', 'request_id': request_id, 'response': response}

    @staticmethod
    def _result(*, num_turns: int, is_error: bool = False, result: str | None = None) -> dict[str, Any]:
        return {
            'type': 'result',
            'subtype': 'error_during_execution' if is_error else 'success',
            'duration_ms': 1,
            'duration_api_ms': 1,
            'is_error': is_error,
            'num_turns': num_turns,
            'session_id': 'fake-session',
            'result': result,
        }

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        request_id = f'cli_{self._counter}'
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._send({'type': 'control_request', 'request_id': request_id, 'request': request})
        response = await future
        assert response['subtype'] == 'success', response
        return response['response']

    async def _mcp(self, method: str, params: dict[str, Any], *, notification: bool = False) -> dict[str, Any]:
        message: dict[str, Any] = {'jsonrpc': '2.0', 'method': method, 'params': params}
        if not notification:
            message['id'] = self._counter + 1
        response = await self._request({'subtype': 'mcp_message', 'server_name': 'pydantic_ai', 'message': message})
        return response['mcp_response']

    async def _turn(self) -> None:
        await self._mcp(
            'initialize',
            {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'fake', 'version': '0'}},
        )
        await self._mcp('notifications/initialized', {}, notification=True)
        self._send(self._init())
        while True:
            response = self.script.responses.pop(0)
            if response.exit or response.crash is not None:
                self._outbox.put_nowait(response.crash)
                return
            if response.error is not None:
                self._report_error(response)
                return
            if response.cut is not None:
                self._stream(response, cut=True)
                if response.cut == 'exit':
                    self._outbox.put_nowait(None)
                elif response.cut == 'crash':
                    self._outbox.put_nowait(ProcessError('Command failed', exit_code=1))
                else:
                    self._send(self._result(num_turns=1, is_error=True, result='Connection error.'))
                return
            self._stream(response)
            if response.stop_reason != 'tool_use':
                self._send(self._result(num_turns=1))
                return
            for block in response.blocks:
                if block['type'] == 'tool_use':
                    await self._call_tool(block)

    def _report_error(self, response: FakeResponse) -> None:
        if response.rate_limit_resets_at is not None:
            info = {'status': 'rejected', 'resetsAt': response.rate_limit_resets_at, 'rateLimitType': 'five_hour'}
            self._send({'type': 'rate_limit_event', 'rate_limit_info': info, 'uuid': 'u', 'session_id': 'fake-session'})
        message = {'role': 'assistant', 'model': 'claude', 'content': [text(response.error_text)]}
        self._send({'type': 'assistant', 'message': message, 'parent_tool_use_id': None, 'error': response.error})
        self._send(self._result(num_turns=1, is_error=True, result=response.error_text))

    def _stream(self, response: FakeResponse, *, cut: bool = False) -> None:
        message: dict[str, Any] = {
            'id': 'msg_fake',
            'type': 'message',
            'role': 'assistant',
            'model': self.options.model,
            'content': [],
            'stop_reason': None,
            'stop_sequence': None,
            'usage': response.usage,
        }
        events: list[dict[str, Any]] = [{'type': 'message_start', 'message': message}]
        for index, block in enumerate(response.blocks):
            if block['type'] == 'text':
                events += [
                    {'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}},
                    {
                        'type': 'content_block_delta',
                        'index': index,
                        'delta': {'type': 'text_delta', 'text': block['text']},
                    },
                ]
            elif block['type'] not in ('tool_use', 'thinking'):
                # A block type newer than the `anthropic` types, followed by a delta type they do know.
                events += [
                    {'type': 'content_block_start', 'index': index, 'content_block': {'type': block['type']}},
                    {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': ''}},
                ]
            elif block['type'] == 'thinking':
                start = {'type': 'thinking', 'thinking': '', 'signature': ''}
                events += [
                    {'type': 'content_block_start', 'index': index, 'content_block': start},
                    {
                        'type': 'content_block_delta',
                        'index': index,
                        'delta': {'type': 'thinking_delta', 'thinking': block['thinking']},
                    },
                    {
                        'type': 'content_block_delta',
                        'index': index,
                        'delta': {'type': 'signature_delta', 'signature': block['signature']},
                    },
                ]
            else:
                start: dict[str, Any] = {**block, 'input': {}}
                delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
                events += [
                    {'type': 'content_block_start', 'index': index, 'content_block': start},
                    {'type': 'content_block_delta', 'index': index, 'delta': delta},
                ]
            events.append({'type': 'content_block_stop', 'index': index})
        if cut:
            events = events[:1]
        else:
            stop = {'stop_reason': response.stop_reason, 'stop_sequence': None}
            events.append(
                {'type': 'message_delta', 'delta': stop, 'usage': {'output_tokens': response.usage['output_tokens']}}
            )
        for event in events:
            self._send({'type': 'stream_event', 'uuid': 'u', 'session_id': 'fake-session', 'event': event})
        if cut:
            return
        # Claude Code reports the whole message before its `message_stop` event.
        assistant = {**message, 'content': response.blocks, 'stop_reason': response.stop_reason}
        self._send({'type': 'assistant', 'message': assistant, 'parent_tool_use_id': None})
        self._send(
            {'type': 'stream_event', 'uuid': 'u', 'session_id': 'fake-session', 'event': {'type': 'message_stop'}}
        )

    async def _call_tool(self, block: dict[str, Any]) -> None:
        hook_input = {
            'hook_event_name': 'PreToolUse',
            'session_id': 'fake-session',
            'transcript_path': '',
            'cwd': '',
            'tool_name': block['name'],
            'tool_input': block['input'],
            'tool_use_id': block['id'],
        }
        for callback_id in self._hook_ids:
            output = await self._request(
                {
                    'subtype': 'hook_callback',
                    'callback_id': callback_id,
                    'input': hook_input,
                    'tool_use_id': block['id'],
                }
            )
            assert output['hookSpecificOutput']['permissionDecision'] == 'allow'
        name = block['name'].removeprefix('mcp__pydantic_ai__')
        result = await self._mcp('tools/call', {'name': name, 'arguments': block['input']})
        self.tool_results.append({'tool_use_id': block['id'], **result['result']})
