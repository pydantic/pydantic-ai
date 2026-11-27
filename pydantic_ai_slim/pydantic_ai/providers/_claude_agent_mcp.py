"""Claude Code's own MCP tool calls, as the Messages API's MCP connector blocks that `AnthropicModel` reads.

With `claude_agent_inherit_mcp_servers`, Claude Code also offers the model the user's own MCP servers and claude.ai
connectors, and runs their tools itself: a turn that calls one is several API calls inside Claude Code, the tool's
result coming in between as a user message. `ResponseEvents` turns those into the events of one response, with each
call and result as the `mcp_tool_use` and `mcp_tool_result` blocks of Anthropic's MCP connector, which `AnthropicModel`
maps to `NativeToolCallPart` and `NativeToolReturnPart`. `replay_blocks` turns them back for a new session.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

MCP_PREFIX = 'mcp__'

_USAGE_FIELDS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def tool_name_server(name: str) -> str:
    """The MCP server name as Claude Code writes it in its tool names, `mcp__<server>__<tool>`."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def server_and_tool(name: str, own_prefix: str) -> tuple[str, str] | None:
    """The server and tool of a call Claude Code runs itself: one of an MCP server other than Pydantic AI's."""
    if not name.startswith(MCP_PREFIX) or name.startswith(own_prefix):
        return None
    server, separator, tool = name.removeprefix(MCP_PREFIX).partition('__')
    return (server, tool) if separator else None


def _result_content(content: str | list[dict[str, JsonValue]] | None) -> list[JsonValue]:
    """An MCP connector result's content: its text blocks, with a note for anything else."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    return [
        block if block.get('type') == 'text' else {'type': 'text', 'text': f'[{block.get("type")} content]'}
        for block in content
    ]


@dataclass
class ResponseEvents:
    """Turns the API calls of one Claude Code turn into the server-sent events of one response."""

    own_prefix: str
    server_names: Mapping[str, str]
    """The configured name of each server, by the name Claude Code writes in its tool names."""
    external_calls: set[str]
    """The IDs of the calls Claude Code runs itself, shared with the session so a later response can find them."""
    offset: int = 0
    started: bool = False
    continuing: bool = False
    """Whether the last API call stopped for Claude Code's own tool calls alone, so the response goes on."""
    _blocks: int = 0
    _calls_ours: bool = False
    _calls_external: bool = False
    _usage: dict[str, int] = field(default_factory=dict[str, int])

    def stream_event(self, event: dict[str, JsonValue]) -> Iterator[dict[str, JsonValue]]:
        match event.get('type'):
            case 'message_start':
                self._blocks, self._calls_ours, self._calls_external, self.continuing = 0, False, False, False
                if not self.started:
                    self.started = True
                    yield event
            case 'content_block_start':
                yield self._block_start(event)
            case 'content_block_delta' | 'content_block_stop':
                yield self._shifted(event)
            case 'message_delta':
                yield from self._message_delta(event)
            case 'message_stop':
                if self.continuing:
                    self.offset += self._blocks
                else:
                    yield event
            case _:  # pragma: no cover
                yield event

    def tool_result(
        self, tool_use_id: str, content: str | list[dict[str, JsonValue]] | None, is_error: bool | None
    ) -> Iterator[dict[str, JsonValue]]:
        """The result of one of Claude Code's own tool calls, as an MCP connector result block."""
        block: dict[str, JsonValue] = {
            'type': 'mcp_tool_result',
            'tool_use_id': tool_use_id,
            'is_error': bool(is_error),
            'content': _result_content(content),
        }
        yield {'type': 'content_block_start', 'index': self.offset, 'content_block': block}
        yield {'type': 'content_block_stop', 'index': self.offset}
        self.offset += 1

    def _shifted(self, event: dict[str, JsonValue]) -> dict[str, JsonValue]:
        index = event.get('index')
        assert isinstance(index, int)
        self._blocks = max(self._blocks, index + 1)
        return {**event, 'index': index + self.offset}

    def _block_start(self, event: dict[str, JsonValue]) -> dict[str, JsonValue]:
        event = self._shifted(event)
        block = event.get('content_block')
        if not isinstance(block, dict) or block.get('type') != 'tool_use' or not isinstance(block.get('name'), str):
            return event
        name = str(block['name'])
        if (external := server_and_tool(name, self.own_prefix)) is None:
            self._calls_ours = True
            return event
        self._calls_external = True
        self.external_calls.add(str(block.get('id')))
        server, tool = external
        mcp_block: dict[str, JsonValue] = {
            'type': 'mcp_tool_use',
            'id': block.get('id'),
            'name': tool,
            'server_name': self.server_names.get(server, server),
            'input': {},
        }
        return {**event, 'content_block': mcp_block}

    def _message_delta(self, event: dict[str, JsonValue]) -> Iterator[dict[str, JsonValue]]:
        delta, usage = event.get('delta'), event.get('usage')
        stop_reason = delta.get('stop_reason') if isinstance(delta, dict) else None
        this_call = {
            name: value
            for name in _USAGE_FIELDS
            if isinstance(usage, dict) and isinstance(value := usage.get(name), int)
        }
        self.continuing = stop_reason == 'tool_use' and self._calls_external and not self._calls_ours
        if self.continuing:
            for name, value in this_call.items():
                self._usage[name] = self._usage.get(name, 0) + value
            return
        if self._usage and isinstance(usage, dict):
            # `AnthropicModel` reads the last usage as the whole response's, so it's the sum of its API calls.
            usage = {**usage, **{name: self._usage.get(name, 0) + this_call.get(name, 0) for name in _USAGE_FIELDS}}
            event = {**event, 'usage': usage}
        yield event


def replay_blocks(
    role: Literal['user', 'assistant'], blocks: list[dict[str, JsonValue]], own_prefix: str
) -> Iterator[tuple[Literal['user', 'assistant'], list[dict[str, JsonValue]]]]:
    """The frames a message is sent to a new session in: Claude Code's own calls and results as its tool blocks."""
    if role == 'user':
        yield role, blocks
        return
    frames: list[tuple[Literal['user', 'assistant'], list[dict[str, JsonValue]]]] = [('assistant', [])]
    for block in blocks:
        match block.get('type'):
            case 'tool_use':
                frames[-1][1].append({**block, 'name': f'{own_prefix}{block.get("name")}'})
            case 'mcp_tool_use':
                server = tool_name_server(str(block.get('server_name')))
                call: dict[str, JsonValue] = {
                    'type': 'tool_use',
                    'id': block.get('id'),
                    'name': f'{MCP_PREFIX}{server}__{block.get("name")}',
                    'input': block.get('input'),
                }
                frames[-1][1].append(call)
            case 'mcp_tool_result':
                result: dict[str, JsonValue] = {
                    'type': 'tool_result',
                    'tool_use_id': block.get('tool_use_id'),
                    'content': block.get('content'),
                    'is_error': bool(block.get('is_error')),
                }
                frames.append(('user', [result]))
                frames.append(('assistant', []))
            case _:
                frames[-1][1].append(block)
    yield from (frame for frame in frames if frame[1])
