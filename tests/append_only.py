"""Drive one conversation through real provider adapters and check that its requests only ever append.

The provider prompt cache is keyed on the bytes we send: each request in a conversation must start
with the previous request's bytes, or every token after the first difference is billed and processed
again. `run_conversation` runs a fixed multi-request scenario (tool calls, then a second run that
continues from `message_history`, with the clock advancing between requests and optional state-changing
tool calls) against Anthropic, OpenAI Responses and OpenAI Chat over a mock HTTP transport that returns
scripted provider responses. `ConversationResult` then answers three questions:

- `prefix_breaks`: where a request stops starting with the previous one, ignoring the providers' own
  cache-breakpoint markers (Anthropic's `cache_control`, OpenAI's `prompt_cache_breakpoint`), which roll
  forward by design, and comparing the top-level fields that are part of the cache key;
- `unaddressable_instructions`: instruction blocks without an `id`, whose changes can only be sent by
  rewriting the system prompt rather than appended (#8188);
- `nondeterministic_projection`: whether building the same request twice from the same history, at the
  same instant, produced different bytes.

`tests/test_history_append_only.py` (core capabilities) and `tests/harness/test_history_append_only.py`
(Harness capabilities) run it over every capability the packages ship.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeGuard

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from .cassette_utils import PrefixBlock, anthropic_deferred_tool_blocks, canonical_prefix_blocks
from .conftest import try_import

Provider = Literal['anthropic', 'openai-responses', 'openai-chat']
PROVIDERS: tuple[Provider, ...] = ('anthropic', 'openai-responses', 'openai-chat')

ToolCall = tuple[str, dict[str, Any]]

START = datetime(2026, 10, 16, 9, 58, tzinfo=UTC)
"""Just before the hour, so the first clock step crosses it."""

STEP = timedelta(minutes=61)
"""How far the clock moves after each request: over an hour, so any per-hour or per-minute rendering shows."""

_ENDPOINTS: dict[Provider, str] = {
    'anthropic': '/v1/messages',
    'openai-responses': '/responses',
    'openai-chat': '/chat/completions',
}


# --- The injected clock ---------------------------------------------------------------------------

_real_datetime = datetime


@dataclass
class Clock:
    """The time every `datetime.now()` in `pydantic_ai` and `pydantic_ai_harness` sees during a scenario."""

    now: datetime = START

    def advance(self, by: timedelta = STEP) -> None:
        self.now += by


CLOCK = Clock()
"""The scenario clock, for fixtures and agent features that render the time themselves."""


class _ClockDatetime(_real_datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return CLOCK.now.astimezone(tz) if tz is not None else CLOCK.now.astimezone().replace(tzinfo=None)


def inject_clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Make `datetime.now()` read a clock the scenario advances, in every loaded library module.

    Swaps the `datetime` name in each `pydantic_ai*` module that imported the class, which is how the
    libraries read the time (`pydantic_ai._utils.now_utc` among them). `monkeypatch` restores them.
    Code that reads the time any other way (`import datetime`, `time.time()`, or a module first imported
    during the run) sees real time, which barely moves within a test, so a prefix that renders it would
    still pass: route new time reads through `pydantic_ai._utils.now_utc`.
    """
    CLOCK.now = START
    for name, module in list(sys.modules.items()):
        if name.startswith('pydantic_ai') and getattr(module, 'datetime', None) is _real_datetime:
            monkeypatch.setattr(module, 'datetime', _ClockDatetime)
    return CLOCK


# --- Scripted provider responses --------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One scripted model response: tool calls, or a final text."""

    calls: tuple[ToolCall, ...] = ()
    text: str = 'Done.'


def _anthropic_response(turn: Turn, index: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {'type': 'tool_use', 'id': f'toolu_{index}_{i}', 'name': name, 'input': args}
        for i, (name, args) in enumerate(turn.calls)
    ] or [{'type': 'text', 'text': turn.text}]
    return {
        'id': f'msg_{index}',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-opus-5-5',
        'content': content,
        'stop_reason': 'tool_use' if turn.calls else 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 5},
    }


def _anthropic_event_stream(message: dict[str, Any]) -> bytes:
    """The same message as server-sent events, for requests the SDK streams behind the scenes (large `max_tokens`)."""
    events: list[tuple[str, dict[str, Any]]] = [
        ('message_start', {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}})
    ]
    for index, block in enumerate(message['content']):
        start: dict[str, Any]
        delta: dict[str, Any]
        if block['type'] == 'tool_use':
            start, delta = (
                {**block, 'input': {}},
                {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])},
            )
        else:
            start, delta = {'type': 'text', 'text': ''}, {'type': 'text_delta', 'text': block['text']}
        events += [
            ('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': start}),
            ('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': delta}),
            ('content_block_stop', {'type': 'content_block_stop', 'index': index}),
        ]
    events += [
        (
            'message_delta',
            {
                'type': 'message_delta',
                'delta': {'stop_reason': message['stop_reason'], 'stop_sequence': None},
                'usage': {'output_tokens': 5},
            },
        ),
        ('message_stop', {'type': 'message_stop'}),
    ]
    return ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events).encode()


def _responses_response(turn: Turn, index: int) -> dict[str, Any]:
    output: list[dict[str, Any]] = [
        {
            'type': 'function_call',
            'id': f'fc_{index}_{i}',
            'call_id': f'call_{index}_{i}',
            'name': name,
            'arguments': json.dumps(args, separators=(',', ':')),
            'status': 'completed',
        }
        for i, (name, args) in enumerate(turn.calls)
    ] or [
        {
            'type': 'message',
            'id': f'msg_{index}',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': turn.text, 'annotations': []}],
        }
    ]
    return {
        'id': f'resp_{index}',
        'object': 'response',
        'created_at': 0,
        'model': 'gpt-5.6-sol',
        'status': 'completed',
        'output': output,
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 5,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 15,
        },
    }


def _chat_response(turn: Turn, index: int) -> dict[str, Any]:
    message: dict[str, Any] = {'role': 'assistant', 'content': None if turn.calls else turn.text}
    if turn.calls:
        message['tool_calls'] = [
            {
                'id': f'call_{index}_{i}',
                'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args, separators=(',', ':'))},
            }
            for i, (name, args) in enumerate(turn.calls)
        ]
    return {
        'id': f'chatcmpl_{index}',
        'object': 'chat.completion',
        'created': 0,
        'model': 'gpt-5.6-sol',
        'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if turn.calls else 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
    }


def _responses_event_stream(response: dict[str, Any]) -> bytes:
    events: list[dict[str, Any]] = [
        {'type': 'response.created', 'response': {**response, 'status': 'in_progress', 'output': []}}
    ]
    for index, item in enumerate(response['output']):
        if item['type'] == 'message':
            text = item['content'][0]['text']
            events += [
                {'type': 'response.output_item.added', 'output_index': index, 'item': {**item, 'content': []}},
                {
                    'type': 'response.output_text.delta',
                    'item_id': item['id'],
                    'output_index': index,
                    'content_index': 0,
                    'delta': text,
                    'logprobs': [],
                },
            ]
        else:
            events.append({'type': 'response.output_item.added', 'output_index': index, 'item': item})
        events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    events.append({'type': 'response.completed', 'response': response})
    return ''.join(
        f'event: {event["type"]}\ndata: {json.dumps({**event, "sequence_number": number})}\n\n'
        for number, event in enumerate(events)
    ).encode()


def _chat_event_stream(response: dict[str, Any]) -> bytes:
    choice = response['choices'][0]
    message = choice['message']
    delta: dict[str, Any] = {'role': 'assistant', 'content': message['content']}
    if tool_calls := message.get('tool_calls'):
        delta['tool_calls'] = [{**call, 'index': index} for index, call in enumerate(tool_calls)]
    base = {key: response[key] for key in ('id', 'created', 'model')} | {'object': 'chat.completion.chunk'}
    chunks: list[dict[str, Any]] = [
        {**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
        {
            **base,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}],
            'usage': response['usage'],
        },
    ]
    return (''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n').encode()


_EVENT_STREAMS: dict[Provider, Callable[[dict[str, Any]], bytes]] = {
    'anthropic': _anthropic_event_stream,
    'openai-responses': _responses_event_stream,
    'openai-chat': _chat_event_stream,
}


_RESPONSE_BUILDERS: dict[Provider, Callable[[Turn, int], dict[str, Any]]] = {
    'anthropic': _anthropic_response,
    'openai-responses': _responses_response,
    'openai-chat': _chat_response,
}


@dataclass
class _ScriptedTransport:
    """Answers each generation request with the next scripted turn, recording what was sent."""

    provider: Provider
    clock: Clock
    script: list[Turn] = field(default_factory=list[Turn])
    sent: list[bytes] = field(default_factory=list[bytes])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith(_ENDPOINTS[self.provider]), f'unexpected request to {request.url}'
        self.sent.append(request.read())
        turn = self.script.pop(0) if self.script else Turn()
        response = _RESPONSE_BUILDERS[self.provider](turn, len(self.sent))
        self.clock.advance()
        if json.loads(self.sent[-1]).get('stream'):
            stream = _EVENT_STREAMS[self.provider](response)
            return httpx2.Response(200, content=stream, headers={'content-type': 'text/event-stream'})
        return httpx2.Response(200, json=response)


class _RecordsParameters(Model):
    """Records the `ModelRequestParameters` each request is built from, after every capability hook ran."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.recorded_parameters.append(model_request_parameters)
        # Only ever mixed in ahead of a provider model, whose `request` this calls.
        return await super().request(messages, model_settings, model_request_parameters)  # pyright: ignore[reportAbstractUsage]

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        self.recorded_parameters.append(model_request_parameters)
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream


with try_import() as provider_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    class _RecordingAnthropicModel(_RecordsParameters, AnthropicModel):
        pass

    class _RecordingResponsesModel(_RecordsParameters, OpenAIResponsesModel):
        pass

    class _RecordingChatModel(_RecordsParameters, OpenAIChatModel):
        pass


def _model(provider: Provider, transport: _ScriptedTransport) -> _RecordsParameters:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(transport.handle))
    if provider == 'anthropic':
        return _RecordingAnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key='x', http_client=client))
    model_class = _RecordingResponsesModel if provider == 'openai-responses' else _RecordingChatModel
    return model_class('gpt-5.6-sol', provider=OpenAIProvider(api_key='x', http_client=client))


# --- The scenario -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """What a case adds to the default conversation.

    `state_calls` are tool calls that change the capability's state (a todo, a memory, a file); the first
    is made in the first run and the second in the continued run, alongside the default `look_up` calls.
    `look_up` builds the call that looks a topic up, for capabilities that change how tools are called.
    """

    state_calls: tuple[ToolCall, ...] = ()
    look_up: Callable[[str], ToolCall] = lambda topic: ('look_up', {'topic': topic})
    configure: Callable[[Agent[Any, Any], Clock], None] | None = None
    check_ids: bool = True
    """Whether every instruction block must have an `id`: framework blocks must, a user's own needn't."""

    def script(self) -> list[Turn]:
        first, second = (*self.state_calls, None, None)[:2]
        return [
            # The first run: two steps of tool calls, then the answer.
            Turn(calls=(self.look_up('weather'),)),
            Turn(calls=(self.look_up('news'), *([first] if first else []))),
            Turn(text='Here is the weather and the news.'),
            # The continued run.
            Turn(calls=(second or self.look_up('sports'),)),
            Turn(text='Here is more.'),
        ]


async def drain_events(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
    """An event stream handler that consumes the events, so a run streams."""
    async for _ in events:
        pass


def look_up(topic: str) -> str:
    """Look up the latest on a topic."""
    return f'The latest on {topic}: all quiet.'


@dataclass
class ConversationResult:
    provider: Provider
    conversation: list[bytes]
    """Every generation request of the two runs, in order."""
    replays: tuple[bytes, bytes]
    """The same continuation request built twice from the final history at the same instant."""
    parameters: list[ModelRequestParameters]

    def changes(self) -> list[Change]:
        """Every block of a request that the next request doesn't repeat at the same position.

        Each section must start with the previous request's blocks exactly, with new blocks only after them.
        From the first position that differs (or that the later request doesn't have), every differing position
        is a change, so an expected change can't hide another one in the same request: a block inserted ahead
        of earlier ones shows as each earlier block it displaced. New blocks are only free at the end of the last
        section the earlier request has blocks in; a tool or system block added ahead of sent messages counts. Anthropic leaves `defer_loading: true` tools
        out of its cache key, so `canonical_prefix_blocks` drops them; their wire contract is still append-only
        in first-reveal order, so they're a section of their own.
        """
        blocks = [_cache_blocks(body, self.provider) for body in self.conversation]
        # Without recorded parameters (a hand-built result), instruction changes can't be told apart.
        parameters: list[ModelRequestParameters | None] = [*self.parameters] or [None] * len(blocks)
        return [
            change
            for index, (earlier, later) in enumerate(zip(blocks, blocks[1:]), start=1)
            for change in _changes(index, earlier, later, parameters[index - 1], parameters[index])
        ]

    def unaddressable_instructions(self) -> dict[str, str]:
        """Instruction parts the model saw without an `id`, by identity (their position), with their text.

        Identified by position, not text, so two parts with the same content are two violations.
        """
        found: dict[str, str] = {}
        for parameters in self.parameters:
            for position, part in enumerate(parameters.instruction_parts or ()):
                if part.id is None and part.content.strip():
                    found.setdefault(f'instruction:anonymous@{position}', _shorten(part.content, 120))
        return found

    def nondeterministic_projection(self) -> list[Change] | None:
        """How the same request built twice from the same history differs, or `None` if the bytes are identical."""
        first, second = self.replays
        if first == second:
            return None
        earlier, later = (_cache_blocks(body, self.provider) for body in (first, second))
        return _changes(1, earlier, later, None, None) or [
            Change(1, 'request', 0, 'request', *_around_difference(first.decode(), second.decode()), in_tail=False)
        ]


async def run_conversation(
    provider: Provider,
    capabilities: Sequence[AbstractCapability[Any]],
    scenario: Scenario,
    clock: Clock,
) -> ConversationResult:
    transport = _ScriptedTransport(provider, clock, script=scenario.script())
    model = _model(provider, transport)
    agent = Agent(
        model,
        instructions='You are a helpful assistant.',
        toolsets=[FunctionToolset([look_up], id='look_up')],
        name='append-only',
        capabilities=list(capabilities),
    )
    if scenario.configure is not None:
        scenario.configure(agent, clock)

    first = await agent.run('What is the weather and the news?')
    # Streamed, so a streamed continuation is checked against the requests a non-streamed run built.
    second = await agent.run('And what else?', message_history=first.all_messages(), event_stream_handler=drain_events)
    conversation = list(transport.sent)
    parameters = list(model.recorded_parameters)

    history = second.all_messages()
    replay_at = clock.now
    replays: list[bytes] = []
    for _ in range(2):
        clock.now = replay_at
        transport.script = [Turn(text='Nothing else.')]
        await agent.run('Anything more?', message_history=history)
        replays.append(transport.sent[-1])

    assert len(parameters) == len(conversation)
    return ConversationResult(provider, conversation, (replays[0], replays[1]), parameters)


# --- Comparing requests ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """A position in a section of request `request_index - 1` that request `request_index` doesn't repeat."""

    request_index: int
    section: str
    """`settings`, `tools`, `system` (with any leading system or developer messages), `messages`,
    `deferred-tools`, or `request` when the bytes differ outside them."""
    block_index: int
    """The position within the section of the earlier request, or of the later one for a block inserted there."""
    identity: str
    """What changed, from structure rather than text: in `system`, the ids of the instruction parts that changed
    (`instruction:<id>`, or `instruction:anonymous@<position>` for a part without one), or `system-prompt` when
    none did; `tool:<name>`; `setting:<field>`; in `messages`, the earlier part's kind (`user-prompt`,
    `tool-return`, `model-response`, `system-prompt`)."""
    earlier: str
    later: str
    in_tail: bool
    """Whether it's the last message of the previous request: a request-only tail was dropped or replaced."""

    def __str__(self) -> str:
        return (
            f'request {self.request_index} changed {self.section} block {self.block_index} ({self.identity}) of '
            f'request {self.request_index - 1}:\n    before: {self.earlier}\n    after:  {self.later}'
        )


_CACHE_MARKERS = frozenset({'cache_control', 'prompt_cache_breakpoint'})
"""Anthropic's and OpenAI's explicit cache breakpoints, which roll forward with each request by design."""


def _strip_cache_markers(body: dict[str, Any], *, unwrap: bool) -> dict[str, Any]:
    """Drop cache breakpoints from where providers put them, and undo the wrapping that placing one needs.

    Markers sit on tool definitions, system blocks, messages and their content parts (one level deeper for an
    Anthropic `tool_result`), so only those are stripped: the same key inside tool arguments or a JSON schema
    is data, and a change to it is a real prefix change. With `unwrap` (OpenAI), a single text part that carried
    the marker is read back as the string it stands for: OpenAI's breakpoint sits on a content part, so a
    plain-string message body or tool output that gets one is sent as a single text part instead (with the text
    unchanged), and goes back to a string once the breakpoint moves on. An unmarked array stays an array, so a
    real switch between array and string still counts.
    """
    stripped = dict(body)
    for section in ('tools', 'system', 'messages', 'input'):
        if isinstance(items := stripped.get(section), list):
            stripped[section] = [_strip_item(item, unwrap=unwrap) for item in items]  # pyright: ignore[reportUnknownVariableType]
    return stripped


def _strip_item(item: Any, *, unwrap: bool, depth: int = 0) -> dict[str, Any]:
    assert is_str_dict(item), f'expected a JSON object, got {item!r}'
    stripped = {key: value for key, value in item.items() if key not in _CACHE_MARKERS}
    for key in ('content', 'output'):
        if depth < 2 and _is_list(parts := stripped.get(key)):
            marked = (
                unwrap
                and len(parts) == 1
                and is_str_dict(first := parts[0])
                and not first.keys().isdisjoint(_CACHE_MARKERS)
            )
            parts = [_strip_item(part, unwrap=unwrap, depth=depth + 1) for part in parts]
            # Only a text part that carried a marker was wrapped for it; an unmarked array is what the request sends.
            text_part = marked and parts[0].keys() == {'type', 'text'} and parts[0]['type'] in ('text', 'input_text')
            stripped[key] = parts[0]['text'] if text_part else parts
    return stripped


def _is_list(value: Any) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


_CACHE_KEYED_SETTINGS: dict[Provider, tuple[str, ...]] = {
    'anthropic': ('model', 'tool_choice', 'thinking'),
    'openai-responses': ('model', 'tool_choice', 'reasoning', 'prompt_cache_key'),
    'openai-chat': ('model', 'tool_choice', 'reasoning_effort', 'prompt_cache_key'),
}
"""Top-level request fields that are part of the provider's cache key: changing one misses the cache like a prefix edit.

Anthropic documents `tool_choice` and the thinking parameters as invalidating the message cache; OpenAI caches per
model, per `prompt_cache_key` (requests are routed by it) and, as measured, per reasoning effort.
"""


def _cache_blocks(body: bytes, provider: Provider) -> list[PrefixBlock]:
    # Only OpenAI wraps a string body in a text part to carry its marker; Anthropic's markers change no shapes.
    parsed = _strip_cache_markers(json.loads(body), unwrap=provider != 'anthropic')
    canonical = canonical_prefix_blocks(parsed, f'https://api.example.com{_ENDPOINTS[provider]}')
    assert canonical is not None
    settings = {key: parsed[key] for key in _CACHE_KEYED_SETTINGS[provider] if key in parsed}
    deferred = (
        [('deferred-tools', tool) for tool in anthropic_deferred_tool_blocks(parsed)] if provider == 'anthropic' else []
    )
    blocks: list[PrefixBlock] = [('settings', json.dumps(settings))]
    # Leading system or developer messages are the standing system prompt on the APIs that send it as messages.
    standing = True
    for level, block in canonical[1]:
        standing = standing and (level != 'messages' or _is_system_message(block))
        blocks.append(('system' if level == 'messages' and standing else level, block))
    return [*blocks, *deferred]


def _changes(
    index: int,
    earlier: list[PrefixBlock],
    later: list[PrefixBlock],
    earlier_parameters: ModelRequestParameters | None,
    later_parameters: ModelRequestParameters | None,
) -> list[Change]:
    changes: list[Change] = []
    instructions = ', '.join(_changed_instructions(earlier_parameters, later_parameters)) or 'system-prompt'
    # Sections in the order the later request serializes them, then any it no longer has.
    sections = list(dict.fromkeys(level for level, _ in [*later, *earlier]))
    for position, section in enumerate(sections):
        before = [block for level, block in earlier if level == section]
        after = [block for level, block in later if level == section]
        for i, old in enumerate(before):
            new = after[i] if i < len(after) else '<missing>'
            if new == old:
                continue
            identity = instructions if section == 'system' else _block_identity(section, old, new)
            changes.append(
                Change(
                    index,
                    section,
                    i,
                    identity,
                    *_around_difference(old, new),
                    in_tail=section == 'messages' and i == len(before) - 1,
                )
            )
        # Appending is only free in the last section the earlier request has blocks in: a new tool or system
        # block goes ahead of the messages it already sent, so it moves their place in the prefix. Anthropic's
        # deferred tools are outside its cache key, so where they sit doesn't matter.
        if any(level in sections[position + 1 :] and level != 'deferred-tools' for level, _ in earlier):
            for i in range(len(before), len(after)):
                identity = instructions if section == 'system' else _block_identity(section, after[i], after[i])
                changes.append(Change(index, section, i, identity, '<missing>', _shorten(after[i], 120), in_tail=False))
    return changes


def _changed_instructions(earlier: ModelRequestParameters | None, later: ModelRequestParameters | None) -> list[str]:
    """The instruction parts whose content differs between two requests: by id, or by position without one."""
    if earlier is None or later is None:
        return []

    def by_identity(parameters: ModelRequestParameters) -> dict[str, str]:
        return {
            f'instruction:{part.id}' if part.id is not None else f'instruction:anonymous@{position}': part.content
            for position, part in enumerate(parameters.instruction_parts or ())
        }

    before, after = by_identity(earlier), by_identity(later)
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


def _block_identity(section: str, block: str, other: str) -> str:
    """What a changed block is, from its structure: a tool's name, a setting's field, or a history part's kind."""
    value = json.loads(block)
    assert is_str_dict(value), f'expected a JSON object, got {block!r}'
    if section == 'settings':
        other_value = json.loads(other)
        assert is_str_dict(other_value), f'expected a JSON object, got {other!r}'
        keys = sorted(value.keys() | other_value.keys())
        return ', '.join(f'setting:{key}' for key in keys if value.get(key) != other_value.get(key))
    if section in ('tools', 'deferred-tools'):
        function = value.get('function')
        name = function.get('name') if is_str_dict(function) else value.get('name') or value.get('type')
        return f'tool:{name}'
    role, kind, content = value.get('role'), value.get('type'), value.get('content')
    if role in ('system', 'developer'):
        return 'system-prompt'
    if role == 'tool' or kind == 'function_call_output':
        return 'tool-return'
    if role == 'user':
        is_tool_result = _is_list(content) and any(
            is_str_dict(part) and part.get('type') == 'tool_result' for part in content
        )
        return 'tool-return' if is_tool_result else 'user-prompt'
    return 'model-response'


def _is_system_message(block: str) -> bool:
    message = json.loads(block)
    return is_str_dict(message) and message.get('role') in ('system', 'developer')


def _around_difference(a: str, b: str, context: int = 100) -> tuple[str, str]:
    """The two strings around the first character where they differ."""
    offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, offset - context)
    prefix = '…' if start else ''
    return prefix + a[start : offset + context], prefix + b[start : offset + context]


def _shorten(text: str, limit: int) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= limit else f'{flat[: limit - 1]}…'


# --- Discovery and the expectation tables --------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRef:
    """A capability class found in source, imported only when a test runs it."""

    name: str
    module: str
    """The module that defines it."""

    def load(self) -> type[AbstractCapability[Any]]:
        """Import the class. Gate optional dependencies before calling this: any `ImportError` here is a failure."""
        cls = getattr(importlib.import_module(self.module), self.name)
        assert _is_capability_class(cls), f'`{self.module}.{self.name}` is not a capability class'
        return cls


def discover_capabilities(packages: Sequence[str]) -> dict[str, CapabilityRef]:
    """Every public capability class defined in `packages`, by class name, found by reading their source.

    Reads the source rather than an export list, so a new capability is a new case without anyone
    registering it, and rather than importing it, so collection never imports an integration's SDK.
    Class definitions are parsed with `ast`, so multi-line and nested-generic bases are followed.
    A class is a capability when its bases lead to `AbstractCapability`, following base names across
    `pydantic_ai` and `packages`. It is public when it isn't `_`-prefixed and is importable under its name
    from its module path with the private segments dropped: `pydantic_ai.capabilities._tool_search.ToolSearch`
    is public because `pydantic_ai.capabilities` imports `ToolSearch`.
    """
    bases: dict[str, set[str]] = {}
    defined: dict[str, list[str]] = {}
    for package in ('pydantic_ai', *packages):
        for module, tree in _module_trees(package):
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    bases.setdefault(node.name, set()).update(_base_names(node))
                    if package in packages:
                        defined.setdefault(module, []).append(node.name)

    capabilities = {'AbstractCapability'}
    while new := {name for name, names in bases.items() if name not in capabilities and names & capabilities}:
        capabilities |= new

    found: dict[str, CapabilityRef] = {}
    for module, names in defined.items():
        public_module = '.'.join(segment for segment in module.split('.') if not segment.startswith('_'))
        exported = _exported_names(public_module) if public_module != module else set(names)
        for name in names:
            if name in capabilities and not name.startswith('_') and name in exported:
                ref = CapabilityRef(name, module)
                # Cases and table entries are keyed by class name, so two public capabilities sharing one
                # would silently test only the second.
                assert name not in found, f'two public capabilities are named `{name}`: {found[name].module}, {module}'
                found[name] = ref
    return found


def _module_trees(package: str) -> Iterator[tuple[str, ast.Module]]:
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations is not None
    root = Path(next(iter(spec.submodule_search_locations)))
    for path in sorted(root.rglob('*.py')):
        parts = path.relative_to(root).with_suffix('').parts
        module = '.'.join((package, *(parts[:-1] if parts[-1] == '__init__' else parts)))
        yield module, ast.parse(path.read_text(encoding='utf-8'))


def _base_names(node: ast.ClassDef) -> set[str]:
    """The unqualified names of a class's bases, without their type arguments."""
    names: set[str] = set()
    for base in node.bases:
        while isinstance(base, ast.Subscript):
            base = base.value
        names.add(ast.unparse(base).rsplit('.', 1)[-1])
    return names


def _exported_names(module: str) -> set[str]:
    """Names a module imports or defines at the top level, read from its source."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:  # pragma: no cover
        return set()
    tree = ast.parse(Path(spec.origin).read_text(encoding='utf-8'))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def _is_capability_class(obj: object) -> TypeGuard[type[AbstractCapability[Any]]]:
    return inspect.isclass(obj) and issubclass(obj, AbstractCapability)


def skip_unless_installed(name: str, optional_dependencies: Mapping[str, str]) -> None:
    """Skip the case for `name` when the module its optional dependency group provides isn't installed.

    Checked with `find_spec` rather than imported, since several of these SDKs are kept out of collection
    (`TID253`). Only capabilities listed here may be skipped: any other `ImportError` fails the test.
    """
    module = optional_dependencies.get(name)
    if module is not None and importlib.util.find_spec(module) is None:  # pragma: lax no cover
        pytest.skip(f'`{name}` needs `{module}`, from an optional dependency group that is not installed')


Fixture = Callable[[Path], 'AbstractCapability[Any] | Sequence[AbstractCapability[Any]]']
"""Builds a configured capability, plus any it depends on (a workspace, say), in the test's `tmp_path`."""


def build_capabilities(
    name: str, load: Callable[[], type[AbstractCapability[Any]]], fixture: Fixture | None, tmp_path: Path
) -> list[AbstractCapability[Any]]:
    """The capability under test with its defaults, or as its fixture builds it."""
    if fixture is not None:
        built = fixture(tmp_path)
        return [built] if isinstance(built, AbstractCapability) else list(built)
    cls = load()
    try:
        return [cls()]
    except (TypeError, ValueError, UserError) as exc:
        pytest.fail(
            f'`{name}()` cannot be built with its defaults ({exc}). Add a `FIXTURES` entry that configures it, '
            "or an `EXEMPT` entry saying why it can't run offline.",
            pytrace=False,
        )


@dataclass(frozen=True)
class Exempt:
    """A capability the test can't construct generically, with the reason."""

    reason: str


Check = Literal['prefix', 'ids', 'determinism']


@dataclass(frozen=True)
class Known:
    """Expected failures of one check, linked to the issue or PR that fixes them.

    An entry explains exactly `count` violations of `check` in `section` with the structural `identity` (see
    `Change.identity`; the `ids` check uses section `instructions` and `instruction:anonymous@<position>`), on
    each of `providers`. Matching never reads model-visible text, which untrusted content can imitate. One
    violation more or fewer fails the test, so a known break can't absorb a new one.
    """

    check: Check
    issue: str
    section: str
    identity: str
    count: int = 1
    providers: tuple[Provider, ...] = PROVIDERS


@dataclass(frozen=True)
class Allowed:
    """A documented prefix mover, checked exactly: it must move the prefix, and only as documented.

    Every change must be one of `changes`, named by `(section, identity)` like `Change`: there are no
    whole-section wildcards, so an allowed move can't carry an unrelated one along. A compaction-style mover may
    change the prefix in at most `max_moves` requests. A `tail` mover sends request-only content at the end of
    each request (behind a cache breakpoint) and drops it from the next, so it may change only the last message.
    """

    reason: str
    changes: tuple[tuple[str, str], ...]
    max_moves: int = 1
    tail: bool = False
    providers: tuple[Provider, ...] = PROVIDERS
    """The providers this mover applies to; on the others the capability must keep the prefix."""


@dataclass(frozen=True)
class Violation:
    section: str
    identity: str
    report: str


def _change_violations(changes: Sequence[Change], prefix: str = '') -> list[Violation]:
    return [Violation(change.section, change.identity, f'{prefix}{change}') for change in changes]


def _violations(
    result: ConversationResult, *, allowed: Allowed | None, check_ids: bool
) -> dict[Check, tuple[str, list[Violation]]]:
    """Every violation of each check, as a summary and one item per violation."""
    violations: dict[Check, tuple[str, list[Violation]]] = {}
    changes = result.changes()
    if allowed is not None and result.provider not in allowed.providers:
        allowed = None
    prefix: list[Violation] = []
    if allowed is None:
        prefix = _change_violations(changes)
    elif not changes:
        prefix = [
            Violation('-', 'allowed', f'is listed in `ALLOWED` ({allowed.reason}) but no longer moves the prefix')
        ]
    else:
        # Each change is checked on its own, so an allowed move can't carry another one along.
        outside = [change for change in changes if (change.section, change.identity) not in allowed.changes]
        prefix += _change_violations(outside, 'is not one of the changes its `ALLOWED` entry permits: ')
        if allowed.tail:
            deep = [change for change in changes if not change.in_tail and change not in outside]
            prefix += _change_violations(deep, 'may only replace the last message of each request, but ')
        elif (moves := len({change.request_index for change in changes})) > allowed.max_moves:
            prefix.append(
                Violation(
                    '-',
                    'allowed',
                    f'moved the prefix in {moves} requests, more than the {allowed.max_moves} its `ALLOWED` entry permits',
                )
            )
    if prefix:
        violations['prefix'] = ('broke the append-only prefix', prefix)
    if check_ids and (unaddressable := result.unaddressable_instructions()):
        violations['ids'] = (
            'contributed instruction blocks without an `id`, so changes to them rewrite the prefix',
            [Violation('instructions', identity, f'{identity}: {text}') for identity, text in unaddressable.items()],
        )
    if (differences := result.nondeterministic_projection()) is not None:
        violations['determinism'] = (
            'built the same request twice from the same history with different bytes',
            _change_violations(differences),
        )
    return violations


def check_conversation(
    name: str,
    result: ConversationResult,
    *,
    known: Sequence[Known] = (),
    allowed: Allowed | None = None,
    check_ids: bool = True,
) -> None:
    """Fail on any violation not listed in `known`, and on any `known` entry that no longer fails."""
    violations = _violations(result, allowed=allowed, check_ids=check_ids)

    # Each violation must be matched by an entry for its identity with room for it, so a known break can't
    # absorb a new one in the same conversation.
    applicable = [k for k in known if result.provider in k.providers]
    uses = [0] * len(applicable)
    unexpected: list[str] = []
    for check, (summary, items) in violations.items():
        unmatched: list[str] = []
        for item in items:
            index = next(
                (
                    i
                    for i, k in enumerate(applicable)
                    if k.check == check
                    and (k.section, k.identity) == (item.section, item.identity)
                    and uses[i] < k.count
                ),
                None,
            )
            if index is None:
                unmatched.append(item.report)
            else:
                uses[index] += 1
        if unmatched:
            unexpected.append(f'[{check}] {name} on {result.provider} {summary}:\n  ' + '\n  '.join(unmatched))
    expected = [f'[{k.check}] {k.issue}' for k, used in zip(applicable, uses) if used]
    stale = [
        f'[{k.check}] {k.issue} (explained {used} of {k.count})' for k, used in zip(applicable, uses) if used != k.count
    ]
    if unexpected or stale:
        message = '\n\n'.join(unexpected)
        if stale:
            message += (
                f'\n\n{name} on {result.provider} no longer fails these checks as expected: fixed, or failing '
                f'differently. Update or remove these `KNOWN` entries: {stale}'
            )
        pytest.fail(message.strip(), pytrace=False)
    if expected:
        pytest.xfail(f'known: {", ".join(expected)}')
