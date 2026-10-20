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
import re
import sys
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterator, Sequence
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

from .cassette_utils import PrefixBlock, canonical_prefix_blocks
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

    def prefix_breaks(self) -> list[PrefixBreak]:
        blocks = [_cache_blocks(body, self.provider) for body in self.conversation]
        return [
            prefix_break
            for index, (earlier, later) in enumerate(zip(blocks, blocks[1:]), start=1)
            if (prefix_break := _first_divergence(index, earlier, later)) is not None
        ]

    def unaddressable_instructions(self) -> list[str]:
        """Distinct instruction blocks the model saw that carry no `id`."""
        seen: dict[str, None] = {}
        for parameters in self.parameters:
            for part in parameters.instruction_parts or ():
                if part.id is None and part.content.strip():
                    seen.setdefault(_shorten(part.content, 120))
        return list(seen)

    def nondeterministic_projection(self) -> str | None:
        first, second = self.replays
        if first == second:
            return None
        return _describe_byte_difference(first, second)


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
class PrefixBreak:
    """Request `request_index` does not start with request `request_index - 1`."""

    request_index: int
    section: str
    block_index: int
    earlier: str
    later: str
    in_tail: bool
    """Whether only the last block of the previous request changed: a request-only tail was dropped or replaced."""

    def __str__(self) -> str:
        return (
            f'request {self.request_index} rewrote {self.section} block {self.block_index} of request '
            f'{self.request_index - 1}:\n    before: {self.earlier}\n    after:  {self.later}'
        )


_CACHE_MARKERS = frozenset({'cache_control', 'prompt_cache_breakpoint'})
"""Anthropic's and OpenAI's explicit cache breakpoints, which roll forward with each request by design."""


def _strip_cache_markers(value: Any) -> Any:
    """Drop cache breakpoints, and undo the wrapping that placing one needs.

    OpenAI's breakpoint sits on a content part, so a plain-string message body or tool output that gets one
    is sent as a single text part instead (with the text unchanged), and goes back to a string once the
    breakpoint moves on. With the marker gone, that single part is read back as the string it stands for.
    """
    if is_str_dict(value):
        stripped = {key: _strip_cache_markers(item) for key, item in value.items() if key not in _CACHE_MARKERS}
        for key in ('content', 'output'):
            if (text := _single_text_part(stripped.get(key))) is not None:
                stripped[key] = text
        return stripped
    if isinstance(value, list):
        return [_strip_cache_markers(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _single_text_part(content: Any) -> str | None:
    if isinstance(content, list) and len(content) == 1:  # pyright: ignore[reportUnknownArgumentType]
        part = content[0]  # pyright: ignore[reportUnknownVariableType]
        if is_str_dict(part) and part.keys() == {'type', 'text'} and part['type'] in ('text', 'input_text'):
            return part['text']
    return None


_CACHE_KEYED_SETTINGS: dict[Provider, tuple[str, ...]] = {
    'anthropic': ('model', 'tool_choice', 'thinking'),
    'openai-responses': ('model', 'tool_choice', 'reasoning'),
    'openai-chat': ('model', 'tool_choice', 'reasoning_effort'),
}
"""Top-level request fields that are part of the provider's cache key: changing one misses the cache like a prefix edit.

Anthropic documents `tool_choice` and the thinking parameters as invalidating the message cache; OpenAI caches per
model and, as measured, per reasoning effort.
"""


def _cache_blocks(body: bytes, provider: Provider) -> list[PrefixBlock]:
    parsed = _strip_cache_markers(json.loads(body))
    canonical = canonical_prefix_blocks(parsed, f'https://api.example.com{_ENDPOINTS[provider]}')
    assert canonical is not None
    settings = {key: parsed[key] for key in _CACHE_KEYED_SETTINGS[provider] if key in parsed}
    return [('settings', json.dumps(settings)), *canonical[1]]


def _first_divergence(index: int, earlier: list[PrefixBlock], later: list[PrefixBlock]) -> PrefixBreak | None:
    if later[: len(earlier)] == earlier:
        return None
    block_index = next((i for i, (a, b) in enumerate(zip(earlier, later)) if a != b), min(len(earlier), len(later)))
    before = earlier[block_index] if block_index < len(earlier) else ('-', '<missing>')
    after = later[block_index] if block_index < len(later) else ('-', '<missing>')
    section = before[0] if before[0] != '-' else after[0]
    before_text, after_text = _around_difference(before[1], after[1])
    return PrefixBreak(index, section, block_index, before_text, after_text, in_tail=block_index == len(earlier) - 1)


def _around_difference(a: str, b: str, context: int = 100) -> tuple[str, str]:
    """The two strings around the first character where they differ."""
    offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, offset - context)
    prefix = '…' if start else ''
    return prefix + a[start : offset + context], prefix + b[start : offset + context]


def _describe_byte_difference(a: bytes, b: bytes) -> str:
    before, after = _around_difference(a.decode(errors='replace'), b.decode(errors='replace'))
    return f'\n    first:  {before}\n    second: {after}'


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
        """Import the class, skipping the test when its optional dependency group isn't installed."""
        try:
            module = importlib.import_module(self.module)
        except ImportError as exc:  # pragma: lax no cover
            pytest.skip(f'`{self.name}` needs an optional dependency that is not installed: {exc}')
        cls = getattr(module, self.name)
        assert _is_capability_class(cls), f'`{self.module}.{self.name}` is not a capability class'
        return cls


def discover_capabilities(packages: Sequence[str]) -> dict[str, CapabilityRef]:
    """Every public capability class defined in `packages`, by class name, found by reading their source.

    Reads the source rather than an export list, so a new capability is a new case without anyone
    registering it, and rather than importing it, so collection never imports an integration's SDK.
    A class is a capability when its bases lead to `AbstractCapability`, following base names across
    `pydantic_ai` and `packages`. It is public when it isn't `_`-prefixed and is importable under its name
    from its module path with the private segments dropped: `pydantic_ai.capabilities._tool_search.ToolSearch`
    is public because `pydantic_ai.capabilities` imports `ToolSearch`.
    """
    bases: dict[str, set[str]] = {}
    defined: dict[str, list[str]] = {}
    for package in ('pydantic_ai', *packages):
        for module, source in _module_sources(package):
            for match in _CLASS_STATEMENT.finditer(source):
                name = match['name']
                bases.setdefault(name, set()).update(_base_names(match['bases']))
                if package in packages:
                    defined.setdefault(module, []).append(name)

    capabilities = {'AbstractCapability'}
    while new := {name for name, names in bases.items() if name not in capabilities and names & capabilities}:
        capabilities |= new

    found: dict[str, CapabilityRef] = {}
    for module, names in defined.items():
        public_module = '.'.join(segment for segment in module.split('.') if not segment.startswith('_'))
        exported = _exported_names(public_module) if public_module != module else set(names)
        for name in names:
            if name in capabilities and not name.startswith('_') and name in exported:
                found[name] = CapabilityRef(name, module)
    return found


_CLASS_STATEMENT = re.compile(r'^class\s+(?P<name>\w+)\s*\((?P<bases>[^)]*)\)\s*:', re.MULTILINE)
"""A top-level class statement. A regex rather than `ast`, which takes most of a second over both packages."""


def _module_sources(package: str) -> Iterator[tuple[str, str]]:
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations is not None
    root = Path(next(iter(spec.submodule_search_locations)))
    for path in sorted(root.rglob('*.py')):
        parts = path.relative_to(root).with_suffix('').parts
        module = '.'.join((package, *(parts[:-1] if parts[-1] == '__init__' else parts)))
        yield module, path.read_text(encoding='utf-8')


def _base_names(bases: str) -> set[str]:
    """The unqualified names of a class statement's bases, without their type arguments or keywords.

    Only the base list goes through `ast` (parsing whole modules is what the regex avoids), so nested type
    arguments like `AbstractCapability[dict[str, Any]]` resolve to `AbstractCapability`.
    """
    statement = ast.parse(f'class _({bases}): pass').body[0]
    assert isinstance(statement, ast.ClassDef)
    names: set[str] = set()
    for base in statement.bases:
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
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
    """An expected failure of one check, linked to the issue or PR that fixes it.

    `match` must appear in the failure report, so a new, different failure of the same check still fails.
    """

    check: Check
    issue: str
    match: str
    providers: tuple[Provider, ...] = PROVIDERS


@dataclass(frozen=True)
class Allowed:
    """A documented prefix mover, checked exactly: it must move the prefix, and only as documented.

    A compaction-style mover may rewrite the prefix at most `max_moves` times. A `tail` mover sends request-only
    content at the end of each request (behind a cache breakpoint) and drops it from the next, so it may change
    the last block of every request and nothing before it.
    """

    reason: str
    max_moves: int = 1
    tail: bool = False
    providers: tuple[Provider, ...] = PROVIDERS
    """The providers this mover applies to; on the others the capability must keep the prefix."""


def _violations(
    result: ConversationResult, *, allowed: Allowed | None, check_ids: bool
) -> dict[Check, tuple[str, list[str]]]:
    """Every violation of each check, as a summary and one item per violation."""
    violations: dict[Check, tuple[str, list[str]]] = {}
    breaks = result.prefix_breaks()
    if allowed is not None and result.provider not in allowed.providers:
        allowed = None
    if allowed is None:
        if breaks:
            violations['prefix'] = ('rewrote the prefix', [str(b) for b in breaks])
    elif not breaks:
        violations['prefix'] = (
            'no longer moves the prefix',
            [f'is listed in `ALLOWED` ({allowed.reason}) but no longer moves the prefix'],
        )
    elif allowed.tail:
        if deep := [str(b) for b in breaks if not b.in_tail]:
            violations['prefix'] = (
                'may only replace the last block of each request, but rewrote earlier history',
                deep,
            )
    elif len(breaks) > allowed.max_moves:
        violations['prefix'] = (
            f'moved the prefix {len(breaks)} times, more than the {allowed.max_moves} its `ALLOWED` entry permits',
            [str(b) for b in breaks],
        )
    if check_ids and (unaddressable := result.unaddressable_instructions()):
        violations['ids'] = (
            'contributed instruction blocks without an `id`, so changes to them rewrite the prefix',
            unaddressable,
        )
    if difference := result.nondeterministic_projection():
        violations['determinism'] = (
            'built the same request twice from the same history with different bytes',
            [difference.strip()],
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

    # Each violation must be matched by some entry, so a known break can't hide an unrelated one in the
    # same conversation.
    unexpected: list[str] = []
    used: set[int] = set()
    for check, (summary, items) in violations.items():
        unmatched: list[str] = []
        for item in items:
            index = next(
                (
                    i
                    for i, k in enumerate(known)
                    if k.check == check and result.provider in k.providers and k.match in item
                ),
                None,
            )
            if index is None:
                unmatched.append(item)
            else:
                used.add(index)
        if unmatched:
            unexpected.append(f'[{check}] {name} on {result.provider} {summary}:\n  ' + '\n  '.join(unmatched))
    expected = [f'[{known[i].check}] {known[i].issue}' for i in sorted(used)]
    stale = [f'[{k.check}] {k.issue}' for i, k in enumerate(known) if result.provider in k.providers and i not in used]
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
