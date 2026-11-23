"""Replayed conversations send each response's provider fields back, and read the prompt cache they write.

`fail_replay_fidelity_violations` in `tests/conftest.py` checks every test that replays a cassette, once it passed, and
fails it on:

- `field-not-echoed`: a response the cassette played carries an opaque field `ECHO_FIELDS` lists, such as a thinking
  signature, encrypted reasoning or a tool call id, and the request the code sends next in that conversation doesn't
  carry the same value back. That request is read from what the code sends now, cassetter's `sent_requests`, not from
  the recording, so an adapter that stops sending a field back fails without a re-recording. Requests are paired by
  message history, not by their order in the file. The code's own rendering of the played request is the sent request
  to the same endpoint that shares the most leading messages with it, since the two can differ by an id the code
  generates afresh on every run. The next request is the shortest longer sent request that extends that rendering, or
  shares the most leading messages with it when the next request drops a request-scoped message, and the field must
  be in the messages after the shared ones. An OpenAI Responses request whose `previous_response_id` names the played
  response carries all of that response's fields, and a function tool call id that a profile's `tool_call_id_length`
  rewrites counts as sent back in its rewritten form.
- `cache-not-read`: an Anthropic or Bedrock request with explicit cache markers wrote the prompt cache, the next
  recorded request with markers extends its prefix, markers aside, with equal `CACHE_KEY_PARAMETERS`, the prefix the
  first one cached is at least the model's `CACHE_MINIMUM_TOKENS`, and the second read nothing from the cache. This
  reads the recorded usage, so it fails a cassette recorded while a request shape busted the cache. A thinking request
  that adds a user prompt is left out: the provider may drop earlier thinking from the prompt, which moves the prefix.
  Implicit caching (OpenAI, Gemini) is best effort, so it is out.

`ECHO_FIELDS` classifies every wire shape the check knows: the fields its responses carry that the next request sends
back, each with the adapter function that does it, or `Exempt` with the reason the shape is left out. The comment
above it lists the opaque fields left out of the table on purpose.

A test whose conversation drops a field or loses a cache read on purpose, for example by continuing history on a model
that can't take the field, takes `@pytest.mark.drops_replay_field(reason=...)`. A marked test whose conversation has
no violation fails, so the marker can't outlive the drop.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import unquote, urlparse

import pytest
from cassetter import Body, Cassette, HttpInteraction, RecordedRequest

from pydantic_ai._utils import is_list, is_str_dict
from pydantic_ai.models._tool_call_ids import hashed_tool_call_id

from .cassette_hooks import normalize_uri
from .cassette_utils import PrefixBlock, canonical_prefix_blocks, classify_prefix_pair, is_new_user_turn
from .manifests import Exempt

WireShape = Literal['anthropic', 'openai-chat', 'openai-responses', 'google', 'bedrock', 'cohere', 'xai']
"""The provider APIs whose conversations the check knows. `_wire_shape` tells apart the HTTP ones; xAI's is gRPC."""

EchoName = Literal[
    'thinking.signature',
    'redacted_thinking.data',
    'tool_use.id',
    'server_tool_use.id',
    'server_tool_use.input',
    'reasoning.encrypted_content',
    'function_call.call_id',
    'tool_calls.id',
    'tool_calls.extra_content',
    'thoughtSignature',
    'functionCall.id',
    'reasoningContent.signature',
    'reasoningContent.redactedContent',
    'toolUse.toolUseId',
    'tool_plan',
]

Echo = tuple[EchoName, str]
"""A field and its value, normalized so a response and a request carrying the same value compare equal."""

ReplayFidelityCode = Literal['field-not-echoed', 'cache-not-read']


@dataclass(frozen=True)
class EchoField:
    """An opaque field a response carries, which the next request in the conversation sends back unchanged."""

    response: str
    """Where the field sits in the response."""
    request: str
    """Where the next request carries it back."""
    sender: str
    """The adapter function that puts it in the next request."""
    reason: str
    """What the provider does with it."""
    function_tool_call_id: bool = False
    """Whether the field is a function tool call id.

    When the model's profile sets `tool_call_id_length` and the id doesn't fit it, as Mistral's does,
    `Model.prepare_messages` sends `hashed_tool_call_id` of the id back instead, which counts as sending it back.
    """


@dataclass(frozen=True)
class Echoes:
    """The fields of one wire shape's responses that the next request sends back."""

    reason: str
    fields: Mapping[EchoName, EchoField]


# Opaque fields left out on purpose:
# - OpenAI Responses item `id`s: they point the server at its stored copy of an item, so the adapter sends them only
#   for the same provider and while `openai_send_reasoning_ids` is on, and a stateless request needs none.
# - Anthropic `container.id`: it goes back only while the code execution tool is enabled.
# - The message-level `extra_content` of Gemini through Chat Completions: Gemini requires a thought signature back only
#   on function calls, which `tool_calls[].extra_content` carries.
# - Groq and Cerebras reasoning: plain text, which each model profile replays its own way, or not at all.
ECHO_FIELDS: dict[WireShape, Echoes | Exempt] = {
    'anthropic': Echoes(
        reason='The Messages API verifies each thinking block by its signature, and pairs each tool result with its call by id',
        fields={
            'thinking.signature': EchoField(
                response='content[type=thinking].signature',
                request='messages[role=assistant].content[type=thinking].signature',
                sender='AnthropicModel._map_thinking_part',
                reason='Anthropic rejects a thinking block whose signature is missing or changed, and needs the block back '
                'beside the tool use it led to',
            ),
            'redacted_thinking.data': EchoField(
                response='content[type=redacted_thinking].data',
                request='messages[role=assistant].content[type=redacted_thinking].data',
                sender='AnthropicModel._map_thinking_part',
                reason='Redacted thinking is nothing but this encrypted payload, which Anthropic decrypts to restore it',
            ),
            'tool_use.id': EchoField(
                response='content[type=tool_use].id',
                request='messages[role=assistant].content[type=tool_use].id',
                sender='AnthropicModel._map_model_response',
                reason='Each `tool_result` names its `tool_use` by this id',
                function_tool_call_id=True,
            ),
            'server_tool_use.id': EchoField(
                response='content[type=server_tool_use].id',
                request='messages[role=assistant].content[type=server_tool_use].id',
                sender='AnthropicModel._map_native_tool_call_part',
                reason='The server tool result block that follows names its call by this id',
            ),
            'server_tool_use.input': EchoField(
                response='content[type=server_tool_use].{id,input}',
                request='messages[role=assistant].content[type=server_tool_use].{id,input}',
                sender='AnthropicModel._map_native_tool_call_part',
                reason='A replayed server tool call carries the input the server ran it with; compared by value, with its id',
            ),
        },
    ),
    'openai-responses': Echoes(
        reason='A stateless Responses request carries the reasoning and function calls of earlier turns itself, unless '
        '`previous_response_id` points the server at them',
        fields={
            'reasoning.encrypted_content': EchoField(
                response='output[type=reasoning].encrypted_content, streamed in `response.output_item.done`',
                request='input[type=reasoning].encrypted_content',
                sender='OpenAIResponsesModel._map_thinking_to_reasoning_item',
                reason='Without stored responses, the encrypted reasoning is the only way the model gets its reasoning back',
            ),
            'function_call.call_id': EchoField(
                response='output[type=function_call].call_id, streamed in `response.output_item.done`',
                request='input[type=function_call].call_id',
                sender='OpenAIResponsesModel._map_function_call',
                reason='Each `function_call_output` names its call by this id',
                function_tool_call_id=True,
            ),
        },
    ),
    'openai-chat': Echoes(
        reason='Chat Completions, and the APIs that copy it such as Mistral, pair each tool message with its assistant '
        'tool call by id',
        fields={
            'tool_calls.id': EchoField(
                response='choices[0].message.tool_calls[].id',
                request='messages[role=assistant].tool_calls[].id',
                sender='OpenAIChatModel._map_tool_call',
                reason='Each tool message names its call by `tool_call_id`',
                function_tool_call_id=True,
            ),
            'tool_calls.extra_content': EchoField(
                response='choices[0].message.tool_calls[].extra_content',
                request='messages[role=assistant].tool_calls[].extra_content',
                sender='OpenAIChatModel._MapModelResponseContext._map_response_tool_call_part',
                reason="Gemini's OpenAI-compatible API carries a function call's thought signature here, and requires it "
                'back; compared by value',
            ),
        },
    ),
    'google': Echoes(
        reason='Gemini validates the thought signature on each part that carried one, and pairs function responses with '
        'their calls by id',
        fields={
            'thoughtSignature': EchoField(
                response='candidates[0].content.parts[].thoughtSignature',
                request='contents[role=model].parts[].thoughtSignature',
                sender='google._content_model_response',
                reason='Gemini rejects a function call that lost its signature, and loses the reasoning behind any other '
                'part that did; base64 and base64url compare equal',
            ),
            'functionCall.id': EchoField(
                response='candidates[0].content.parts[].functionCall.id',
                request='contents[role=model].parts[].functionCall.id',
                sender='google._function_call_part_dict',
                reason='Each `functionResponse` names its call by this id',
                function_tool_call_id=True,
            ),
        },
    ),
    'bedrock': Echoes(
        reason='Converse verifies Claude reasoning by its signature, and pairs each tool result with its call by '
        '`toolUseId`',
        fields={
            'reasoningContent.signature': EchoField(
                response='output.message.content[].reasoningContent.reasoningText.signature',
                request='messages[role=assistant].content[].reasoningContent.reasoningText.signature',
                sender='BedrockConverseModel._map_thinking_part',
                reason='Claude on Bedrock rejects replayed reasoning whose signature is missing or changed',
            ),
            'reasoningContent.redactedContent': EchoField(
                response='output.message.content[].reasoningContent.redactedContent',
                request='messages[role=assistant].content[].reasoningContent.redactedContent',
                sender='BedrockConverseModel._map_thinking_part',
                reason='Redacted reasoning is nothing but these encrypted bytes, which the model decrypts to restore it',
            ),
            'toolUse.toolUseId': EchoField(
                response='output.message.content[].toolUse.toolUseId',
                request='messages[role=assistant].content[].toolUse.toolUseId',
                sender='BedrockConverseModel._map_tool_call',
                reason='Each `toolResult` names its call by this id',
                function_tool_call_id=True,
            ),
        },
    ),
    'cohere': Echoes(
        reason='Cohere tool use sends the assistant tool-call message back, plan included, and pairs tool results with '
        'their calls by id',
        fields={
            'tool_plan': EchoField(
                response='message.tool_plan, streamed in `tool-plan-delta`',
                request='messages[role=assistant].tool_plan',
                sender='CohereModel._map_model_response',
                reason='https://docs.cohere.com/docs/tool-use-overview sends the plan back beside the tool calls it led to',
            ),
            'tool_calls.id': EchoField(
                response='message.tool_calls[].id, streamed in `tool-call-start`',
                request='messages[role=assistant].tool_calls[].id',
                sender='CohereModel._map_tool_call',
                reason='Each tool message names its call by `tool_call_id`',
                function_tool_call_id=True,
            ),
        },
    ),
    'xai': Exempt(
        '`XaiModel` talks gRPC, and its cassettes replay through `tests/models/xai_proto_cassettes.py`, not cassetter, '
        'so there is no sent request to read'
    ),
}

# Request parameters that are part of the provider's prompt-cache key, by the dotted path they sit at: two requests
# that differ in one don't share a cache entry. Whether any image is in the prompt counts too. See "What invalidates
# the cache" in https://platform.claude.com/docs/en/build-with-claude/prompt-caching; Bedrock passes Claude's
# parameters in `additionalModelRequestFields`, and its model is in the URL.
CACHE_KEY_PARAMETERS: dict[WireShape, tuple[str, ...]] = {
    'anthropic': ('model', 'tool_choice', 'thinking', 'output_config', 'speed', 'context_management'),
    'bedrock': ('toolConfig.toolChoice', 'additionalModelRequestFields'),
}

# The fewest tokens a cached prefix can hold, by a model name fragment; the longest fragment in the model id applies,
# and a model not listed gets the largest minimum. Claude's are from "Minimum cacheable prompt length" in
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching, which the table in
# https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html repeats for the Claude models on Bedrock and
# extends with Nova's.
CACHE_MINIMUM_TOKENS: dict[str, int] = {
    'claude-fable-5-2': 512,
    'claude-mythos-5-2': 512,
    'claude-fable-5-1': 512,
    'claude-mythos-5-1': 512,
    'claude-opus-5-5': 512,
    'claude-opus-5': 512,
    'claude-sonnet-5-5': 512,
    'claude-fable-5': 512,
    'claude-mythos-5': 512,
    'claude-haiku-5-5': 512,
    'claude-mythos-preview': 2048,
    'claude-opus-4-7': 2048,
    'claude-opus-4-6': 4096,
    'claude-opus-4-5': 4096,
    'claude-haiku-4-5': 4096,
    'claude-opus-4-8': 1024,
    'claude-sonnet-5': 1024,
    'claude-sonnet-4-6': 1024,
    'claude-sonnet-4-5': 1024,
    'claude-opus-4-1': 1024,
    'claude-opus-4': 1024,
    'claude-sonnet-4': 1024,
    'nova': 1024,
}


@dataclass(frozen=True)
class EchoCheck:
    """Whether the next request in a conversation carried back one field of a played response."""

    shape: WireShape
    field: EchoName
    interaction: int
    """The index of the played interaction whose response carried the field."""
    value: str
    echoed: bool


@dataclass(frozen=True)
class ReplayFidelityViolation:
    code: ReplayFidelityCode
    shape: WireShape
    interaction: int
    """The index of the played interaction whose response the violation is about."""
    detail: str


@dataclass(frozen=True)
class _Request:
    shape: WireShape
    endpoint: tuple[str, str]
    """The host and path, normalized as cassette matching does, with streaming and non-streaming paths alike."""
    body: dict[str, object]
    history: list[object]


@dataclass(frozen=True)
class _Response:
    fields: list[Echo] = field(default_factory=list[Echo])
    response_id: str | None = None
    cache_read: int = 0
    cache_write: int = 0


@dataclass(frozen=True)
class _CacheTurn:
    interaction: int
    blocks: list[PrefixBlock]
    key: str
    """The cache-key parameters, serialized."""
    model: str
    thinking: bool
    response: _Response


def _wire_shape(uri: str, body: Mapping[str, object]) -> WireShape | None:
    """The provider API an HTTP request with this URI and JSON body speaks, if the check knows it."""
    parsed = urlparse(uri)
    host, path = parsed.hostname or '', parsed.path
    if path.endswith('/v1/messages') or (host == 'api.anthropic.com' and path.endswith('/messages')):
        return 'anthropic'
    if path.endswith((':rawPredict', ':streamRawPredict')) and 'anthropic_version' in body:
        return 'anthropic'
    if path.endswith('/chat/completions'):
        return 'openai-chat'
    if path.endswith('/responses'):
        return 'openai-responses'
    if path.endswith((':generateContent', ':streamGenerateContent')):
        return 'google'
    if path.endswith(('/converse', '/converse-stream')):
        return 'bedrock'
    if 'cohere' in host and path.endswith('/v2/chat'):
        return 'cohere'
    return None


def _parse_request(method: str, uri: str, body: object) -> _Request | None:
    if method.upper() != 'POST' or not is_str_dict(body) or (shape := _wire_shape(uri, body)) is None:
        return None
    parsed = urlparse(normalize_uri(uri))
    path = unquote(parsed.path).replace(':streamGenerateContent', ':generateContent')
    path = path.replace(':streamRawPredict', ':rawPredict').removesuffix('-stream')
    history = body.get({'google': 'contents', 'openai-responses': 'input'}.get(shape, 'messages'))
    return _Request(
        shape,
        (parsed.hostname or '', path),
        body,
        history if is_list(history) else [history] if isinstance(history, str) else [],
    )


def _body_content(body: Body) -> object:
    """A recorded body: parsed JSON, text (an event stream), or bytes (an AWS event stream)."""
    if body.body_type == 'text' and isinstance(body.content, str) and body.content.lstrip().startswith(('{', '[')):
        return _json_or_none(body.content)
    return body.content


def _json_or_none(text: str | bytes) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _base64(value: str) -> str:
    """`value` with base64url's alphabet mapped to base64's and the padding dropped."""
    return value.replace('-', '+').replace('_', '/').rstrip('=')


def _events(body: object) -> list[object]:
    """A response body as the list of JSON documents it streams: one for a plain JSON response."""
    if isinstance(body, str):
        return [
            event
            for line in body.splitlines()
            if line.startswith('data:') and (event := _json_or_none(line[5:].strip())) is not None
        ]
    if is_list(body):
        return list(body)
    return [body]


def _eventstream_payloads(data: bytes) -> Iterator[dict[str, object]]:
    """The JSON payloads of an AWS event stream.

    Each message is a prelude of its total length, its headers' length and a checksum (4 bytes each), the headers, the
    payload, and a checksum. A payload's keys say its event type, so the headers are skipped.
    """
    offset = 0
    while offset + 16 <= len(data):
        total, headers = struct.unpack_from('>II', data, offset)
        if total < 16:
            return
        payload = _json_or_none(data[offset + 12 + headers : offset + total - 4])
        if is_str_dict(payload):
            yield payload
        offset += total


def _usage_tokens(usage: object, key: str) -> int:
    value = usage.get(key) if is_str_dict(usage) else None
    return value if isinstance(value, int) else 0


def _anthropic_response(body: object) -> _Response:
    if is_str_dict(body):
        content = body.get('content')
        blocks: list[object] = list(content) if is_list(content) else []
        usage: dict[str, object] = dict(body.get('usage') or {})
    else:
        blocks, usage = _fold_anthropic_stream(body)
    fields: list[Echo] = []
    for block in blocks:
        fields += _anthropic_block_echoes(block)
    return _Response(
        fields,
        cache_read=_usage_tokens(usage, 'cache_read_input_tokens'),
        cache_write=_usage_tokens(usage, 'cache_creation_input_tokens'),
    )


def _fold_anthropic_stream(body: object) -> tuple[list[object], dict[str, object]]:
    """The content blocks and usage a streamed Anthropic message built."""
    blocks: dict[int, dict[str, object]] = {}
    inputs: dict[int, str] = defaultdict(str)
    usage: dict[str, object] = {}
    for event in _events(body):
        if not is_str_dict(event):
            continue
        index, delta = event.get('index'), event.get('delta')
        if event.get('type') == 'message_start' and is_str_dict(message := event.get('message')):
            usage.update(message.get('usage') or {})
        elif event.get('type') == 'message_delta' and is_str_dict(update := event.get('usage')):
            usage.update({key: value for key, value in update.items() if value is not None})
        elif isinstance(index, int) and is_str_dict(block := event.get('content_block')):
            blocks[index] = dict(block)
        elif isinstance(index, int) and index in blocks and is_str_dict(delta):
            if delta.get('type') == 'signature_delta':
                blocks[index]['signature'] = str(blocks[index].get('signature') or '') + str(
                    delta.get('signature') or ''
                )
            elif delta.get('type') == 'input_json_delta':
                inputs[index] += str(delta.get('partial_json') or '')
    for index, partial in inputs.items():
        if partial:
            blocks[index]['input'] = _json_or_none(partial)
    return [blocks[index] for index in sorted(blocks)], usage


def _anthropic_block_echoes(block: object) -> list[Echo]:
    if not is_str_dict(block):
        return []
    kind, block_id = block.get('type'), block.get('id')
    if kind == 'thinking' and isinstance(signature := block.get('signature'), str) and signature:
        return [('thinking.signature', signature)]
    if kind == 'redacted_thinking' and isinstance(data := block.get('data'), str):
        return [('redacted_thinking.data', data)]
    if kind == 'tool_use' and isinstance(block_id, str):
        return [('tool_use.id', block_id)]
    if kind == 'server_tool_use' and isinstance(block_id, str):
        return [('server_tool_use.id', block_id), ('server_tool_use.input', _canonical([block_id, block.get('input')]))]
    return []


def _anthropic_request(history: list[object]) -> set[Echo]:
    echoes: set[Echo] = set()
    for message in history:
        if is_str_dict(message) and message.get('role') == 'assistant' and is_list(content := message.get('content')):
            for block in content:
                echoes.update(_anthropic_block_echoes(block))
    return echoes


def _responses_response(body: object) -> _Response:
    response: object = body
    done: dict[int, object] = {}
    if not (is_str_dict(body) and 'output' in body):
        for event in _events(body):
            if is_str_dict(event) and is_str_dict(event.get('response')):
                response = event['response']
            elif (
                is_str_dict(event)
                and event.get('type') == 'response.output_item.done'
                and isinstance(index := event.get('output_index'), int)
            ):
                done[index] = event.get('item')
    if not is_str_dict(response):
        return _Response()
    # A stream's parts are built from `response.output_item.done`; `response.completed` may re-encrypt the reasoning.
    output = [done[index] for index in sorted(done)] if done else response.get('output')
    response_id = response.get('id')
    return _Response(
        _responses_item_echoes(output if is_list(output) else []),
        response_id=response_id if isinstance(response_id, str) else None,
    )


def _responses_item_echoes(items: list[object]) -> list[Echo]:
    echoes: list[Echo] = []
    for item in items:
        if not is_str_dict(item):
            continue
        if item.get('type') == 'reasoning' and isinstance(content := item.get('encrypted_content'), str) and content:
            echoes.append(('reasoning.encrypted_content', content))
        elif item.get('type') == 'function_call' and isinstance(call_id := item.get('call_id'), str):
            echoes.append(('function_call.call_id', call_id))
    return echoes


def _responses_request(history: list[object]) -> set[Echo]:
    return set(_responses_item_echoes(history))


def _chat_response(body: object) -> _Response:
    calls: list[dict[str, object]] = []
    by_index: dict[int, dict[str, object]] = {}
    for event in _events(body):
        choices = event.get('choices') if is_str_dict(event) else None
        choice = choices[0] if is_list(choices) and choices else None
        if is_str_dict(choice) and is_str_dict(message := choice.get('message')):
            tool_calls = message.get('tool_calls')
            calls += [call for call in tool_calls if is_str_dict(call)] if is_list(tool_calls) else []
        elif (
            is_str_dict(choice)
            and is_str_dict(delta := choice.get('delta'))
            and is_list(chunks := delta.get('tool_calls'))
        ):
            for position, chunk in enumerate(chunks):
                if is_str_dict(chunk):
                    index = chunk.get('index')
                    slot = by_index.setdefault(index if isinstance(index, int) else position, {})
                    # A call's id comes with its first fragment.
                    for key in ('id', 'extra_content'):
                        if chunk.get(key):
                            slot.setdefault(key, chunk[key])
    return _Response(_chat_call_echoes([*calls, *(by_index[index] for index in sorted(by_index))]))


def _chat_call_echoes(calls: list[object]) -> list[Echo]:
    echoes: list[Echo] = []
    for call in calls:
        if not is_str_dict(call):
            continue
        if isinstance(call_id := call.get('id'), str) and call_id:
            echoes.append(('tool_calls.id', call_id))
        if extra_content := call.get('extra_content'):
            echoes.append(('tool_calls.extra_content', _canonical(extra_content)))
    return echoes


def _chat_request(history: list[object]) -> set[Echo]:
    echoes: set[Echo] = set()
    for message in history:
        if is_str_dict(message) and message.get('role') == 'assistant' and is_list(calls := message.get('tool_calls')):
            echoes.update(_chat_call_echoes(calls))
    return echoes


def _google_response(body: object) -> _Response:
    fields: list[Echo] = []
    for chunk in _events(body):
        candidates = chunk.get('candidates') if is_str_dict(chunk) else None
        candidate = candidates[0] if is_list(candidates) and candidates else None
        content = candidate.get('content') if is_str_dict(candidate) else None
        if is_str_dict(content) and is_list(parts := content.get('parts')):
            fields += _google_part_echoes(parts)
    return _Response(fields)


def _google_part_echoes(parts: list[object]) -> list[Echo]:
    echoes: list[Echo] = []
    for part in parts:
        if not is_str_dict(part):
            continue
        if isinstance(signature := part.get('thoughtSignature') or part.get('thought_signature'), str) and signature:
            echoes.append(('thoughtSignature', _base64(signature)))
        call = part.get('functionCall') or part.get('function_call')
        if is_str_dict(call) and isinstance(call_id := call.get('id'), str) and call_id:
            echoes.append(('functionCall.id', call_id))
    return echoes


def _google_request(history: list[object]) -> set[Echo]:
    echoes: set[Echo] = set()
    for content in history:
        if is_str_dict(content) and content.get('role') == 'model' and is_list(parts := content.get('parts')):
            echoes.update(_google_part_echoes(parts))
    return echoes


def _bedrock_response(body: object) -> _Response:
    if isinstance(body, bytes):
        return _fold_bedrock_stream(body)
    output = body.get('output') if is_str_dict(body) else None
    message = output.get('message') if is_str_dict(output) else None
    content = message.get('content') if is_str_dict(message) else None
    usage = body.get('usage') if is_str_dict(body) else None
    return _Response(
        _bedrock_block_echoes(content if is_list(content) else []),
        cache_read=_usage_tokens(usage, 'cacheReadInputTokens'),
        cache_write=_usage_tokens(usage, 'cacheWriteInputTokens'),
    )


def _fold_bedrock_stream(data: bytes) -> _Response:
    fields: list[Echo] = []
    signatures: dict[object, str] = defaultdict(str)
    redacted: dict[object, str] = defaultdict(str)
    usage: object = None
    for payload in _eventstream_payloads(data):
        start, delta = payload.get('start'), payload.get('delta')
        if (
            is_str_dict(start)
            and is_str_dict(tool_use := start.get('toolUse'))
            and isinstance(tool_use_id := tool_use.get('toolUseId'), str)
        ):
            fields.append(('toolUse.toolUseId', tool_use_id))
        elif is_str_dict(delta) and is_str_dict(reasoning := delta.get('reasoningContent')):
            signatures[payload.get('contentBlockIndex')] += str(reasoning.get('signature') or '')
            redacted[payload.get('contentBlockIndex')] += str(reasoning.get('redactedContent') or '')
        elif 'usage' in payload:
            usage = payload['usage']
    fields.extend(('reasoningContent.signature', signature) for signature in signatures.values() if signature)
    fields.extend(('reasoningContent.redactedContent', _base64(content)) for content in redacted.values() if content)
    return _Response(
        fields,
        cache_read=_usage_tokens(usage, 'cacheReadInputTokens'),
        cache_write=_usage_tokens(usage, 'cacheWriteInputTokens'),
    )


def _bedrock_block_echoes(blocks: list[object]) -> list[Echo]:
    echoes: list[Echo] = []
    for block in blocks:
        if not is_str_dict(block):
            continue
        reasoning, tool_use = block.get('reasoningContent'), block.get('toolUse')
        if is_str_dict(reasoning):
            text = reasoning.get('reasoningText')
            if is_str_dict(text) and isinstance(signature := text.get('signature'), str) and signature:
                echoes.append(('reasoningContent.signature', signature))
            if isinstance(content := reasoning.get('redactedContent'), str) and content:
                echoes.append(('reasoningContent.redactedContent', _base64(content)))
        if is_str_dict(tool_use) and isinstance(tool_use_id := tool_use.get('toolUseId'), str):
            echoes.append(('toolUse.toolUseId', tool_use_id))
    return echoes


def _bedrock_request(history: list[object]) -> set[Echo]:
    echoes: set[Echo] = set()
    for message in history:
        if is_str_dict(message) and message.get('role') == 'assistant' and is_list(content := message.get('content')):
            echoes.update(_bedrock_block_echoes(content))
    return echoes


def _cohere_response(body: object) -> _Response:
    if is_str_dict(body) and is_str_dict(message := body.get('message')):
        tool_calls = message.get('tool_calls')
        return _Response(_cohere_message_echoes(message.get('tool_plan'), tool_calls if is_list(tool_calls) else []))
    plan = ''
    calls: list[object] = []
    for event in _events(body):
        delta = event.get('delta') if is_str_dict(event) else None
        message = delta.get('message') if is_str_dict(delta) else None
        if not (is_str_dict(event) and is_str_dict(message)):
            continue
        if event.get('type') == 'tool-plan-delta':
            plan += str(message.get('tool_plan') or '')
        elif event.get('type') == 'tool-call-start':
            calls.append(message.get('tool_calls'))
    return _Response(_cohere_message_echoes(plan, calls))


def _cohere_message_echoes(plan: object, calls: list[object]) -> list[Echo]:
    echoes: list[Echo] = [('tool_plan', plan)] if isinstance(plan, str) and plan else []
    for call in calls:
        if is_str_dict(call) and isinstance(call_id := call.get('id'), str) and call_id:
            echoes.append(('tool_calls.id', call_id))
    return echoes


def _cohere_request(history: list[object]) -> set[Echo]:
    echoes: set[Echo] = set()
    for message in history:
        if is_str_dict(message) and message.get('role') == 'assistant':
            calls = message.get('tool_calls')
            echoes.update(_cohere_message_echoes(message.get('tool_plan'), calls if is_list(calls) else []))
    return echoes


_READERS: dict[WireShape, tuple[Callable[[object], _Response], Callable[[list[object]], set[Echo]]]] = {
    'anthropic': (_anthropic_response, _anthropic_request),
    'openai-responses': (_responses_response, _responses_request),
    'openai-chat': (_chat_response, _chat_request),
    'google': (_google_response, _google_request),
    'bedrock': (_bedrock_response, _bedrock_request),
    'cohere': (_cohere_response, _cohere_request),
}
"""How to read the fields `ECHO_FIELDS` lists for a shape out of a response, and out of a request's history."""


def _shared(a: list[object], b: list[object]) -> int:
    """How many leading messages two histories share."""
    return next((index for index, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _continuations(history: list[object], sent: list[_Request]) -> list[list[object]]:
    """What the sent requests continuing the conversation of the played request `history` add to it.

    The code's own rendering of the played request is the sent request sharing the most leading messages with
    `history`, the nearest in length and then the first sent among ties: it can differ from the recording by an id
    the code generates afresh on every run, or by a message the code now renders differently. The requests that continue it are the shortest longer sent requests
    that share the most leading messages with that rendering: all of it, unless the next request drops a
    request-scoped message. Each one contributes the messages after the ones it shares.
    """
    if not sent:
        return []
    rendered = min(
        (request.history for request in sent),
        key=lambda candidate: (-_shared(history, candidate), abs(len(candidate) - len(history))),
    )
    if not _shared(history, rendered):
        return []
    shared = {
        index: _shared(rendered, request.history)
        for index, request in enumerate(sent)
        if len(request.history) > len(rendered)
    }
    most = max(shared.values(), default=0)
    shortest = min((len(sent[index].history) for index, count in shared.items() if count == most), default=0)
    return [
        sent[index].history[most:]
        for index, count in shared.items()
        if most and count == most and len(sent[index].history) == shortest
    ]


def _is_sent_back(echo: Echo, echoed: set[Echo], field: EchoField) -> bool:
    """Whether `echoed` holds `echo`, or, for a function tool call id, the id a profile's `tool_call_id_length` made of it."""
    name, value = echo
    return echo in echoed or (
        field.function_tool_call_id
        and any(
            other == name
            and candidate.isascii()
            and candidate.isalnum()
            and candidate == hashed_tool_call_id(value, len(candidate))
            for other, candidate in echoed
        )
    )


def iter_echo_checks(interactions: Sequence[HttpInteraction], sent: Sequence[RecordedRequest]) -> Iterator[EchoCheck]:
    """Check each `ECHO_FIELDS` field of each played response against the request sent next in its conversation."""
    sent_by_endpoint: dict[tuple[WireShape, tuple[str, str]], list[_Request]] = defaultdict(list)
    for sent_request in sent:
        body = _json_or_none(sent_request.body) if isinstance(sent_request.body, str) else None
        if (request := _parse_request(sent_request.method, sent_request.uri, body)) is not None:
            sent_by_endpoint[(request.shape, request.endpoint)].append(request)
    for index, interaction in enumerate(interactions):
        request = _parse_request(
            interaction.request.method, interaction.request.uri, _body_content(interaction.request.body)
        )
        if (
            request is None
            or not request.history
            or interaction.response.status >= 400
            or not isinstance(echoes := ECHO_FIELDS[request.shape], Echoes)
        ):
            continue
        read_response, read_request = _READERS[request.shape]
        response = read_response(_body_content(interaction.response.body))
        candidates = sent_by_endpoint[(request.shape, request.endpoint)]
        if response.response_id is not None and any(
            candidate.body.get('previous_response_id') == response.response_id for candidate in candidates
        ):
            # The server holds the response, so the request needs none of its fields.
            yield from (EchoCheck(request.shape, name, index, value, True) for name, value in response.fields)
            continue
        if not response.fields or not (continuations := _continuations(request.history, candidates)):
            continue
        echoed: set[Echo] = set()
        for added in continuations:
            echoed |= read_request(added)
        for echo in response.fields:
            yield EchoCheck(request.shape, echo[0], index, echo[1], _is_sent_back(echo, echoed, echoes.fields[echo[0]]))


def _parameter(body: Mapping[str, object], path: str) -> object:
    value: object = body
    for key in path.split('.'):
        value = value.get(key) if is_str_dict(value) else None
    return value


def _has_cache_marker(value: object) -> bool:
    if is_str_dict(value):
        return 'cache_control' in value or 'cachePoint' in value or any(map(_has_cache_marker, value.values()))
    return is_list(value) and any(map(_has_cache_marker, value))


def _without_cache_markers(value: object) -> object:
    if is_str_dict(value):
        return {key: _without_cache_markers(item) for key, item in value.items() if key != 'cache_control'}
    if is_list(value):
        return [_without_cache_markers(item) for item in value if not (is_str_dict(item) and 'cachePoint' in item)]
    return value


def _cache_turn(index: int, request: _Request, interaction: HttpInteraction) -> _CacheTurn | None:
    stripped = _without_cache_markers(request.body)
    canonical = canonical_prefix_blocks(stripped, interaction.request.uri) if is_str_dict(stripped) else None
    if canonical is None:
        return None
    blocks = canonical[1]
    image = '"type": "image"' if request.shape == 'anthropic' else '{"image": '
    parameters = [_parameter(request.body, path) for path in CACHE_KEY_PARAMETERS[request.shape]]
    if request.shape == 'anthropic':
        model, thinking = str(request.body.get('model')), request.body.get('thinking')
    else:
        model = request.endpoint[1].removeprefix('/model/').split('/')[0]
        thinking = _parameter(request.body, 'additionalModelRequestFields.thinking')
    return _CacheTurn(
        index,
        blocks,
        _canonical([*parameters, any(image in block for _, block in blocks)]),
        model,
        is_str_dict(thinking) and thinking.get('type') != 'disabled',
        _READERS[request.shape][0](_body_content(interaction.response.body)),
    )


def _cache_minimum_tokens(model: str) -> int:
    """The fewest tokens `model` caches, from `CACHE_MINIMUM_TOKENS`."""
    fragments = [fragment for fragment in CACHE_MINIMUM_TOKENS if fragment in model]
    if not fragments:
        return max(CACHE_MINIMUM_TOKENS.values())
    return CACHE_MINIMUM_TOKENS[max(fragments, key=len)]


def iter_cache_misses(interactions: Sequence[HttpInteraction]) -> Iterator[ReplayFidelityViolation]:
    """Find each explicit cache write that the next recorded request extending its prefix doesn't read."""
    turns: dict[tuple[WireShape, tuple[str, str]], list[_CacheTurn]] = defaultdict(list)
    for index, interaction in enumerate(interactions):
        request = _parse_request(
            interaction.request.method, interaction.request.uri, _body_content(interaction.request.body)
        )
        if (
            request is not None
            and request.shape in CACHE_KEY_PARAMETERS
            and interaction.response.status < 400
            and _has_cache_marker(request.body)
            and (turn := _cache_turn(index, request, interaction)) is not None
        ):
            turns[(request.shape, request.endpoint)].append(turn)
    for (shape, _), endpoint_turns in turns.items():
        for position, turn in enumerate(endpoint_turns):
            later = next(
                (
                    candidate
                    for candidate in endpoint_turns[position + 1 :]
                    if candidate.key == turn.key
                    and classify_prefix_pair(turn.blocks, candidate.blocks)[0] == 'extension'
                ),
                None,
            )
            cached = turn.response.cache_read + turn.response.cache_write
            if (
                later is None
                or not turn.response.cache_write
                or later.response.cache_read
                or cached < _cache_minimum_tokens(turn.model)
                or (
                    turn.thinking
                    and any(
                        level == 'messages' and is_new_user_turn(block)
                        for level, block in later.blocks[len(turn.blocks) :]
                    )
                )
            ):
                continue
            yield ReplayFidelityViolation(
                'cache-not-read',
                shape,
                turn.interaction,
                f'cached {cached} tokens, and interaction {later.interaction}, which extends its prefix with the same '
                'cache-key parameters, read none',
            )


def replay_fidelity_violations(
    interactions: Sequence[HttpInteraction], sent: Sequence[RecordedRequest]
) -> list[ReplayFidelityViolation]:
    """The violations of a conversation whose played `interactions` led the code to send `sent`."""
    if len(interactions) < 2:
        return []
    violations: list[ReplayFidelityViolation] = []
    for check in iter_echo_checks(interactions, sent):
        if not check.echoed:
            echoes = ECHO_FIELDS[check.shape]
            assert isinstance(echoes, Echoes)
            violations.append(
                ReplayFidelityViolation(
                    'field-not-echoed',
                    check.shape,
                    check.interaction,
                    f'`{check.field}` {check.value[:60]!r} is not in the next request, where '
                    f'`{echoes.fields[check.field].sender}` puts it',
                )
            )
    violations += iter_cache_misses(interactions)
    return violations


def check_replay_fidelity(node: pytest.Item, cassette: Cassette) -> None:
    """Fail when a played conversation drops a provider field it must echo, or loses a cache read."""
    violations = replay_fidelity_violations(cassette.interactions, cassette.sent_requests)
    marker = node.get_closest_marker('drops_replay_field')
    if marker is not None:
        reason = marker.kwargs.get('reason')
        if not (isinstance(reason, str) and reason.strip()):
            pytest.fail(
                '@pytest.mark.drops_replay_field requires reason=... (a non-empty string) explaining why this '
                "test's conversation drops a provider field or a cache read"
            )
        if not violations:
            pytest.fail(
                '@pytest.mark.drops_replay_field is stale: every played response had its fields sent back and every '
                'cache write was read, so remove the marker'
            )
        return
    if violations:
        details = '\n'.join(
            f'{cassette.path} [{violation.shape}] interaction {violation.interaction}, {violation.code}: '
            f'{violation.detail}'
            for violation in violations
        )
        pytest.fail(
            f'{details}\nA field the next request drops makes the provider reject the conversation or lose what the '
            "field carried, and a cache write the next request doesn't read is paid for nothing. Fix the adapter. If "
            'the provider never needs the field back, fix its entry in `ECHO_FIELDS` in `tests/replay_fidelity.py`; if '
            "this test's conversation does this on purpose, add @pytest.mark.drops_replay_field(reason=...) to the test",
            pytrace=False,
        )
