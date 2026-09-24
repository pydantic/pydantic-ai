"""The boundary between Pydantic AI's message model and babel's canonical IR.

`messages_to_ir` renders a message history as a babel IR request, `ir_to_model_response` reads a
decoded IR response back into a `ModelResponse`, and `fold_stream_emits` routes babel's stream emits
into the parts manager. Everything below the IR, the actual provider wire mapping, is babel's
compiled transform; this module only crosses the boundary in both directions.
"""

from __future__ import annotations as _annotations

import dataclasses
import re
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias, TypeVar, cast

from llm_transform import ir_build
from llm_transform.ir_types import IRRequestDict, MessageDict, PartDict, TextPartDict
from llm_transform.registry import canonical_json

from ... import _utils
from ..._parts_manager import ModelResponsePartsManager
from ...exceptions import UserError
from ...messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    FilePart,
    FileUrl,
    FinishReason,
    ImageUrl,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from ...usage import RequestUsage
from .. import StreamedResponse, download_item

BabelFormat: TypeAlias = Literal['openai-chat', 'anthropic-messages', 'gemini', 'bedrock-converse']
"""The babel wire formats the drop-in models speak."""

IR: TypeAlias = dict[str, Any]
"""A decoded babel IR node: plain JSON-shaped data, see the `llm_transform.ir_types` TypedDicts for the shapes."""

_MediaKind: TypeAlias = Literal['image', 'audio', 'document', 'video']

_T = TypeVar('_T')

_MEDIA_KIND_PREFIXES: tuple[tuple[str, _MediaKind], ...] = (
    ('image/', 'image'),
    ('audio/', 'audio'),
    ('video/', 'video'),
)

# Anthropic and Bedrock report the uncached input tokens apart from the cache reads and writes;
# the other wires (and `RequestUsage.input_tokens`) report the inclusive total.
_DISJOINT_INPUT_TOKEN_FORMATS: frozenset[BabelFormat] = frozenset({'anthropic-messages', 'bedrock-converse'})

# A reasoning part's replay signature has no cross-provider meaning, so babel carries it in the
# source provider's `provider_ext` bucket. These tables map a `ThinkingPart.provider_name` to that
# bucket on the way in, and a response format to its bucket on the way out.
_SIGNATURE_NAMESPACE: dict[str, tuple[BabelFormat, str]] = {
    'anthropic': ('anthropic-messages', 'signature'),
    'google': ('gemini', 'thoughtSignature'),
    'google-gla': ('gemini', 'thoughtSignature'),
    'google-vertex': ('gemini', 'thoughtSignature'),
    'google-cloud': ('gemini', 'thoughtSignature'),
    'bedrock': ('bedrock-converse', 'signature'),
}
_SIGNATURE_FIELD: dict[BabelFormat, str] = {
    'anthropic-messages': 'signature',
    'gemini': 'thoughtSignature',
    'bedrock-converse': 'signature',
}

# The `provider_ext` field babel captures a response's id under, per format. Bedrock's request id
# is not in the response body, so its model passes it in.
_RESPONSE_ID_FIELD: dict[BabelFormat, str] = {
    'openai-chat': 'id',
    'anthropic-messages': 'id',
    'gemini': 'responseId',
}

# babel `StopReason` -> Pydantic AI `FinishReason`. Exhaustive over babel's `StopReason` literal,
# which `tests/models/babel/test_adapters.py` pins.
_FINISH_REASON: dict[str, FinishReason | None] = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
    'content_filter': 'content_filter',
    'refusal': 'stop',
    'malformed_tool_use': 'tool_call',
    'context_window_exceeded': 'length',
    'other': None,
}

# babel `StopReason` -> the raw OpenAI `finish_reason` the native model records in `provider_details`.
_OPENAI_RAW_FINISH_REASON: dict[str, str] = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'content_filter': 'content_filter',
    'refusal': 'stop',
}

_CAMEL_CASE_BOUNDARY = re.compile(r'([A-Z])')


def messages_to_ir(
    messages: Sequence[ModelMessage],
    *,
    model_name: str | None = None,
    provider_name: str | None = None,
    instruction_parts: Sequence[InstructionPart] | None = None,
) -> IRRequestDict:
    """Render a Pydantic AI message history as a babel IR request.

    System prompts and `instruction_parts` become IR system segments; user, tool-return and retry
    parts become user and tool messages; assistant parts become an assistant message. Media is
    carried as URLs or base64 as given, see `download_url_media` for the wires that need bytes.

    A `CachePoint` attaches an Anthropic `cache_control` breakpoint to the preceding content part.
    A `ThinkingPart` keeps its replay signature only under its own provider's namespace, so a turn
    produced by one provider never replays a signature to another, and a redacted one (an encrypted
    blob only its provider can read) replays only to `provider_name`. Server-side tool parts from
    `provider_name` replay as provider-executed tool calls; those from other providers are dropped,
    as the native models do.

    Raises:
        UserError: For a `CachePoint` with no preceding content, or a part kind babel cannot render.
    """
    system: list[TextPartDict] = []
    ir_messages: list[MessageDict] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            ir_messages.extend(_request_to_ir(message, system))
        else:
            content = [
                part_ir for part in message.parts if (part_ir := _response_part_to_ir(part, provider_name)) is not None
            ]
            if content:
                ir_messages.append(ir_build.assistant(content))
    system.extend(ir_build.text(part.content) for part in instruction_parts or ())
    return ir_build.request(model=model_name, messages=ir_messages, system=system or None)


def _request_to_ir(message: ModelRequest, system: list[TextPartDict]) -> list[MessageDict]:
    user_parts: list[PartDict] = []
    tool_parts: list[PartDict] = []
    # Media a tool returned trails the tool result as a user message, the split native models make
    # for wires whose tool results only take text.
    tool_media_parts: list[PartDict] = []
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            system.append(ir_build.text(part.content))
        elif isinstance(part, UserPromptPart):
            if isinstance(part.content, str):
                user_parts.append(ir_build.text(part.content))
            else:
                for item in part.content:
                    if isinstance(item, CachePoint):
                        _apply_cache_point(user_parts, item)
                    else:
                        user_parts.append(_user_content_to_ir(item))
        elif isinstance(part, ToolReturnPart):
            text, files = part.model_response_str_and_user_content()
            if files:
                tool_parts.append(ir_build.tool_result(text, id=part.tool_call_id, name=part.tool_name))
                tool_media_parts.extend(_user_content_to_ir(file) for file in files if not isinstance(file, CachePoint))
            else:
                tool_parts.append(ir_build.tool_result(part.content, id=part.tool_call_id, name=part.tool_name))
        elif isinstance(part, RetryPromptPart):
            if part.tool_name is None:
                user_parts.append(ir_build.text(part.model_response()))
            else:
                tool_parts.append(
                    ir_build.tool_result(
                        part.model_response(), id=part.tool_call_id, name=part.tool_name, is_error=True
                    )
                )
        else:
            raise UserError(f'`{type(part).__name__}` is not supported by babel models')
    ir_messages: list[MessageDict] = []
    if tool_parts:
        ir_messages.append(ir_build.tool(tool_parts))
    if user_parts:
        ir_messages.append(ir_build.user(user_parts))
    if tool_media_parts:
        ir_messages.append(ir_build.user(tool_media_parts))
    return ir_messages


def _user_content_to_ir(item: str | TextContent | MultiModalContent) -> PartDict:
    if isinstance(item, str):
        return ir_build.text(item)
    if isinstance(item, TextContent):
        return ir_build.text(item.content)
    if isinstance(item, UploadedFile):
        return ir_build.file(item.file_id, source='file_id', media_kind=_media_kind(item.media_type))
    if isinstance(item, BinaryContent):
        return ir_build.file(
            item.base64,
            media_kind=_media_kind(item.media_type),
            media_type=item.media_type,
            provider_ext=_image_detail_provider_ext(item.vendor_metadata),
        )
    return ir_build.file(
        item.url,
        source='url',
        media_kind=_url_media_kind(item),
        provider_ext=_image_detail_provider_ext(item.vendor_metadata),
    )


def _media_kind(media_type: str) -> _MediaKind:
    for prefix, kind in _MEDIA_KIND_PREFIXES:
        if media_type.startswith(prefix):
            return kind
    return 'document'


def _url_media_kind(item: FileUrl) -> _MediaKind:
    if isinstance(item, ImageUrl):
        return 'image'
    if isinstance(item, AudioUrl):
        return 'audio'
    if isinstance(item, VideoUrl):
        return 'video'
    return 'document'


def _image_detail_provider_ext(vendor_metadata: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Carry an OpenAI image `detail` from `vendor_metadata` so the `openai-chat` encoder emits it."""
    if vendor_metadata and 'detail' in vendor_metadata:
        return {'openai-chat': {'detail': vendor_metadata['detail']}}
    return None


def _apply_cache_point(parts: list[PartDict], cache_point: CachePoint) -> None:
    if not parts:
        raise UserError(
            'CachePoint cannot be the first content in a user message - there must be previous content '
            'to attach the CachePoint to. To cache system instructions or tool definitions, use the '
            '`anthropic_cache_instructions` or `anthropic_cache_tool_definitions` settings instead.'
        )
    part = parts[-1]
    provider_ext = part.get('provider_ext') or {}
    anthropic_ext = provider_ext.get('anthropic-messages') or {}
    cache_control = {'type': 'ephemeral', 'ttl': cache_point.ttl}
    # Rebuilt rather than assigned in place: `PartDict` is a union of TypedDicts, which cannot be
    # written through, and every member carries `provider_ext`, so the spread keeps the part's shape.
    parts[-1] = cast(
        PartDict,
        {
            **part,
            'provider_ext': {**provider_ext, 'anthropic-messages': {**anthropic_ext, 'cache_control': cache_control}},
        },
    )


def _response_part_to_ir(part: ModelResponsePart, provider_name: str | None) -> PartDict | None:
    if isinstance(part, TextPart):
        return ir_build.text(part.content)
    if isinstance(part, ThinkingPart):
        if part.id == 'redacted_thinking':
            # A redacted block has no readable content; the encrypted blob is its `signature`, which
            # only the provider that issued it can decrypt, so it never replays anywhere else.
            if part.provider_name == provider_name and part.signature:
                return ir_build.reasoning(part.signature, redacted=True)
            return None
        return ir_build.reasoning(part.content, provider_ext=_signature_provider_ext(part))
    if isinstance(part, NativeToolCallPart):
        if part.provider_name != provider_name:
            return None
        return ir_build.tool_call(part.tool_name, part.args_as_dict(), id=part.tool_call_id, provider_executed=True)
    if isinstance(part, NativeToolReturnPart):
        if part.provider_name != provider_name:
            return None
        return ir_build.tool_result(part.content, id=part.tool_call_id, name=part.tool_name, provider_executed=True)
    if isinstance(part, ToolCallPart):
        return ir_build.tool_call(part.tool_name, part.args_as_dict(), id=part.tool_call_id)
    if isinstance(part, CompactionPart):
        # Only Anthropic renders a compaction block; elsewhere the summary survives as plain text.
        marker = {'anthropic-messages': {'block': 'compaction'}} if part.provider_name == 'anthropic' else None
        return ir_build.text(part.content or '', provider_ext=marker)
    if isinstance(part, FilePart):
        return None
    raise UserError(f'`{type(part).__name__}` is not supported by babel models')


def _signature_provider_ext(part: ThinkingPart) -> dict[str, dict[str, Any]] | None:
    """The namespaced `provider_ext` home for a thinking part's replay signature.

    `None` when there is no signature or the provider has no known namespace: a signature is never
    filed under a guessed provider, since that is exactly the cross-provider replay a namespace prevents.
    """
    namespace = _SIGNATURE_NAMESPACE.get(part.provider_name or '')
    if part.signature and namespace:
        fmt, field = namespace
        return {fmt: {field: part.signature}}
    return None


def ir_to_model_response(
    ir: IR,
    *,
    fmt: BabelFormat,
    provider_name: str,
    provider_url: str,
    usage: RequestUsage | None = None,
    model_name: str | None = None,
    provider_response_id: str | None = None,
) -> ModelResponse:
    """Read a decoded babel IR response into a `ModelResponse`.

    Text, reasoning and tool-call parts of the first candidate are mapped; a reasoning part's replay
    signature is read from `fmt`'s `provider_ext` bucket and the resulting `ThinkingPart` is tagged
    with `provider_name`, so it replays through `messages_to_ir` under the same namespace.
    Provider-executed tool calls and results become native tool parts. Grounding sources are not
    carried.

    Args:
        ir: The IR response from `llm_transform.registry.decode_response`.
        fmt: The babel format the response was decoded from.
        provider_name: The provider the response came from, recorded on the response and its parts.
        provider_url: The provider's base URL.
        usage: The request usage. Defaults to the IR's token counts, with Anthropic's and Bedrock's
            cache reads and writes folded into `input_tokens` to match `RequestUsage`'s inclusive counts.
        model_name: The model name to record when the response body carries none, as Bedrock's does not.
        provider_response_id: The response id when it is not in the body, as Bedrock's is not.
    """
    candidate: IR = ir['candidates'][0]
    content: list[IR] = candidate['content']
    parts: list[ModelResponsePart] = []
    for part in content:
        kind = part['kind']
        if kind == 'text':
            if part['text']:
                parts.append(_text_ir_to_part(part, provider_name))
        elif kind == 'reasoning':
            if part.get('redacted'):
                parts.append(
                    ThinkingPart(
                        id='redacted_thinking', content='', signature=part['text'], provider_name=provider_name
                    )
                )
            else:
                provider_ext: dict[str, Any] = part.get('provider_ext') or {}
                bucket: dict[str, Any] = provider_ext.get(fmt) or {}
                signature = bucket.get(_SIGNATURE_FIELD.get(fmt, ''))
                parts.append(ThinkingPart(content=part['text'], signature=signature, provider_name=provider_name))
        elif kind == 'tool_call':
            if part.get('provider_executed'):
                parts.append(
                    NativeToolCallPart(
                        tool_name=part['name'],
                        args=_tool_args(part['input']),
                        tool_call_id=part.get('id') or '',
                        provider_name=provider_name,
                    )
                )
            else:
                parts.append(
                    ToolCallPart(
                        tool_name=part['name'], args=_tool_args(part['input']), tool_call_id=part.get('id') or ''
                    )
                )
        elif kind == 'tool_result' and part.get('provider_executed'):
            parts.append(
                NativeToolReturnPart(
                    tool_name=part.get('name') or '',
                    content=part['content'],
                    tool_call_id=part.get('id') or '',
                    provider_name=provider_name,
                )
            )
    response_provider_ext: dict[str, Any] = ir.get('provider_ext') or {}
    response_ext: dict[str, Any] = response_provider_ext.get(fmt) or {}
    return ModelResponse(
        parts=parts,
        usage=usage if usage is not None else _ir_usage(ir, fmt),
        model_name=ir.get('model') or model_name,
        provider_name=provider_name,
        provider_url=provider_url,
        provider_response_id=provider_response_id or response_ext.get(_RESPONSE_ID_FIELD.get(fmt, '')),
        provider_details=_openai_provider_details(candidate, response_ext) if fmt == 'openai-chat' else None,
        finish_reason=_FINISH_REASON.get(candidate['stop_reason']),
    )


def _text_ir_to_part(part: IR, provider_name: str) -> TextPart | CompactionPart:
    provider_ext: dict[str, Any] = part.get('provider_ext') or {}
    anthropic_ext: dict[str, Any] = provider_ext.get('anthropic-messages') or {}
    if anthropic_ext.get('block') == 'compaction':
        return CompactionPart(content=part['text'], provider_name=provider_name)
    return TextPart(content=part['text'])


def _tool_args(value: Any) -> Any:
    """Serialize decoded tool-call arguments to the string form a `ToolCallPart` carries.

    Decoded IR keeps JSON numbers as byte-faithful tokens the standard library cannot dump, so babel's
    `canonical_json` serializes them, matching how the native models forward the wire's argument string.
    """
    return canonical_json(value) if isinstance(value, dict | list) else value


def _ir_usage(ir: IR, fmt: BabelFormat) -> RequestUsage:
    usage: dict[str, Any] = ir.get('usage') or {}
    cache_read: int = usage.get('cache_read_tokens') or 0
    cache_write: int = usage.get('cache_write_tokens') or 0
    input_tokens: int = usage.get('input_tokens') or 0
    if fmt in _DISJOINT_INPUT_TOKEN_FORMATS:
        input_tokens += cache_read + cache_write
    return RequestUsage(
        input_tokens=input_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=usage.get('output_tokens') or 0,
    )


def _openai_provider_details(candidate: IR, response_ext: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the `provider_details` the native OpenAI model records: the raw `finish_reason` and the `created` timestamp."""
    provider_details: dict[str, Any] = {}
    raw_finish_reason = _OPENAI_RAW_FINISH_REASON.get(candidate['stop_reason'])
    if raw_finish_reason is not None:
        provider_details['finish_reason'] = raw_finish_reason
    created = response_ext.get('created')
    if created is not None:
        provider_details['timestamp'] = datetime.fromtimestamp(int(created), tz=timezone.utc)
    return provider_details or None


def fold_stream_emits(
    emits: Sequence[IR],
    parts_manager: ModelResponsePartsManager,
    response: StreamedResponse,
    *,
    provider_name: str | None = None,
    on_usage: Callable[[IR], None] | None = None,
) -> Iterator[ModelResponseStreamEvent]:
    """Route the emits of one babel `stream_step` into the parts manager, yielding the resulting events.

    Text and reasoning deltas stream through the parts manager under fixed vendor part ids. A
    `tool_call_start` opens the tool-call part with its name; the later `tool_call` fills in the
    complete arguments, so a streamed tool call arrives as a start event and one argument delta.
    A `meta` emit sets the response's finish reason; a `usage` emit is handed to `on_usage`, since
    accumulating usage across chunks is the caller's job.
    """
    for emit in emits:
        kind = emit['kind']
        if kind == 'text':
            yield from parts_manager.handle_text_delta(vendor_part_id='content', content=emit['text'])
        elif kind == 'reasoning':
            yield from parts_manager.handle_thinking_delta(
                vendor_part_id='thinking',
                content=emit.get('text'),
                signature=emit.get('signature'),
                provider_name=provider_name,
            )
        elif kind in ('tool_call_start', 'tool_call'):
            # The start carries the name and the completion the arguments; resending the name with
            # the arguments would append it to the name the parts manager accumulated.
            event = parts_manager.handle_tool_call_delta(
                vendor_part_id=emit.get('id') or emit['name'],
                tool_name=emit['name'] if kind == 'tool_call_start' else None,
                args=_tool_args(emit['input']) if kind == 'tool_call' else None,
                tool_call_id=emit.get('id'),
            )
            if event is not None:
                yield event
        elif kind == 'usage':
            if on_usage is not None:
                on_usage(emit)
        elif kind == 'meta':
            response.finish_reason = _FINISH_REASON.get(emit['stop_reason'])


async def download_url_media(messages: Sequence[ModelMessage], url_ok: frozenset[str]) -> list[ModelMessage]:
    """Replace URL media a wire cannot take as a URL with downloaded `BinaryContent`.

    `url_ok` is the set of media kinds the target wire accepts by URL (babel's `MEDIA_URL_OK` table);
    every other `FileUrl` in a user prompt or a tool return, and any with `force_download` set, is
    downloaded with `download_item`. Returns a new history; messages without such media are shared,
    not copied.
    """
    result: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [await _download_request_media(part, url_ok) for part in message.parts]
            if any(new is not old for new, old in zip(parts, message.parts)):
                message = dataclasses.replace(message, parts=parts)
        result.append(message)
    return result


async def _download_request_media(part: ModelRequestPart, url_ok: frozenset[str]) -> ModelRequestPart:
    if isinstance(part, UserPromptPart):
        if isinstance(part.content, str):
            return part
        items: list[UserContent] = [await _download_file_url(item, url_ok) for item in part.content]
        return _replace_content(part, items)
    if isinstance(part, ToolReturnPart):
        # A tool's files trail its result as user content (see `_request_to_ir`), so they need the
        # same treatment as a user prompt's. A `ToolReturnPart` holds a file directly or in a list.
        tool_content: Any = part.content
        if isinstance(tool_content, list):
            content: list[Any] = [await _download_file_url(item, url_ok) for item in cast(list[Any], tool_content)]
            return _replace_content(part, content)
        if isinstance(tool_content, FileUrl):
            return _replace_content(part, [await _download_file_url(tool_content, url_ok)], single=True)
    return part


_RequestPartT = TypeVar('_RequestPartT', UserPromptPart, ToolReturnPart)


def _replace_content(part: _RequestPartT, items: list[Any], *, single: bool = False) -> _RequestPartT:
    """`part` with `items` as its content, or `part` itself when nothing in it was downloaded."""
    old = [part.content] if single else cast(list[Any], part.content)
    if all(new is old_item for new, old_item in zip(items, old)):
        return part
    return dataclasses.replace(part, content=items[0] if single else items)


async def _download_file_url(item: _T, url_ok: frozenset[str]) -> _T | BinaryContent:
    if isinstance(item, FileUrl) and (item.force_download or _url_media_kind(item) not in url_ok):
        downloaded = await download_item(item, data_format='bytes')
        return BinaryContent(
            data=downloaded['data'], media_type=downloaded['data_type'], vendor_metadata=item.vendor_metadata
        )
    return item


def gemini_rest_to_sdk(node: Any) -> Any:
    """Convert a babel Gemini REST-wire node to the `google.genai` SDK's snake_case dict shape.

    Only structural keys are converted; tool-call `args`, `response` payloads and JSON schemas are
    user data and stay untouched. A scalar `functionResponse.response` is wrapped in
    `{'return_value': ...}`, the object the SDK requires, as the native model does.
    """
    return _wrap_scalar_function_responses(_rekey(node, _camel_to_snake))


_GEMINI_OPAQUE_KEYS = frozenset({'args', 'response', 'responseSchema', 'response_schema'})


def _rekey(value: Any, key_fn: Callable[[str], str]) -> Any:
    if _utils.is_str_dict(value):
        return {
            key_fn(key): item if key in _GEMINI_OPAQUE_KEYS else _rekey(item, key_fn) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rekey(item, key_fn) for item in cast(list[Any], value)]
    return value


def _wrap_scalar_function_responses(value: Any) -> Any:
    if _utils.is_str_dict(value):
        function_response = value.get('function_response')
        if _utils.is_str_dict(function_response) and not isinstance(function_response.get('response'), dict):
            function_response = {**function_response, 'response': {'return_value': function_response.get('response')}}
            value = {**value, 'function_response': function_response}
        return {key: _wrap_scalar_function_responses(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wrap_scalar_function_responses(item) for item in cast(list[Any], value)]
    return value


def _camel_to_snake(key: str) -> str:
    return _CAMEL_CASE_BOUNDARY.sub(r'_\1', key).lower()
