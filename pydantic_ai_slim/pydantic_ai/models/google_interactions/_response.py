"""Response steps shared by ordinary and streamed Google interactions."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Literal

from google.genai import interactions as gi
from pydantic import JsonValue, TypeAdapter

from ... import _utils
from ...exceptions import UnexpectedModelBehavior
from ...messages import (
    AudioUrl,
    BinaryContent,
    Citation,
    ContentCitationAnchor,
    DocumentCitationSource,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    VideoUrl,
    WebCitationSource,
)
from ...usage import RequestUsage
from .. import _google_common, download_item

STEP_PARAMS = TypeAdapter(list[dict[str, JsonValue]])
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_CONTENT_OBJECTS = TypeAdapter(list[dict[str, JsonValue]])

NATIVE_TOOL_NAMES: dict[str, str] = {
    'google_search': 'web_search',
    'url_context': 'web_fetch',
    'code_execution': 'code_execution',
    'file_search': 'file_search',
    'mcp_server_tool': 'mcp_server',
    'google_maps': 'google_maps',
    'retrieval': 'retrieval',
    'processing': 'processing',
}


def media_kind(media_type: str) -> Literal['image', 'audio', 'video', 'document']:
    kind = media_type.partition('/')[0]
    if kind in ('image', 'audio', 'video'):
        return kind
    return 'document'


def replay_steps(response: ModelResponse) -> list[dict[str, JsonValue]] | None:
    """Validate persisted wire steps at the one boundary where they reenter a request."""
    if response.provider_details is None or 'google_interactions_steps' not in response.provider_details:
        return None
    return STEP_PARAMS.validate_python(response.provider_details['google_interactions_steps'])


def mark_step(parts: list[ModelResponsePart], index: int) -> list[ModelResponsePart]:
    for content_index, part in enumerate(parts):
        part.provider_details = {
            **(part.provider_details or {}),
            'google_interactions_step_index': index,
            'google_interactions_content_index': content_index,
        }
        if isinstance(part, FilePart):
            part.provider_details['google_interactions_content_sha256'] = hashlib.sha256(part.content.data).hexdigest()
            part.provider_details['google_interactions_content_media_type'] = part.content.media_type
        elif isinstance(part, TextPart):
            part.provider_details['google_interactions_text_sha256'] = hashlib.sha256(part.content.encode()).hexdigest()
    return parts


def step_metadata(raw: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Retain replay-only fields; normalized parts own the visible payloads."""
    metadata: dict[str, JsonValue] = dict(raw)
    step_type = str(raw.get('type', ''))
    if step_type == 'function_call' or step_type.removesuffix('_call') in NATIVE_TOOL_NAMES:
        metadata.pop('arguments', None)
    elif step_type == 'function_result' or step_type.removesuffix('_result') in NATIVE_TOOL_NAMES:
        metadata.pop('result', None)
    if raw.get('type') == 'model_output':
        content = _CONTENT_OBJECTS.validate_python(raw.get('content', []))
        for value in content:
            value.pop('text' if value.get('type') == 'text' else 'data', None)
        metadata['content'] = [value for value in content]
    return metadata


def mcp_tool_name(server_name: str | None) -> str:
    return f'mcp_server:{server_name}' if server_name else 'mcp_server'


def mcp_server_names(steps: Iterable[gi.Step]) -> dict[str, str]:
    all_steps: list[gi.Step] = list(steps)
    executed = {step.call_id for step in all_steps if isinstance(step, gi.FunctionResultStep)}
    names: dict[str, str] = {}
    for step in all_steps:
        if isinstance(step, gi.MCPServerToolCallStep):
            names[step.id] = step.server_name
        elif isinstance(step, gi.FunctionCallStep) and step.id in executed:
            names[step.id] = step.name.partition(':')[0]
    return names


def usage_from_interaction(usage: gi.Usage | None) -> RequestUsage:
    if usage is None:
        return RequestUsage(unmeasured_requests=1)
    details: dict[str, int] = {}
    if usage.total_thought_tokens is not None:
        details['thoughts_tokens'] = usage.total_thought_tokens
    if usage.total_tool_use_tokens is not None:
        details['tool_use_tokens'] = usage.total_tool_use_tokens
    for tool_count in usage.grounding_tool_count or []:
        if tool_count.type is not None and tool_count.count is not None:
            details[f'{tool_count.type}_count'] = tool_count.count
    for prefix, modalities in (
        ('input', usage.input_tokens_by_modality),
        ('output', usage.output_tokens_by_modality),
        ('cached', usage.cached_tokens_by_modality),
        ('tool_use', usage.tool_use_tokens_by_modality),
    ):
        for modality in modalities or []:
            details[f'{prefix}_{modality.modality}_tokens'] = modality.tokens or 0
    return RequestUsage(
        input_tokens=usage.total_input_tokens or 0,
        output_tokens=(usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0),
        cache_read_tokens=usage.total_cached_tokens or 0,
        output_reasoning_tokens=usage.total_thought_tokens or 0,
        input_audio_tokens=sum(m.tokens or 0 for m in usage.input_tokens_by_modality or [] if m.modality == 'audio'),
        output_audio_tokens=sum(m.tokens or 0 for m in usage.output_tokens_by_modality or [] if m.modality == 'audio'),
        cache_audio_read_tokens=sum(
            m.tokens or 0 for m in usage.cached_tokens_by_modality or [] if m.modality == 'audio'
        ),
        details=details,
    )


async def process_response(
    response: gi.Interaction,
    model_name: str,
    provider_name: str,
    provider_url: str,
    canonical_system: str | None,
) -> ModelResponse:
    parts: list[ModelResponsePart] = []
    server_calls = mcp_server_names(response.steps or [])
    for index, step in enumerate(response.steps or []):
        parts.extend(mark_step(await map_step(step, provider_name, server_calls=server_calls), index))
    return build_response(response, parts, model_name, provider_name, provider_url, canonical_system)


def build_response(
    response: gi.Interaction,
    parts: list[ModelResponsePart],
    model_name: str,
    provider_name: str,
    provider_url: str,
    canonical_system: str | None,
) -> ModelResponse:
    details: dict[str, JsonValue] = {
        'status': response.status,
        'google_interactions_steps': [
            step_metadata(JSON_OBJECT.validate_python(step.model_dump(mode='json', by_alias=True, exclude_none=True)))
            for step in response.steps or []
        ],
    }
    if response.usage is not None:
        details['usage'] = JSON_OBJECT.validate_python(response.usage.model_dump(mode='json', exclude_none=True))
    if response.service_tier is not None:
        details['service_tier'] = response.service_tier
    if response.errors:
        details['errors'] = [
            JSON_OBJECT.validate_python(error.model_dump(mode='json', exclude_none=True)) for error in response.errors
        ]
    finish_reasons: dict[str, FinishReason] = {
        'completed': 'stop',
        'requires_action': 'tool_call',
        'incomplete': 'length',
        'budget_exceeded': 'length',
        'failed': 'error',
        'cancelled': 'error',
    }
    return ModelResponse(
        parts=parts,
        model_name=response.model or model_name,
        provider_name=provider_name,
        provider_url=provider_url,
        canonical_system=canonical_system,
        provider_response_id=response.id or None,
        timestamp=datetime.fromisoformat(response.created.replace('Z', '+00:00'))
        if response.created
        else _utils.now_utc(),
        usage=usage_from_interaction(response.usage),
        provider_details=details,
        finish_reason=finish_reasons.get(response.status),
    )


NativeCallStep = (
    gi.GoogleSearchCallStep
    | gi.URLContextCallStep
    | gi.CodeExecutionCallStep
    | gi.FileSearchCallStep
    | gi.MCPServerToolCallStep
    | gi.GoogleMapsCallStep
    | gi.RetrievalCallStep
    | gi.ProcessingCallStep
)
NativeResultStep = (
    gi.GoogleSearchResultStep
    | gi.URLContextResultStep
    | gi.CodeExecutionResultStep
    | gi.FileSearchResultStep
    | gi.MCPServerToolResultStep
    | gi.GoogleMapsResultStep
    | gi.RetrievalResultStep
    | gi.ProcessingResultStep
)


async def map_step(
    step: gi.Step, provider_name: str, *, server_calls: Mapping[str, str] | None = None
) -> list[ModelResponsePart]:
    """Map one wire step, including signatures that carry no visible text."""
    server_calls = server_calls or {}
    match step:
        case gi.ThoughtStep():
            return map_thought(step, provider_name)
        case gi.FunctionCallStep():
            return map_function_call(step, provider_name, server_calls)
        case gi.FunctionResultStep():
            return map_function_result(step, provider_name, server_calls)
        case gi.ModelOutputStep():
            return [await map_content(content, provider_name) for content in step.content or []]
        case _ if isinstance(step, NativeCallStep):
            return map_native_call(step, provider_name)
        case _ if isinstance(step, NativeResultStep):
            return map_native_result(step, provider_name, server_calls)
        case _:
            raise UnexpectedModelBehavior(f'Unexpected Google Interactions response step: {step.type!r}')


def map_thought(step: gi.ThoughtStep, provider_name: str) -> list[ModelResponsePart]:
    text = ''.join(item.text for item in step.summary or [] if isinstance(item, gi.TextContent))
    return [ThinkingPart(text, signature=step.signature, provider_name=provider_name)]


def map_function_result(
    step: gi.FunctionResultStep, provider_name: str, server_calls: Mapping[str, str]
) -> list[ModelResponsePart]:
    raw = JSON_OBJECT.validate_python(step.model_dump(mode='json', by_alias=True, exclude_none=True))
    return [
        NativeToolReturnPart(
            mcp_tool_name(server_calls.get(step.call_id)),
            raw.get('result'),
            step.call_id,
            provider_name=provider_name,
            provider_details=step_metadata(raw),
        )
    ]


def map_native_call(step: NativeCallStep, provider_name: str) -> list[ModelResponsePart]:
    raw = JSON_OBJECT.validate_python(step.model_dump(mode='json', by_alias=True, exclude_none=True))
    return [
        NativeToolCallPart(
            mcp_tool_name(step.server_name)
            if isinstance(step, gi.MCPServerToolCallStep)
            else NATIVE_TOOL_NAMES[step.type.removesuffix('_call')],
            JSON_OBJECT.validate_python(raw.get('arguments', {})),
            step.id,
            provider_name=provider_name,
            provider_details=step_metadata(raw),
        )
    ]


def map_native_result(
    step: NativeResultStep, provider_name: str, server_calls: Mapping[str, str]
) -> list[ModelResponsePart]:
    raw = JSON_OBJECT.validate_python(step.model_dump(mode='json', by_alias=True, exclude_none=True))
    return [
        NativeToolReturnPart(
            mcp_tool_name(step.server_name or server_calls.get(step.call_id))
            if isinstance(step, gi.MCPServerToolResultStep)
            else NATIVE_TOOL_NAMES[step.type.removesuffix('_result')],
            raw.get('result'),
            step.call_id,
            provider_name=provider_name,
            provider_details=step_metadata(raw),
        )
    ]


def map_function_call(
    step: gi.FunctionCallStep, provider_name: str, server_calls: Mapping[str, str]
) -> list[ModelResponsePart]:
    if step.id in server_calls:
        raw = JSON_OBJECT.validate_python(step.model_dump(mode='json', by_alias=True, exclude_none=True))
        return [
            NativeToolCallPart(
                mcp_tool_name(server_calls.get(step.id)),
                step.arguments,
                step.id,
                provider_name=provider_name,
                provider_details=step_metadata(raw),
            )
        ]
    return [ToolCallPart(step.name, step.arguments, step.id, provider_name=provider_name)]


async def map_content(content: gi.Content, provider_name: str) -> ModelResponsePart:
    if isinstance(content, gi.TextContent):
        return map_text_content(content, provider_name)
    elif isinstance(content, (gi.ImageContent, gi.AudioContent, gi.VideoContent, gi.DocumentContent)):
        return await map_media_content(content, provider_name)
    else:
        raise UnexpectedModelBehavior(f'Unexpected Google Interactions response content: {content.type!r}')


def map_text_content(content: gi.TextContent, provider_name: str) -> TextPart:
    citations: list[Citation] = []
    for annotation in content.annotations or []:
        if citation := map_annotation(content.text, annotation):
            citations.append(citation)
    return TextPart(content.text, provider_name=provider_name, citations=citations or None)


def map_annotation(text: str, annotation: gi.Annotation) -> Citation | None:
    if isinstance(annotation, (gi.URLCitation, gi.PlaceCitation)):
        if annotation.url is None:
            return None
        title = annotation.title if isinstance(annotation, gi.URLCitation) else annotation.name
        source = WebCitationSource(annotation.url, title=title)
    elif isinstance(annotation, gi.FileCitation):
        source = DocumentCitationSource(
            document_id=annotation.document_uri or annotation.media_id, title=annotation.file_name
        )
    else:
        return None
    anchor = None
    if annotation.start_index is not None and annotation.end_index is not None:
        start = _google_common.character_index(text, annotation.start_index)
        end = _google_common.character_index(text, annotation.end_index)
        if start is not None and end is not None and start < end:
            anchor = ContentCitationAnchor(start=start, end=end)
    return Citation([source], anchor=anchor)


async def map_media_content(
    content: gi.ImageContent | gi.AudioContent | gi.VideoContent | gi.DocumentContent, provider_name: str
) -> FilePart:
    if content.data is not None:
        data = base64.b64decode(content.data)
        mime_type = content.mime_type or 'application/octet-stream'
    elif content.uri is not None:
        url_class = (
            ImageUrl
            if isinstance(content, gi.ImageContent)
            else AudioUrl
            if isinstance(content, gi.AudioContent)
            else VideoUrl
            if isinstance(content, gi.VideoContent)
            else DocumentUrl
        )
        downloaded = await download_item(url_class(content.uri), data_format='bytes')
        data, mime_type = downloaded['data'], downloaded['data_type']
    else:
        raise UnexpectedModelBehavior('Google returned media without data or a URI.')
    return FilePart(BinaryContent.narrow_type(BinaryContent(data, media_type=mime_type)), provider_name=provider_name)
