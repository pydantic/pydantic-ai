from __future__ import annotations as _annotations

import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from urllib.parse import urlparse

from typing_extensions import TypeVar, assert_never

from .messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    FileUrl,
    ForceDownloadMode,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    ToolReturnContent,
    UploadedFile,
    UserContent,
    UserPromptPart,
)

__all__ = ('sanitize_message_history',)


_FileUrlT = TypeVar('_FileUrlT', bound=FileUrl)
"""TypeVar for a [`FileUrl`][pydantic_ai.messages.FileUrl] subclass, used to preserve the concrete
subclass (`ImageUrl`, `DocumentUrl`, etc.) when sanitizing a file URL."""


def sanitize_message_history(
    messages: Sequence[ModelMessage],
    *,
    strip_system_prompts: bool = True,
    allowed_file_url_schemes: Collection[str] = ('http', 'https'),
    allowed_file_url_force_download: Collection[ForceDownloadMode] = (),
    preserve_file_data: bool = False,
    resolved_tool_call_ids: Collection[str] = (),
) -> list[ModelMessage]:
    """Strip message history parts that aren't safe to honor from untrusted input.

    This is the same default sanitization the [UI adapters](../ui/overview.md) apply to
    client-submitted messages before they're passed to an agent. Use it when loading
    `message_history` from a source the application does not fully trust, such as a browser request.

    By default it strips:

    - [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s (disable with
      `strip_system_prompts=False`). The system prompt is the server's to own; a client that can
      inject one can override the agent's behavior. If stripping leaves a `ModelRequest` with no
      parts, the request is dropped from history entirely.
    - [`FileUrl`][pydantic_ai.messages.FileUrl] parts whose URL scheme is not in
      `allowed_file_url_schemes` (default `http`/`https`). Non-HTTP schemes like `s3://` or `gs://`
      cause the model provider to fetch the object using the server-side IAM role, so they should
      only be accepted from trusted clients.
    - [`FileUrl.force_download`][pydantic_ai.messages.FileUrl.force_download] values other than
      `False` that aren't in `allowed_file_url_force_download`, resetting them to `False`. Both
      `True` and `'allow-local'` are reset by default. Applies to file URLs in user content and
      those nested in tool return parts.
    - [`UploadedFile`][pydantic_ai.messages.UploadedFile] items unless `preserve_file_data=True`.
      Like a non-HTTP `FileUrl`, an `UploadedFile` references an object the model provider fetches
      using the server-side IAM role. Applies to uploaded files in user content and those nested in
      tool return parts.
    - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s and
      [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart]s at the end of the history
      that aren't in `resolved_tool_call_ids`. An unresolved tool call at the end of client-supplied
      history doesn't correspond to a paused agent run and shouldn't be executed. If stripping
      leaves the final response with no parts, the response is dropped from history entirely.

    Args:
        messages: Message history to sanitize.
        strip_system_prompts: Whether to strip
            [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s.
        allowed_file_url_schemes: URL schemes allowed for [`FileUrl`][pydantic_ai.messages.FileUrl]
            parts. Defaults to `http` and `https`.
        allowed_file_url_force_download: Additional
            [`FileUrl.force_download`][pydantic_ai.messages.FileUrl.force_download] values to allow.
            `False` is always allowed. Defaults to no additional values.
        preserve_file_data: Whether to keep [`UploadedFile`][pydantic_ai.messages.UploadedFile] items.
        resolved_tool_call_ids: Tool call IDs to preserve when the final response ends with tool calls.
            Use this for human-in-the-loop resumption when matching tool results are being submitted
            with the same request.
    """
    allowed_schemes = {scheme.lower() for scheme in allowed_file_url_schemes}
    allowed_force_download = set(allowed_file_url_force_download)
    resolved_ids = set(resolved_tool_call_ids)

    stripped_system_prompt = False
    disallowed_url_schemes: set[str] = set()
    reset_force_download_values: set[ForceDownloadMode] = set()
    dropped_uploaded_file_providers: set[str] = set()
    dangling_tool_call_names: list[str] = []

    sanitized: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            new_request_parts, request_stripped_system_prompt = _sanitize_request_parts(
                message.parts,
                strip_system_prompts=strip_system_prompts,
                allowed_file_url_schemes=allowed_schemes,
                allowed_file_url_force_download=allowed_force_download,
                preserve_file_data=preserve_file_data,
                disallowed_schemes=disallowed_url_schemes,
                reset_force_download_values=reset_force_download_values,
                dropped_uploaded_file_providers=dropped_uploaded_file_providers,
            )
            stripped_system_prompt = stripped_system_prompt or request_stripped_system_prompt
            if new_request_parts:
                sanitized.append(replace(message, parts=new_request_parts))
            # Otherwise drop the request entirely so we don't leave an empty
            # `ModelRequest(parts=[])` in history.
        elif isinstance(message, ModelResponse):
            new_response_parts = _sanitize_response_parts(
                message.parts,
                allowed_file_url_schemes=allowed_schemes,
                allowed_file_url_force_download=allowed_force_download,
                preserve_file_data=preserve_file_data,
                disallowed_schemes=disallowed_url_schemes,
                reset_force_download_values=reset_force_download_values,
                dropped_uploaded_file_providers=dropped_uploaded_file_providers,
            )
            if new_response_parts:
                sanitized.append(replace(message, parts=new_response_parts))
            # Otherwise drop the response entirely so we don't leave an empty
            # `ModelResponse(parts=[])` in history.
        else:
            assert_never(message)

    _strip_dangling_tail_tool_calls(sanitized, resolved_ids, dangling_tool_call_names)

    if stripped_system_prompt:
        warnings.warn(
            'Client-submitted system prompts were stripped. Pass `strip_system_prompts=False` only when the '
            "client is trusted to own the system prompt, or set `manage_system_prompt='client'` on a UI adapter.",
            UserWarning,
            stacklevel=2,
        )

    if disallowed_url_schemes:
        warnings.warn(
            f'Client-submitted file URLs with scheme(s) {sorted(disallowed_url_schemes)!r} '
            f'were dropped because those schemes are not in `allowed_file_url_schemes` '
            f'(currently {sorted(allowed_schemes)!r}). Non-HTTP schemes like '
            f'`s3://` or `gs://` are fetched by the model provider using the server-side IAM role, '
            f'so they should only be accepted from trusted clients. To allow a scheme, add it to '
            f'`allowed_file_url_schemes`.',
            UserWarning,
            stacklevel=2,
        )

    if reset_force_download_values:
        warnings.warn(
            f'Client-submitted file URLs with `force_download` value(s) '
            f'{sorted(reset_force_download_values, key=repr)!r} were reset to `False` because '
            f'those values are not in `allowed_file_url_force_download` '
            f'(currently {sorted(allowed_force_download, key=repr)!r}). '
            f"`'allow-local'` opts the URL out of the SSRF private-IP block and `True` makes "
            f'the server fetch the file itself, so neither should be accepted from untrusted '
            f'clients. To allow a value, add it to `allowed_file_url_force_download`, or set '
            f'it only on trusted server-side `message_history` passed directly to `Agent.run`.',
            UserWarning,
            stacklevel=2,
        )

    if dropped_uploaded_file_providers:
        warnings.warn(
            f'Client-submitted uploaded file(s) for provider(s) {sorted(dropped_uploaded_file_providers)!r} '
            f'were dropped because `preserve_file_data` is `False` (the default). Like a non-HTTP file URL, '
            f'an uploaded file references an object the model provider fetches using the server-side IAM role '
            f'or service account, so it should only be accepted from trusted clients. To keep uploaded files '
            f'from the client, set `preserve_file_data=True`, or pass them on trusted server-side '
            f'`message_history` directly to `Agent.run`.',
            UserWarning,
            stacklevel=2,
        )

    if dangling_tool_call_names:
        warnings.warn(
            f'Client-submitted history ended with unresolved tool call(s) '
            f'{sorted(set(dangling_tool_call_names))!r}, which were stripped. Tool calls are '
            f'produced by the model on the server side, so an unresolved tool call at the end '
            f'of client-supplied history does not correspond to a paused agent run. For '
            f'human-in-the-loop resumption, pass matching tool call IDs to `resolved_tool_call_ids`, '
            f'or pass matching `deferred_tool_results` to a UI adapter run method.',
            UserWarning,
            stacklevel=2,
        )

    return sanitized


def _strip_dangling_tail_tool_calls(
    sanitized: list[ModelMessage],
    resolved_tool_call_ids: set[str],
    dangling_names: list[str],
) -> None:
    """Strip unresolved (dangling) tool calls from the surviving tail of already-sanitized history.

    The tail is only known once empty messages have been dropped: a trailing `ModelRequest` that
    sanitized to empty (e.g. a client-supplied system prompt) is gone, which can re-expose an earlier
    [`ModelResponse`][pydantic_ai.messages.ModelResponse] whose tool calls a promptless run would
    dispatch directly. Anchoring on the pre-drop index would miss that re-exposed response. Walks
    back over trailing responses so several dropped messages can't hide a dangling call, keeping
    calls in `resolved_tool_call_ids` so a same-request human-in-the-loop resume still works.

    Mutates `sanitized` (dropping/rewriting trailing responses) and appends stripped tool names to
    `dangling_names` in place.
    """
    while sanitized and isinstance(tail := sanitized[-1], ModelResponse):
        kept_parts: list[ModelResponsePart] = []
        for part in tail.parts:
            if isinstance(part, BaseToolCallPart) and part.tool_call_id not in resolved_tool_call_ids:
                dangling_names.append(part.tool_name)
            else:
                kept_parts.append(part)
        if len(kept_parts) == len(tail.parts):
            break
        if kept_parts:
            sanitized[-1] = replace(tail, parts=kept_parts)
            break
        sanitized.pop()


def _sanitize_request_parts(
    parts: Sequence[ModelRequestPart],
    *,
    strip_system_prompts: bool,
    allowed_file_url_schemes: set[str],
    allowed_file_url_force_download: set[ForceDownloadMode],
    preserve_file_data: bool,
    disallowed_schemes: set[str],
    reset_force_download_values: set[ForceDownloadMode],
    dropped_uploaded_file_providers: set[str],
) -> tuple[list[ModelRequestPart], bool]:
    """Sanitize the parts of an untrusted [`ModelRequest`][pydantic_ai.messages.ModelRequest].

    `disallowed_schemes`, `reset_force_download_values`, and `dropped_uploaded_file_providers` are
    updated in place with any non-allowlisted file URL schemes, `force_download` values, and dropped
    uploaded file providers encountered.
    Returns the kept parts and whether any [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s
    were stripped.
    """
    stripped_system_prompt = False
    new_parts: list[ModelRequestPart] = []
    for part in parts:
        if strip_system_prompts and isinstance(part, SystemPromptPart):
            stripped_system_prompt = True
            continue
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
            filtered_content = _filter_user_content(
                part.content,
                allowed_file_url_schemes,
                allowed_file_url_force_download,
                preserve_file_data,
                disallowed_schemes,
                reset_force_download_values,
                dropped_uploaded_file_providers,
            )
            new_parts.append(replace(part, content=filtered_content))
        elif isinstance(part, BaseToolReturnPart) and part.tool_kind is None:
            # Skip narrower subclasses (`tool_kind` set): their `content` is a typed
            # `TypedDict` with required fields, and stripping a `FileUrl`-bearing key
            # during sanitization would leave it schema-invalid.
            keep_content, sanitized_content = _sanitize_tool_return_content(
                part.content,
                allowed_file_url_schemes,
                allowed_file_url_force_download,
                preserve_file_data,
                disallowed_schemes,
                reset_force_download_values,
                dropped_uploaded_file_providers,
            )
            new_parts.append(replace(part, content=sanitized_content if keep_content else None))
        else:
            new_parts.append(part)
    return new_parts, stripped_system_prompt


def _filter_user_content(
    content: Sequence[UserContent],
    allowed_file_url_schemes: set[str],
    allowed_file_url_force_download: set[ForceDownloadMode],
    preserve_file_data: bool,
    disallowed_schemes: set[str],
    reset_force_download_values: set[ForceDownloadMode],
    dropped_uploaded_file_providers: set[str],
) -> list[UserContent]:
    """Sanitize untrusted file references (file URLs and uploaded files) in user content.

    Drops file URLs whose scheme isn't in the allowlist, and resets `force_download` values that
    aren't `False` and aren't in `allowed_file_url_force_download` on kept items to `False`. Drops
    uploaded files unless `preserve_file_data` is set.

    `disallowed_schemes`, `reset_force_download_values`, and `dropped_uploaded_file_providers` are
    updated in place with any disallowed schemes, reset `force_download` values, and dropped uploaded
    file providers encountered.
    """
    filtered: list[UserContent] = []
    for item in content:
        if isinstance(item, FileUrl):
            scheme = urlparse(item.url).scheme.lower()
            if scheme and scheme not in allowed_file_url_schemes:
                disallowed_schemes.add(scheme)
                continue
            item = _sanitize_file_url(item, allowed_file_url_force_download, reset_force_download_values)
        elif isinstance(item, UploadedFile) and not preserve_file_data:
            dropped_uploaded_file_providers.add(item.provider_name)
            continue
        filtered.append(item)
    return filtered


def _sanitize_file_url(
    file_url: _FileUrlT,
    allowed_file_url_force_download: set[ForceDownloadMode],
    reset_force_download_values: set[ForceDownloadMode],
) -> _FileUrlT:
    """Reset a [`FileUrl`][pydantic_ai.messages.FileUrl]'s `force_download` if it's not allowlisted.

    `reset_force_download_values` is updated in place with the original value when it's reset.
    """
    if file_url.force_download is not False and file_url.force_download not in allowed_file_url_force_download:
        reset_force_download_values.add(file_url.force_download)
        return replace(file_url, force_download=False)
    return file_url


def _sanitize_tool_return_content(
    content: ToolReturnContent,
    allowed_file_url_schemes: set[str],
    allowed_file_url_force_download: set[ForceDownloadMode],
    preserve_file_data: bool,
    disallowed_schemes: set[str],
    reset_force_download_values: set[ForceDownloadMode],
    dropped_uploaded_file_providers: set[str],
) -> tuple[bool, ToolReturnContent]:
    """Recursively sanitize file references (file URLs and uploaded files) nested in tool return content.

    Tool return content is an arbitrarily nested structure of files, sequences, and mappings,
    so any `FileUrl` or `UploadedFile` it contains — including those introduced by multimodal tool
    returns — is walked and sanitized the same way file references in user content are: file URL
    schemes and `force_download` are checked, and uploaded files are dropped unless `preserve_file_data`
    is set.

    `disallowed_schemes`, `reset_force_download_values`, and `dropped_uploaded_file_providers` are
    updated in place with any disallowed schemes, reset `force_download` values, and dropped uploaded
    file providers encountered.
    """
    if isinstance(content, FileUrl):
        scheme = urlparse(content.url).scheme.lower()
        if scheme and scheme not in allowed_file_url_schemes:
            disallowed_schemes.add(scheme)
            return False, content
        return True, _sanitize_file_url(content, allowed_file_url_force_download, reset_force_download_values)
    if isinstance(content, UploadedFile):
        if not preserve_file_data:
            dropped_uploaded_file_providers.add(content.provider_name)
            return False, content
        return True, content
    # `ToolReturnContent` is a recursive `TypeAliasType` at runtime (for Pydantic validation)
    # but resolves to `Any` at type-check time, so pyright can't infer the element types.
    if isinstance(content, Mapping):
        mapping: Mapping[str, ToolReturnContent] = content  # pyright: ignore[reportUnknownVariableType]
        sanitized_mapping: dict[str, ToolReturnContent] = {}
        for key, value in mapping.items():
            keep, sanitized_value = _sanitize_tool_return_content(
                value,
                allowed_file_url_schemes,
                allowed_file_url_force_download,
                preserve_file_data,
                disallowed_schemes,
                reset_force_download_values,
                dropped_uploaded_file_providers,
            )
            if keep:
                sanitized_mapping[key] = sanitized_value
        return True, sanitized_mapping
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        sequence: Sequence[ToolReturnContent] = content  # pyright: ignore[reportUnknownVariableType]
        sanitized_sequence: list[ToolReturnContent] = []
        for item in sequence:
            keep, sanitized_item = _sanitize_tool_return_content(
                item,
                allowed_file_url_schemes,
                allowed_file_url_force_download,
                preserve_file_data,
                disallowed_schemes,
                reset_force_download_values,
                dropped_uploaded_file_providers,
            )
            if keep:
                sanitized_sequence.append(sanitized_item)
        return True, sanitized_sequence
    return True, content


def _sanitize_response_parts(
    parts: Sequence[ModelResponsePart],
    *,
    allowed_file_url_schemes: set[str],
    allowed_file_url_force_download: set[ForceDownloadMode],
    preserve_file_data: bool,
    disallowed_schemes: set[str],
    reset_force_download_values: set[ForceDownloadMode],
    dropped_uploaded_file_providers: set[str],
) -> list[ModelResponsePart]:
    """Sanitize the file references nested in an untrusted response's tool return parts.

    Drops non-allowlisted schemes and resets non-allowlisted `force_download` values on
    [`FileUrl`][pydantic_ai.messages.FileUrl]s nested in tool return parts, and drops
    [`UploadedFile`][pydantic_ai.messages.UploadedFile]s nested in tool return parts unless
    `preserve_file_data` is set. Unresolved (dangling) tool calls are stripped separately, from
    the surviving tail, by `sanitize_message_history`.

    `disallowed_schemes`, `reset_force_download_values`, and `dropped_uploaded_file_providers` are
    updated in place with any disallowed schemes, reset `force_download` values, and dropped uploaded
    file providers encountered.
    """
    new_parts: list[ModelResponsePart] = []
    for part in parts:
        if isinstance(part, BaseToolReturnPart) and part.tool_kind is None:
            # Skip narrower subclasses (`tool_kind` set): their `content` is a typed
            # `TypedDict` with required fields, and stripping a `FileUrl`-bearing key
            # during sanitization would leave it schema-invalid.
            keep_content, sanitized_content = _sanitize_tool_return_content(
                part.content,
                allowed_file_url_schemes,
                allowed_file_url_force_download,
                preserve_file_data,
                disallowed_schemes,
                reset_force_download_values,
                dropped_uploaded_file_providers,
            )
            new_parts.append(replace(part, content=sanitized_content if keep_content else None))
        else:
            new_parts.append(part)
    return new_parts
