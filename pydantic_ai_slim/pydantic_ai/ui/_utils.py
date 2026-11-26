"""Helpers shared by the UI protocol adapters."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final, TypeVar, assert_never, overload

from pydantic_ai import messages as _messages
from pydantic_ai._mcp_apps import MCP_APP_METADATA_KEY, MCPAppToolResult
from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    BaseToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]
    TextContent,
    ToolCallPart,
    ToolReturnContent,
    ToolReturnPart,
    ToolUIVisibility,
    _translate_legacy_retry_part,  # pyright: ignore[reportPrivateUsage]
)

INTERNAL_METADATA_KEY = _messages._PYDANTIC_AI_METADATA_KEY  # pyright: ignore[reportPrivateUsage]
"""Reserved key in UI message metadata for state the adapters own; never read from client-controlled metadata."""
_UI_MESSAGE_ID_KEY = 'ui_message_id'

MCP_APPS_UI_TYPE: Final = 'mcp-apps'
"""Name of the UI event that carries an MCP Apps record: the AG-UI activity type and the Vercel AI `data-*` suffix.

The AG-UI MCP Apps middleware fixes this name, and renderers such as CopilotKit's only render activities of this
type. Vercel AI reuses it so both protocols send the same event.
"""


def set_ui_message_id(message: ModelMessage, ui_message_id: str) -> None:
    """Keep the UI message id that `message` was loaded from, for `get_ui_message_id` to restore.

    It lives under the reserved `__pydantic_ai__` namespace, which the UI adapters never write to or
    read from client-controlled message metadata.
    """
    metadata = message.metadata or {}
    namespace = metadata.get(INTERNAL_METADATA_KEY)
    message.metadata = {
        **metadata,
        INTERNAL_METADATA_KEY: {**(namespace if is_str_dict(namespace) else {}), _UI_MESSAGE_ID_KEY: ui_message_id},
    }


def get_ui_message_id(message: ModelMessage) -> str | None:
    """Return the UI message id kept by `set_ui_message_id`, if any."""
    namespace = (message.metadata or {}).get(INTERNAL_METADATA_KEY)
    ui_message_id = namespace.get(_UI_MESSAGE_ID_KEY) if is_str_dict(namespace) else None
    return ui_message_id if isinstance(ui_message_id, str) else None


def without_ui_message_id(message: ModelMessage) -> ModelMessage:
    """Return a copy of `message` without the UI message id kept by `set_ui_message_id`.

    Leaves out the reserved namespace, and then `metadata` itself, if nothing else is left in them.
    """
    metadata = message.metadata
    namespace = (metadata or {}).get(INTERNAL_METADATA_KEY)
    if metadata is None or not is_str_dict(namespace) or _UI_MESSAGE_ID_KEY not in namespace:
        return message
    rest = {key: value for key, value in namespace.items() if key != _UI_MESSAGE_ID_KEY}
    new_metadata = {key: value for key, value in metadata.items() if key != INTERNAL_METADATA_KEY}
    if rest:
        new_metadata[INTERNAL_METADATA_KEY] = rest
    return replace(message, metadata=new_metadata or None)


def without_internal_text_metadata(content: ToolReturnContent) -> ToolReturnContent:
    """Tool return `content` with the reserved namespace left out of the metadata of each `TextContent` in it.

    For dumping to a client: the framework keeps state there, such as the mark an earlier version put on a frame a capability wrote, and it
    stays server-side like the reserved namespace of message metadata. Sequences and mappings are rebuilt as lists and
    dicts, which dump the same.
    """
    if isinstance(content, TextContent):
        return _messages._text_without_internal_metadata(content)  # pyright: ignore[reportPrivateUsage]
    # `ToolReturnContent` resolves to `Any` at type-check time, so pyright can't infer the element types.
    if isinstance(content, Mapping):
        mapping: Mapping[str, ToolReturnContent] = content  # pyright: ignore[reportUnknownVariableType]
        return {key: without_internal_text_metadata(value) for key, value in mapping.items()}
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        sequence: Sequence[ToolReturnContent] = content  # pyright: ignore[reportUnknownVariableType]
        return [without_internal_text_metadata(item) for item in sequence]
    return content


def client_error_text(part: BaseToolReturnPart[Any]) -> str:
    """The unwrapped error text of an errored or denied tool return, for sending to a client.

    Like `without_internal_text_metadata`, it leaves the reserved namespace out of the metadata of each
    `TextContent`, which the text otherwise includes as JSON.
    """
    return replace(part, content=without_internal_text_metadata(part.content)).model_response_str(wrap_if_error=False)


def mcp_app_ui_payload(app: MCPAppToolResult) -> dict[str, Any]:
    """Build the payload UI adapters send for an MCP App tool call.

    The keys match the `mcp-apps` activity content of the AG-UI MCP Apps middleware, so frontends can
    share one renderer across protocols. The middleware identifies the server by both `serverId` and
    `serverHash` (a digest of its connection config), and renderers such as CopilotKit's require
    `serverHash`, so the toolset ID fills both.
    """
    payload: dict[str, Any] = {
        'resourceUri': app['resource_uri'],
        'toolInput': app['tool_input'],
        'result': app['result'],
    }
    if app['toolset_id'] is not None:
        payload['serverId'] = payload['serverHash'] = app['toolset_id']
    return payload


REDACTED_TOOL_RETURN_CONTENT: Final = '[REDACTED]'
"""What a client is sent in place of the content of a return whose tool is `'redacted'`."""

_ReturnPartT = TypeVar('_ReturnPartT', bound=BaseToolReturnPart[Any])


def ui_visibility_of(
    part: ToolCallPart | NativeToolCallPart | BaseToolReturnPart[Any],
    *,
    call_visibility: ToolUIVisibility | None,
    default: ToolUIVisibility,
) -> ToolUIVisibility:
    """What a client is sent for `part`: its own stamp, else its call's (for a return), else the adapter's default.

    A value outside `ToolUIVisibility` raises rather than reading as `'public'`.
    """
    visibility = part.ui_visibility or call_visibility or default
    match visibility:
        case 'public' | 'redacted' | 'private':
            return visibility
        case _:
            assert_never(visibility)


def client_tool_return(part: _ReturnPartT, visibility: ToolUIVisibility) -> _ReturnPartT | None:
    """`part` as a client is sent it under `visibility`: `None` when it is `'private'`.

    A `'redacted'` return's content becomes the placeholder, a retry loses the feedback it carries, and the MCP Apps
    record, which holds the full result, leaves the metadata. The rest of the metadata stays: the events a tool puts
    there are the channel it chose for the client.
    """
    match visibility:
        case 'public':
            return part
        case 'private':
            return None
        case 'redacted':
            metadata = part.metadata
            if is_str_dict(metadata) and MCP_APP_METADATA_KEY in metadata:
                metadata = {key: value for key, value in metadata.items() if key != MCP_APP_METADATA_KEY} or None
            if isinstance(part, ToolReturnPart):
                return replace(part, content=REDACTED_TOOL_RETURN_CONTENT, metadata=metadata, retry=None)
            return replace(part, content=REDACTED_TOOL_RETURN_CONTENT, metadata=metadata)
        case _:
            assert_never(visibility)


def warn_client_held_history_with_hidden_tools(*, stacklevel: int) -> None:
    """Warn that a client holding the history sends back what it was sent of a non-public tool as the tool's result."""
    warnings.warn(
        "A tool's `ui_visibility` keeps its result from the client, but the client holds the history, so it sends "
        'back the placeholder (or nothing) as the result and the model loses the real one. '
        "Keep the history on the server with `manage_message_history='server'` or a `conversation_store`.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _return_visibility(
    part: BaseToolReturnPart[Any], calls: Mapping[str, ToolUIVisibility], default_ui_visibility: ToolUIVisibility
) -> ToolUIVisibility:
    return ui_visibility_of(part, call_visibility=calls.get(part.tool_call_id), default=default_ui_visibility)


@overload
def _with_parts(message: ModelResponse, parts: list[ModelResponsePart]) -> ModelResponse | None: ...
@overload
def _with_parts(message: ModelRequest, parts: list[ModelRequestPart]) -> ModelRequest | None: ...
def _with_parts(message: ModelMessage, parts: list[ModelResponsePart] | list[ModelRequestPart]) -> ModelMessage | None:
    """`message` with the parts `client_view` shows: itself if they are its own, `None` if there are none left."""
    if len(parts) == len(message.parts) and all(new is old for new, old in zip(parts, message.parts)):
        return message
    return replace(message, parts=parts) if parts else None


def client_response_part(
    part: ModelResponsePart, calls: dict[str, ToolUIVisibility], default_ui_visibility: ToolUIVisibility
) -> ModelResponsePart | None:
    """`part` as a client is sent it, recording each tool call's visibility in `calls` for its return; `None` for nothing."""
    if isinstance(part, (ToolCallPart, NativeToolCallPart)):
        visibility = calls[part.tool_call_id] = ui_visibility_of(
            part, call_visibility=None, default=default_ui_visibility
        )
        return None if visibility == 'private' else part
    if isinstance(part, NativeToolReturnPart):
        return client_tool_return(part, _return_visibility(part, calls, default_ui_visibility))
    return part


def _client_request_part(
    part: ModelRequestPart, calls: Mapping[str, ToolUIVisibility], default_ui_visibility: ToolUIVisibility
) -> ModelRequestPart | None:
    """`part` as `client_view` shows it.

    A legacy retry bound to a tool call is that call's return, as the dumps translate it, so it takes the call's
    visibility; a shown one is left for the dumps to translate.
    """
    if isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
        translated = _translate_legacy_retry_part(part)
        if not isinstance(translated, ToolReturnPart):
            return part
        shown = client_tool_return(translated, _return_visibility(translated, calls, default_ui_visibility))
        return part if shown is translated else shown
    if isinstance(part, ToolReturnPart):
        return client_tool_return(part, _return_visibility(part, calls, default_ui_visibility))
    return part


def client_view(messages: Sequence[ModelMessage], default_ui_visibility: ToolUIVisibility) -> list[ModelMessage]:
    """`messages` as a client is sent them, by each tool part's `ui_visibility` stamp.

    A `'private'` call and its return are left out, a `'redacted'` return gets the placeholder, and a message
    left with no parts is dropped. A part with no stamp takes its call's, then `default_ui_visibility`. The messages
    themselves are not changed.
    """
    calls: dict[str, ToolUIVisibility] = {}
    view: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            response_parts = [
                shown
                for part in message.parts
                if (shown := client_response_part(part, calls, default_ui_visibility)) is not None
            ]
            projected: ModelMessage | None = _with_parts(message, response_parts)
        else:
            request_parts = [
                shown
                for part in message.parts
                if (shown := _client_request_part(part, calls, default_ui_visibility)) is not None
            ]
            projected = _with_parts(message, request_parts)
        if projected is not None:
            view.append(projected)
    return view
