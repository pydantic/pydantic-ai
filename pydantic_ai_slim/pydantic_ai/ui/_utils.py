"""Helpers shared by the UI protocol adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from pydantic_ai import messages as _messages
from pydantic_ai._mcp_apps import MCPAppToolResult
from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import ModelMessage, TextContent, ToolReturnContent

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

    For dumping to a client: the framework keeps state there, such as the mark on a frame a capability wrote, and it
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
