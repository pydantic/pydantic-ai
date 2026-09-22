"""Keep a UI protocol's message id on the `ModelMessage` it was loaded into, so a dump can restore it."""

from __future__ import annotations

from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import ModelMessage

_INTERNAL_METADATA_KEY = '__pydantic_ai__'
_UI_MESSAGE_ID_KEY = 'ui_message_id'


def set_ui_message_id(message: ModelMessage, ui_message_id: str) -> None:
    """Keep the UI message id that `message` was loaded from, for `get_ui_message_id` to restore.

    It lives under the reserved `__pydantic_ai__` namespace, which the UI adapters never write to or
    read from client-controlled message metadata.
    """
    metadata = message.metadata or {}
    namespace = metadata.get(_INTERNAL_METADATA_KEY)
    message.metadata = {
        **metadata,
        _INTERNAL_METADATA_KEY: {**(namespace if is_str_dict(namespace) else {}), _UI_MESSAGE_ID_KEY: ui_message_id},
    }


def get_ui_message_id(message: ModelMessage) -> str | None:
    """Return the UI message id kept by `set_ui_message_id`, if any."""
    namespace = (message.metadata or {}).get(_INTERNAL_METADATA_KEY)
    ui_message_id = namespace.get(_UI_MESSAGE_ID_KEY) if is_str_dict(namespace) else None
    return ui_message_id if isinstance(ui_message_id, str) else None
