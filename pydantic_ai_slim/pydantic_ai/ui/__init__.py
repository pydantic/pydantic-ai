from __future__ import annotations

from typing import TYPE_CHECKING

from ._adapter import StateDeps, StateHandler, UIAdapter
from ._event_stream import SSE_CONTENT_TYPE, NativeEvent, OnCancelFunc, OnCompleteFunc, UIEventStream
from ._messages_builder import BuilderCheckpoint, MessagesBuilder

if TYPE_CHECKING:
    from ._web import DEFAULT_HTML_URL, OFFLINE_HTML_URL

__all__ = [
    'UIAdapter',
    'UIEventStream',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'NativeEvent',
    'OnCompleteFunc',
    'OnCancelFunc',
    'MessagesBuilder',
    'BuilderCheckpoint',
    'DEFAULT_HTML_URL',
    'OFFLINE_HTML_URL',
]


def __getattr__(name: str) -> object:
    if name == 'DEFAULT_HTML_URL':
        from ._web import DEFAULT_HTML_URL

        return DEFAULT_HTML_URL
    if name == 'OFFLINE_HTML_URL':
        from ._web import OFFLINE_HTML_URL

        return OFFLINE_HTML_URL
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
