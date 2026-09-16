"""Streaming terminal conversations with lazy exports for import-time branding."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._app import chat, create_agent
    from ._rendering import StreamRenderer
    from ._session import Session

__all__ = ['Session', 'StreamRenderer', 'chat', 'create_agent']


def __getattr__(name: str) -> object:
    if name in ('chat', 'create_agent'):
        from ._app import chat, create_agent

        return chat if name == 'chat' else create_agent
    if name == 'Session':
        from ._session import Session

        return Session
    if name == 'StreamRenderer':
        from ._rendering import StreamRenderer

        return StreamRenderer
    raise AttributeError(name)
