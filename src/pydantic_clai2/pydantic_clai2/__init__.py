"""Streaming terminal conversations with lazy exports for import-time branding."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_clai2._app import DEFAULT_PLUGINS, chat, create_agent, open_stock_agent
    from pydantic_clai2.runtime._session import Session
    from pydantic_clai2.ui.rendering._rendering import StreamRenderer

__all__ = ['DEFAULT_PLUGINS', 'Session', 'StreamRenderer', 'chat', 'create_agent', 'open_stock_agent']


def __getattr__(name: str) -> object:
    if name in ('chat', 'create_agent', 'open_stock_agent', 'DEFAULT_PLUGINS'):
        from pydantic_clai2 import _app

        return getattr(_app, name)
    if name == 'Session':
        from pydantic_clai2.runtime._session import Session

        return Session
    if name == 'StreamRenderer':
        from pydantic_clai2.ui.rendering._rendering import StreamRenderer

        return StreamRenderer
    raise AttributeError(name)
