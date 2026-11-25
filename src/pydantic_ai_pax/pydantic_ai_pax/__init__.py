"""Streaming terminal conversations with lazy exports for import-time branding."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_pax.app import DEFAULT_PLUGINS, chat, create_agent, open_stock_agent
    from pydantic_ai_pax.runtime.session import Session
    from pydantic_ai_pax.ui.rendering.rendering import StreamRenderer

__all__ = ['DEFAULT_PLUGINS', 'Session', 'StreamRenderer', 'chat', 'create_agent', 'open_stock_agent']


def __getattr__(name: str) -> object:
    if name in ('chat', 'create_agent', 'open_stock_agent', 'DEFAULT_PLUGINS'):
        from pydantic_ai_pax import app

        return getattr(app, name)
    if name == 'Session':
        from pydantic_ai_pax.runtime.session import Session

        return Session
    if name == 'StreamRenderer':
        from pydantic_ai_pax.ui.rendering.rendering import StreamRenderer

        return StreamRenderer
    raise AttributeError(name)
