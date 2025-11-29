"""Web-based chat UI for Pydantic AI agents."""

from .app import ModelsParam, create_web_app

__all__ = [
    'create_web_app',
    'ModelsParam',
]
