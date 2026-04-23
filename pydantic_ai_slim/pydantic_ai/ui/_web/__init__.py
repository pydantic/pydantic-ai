"""Web-based chat UI for Pydantic AI agents."""

from .api import ModelsParam
from .app import DEFAULT_HTML_URL, create_web_app

__all__ = [
    'create_web_app',
    'ModelsParam',
    'DEFAULT_HTML_URL',
]
