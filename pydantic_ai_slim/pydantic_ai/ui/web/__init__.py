"""Web-based chat UI for Pydantic AI agents."""

from ._mcp import load_mcp_server_tools
from .api import AIModel, BuiltinToolInfo, add_api_routes
from .app import ModelsParam, create_web_app, format_model_display_name

__all__ = [
    'create_web_app',
    'add_api_routes',
    'AIModel',
    'BuiltinToolInfo',
    'ModelsParam',
    'load_mcp_server_tools',
    'format_model_display_name',
]
