"""Web-based chat UI for Pydantic AI agents."""

from ._mcp import load_mcp_server_tools
from .api import AIModel, BuiltinToolInfo, add_api_routes
from .app import create_web_app

__all__ = [
    'create_web_app',
    'add_api_routes',
    'AIModel',
    'BuiltinToolInfo',
    'load_mcp_server_tools',
    'format_model_display_name',
]


def format_model_display_name(model_name: str) -> str:
    """Format model name for display in UI.

    Handles common patterns:
    - gpt-5 -> GPT 5
    - claude-sonnet-4-5 -> Claude Sonnet 4.5
    - gemini-2.5-pro -> Gemini 2.5 Pro
    """
    parts = model_name.split('-')
    result: list[str] = []

    for i, part in enumerate(parts):
        if i == 0 and part.lower() == 'gpt':
            result.append(part.upper())
        elif part.replace('.', '').isdigit():
            if result and result[-1].replace('.', '').isdigit():
                result[-1] = f'{result[-1]}.{part}'
            else:
                result.append(part)
        else:
            result.append(part.capitalize())

    return ' '.join(result)
