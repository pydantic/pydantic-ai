"""Tool injection for MCP elicitation support."""

import json
from functools import lru_cache
from typing import Any, Callable, cast

from pyodide.webloop import run_sync  # type: ignore[import-untyped]


def _create_elicitation_request(tool_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Create an elicitation request object for tool execution."""
    if args and len(args) == 1:
        first_arg = args[0]
        if isinstance(first_arg, str):
            tool_args = {'query': first_arg, **kwargs}
        elif isinstance(first_arg, dict):
            tool_args = {**first_arg, **kwargs}  # type: ignore[arg-type]
        else:
            tool_args = kwargs.copy()
    else:
        tool_args = kwargs.copy()

    return {
        'message': json.dumps({'tool_name': tool_name, 'arguments': tool_args}),
        'requestedSchema': _get_tool_schema(tool_name),
    }


@lru_cache(maxsize=128)
def _get_tool_schema(tool_name: str) -> dict[str, Any]:
    """Get cached schema for a tool."""
    return {
        'type': 'object',
        'properties': {'result': {'type': 'string', 'description': f'Result of executing {tool_name} tool'}},
        'required': ['result'],
    }


def _handle_tool_callback_result(result: Any, tool_name: str) -> Any:
    """Handle the result from a tool callback, including promise resolution."""
    if hasattr(result, 'then'):
        try:
            resolved_result = cast(dict[str, Any], run_sync(result))
            if isinstance(resolved_result, dict):
                if resolved_result.get('action') == 'accept':
                    content = cast(dict[str, Any], resolved_result.get('content', {}))
                    return content.get('result')
                elif resolved_result.get('action') == 'decline':
                    raise Exception(f'Tool {tool_name} execution declined')
                elif resolved_result.get('action') == 'cancel':
                    raise Exception(f'Tool {tool_name} execution cancelled')
            return resolved_result
        except Exception as promise_error:
            raise Exception(f'Tool {tool_name} promise error: {str(promise_error)}')
    else:
        return result


def _create_tool_function(
    tool_name: str, tool_callback: Callable[[Any], Any], globals_dict: dict[str, Any]
) -> Callable[..., Any]:
    """Create a tool function that can be called from Python."""

    def tool_function(*args: Any, **kwargs: Any) -> Any:
        """Tool function that calls the MCP elicitation callback."""
        elicitation_request = _create_elicitation_request(tool_name=tool_name, args=args, kwargs=kwargs)

        try:
            result = tool_callback(elicitation_request)
            return _handle_tool_callback_result(result, tool_name)
        except Exception as e:
            raise Exception(f'Tool {tool_name} failed: {str(e)}')

    return tool_function


def inject_tool_functions(
    globals_dict: dict[str, Any],
    available_tools: list[str],
    tool_callback: Callable[[Any], Any] | None = None,
) -> None:
    """Inject tool functions into the global namespace.

    Args:
        globals_dict: Global namespace to inject tools into
        available_tools: List of available tool names
        tool_callback: Optional callback for tool execution
    """
    if not available_tools:
        return

    for tool_name in available_tools:
        if tool_callback is not None:
            python_name = tool_name.replace('-', '_')
            globals_dict[python_name] = _create_tool_function(tool_name, tool_callback, globals_dict)
