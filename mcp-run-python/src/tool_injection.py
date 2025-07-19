"""Tool injection for MCP elicitation support."""

import json
from typing import Any, Callable, cast


def _create_elicitation_request(tool_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Create an elicitation request object for tool execution."""
    tool_args: dict[str, Any] = {}

    # Handle positional arguments
    if args:
        if len(args) == 1 and isinstance(args[0], str):
            # Single string argument - assume it's a query
            tool_args['query'] = args[0]
        elif len(args) == 1 and isinstance(args[0], dict):
            # Single dict argument
            tool_args.update(args[0])  # type: ignore[arg-type]

    # Add keyword arguments
    tool_args.update(kwargs)

    tool_execution_data: dict[str, Any] = {'tool_name': tool_name, 'arguments': tool_args}
    return {
        'message': json.dumps(tool_execution_data),
        'requestedSchema': {
            'type': 'object',
            'properties': {'result': {'type': 'string', 'description': f'Result of executing {tool_name} tool'}},
            'required': ['result'],
        },
    }


def _handle_tool_callback_result(result: Any, tool_name: str) -> Any:
    """Handle the result from a tool callback, including promise resolution."""
    # Handle PyodideFuture (JavaScript Promise)
    if hasattr(result, 'then'):
        try:
            # Import at runtime to avoid dependency issues
            from pyodide.webloop import run_sync  # type: ignore[import-untyped]

            # Use cast to tell the type checker what we expect
            resolved_result = cast(dict[str, Any], run_sync(result))

            # Extract result from elicitation response
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


def _create_tool_function(tool_name: str, tool_callback: Callable[[Any], Any]) -> Callable[..., Any]:
    """Create a tool function that can be called from Python."""

    def tool_function(*args: Any, **kwargs: Any) -> Any:
        """Synchronous tool function that handles the async callback properly."""
        # Note: tool_callback is guaranteed to be not None due to check in inject_tool_functions

        elicitation_request = _create_elicitation_request(tool_name, args, kwargs)

        try:
            result = tool_callback(elicitation_request)
            return _handle_tool_callback_result(result, tool_name)
        except Exception as e:
            raise Exception(f'Tool {tool_name} failed: {str(e)}')

    return tool_function


def inject_tool_functions(
    globals_dict: dict[str, Any], available_tools: list[str], tool_callback: Callable[[Any], Any] | None = None
) -> None:
    """Inject tool functions into the global namespace.

    Args:
        globals_dict: Global namespace to inject tools into
        available_tools: List of available tool names
        tool_callback: Optional callback for tool execution
    """
    if not available_tools:
        return

    # Inject tool functions into globals
    for tool_name in available_tools:
        if tool_callback is not None:
            globals_dict[tool_name] = _create_tool_function(tool_name, tool_callback)
