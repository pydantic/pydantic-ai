from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

try:
    from stackone_ai import StackOneToolSet
except ImportError as _import_error:
    raise ImportError(
        'Please install `stackone-ai` to use StackOne tools. '
        'Note that stackone-ai requires Python 3.11 or higher. '
        'Install with: pip install stackone-ai'
    ) from _import_error


def tool_from_stackone(
    tool_name: str,
    *,
    account_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Creates a Pydantic AI tool proxy from a StackOne tool.

    Args:
        tool_name: The name of the StackOne tool to wrap (e.g., "hris_list_employees").
        account_id: The StackOne account ID. If not provided, uses STACKONE_ACCOUNT_ID env var.
        api_key: The StackOne API key. If not provided, uses STACKONE_API_KEY env var.
        base_url: Custom base URL for StackOne API. Optional.

    Returns:
        A Pydantic AI tool that corresponds to the StackOne tool.
    """
    # Initialize StackOneToolSet
    stackone_toolset = StackOneToolSet(
        api_key=api_key or os.getenv('STACKONE_API_KEY'),
        account_id=account_id or os.getenv('STACKONE_ACCOUNT_ID'),
        **({'base_url': base_url} if base_url else {}),
    )

    # Get tools that match the specific tool name
    tools = stackone_toolset.get_tools([tool_name])

    # Get the specific tool
    stackone_tool = tools.get_tool(tool_name)

    # Extract tool information from the StackOne tool
    # We'll use the tool's call method and extract schema information
    def implementation(**kwargs: Any) -> str:
        """Execute the StackOne tool with provided arguments."""
        try:
            result = stackone_tool.call(**kwargs)
            # Convert result to string if it's not already
            if isinstance(result, str):
                return result
            # For complex objects, return JSON representation
            import json

            return json.dumps(result, default=str)
        except Exception as e:
            return f'Error executing StackOne tool: {str(e)}'

    # Create a basic JSON schema for the tool
    # In a real implementation, you'd want to extract this from the StackOne tool's schema
    json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}, 'additionalProperties': True, 'required': []}

    return Tool.from_schema(
        function=implementation,
        name=tool_name,
        description=f'StackOne tool: {tool_name}',
        json_schema=json_schema,
    )


class StackOneToolset(FunctionToolset):
    """A toolset that wraps StackOne tools."""

    def __init__(
        self,
        tool_patterns: Sequence[str] | str | None = None,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        id: str | None = None,
    ):
        """Initialize StackOneToolset.

        Args:
            tool_patterns: Glob patterns to filter tools (e.g., ["hris_*", "!hris_delete_*"]).
                         If None, includes all tools. Can be a single string or list of strings.
            account_id: The StackOne account ID. If not provided, uses STACKONE_ACCOUNT_ID env var.
            api_key: The StackOne API key. If not provided, uses STACKONE_API_KEY env var.
            base_url: Custom base URL for StackOne API. Optional.
            id: Optional toolset ID.
        """
        # Initialize StackOneToolSet
        self._stackone_toolset = StackOneToolSet(
            api_key=api_key or os.getenv('STACKONE_API_KEY'),
            account_id=account_id or os.getenv('STACKONE_ACCOUNT_ID'),
            **({'base_url': base_url} if base_url else {}),
        )

        # Handle tool patterns
        if tool_patterns is None:
            patterns = ['*']
        elif isinstance(tool_patterns, str):
            patterns = [tool_patterns]
        else:
            patterns = list(tool_patterns)

        # Get tools from StackOneToolSet
        self._stackone_tools = self._stackone_toolset.get_tools(patterns)

        # Convert StackOne tools to Pydantic AI tools
        pydantic_tools = []

        # Get available tool names
        # This is a simplified approach - in practice, you'd want to
        # inspect the StackOne tools more thoroughly
        try:
            # Try to get meta tools to discover available tools
            meta_tools = self._stackone_tools.meta_tools()
            filter_tool = meta_tools.get_tool('meta_filter_relevant_tools')

            # Get a broad search to find available tools
            available_tools = filter_tool.call(query='', limit=100, min_score=0.0)

            if isinstance(available_tools, dict) and 'tools' in available_tools:
                tool_names = [tool.get('name', '') for tool in available_tools['tools'] if tool.get('name')]
            else:
                # Fallback to common HRIS tool names if meta discovery fails
                tool_names = [
                    'hris_list_employees',
                    'hris_get_employee',
                    'hris_create_employee',
                    'hris_update_employee',
                    'hris_list_departments',
                    'hris_get_department',
                ]
        except Exception:
            # If meta tools fail, use common tool names as fallback
            tool_names = [
                'hris_list_employees',
                'hris_get_employee',
                'hris_create_employee',
                'hris_update_employee',
                'hris_list_departments',
                'hris_get_department',
            ]

        # Create Pydantic AI tools for each discovered tool
        for tool_name in tool_names:
            try:
                tool = self._create_tool_from_name(tool_name)
                if tool:
                    pydantic_tools.append(tool)
            except Exception:
                # Skip tools that can't be created
                continue

        super().__init__(pydantic_tools, id=id)

    def _create_tool_from_name(self, tool_name: str) -> Tool | None:
        """Create a Pydantic AI tool from a StackOne tool name."""
        try:
            # Get the specific tool from StackOne
            stackone_tool = self._stackone_tools.get_tool(tool_name)

            def implementation(**kwargs: Any) -> str:
                """Execute the StackOne tool with provided arguments."""
                try:
                    result = stackone_tool.call(**kwargs)
                    # Convert result to string if it's not already
                    if isinstance(result, str):
                        return result
                    # For complex objects, return JSON representation
                    import json

                    return json.dumps(result, default=str)
                except Exception as e:
                    return f"Error executing StackOne tool '{tool_name}': {str(e)}"

            # Create a basic JSON schema for the tool
            json_schema: JsonSchemaValue = {
                'type': 'object',
                'properties': {},
                'additionalProperties': True,
                'required': [],
            }

            return Tool.from_schema(
                function=implementation,
                name=tool_name,
                description=f'StackOne tool: {tool_name}',
                json_schema=json_schema,
            )

        except Exception:
            return None
