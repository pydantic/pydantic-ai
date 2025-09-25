from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import Tool

try:
    from aci import ACI
except ImportError as _import_error:
    raise ImportError('Please install `aci-sdk` to use ACI.dev tools') from _import_error


def _clean_schema(schema):
    if isinstance(schema, dict):
        # Remove non-standard keys (e.g., 'visible')
        return {k: _clean_schema(v) for k, v in schema.items() if k not in {'visible'}}
    elif isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    else:
        return schema


def tool_from_aci(aci_function: str, linked_account_owner_id: str) -> Tool:
    """Creates a Pydantic AI tool proxy from an ACI.dev function.

    Args:
        aci_function: The ACI.dev function to wrap.
        linked_account_owner_id: The ACI user ID to execute the function on behalf of.

    Returns:
        A Pydantic AI tool that corresponds to the ACI.dev tool.
    """
    aci = ACI()
    function_definition = aci.functions.get_definition(aci_function)
    function_name = function_definition['function']['name']
    function_description = function_definition['function']['description']
    inputs = function_definition['function']['parameters']

    json_schema = {
        'additionalProperties': inputs.get('additionalProperties', False),
        'properties': inputs.get('properties', {}),
        'required': inputs.get('required', []),
        # Default to 'object' if not specified
        'type': inputs.get('type', 'object'),
    }

    # Clean the schema
    json_schema = _clean_schema(json_schema)

    def implementation(*args: Any, **kwargs: Any) -> str:
        if args:
            raise TypeError('Positional arguments are not allowed')
        return aci.handle_function_call(
            function_name,
            kwargs,
            linked_account_owner_id=linked_account_owner_id,
            allowed_apps_only=True,
        )

    return Tool.from_schema(
        function=implementation,
        name=function_name,
        description=function_description,
        json_schema=json_schema,
    )


class ACIToolset(FunctionToolset):
    """A toolset that wraps ACI.dev tools."""

    def __init__(self, aci_functions: Sequence[str], linked_account_owner_id: str, *, id: str | None = None):
        super().__init__(
            [tool_from_aci(aci_function, linked_account_owner_id) for aci_function in aci_functions], id=id
        )
