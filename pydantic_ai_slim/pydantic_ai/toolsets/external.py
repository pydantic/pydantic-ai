from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from pydantic_core import SchemaValidator, core_schema

from .. import _utils
from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


class ExternalToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds tools whose results will be produced outside of the Pydantic AI agent run in which they were called.

    See [toolset docs](../toolsets.md#external-toolset) for more information.
    """

    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = [
            replace(tool_def, parameters_json_schema=_validate_parameters_json_schema(tool_def))
            for tool_def in tool_defs
        ]
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='external'),
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        raise NotImplementedError('External tools cannot be called directly')


def _validate_parameters_json_schema(tool_def: ToolDefinition) -> dict[str, Any]:
    try:
        schema = _utils.check_object_json_schema(tool_def.parameters_json_schema)
    except UserError as exc:
        raise UserError(f'Invalid parameters_json_schema for external tool {tool_def.name!r}: {exc}') from exc

    properties_value = schema.get('properties', {})
    if not isinstance(properties_value, dict):
        raise UserError(
            f'Invalid parameters_json_schema for external tool {tool_def.name!r}: properties must be an object'
        )
    properties = cast(dict[str, Any], properties_value)

    required_value = schema.get('required', [])
    if not isinstance(required_value, list):
        raise UserError(
            f'Invalid parameters_json_schema for external tool {tool_def.name!r}: required must be a list of strings'
        )

    required: list[str] = []
    for key in cast(list[Any], required_value):
        if not isinstance(key, str):
            raise UserError(
                f'Invalid parameters_json_schema for external tool {tool_def.name!r}: '
                'required must be a list of strings'
            )
        required.append(key)

    missing_required = sorted(set(required) - set(properties))
    if missing_required:
        missing = ', '.join(repr(key) for key in missing_required)
        raise UserError(
            f'Invalid parameters_json_schema for external tool {tool_def.name!r}: '
            f'required keys are missing from properties: {missing}'
        )

    return schema
