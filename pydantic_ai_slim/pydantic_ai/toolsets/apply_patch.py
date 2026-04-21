"""Apply patch toolset for file patching with provider-native format support.

Uses provider-native format when supported (OpenAI ``apply_patch``),
falls back to a standard function tool otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..tools import ApplyPatchNativeDefinition, ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

__all__ = (
    'ApplyPatchOperation',
    'ApplyPatchOutput',
    'ApplyPatchToolset',
)


@dataclass(kw_only=True)
class ApplyPatchOperation:
    """A file patch operation from the model."""

    operation_type: Literal['create_file', 'update_file', 'delete_file']
    path: str
    diff: str | None = None
    content: str | None = None


@dataclass(kw_only=True)
class ApplyPatchOutput:
    """Result of a patch operation."""

    status: Literal['completed', 'failed']
    output: str | None = None


ApplyPatchExecuteFunc = Callable[[ApplyPatchOperation], Awaitable[ApplyPatchOutput]]
"""Callback type for executing patch operations."""

_APPLY_PATCH_ARGS_VALIDATOR = SchemaValidator(
    core_schema.typed_dict_schema(
        {
            'operation_type': core_schema.typed_dict_field(
                core_schema.str_schema(pattern='^(create_file|update_file|delete_file)$')
            ),
            'path': core_schema.typed_dict_field(core_schema.str_schema()),
            'diff': core_schema.typed_dict_field(
                core_schema.with_default_schema(core_schema.nullable_schema(core_schema.str_schema()), default=None),
                required=False,
            ),
            'content': core_schema.typed_dict_field(
                core_schema.with_default_schema(core_schema.nullable_schema(core_schema.str_schema()), default=None),
                required=False,
            ),
        }
    )
)

_APPLY_PATCH_TOOL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'operation_type': {
            'type': 'string',
            'enum': ['create_file', 'update_file', 'delete_file'],
            'description': 'The type of file operation.',
        },
        'path': {'type': 'string', 'description': 'The file path to operate on.'},
        'diff': {'type': 'string', 'description': 'V4A format diff for update_file or create_file.'},
        'content': {'type': 'string', 'description': 'Full file content for create_file.'},
    },
    'required': ['operation_type', 'path'],
}


@dataclass
class ApplyPatchToolset(AbstractToolset[AgentDepsT]):
    """Toolset for file patching with provider-native format.

    Uses provider-native format when supported (OpenAI ``apply_patch``),
    falls back to a standard function tool otherwise.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.toolsets import ApplyPatchToolset

        async def my_patcher(op):
            # Your patching logic here
            return ApplyPatchOutput(status='completed')

        agent = Agent('openai:gpt-5.4', toolsets=[ApplyPatchToolset(execute=my_patcher)])
    """

    execute: ApplyPatchExecuteFunc
    """The callback that executes patch operations."""

    _: Any = field(init=False, repr=False, default=None)  # KW_ONLY sentinel

    tool_name: str = 'apply_patch'
    """The name of the tool exposed to the model."""

    description: str = 'Apply a patch to create, update, or delete a file.'
    """Description of the tool."""

    max_retries: int = 1
    """Maximum number of retries for failed tool calls."""

    _id: str | None = field(default=None, repr=False)

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tool_def = ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters_json_schema=_APPLY_PATCH_TOOL_SCHEMA,
            native_definition=ApplyPatchNativeDefinition(),
        )
        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=_APPLY_PATCH_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> str:
        operation = ApplyPatchOperation(
            operation_type=tool_args['operation_type'],
            path=tool_args['path'],
            diff=tool_args.get('diff'),
            content=tool_args.get('content'),
        )
        result = await self.execute(operation)
        return json.dumps({'status': result.status, 'output': result.output})
