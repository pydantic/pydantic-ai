"""Text editor toolset for local file editing with provider-native format support.

Uses provider-native format when supported (Anthropic ``text_editor_20250728``),
falls back to a standard function tool otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_core import SchemaValidator, core_schema
from typing_extensions import NotRequired, TypedDict

from .._run_context import AgentDepsT, RunContext
from ..tools import TextEditorNativeDefinition, ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

__all__ = (
    'TextEditorCommand',
    'TextEditorOutput',
    'TextEditorToolset',
)


class ViewCommand(TypedDict):
    """View a file or a range of lines."""

    command: Literal['view']
    path: str
    view_range: NotRequired[list[int]]


class StrReplaceCommand(TypedDict):
    """Replace a string in a file."""

    command: Literal['str_replace']
    path: str
    old_str: str
    new_str: str


class CreateCommand(TypedDict):
    """Create a new file with content."""

    command: Literal['create']
    path: str
    file_text: str


class InsertCommand(TypedDict):
    """Insert text at a specific line."""

    command: Literal['insert']
    path: str
    insert_line: int
    insert_text: str


TextEditorCommand = ViewCommand | StrReplaceCommand | CreateCommand | InsertCommand
"""Discriminated union of text editor commands, keyed on ``command``."""


@dataclass(kw_only=True)
class TextEditorOutput:
    """Result of a text editor operation."""

    output: str
    success: bool = True


TextEditorExecuteFunc = Callable[[TextEditorCommand], Awaitable[TextEditorOutput]]
"""Callback type for executing text editor commands."""

_TEXT_EDITOR_ARGS_VALIDATOR = SchemaValidator(
    core_schema.typed_dict_schema(
        {
            'command': core_schema.typed_dict_field(
                core_schema.str_schema(pattern='^(view|str_replace|create|insert)$')
            ),
            'path': core_schema.typed_dict_field(core_schema.str_schema()),
            'view_range': core_schema.typed_dict_field(
                core_schema.with_default_schema(
                    core_schema.nullable_schema(core_schema.list_schema(core_schema.int_schema())),
                    default=None,
                ),
                required=False,
            ),
            'old_str': core_schema.typed_dict_field(core_schema.str_schema(), required=False),
            'new_str': core_schema.typed_dict_field(core_schema.str_schema(), required=False),
            'file_text': core_schema.typed_dict_field(core_schema.str_schema(), required=False),
            'insert_line': core_schema.typed_dict_field(core_schema.int_schema(), required=False),
            'insert_text': core_schema.typed_dict_field(core_schema.str_schema(), required=False),
        }
    )
)

_TEXT_EDITOR_TOOL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'command': {
            'type': 'string',
            'enum': ['view', 'str_replace', 'create', 'insert'],
            'description': 'The command to execute.',
        },
        'path': {'type': 'string', 'description': 'The file path to operate on.'},
        'view_range': {
            'type': 'array',
            'items': {'type': 'integer'},
            'description': 'Range of lines to view [start, end], 1-indexed. Only for view command.',
        },
        'old_str': {'type': 'string', 'description': 'String to replace. Only for str_replace command.'},
        'new_str': {'type': 'string', 'description': 'Replacement string. Only for str_replace command.'},
        'file_text': {'type': 'string', 'description': 'File content. Only for create command.'},
        'insert_line': {'type': 'integer', 'description': 'Line number to insert at. Only for insert command.'},
        'insert_text': {'type': 'string', 'description': 'Text to insert. Only for insert command.'},
    },
    'required': ['command', 'path'],
}


@dataclass
class TextEditorToolset(AbstractToolset[AgentDepsT]):
    """Toolset for local text editor operations with provider-native format.

    Uses provider-native format when supported (Anthropic ``text_editor_20250728``),
    falls back to a standard function tool otherwise.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.toolsets import TextEditorToolset

        async def my_editor(cmd):
            # Your editor logic here
            return TextEditorOutput(output='OK')

        agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[TextEditorToolset(execute=my_editor)])
    """

    execute: TextEditorExecuteFunc
    """The callback that executes text editor commands."""

    _: Any = field(init=False, repr=False, default=None)  # KW_ONLY sentinel

    tool_name: str = 'str_replace_based_edit_tool'
    """The name of the tool exposed to the model."""

    description: str = 'A text editor tool for viewing and editing files.'
    """Description of the tool."""

    max_characters: int | None = None
    """Maximum characters for the text editor. Passed through to the native tool definition."""

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
            parameters_json_schema=_TEXT_EDITOR_TOOL_SCHEMA,
            native_definition=TextEditorNativeDefinition(max_characters=self.max_characters),
        )
        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=_TEXT_EDITOR_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> str:
        command: TextEditorCommand = tool_args  # type: ignore[assignment]
        result = await self.execute(command)
        return json.dumps({'output': result.output, 'success': result.success})
