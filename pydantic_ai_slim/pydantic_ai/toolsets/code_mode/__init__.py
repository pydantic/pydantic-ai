"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

import copy
import keyword
import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any, TypeAlias, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypedDict

from pydantic_ai.messages import tool_return_ta
from pydantic_ai.runtime import RuntimeName, get_runtime
from pydantic_ai.runtime.abstract import (
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)

from ... import exceptions
from ..._python_signature import (
    FunctionSignature,
    TypeSignature,
    collect_unique_referenced_types,
    dedup_referenced_types,
)
from ..._run_context import AgentDepsT, RunContext
from ..._tool_manager import ToolManager
from ...exceptions import ApprovalRequired, CallDeferred, ModelRetry
from ...messages import ToolCallPart
from ...tools import ToolDefinition
from ..abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool
from ..wrapper import WrapperToolset

__all__ = (
    'CodeModeToolset',
    'DescriptionFunc',
    'FunctionSignature',
    'TypeSignature',
    'build_default_code_mode_description',
    'ToolCallback',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_VALIDATOR = _CODE_ADAPTER.validator
_CODE_JSON_SCHEMA = _CODE_ADAPTER.json_schema()
_CODE_MODE_TOOL_NAME = 'run_code_with_tools'

# TODO: The first line of the prompt should be customizable by the user using Prompt Templates #3656
# TODO (DouweM): Explain the use case of codemode to the model, e.g. filtering data to save context. See if we can find the prompt that Anthropic PTC uses.
_CODE_MODE_PROMPT = """
Use this tool to run Python code that can call other tools as functions, also known as "code mode" or "programmatic tool calling".

You can use it to:
- filter tool return data to save context,
- perform complex operations that would take many model calls using standard tool calling, or
- pass the result of one tool to another without it entering your context window.

Execution model:
- Each call to this tool runs in a completely isolated environment — variables don't persist between calls
- If a previous call failed, you must rewrite the entire program from scratch — you cannot reference variables or results from a failed attempt
- All functions are async. You can create new functions for convenience.
- This tool is for calling and chaining tools programmatically — don't use it just to format or print your final analysis. Write your report as regular text in your response.
"""
# TODO (Douwe): dynamic based on whether a toolset is mounted (codemode) or not (regular code execution)


DescriptionFunc: TypeAlias = Callable[[list[FunctionSignature], list[TypeSignature], str | None], str]
"""Callback type for building the code mode tool description.

Receives the function signatures, their referenced types, and optional
runtime-specific instructions. Returns the complete tool description string.
"""


def build_default_code_mode_description(
    signatures: list[FunctionSignature],
    referenced_types: list[TypeSignature],
    runtime_instructions: str | None,
    *,
    description: str = _CODE_MODE_PROMPT,
) -> str:
    """Build the default code mode tool description with the given tool signatures.

    This is the default description builder used by CodeModeToolset. Users can provide
    their own callback via the ``description`` parameter, or pass a string to customize
    just the preamble text while keeping the default structure.

    Args:
        signatures: List of Python function signatures for available tools.
        referenced_types: Unique type definitions referenced by the signatures.
        runtime_instructions: Runtime-specific text to include in the description (from
            `CodeRuntime.instructions`). Inserted verbatim if non-empty.
        description: Custom preamble text to use instead of the built-in default.

    Returns:
        The complete description string with preamble, available types, and function signatures.
    """
    parts = [description]

    if runtime_instructions:
        parts.append(runtime_instructions)

    parts.append('```python')

    if referenced_types:
        parts.append('# Available types:')
        parts.append('\n\n'.join(str(t) for t in referenced_types))

    parts.append('# Available functions:')
    parts.extend(str(sig) for sig in signatures)

    parts.append('```')

    return '\n\n'.join(parts)


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    signatures: list[FunctionSignature]
    referenced_types: list[TypeSignature]
    name_map: dict[str, str]
    tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass(init=False)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context.

    Args:
        wrapped: The underlying toolset to wrap.
        description: Custom tool description. Can be a string (used as the preamble text
            with the default structure) or a `DescriptionFunc` callback for full control.
            Defaults to `build_default_code_mode_description`.
        max_retries: Maximum number of retries for code execution errors (type/syntax/runtime).
            Defaults to 3. Increase for complex code generation tasks or less capable models.
    """

    _: KW_ONLY

    runtime: CodeRuntime
    description: str | DescriptionFunc
    max_retries: int = 3

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        runtime: CodeRuntime | RuntimeName = 'monty',
        description: str | DescriptionFunc = build_default_code_mode_description,
        max_retries: int = 3,
    ) -> None:
        if isinstance(runtime, str):
            runtime = get_runtime(runtime)
        self.runtime = runtime

        self.description = description
        self.max_retries = max_retries
        super().__init__(wrapped)

    async def __aenter__(self) -> Self:
        await self.runtime.__aenter__()
        try:
            return await super().__aenter__()
        except BaseException:
            await self.runtime.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            return await super().__aexit__(*args)
        finally:
            await self.runtime.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)

        deferred_tools = [name for name, tool in tools.items() if tool.tool_def.defer]
        if deferred_tools:
            raise exceptions.UserError(
                'Tool approval and deferral are not yet supported in code mode. '
                'Ensure wrapped tools do not use approval or deferral when used with CodeModeToolset.'
            )

        # Build sanitized name map: {sanitized_name: original_name}
        # Code mode presents tools as Python function signatures to the LLM, which writes
        # Python code calling them. Tool names from MCP etc. may not be valid Python
        # identifiers (e.g. 'search-records', 'get.data'), so we sanitize them here.
        # We don't use RenamedToolset because the name map is computed dynamically at
        # get_tools() time and the renaming is internal (never exposed to the agent
        # framework — all tools are collapsed into a single 'run_code_with_tools' tool).
        name_map: dict[str, str] = {}  # {sanitized: original}
        for original_name in tools:
            sanitized = _sanitize_tool_name(original_name)
            base = sanitized
            counter = 2
            while sanitized in name_map:
                sanitized = f'{base}_{counter}'
                counter += 1
            name_map[sanitized] = original_name

        signatures: list[FunctionSignature] = []
        for sanitized_name, original_name in name_map.items():
            sig = copy.deepcopy(tools[original_name].python_signature)
            sig.name = sanitized_name
            signatures.append(sig)

        dedup_referenced_types(signatures)
        referenced_types = collect_unique_referenced_types(signatures)

        if isinstance(self.description, str):
            tool_description = build_default_code_mode_description(
                signatures, referenced_types, self.runtime.instructions, description=self.description
            )
        else:
            tool_description = self.description(signatures, referenced_types, self.runtime.instructions)

        return {
            _CODE_MODE_TOOL_NAME: _CodeModeTool(
                toolset=self,
                signatures=signatures,
                referenced_types=referenced_types,
                name_map=name_map,
                tools=tools,
                tool_def=ToolDefinition(
                    name=_CODE_MODE_TOOL_NAME,
                    parameters_json_schema=_CODE_JSON_SCHEMA,
                    description=tool_description,
                ),
                max_retries=self.max_retries,
                args_validator=cast(SchemaValidatorProt, _CODE_VALIDATOR),
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert name == _CODE_MODE_TOOL_NAME
        assert isinstance(tool, _CodeModeTool)

        code = tool_args.get('code')
        assert isinstance(code, str)

        tool_manager = ToolManager(
            toolset=self.wrapped,
            ctx=ctx,
            tools=tool.tools,
        )

        async def call_tool_callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_name = tool.name_map.get(sanitized_name, sanitized_name)

            try:
                if call.args:
                    raise ModelRetry(
                        'Positional arguments are not supported in code mode tool calls. All parameters are keyword-only.'
                    )

                tool_call = ToolCallPart(tool_name=original_name, args=call.kwargs, tool_call_id=call.call_id)
                # Route through full ToolManager flow:
                # handle_call → _call_function_tool (tracing + usage) → _call_tool_callback (validate + enrich + call)
                # wrap_validation_errors=False: let raw errors propagate to the runtime.
                result = await tool_manager.handle_call(tool_call, wrap_validation_errors=False)

                return tool_return_ta.dump_python(result)
            except (CallDeferred, ApprovalRequired):
                raise exceptions.UserError(
                    'Tool approval and deferral are not yet supported in code mode. '
                    'Ensure wrapped tools do not use approval or deferral when used with CodeModeToolset.'
                )
            except (ModelRetry, ValidationError) as e:
                # Wrap retryable errors with context about which function call failed,
                # so the model knows which specific call to fix. Other exceptions
                # propagate directly to user code (same as regular tool execution).
                raise CodeRuntimeError(f'Call to {sanitized_name!r} failed: {e}') from e

        try:
            return await self.runtime.run(
                code,
                call_tool_callback,
                functions={sig.name: sig for sig in tool.signatures},
                referenced_types=tool.referenced_types,
            )
        except CodeTypingError as e:
            raise ModelRetry(f'Type error in generated code:\n{e.message}') from e
        except CodeSyntaxError as e:
            raise ModelRetry(f'Syntax error in generated code:\n{e.message}') from e
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}') from e


def _sanitize_tool_name(name: str) -> str:
    """Convert a tool name to a valid Python identifier.

    Args:
        name: The original tool name (may contain hyphens, dots, etc.)

    Returns:
        A valid Python identifier in snake_case.

    Examples:
        >>> _sanitize_tool_name('search-records')
        'search_records'
        >>> _sanitize_tool_name('get.user.data')
        'get_user_data'
        >>> _sanitize_tool_name('class')  # Python keyword
        'class_'
    """
    # Replace common separators with underscores
    sanitized = re.sub(r'[-.\s]+', '_', name)

    # Remove any remaining invalid characters (keep alphanumeric and underscore)
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'

    # Handle empty result
    if not sanitized:
        sanitized = 'tool'

    # Convert to snake_case if it's camelCase or PascalCase
    # Insert underscore before uppercase letters that follow lowercase
    sanitized = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', sanitized)
    sanitized = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', sanitized).lower()

    # Handle Python keywords by appending underscore
    if keyword.iskeyword(sanitized):
        sanitized = f'{sanitized}_'

    return sanitized
