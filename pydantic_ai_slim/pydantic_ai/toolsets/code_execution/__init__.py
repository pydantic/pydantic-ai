"""Code execution toolset that optionally wraps tools as Python functions callable from generated code."""

from __future__ import annotations

import copy
import keyword
import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, Literal, TypeAlias, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypedDict, assert_never

from pydantic_ai.messages import tool_return_ta

from ... import exceptions
from ..._python_signature import (
    FunctionSignature,
    TypeSignature,
    collect_unique_referenced_types,
    dedup_referenced_types,
)
from ..._run_context import AgentDepsT, RunContext
from ..._tool_manager import ToolManager, _parallel_execution_mode_ctx_var  # pyright: ignore[reportPrivateUsage]
from ...exceptions import ApprovalRequired, CallDeferred, ModelRetry
from ...messages import ToolCallPart
from ...tools import ToolDefinition
from ..abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool
from ._abstract import (
    CodeExecutionError,
    CodeExecutionTimeout,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)
from ._transport import DriverBasedRuntime, DriverTransport

try:
    from .monty import MontyRuntime
except ImportError:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .monty import MontyRuntime

from .docker import DockerRuntime, DockerSecuritySettings

__all__ = (
    'CodeExecutionToolset',
    'CodeRuntime',
    'CodeRuntimeError',
    'CodeExecutionError',
    'CodeSyntaxError',
    'CodeTypingError',
    'CodeExecutionTimeout',
    'DescriptionFunc',
    'DockerRuntime',
    'DockerSecuritySettings',
    'DriverBasedRuntime',
    'DriverTransport',
    'FunctionCall',
    'FunctionSignature',
    'MontyRuntime',
    'ToolCallback',
    'TypeSignature',
    'build_default_description',
)


RuntimeName = Literal['monty', 'docker']


def get_runtime(name: RuntimeName) -> CodeRuntime:
    if name == 'monty':
        from .monty import MontyRuntime

        return MontyRuntime()
    elif name == 'docker':
        return DockerRuntime()
    else:
        assert_never(name)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_VALIDATOR = _CODE_ADAPTER.validator
_CODE_JSON_SCHEMA = _CODE_ADAPTER.json_schema()
_TOOL_NAME = 'run_code'

_BASE_PROMPT = """
Use this tool to run Python code.

Execution model:
- Each call runs in a completely isolated environment — variables don't persist between calls
- If a previous call failed, you must rewrite the entire program from scratch
- All functions are async. You can create new functions for convenience.
- This tool is for running code — don't use it just to format or print your final analysis.
"""

_TOOLS_PROMPT = """
Use this tool to run Python code that can call other tools as functions.

You can use it to:
- filter tool return data to save context,
- perform complex operations that would take many model calls using standard tool calling, or
- pass the result of one tool to another without it entering your context window.

Execution model:
- Each call to this tool runs in a completely isolated environment — no variables, results, or state persist between calls. You MUST do all your work (fetching data, processing it, and producing the final result) in a single code block. Do not split work across multiple calls expecting to use earlier results.
- If a previous call failed, you must rewrite the entire program from scratch — you cannot reference variables or results from a failed attempt.
- You can create new functions for convenience.
- This tool is for calling and chaining tools programmatically — don't use it just to format or print your final analysis. Write your report as regular text in your response.
"""


DescriptionFunc: TypeAlias = Callable[[list[FunctionSignature], list[TypeSignature], str | None], str]
"""Callback type for building the code execution tool description.

Receives the function signatures, their referenced types, and optional
runtime-specific instructions. Returns the complete tool description string.
"""


def build_default_description(
    signatures: list[FunctionSignature],
    referenced_types: list[TypeSignature],
    runtime_instructions: str | None,
    *,
    description: str | None = None,
) -> str:
    """Build the default code execution tool description with the given tool signatures.

    This is the default description builder used by CodeExecutionToolset. Users can provide
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
    if description is None:
        description = _TOOLS_PROMPT if signatures else _BASE_PROMPT

    parts = [description]

    if runtime_instructions:
        parts.append(runtime_instructions)

    if signatures:
        parts.append('```python')

        if referenced_types:
            parts.append('# Available types:')
            parts.extend(str(t) for t in referenced_types)

        parts.append('# Available functions:')
        parts.extend(str(sig) for sig in signatures)

        parts.append('```')

    return '\n\n'.join(parts)


@dataclass(kw_only=True)
class _CodeExecutionTool(ToolsetTool[AgentDepsT]):
    signatures: list[FunctionSignature]
    referenced_types: list[TypeSignature]
    name_map: dict[str, str]
    tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass(init=False)
class CodeExecutionToolset(AbstractToolset[AgentDepsT]):
    """A toolset that executes Python code, optionally with access to wrapped tools as callable functions.

    When a ``toolset`` is provided, its tools are exposed as callable Python functions in the code
    execution context. When no ``toolset`` is provided, it acts as a pure code execution environment.

    Args:
        runtime: The code execution runtime. Can be a runtime instance or a string shorthand
            (``'monty'`` or ``'docker'``). Defaults to ``'monty'``.
        toolset: Optional underlying toolset to wrap. When provided, its tools are exposed as
            callable Python functions in the code execution context.
        description: Custom tool description. Can be a string (used as the preamble text
            with the default structure) or a `DescriptionFunc` callback for full control.
            Defaults to `build_default_description`.
        max_retries: Maximum number of retries for code execution errors (type/syntax/runtime).
            Defaults to 3. Increase for complex code generation tasks or less capable models.
    """

    runtime: CodeRuntime

    _: KW_ONLY

    toolset: AbstractToolset[AgentDepsT] | None
    description: str | DescriptionFunc
    max_retries: int = 3

    def __init__(
        self,
        runtime: CodeRuntime | RuntimeName = 'monty',
        *,
        toolset: AbstractToolset[AgentDepsT] | None = None,
        description: str | DescriptionFunc = build_default_description,
        max_retries: int = 3,
    ) -> None:
        if isinstance(runtime, str):
            runtime = get_runtime(runtime)
        self.runtime = runtime
        self.toolset = toolset
        self.description = description
        self.max_retries = max_retries

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:  # pragma: no cover
        if self.toolset is not None:
            return f'CodeExecutionToolset({self.toolset.label})'
        return 'CodeExecutionToolset'

    async def __aenter__(self) -> Self:
        await self.runtime.__aenter__()
        try:
            if self.toolset is not None:
                await self.toolset.__aenter__()
        except BaseException:
            await self.runtime.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            if self.toolset is not None:
                return await self.toolset.__aexit__(*args)
            return None
        finally:
            await self.runtime.__aexit__(*args)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:  # pragma: no cover
        if self.toolset is not None:
            self.toolset.apply(visitor)
        else:
            visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        if self.toolset is not None:
            return replace(self, toolset=self.toolset.visit_and_replace(visitor))
        return visitor(self)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        if self.toolset is not None:
            wrapped_tools = await self.toolset.get_tools(ctx)

            deferred_tools = [name for name, tool in wrapped_tools.items() if tool.tool_def.defer]
            if deferred_tools:
                raise exceptions.UserError(
                    'Tool approval and deferral are not yet supported in code execution mode. '
                    'Ensure wrapped tools do not use approval or deferral when used with CodeExecutionToolset.'
                )

        if wrapped_tools:
            # Build sanitized name map: {sanitized_name: original_name}
            # Code execution presents tools as Python function signatures to the LLM, which writes
            # Python code calling them. Tool names from MCP etc. may not be valid Python
            # identifiers (e.g. 'search-records', 'get.data'), so we sanitize them here.
            name_map: dict[str, str] = {}  # {sanitized: original}
            for original_name in wrapped_tools:
                sanitized = _sanitize_tool_name(original_name)
                base = sanitized
                counter = 2
                while sanitized in name_map:
                    sanitized = f'{base}_{counter}'
                    counter += 1
                name_map[sanitized] = original_name

            global_sequential = _parallel_execution_mode_ctx_var.get() in ('sequential', 'parallel_ordered_events')

            signatures: list[FunctionSignature] = []
            for sanitized_name, original_name in name_map.items():
                sig = copy.deepcopy(wrapped_tools[original_name].python_signature)
                sig.name = sanitized_name
                sig.is_async = not (global_sequential or wrapped_tools[original_name].tool_def.sequential)
                signatures.append(sig)

            dedup_referenced_types(signatures)
            referenced_types = collect_unique_referenced_types(signatures)
        else:
            name_map = {}
            signatures = []
            referenced_types = []

        if isinstance(self.description, str):
            tool_description = build_default_description(
                signatures, referenced_types, self.runtime.instructions, description=self.description
            )
        else:
            tool_description = self.description(signatures, referenced_types, self.runtime.instructions)

        return {
            _TOOL_NAME: _CodeExecutionTool(
                toolset=self,
                signatures=signatures,
                referenced_types=referenced_types,
                name_map=name_map,
                tools=wrapped_tools,
                tool_def=ToolDefinition(
                    name=_TOOL_NAME,
                    parameters_json_schema=_CODE_JSON_SCHEMA,
                    description=tool_description,
                    metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                ),
                max_retries=self.max_retries,
                args_validator=cast(SchemaValidatorProt, _CODE_VALIDATOR),
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert name == _TOOL_NAME
        assert isinstance(tool, _CodeExecutionTool)

        code = tool_args.get('code')
        assert isinstance(code, str)

        tool_manager: ToolManager[AgentDepsT] | None = None
        if self.toolset is not None:
            tool_manager = ToolManager(
                toolset=self.toolset,
                ctx=ctx,
                tools=tool.tools,
            )

        async def call_tool_callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_name = tool.name_map.get(sanitized_name, sanitized_name)

            try:
                if tool_manager is None:  # pragma: no cover
                    raise ModelRetry('No tools available')

                if call.args:
                    raise ModelRetry(
                        'Positional arguments are not supported in code mode tool calls. All parameters are keyword-only.'
                    )

                tool_call = ToolCallPart(tool_name=original_name, args=call.kwargs, tool_call_id=call.call_id)
                result = await tool_manager.handle_call(tool_call, wrap_validation_errors=False)

                return tool_return_ta.dump_python(result)
            except (CallDeferred, ApprovalRequired):
                raise exceptions.UserError(
                    'Tool approval and deferral are not yet supported in code execution mode. '
                    'Ensure wrapped tools do not use approval or deferral when used with CodeExecutionToolset.'
                )
            except (ModelRetry, ValidationError) as e:
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
