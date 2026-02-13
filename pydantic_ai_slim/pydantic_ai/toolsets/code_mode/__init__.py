"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any, cast

from pydantic import TypeAdapter
from typing_extensions import Self, TypedDict

from pydantic_ai.messages import tool_return_ta
from pydantic_ai.runtime import RuntimeName, get_runtime
from pydantic_ai.runtime.abstract import (
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)

from ... import exceptions
from ..._python_signature import FunctionSignature, collect_unique_referenced_types, dedup_referenced_types
from ..._run_context import AgentDepsT, RunContext
from ..._tool_manager import ToolManager
from ...exceptions import ApprovalRequired, CallDeferred, ModelRetry
from ...messages import ToolCallPart
from ...tools import ToolDefinition
from ..abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool
from ..wrapper import WrapperToolset
from ._sanitization import sanitize_tool_name

__all__ = (
    'CodeModeToolset',
    'FunctionSignature',
    'build_code_mode_prompt',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code_with_tools'

# TODO: The first line of the prompt should be customizable by the user using Prompt Templates #3656
_CODE_MODE_PROMPT = """
Use `run_code_with_tools` to write Python code that calls the available functions. You can make a single call or combine multiple steps in a script — use your judgment based on the task.

Execution model:
- Each `run_code_with_tools` call runs in an isolated environment — variables don't persist between calls
- Functions are async — call with `await`, e.g. `result = await get_items()`
- To run independent calls concurrently, fire them first, then await:
  ```python
  future_a = get_items()    # starts immediately
  future_b = get_users()    # starts immediately
  items = await future_a    # wait for results
  users = await future_b
  ```
- The last expression evaluated is the return value
- Return raw data when it answers the question directly; extract or transform when needed
"""


def build_code_mode_prompt(signatures: list[FunctionSignature], runtime_instructions: str | None) -> str:
    """Build the default code mode prompt with the given tool signatures.

    This is the default prompt builder used by CodeModeToolset. Users can provide
    their own prompt_builder callback to customize the prompt entirely.

    Args:
        signatures: List of Python function signatures for available tools.
        runtime_instructions: Runtime-specific text to include in the prompt (from
            `CodeRuntime.instructions()`). Inserted verbatim if non-empty.

    Returns:
        The complete prompt describing code mode capabilities and available functions.
    """
    parts = [_CODE_MODE_PROMPT]

    if runtime_instructions:
        parts.append(runtime_instructions)

    parts.append('```python')

    referenced_types = collect_unique_referenced_types(signatures)

    if referenced_types:
        parts.append('# Available types:')
        parts.append('\n\n'.join(t.render() for t in referenced_types))

    parts.append('# Available functions:')
    parts.extend(str(sig) for sig in signatures)

    parts.append('```')

    return '\n\n'.join(parts)


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]
    cached_signatures: list[FunctionSignature]
    name_map: dict[str, str]
    original_name_tools: dict[str, ToolsetTool[AgentDepsT]]


@dataclass(init=False)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context.

    Args:
        wrapped: The underlying toolset to wrap.
        prompt_builder: Optional callback to build a custom prompt. If not provided,
            uses `build_code_mode_prompt`. The callback receives `signatures` as a
            keyword argument containing the list of Python function signatures.
        max_retries: Maximum number of retries for code execution errors (type/syntax/runtime).
            Defaults to 3. Increase for complex code generation tasks or less capable models.
    """

    _: KW_ONLY

    runtime: CodeRuntime
    prompt_builder: Callable[[list[FunctionSignature], str | None], str] = build_code_mode_prompt
    max_retries: int = 3

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        runtime: CodeRuntime | RuntimeName = 'monty',
        prompt_builder: Callable[[list[FunctionSignature], str | None], str] = build_code_mode_prompt,
        max_retries: int = 3,
    ) -> None:
        if isinstance(runtime, str):
            runtime = get_runtime(runtime)
        self.runtime = runtime
        self.prompt_builder = prompt_builder
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
        wrapped_tools = await super().get_tools(ctx)

        # Build sanitized name map: {sanitized_name: original_name}
        # Code mode presents tools as Python function signatures to the LLM, which writes
        # Python code calling them. Tool names from MCP etc. may not be valid Python
        # identifiers (e.g. 'search-records', 'get.data'), so we sanitize them here.
        # We don't use RenamedToolset because the name map is computed dynamically at
        # get_tools() time and the renaming is internal (never exposed to the agent
        # framework — all tools are collapsed into a single 'run_code_with_tools' tool).
        name_map: dict[str, str] = {}  # {sanitized: original}
        for original_name in wrapped_tools:
            sanitized = sanitize_tool_name(original_name)
            base = sanitized
            counter = 2
            while sanitized in name_map:
                sanitized = f'{base}_{counter}'
                counter += 1
            name_map[sanitized] = original_name

        sanitized_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        signatures: list[FunctionSignature] = []

        for sanitized_name, original_name in name_map.items():
            tool = wrapped_tools[original_name]
            sanitized_tools[sanitized_name] = tool
            signatures.append(tool.python_signature(name_override=sanitized_name))

        dedup_referenced_types(signatures)

        # TODO: (Aditya), Check if tools need approved? Seq / Approval / Warn the users

        # Pre-compute reverse mapping (original name → tool)
        original_name_tools: dict[str, ToolsetTool[AgentDepsT]] = {name_map[s]: t for s, t in sanitized_tools.items()}

        description = self.prompt_builder(signatures, self.runtime.instructions)
        return {
            _CODE_MODE_TOOL_NAME: _CodeModeTool(
                toolset=self,
                original_tools=sanitized_tools,
                cached_signatures=signatures,
                name_map=name_map,
                original_name_tools=original_name_tools,
                tool_def=ToolDefinition(
                    name=_CODE_MODE_TOOL_NAME,
                    parameters_json_schema=_CODE_ADAPTER.json_schema(),
                    description=description,
                ),
                max_retries=self.max_retries,
                args_validator=cast(SchemaValidatorProt, _CODE_ADAPTER.validator),
            )
        }

    @staticmethod
    def _build_tool_kwargs(call: FunctionCall) -> dict[str, Any]:
        """Build tool kwargs from a FunctionCall.

        All tool signatures use keyword-only parameters (via `*`), so positional
        args should never appear. If they do, the generated code is malformed.
        """
        # TODO (DouweM): May not be needed since we force kwargs via `*` in the signature
        # ^ Confirmed not needed — now we just error if it happens anyway
        if call.args:
            raise ModelRetry(
                f'Positional arguments are not supported in code mode tool calls '
                f'(function {call.function_name!r} received {len(call.args)} positional arg(s)). '
                f'All parameters are keyword-only.'
            )
        return dict(call.kwargs)

    def _make_tool_callback(
        self,
        tool: _CodeModeTool[AgentDepsT],
        code_mode_tool_manager: ToolManager[AgentDepsT],
        name_map: dict[str, str],
    ) -> ToolCallback:
        """Create a callback for the runtime to invoke when code calls external functions.

        Args:
            tool: The code mode tool with original tools mapping.
            code_mode_tool_manager: ToolManager for executing nested tool calls.
            name_map: Mapping from sanitized names to original tool names.
        """

        async def callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_name = name_map.get(sanitized_name, sanitized_name)

            tool_kwargs = self._build_tool_kwargs(call)
            tool_call_part = ToolCallPart(tool_name=original_name, args=tool_kwargs)

            # Route through full ToolManager flow:
            # handle_call → _call_function_tool (tracing + usage) → _call_tool (validate + enrich + call)
            # wrap_validation_errors=False: let raw errors propagate to the runtime.
            # Tool exceptions bubble up to user code (same behavior as regular tools).

            try:
                result = await code_mode_tool_manager.handle_call(
                    tool_call_part,
                    wrap_validation_errors=False,
                )
                return tool_return_ta.dump_python(result, mode='json')
            except (CallDeferred, ApprovalRequired):
                raise

        return callback

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        code = tool_args['code']
        if name != _CODE_MODE_TOOL_NAME:
            raise exceptions.UserError(
                f'CodeModeToolset.call_tool expected tool name {_CODE_MODE_TOOL_NAME!r}, got {name!r}'
            )
        if not isinstance(tool, _CodeModeTool):
            raise exceptions.UserError(f'CodeModeToolset.call_tool expected _CodeModeTool, got {type(tool).__name__}')
        if not isinstance(code, str):
            raise exceptions.UserError(
                f'CodeModeToolset.call_tool expected code to be a string, got {type(code).__name__}'
            )

        code_mode_tool_manager = ToolManager(
            toolset=self.wrapped,
            ctx=ctx,
            tools=tool.original_name_tools,
        )

        callback = self._make_tool_callback(tool, code_mode_tool_manager, tool.name_map)
        functions = list(tool.original_tools.keys())

        try:
            return await self.runtime.run(code, functions, callback, signatures=tool.cached_signatures)
        except CodeTypingError as e:
            raise ModelRetry(f'Type error in generated code:\n{e.message}') from e
        except CodeSyntaxError as e:
            raise ModelRetry(f'Syntax error in generated code:\n{e.message}') from e
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}') from e
        except CodeInterruptedError:
            raise exceptions.UserError(
                'Tool approval and deferral are not yet supported in code mode. '
                'Ensure wrapped tools do not use approval or deferral when used with CodeModeToolset.'
            )
