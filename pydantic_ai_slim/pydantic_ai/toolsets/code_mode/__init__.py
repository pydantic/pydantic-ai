"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

from pydantic import TypeAdapter
from typing_extensions import TypedDict

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
from ..._python_signature import Signature
from ..._run_context import AgentDepsT, RunContext
from ..._tool_manager import ToolManager
from ...exceptions import ModelRetry
from ...messages import ToolCallPart
from ...tools import ToolDefinition
from ..abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool
from ..wrapper import WrapperToolset
from ._sanitization import sanitize_tool_name

__all__ = (
    'CodeModeToolset',
    'build_code_mode_prompt',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code_with_tools'

_TYPEDDICT_NAME_RE = re.compile(r'^class (\w+)\(TypedDict\):')

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


def _dedup_typeddicts(signatures: list[Signature]) -> None:
    """Deduplicate TypedDict definitions across multiple tool signatures in place.

    When multiple tools share the same TypedDict type (same name and definition),
    the definition is kept on the first signature and removed from subsequent ones.
    When different TypedDict types share the same name (conflict), the later one is
    renamed by prefixing the tool name to disambiguate.
    """
    # Collect all (name, definition) pairs, tracking which signature owns each
    seen: dict[str, str] = {}  # {name: definition}

    for sig in signatures:
        deduped: list[str] = []
        for td_str in sig.typeddicts:
            m = _TYPEDDICT_NAME_RE.match(td_str)
            if not m:
                deduped.append(td_str)
                continue

            td_name = m.group(1)
            if td_name not in seen:
                # First occurrence — keep it
                seen[td_name] = td_str
                deduped.append(td_str)
            elif seen[td_name] == td_str:
                # Same name, same definition — deduplicate (skip)
                pass
            else:
                # Same name, different definition — rename to avoid conflict
                new_name = f'{sig.name}_{td_name}'
                renamed_td = td_str.replace(f'class {td_name}(TypedDict):', f'class {new_name}(TypedDict):')
                # Update references using word boundaries to avoid corrupting names
                # that contain td_name as a substring (e.g. renaming "User" must not
                # affect "UserMeta").
                td_pattern = re.compile(r'\b' + re.escape(td_name) + r'\b')
                sig.params = [td_pattern.sub(new_name, p) for p in sig.params]
                sig.return_type = td_pattern.sub(new_name, sig.return_type)
                # Update references in other TypedDicts already in this signature's deduped list
                deduped = [td_pattern.sub(new_name, d) for d in deduped]
                seen[new_name] = renamed_td
                deduped.append(renamed_td)
        sig.typeddicts = deduped


def build_code_mode_prompt(signatures: list[Signature], runtime_instructions: str | None) -> str:
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

    parts.append('Available functions:')
    joined_signatures = '\n\n'.join(sig.with_typeddicts() for sig in signatures)
    parts.append(f'```python\n{joined_signatures}\n```')
    return '\n\n'.join(parts)


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


def _get_tool_signature(
    tool: ToolsetTool[Any],
    name_override: str | None = None,
) -> Signature:
    """Get a Signature object for a tool.

    Delegates to `tool.python_signature()`, which uses inspect-based signatures
    for function tools and JSON-schema-based signatures for external tools (MCP, etc.).

    Args:
        tool: The tool to generate a signature for.
        name_override: Optional name to use instead of the tool's original name.
            Used to show sanitized names (valid Python identifiers) to the LLM.

    Returns:
        A Signature object. Use str(sig) for type checking, sig.with_typeddicts('...') for LLM.
    """
    return tool.python_signature(name_override=name_override)


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
    prompt_builder: Callable[[list[Signature], str | None], str] = build_code_mode_prompt
    max_retries: int = 3

    # Note: _cached_signatures and _name_map are mutable instance state populated in get_tools()
    # and consumed in call_tool(). This class is not safe for concurrent use from multiple agent
    # runs on the same instance. The agent framework runs a single loop per toolset instance,
    # so this is fine in practice.
    _cached_signatures: list[Signature] | None = field(default=None, init=False, repr=False)
    _name_map: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False)

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        runtime: CodeRuntime | RuntimeName = 'monty',
        prompt_builder: Callable[[list[Signature], str | None], str] = build_code_mode_prompt,
        max_retries: int = 3,
    ) -> None:
        if isinstance(runtime, str):
            runtime = get_runtime(runtime)
        self.runtime = runtime
        self.prompt_builder = prompt_builder
        self.max_retries = max_retries
        super().__init__(wrapped)

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
        self._name_map = name_map

        sanitized_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        signatures: list[Signature] = []

        for sanitized_name, original_name in name_map.items():
            tool = wrapped_tools[original_name]
            sanitized_tools[sanitized_name] = tool
            signatures.append(_get_tool_signature(tool, name_override=sanitized_name))

        _dedup_typeddicts(signatures)

        self._cached_signatures = signatures

        description = self.prompt_builder(signatures, self.runtime.instructions)
        return {
            _CODE_MODE_TOOL_NAME: _CodeModeTool(
                toolset=self,
                original_tools=sanitized_tools,
                tool_def=ToolDefinition(
                    name=_CODE_MODE_TOOL_NAME,
                    parameters_json_schema=_CODE_ADAPTER.json_schema(),
                    description=description,
                ),
                max_retries=self.max_retries,
                args_validator=cast(SchemaValidatorProt, _CODE_ADAPTER.validator),
            )
        }

    def _build_tool_kwargs(
        self,
        call: FunctionCall,
        tool: _CodeModeTool[AgentDepsT],
        sanitized_name: str,
    ) -> dict[str, Any]:
        """Build tool kwargs from FunctionCall, handling positional args fallback."""
        tool_kwargs: dict[str, Any] = dict(call.kwargs)
        if call.args:
            # TODO (DouweM): May not be needed since we force kwargs via `*` in the signature
            # Positional args are mapped using JSON schema property order, which may not match
            # the tool's actual parameter order. The prompt instructs models to use keyword
            # arguments only, but we handle positional args as a fallback for non-compliant models.
            original_tool = tool.original_tools[sanitized_name]
            param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
            for i, arg in enumerate(call.args):
                if i < len(param_names):
                    tool_kwargs[param_names[i]] = arg
        return tool_kwargs

    def _make_tool_callback(
        self,
        tool: _CodeModeTool[AgentDepsT],
        code_mode_tool_manager: ToolManager[AgentDepsT],
        sanitized_to_original: dict[str, str],
    ) -> ToolCallback:
        """Create a callback for the runtime to invoke when code calls external functions.

        Args:
            tool: The code mode tool with original tools mapping.
            code_mode_tool_manager: ToolManager for executing nested tool calls.
            sanitized_to_original: Mapping from sanitized names to original tool names.
        """

        async def callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_name = sanitized_to_original.get(sanitized_name, sanitized_name)

            tool_kwargs = self._build_tool_kwargs(call, tool, sanitized_name)
            tool_call_part = ToolCallPart(tool_name=original_name, args=tool_kwargs)

            # Route through full ToolManager flow:
            # handle_call → _call_function_tool (tracing + usage) → _call_tool (validate + enrich + call)
            # wrap_validation_errors=False: let raw errors propagate to the runtime.
            # Tool exceptions bubble up to user code (same behavior as regular tools).
            return await code_mode_tool_manager.handle_call(
                tool_call_part,
                wrap_validation_errors=False,
            )

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
        if self._cached_signatures is None:
            raise exceptions.UserError('CodeModeToolset.call_tool called before get_tools — signatures not initialized')

        original_name_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        sanitized_to_original: dict[str, str] = {}
        for sanitized, t in tool.original_tools.items():
            orig = self._name_map.get(sanitized, sanitized)
            original_name_tools[orig] = t
            sanitized_to_original[sanitized] = orig

        code_mode_tool_manager = ToolManager(
            toolset=self.wrapped,
            ctx=ctx,
            tools=original_name_tools,
        )

        callback = self._make_tool_callback(tool, code_mode_tool_manager, sanitized_to_original)
        functions = list(tool.original_tools.keys())

        try:
            return await self.runtime.run(code, functions, callback, signatures=self._cached_signatures)
        except CodeTypingError as e:
            raise ModelRetry(f'Type error in generated code:\n{e.message}') from e
        except CodeSyntaxError as e:
            raise ModelRetry(f'Syntax error in generated code:\n{e.message}') from e
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}') from e
        except CodeInterruptedError:
            raise exceptions.UserError(
                'Tool approval and deferral are not supported in code mode. '
                'Ensure wrapped tools do not use approval or deferral when used with CodeModeToolset.'
            )
