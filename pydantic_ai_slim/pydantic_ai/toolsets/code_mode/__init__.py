"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import TypeAdapter
from typing_extensions import NotRequired, TypedDict

from pydantic_ai.runtime.abstract import (
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)
from pydantic_ai.runtime.monty import MontyRuntime

from ... import exceptions
from ..._run_context import AgentDepsT, RunContext
from ..._tool_manager import ToolManager
from ...exceptions import ApprovalRequired, CallDeferred, ModelRetry
from ...messages import ToolCallPart
from ...tools import ToolApproved, ToolDefinition, ToolDenied
from ..abstract import SchemaValidatorProt, ToolsetTool
from ..function import FunctionToolset, FunctionToolsetTool
from ..wrapper import WrapperToolset
from .sanitization import ToolNameMapping
from .signature import Signature, signature_from_function, signature_from_schema

# Type alias for description handler callback
# Takes (description, tool_definition) and returns processed description
DescriptionHandler = Callable[[str, ToolDefinition], str]


class InterruptedCall(TypedDict):
    """Details of a nested tool call that was interrupted during code execution.

    Attributes:
        call_id: Unique identifier for this interrupted call.
        tool_name: The name of the tool that was called.
        args: Positional arguments passed to the tool.
        kwargs: Keyword arguments passed to the tool.
        type: Whether this call needs 'approval' or 'external' execution.
    """

    call_id: str
    tool_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    type: Literal['approval', 'external']


class CodeModeContext(TypedDict):
    """Context for code mode tool calls (run_code).

    This TypedDict defines the structure of context passed between code mode interruption
    and resumption. Use this type for compile-time checking of required fields.

    When code execution is interrupted (ApprovalRequired/CallDeferred), this context
    is returned in `DeferredToolRequests.context[tool_call_id]`.

    When resuming, pass this context back in `DeferredToolResults.context[tool_call_id]`
    with the added `results` field.

    Attributes:
        checkpoint: Monty runtime checkpoint for resuming execution.
        interrupted_calls: List of nested tool calls that were interrupted.
        completed_results: Results from calls that succeeded before the interruption.
            Keyed by string call_id, values in Monty result format. Passed back on
            resume so those calls are not re-executed.
        results: Map of call_id to result for each interrupted call. Required when resuming.
            Values can be: ToolApproved(), ToolDenied(message=...), or any external result.

    Example:
        ```python
        # Receiving context from DeferredToolRequests
        ctx: CodeModeContext = result.output.context[tool_call_id]
        checkpoint = ctx['checkpoint']  # bytes
        interrupted = ctx['interrupted_calls']  # list[InterruptedCall]

        # Building resume context with results
        resume_ctx: CodeModeContext = {
            'checkpoint': ctx['checkpoint'],
            'interrupted_calls': ctx['interrupted_calls'],
            'completed_results': ctx.get('completed_results', {}),
            'results': {ic['call_id']: ToolApproved() for ic in interrupted},
        }
        ```
    """

    checkpoint: bytes
    interrupted_calls: list[InterruptedCall]
    completed_results: NotRequired[dict[str, Any]]
    results: NotRequired[dict[str, Any]]


__all__ = (
    'CodeModeToolset',
    'DescriptionHandler',
    'Signature',
    'build_code_mode_prompt',
    'signature_from_function',
    'signature_from_schema',
    'ApprovalRequired',
    'CallDeferred',
    'CodeModeContext',
    'InterruptedCall',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'


def build_code_mode_prompt(*, signatures: list[str]) -> str:
    """Build the default code mode prompt with the given tool signatures.

    This is the default prompt builder used by CodeModeToolset. Users can provide
    their own prompt_builder callback to customize the prompt entirely.

    Args:
        signatures: List of Python function signatures for available tools.

    Returns:
        The complete prompt describing code mode capabilities and available functions.
    """
    functions_block = '\n\n'.join(signatures)
    # TODO: The first line of the prompt should be customizable by the user using Prompt Templates #3656
    return f"""\
ALWAYS use run_code to solve the ENTIRE task in a single code block. Do not call tools individually - write one comprehensive Python script that fetches all data, processes it, and returns the complete answer.

CRITICAL execution model:
- Solve the COMPLETE problem in ONE run_code call - not partial solutions
- Each run_code call is ISOLATED - variables do NOT persist between calls
- Plan your entire solution before writing code, then implement it all at once


CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):
- No imports - use only the provided functions and builtins (len, sum, str, etc.) or write your own functions.

How to write effective code:
- External functions return coroutines - use `await` to get results
- For sequential execution: `items = await get_items()`
- For parallel execution of independent calls, fire first then await:
  ```python
  future_items = get_items()   # Fire (no await)
  future_users = get_users()   # Fire (no await)
  items = await future_items   # Both execute in parallel
  users = await future_users
  ```
- ALWAYS use keyword arguments (e.g., `get_user(id=123)` not `get_user(123)`)
- Use for loops to handle multiple items
- NEVER return raw tool results - always extract/filter to only what you need
- The last expression evaluated becomes the return value - make it a processed summary, not raw data

Available functions:

```python
{functions_block}
```

Example - parallel fetching, then sequential processing:
```python
# PARALLEL: Fire independent calls first (no await yet)
future_items = get_items(category="electronics")
future_users = get_users(status="active")

# Await results - both calls execute in parallel
items = await future_items
users = await future_users

# SEQUENTIAL: Process items (each depends on previous result)
results = []
total = 0
for item in items:
    details = await get_item_details(id=item["id"])
    if details["status"] == "active":
        total = total + details["price"]
        results.append({{"name": item["name"], "price": details["price"]}})

# Return processed summary, NOT raw data
{{"total": total, "count": len(results), "items": results, "user_count": len(users)}}
```"""


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


def _get_tool_signature(
    tool: ToolsetTool[Any],
    name_override: str | None = None,
    description_handler: DescriptionHandler | None = None,
) -> Signature:
    """Get a Signature object for a tool.

    For native function tools, uses the original function's signature (including return type).
    For external tools (MCP, etc.), converts the JSON schema to a signature.

    Args:
        tool: The tool to generate a signature for.
        name_override: Optional name to use instead of the tool's original name.
            Used to show sanitized names (valid Python identifiers) to the LLM.
        description_handler: Optional callback to process/truncate tool descriptions.

    Returns:
        A Signature object. Use str(sig) for type checking, sig.with_typeddicts('...') for LLM.

    Note: Code mode always includes return types because the model needs to know
    what structure each function returns to write correct code.
    """
    signature_name = name_override or tool.tool_def.name

    # Process description through handler if provided
    description = tool.tool_def.description
    if description and description_handler:
        description = description_handler(description, tool.tool_def)

    if isinstance(tool, FunctionToolsetTool) and isinstance(tool.toolset, FunctionToolset):
        tool_name = tool.tool_def.name
        if tool_name in tool.toolset.tools:
            original_tool = tool.toolset.tools[tool_name]
            return signature_from_function(
                original_tool.function,
                name=signature_name,
                description=description,
                include_return_type=True,  # Always show return types in code mode
            )

    # For external tools (MCP, etc.), convert JSON schema to signature
    return signature_from_schema(
        name=signature_name,
        parameters_json_schema=tool.tool_def.parameters_json_schema,
        description=description,
        return_json_schema=tool.tool_def.return_schema,  # Always include if available
        namespace_defs=True,
    )


@dataclass(kw_only=True)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """A toolset that exposes wrapped tools as callable Python functions in a code execution context.

    Args:
        wrapped: The underlying toolset to wrap.
        prompt_builder: Optional callback to build a custom prompt. If not provided,
            uses `build_code_mode_prompt`. The callback receives `signatures` as a
            keyword argument containing the list of Python function signatures.
        max_retries: Maximum number of retries for code execution errors (type/syntax/runtime).
            Defaults to 3. Increase for complex code generation tasks or less capable models.
        tool_name_prefix: Optional prefix to add to all sanitized tool names (e.g., MCP server name).
            Helps avoid name collisions when combining tools from multiple sources.
        description_handler: Optional callback to process tool descriptions on a per-tool basis.
            Takes (description, tool_definition) and returns the processed description.
            Useful for truncating long descriptions, removing embedded JSON schemas, etc.

            .. warning::
                This callback is not serializable. If you need to serialize the toolset
                (e.g., for distributed execution), you'll need to recreate it with the
                callback after deserialization.
    """

    runtime: CodeRuntime = field(default_factory=MontyRuntime)
    prompt_builder: Callable[..., str] = build_code_mode_prompt
    max_retries: int = 3
    tool_name_prefix: str | None = None
    description_handler: DescriptionHandler | None = None
    _cached_signatures: list[str] = field(default_factory=lambda: [], init=False, repr=False)
    _name_mapping: ToolNameMapping = field(default_factory=ToolNameMapping, init=False, repr=False)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = await super().get_tools(ctx)

        # Sanitize tool names to valid Python identifiers and build mapping
        self._name_mapping = ToolNameMapping(prefix=self.tool_name_prefix)
        sanitized_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        available_functions: list[str] = []

        for original_name, tool in wrapped_tools.items():
            sanitized_name = self._name_mapping.add(original_name)
            sanitized_tools[sanitized_name] = tool
            sig = _get_tool_signature(
                tool,
                name_override=sanitized_name,
                description_handler=self.description_handler,
            )
            # Use `...` body for both LLM display and type checking
            # (Monty converts to `raise NotImplementedError()` internally for ty compatibility)
            available_functions.append(sig.with_typeddicts())

        self._cached_signatures = available_functions

        description = self.prompt_builder(signatures=available_functions)
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
        results_map: dict[str, Any] | None = None,
    ) -> ToolCallback:
        """Create a callback for Monty to invoke when code calls external functions.

        Args:
            tool: The code mode tool with original tools mapping.
            code_mode_tool_manager: ToolManager for executing nested tool calls.
            sanitized_to_original: Mapping from sanitized names to original tool names.
            results_map: Optional mapping of call_id to pre-resolved results (for resume path).
                Values can be ToolApproved (execute with approval), ToolDenied (raise error),
                or any other value (return directly as external result).
        """

        async def callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_name = sanitized_to_original.get(sanitized_name, sanitized_name)

            # Check for pre-resolved result (resume path)
            if results_map is not None and call.call_id in results_map:
                result = results_map[call.call_id]

                if isinstance(result, ToolApproved):
                    # Approved - execute with approval flag
                    tool_kwargs = self._build_tool_kwargs(call, tool, sanitized_name)
                    tool_call_part = ToolCallPart(tool_name=original_name, args=tool_kwargs)
                    return await code_mode_tool_manager.handle_call(
                        tool_call_part, wrap_validation_errors=False, approved=True
                    )
                elif isinstance(result, ToolDenied):
                    # Denied - raise to tell LLM
                    raise ModelRetry(result.message)
                else:
                    # External result - return directly
                    return result

            # Normal path - build and execute (may raise ApprovalRequired)
            tool_kwargs = self._build_tool_kwargs(call, tool, sanitized_name)
            tool_call_part = ToolCallPart(tool_name=original_name, args=tool_kwargs)

            # Route through full ToolManager flow:
            # handle_call → _call_function_tool (tracing + usage) → _call_tool (validate + enrich + call)
            # wrap_validation_errors=False: let raw errors propagate to the runtime.
            # If UnexpectedModelBehavior is raised (e.g. max_retries=0 tool), it propagates
            # naturally through the Monty runtime as CodeRuntimeError → outer ModelRetry.
            return await code_mode_tool_manager.handle_call(
                tool_call_part,
                wrap_validation_errors=False,
            )

        return callback

    async def _handle_resume(
        self,
        ctx: RunContext[AgentDepsT],
        tool: _CodeModeTool[AgentDepsT],
        code_mode_tool_manager: ToolManager[AgentDepsT],
        sanitized_to_original: dict[str, str],
    ) -> Any:
        """Handle resumption from a checkpoint after approval/external results are provided.

        Args:
            ctx: The run context with tool_call_context containing CodeModeContext with checkpoint and results.
            tool: The code mode tool.
            code_mode_tool_manager: ToolManager for executing nested tool calls.
            sanitized_to_original: Mapping from sanitized names to original tool names.

        Returns:
            The result of the resumed code execution.

        Raises:
            UserError: If context is missing required data (checkpoint, results).
            ModelRetry: If code execution fails after resume.
        """
        raw_context = ctx.tool_call_context
        if raw_context is None:
            raise exceptions.UserError(
                'Code mode resume requires context with checkpoint. '
                'Pass back the original DeferredToolRequests.context[tool_call_id] '
                'with an added "results" key mapping call_id to ToolApproved(), ToolDenied(), or external result.'
            )

        checkpoint = raw_context.get('checkpoint')
        if checkpoint is None:
            raise exceptions.UserError(
                'Code mode resume requires checkpoint in context. '
                'The checkpoint should be preserved from the original DeferredToolRequests.context.'
            )

        context = cast(CodeModeContext, raw_context)
        interrupted_calls = context.get('interrupted_calls', [])
        results_map = context.get('results')

        if not results_map:
            raise exceptions.UserError(
                'Code mode resume requires context with results for nested calls. '
                'Add a "results" key to the context mapping call_id to '
                'ToolApproved(), ToolDenied(), or the external result.'
            )

        missing_results = [ic['call_id'] for ic in interrupted_calls if ic['call_id'] not in results_map]
        if missing_results:
            raise exceptions.UserError(
                f'Missing results for interrupted calls: {missing_results}. '
                'All interrupted calls must have a result (ToolApproved, ToolDenied, or value).'
            )

        # Recover results from calls that completed before the interruption.
        # Keys are converted back from str to int for Monty's result format.
        completed_results_raw = context.get('completed_results', {})
        completed_results = {int(k): v for k, v in completed_results_raw.items()} if completed_results_raw else None

        callback = self._make_tool_callback(tool, code_mode_tool_manager, sanitized_to_original, results_map)

        try:
            return await self.runtime.resume(
                checkpoint,
                callback,
                cast(list[dict[str, Any]], interrupted_calls),
                completed_results=completed_results,
            )
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}')

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        code = tool_args['code']
        if name != _CODE_MODE_TOOL_NAME:
            raise exceptions.UserError(f'CodeModeToolset.call_tool expected tool name {_CODE_MODE_TOOL_NAME!r}, got {name!r}')
        if not isinstance(tool, _CodeModeTool):
            raise exceptions.UserError(f'CodeModeToolset.call_tool expected _CodeModeTool, got {type(tool).__name__}')
        if not isinstance(code, str):
            raise exceptions.UserError(f'CodeModeToolset.call_tool expected code to be a string, got {type(code).__name__}')

        original_name_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        sanitized_to_original: dict[str, str] = {}
        for sanitized, t in tool.original_tools.items():
            orig = self._name_mapping.get_original(sanitized) or sanitized
            original_name_tools[orig] = t
            sanitized_to_original[sanitized] = orig

        code_mode_tool_manager = ToolManager(
            toolset=self.wrapped,
            ctx=ctx,
            tools=original_name_tools,
        )

        # Check if this is a resume from deferred (tool_call_approved indicates resumption)
        if ctx.tool_call_approved:
            return await self._handle_resume(ctx, tool, code_mode_tool_manager, sanitized_to_original)

        # Normal execution path
        callback = self._make_tool_callback(tool, code_mode_tool_manager, sanitized_to_original, results_map=None)

        try:
            functions = list(tool.original_tools.keys())
            return await self.runtime.run(code, functions, callback, signatures=self._cached_signatures)
        except CodeTypingError as e:
            raise ModelRetry(f'Type error in generated code:\n{e.message}')
        except CodeSyntaxError as e:
            raise ModelRetry(f'Syntax error in generated code:\n{e.message}')
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}')
        except CodeInterruptedError as e:
            # Raise ApprovalRequired so run_code ends up in approvals (not calls)
            # This ensures ToolApproved flows through correctly on resume
            # The context contains checkpoint and interrupted_calls info
            # We use context (not metadata) because this data is REQUIRED for resumption
            interrupted_calls: list[InterruptedCall] = [
                {
                    'call_id': ic.call.call_id,
                    'tool_name': ic.call.function_name,
                    'args': ic.call.args,
                    'kwargs': ic.call.kwargs,
                    'type': 'approval' if isinstance(ic.type, ApprovalRequired) else 'external',
                }
                for ic in e.interrupted_calls
            ]
            code_mode_context: CodeModeContext = {
                'checkpoint': e.checkpoint,
                'interrupted_calls': interrupted_calls,
                'completed_results': {str(k): v for k, v in e.completed_results.items()},
            }
            raise ApprovalRequired(context=cast(dict[str, Any], code_mode_context))
