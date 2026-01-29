"""Code mode toolset that wraps tools as Python functions callable from generated code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai.runtime.abstract import (
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    CodeTypingError,
    FunctionCall,
    ToolCallback,
)
from pydantic_ai.runtime.monty import MontyRuntime

from ..._run_context import AgentDepsT, RunContext
from ...exceptions import ModelRetry
from ...tools import ToolDefinition
from ..abstract import SchemaValidatorProt, ToolsetTool
from ..function import FunctionToolset, FunctionToolsetTool
from ..wrapper import WrapperToolset
from .sanitization import ToolNameMapping
from .signature import SignatureResult, signature_from_function, signature_from_schema

# Type alias for description handler callback
# Takes (description, tool_definition) and returns processed description
DescriptionHandler = Callable[[str, ToolDefinition], str]

__all__ = (
    'CodeModeToolset',
    'DescriptionHandler',
    'SignatureResult',
    'build_code_mode_prompt',
    'signature_from_function',
    'signature_from_schema',
)


class _CodeToolArguments(TypedDict):
    code: str


_CODE_ADAPTER = TypeAdapter(_CodeToolArguments)
_CODE_MODE_TOOL_NAME = 'run_code'


# We will attempt to remove all restrictions
# CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):
# - No imports - use only the provided functions and builtins (len, sum, str, etc.)
# - No while loops - use for loops instead
# - No comprehensions (list/dict/set) or generator expressions - use explicit for loops
# - No lambdas - define logic inline
# - No tuple unpacking (e.g., `a, b = 1, 2`) - assign variables separately
# - No list indexing or slicing (e.g., `lst[0]`, `lst[:10]`) - use for loops to iterate
# - No break or continue statements - use conditional logic instead
# - No string methods (.join, .split, .upper, etc.) - return data structures, not formatted strings

# What DOES work:
# - Dict assignment: `d["key"] = value`
# - Dict methods: `.get()`, `.keys()`, `.values()`, `.items()`
# - List methods: `.append()`
# - F-strings: `f"value is {{x}}"`
# - Builtins: `len()`, `sum()`, `str()`, `list()`, `range()`


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
- ALWAYS use `await` when calling external functions (e.g., `items = await get_items()`)
- ALWAYS use keyword arguments when calling functions (e.g., `get_user(id=123)` not `get_user(123)`)
- Use for loops to handle multiple items
- NEVER return raw tool results - always extract/filter to only what you need
- The last expression evaluated becomes the return value - make it a processed summary, not raw data

Available functions:

```python
{functions_block}
```

Example - fetching, filtering, and summarizing in one execution:
```python
# Fetch data
items = await get_items(category="electronics")

# Process immediately - extract only needed fields
results = []
total = 0
for item in items:
    details = await get_item_details(id=item["id"])
    if details["status"] == "active":
        total = total + details["price"]
        results.append({{"name": item["name"], "price": details["price"]}})

# Return processed summary, NOT raw data
{{"total": total, "count": len(results), "items": results}}
```"""


@dataclass(kw_only=True)
class _CodeModeTool(ToolsetTool[AgentDepsT]):
    original_tools: dict[str, ToolsetTool[AgentDepsT]]


def _get_tool_signature(
    tool: ToolsetTool[Any],
    name_override: str | None = None,
    description_handler: DescriptionHandler | None = None,
) -> str:
    """Get a Python signature string for a tool.

    For native function tools, uses the original function's signature (including return type).
    For external tools (MCP, etc.), converts the JSON schema to a signature.

    Args:
        tool: The tool to generate a signature for.
        name_override: Optional name to use instead of the tool's original name.
            Used to show sanitized names (valid Python identifiers) to the LLM.
        description_handler: Optional callback to process/truncate tool descriptions.

    Note: Code mode always includes return types because the model needs to know
    what structure each function returns to write correct code.
    """
    # Code mode MUST show return types - without them the model can't know
    # that get_weather() returns a dict with 'temperature' key vs just a number.
    # We ignore tool.include_return_schema here because code mode has different needs
    # than traditional tool calling.
    # TODO: For native function tools, we call signature_from_function which uses
    # inspect.signature() and get_type_hints() every time get_tools() is called.
    # This re-inspection is needed to filter out RunContext params and format the signature,
    # but could be cached/precomputed when the tool is registered. This becomes non-optimal
    # when: (1) get_tools() is called frequently during an agent run, (2) there are many
    # tools in the toolset, (3) the toolset is reused across multiple agent runs.
    # Consider storing the formatted signature string on FunctionToolsetTool at registration time.
    signature_name = name_override or tool.tool_def.name

    # Process description through handler if provided
    description = tool.tool_def.description
    if description and description_handler:
        description = description_handler(description, tool.tool_def)

    if isinstance(tool, FunctionToolsetTool) and isinstance(tool.toolset, FunctionToolset):
        tool_name = tool.tool_def.name
        if tool_name in tool.toolset.tools:
            original_tool = tool.toolset.tools[tool_name]
            result = signature_from_function(
                original_tool.function,
                name=signature_name,
                description=description,
                include_return_type=True,  # Always show return types in code mode
            )
            if result.typeddict_defs:
                return '\n\n'.join(result.typeddict_defs) + '\n\n' + result.signature
            return result.signature

    # For external tools (MCP, etc.), convert JSON schema to signature
    result = signature_from_schema(
        name=signature_name,
        parameters_json_schema=tool.tool_def.parameters_json_schema,
        description=description,
        return_json_schema=tool.tool_def.return_schema,  # Always include if available
        namespace_defs=True,
    )

    if result.typeddict_defs:
        return '\n\n'.join(result.typeddict_defs) + '\n\n' + result.signature
    return result.signature


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
    # The user needs to provide a runtime to be used with the toolset. If not provided we will use Monty by default.
    # At the moment we only support Monty as well, need to check if Modal or some other Sandbox can also be used with this.
    prompt_builder: Callable[..., str] = build_code_mode_prompt
    max_retries: int = 3
    tool_name_prefix: str | None = None
    # TODO: description_handler is not serializable. For distributed execution scenarios,
    # we may need an alternative approach (e.g., named handlers registered in a registry,
    # or description processing at tool registration time rather than at signature generation).
    # Claude added the above comment, yet to validate

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
            signature = _get_tool_signature(
                tool,
                name_override=sanitized_name,
                description_handler=self.description_handler,
            )
            available_functions.append(signature)

        self._cached_signatures = available_functions

        # TODO: This dumps all tool signatures up-front, which can bloat context for large toolsets.
        # Example: hundreds of MCP tools can push tens of thousands of tokens into the prompt,
        # defeating the progressive-disclosure approach described in code-mode references.
        # Consider: progressive discovery (list tool names first, fetch signatures on demand).
        # David to look into this
        llm_signatures = [sig.replace('raise NotImplementedError()', '...') for sig in available_functions]
        description = self.prompt_builder(signatures=llm_signatures)
        # TODO: Ideally we'd use kind='output' to make the code result be the final answer
        # without a second LLM call. However, output tools are treated differently by models -
        # they expect to provide structured output directly, not execute code. We need a way
        # to have a function tool whose result becomes the final output without another LLM call.
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

    def _make_tool_callback(
        self,
        tool: _CodeModeTool[AgentDepsT],
        ctx: RunContext[AgentDepsT],
        # checkpoint: dict[str, Any] | None,
    ) -> ToolCallback:
        # Match approval by tool name rather than a boolean flag. With concurrent
        # execution, multiple callbacks fire simultaneously — a boolean consumed by
        # whichever task runs first would approve the wrong call. String matching
        # ensures only the callback whose original_name matches gets approved.
        # approved_tool_name: str | None = checkpoint.get('tool_name') if checkpoint else None
        # override_args: dict[str, Any] | None = (
        #     ctx.tool_call_metadata.get('_override_args') if checkpoint and ctx.tool_call_metadata else None
        # )

        async def callback(call: FunctionCall) -> Any:
            sanitized_name = call.function_name
            original_tool = tool.original_tools[sanitized_name]
            original_name = self._name_mapping.get_original(sanitized_name) or sanitized_name

            # Build kwargs (positional arg fallback)
            tool_kwargs = dict(call.kwargs)
            if call.args:
                # Positional args are mapped using JSON schema property order, which may not match
                # the tool's actual parameter order. The prompt instructs models to use keyword
                # arguments only, but we handle positional args as a fallback for non-compliant models.
                param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
                for i, arg in enumerate(call.args):
                    if i < len(param_names):
                        tool_kwargs[param_names[i]] = arg

            span_attributes = {
                'gen_ai.tool.name': original_name,
                'code_mode.inner_tool': True,
                'code_mode.sanitized_name': sanitized_name,
                'logfire.msg': f'code mode calling: {original_name}',
            }

            # TODO: Consider letting tool manager handle the span(Discussion with Douwe)?
            span_name = f'code_mode_tool:{original_name}'
            with ctx.tracer.start_as_current_span(span_name, attributes=span_attributes):
                return await super(CodeModeToolset, self).call_tool(original_name, tool_kwargs, ctx, original_tool)

        return callback

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        code = tool_args['code']
        assert name == _CODE_MODE_TOOL_NAME
        assert isinstance(tool, _CodeModeTool)
        assert isinstance(code, str)

        callback = self._make_tool_callback(tool, ctx)

        try:
            functions = list(tool.original_tools.keys())

            return await self.runtime.run(
                code, functions, callback, self._cached_signatures
            )  # I don't know if it is wise to send a shared mutable state like this through?

        except CodeTypingError as e:
            raise ModelRetry(f'Type error in generated code:\n{e.message}')
        except CodeSyntaxError as e:
            raise ModelRetry(f'Syntax error in generated code:\n{e.message}')
        except CodeRuntimeError as e:
            raise ModelRetry(f'Runtime error in generated code:\n{e.message}')
