from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated, Any, Concatenate, Generic, Literal, TypeAlias, Union, cast

from pydantic import AliasChoices, Field
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import ParamSpec, Self, TypeVar

from . import _function_schema, _utils
from ._deferred import (
    DeferredToolApprovalResult as DeferredToolApprovalResult,
    DeferredToolCallResult as DeferredToolCallResult,
    DeferredToolRequests as DeferredToolRequests,
    DeferredToolResult as DeferredToolResult,
    DeferredToolResults as DeferredToolResults,
    ToolApproved as ToolApproved,
    ToolDenied as ToolDenied,
)
from ._run_context import AgentDepsT, RunContext
from .exceptions import UserError
from .function_signature import FunctionSignature
from .messages import ToolPartKind
from .native_tools import AbstractNativeTool

__all__ = (
    'AgentDepsT',
    'ArgsValidatorFunc',
    'DocstringFormat',
    'RunContext',
    'SystemPromptFunc',
    'ToolFuncContext',
    'ToolFuncPlain',
    'ToolFuncEither',
    'ToolParams',
    'ToolPrepareFunc',
    'ToolsPrepareFunc',
    'ToolSelectorFunc',
    'ToolSelector',
    'matches_tool_selector',
    'AgentNativeTool',
    'NativeToolFunc',
    'Tool',
    'ObjectJsonSchema',
    'ToolDefinition',
    'DeferredToolRequests',
    'DeferredToolResults',
    'ToolApproved',
    'ToolDenied',
)


ToolParams = ParamSpec('ToolParams', default=...)
"""Retrieval function param spec."""

SystemPromptFunc: TypeAlias = (
    Callable[[RunContext[AgentDepsT]], str | None]
    | Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]
    | Callable[[], str | None]
    | Callable[[], Awaitable[str | None]]
)
"""A function that may or may not take `RunContext` as an argument, and may or may not be async.

Functions which return None are excluded from model requests.

Usage `SystemPromptFunc[AgentDepsT]`.
"""

ToolFuncContext: TypeAlias = Callable[Concatenate[RunContext[AgentDepsT], ToolParams], Any]
"""A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDepsT, ToolParams]`.
"""
ToolFuncPlain: TypeAlias = Callable[ToolParams, Any]
"""A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.
"""
ToolFuncEither: TypeAlias = ToolFuncContext[AgentDepsT, ToolParams] | ToolFuncPlain[ToolParams]
"""Either kind of tool function.

This is just a union of [`ToolFuncContext`][pydantic_ai.tools.ToolFuncContext] and
[`ToolFuncPlain`][pydantic_ai.tools.ToolFuncPlain].

Usage `ToolFuncEither[AgentDepsT, ToolParams]`.
"""
ArgsValidatorFunc: TypeAlias = (
    Callable[Concatenate[RunContext[AgentDepsT], ToolParams], Awaitable[None]]
    | Callable[Concatenate[RunContext[AgentDepsT], ToolParams], None]
)
"""A function that validates tool arguments before execution.

The validator receives the same typed parameters as the tool function,
with [`RunContext`][pydantic_ai.tools.RunContext] as the first argument for dependency access.

Should raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on validation failure.
"""
ToolPrepareFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], 'ToolDefinition'],
    Union[Awaitable['ToolDefinition | None'], 'ToolDefinition', None],
]
"""Definition of a function that can prepare a tool definition at call time.
Both sync and async functions are accepted.

See [tool docs](../tools-advanced.md#tool-prepare) for more information.

Example — here `only_if_42` is valid as a `ToolPrepareFunc`:

```python {noqa="I001"}
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps == 42:
        return tool_def

def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'

hitchhiker = Tool(hitchhiker, prepare=only_if_42)
```

Usage `ToolPrepareFunc[AgentDepsT]`.
"""

ToolsPrepareFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], list['ToolDefinition']],
    Awaitable[list['ToolDefinition']] | list['ToolDefinition'],
]
"""Definition of a function that can prepare the tool definition of all tools for each step.
This is useful if you want to customize the definition of multiple tools or you want to register
a subset of tools for a given step. Both sync and async functions are accepted.

Example — here `turn_on_strict_if_openai` is valid as a `ToolsPrepareFunc`:

```python {noqa="I001"}
from dataclasses import replace

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition


def turn_on_strict_if_openai(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs

agent = Agent('openai:gpt-5.2', capabilities=[PrepareTools(turn_on_strict_if_openai)])
```

Usage `ToolsPrepareFunc[AgentDepsT]`.
"""

ToolSelectorFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], 'ToolDefinition'],
    bool | Awaitable[bool],
]
"""A callable that decides whether a tool matches a selection criterion.

Receives the run context and a tool definition, returns `True` if the tool is selected.
Both sync and async functions are accepted.

Usage `ToolSelectorFunc[AgentDepsT]`.
"""

ToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any] | ToolSelectorFunc[AgentDepsT]
"""Specifies which tools a capability or toolset wrapper should apply to.

- `'all'`: matches every tool (default for most capabilities).
- `Sequence[str]`: matches tools whose names are in the sequence.
- `dict[str, Any]`: matches tools whose
  [`metadata`][pydantic_ai.tools.ToolDefinition.metadata] contains all the
  specified key-value pairs (deep inclusion check — nested dicts are compared
  recursively, and the tool's metadata may have additional keys).
- `Callable[[RunContext, ToolDefinition], bool | Awaitable[bool]]`:
  custom sync or async predicate.

The first three forms are serializable for use in agent specs (YAML/JSON).

Usage `ToolSelector[AgentDepsT]`.
"""


def _metadata_includes(metadata: dict[str, Any], selector: dict[str, Any]) -> bool:
    """Check whether *metadata* deeply includes all key-value pairs from *selector*."""
    for key, expected in selector.items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if isinstance(expected, dict) and isinstance(actual, dict):
            if not _metadata_includes(cast(dict[str, Any], actual), cast(dict[str, Any], expected)):
                return False
        elif actual != expected:
            return False
    return True


async def matches_tool_selector(
    selector: ToolSelector[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    tool_def: ToolDefinition,
) -> bool:
    """Check whether a tool definition matches a [`ToolSelector`][pydantic_ai.tools.ToolSelector].

    Args:
        selector: The selector to check against.
        ctx: The current run context.
        tool_def: The tool definition to test.

    Returns:
        `True` if the tool matches the selector.
    """
    if selector == 'all':
        return True
    if callable(selector):
        result = selector(ctx, tool_def)
        if inspect.isawaitable(result):
            return await result
        return result
    if isinstance(selector, dict):
        metadata: dict[str, Any] = tool_def.metadata or {}
        return _metadata_includes(metadata, selector)
    if isinstance(selector, str):
        return tool_def.name == selector
    # Sequence[str] — match by tool name
    return tool_def.name in selector


NativeToolFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]], Awaitable[AbstractNativeTool | None] | AbstractNativeTool | None
]
"""Definition of a function that can prepare a native tool at call time.

This is useful if you want to customize the native tool based on the run context (e.g. user dependencies),
or omit it completely from a step.
"""

AgentNativeTool: TypeAlias = AbstractNativeTool | NativeToolFunc[AgentDepsT]
"""A native tool or a function that dynamically produces one.

This is a convenience alias for `AbstractNativeTool | NativeToolFunc[AgentDepsT]`.
"""

DocstringFormat: TypeAlias = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats.

* `'google'` — [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
* `'numpy'` — [Numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings.
* `'sphinx'` — [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) docstrings.
* `'auto'` — Automatically infer the format based on the structure of the docstring.
"""


A = TypeVar('A')


class GenerateToolJsonSchema(GenerateJsonSchema):
    def _named_required_fields_schema(self, named_required_fields: Sequence[tuple[str, bool, Any]]) -> JsonSchemaValue:
        # Remove largely-useless property titles
        s = super()._named_required_fields_schema(named_required_fields)
        for p in s.get('properties', {}):
            s['properties'][p].pop('title', None)
        return s


ToolAgentDepsT = TypeVar('ToolAgentDepsT', default=object, contravariant=True)
"""Type variable for agent dependencies for a tool."""


def _validate_max_retries(max_retries: int | None) -> None:
    if max_retries is not None and max_retries < 0:
        raise UserError(f'max_retries must be >= 0, got {max_retries}')


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and timeout <= 0:
        raise UserError(f'timeout must be > 0, got {timeout}')


@dataclass(init=False)
class Tool(Generic[ToolAgentDepsT]):
    """A tool function for an agent."""

    function: ToolFuncEither[ToolAgentDepsT]
    takes_ctx: bool
    max_retries: int | None
    name: str
    description: str | None
    prepare: ToolPrepareFunc[ToolAgentDepsT] | None
    args_validator: ArgsValidatorFunc[ToolAgentDepsT, ...] | None
    docstring_format: DocstringFormat
    require_parameter_descriptions: bool
    strict: bool | None
    sequential: bool
    requires_approval: bool
    metadata: dict[str, Any] | None
    timeout: float | None
    defer_loading: bool
    include_return_schema: bool | None
    function_schema: _function_schema.FunctionSchema
    """
    The base JSON schema for the tool's parameters.

    This schema may be modified by the `prepare` function or by the Model class prior to including it in an API request.
    """

    def __init__(
        self,
        function: ToolFuncEither[ToolAgentDepsT, ToolParams],
        *,
        takes_ctx: bool | None = None,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
        prepare: ToolPrepareFunc[ToolAgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[ToolAgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
        function_schema: _function_schema.FunctionSchema | None = None,
    ):
        """Create a new tool instance.

        Example usage:

        ```python {noqa="I001"}
        from pydantic_ai import Agent, RunContext, Tool

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        agent = Agent('test', tools=[Tool(my_tool)])
        ```

        or with a custom prepare method:

        ```python {noqa="I001"}

        from pydantic_ai import Agent, RunContext, Tool
        from pydantic_ai.tools import ToolDefinition

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        async def prep_my_tool(
            ctx: RunContext[int], tool_def: ToolDefinition
        ) -> ToolDefinition | None:
            # only register the tool if `deps == 42`
            if ctx.deps == 42:
                return tool_def

        agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])
        ```


        Args:
            function: The Python function to call as the tool.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument,
                this is inferred if unset.
            max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
            name: Name of the tool, inferred from the function if `None`.
            description: Description of the tool, inferred from the function if `None`.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            args_validator: custom method to validate tool arguments after schema validation has passed,
                before execution. The validator receives the already-validated and type-converted parameters,
                with `RunContext` as the first argument.
                Should raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on validation failure,
                return `None` on success.
                See [`ArgsValidatorFunc`][pydantic_ai.tools.ArgsValidatorFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether this tool acts as a barrier that runs alone, not overlapping with other tool calls.
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            timeout: Timeout in seconds for tool execution. If the tool takes longer, a retry prompt is returned to the model.
                Defaults to None (no timeout).
            defer_loading: Whether to hide this tool until it's discovered via tool search. Defaults to False.
                See [Tool Search](../tools-advanced.md#tool-search) for more info.
            include_return_schema: Whether to include the return schema in the tool definition sent to the model.
                If `None`, defaults to `False` unless the [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is used.
            function_schema: The function schema to use for the tool. If not provided, it will be generated.
        """
        _validate_max_retries(max_retries)
        _validate_timeout(timeout)
        self.function = function
        self.name = name or function.__name__
        self.function_schema = function_schema or _function_schema.function_schema(
            function,
            schema_generator,
            tool_name=self.name,
            takes_ctx=takes_ctx,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
        )
        self.takes_ctx = self.function_schema.takes_ctx
        self.max_retries = max_retries
        self.description = description or self.function_schema.description
        self.prepare = prepare
        self.args_validator = args_validator
        self.docstring_format = docstring_format
        self.require_parameter_descriptions = require_parameter_descriptions
        self.strict = strict
        self.sequential = sequential
        self.requires_approval = requires_approval
        self.metadata = metadata
        self.timeout = timeout
        self.defer_loading = defer_loading
        self.include_return_schema = include_return_schema

    @classmethod
    def from_schema(
        cls,
        function: Callable[..., Any],
        name: str,
        description: str | None,
        json_schema: JsonSchemaValue,
        takes_ctx: bool = False,
        sequential: bool = False,
        args_validator: ArgsValidatorFunc[Any, ...] | None = None,
    ) -> Self:
        """Creates a Pydantic tool from a function and a JSON schema.

        Args:
            function: The function to call.
                This will be called with keywords only. Schema validation of
                the arguments is skipped, but a custom `args_validator` will
                still run if provided.
            name: The unique name of the tool that clearly communicates its purpose
            description: Used to tell the model how/when/why to use the tool.
                You can provide few-shot examples as a part of the description.
            json_schema: The schema for the function arguments
            takes_ctx: An optional boolean parameter indicating whether the function
                accepts the context object as an argument.
            sequential: Whether this tool acts as a barrier that runs alone, not overlapping with other tool calls.
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info. Defaults to False.
            args_validator: custom method to validate tool arguments after schema validation has passed,
                before execution. The validator receives the already-validated and type-converted parameters,
                with `RunContext` as the first argument.
                Should raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on validation failure,
                return `None` on success.
                See [`ArgsValidatorFunc`][pydantic_ai.tools.ArgsValidatorFunc].

        Returns:
            A Pydantic tool that calls the function
        """
        function_schema = _function_schema.FunctionSchema(
            function=function,
            name=name,
            description=description,
            validator=SchemaValidator(schema=core_schema.any_schema()),
            json_schema=json_schema,
            takes_ctx=takes_ctx,
            is_async=_utils.is_async_callable(function),
        )

        tool = cls(
            function,
            takes_ctx=takes_ctx,
            name=name,
            description=description,
            function_schema=function_schema,
            sequential=sequential,
            args_validator=args_validator,
        )
        return tool

    @property
    def tool_def(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.function_schema.json_schema,
            strict=self.strict,
            sequential=self.sequential,
            metadata=self.metadata,
            timeout=self.timeout,
            defer_loading=self.defer_loading,
            kind='unapproved' if self.requires_approval else 'function',
            return_schema=self.function_schema.return_schema,
            include_return_schema=self.include_return_schema,
        )

    async def prepare_tool_def(self, ctx: RunContext[ToolAgentDepsT]) -> ToolDefinition | None:
        """Get the tool definition.

        By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
        if it's set.

        Returns:
            return a `ToolDefinition` or `None` if the tools should not be registered for this run.
        """
        tool_def = self.tool_def

        if self.prepare is not None:
            result = self.prepare(ctx, tool_def)
            if inspect.isawaitable(result):
                return await result
            return result
        else:
            return tool_def


ObjectJsonSchema: TypeAlias = dict[str, Any]
"""Type representing JSON schema of an object, e.g. where `"type": "object"`.

This type is used to define tools parameters (aka arguments) in [ToolDefinition][pydantic_ai.tools.ToolDefinition].

With PEP-728 this should be a TypedDict with `type: Literal['object']`, and `extra_parts=Any`
"""

ToolKind: TypeAlias = Literal['function', 'output', 'external', 'unapproved']
"""Kind of tool."""


@dataclass(repr=False, kw_only=True)
class ToolDefinition:
    """Definition of a tool passed to a model.

    This is used for both function tools and output tools.
    """

    name: str
    """The name of the tool."""

    parameters_json_schema: ObjectJsonSchema = field(default_factory=lambda: {'type': 'object', 'properties': {}})
    """The JSON schema for the tool's parameters."""

    description: str | None = None
    """The description of the tool."""

    outer_typed_dict_key: str | None = None
    """The key in the outer [TypedDict] that wraps an output tool.

    This will only be set for output tools which don't have an `object` JSON schema.
    """

    strict: bool | None = None
    """Whether to enforce (vendor-specific) strict JSON schema validation for tool calls.

    Setting this to `True` while using a supported model generally imposes some restrictions on the tool's JSON schema
    in exchange for guaranteeing the API responses strictly match that schema.

    When `False`, the model may be free to generate other properties or types (depending on the vendor).
    When `None` (the default), the value will be inferred based on the compatibility of the parameters_json_schema.

    Note: this is currently supported by OpenAI and Anthropic models.
    """

    sequential: bool = False
    """Whether this tool acts as a barrier that runs alone, not overlapping with other tool calls.

    A `sequential=True` tool acts as a barrier: it runs alone, with tools the model emitted before it
    completing first and tools emitted after it starting only once it finishes. Other tools still run
    in parallel around it. To run an entire run's tools serially, use
    [`ToolManager.parallel_execution_mode('sequential')`][pydantic_ai.tool_manager.ToolManager.parallel_execution_mode]
    instead.
    """

    kind: ToolKind = field(default='function')
    """The kind of tool:

    - `'function'`: a tool that will be executed by Pydantic AI during an agent run and has its result returned to the model
    - `'output'`: a tool that passes through an output value that ends the run
    - `'external'`: a tool whose result will be produced outside of the Pydantic AI agent run in which it was called, because it depends on an upstream service (or user) or could take longer to generate than it's reasonable to keep the agent process running.
        See the [tools documentation](../deferred-tools.md#deferred-tools) for more info.
    - `'unapproved'`: a tool that requires human-in-the-loop approval.
        See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
    """

    metadata: dict[str, Any] | None = None
    """Tool metadata that can be set by the toolset this tool came from. It is not sent to the model, but can be used for filtering and tool behavior customization.

    For MCP tools, this contains the `meta` and `annotations` fields from the tool definition, as well as a `task` flag indicating whether the server declares support for task-augmented execution.
    """

    timeout: float | None = None
    """Timeout in seconds for tool execution.

    If the tool takes longer than this, a retry prompt is returned to the model.
    Defaults to None (no timeout).
    """

    defer_loading: bool = False
    """Whether this tool should be hidden from the model until something explicitly surfaces it.

    Carries two meanings depending on where in the pipeline you observe it:

    1. **User-input intent** — set on `Tool(defer_loading=True)` (or via a custom toolset)
       to opt this tool into deferred loading. This is what `prepare_tools` hooks and other
       pre-toolset-wrapping consumers see, and is the value users persist on `ToolDefinition`.
    2. **Current visibility state** — after a toolset like
       the internal `ToolSearchToolset` processes
       the corpus, it flips this field to `False` for tools whose discovery shows up in
       message history, so downstream `Model.prepare_request` filtering and adapter wire
       formatting can read "should this be on the wire?" off a single boolean.

    The dual meaning is acknowledged tech debt: a future `RunContext.loaded_tools` /
    equivalent will surface (2) as a derived view so this field cleanly stays a user-input
    flag. Until then, the toolset-set value flows through agent-graph plumbing on a per-step
    `ToolDefinition` instance built via `replace(...)`; user-persisted definitions are not
    mutated.

    See [Tool Search](../tools-advanced.md#tool-search) for more info.
    """

    unless_native: Annotated[
        str | None,
        # Old names were `prefer_builtin` and (after the builtin → native rename in https://github.com/pydantic/pydantic-ai/issues/5338)
        # `prefer_native`; keep accepting both for serialized-history backward compat.
        Field(validation_alias=AliasChoices('unless_native', 'prefer_native', 'prefer_builtin')),
    ] = None
    """If set, this tool is dropped from the wire when the named native tool is supported by the model.

    Generic version of the old `prefer_builtin` flag: a function tool carrying
    `unless_native='web_search'` is treated as a local fallback for the
    [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] native tool and silently
    removed from the request whenever the model handles `WebSearchTool` natively. It
    stays in the request when the native tool isn't supported.
    """

    with_native: str | None = None
    """If set, this tool is kept on the wire when the named native tool is supported, with the
    native tool's adapter applying any wire-format adjustments (e.g. setting `defer_loading=True`
    on the request param for the framework-managed tool-search native tool).

    Symmetric pair with `unless_native`:

    * `unless_native='X'` — drop me from the wire when X is supported (local fallback).
    * `with_native='X'` — keep me on the wire when X is supported, formatted via X's adapter
      (corpus member managed by the native tool).

    When the named native tool is unsupported, a tool with `with_native` and `defer_loading=True`
    is dropped (the corpus member is currently undiscovered, so the model can't call it on
    this provider); otherwise it's kept as a regular function tool.
    """

    # Implementation note for new typed native tools: registering a new tool_kind value
    # requires (1) extending the ToolPartKind Literal in messages.py, (2) defining
    # the typed subclass + narrower under pydantic_ai/<your_native_tool>.py and registering
    # in _TOOL_CALL_NARROWERS / _NATIVE_CALL_NARROWERS / _TOOL_RETURN_NARROWERS /
    # _NATIVE_RETURN_NARROWERS, (3) adding the (part_kind, tool_kind) → Tag entries
    # in messages.py's _TYPED_PART_TAGS and _TYPED_PART_TAGS_BY_TYPE registries, and
    # (4) extending the ModelResponsePart / ModelRequestPart Annotated unions with
    # the new typed subclasses.
    tool_kind: ToolPartKind | None = None
    """Discriminator for a cross-provider typed call/return shape (e.g. `'tool-search'`).

    Set by the framework when a tool emits parts that should be promoted to a typed
    subclass (such as [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart]
    and [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart]). Leave as
    `None` for user-defined function tools — they go through the standard
    [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] /
    [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] shapes.

    To detect a tool-search part regardless of execution path (native server-side vs.
    local fallback), check `part.tool_kind == 'tool-search'` — this works across both
    call/return and both server/local variants.

    Distinct from [`kind`][pydantic_ai.tools.ToolDefinition.kind], which is about invocation
    semantics (`'function'` / `'output'` / `'external'` / `'unapproved'`).
    """

    return_schema: ObjectJsonSchema | None = None
    """The JSON schema for the tool's return value.

    For models that natively support return schemas (e.g. Google Gemini), this is passed as a
    structured field in the API request. For other models, it is injected into the tool's
    description as JSON text. Only included when `include_return_schema` resolves to `True`.
    """

    include_return_schema: bool | None = None
    """Whether to include the return schema in the tool definition sent to the model.

    When `True`, the `return_schema` will be preserved and sent to the model.
    When `False`, the `return_schema` will be cleared before sending.
    When `None` (default), defaults to `False` unless the
    [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is used.
    """

    toolset_id: str | None = None
    """The ID of the toolset that this tool belongs to.

    Set automatically when tools are collected from toolsets. Can be used by capabilities
    (e.g. durable execution) to apply per-toolset configuration to tool operations.
    """

    capability_id: str | None = None
    """The id of the capability that contributed this tool, or `None` if the tool is not owned by a capability.

    Assigned once when the run's capabilities are set up and then carried on the `ToolDefinition`
    for the rest of that run — it does not change or reset between steps. For a tool owned by a
    deferred capability it gates visibility: the tool is revealed once that capability's id appears
    in [`RunContext.loaded_capability_ids`][pydantic_ai.tools.RunContext.loaded_capability_ids].
    """

    @cached_property
    def function_signature(self) -> FunctionSignature:
        """The function signature shape for this tool.

        Lazily computed from `parameters_json_schema` and `return_schema` on first access.
        Name and description are not stored on the signature — pass them at render time
        via `sig.render(body, name=td.name, description=td.description)`.
        """
        return FunctionSignature.from_schema(
            name=self.name,
            parameters_schema=self.parameters_json_schema,
            return_schema=self.return_schema,
        )

    def render_signature(self, body: str, **kwargs: Any) -> str:
        """Render the function signature with this tool's name and description.

        Convenience wrapper around `self.function_signature.render()` that
        supplies `name` and `description` from this tool definition.
        """
        return self.function_signature.render(body, name=self.name, description=self.description, **kwargs)

    @property
    def defer(self) -> bool:
        """Whether calls to this tool will be deferred.

        See the [tools documentation](../deferred-tools.md#deferred-tools) for more info.
        """
        return self.kind in ('external', 'unapproved')

    __repr__ = _utils.dataclasses_no_defaults_repr
