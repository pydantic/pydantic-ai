from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from typing import Annotated, Any, Concatenate, Generic, Literal, TypeAlias, Union, cast

from pydantic import Discriminator, Tag
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import ParamSpec, Self, TypeVar

from . import _function_schema, _utils
from ._run_context import AgentDepsT, RunContext
from .builtin_tools import AbstractBuiltinTool
from .exceptions import ModelRetry
from .function_signature import FunctionSignature
from .messages import RetryPromptPart, ToolCallPart, ToolReturn

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
    'AgentBuiltinTool',
    'BuiltinToolFunc',
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
"""A function that may or maybe not take `RunContext` as an argument, and may or may not be async.

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
    Awaitable['list[ToolDefinition] | None'] | list['ToolDefinition'] | None,
]
"""Definition of a function that can prepare the tool definition of all tools for each step.
This is useful if you want to customize the definition of multiple tools or you want to register
a subset of tools for a given step. Both sync and async functions are accepted.

Example — here `turn_on_strict_if_openai` is valid as a `ToolsPrepareFunc`:

```python {noqa="I001"}
from dataclasses import replace

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition


def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs

agent = Agent('openai:gpt-5.2', prepare_tools=turn_on_strict_if_openai)
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


BuiltinToolFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]], Awaitable[AbstractBuiltinTool | None] | AbstractBuiltinTool | None
]
"""Definition of a function that can prepare a builtin tool at call time.

This is useful if you want to customize the builtin tool based on the run context (e.g. user dependencies),
or omit it completely from a step.
"""

AgentBuiltinTool: TypeAlias = AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]
"""A builtin tool or a function that dynamically produces one.

This is a convenience alias for `AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]`.
"""

DocstringFormat: TypeAlias = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats.

* `'google'` — [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
* `'numpy'` — [Numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings.
* `'sphinx'` — [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) docstrings.
* `'auto'` — Automatically infer the format based on the structure of the docstring.
"""


@dataclass(kw_only=True)
class DeferredToolRequests:
    """Tool calls that require approval or external execution.

    This can be used as an agent's `output_type` and will be used as the output of the agent run if the model called any deferred tools.

    Results can be passed to the next agent run using a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object with the same tool call IDs.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: list[ToolCallPart] = field(default_factory=list[ToolCallPart])
    """Tool calls that require external execution."""
    approvals: list[ToolCallPart] = field(default_factory=list[ToolCallPart])
    """Tool calls that require human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Metadata for deferred tool calls, keyed by `tool_call_id`."""


@dataclass(kw_only=True)
class ToolApproved:
    """Indicates that a tool call has been approved and that the tool function should be executed."""

    override_args: dict[str, Any] | None = None
    """Optional tool call arguments to use instead of the original arguments."""

    kind: Literal['tool-approved'] = 'tool-approved'


@dataclass
class ToolDenied:
    """Indicates that a tool call has been denied and that a denial message should be returned to the model."""

    message: str = 'The tool call was denied.'
    """The message to return to the model."""

    _: KW_ONLY

    kind: Literal['tool-denied'] = 'tool-denied'


def _deferred_tool_call_result_discriminator(x: Any) -> str | None:
    if isinstance(x, dict):
        if 'kind' in x:
            return cast(str, x['kind'])
        elif 'part_kind' in x:
            return cast(str, x['part_kind'])
    else:
        if hasattr(x, 'kind'):
            return cast(str, x.kind)
        elif hasattr(x, 'part_kind'):
            return cast(str, x.part_kind)
    return None


DeferredToolApprovalResult: TypeAlias = Annotated[ToolApproved | ToolDenied, Discriminator('kind')]
"""Result for a tool call that required human-in-the-loop approval."""
DeferredToolCallResult: TypeAlias = Annotated[
    Annotated[ToolReturn, Tag('tool-return')]
    | Annotated[ModelRetry, Tag('model-retry')]
    | Annotated[RetryPromptPart, Tag('retry-prompt')],
    Discriminator(_deferred_tool_call_result_discriminator),
]
"""Result for a tool call that required external execution."""
DeferredToolResult = DeferredToolApprovalResult | DeferredToolCallResult
"""Result for a tool call that required approval or external execution."""


@dataclass(kw_only=True)
class DeferredToolResults:
    """Results for deferred tool calls from a previous run that required approval or external execution.

    The tool call IDs need to match those from the [`DeferredToolRequests`][pydantic_ai.output.DeferredToolRequests] output object from the previous run.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: dict[str, DeferredToolCallResult | Any] = field(default_factory=dict[str, DeferredToolCallResult | Any])
    """Map of tool call IDs to results for tool calls that required external execution."""
    approvals: dict[str, bool | DeferredToolApprovalResult] = field(
        default_factory=dict[str, bool | DeferredToolApprovalResult]
    )
    """Map of tool call IDs to results for tool calls that required human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Metadata for deferred tool calls, keyed by `tool_call_id`. Each value will be available in the tool's RunContext as `tool_call_metadata`."""


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
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
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
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
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
    """Whether this tool requires a sequential/serial execution environment."""

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

    For MCP tools, this contains the `meta`, `annotations`, and `output_schema` fields from the tool definition.
    """

    timeout: float | None = None
    """Timeout in seconds for tool execution.

    If the tool takes longer than this, a retry prompt is returned to the model.
    Defaults to None (no timeout).
    """

    defer_loading: bool = False
    """Whether this tool should be hidden from the model until discovered via tool search.

    See [Tool Search](../tools-advanced.md#tool-search) for more info.
    """

    prefer_builtin: str | None = None
    """If set, this function tool is a local fallback for the builtin tool with the given unique_id.

    When the model supports the corresponding builtin tool natively, this function tool is
    removed from the request. When the model does not support the builtin, the builtin is
    removed and this function tool stays.
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
