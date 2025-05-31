from __future__ import annotations as _annotations

import dataclasses
import json
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union

from opentelemetry.trace import Tracer
from pydantic import ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Concatenate, ParamSpec, TypeAlias, TypeVar

from . import _function_schema, _utils, messages as _messages
from .exceptions import ModelRetry, UnexpectedModelBehavior

if TYPE_CHECKING:
    from .models import Model
    from .result import Usage

__all__ = (
    'AgentDepsT',
    'DocstringFormat',
    'RunContext',
    'SystemPromptFunc',
    'ToolFuncContext',
    'ToolFuncPlain',
    'ToolFuncEither',
    'ToolParams',
    'ToolPrepareFunc',
    'ToolsPrepareFunc',
    'Tool',
    'ObjectJsonSchema',
    'ToolDefinition',
)

AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
"""Type variable for agent dependencies."""


@dataclasses.dataclass(repr=False)
class RunContext(Generic[AgentDepsT]):
    """Information about the current call."""

    deps: AgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: Usage
    """LLM usage associated with the run."""
    prompt: str | Sequence[_messages.UserContent] | None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries so far."""
    run_step: int = 0
    """The current step in the run."""

    def replace_with(
        self, retry: int | None = None, tool_name: str | None | _utils.Unset = _utils.UNSET
    ) -> RunContext[AgentDepsT]:
        # Create a new `RunContext` a new `retry` value and `tool_name`.
        kwargs = {}
        if retry is not None:
            kwargs['retry'] = retry
        if tool_name is not _utils.UNSET:  # pragma: no branch
            kwargs['tool_name'] = tool_name
        return dataclasses.replace(self, **kwargs)

    __repr__ = _utils.dataclasses_no_defaults_repr


ToolParams = ParamSpec('ToolParams', default=...)
"""Retrieval function param spec."""

SystemPromptFunc = Union[
    Callable[[RunContext[AgentDepsT]], str],
    Callable[[RunContext[AgentDepsT]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]
"""A function that may or maybe not take `RunContext` as an argument, and may or may not be async.

Usage `SystemPromptFunc[AgentDepsT]`.
"""

ToolFuncContext = Callable[Concatenate[RunContext[AgentDepsT], ToolParams], Any]
"""A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDepsT, ToolParams]`.
"""
ToolFuncPlain = Callable[ToolParams, Any]
"""A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.
"""
ToolFuncEither = Union[ToolFuncContext[AgentDepsT, ToolParams], ToolFuncPlain[ToolParams]]
"""Either kind of tool function.

This is just a union of [`ToolFuncContext`][pydantic_ai.tools.ToolFuncContext] and
[`ToolFuncPlain`][pydantic_ai.tools.ToolFuncPlain].

Usage `ToolFuncEither[AgentDepsT, ToolParams]`.
"""
ToolPrepareFunc: TypeAlias = 'Callable[[RunContext[AgentDepsT], ToolDefinition], Awaitable[ToolDefinition | None]]'
"""Definition of a function that can prepare a tool definition at call time.

See [tool docs](../tools.md#tool-prepare) for more information.

Example — here `only_if_42` is valid as a `ToolPrepareFunc`:

```python {noqa="I001"}
from typing import Union

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
    if ctx.deps == 42:
        return tool_def

def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'

hitchhiker = Tool(hitchhiker, prepare=only_if_42)
```

Usage `ToolPrepareFunc[AgentDepsT]`.
"""

ToolsPrepareFunc: TypeAlias = (
    'Callable[[RunContext[AgentDepsT], list[ToolDefinition]], Awaitable[list[ToolDefinition] | None]]'
)
"""Definition of a function that can prepare the tool definition of all tools for each step.
This is useful if you want to customize the definition of multiple tools or you want to register
a subset of tools for a given step.

Example — here `turn_on_strict_if_openai` is valid as a `ToolsPrepareFunc`:

```python {noqa="I001"}
from dataclasses import replace
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition


async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs

agent = Agent('openai:gpt-4o', prepare_tools=turn_on_strict_if_openai)
```

Usage `ToolsPrepareFunc[AgentDepsT]`.
"""


DocstringFormat = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats.

* `'google'` — [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
* `'numpy'` — [Numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings.
* `'sphinx'` — [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) docstrings.
* `'auto'` — Automatically infer the format based on the structure of the docstring.
"""

A = TypeVar('A')


class GenerateToolJsonSchema(GenerateJsonSchema):
    def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> JsonSchemaValue:
        s = super().typed_dict_schema(schema)
        total = schema.get('total')
        if 'additionalProperties' not in s and (total is True or total is None):
            s['additionalProperties'] = False
        return s

    def _named_required_fields_schema(self, named_required_fields: Sequence[tuple[str, bool, Any]]) -> JsonSchemaValue:
        # Remove largely-useless property titles
        s = super()._named_required_fields_schema(named_required_fields)
        for p in s.get('properties', {}):
            s['properties'][p].pop('title', None)
        return s


@dataclass(init=False)
class Tool(Generic[AgentDepsT]):
    """A tool function for an agent."""

    function: ToolFuncEither[AgentDepsT]
    takes_ctx: bool
    max_retries: int | None
    name: str
    description: str
    prepare: ToolPrepareFunc[AgentDepsT] | None
    docstring_format: DocstringFormat
    require_parameter_descriptions: bool
    strict: bool | None
    function_schema: _function_schema.FunctionSchema
    """
    The base JSON schema for the tool's parameters.

    This schema may be modified by the `prepare` function or by the Model class prior to including it in an API request.
    """

    # TODO: Move this state off the Tool class, which is otherwise stateless.
    #   This should be tracked inside a specific agent run, not the tool.
    current_retry: int = field(default=0, init=False)

    def __init__(
        self,
        function: ToolFuncEither[AgentDepsT],
        *,
        takes_ctx: bool | None = None,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
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
        from typing import Union

        from pydantic_ai import Agent, RunContext, Tool
        from pydantic_ai.tools import ToolDefinition

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        async def prep_my_tool(
            ctx: RunContext[int], tool_def: ToolDefinition
        ) -> Union[ToolDefinition, None]:
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
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            function_schema: The function schema to use for the tool. If not provided, it will be generated.
        """
        self.function = function
        self.function_schema = function_schema or _function_schema.function_schema(
            function,
            schema_generator,
            takes_ctx=takes_ctx,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
        )
        self.takes_ctx = self.function_schema.takes_ctx
        self.max_retries = max_retries
        self.name = name or function.__name__
        self.description = description or self.function_schema.description
        self.prepare = prepare
        self.docstring_format = docstring_format
        self.require_parameter_descriptions = require_parameter_descriptions
        self.strict = strict

    async def prepare_tool_def(self, ctx: RunContext[AgentDepsT]) -> ToolDefinition | None:
        """Get the tool definition.

        By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
        if it's set.

        Returns:
            return a `ToolDefinition` or `None` if the tools should not be registered for this run.
        """
        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.function_schema.json_schema,
            strict=self.strict,
        )
        if self.prepare is not None:
            return await self.prepare(ctx, tool_def)
        else:
            return tool_def

    async def run(
        self, message: _messages.ToolCallPart, run_context: RunContext[AgentDepsT], tracer: Tracer
    ) -> _messages.ToolReturnPart | _messages.RetryPromptPart:
        """Run the tool function asynchronously.

        This method wraps `_run` in an OpenTelemetry span.

        See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>.
        """
        span_attributes = {
            'gen_ai.tool.name': self.name,
            # NOTE: this means `gen_ai.tool.call.id` will be included even if it was generated by pydantic-ai
            'gen_ai.tool.call.id': message.tool_call_id,
            'tool_arguments': message.args_as_json_str(),
            'logfire.msg': f'running tool: {self.name}',
            # add the JSON schema so these attributes are formatted nicely in Logfire
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        'tool_arguments': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                }
            ),
        }
        with tracer.start_as_current_span('running tool', attributes=span_attributes):
            return await self._run(message, run_context)

    async def _run(
        self, message: _messages.ToolCallPart, run_context: RunContext[AgentDepsT]
    ) -> _messages.ToolReturnPart | _messages.RetryPromptPart:
        try:
            validator = self.function_schema.validator
            if isinstance(message.args, str):
                args_dict = validator.validate_json(message.args or '{}')
            else:
                args_dict = validator.validate_python(message.args or {})
        except ValidationError as e:
            return self._on_error(e, message)

        ctx = dataclasses.replace(
            run_context,
            retry=self.current_retry,
            tool_name=message.tool_name,
            tool_call_id=message.tool_call_id,
        )
        try:
            response_content = await self.function_schema.call(args_dict, ctx)
        except ModelRetry as e:
            return self._on_error(e, message)

        self.current_retry = 0
        return _messages.ToolReturnPart(
            tool_name=message.tool_name,
            content=response_content,
            tool_call_id=message.tool_call_id,
        )

    def _on_error(
        self, exc: ValidationError | ModelRetry, call_message: _messages.ToolCallPart
    ) -> _messages.RetryPromptPart:
        self.current_retry += 1
        if self.max_retries is None or self.current_retry > self.max_retries:
            raise UnexpectedModelBehavior(f'Tool exceeded max retries count of {self.max_retries}') from exc
        else:
            if isinstance(exc, ValidationError):
                content = exc.errors(include_url=False, include_context=False)
            else:
                content = exc.message
            return _messages.RetryPromptPart(
                tool_name=call_message.tool_name,
                content=content,
                tool_call_id=call_message.tool_call_id,
            )


ObjectJsonSchema: TypeAlias = dict[str, Any]
"""Type representing JSON schema of an object, e.g. where `"type": "object"`.

This type is used to define tools parameters (aka arguments) in [ToolDefinition][pydantic_ai.tools.ToolDefinition].

With PEP-728 this should be a TypedDict with `type: Literal['object']`, and `extra_parts=Any`
"""


@dataclass(repr=False)
class ToolDefinition:
    """Definition of a tool passed to a model.

    This is used for both function tools and output tools.
    """

    name: str
    """The name of the tool."""

    description: str
    """The description of the tool."""

    parameters_json_schema: ObjectJsonSchema
    """The JSON schema for the tool's parameters."""

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

    Note: this is currently only supported by OpenAI models.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr
