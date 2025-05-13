from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Union, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict, TypeVar, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from . import _utils, messages as _messages
from .exceptions import ModelRetry
from .tools import AgentDepsT, GenerateToolJsonSchema, ObjectJsonSchema, RunContext, ToolDefinition

T = TypeVar('T')
"""An invariant TypeVar."""
OutputDataT_inv = TypeVar('OutputDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `OutputValidator` and `OutputValidatorFunc` because the output data type is used
in both the input and output of a `OutputValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing OutputValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to OutputValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""
OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the result data type of a run."""

OutputValidatorFunc = Union[
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv],
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]],
    Callable[[OutputDataT_inv], OutputDataT_inv],
    Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]],
]
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""

DEFAULT_OUTPUT_TOOL_NAME = 'final_result'


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use tools for outputs, and customize the tool."""

    output_type: type[OutputDataT]
    # TODO: Add `output_call` support, for calling a function to get the output
    # output_call: Callable[..., OutputDataT] | None
    name: str
    description: str | None
    max_retries: int | None
    strict: bool | None

    def __init__(
        self,
        *,
        type_: type[OutputDataT],
        # call: Callable[..., OutputDataT] | None = None,
        name: str = 'final_result',
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output_type = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict

        # TODO: add support for call and make type_ optional, with the following logic:
        # if type_ is None and call is None:
        #     raise ValueError('Either type_ or call must be provided')
        # if call is not None:
        #     if type_ is None:
        #         type_ = get_type_hints(call).get('return')
        #         if type_ is None:
        #             raise ValueError('Unable to determine type_ from call signature; please provide it explicitly')
        # self.output_call = call


@dataclass(init=False)
class StructuredOutput(Generic[OutputDataT]):
    """Marker class to use structured output for outputs."""

    output_type: type[OutputDataT]
    name: str | None
    description: str | None
    strict: bool | None

    def __init__(
        self,
        *,
        type_: type[OutputDataT],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self.output_type = type_
        self.name = name
        self.description = description
        self.strict = strict


@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        result: T,
        tool_call: _messages.ToolCallPart | None,
        run_context: RunContext[AgentDepsT],
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            tool_call: The original tool call message, `None` if there was no tool call.
            run_context: The current run context.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            ctx = run_context.replace_with(tool_name=tool_call.tool_name if tool_call else None)
            args = ctx, result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[T]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], T], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            m = _messages.RetryPromptPart(content=r.message)
            if tool_call is not None:
                m.tool_name = tool_call.tool_name
                m.tool_call_id = tool_call.tool_call_id
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    """Internal exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: _messages.RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass
class OutputSchema(Generic[OutputDataT]):
    """Model the final response from an agent run.

    Similar to `Tool` but for the final output of running an agent.
    """

    # TODO: Since this is currently called "preferred", models that don't have structured output implemented yet ignore it and use tools (except for Mistral).
    # We should likely raise an error if an unsupported mode is used, _and_ allow the model to pick its own preferred mode if none is forced.
    preferred_mode: Literal['tool', 'structured'] | None  # TODO: Add mode for manual JSON
    output_object_schema: OutputObjectSchema[OutputDataT]
    tools: dict[str, OutputSchemaTool[OutputDataT]]
    allow_text_output: bool  # TODO: Verify structured output works correctly with string as a union member

    @classmethod
    def build(
        cls: type[OutputSchema[T]],
        output_type: type[T] | ToolOutput[T] | StructuredOutput[T],  # TODO: Support a list of output types/markers
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[T] | None:
        """Build an OutputSchema dataclass from a response type."""
        if output_type is str:
            return None

        preferred_mode = None
        if isinstance(output_type, ToolOutput):
            # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
            name = output_type.name
            description = output_type.description
            output_type_ = output_type.output_type
            strict = output_type.strict
            preferred_mode = 'tool'
        elif isinstance(output_type, StructuredOutput):
            name = output_type.name
            description = output_type.description
            output_type_ = output_type.output_type
            strict = output_type.strict
            preferred_mode = 'structured'
        else:
            output_type_ = output_type

        output_object_schema = OutputObjectSchema(
            output_type=output_type_, name=name, description=description, strict=strict
        )

        # No need to include an output tool for string output
        if output_type_option := extract_str_from_union(output_type):
            output_type_ = output_type_option.value
            allow_text_output = True
        else:
            allow_text_output = False

        tools: dict[str, OutputSchemaTool[T]] = {}
        if args := get_union_args(output_type_):
            for i, arg in enumerate(args, start=1):
                tool_name = raw_tool_name = union_tool_name(name, arg)
                while tool_name in tools:
                    tool_name = f'{raw_tool_name}_{i}'

                parameters_object_schema = OutputObjectSchema(output_type=arg, description=description, strict=strict)
                tools[tool_name] = cast(
                    OutputSchemaTool[T],
                    OutputSchemaTool(name=tool_name, parameters_object_schema=parameters_object_schema, multiple=True),
                )
        else:
            tool_name = name or DEFAULT_OUTPUT_TOOL_NAME
            parameters_object_schema = OutputObjectSchema(
                output_type=output_type_, description=description, strict=strict
            )
            tools[tool_name] = cast(
                OutputSchemaTool[T],
                OutputSchemaTool(name=tool_name, parameters_object_schema=parameters_object_schema, multiple=False),
            )

        return cls(
            preferred_mode=preferred_mode,
            tools=tools,
            allow_text_output=allow_text_output,
            output_object_schema=output_object_schema,
        )

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputSchemaTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputSchemaTool[OutputDataT]]]:
        """Find a tool that matches one of the calls."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):
                if result := self.tools.get(part.tool_name):
                    yield part, result

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())

    def tool_defs(self) -> list[ToolDefinition]:
        """Get tool definitions to register with the model."""
        return [t.tool_def for t in self.tools.values()]

    def validate(
        self, data: str | dict[str, Any], allow_partial: bool = False, wrap_validation_errors: bool = True
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            data: The output data to validate.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        return self.output_object_schema.validate(
            data, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )

    # TODO: Build instructions for manual JSON


DEFAULT_DESCRIPTION = 'The final response which ends this conversation'


@dataclass
class OutputObjectDefinition:
    name: str
    json_schema: ObjectJsonSchema
    description: str | None = None
    strict: bool | None = None


@dataclass(init=False)
class OutputObjectSchema(Generic[OutputDataT]):
    definition: OutputObjectDefinition
    type_adapter: TypeAdapter[Any]
    outer_typed_dict_key: str | None = None

    def __init__(
        self,
        *,
        output_type: type[OutputDataT],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if _utils.is_model_like(output_type):
            self.type_adapter = TypeAdapter(output_type)
        else:
            self.outer_typed_dict_key = 'response'
            response_data_typed_dict = TypedDict(  # noqa: UP013
                'response_data_typed_dict',
                {'response': output_type},  # pyright: ignore[reportInvalidTypeForm]
            )
            self.type_adapter = TypeAdapter(response_data_typed_dict)

        json_schema = _utils.check_object_json_schema(
            self.type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
        )
        if self.outer_typed_dict_key:
            # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
            json_schema.pop('title')

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                description = json_schema_description
            else:
                description = f'{description}. {json_schema_description}'

        self.definition = OutputObjectDefinition(
            name=name or getattr(output_type, '__name__', DEFAULT_OUTPUT_TOOL_NAME),
            description=description,
            json_schema=json_schema,
            strict=strict,
        )

    def validate(
        self, data: str | dict[str, Any], allow_partial: bool = False, wrap_validation_errors: bool = True
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            data: The output data to validate.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(data, str):
                output = self.type_adapter.validate_json(data, experimental_allow_partial=pyd_allow_partial)
            else:
                output = self.type_adapter.validate_python(data, experimental_allow_partial=pyd_allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise
        else:
            if k := self.outer_typed_dict_key:
                output = output[k]
            return output


@dataclass(init=False)
class OutputSchemaTool(Generic[OutputDataT]):
    parameters_object_schema: OutputObjectSchema[OutputDataT]
    tool_def: ToolDefinition

    def __init__(self, *, name: str, parameters_object_schema: OutputObjectSchema[OutputDataT], multiple: bool):
        self.parameters_object_schema = parameters_object_schema
        definition = parameters_object_schema.definition

        description = definition.description
        if not description:
            description = DEFAULT_DESCRIPTION
            if multiple:
                description = f'{definition.name}: {description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema=definition.json_schema,
            strict=definition.strict,
            outer_typed_dict_key=parameters_object_schema.outer_typed_dict_key,
        )

    def validate(
        self, tool_call: _messages.ToolCallPart, allow_partial: bool = False, wrap_validation_errors: bool = True
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            tool_call: The tool call from the LLM to validate.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = self.parameters_object_schema.validate(
                tool_call.args, allow_partial=allow_partial, wrap_validation_errors=False
            )
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False),
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise
        else:
            return output


def union_tool_name(base_name: str | None, union_arg: Any) -> str:
    return f'{base_name or DEFAULT_OUTPUT_TOOL_NAME}_{union_arg_name(union_arg)}'


def union_arg_name(union_arg: Any) -> str:
    return union_arg.__name__


def extract_str_from_union(output_type: Any) -> _utils.Option[Any]:
    """Extract the string type from a Union, return the remaining union or remaining type."""
    union_args = get_union_args(output_type)
    if any(t is str for t in union_args):
        remain_args: list[Any] = []
        includes_str = False
        for arg in union_args:
            if arg is str:
                includes_str = True
            else:
                remain_args.append(arg)
        if includes_str:
            if len(remain_args) == 1:
                return _utils.Some(remain_args[0])
            else:
                return _utils.Some(Union[tuple(remain_args)])


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `output_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()
