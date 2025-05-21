from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Union, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import TypeAliasType, TypedDict, TypeVar, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from . import _function_schema, _utils, messages as _messages
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
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'


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


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use tools for outputs, and customize the tool."""

    output_type: SimpleOutputType[OutputDataT]
    name: str | None
    description: str | None
    max_retries: int | None
    strict: bool | None

    def __init__(
        self,
        *,
        type_: SimpleOutputType[OutputDataT],
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output_type = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


T_co = TypeVar('T_co', covariant=True)
# output_type=Type or output_type=function or output_type=object.method
SimpleOutputType = TypeAliasType(
    'SimpleOutputType', Union[type[T_co], Callable[..., Union[T_co, Awaitable[T_co]]]], type_params=(T_co,)
)
# output_type=ToolOutput(<see above>) or <see above>
SimpleOutputTypeOrMarker = TypeAliasType(
    'SimpleOutputTypeOrMarker', Union[SimpleOutputType[T_co], ToolOutput[T_co]], type_params=(T_co,)
)
# output_type=<see above> or [<see above>, ...]
OutputType = TypeAliasType(
    'OutputType', Union[SimpleOutputTypeOrMarker[T_co], Sequence[SimpleOutputTypeOrMarker[T_co]]], type_params=(T_co,)
)


@dataclass
class OutputSchema(Generic[OutputDataT]):
    """Model the final output from an agent run.

    Similar to `Tool` but for the final output of running an agent.
    """

    tools: dict[str, OutputTool[OutputDataT]]
    allow_text_output: bool

    @classmethod
    def build(
        cls: type[OutputSchema[OutputDataT]],
        output_type: OutputType[OutputDataT],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[OutputDataT] | None:
        """Build an OutputSchema dataclass from an output type."""
        if output_type is str:
            return None

        multiple = False

        output_types_or_markers: Sequence[SimpleOutputTypeOrMarker[OutputDataT]]
        if isinstance(output_type, Sequence):
            output_types_or_markers = output_type
            multiple = True
        else:
            output_types_or_markers = [output_type]

        allow_text_output = False
        tools: dict[str, OutputTool[OutputDataT]] = {}
        for output_type_or_marker in output_types_or_markers:
            tool_name = name
            tool_description = description
            tool_strict = strict
            custom_tool_name = False
            if isinstance(output_type_or_marker, ToolOutput):
                output_type_ = output_type_or_marker.output_type
                # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
                tool_name = output_type_or_marker.name
                tool_description = output_type_or_marker.description
                tool_strict = output_type_or_marker.strict
                custom_tool_name = True
            elif output_type_other_than_str := extract_str_from_union(output_type_or_marker):
                output_type_ = output_type_other_than_str.value
                allow_text_output = True
            else:
                output_type_ = output_type_or_marker

            disambiguate_tool_name = multiple and not custom_tool_name
            if args := get_union_args(output_type_):
                multiple = True
                disambiguate_tool_name = True
            else:
                args = (output_type_,)

            base_tool_name = tool_name
            for arg in args:
                tool_name = base_tool_name or DEFAULT_OUTPUT_TOOL_NAME

                if disambiguate_tool_name:
                    tool_name += f'_{arg.__name__}'

                i = 1
                original_tool_name = tool_name
                while tool_name in tools:
                    tool_name = f'{original_tool_name}_{i}'
                    i += 1

                parameters_schema = OutputObjectSchema(
                    output_type=arg, description=tool_description, strict=tool_strict
                )
                tools[tool_name] = cast(
                    OutputTool[OutputDataT],
                    OutputTool(name=tool_name, parameters_schema=parameters_schema, multiple=multiple),
                )

        return cls(
            tools=tools,
            allow_text_output=allow_text_output,
        )

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:  # pragma: no branch
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputTool[OutputDataT]]]:
        """Find a tool that matches one of the calls."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if result := self.tools.get(part.tool_name):
                    yield part, result

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())

    def tool_defs(self) -> list[ToolDefinition]:
        """Get tool definitions to register with the model."""
        return [t.tool_def for t in self.tools.values()]


def allow_text_output(output_schema: OutputSchema[Any] | None) -> bool:
    return output_schema is None or output_schema.allow_text_output


@dataclass
class OutputObjectDefinition:
    name: str
    json_schema: ObjectJsonSchema
    description: str | None = None
    strict: bool | None = None


@dataclass(init=False)
class OutputObjectSchema(Generic[OutputDataT]):
    definition: OutputObjectDefinition
    validator: SchemaValidator
    function_schema: _function_schema.FunctionSchema | None = None
    outer_typed_dict_key: str | None = None

    def __init__(
        self,
        *,
        output_type: SimpleOutputType[OutputDataT],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if inspect.isfunction(output_type) or inspect.ismethod(output_type):
            self.function_schema = _function_schema.function_schema(output_type, GenerateToolJsonSchema)
            self.validator = self.function_schema.validator
            json_schema = self.function_schema.json_schema
        else:
            type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output_type):
                type_adapter = TypeAdapter(output_type)
            else:
                self.outer_typed_dict_key = 'response'
                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': cast(type[OutputDataT], output_type)},  # pyright: ignore[reportInvalidTypeForm]
                )
                type_adapter = TypeAdapter(response_data_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self.validator = cast(SchemaValidator, type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
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

    async def process(
        self,
        data: str | dict[str, Any],
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function.

        Args:
            data: The output data to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        if isinstance(data, str):
            output = self.validator.validate_json(data, allow_partial=pyd_allow_partial)
        else:
            output = self.validator.validate_python(data, allow_partial=pyd_allow_partial)

        if self.function_schema:
            output = await self.function_schema.call(output, run_context)

        if k := self.outer_typed_dict_key:
            output = output[k]
        return output


@dataclass(init=False)
class OutputTool(Generic[OutputDataT]):
    parameters_schema: OutputObjectSchema[OutputDataT]
    tool_def: ToolDefinition

    def __init__(self, *, name: str, parameters_schema: OutputObjectSchema[OutputDataT], multiple: bool):
        self.parameters_schema = parameters_schema
        definition = parameters_schema.definition

        description = definition.description
        if not description:
            description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
            if multiple:
                description = f'{definition.name}: {description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema=definition.json_schema,
            strict=definition.strict,
            outer_typed_dict_key=parameters_schema.outer_typed_dict_key,
        )

    async def process(
        self,
        tool_call: _messages.ToolCallPart,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message.

        Args:
            tool_call: The tool call from the LLM to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = await self.parameters_schema.process(tool_call.args, run_context, allow_partial=allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False),
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=r.message,
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover
        else:
            return output


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
        if includes_str:  # pragma: no branch
            if len(remain_args) == 1:
                return _utils.Some(remain_args[0])
            else:
                return _utils.Some(Union[tuple(remain_args)])  # pragma: no cover


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `output_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()
