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
from .result import DEFAULT_OUTPUT_TOOL_NAME, OutputDataT, OutputDataT_inv, OutputValidatorFunc, ToolOutput
from .tools import AgentDepsT, GenerateToolJsonSchema, RunContext, ToolDefinition

T = TypeVar('T')
"""An invariant TypeVar."""


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

    tools: dict[str, OutputSchemaTool[OutputDataT]]
    allow_text_output: bool

    @classmethod
    def build(
        cls: type[OutputSchema[T]],
        output_type: type[T] | ToolOutput[T],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[T] | None:
        """Build an OutputSchema dataclass from a response type."""
        if output_type is str:
            return None

        if isinstance(output_type, ToolOutput):
            # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
            name = output_type.name
            description = output_type.description
            output_type_ = output_type.output_type
            strict = output_type.strict
        else:
            output_type_ = output_type

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
                tools[tool_name] = cast(
                    OutputSchemaTool[T],
                    OutputSchemaTool(
                        output_type=arg, name=tool_name, description=description, multiple=True, strict=strict
                    ),
                )
        else:
            name = name or DEFAULT_OUTPUT_TOOL_NAME
            tools[name] = cast(
                OutputSchemaTool[T],
                OutputSchemaTool(
                    output_type=output_type_, name=name, description=description, multiple=False, strict=strict
                ),
            )

        return cls(tools=tools, allow_text_output=allow_text_output)

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputSchemaTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:  # pragma: no branch
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputSchemaTool[OutputDataT]]]:
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


DEFAULT_DESCRIPTION = 'The final response which ends this conversation'


@dataclass(init=False)
class OutputSchemaTool(Generic[OutputDataT]):
    tool_def: ToolDefinition
    type_adapter: TypeAdapter[Any]

    def __init__(
        self, *, output_type: type[OutputDataT], name: str, description: str | None, multiple: bool, strict: bool | None
    ):
        """Build a OutputSchemaTool from a response type."""
        if _utils.is_model_like(output_type):
            self.type_adapter = TypeAdapter(output_type)
            outer_typed_dict_key: str | None = None
            # noinspection PyArgumentList
            parameters_json_schema = _utils.check_object_json_schema(
                self.type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )
        else:
            response_data_typed_dict = TypedDict(  # noqa: UP013
                'response_data_typed_dict',
                {'response': output_type},  # pyright: ignore[reportInvalidTypeForm]
            )
            self.type_adapter = TypeAdapter(response_data_typed_dict)
            outer_typed_dict_key = 'response'
            # noinspection PyArgumentList
            parameters_json_schema = _utils.check_object_json_schema(
                self.type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )
            # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
            parameters_json_schema.pop('title')

        if json_schema_description := parameters_json_schema.pop('description', None):
            if description is None:
                tool_description = json_schema_description
            else:
                tool_description = f'{description}. {json_schema_description}'  # pragma: no cover
        else:
            tool_description = description or DEFAULT_DESCRIPTION
            if multiple:
                tool_description = f'{union_arg_name(output_type)}: {tool_description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=tool_description,
            parameters_json_schema=parameters_json_schema,
            outer_typed_dict_key=outer_typed_dict_key,
            strict=strict,
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
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(tool_call.args, str):
                output = self.type_adapter.validate_json(tool_call.args, experimental_allow_partial=pyd_allow_partial)
            else:
                output = self.type_adapter.validate_python(tool_call.args, experimental_allow_partial=pyd_allow_partial)
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
        else:
            if k := self.tool_def.outer_typed_dict_key:
                output = output[k]
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
