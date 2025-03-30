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
from .result import ResultDataT, ResultDataT_inv, ResultValidatorFunc
from .tools import AgentDepsT, RunContext, ToolDefinition

T = TypeVar('T')
"""An invariant TypeVar."""


@dataclass
class ResultValidator(Generic[AgentDepsT, ResultDataT_inv]):
    function: ResultValidatorFunc[AgentDepsT, ResultDataT_inv]
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
class ResultSchema(Generic[ResultDataT]):
    """Model the final response from an agent run.

    Similar to `Tool` but for the final result of running an agent.
    """

    tools: dict[str, ResultTool[ResultDataT]]
    allow_text_result: bool

    @classmethod
    def build(
        cls: type[ResultSchema[T]], response_type: type[T], name: str, description: str | None
    ) -> ResultSchema[T] | None:
        """Build a ResultSchema dataclass from a response type."""
        if response_type is str:
            return None

        if response_type_option := extract_str_from_union(response_type):
            response_type = response_type_option.value
            allow_text_result = True
        else:
            allow_text_result = False

        def _build_tool(a: Any, tool_name_: str, multiple: bool) -> ResultTool[T]:
            return cast(ResultTool[T], ResultTool(a, tool_name_, description, multiple))

        tools: dict[str, ResultTool[T]] = {}
        if args := get_union_args(response_type):
            for i, arg in enumerate(args, start=1):
                tool_name = union_tool_name(name, arg)
                while tool_name in tools:
                    tool_name = f'{tool_name}_{i}'
                tools[tool_name] = _build_tool(arg, tool_name, True)
        else:
            tools[name] = _build_tool(response_type, name, False)

        return cls(tools=tools, allow_text_result=allow_text_result)

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, ResultTool[ResultDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, ResultTool[ResultDataT]]]:
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


DEFAULT_DESCRIPTION = 'The final response which ends this conversation'


@dataclass(init=False)
class ResultTool(Generic[ResultDataT]):
    tool_def: ToolDefinition
    type_adapter: TypeAdapter[Any]

    def __init__(self, response_type: type[ResultDataT], name: str, description: str | None, multiple: bool):
        """Build a ResultTool dataclass from a response type."""
        assert response_type is not str, 'ResultTool does not support str as a response type'

        if _utils.is_model_like(response_type):
            self.type_adapter = TypeAdapter(response_type)
            outer_typed_dict_key: str | None = None
            # noinspection PyArgumentList
            parameters_json_schema = _utils.check_object_json_schema(self.type_adapter.json_schema())
        else:
            response_data_typed_dict = TypedDict('response_data_typed_dict', {'response': response_type})  # noqa
            self.type_adapter = TypeAdapter(response_data_typed_dict)
            outer_typed_dict_key = 'response'
            # noinspection PyArgumentList
            parameters_json_schema = _utils.check_object_json_schema(self.type_adapter.json_schema())
            # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
            parameters_json_schema.pop('title')

        if json_schema_description := parameters_json_schema.pop('description', None):
            if description is None:
                tool_description = json_schema_description
            else:
                tool_description = f'{description}. {json_schema_description}'
        else:
            tool_description = description or DEFAULT_DESCRIPTION
            if multiple:
                tool_description = f'{union_arg_name(response_type)}: {tool_description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=tool_description,
            parameters_json_schema=parameters_json_schema,
            outer_typed_dict_key=outer_typed_dict_key,
        )

    def validate(
        self, tool_call: _messages.ToolCallPart, allow_partial: bool = False, wrap_validation_errors: bool = True
    ) -> ResultDataT:
        """Validate a result message.

        Args:
            tool_call: The tool call from the LLM to validate.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated result data (left) or a retry message (right).
        """
        try:
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(tool_call.args, str):
                result = self.type_adapter.validate_json(tool_call.args, experimental_allow_partial=pyd_allow_partial)
            else:
                result = self.type_adapter.validate_python(tool_call.args, experimental_allow_partial=pyd_allow_partial)
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
            if k := self.tool_def.outer_typed_dict_key:
                result = result[k]
            return result


def union_tool_name(base_name: str, union_arg: Any) -> str:
    return f'{base_name}_{union_arg_name(union_arg)}'


def union_arg_name(union_arg: Any) -> str:
    return union_arg.__name__


def extract_str_from_union(response_type: Any) -> _utils.Option[Any]:
    """Extract the string type from a Union, return the remaining union or remaining type."""
    union_args = get_union_args(response_type)
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
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()
