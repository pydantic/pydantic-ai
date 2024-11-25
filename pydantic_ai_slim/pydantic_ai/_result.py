from __future__ import annotations as _annotations

import inspect
import sys
import types
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Union, cast, get_args, get_origin

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypeAliasType, TypedDict

from . import _utils, messages
from .dependencies import AgentDeps, CallContext, ResultValidatorFunc
from .exceptions import ModelRetry
from .messages import ModelStructuredResponse, ToolCall
from .result import ResultData


@dataclass
class ResultValidator(Generic[AgentDeps, ResultData]):
    function: ResultValidatorFunc[AgentDeps, ResultData]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self, result: ResultData, deps: AgentDeps, retry: int, tool_call: messages.ToolCall | None
    ) -> ResultData:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            deps: The agent dependencies.
            retry: The current retry number.
            tool_call: The original tool call message, `None` if there was no tool call.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            args = CallContext(deps, retry, tool_call.tool_name if tool_call else None), result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[ResultData]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], ResultData], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            m = messages.RetryPrompt(content=r.message)
            if tool_call is not None:
                m.tool_name = tool_call.tool_name
                m.tool_id = tool_call.tool_id
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    """Internal exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: messages.RetryPrompt):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass
class ResultSchema(Generic[ResultData]):
    """Model the final response from an agent run.

    Similar to `Retriever` but for the final result of running an agent.
    """

    tools: dict[str, ResultTool[ResultData]]
    allow_text_result: bool

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str | None) -> Self | None:
        """Build a ResultSchema dataclass from a response type."""
        if response_type is str:
            return None

        if response_type_option := extract_str_from_union(response_type):
            response_type = response_type_option.value
            allow_text_result = True
        else:
            allow_text_result = False

        def _build_tool(a: Any, tool_name_: str, multiple: bool) -> ResultTool[ResultData]:
            return cast(
                ResultTool[ResultData],
                ResultTool.build(a, tool_name_, description, multiple),  # pyright: ignore[reportUnknownMemberType]
            )

        tools: dict[str, ResultTool[ResultData]] = {}
        if args := get_union_args(response_type):
            for i, arg in enumerate(args, start=1):
                tool_name = union_tool_name(name, arg)
                while tool_name in tools:
                    tool_name = f'{tool_name}_{i}'
                tools[tool_name] = _build_tool(arg, tool_name, True)
        else:
            tools[name] = _build_tool(response_type, name, False)

        return cls(tools=tools, allow_text_result=allow_text_result)

    def find_tool(self, message: ModelStructuredResponse) -> tuple[ToolCall, ResultTool[ResultData]] | None:
        """Find a tool that matches one of the calls."""
        for call in message.calls:
            if result := self.tools.get(call.tool_name):
                return call, result

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())


DEFAULT_DESCRIPTION = 'The final response which ends this conversation'


@dataclass
class ResultTool(Generic[ResultData]):
    name: str
    description: str
    type_adapter: TypeAdapter[Any]
    json_schema: _utils.ObjectJsonSchema
    outer_typed_dict_key: str | None

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str | None, multiple: bool) -> Self | None:
        """Build a ResultTool dataclass from a response type."""
        assert response_type is not str, 'ResultTool does not support str as a response type'

        if _utils.is_model_like(response_type):
            type_adapter = TypeAdapter(response_type)
            outer_typed_dict_key: str | None = None
            # noinspection PyArgumentList
            json_schema = _utils.check_object_json_schema(type_adapter.json_schema())
        else:
            response_data_typed_dict = TypedDict('response_data_typed_dict', {'response': response_type})  # noqa
            type_adapter = TypeAdapter(response_data_typed_dict)
            outer_typed_dict_key = 'response'
            # noinspection PyArgumentList
            json_schema = _utils.check_object_json_schema(type_adapter.json_schema())
            # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
            json_schema.pop('title')

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                tool_description = json_schema_description
            else:
                tool_description = f'{description}. {json_schema_description}'
        else:
            tool_description = description or DEFAULT_DESCRIPTION
            if multiple:
                tool_description = f'{union_arg_name(response_type)}: {tool_description}'

        return cls(
            name=name,
            description=tool_description,
            type_adapter=type_adapter,
            json_schema=json_schema,
            outer_typed_dict_key=outer_typed_dict_key,
        )

    def validate(
        self, tool_call: messages.ToolCall, allow_partial: bool = False, wrap_validation_errors: bool = True
    ) -> ResultData:
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
            if isinstance(tool_call.args, messages.ArgsJson):
                result = self.type_adapter.validate_json(
                    tool_call.args.args_json or '', experimental_allow_partial=pyd_allow_partial
                )
            else:
                result = self.type_adapter.validate_python(
                    tool_call.args.args_object, experimental_allow_partial=pyd_allow_partial
                )
        except ValidationError as e:
            if wrap_validation_errors:
                m = messages.RetryPrompt(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False),
                    tool_id=tool_call.tool_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise
        else:
            if k := self.outer_typed_dict_key:
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
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty union."""
    if isinstance(tp, TypeAliasType):
        tp = tp.__value__

    origin = get_origin(tp)
    if origin_is_union(origin):
        return get_args(tp)
    else:
        return ()


if sys.version_info < (3, 10):

    def origin_is_union(tp: type[Any] | None) -> bool:
        return tp is Union

else:

    def origin_is_union(tp: type[Any] | None) -> bool:
        return tp is Union or tp is types.UnionType
