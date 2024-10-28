from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, Union, cast, get_args

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypedDict

from . import _pydantic, _utils, messages
from .messages import LLMToolCalls, ToolCall
from .shared import AgentDeps, CallContext, ModelRetry, ResultData

# A function that always takes `ResultData` and returns `ResultData`,
# but may or maybe not take `CallInfo` as a first argument, and may or may not be async.
# Usage `ResultValidator[AgentDeps, ResultData]`
ResultValidatorFunc = Union[
    Callable[[CallContext[AgentDeps], ResultData], ResultData],
    Callable[[CallContext[AgentDeps], ResultData], Awaitable[ResultData]],
    Callable[[ResultData], ResultData],
    Callable[[ResultData], Awaitable[ResultData]],
]


@dataclass
class ResultValidator(Generic[AgentDeps, ResultData]):
    function: ResultValidatorFunc[AgentDeps, ResultData]
    _takes_ctx: bool = False
    _is_async: bool = False

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
        if args := union_args(response_type):
            for arg in args:
                tool_name = union_tool_name(name, arg)
                tools[tool_name] = _build_tool(arg, tool_name, True)
        else:
            tools[name] = _build_tool(response_type, name, False)

        return cls(tools=tools, allow_text_result=allow_text_result)

    def find_tool(self, message: LLMToolCalls) -> tuple[ToolCall, ResultTool[ResultData]] | None:
        """Find a tool that matches one of the calls."""
        for call in message.calls:
            if result := self.tools.get(call.tool_name):
                return call, result


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

    def validate(self, tool_call: messages.ToolCall) -> ResultData:
        """Validate a result message.

        Returns:
            Either the validated result data (left) or a retry message (right).
        """
        try:
            if isinstance(tool_call.args, messages.ArgsJson):
                result = self.type_adapter.validate_json(tool_call.args.args_json)
            else:
                result = self.type_adapter.validate_python(tool_call.args.args_object)
        except ValidationError as e:
            m = messages.RetryPrompt(
                tool_name=tool_call.tool_name,
                content=e.errors(include_url=False),
                tool_id=tool_call.tool_id,
            )
            raise ToolRetryError(m) from e
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
    if _pydantic.is_union(response_type) and any(t is str for t in get_args(response_type)):
        remain_args: list[Any] = []
        includes_str = False
        for arg in get_args(response_type):
            if arg is str:
                includes_str = True
            else:
                remain_args.append(arg)
        if includes_str:
            if len(remain_args) == 1:
                return _utils.Some(remain_args[0])
            else:
                return _utils.Some(Union[tuple(remain_args)])


def union_args(response_type: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty union."""
    if _pydantic.is_union(response_type):
        return get_args(response_type)
    else:
        return ()
