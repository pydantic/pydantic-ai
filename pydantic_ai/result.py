from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self, TypedDict

from . import _utils, messages
from .call import AgentDeps, CallContext, Retry
from .messages import ArgsJson

ResultData = TypeVar('ResultData')


@dataclass
class Cost:
    """Cost of a run."""

    total_cost: int


@dataclass
class RunResult(Generic[ResultData]):
    """Result of a run."""

    response: ResultData
    message_history: list[messages.Message]
    cost: Cost

    def message_history_json(self) -> str:
        """Return the history of messages as a JSON string."""
        return messages.MessagesTypeAdapter.dump_json(self.message_history).decode()


@dataclass
class ResultSchema(Generic[ResultData]):
    """Model the final response from an agent run.

    Similar to `Retriever` but for the final result of running an agent.
    """

    name: str
    description: str
    type_adapter: TypeAdapter[Any]
    json_schema: _utils.ObjectJsonSchema
    allow_text_result: bool
    outer_typed_dict: bool

    @classmethod
    def build(cls, response_type: type[ResultData], name: str, description: str) -> Self | None:
        """Build a ResultSchema dataclass from a response type."""
        if response_type is str:
            return None

        if _utils.is_model_like(response_type):
            type_adapter = TypeAdapter(response_type)
            outer_typed_dict = False
        else:
            # noinspection PyTypedDict
            response_data_typed_dict = TypedDict('response_data_typed_dict', {'response': response_type})  # noqa
            type_adapter = TypeAdapter(response_data_typed_dict)
            outer_typed_dict = True

        return cls(
            name=name,
            description=description,
            type_adapter=type_adapter,
            json_schema=_utils.check_object_json_schema(type_adapter.json_schema()),
            allow_text_result=_utils.allow_plain_str(response_type),
            outer_typed_dict=outer_typed_dict,
        )

    def validate(self, tool_call: messages.ToolCall) -> ResultData:
        """Validate a result message.

        Returns:
            Either the validated result data (left) or a retry message (right).
        """
        try:
            if isinstance(tool_call.args, ArgsJson):
                result = self.type_adapter.validate_json(tool_call.args.args_json)
            else:
                result = self.type_adapter.validate_python(tool_call.args.args_object)
        except ValidationError as e:
            m = messages.ToolRetry(
                tool_name=tool_call.tool_name,
                content=e.errors(include_url=False),
                tool_id=tool_call.tool_id,
            )
            raise ToolRetryError(m) from e
        else:
            if self.outer_typed_dict:
                result = result['response']
            return result


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
        self, result: ResultData, deps: AgentDeps, retry: int, tool_call: messages.ToolCall
    ) -> ResultData:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            deps: The agent dependencies.
            retry: The current retry number.
            tool_call: The original tool call message.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            args = CallContext(deps, retry), result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[ResultData]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], ResultData], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except Retry as r:
            m = messages.ToolRetry(
                tool_name=tool_call.tool_name,
                content=r.message,
                tool_id=tool_call.tool_id,
            )
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    def __init__(self, tool_retry: messages.ToolRetry):
        self.tool_retry = tool_retry
        super().__init__()
