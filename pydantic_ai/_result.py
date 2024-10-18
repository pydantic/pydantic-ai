from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, Union, cast

from . import _utils, messages
from .result import ResultData
from .shared import AgentDeps, CallContext, ModelRetry

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
        except ModelRetry as r:
            m = messages.ToolRetry(
                tool_name=tool_call.tool_name,
                content=r.message,
                tool_id=tool_call.tool_id,
            )
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    """Internal exception used to indicate a signal a `ToolRetry` message should be returned to the LLM"""

    def __init__(self, tool_retry: messages.ToolRetry):
        self.tool_retry = tool_retry
        super().__init__()
