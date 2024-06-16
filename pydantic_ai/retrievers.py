from __future__ import annotations as _annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Concatenate, Generic, ParamSpec, Self, TypeVar, cast

from pydantic import ValidationError
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaValidator

from . import _pydantic, _utils, messages

AgentContext = TypeVar('AgentContext')
# retrieval function parameters
P = ParamSpec('P')


@dataclass
class CallInfo(Generic[AgentContext]):
    """Information about the current call."""

    context: AgentContext
    # do we allow retries within functions?
    retry: int


# Usage `RetrieverFunc[AgentContext, P]`
RetrieverFunc = Callable[Concatenate[CallInfo[AgentContext], P], str | Awaitable[str]]


@dataclass
class Retriever(Generic[AgentContext, P]):
    """A retriever function for an agent."""

    name: str
    description: str
    function: RetrieverFunc[AgentContext, P]
    is_async: bool
    takes_info: bool
    single_arg_name: str | None
    validator: SchemaValidator
    json_schema: JsonSchemaValue
    max_retries: int
    _current_retry: int = 0

    @classmethod
    def build(cls, function: RetrieverFunc[AgentContext, P], retries: int) -> Self:
        """Build a Retriever dataclass from a function."""
        f = _pydantic.function_schema(function)
        return cls(
            name=function.__name__,
            description=f['description'],
            function=function,
            is_async=inspect.iscoroutinefunction(function),
            takes_info=f['takes_info'],
            single_arg_name=f['single_arg_name'],
            validator=f['validator'],
            json_schema=f['json_schema'],
            max_retries=retries,
        )

    def reset(self) -> None:
        """Reset the current retry count."""
        self._current_retry = 0

    async def run(self, context: AgentContext, message: messages.FunctionCall) -> messages.Message:
        """Run the retriever function asynchronously."""
        try:
            kwargs = self._call_kwargs(message['arguments'])
        except ValidationError as e:
            self._current_retry += 1
            if self._current_retry > self.max_retries:
                # TODO custom error
                raise
            else:
                return messages.FunctionValidationError(
                    role='function-validation-error',
                    timestamp=datetime.now(),
                    function_id=message['function_id'],
                    function_name=message['function_name'],
                    errors=e.errors(),
                )

        self._current_retry = 0
        args = (CallInfo(context, self._current_retry),) if self.takes_info else ()
        if self.is_async:
            response_content = await self.function(*args, **kwargs)  # type: ignore[reportCallIssue]
        else:
            response_content = await _utils.run_in_executor(self.function, *args, **kwargs)  # type: ignore[reportCallIssue]

        return messages.FunctionResponse(
            role='function-response',
            timestamp=datetime.now(),
            function_id=message['function_id'],
            function_name=message['function_name'],
            content=cast(str, response_content),
        )

    def _call_kwargs(self, json_arguments: str) -> dict[str, Any]:
        kwargs = self.validator.validate_json(json_arguments)
        if self.single_arg_name:
            return {self.single_arg_name: kwargs}
        else:
            return kwargs
