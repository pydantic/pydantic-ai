from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union, cast

import pydantic_core
from pydantic import ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import Concatenate, ParamSpec

from . import _pydantic, _utils, messages

AgentDependencies = TypeVar('AgentDependencies')
# retrieval function parameters
P = ParamSpec('P')


@dataclass
class CallContext(Generic[AgentDependencies]):
    """Information about the current call."""

    deps: AgentDependencies
    # do we allow retries within functions?
    retry: int


# Usage `RetrieverContextFunc[AgentDependencies, P]`
RetrieverContextFunc = Callable[Concatenate[CallContext[AgentDependencies], P], Union[str, Awaitable[str]]]
# Usage `RetrieverPlainFunc[P]`
RetrieverPlainFunc = Callable[P, Union[str, Awaitable[str]]]
# Usage `RetrieverEitherFunc[AgentDependencies, P]`
RetrieverEitherFunc = Union[RetrieverContextFunc[AgentDependencies, P], RetrieverPlainFunc[P]]


class Retry(Exception):
    """Exception raised when a retriever function should be retried."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


@dataclass(init=False)
class Retriever(Generic[AgentDependencies, P]):
    """A retriever function for an agent."""

    name: str
    description: str
    function: RetrieverEitherFunc[AgentDependencies, P]
    is_async: bool
    takes_ctx: bool
    single_arg_name: str | None
    positional_fields: list[str]
    var_positional_field: str | None
    validator: SchemaValidator
    json_schema: _utils.ObjectJsonSchema
    max_retries: int
    _current_retry: int = 0

    def __init__(self, function: RetrieverEitherFunc[AgentDependencies, P], takes_ctx: bool, retries: int):
        """Build a Retriever dataclass from a function."""
        f = _pydantic.function_schema(function, takes_ctx)
        self.name = function.__name__
        self.description = f['description']
        self.function = function
        self.is_async = inspect.iscoroutinefunction(function)
        self.takes_ctx = takes_ctx
        self.single_arg_name = f['single_arg_name']
        self.positional_fields = f['positional_fields']
        self.var_positional_field = f['var_positional_field']
        self.validator = f['validator']
        self.json_schema = f['json_schema']
        self.max_retries = retries

    def reset(self) -> None:
        """Reset the current retry count."""
        self._current_retry = 0

    async def run(self, deps: AgentDependencies, message: messages.FunctionCall) -> messages.Message:
        """Run the retriever function asynchronously."""
        try:
            args_dict = self.validator.validate_json(message.arguments)
        except ValidationError as e:
            return self._on_error(e.errors(), message)

        args, kwargs = self._call_args(deps, args_dict)
        try:
            if self.is_async:
                response_content = await self.function(*args, **kwargs)  # type: ignore[reportCallIssue]
            else:
                response_content = await _utils.run_in_executor(
                    self.function,  # type: ignore[reportArgumentType]
                    *args,
                    **kwargs,
                )
        except Retry as e:
            return self._on_error(e.message, message)

        self._current_retry = 0
        return messages.FunctionReturn(
            function_id=message.function_id,
            function_name=message.function_name,
            content=cast(str, response_content),
        )

    def _call_args(self, deps: AgentDependencies, args_dict: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        if self.single_arg_name:
            args_dict = {self.single_arg_name: args_dict}

        args = [CallContext(deps, self._current_retry)] if self.takes_ctx else []
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict

    def _on_error(
        self, content: list[pydantic_core.ErrorDetails] | str, call_message: messages.FunctionCall
    ) -> messages.FunctionRetry:
        self._current_retry += 1
        if self._current_retry > self.max_retries:
            # TODO custom error with details of the retriever
            raise
        else:
            return messages.FunctionRetry(
                function_id=call_message.function_id,
                function_name=call_message.function_name,
                content=content,
            )
