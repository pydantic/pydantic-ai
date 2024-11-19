from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, cast

from pydantic import ValidationError
from pydantic_core import SchemaValidator

from . import _pydantic, _utils, messages
from .dependencies import AgentDeps, CallContext, RetrieverContextFunc, RetrieverParams, RetrieverPlainFunc
from .exceptions import ModelRetry, UnexpectedModelBehavior

# Usage `RetrieverEitherFunc[AgentDependencies, P]`
RetrieverEitherFunc = _utils.Either[
    RetrieverContextFunc[AgentDeps, RetrieverParams], RetrieverPlainFunc[RetrieverParams]
]


@dataclass(init=False)
class Retriever(Generic[AgentDeps, RetrieverParams]):
    """A retriever function for an agent."""

    name: str
    description: str
    function: RetrieverEitherFunc[AgentDeps, RetrieverParams] = field(repr=False)
    is_async: bool
    single_arg_name: str | None
    positional_fields: list[str]
    var_positional_field: str | None
    validator: SchemaValidator = field(repr=False)
    json_schema: _utils.ObjectJsonSchema
    max_retries: int
    _current_retry: int = 0
    outer_typed_dict_key: str | None = None

    def __init__(self, function: RetrieverEitherFunc[AgentDeps, RetrieverParams], retries: int):
        """Build a Retriever dataclass from a function."""
        self.function = function
        # noinspection PyTypeChecker
        f = _pydantic.function_schema(function)
        raw_function = function.whichever()
        self.name = raw_function.__name__
        self.description = f['description']
        self.is_async = inspect.iscoroutinefunction(raw_function)
        self.single_arg_name = f['single_arg_name']
        self.positional_fields = f['positional_fields']
        self.var_positional_field = f['var_positional_field']
        self.validator = f['validator']
        self.json_schema = f['json_schema']
        self.max_retries = retries

    def reset(self) -> None:
        """Reset the current retry count."""
        self._current_retry = 0

    async def run(self, deps: AgentDeps, message: messages.ToolCall) -> messages.Message:
        """Run the retriever function asynchronously."""
        try:
            if isinstance(message.args, messages.ArgsJson):
                args_dict = self.validator.validate_json(message.args.args_json)
            else:
                args_dict = self.validator.validate_python(message.args.args_object)
        except ValidationError as e:
            return self._on_error(e, message)

        args, kwargs = self._call_args(deps, args_dict, message)
        try:
            if self.is_async:
                function = cast(Callable[[Any], Awaitable[str]], self.function.whichever())
                response_content = await function(*args, **kwargs)
            else:
                function = cast(Callable[[Any], str], self.function.whichever())
                response_content = await _utils.run_in_executor(function, *args, **kwargs)
        except ModelRetry as e:
            return self._on_error(e, message)

        self._current_retry = 0
        return messages.ToolReturn(
            tool_name=message.tool_name,
            content=response_content,
            tool_id=message.tool_id,
        )

    def _call_args(
        self, deps: AgentDeps, args_dict: dict[str, Any], message: messages.ToolCall
    ) -> tuple[list[Any], dict[str, Any]]:
        if self.single_arg_name:
            args_dict = {self.single_arg_name: args_dict}

        args = [CallContext(deps, self._current_retry, message.tool_name)] if self.function.is_left() else []
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict

    def _on_error(self, exc: ValidationError | ModelRetry, call_message: messages.ToolCall) -> messages.RetryPrompt:
        self._current_retry += 1
        if self._current_retry > self.max_retries:
            # TODO custom error with details of the retriever
            raise UnexpectedModelBehavior(f'Retriever exceeded max retries count of {self.max_retries}') from exc
        else:
            if isinstance(exc, ValidationError):
                content = exc.errors(include_url=False)
            else:
                content = exc.message
            return messages.RetryPrompt(
                tool_name=call_message.tool_name,
                content=content,
                tool_id=call_message.tool_id,
            )
