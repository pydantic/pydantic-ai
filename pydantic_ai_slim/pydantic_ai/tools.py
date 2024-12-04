from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union, cast

from pydantic import ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import Concatenate, ParamSpec, final

from . import _pydantic, _utils, messages
from .exceptions import ModelRetry, UnexpectedModelBehavior

if TYPE_CHECKING:
    from .result import ResultData
else:
    ResultData = Any


__all__ = (
    'AgentDeps',
    'RunContext',
    'ResultValidatorFunc',
    'SystemPromptFunc',
    'ToolFuncContext',
    'ToolFuncPlain',
    'ToolFuncEither',
    'ToolParams',
    'Tool',
)

AgentDeps = TypeVar('AgentDeps')
"""Type variable for agent dependencies."""


@dataclass
class RunContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    """Dependencies for the agent."""
    retry: int
    """Number of retries so far."""
    tool_name: str | None
    """Name of the tool being called."""


ToolParams = ParamSpec('ToolParams')
"""Retrieval function param spec."""

SystemPromptFunc = Union[
    Callable[[RunContext[AgentDeps]], str],
    Callable[[RunContext[AgentDeps]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]
"""A function that may or maybe not take `RunContext` as an argument, and may or may not be async.

Usage `SystemPromptFunc[AgentDeps]`.
"""

ResultValidatorFunc = Union[
    Callable[[RunContext[AgentDeps], ResultData], ResultData],
    Callable[[RunContext[AgentDeps], ResultData], Awaitable[ResultData]],
    Callable[[ResultData], ResultData],
    Callable[[ResultData], Awaitable[ResultData]],
]
"""
A function that always takes `ResultData` and returns `ResultData`,
but may or maybe not take `CallInfo` as a first argument, and may or may not be async.

Usage `ResultValidator[AgentDeps, ResultData]`.
"""

ToolFuncContext = Callable[Concatenate[RunContext[AgentDeps], ToolParams], Any]
"""A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDeps, ToolParams]`.
"""
ToolFuncPlain = Callable[ToolParams, Any]
"""A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.
"""
ToolFuncEither = Union[ToolFuncContext[AgentDeps, ToolParams], ToolFuncPlain[ToolParams]]
"""Either kind of tool function.

This is just a union of [`ToolFuncContext`][pydantic_ai.tools.ToolFuncContext] and
[`ToolFuncPlain`][pydantic_ai.tools.ToolFuncPlain].

Usage `ToolFuncEither[AgentDeps, ToolParams]`.
"""

A = TypeVar('A')


@final
@dataclass(init=False)
class Tool(Generic[AgentDeps]):
    """A tool function for an agent."""

    function: ToolFuncEither[AgentDeps, ...]
    takes_ctx: bool
    max_retries: int | None
    name: str
    description: str
    _is_async: bool = field(init=False)
    _single_arg_name: str | None = field(init=False)
    _positional_fields: list[str] = field(init=False)
    _var_positional_field: str | None = field(init=False)
    _validator: SchemaValidator = field(init=False, repr=False)
    _json_schema: _utils.ObjectJsonSchema = field(init=False)
    _current_retry: int = field(default=0, init=False)

    def __init__(
        self,
        function: ToolFuncEither[AgentDeps, ...],
        takes_ctx: bool,
        *,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
    ):
        """Create a new tool instance.

        Example usage:

        ```py
        from pydantic_ai import Agent, RunContext, Tool

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        agent = Agent('test', tools=[Tool(my_tool, True)])
        ```

        Args:
            function: The Python function to call as the tool.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument.
            max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
            name: Name of the tool, inferred from the function if `None`.
            description: Description of the tool, inferred from the function if `None`.
        """
        f = _pydantic.function_schema(function, takes_ctx)
        self.function = function
        self.takes_ctx = takes_ctx
        self.max_retries = max_retries
        self.name = name or function.__name__
        self.description = description or f['description']
        self._is_async = inspect.iscoroutinefunction(self.function)
        self._single_arg_name = f['single_arg_name']
        self._positional_fields = f['positional_fields']
        self._var_positional_field = f['var_positional_field']
        self._validator = f['validator']
        self._json_schema = f['json_schema']

    @staticmethod
    def infer(function: ToolFuncEither[A, ...] | Tool[A]) -> Tool[A]:
        """Create a tool from a pure function, inferring whether it takes `RunContext` as its first argument.

        Args:
            function: The tool function to wrap; or for convenience, a `Tool` instance.

        Returns:
            A new `Tool` instance.
        """
        if isinstance(function, Tool):
            return function
        else:
            return Tool(function, takes_ctx=_pydantic.takes_ctx(function))

    def reset(self) -> None:
        """Reset the current retry count."""
        self._current_retry = 0

    async def run(self, deps: AgentDeps, message: messages.ToolCall) -> messages.Message:
        """Run the tool function asynchronously."""
        try:
            if isinstance(message.args, messages.ArgsJson):
                args_dict = self._validator.validate_json(message.args.args_json)
            else:
                args_dict = self._validator.validate_python(message.args.args_dict)
        except ValidationError as e:
            return self._on_error(e, message)

        args, kwargs = self._call_args(deps, args_dict, message)
        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[str]], self.function)
                response_content = await function(*args, **kwargs)
            else:
                function = cast(Callable[[Any], str], self.function)
                response_content = await _utils.run_in_executor(function, *args, **kwargs)
        except ModelRetry as e:
            return self._on_error(e, message)

        self._current_retry = 0
        return messages.ToolReturn(
            tool_name=message.tool_name,
            content=response_content,
            tool_id=message.tool_id,
        )

    @property
    def json_schema(self) -> _utils.ObjectJsonSchema:
        return self._json_schema

    @property
    def outer_typed_dict_key(self) -> str | None:
        return None

    def _call_args(
        self, deps: AgentDeps, args_dict: dict[str, Any], message: messages.ToolCall
    ) -> tuple[list[Any], dict[str, Any]]:
        if self._single_arg_name:
            args_dict = {self._single_arg_name: args_dict}

        args = [RunContext(deps, self._current_retry, message.tool_name)] if self.takes_ctx else []
        for positional_field in self._positional_fields:
            args.append(args_dict.pop(positional_field))
        if self._var_positional_field:
            args.extend(args_dict.pop(self._var_positional_field))

        return args, args_dict

    def _on_error(self, exc: ValidationError | ModelRetry, call_message: messages.ToolCall) -> messages.RetryPrompt:
        self._current_retry += 1
        if self.max_retries is None or self._current_retry > self.max_retries:
            raise UnexpectedModelBehavior(f'Tool exceeded max retries count of {self.max_retries}') from exc
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
