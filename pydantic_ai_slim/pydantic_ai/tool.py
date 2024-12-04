from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union, cast

from pydantic import ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import Concatenate, ParamSpec, TypeAlias

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
    'ToolReturnValue',
    'ToolFuncContext',
    'ToolFuncPlain',
    'ToolFuncEither',
    'ToolParams',
    'JsonData',
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

JsonData: TypeAlias = 'None | str | int | float | Sequence[JsonData] | Mapping[str, JsonData]'
"""Type representing any JSON data."""

ToolReturnValue = Union[JsonData, Awaitable[JsonData]]
"""Return value of a tool function."""
ToolFuncContext = Callable[Concatenate[RunContext[AgentDeps], ToolParams], ToolReturnValue]
"""A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDeps, ToolParams]`.
"""
ToolFuncPlain = Callable[ToolParams, ToolReturnValue]
"""A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.
"""
ToolFuncEither = Union[ToolFuncContext[AgentDeps, ToolParams], ToolFuncPlain[ToolParams]]
"""Either kind of tool function.

This is just a union of [`ToolFuncContext`][pydantic_ai.dependencies.ToolFuncContext] and
[`ToolFuncPlain`][pydantic_ai.dependencies.ToolFuncPlain].

Usage `ToolFuncEither[AgentDeps, ToolParams]`.
"""


@dataclass
class Tool(Generic[AgentDeps]):
    """A tool function for an agent."""

    function: ToolFuncEither[AgentDeps, ...]
    """The Python function to call as the tool."""
    takes_ctx: bool
    """Whether the function takes a [`RunContext`][pydantic_ai.dependencies.RunContext] first argument."""
    max_retries: int = 1
    """Maximum number of retries allowed for this tool."""
    name: str = ''
    """Name of the tool, inferred from the function if left blank."""
    description: str = ''
    """Description of the tool, inferred from the function if left blank."""
    _is_async: bool = field(init=False)
    _single_arg_name: str | None = field(init=False)
    _positional_fields: list[str] = field(init=False)
    _var_positional_field: str | None = field(init=False)
    _validator: SchemaValidator = field(init=False, repr=False)
    _json_schema: _utils.ObjectJsonSchema = field(init=False)
    _current_retry: int = field(default=0, init=False)

    def __post_init__(self):
        f = _pydantic.function_schema(self.function, self.takes_ctx)
        self.name = self.name or self.function.__name__
        self.description = self.description or f['description']
        self._is_async = inspect.iscoroutinefunction(self.function)
        self._single_arg_name = f['single_arg_name']
        self._positional_fields = f['positional_fields']
        self._var_positional_field = f['var_positional_field']
        self._validator = f['validator']
        self._json_schema = f['json_schema']

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
        if self._current_retry > self.max_retries:
            # TODO custom error with details of the tool
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
