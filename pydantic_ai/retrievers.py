from __future__ import annotations as _annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Concatenate, Generic, ParamSpec, Self, TypeVar, cast

from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaValidator

from . import _pydantic, _utils

AgentContext = TypeVar('AgentContext')
# retrieval function parameters
P = ParamSpec('P')


@dataclass
class CallInfo(Generic[AgentContext]):
    """Information about the current call."""

    context: AgentContext
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
    retries: int

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
            retries=retries,
        )

    async def async_run(self, call_info: CallInfo[AgentContext], json_data: str | bytes) -> str:
        """Run the retriever function asynchronously."""
        kwargs = self._call_kwargs(json_data)
        args = (call_info,) if self.takes_info else ()
        if self.is_async:
            return await self.function(*args, **kwargs)  # type: ignore[reportCallIssue]
        else:
            f = cast(Callable[Concatenate[CallInfo[AgentContext], P], str], self.function)
            return await _utils.run_in_executor(f, *args, **kwargs)  # type: ignore[reportCallIssue]

    def _call_kwargs(self, json_data: str | bytes) -> dict[str, Any]:
        kwargs = self.validator.validate_json(json_data)
        if self.single_arg_name:
            return {self.single_arg_name: kwargs}
        else:
            return kwargs
