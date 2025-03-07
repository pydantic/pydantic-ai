from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import partial
from inspect import iscoroutinefunction
from typing import Any, Callable, Concatenate, Generic, Literal, NotRequired, cast

import anyio.to_thread
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic._internal import _typing_extra
from typing_extensions import TypedDict, TypeVar, assert_type

from pydantic_evals.scoring import ScoringContext

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass
class AssertionResult:
    """The result of running an assertion."""

    check: SerializedAssertion

    passed: bool
    """Whether the assertion passed."""

    reason: str | None = None
    """The reason the assertion failed. Will be the error message if an exception is provided"""


AssertionFunctionResult = None | bool | tuple[bool, str]

SyncAssertionFunction = Callable[Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...], AssertionFunctionResult]
AsyncAssertionFunction = Callable[
    Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...], Awaitable[AssertionFunctionResult]
]
AssertionFunction = (
    SyncAssertionFunction[InputsT, OutputT, MetadataT] | AsyncAssertionFunction[InputsT, OutputT, MetadataT]
)

AssertionRegistry = Mapping[str, AssertionFunction[InputsT, OutputT, MetadataT]]

SyncBoundAssertionFunction = Callable[[ScoringContext[InputsT, OutputT, MetadataT]], AssertionFunctionResult]
AsyncBoundAssertionFunction = Callable[
    [ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[AssertionFunctionResult]
]
BoundAssertionFunction = (
    SyncBoundAssertionFunction[InputsT, OutputT, MetadataT] | AsyncBoundAssertionFunction[InputsT, OutputT, MetadataT]
)


@dataclass
class Assertion(Generic[InputsT, OutputT, MetadataT]):
    """An assertion to be run on a scoring context."""

    data: SerializedAssertion

    function: BoundAssertionFunction[InputsT, OutputT, MetadataT]

    async def check(self, ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> AssertionResult:
        try:
            if inspect.iscoroutinefunction(self.function):
                async_function = cast(AsyncBoundAssertionFunction[InputsT, OutputT, MetadataT], self.function)
                result = await async_function(ctx)
            else:
                sync_function = cast(SyncBoundAssertionFunction[InputsT, OutputT, MetadataT], self.function)
                result = await anyio.to_thread.run_sync(sync_function, ctx)
            if isinstance(result, bool):
                assert result  # raise an AssertionError if we got a bool
        except Exception as e:
            return AssertionResult(self.data, passed=False, reason=f'{type(e).__name__}: {str(e)}')

        if isinstance(result, tuple):
            return AssertionResult(self.data, passed=result[0], reason=result[1])
        else:
            assert_type(result, Literal[True] | None)
            # In this case, no exception was raised, so the check passes
            return AssertionResult(self.data, passed=True)


class SerializedAssertion(BaseModel, extra='allow'):
    """A serialized assertion call. Requires a registry to be deserialized."""

    call: str
    """The name of the assertion function to call.

    The registered function should accept a ScoringContext[InputsT, OutputT, MetadataT] as its first argument.
    Any additional required arguments must have values provided in `kwargs`.
    The function should return a boolean indicating whether the assertion passed.

    The call must be registered on the `Evaluation` prior to the call to `run`.
    """

    def __init__(self, call: str, **kwargs: Any):
        super().__init__(call=call, **kwargs)

    def bind(self, registry: AssertionRegistry[InputsT, OutputT, MetadataT]) -> Assertion[InputsT, OutputT, MetadataT]:
        # Note: validate_kwargs should have been called prior to this
        function = registry[self.call]
        if iscoroutinefunction(function):
            bound_async_function = cast(
                AsyncBoundAssertionFunction[InputsT, OutputT, MetadataT],
                partial(function, **(self.__pydantic_extra__ or {})),
            )
            return Assertion[InputsT, OutputT, MetadataT](data=self, function=bound_async_function)
        else:
            bound_sync_function = cast(
                SyncBoundAssertionFunction[InputsT, OutputT, MetadataT],
                partial(function, **(self.__pydantic_extra__ or {})),
            )
            return Assertion[InputsT, OutputT, MetadataT](data=self, function=bound_sync_function)

    def validate_against_registry(self, registry: AssertionRegistry[Any, Any, Any], case_name: str) -> None:
        function = registry.get(self.call)
        if function is None:
            raise ValueError(
                f'Assertion call {self.call!r} is not registered. Registered choices: {list(registry.keys())}'
            )

        signature = inspect.signature(function)
        first_param = list(signature.parameters.values())[0]
        type_hints = _typing_extra.get_function_type_hints(function)
        type_hints.pop(first_param.name)
        type_hints.pop('return', None)
        for p in signature.parameters.values():
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]

        td = TypedDict(function.__name__, type_hints)  # type: ignore
        td.__pydantic_config__ = {'extra': 'forbid'}  # type: ignore
        adapter = TypeAdapter(td)
        try:
            adapter.validate_python(self.__pydantic_extra__ or {})
        except ValidationError as e:
            error_message = str(e)
            error_message = error_message.replace(' for typed-dict', ':')
            raise ValueError(f'Assertion {self.call!r} in case {case_name!r} has {error_message}')
