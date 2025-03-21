from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Concatenate, Generic, NotRequired, cast

import anyio.to_thread
from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    RootModel,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic._internal import _typing_extra
from pydantic_core.core_schema import SerializerFunctionWrapHandler
from typing_extensions import Self, TypedDict, TypeVar

from .._utils import get_unwrapped_function_name
from .context import ScoringContext

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


class AssessmentSpec(BaseModel):
    """The specification of an assessment to be run.

    Note that special "short forms" are supported during serialization/deserialization.

    In particular, each of the following forms is supported for specifying an assessment using a callable known as `my_call`:
    * `'my_call'` Just the (string) name of the call is used if there are no arguments or kwargs to the call other than the scoring context
    * `{'my_call': first_arg}` A single argument is passed as the first argument to the `my_call` assessment
    * `{'my_call': {k1: v1, k2: v2}}` Multiple kwargs are passed to the `my_call` assessment
    * `{'call': 'my_call', 'args': [first_arg], 'kwargs': {k1: v1, k2: v2}}` The full form of the assessment spec, with all arguments and kwargs specified
    """

    call: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[AssessmentSpec]) -> AssessmentSpec:
        try:
            result = handler(value)
            return result
        except ValidationError as exc:
            try:
                deserialized = _SerializedAssessmentSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error; TODO: We should somehow combine the errors
            return deserialized.to_assessment_spec()

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        # In this case, use the standard "long-form" serialization
        if len(self.args) > 1:
            return handler(self)

        # Note: The rest of this logic needs to be kept in sync with the definition of _SerializedAssessmentSpec
        # In this case, use the shortest compatible form of serialization:
        if not self.args and not self.kwargs:
            return self.call
        elif len(self.args) == 1:
            return {self.call: self.args[0]}
        else:
            return {self.call: self.kwargs}


class _SerializedAssessmentSpec(RootModel[str | dict[str, Any]]):
    """The specification of an assessment to be run.

    Corresponds to the serialized form of an assessment call in a yaml file.

    This is just an auxiliary class used to serialize/deserialize instances of AssessmentSpec.
    """

    @field_validator('root')
    @classmethod
    def enforce_one_key(cls, value: str | dict[str, Any]) -> Any:
        if isinstance(value, str):
            return value
        if len(value) != 1:
            raise ValueError(f'Expected a single key containing the call name, found keys {list(value.keys())}')
        return value

    @property
    def _call(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def _args_kwargs(self) -> tuple[list[Any], dict[str, Any]]:
        if isinstance(self.root, str):
            return [], {}  # a plain str means a function call without any arguments
        value = next(iter(self.root.values()))

        if isinstance(value, dict):
            keys: list[Any] = list(value.keys())  # pyright: ignore[reportUnknownArgumentType]
            if all(isinstance(k, str) for k in keys):
                # dict[str, Any]s are treated as kwargs
                return [], cast(dict[str, Any], value)

        # Anything else is passed as the first argument
        return [value], {}

    def to_assessment_spec(self) -> AssessmentSpec:
        call = self._call
        args, kwargs = self._args_kwargs
        return AssessmentSpec(call=call, args=args, kwargs=kwargs)


AssessmentValue = bool | int | float | str


@dataclass
class AssessmentResult:
    """The result of running an assessment."""

    value: AssessmentValue
    reason: str | None = None


AssessmentFunctionResult = AssessmentValue | AssessmentResult | Mapping[str, AssessmentValue | AssessmentResult]


def _convert_to_mapping(
    result: AssessmentFunctionResult, *, scalar_name: str
) -> Mapping[str, AssessmentValue | AssessmentResult]:
    if isinstance(result, Mapping):
        return result
    return {scalar_name: result}


# TODO(DavidM): Add bound=AssessmentValue to the following typevar after we upgrade to pydantic 2.11
AssessmentValueT = TypeVar('AssessmentValueT', default=AssessmentValue, covariant=True)


@dataclass
class AssessmentDetail(Generic[AssessmentValueT]):
    """The details of an assessment that has been run."""

    name: str
    value: AssessmentValueT
    reason: str | None
    source: AssessmentSpec


AssessmentFunction = Callable[
    Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...],
    AssessmentFunctionResult | Awaitable[AssessmentFunctionResult],
]
AssessmentRegistry = Mapping[str, AssessmentFunction[InputsT, OutputT, MetadataT]]
BoundAssessmentFunction = Callable[
    [ScoringContext[InputsT, OutputT, MetadataT]], AssessmentFunctionResult | Awaitable[AssessmentFunctionResult]
]


@dataclass
class Assessment(Generic[InputsT, OutputT, MetadataT]):
    """An assessment to be run on a scoring context."""

    spec: AssessmentSpec
    function: BoundAssessmentFunction[InputsT, OutputT, MetadataT]

    @classmethod
    def from_function(
        cls,
        function: BoundAssessmentFunction[InputsT, OutputT, MetadataT],
    ) -> Self:
        spec = AssessmentSpec(call=get_unwrapped_function_name(function))
        return cls(spec=spec, function=function)

    @classmethod
    def from_registry(
        cls, registry: AssessmentRegistry[InputsT, OutputT, MetadataT], case_name: str | None, spec: AssessmentSpec
    ) -> Self:
        function = Assessment[InputsT, OutputT, MetadataT]._validate_against_registry(registry, case_name, spec)
        bound_function = _bind_assessment_function(function, spec.args, spec.kwargs)
        return cls(spec=spec, function=bound_function)

    async def execute(self, ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> list[AssessmentDetail]:
        if inspect.iscoroutinefunction(self.function):
            results = cast(AssessmentFunctionResult, await self.function(ctx))
        else:
            results = cast(AssessmentFunctionResult, await anyio.to_thread.run_sync(self.function, ctx))

        results = _convert_to_mapping(results, scalar_name=self.spec.call)

        details: list[AssessmentDetail] = []
        for name, result in results.items():
            if not isinstance(result, AssessmentResult):
                result = AssessmentResult(value=result)
            details.append(AssessmentDetail(name=name, value=result.value, reason=result.reason, source=self.spec))

        return details

    @staticmethod
    def _validate_against_registry(
        registry: AssessmentRegistry[InputsT, OutputT, MetadataT], case_name: str | None, spec: AssessmentSpec
    ) -> AssessmentFunction[InputsT, OutputT, MetadataT]:
        # Note: a lot of this code is duplicated in pydantic_evals.datasets.Dataset.model_json_schema_with_assessments,
        # but I don't see a good way to refactor/unify it yet. (Feel free to try!)

        function = registry.get(spec.call)
        if function is None:
            raise ValueError(
                f'Assessment call {spec.call!r} is not registered. Registered choices: {list(registry.keys())}'
            )

        signature = inspect.signature(function)

        scoring_context_param, *other_params = signature.parameters.values()
        type_hints = _typing_extra.get_function_type_hints(function)
        type_hints.pop(scoring_context_param.name, None)
        type_hints.pop('return', None)
        for p in other_params:
            type_hints.setdefault(p.name, Any)
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]

        td = TypedDict(function.__name__, type_hints)  # type: ignore
        td.__pydantic_config__ = {'extra': 'forbid'}  # type: ignore
        adapter = TypeAdapter(td)

        # TODO(DavidM): We should cache the call above this point; it only depends on the registry and spec.call
        #   and may come with non-trivial overhead for large datasets

        try:
            # Include `...` as the first argument to account for the scoring_context parameter
            bound_arguments = signature.bind(..., *spec.args, **spec.kwargs).arguments
            bound_arguments.pop(scoring_context_param.name)
        except TypeError as e:
            case_detail = f'case {case_name!r}' if case_name is not None else 'dataset'
            raise ValueError(f'Failed to bind arguments for {case_detail} assessment {spec.call!r}: {e}')

        try:
            adapter.validate_python(bound_arguments)
        except ValidationError as e:
            error_message = str(e).replace(' for typed-dict', ':')
            raise ValueError(f'Assessment {spec.call!r} in case {case_name!r} has {error_message}') from e

        return function


def _bind_assessment_function(
    function: AssessmentFunction[InputsT, OutputT, MetadataT], args: list[Any], kwargs: dict[str, Any]
) -> BoundAssessmentFunction[InputsT, OutputT, MetadataT]:
    """Bind spec.args and spec.kwargs to `function` without using functools.partial.

    Returns a function (sync or async) that, when called,
    invokes `function` with spec.args/kwargs already included.
    """
    # If there are no extra arguments to bind, just return the original function
    if not args and not kwargs:
        return function

    # Validate that these arguments can bind to the function.
    # This will raise a TypeError if they don't fit the signature.
    inspect.signature(function).bind(..., *args, **kwargs)

    # Decide which wrapper to return based on whether `function` is async.
    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def bound_async_function(ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> AssessmentFunctionResult:
            return await function(ctx, *args, **kwargs)

        return bound_async_function
    else:

        @wraps(function)
        def bound_function(ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> AssessmentFunctionResult:
            return cast(AssessmentFunctionResult, function(ctx, *args, **kwargs))

        return bound_function
