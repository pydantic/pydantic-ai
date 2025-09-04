from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Awaitable, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Generic, cast

from pydantic import (
    ConfigDict,
    model_serializer,
)
from pydantic_core import to_jsonable_python
from pydantic_core.core_schema import SerializationInfo
from typing_extensions import TypeVar, deprecated

from pydantic_ai import _utils

from .._utils import get_event_loop
from .context import EvaluatorContext
from .spec import EvaluatorSpec

__all__ = (
    'EvaluationReason',
    'EvaluationResult',
    'EvaluationScalar',
    'Evaluator',
    'EvaluatorFailure',
    'EvaluatorOutput',
    'EvaluatorSpec',
)

EvaluationScalar = bool | int | float | str
"""The most primitive output allowed as an output from an Evaluator.

`int` and `float` are treated as scores, `str` as labels, and `bool` as assertions.
"""


@dataclass
class EvaluationReason:
    """The result of running an evaluator with an optional explanation.

    Contains a scalar value and an optional "reason" explaining the value.

    Args:
        value: The scalar result of the evaluation (boolean, integer, float, or string).
        reason: An optional explanation of the evaluation result.
    """

    value: EvaluationScalar
    reason: str | None = None


EvaluatorOutput = EvaluationScalar | EvaluationReason | Mapping[str, EvaluationScalar | EvaluationReason]
"""Type for the output of an evaluator, which can be a scalar, an EvaluationReason, or a mapping of names to either."""


# TODO(DavidM): Add bound=EvaluationScalar to the following typevar once pydantic 2.11 is the min supported version
EvaluationScalarT = TypeVar('EvaluationScalarT', default=EvaluationScalar, covariant=True)
"""Type variable for the scalar result type of an evaluation."""

T = TypeVar('T')


@dataclass
class EvaluationResult(Generic[EvaluationScalarT]):
    """The details of an individual evaluation result.

    Contains the name, value, reason, and source evaluator for a single evaluation.

    Args:
        name: The name of the evaluation.
        value: The scalar result of the evaluation.
        reason: An optional explanation of the evaluation result.
        source: The spec of the evaluator that produced this result.
    """

    name: str
    value: EvaluationScalarT
    reason: str | None
    source: EvaluatorSpec

    def downcast(self, *value_types: type[T]) -> EvaluationResult[T] | None:
        """Attempt to downcast this result to a more specific type.

        Args:
            *value_types: The types to check the value against.

        Returns:
            A downcast version of this result if the value is an instance of one of the given types,
            otherwise None.
        """
        # Check if value matches any of the target types, handling bool as a special case
        for value_type in value_types:
            if isinstance(self.value, value_type):
                # Only match bool with explicit bool type
                if isinstance(self.value, bool) and value_type is not bool:
                    continue
                return cast(EvaluationResult[T], self)
        return None


@dataclass
class EvaluatorFailure:
    """Represents a failure raised during the execution of an evaluator."""

    name: str
    error_message: str
    error_stacktrace: str
    source: EvaluatorSpec


# Evaluators are contravariant in all of its parameters.
InputsT = TypeVar('InputsT', default=Any, contravariant=True)
"""Type variable for the inputs type of the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any, contravariant=True)
"""Type variable for the output type of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)
"""Type variable for the metadata type of the task being evaluated."""


class _StrictABCMeta(ABCMeta):
    """An ABC-like metaclass that goes further and disallows even defining abstract subclasses."""

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any):
        result = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Check if this class is a proper subclass of a _StrictABC instance
        is_proper_subclass = any(isinstance(c, _StrictABCMeta) for c in result.__mro__[1:])
        if is_proper_subclass and result.__abstractmethods__:
            abstractmethods = ', '.join([f'{m!r}' for m in result.__abstractmethods__])
            raise TypeError(f'{name} must implement all abstract methods: {abstractmethods}')
        return result


@dataclass(repr=False)
class Evaluator(Generic[InputsT, OutputT, MetadataT], metaclass=_StrictABCMeta):
    """Base class for all evaluators.

    Evaluators can assess the performance of a task in a variety of ways, as a function of the EvaluatorContext.

    Subclasses must implement the `evaluate` method. Note it can be defined with either `def` or `async def`.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class ExactMatch(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return ctx.output == ctx.expected_output
    ```
    """

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the 'name' of this Evaluator to use during serialization.

        Returns:
            The name of the Evaluator, which is typically the class name.
        """
        return cls.__name__

    @classmethod
    @deprecated('`name` has been renamed, use `get_serialization_name` instead.')
    def name(cls) -> str:
        """`name` has been renamed, use `get_serialization_name` instead."""
        return cls.get_serialization_name()

    def get_default_evaluation_name(self) -> str:
        """Return the default name to use in reports for the output of this evaluator.

        By default, if the evaluator has an attribute called `evaluation_name` of type string, that will be used.
        Otherwise, the serialization name of the evaluator (which is usually the class name) will be used.

        This can be overridden to get a more descriptive name in evaluation reports, e.g. using instance information.

        Note that evaluators that return a mapping of results will always use the keys of that mapping as the names
        of the associated evaluation results.
        """
        evaluation_name = getattr(self, 'evaluation_name', None)
        if isinstance(evaluation_name, str):
            # If the evaluator has an attribute `name` of type string, use that
            return evaluation_name

        return self.get_serialization_name()

    @abstractmethod
    def evaluate(
        self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:  # pragma: no cover
        """Evaluate the task output in the given context.

        This is the main evaluation method that subclasses must implement. It can be either synchronous
        or asynchronous, returning either an EvaluatorOutput directly or an Awaitable[EvaluatorOutput].

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those. Can be returned either synchronously or as an
            awaitable for asynchronous evaluation.
        """
        raise NotImplementedError('You must implement `evaluate`.')

    def evaluate_sync(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Run the evaluator synchronously, handling both sync and async implementations.

        This method ensures synchronous execution by running any async evaluate implementation
        to completion using run_until_complete.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.
        """
        output = self.evaluate(ctx)
        if inspect.iscoroutine(output):  # pragma: no cover
            return get_event_loop().run_until_complete(output)
        else:
            return cast(EvaluatorOutput, output)

    async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Run the evaluator asynchronously, handling both sync and async implementations.

        This method ensures asynchronous execution by properly awaiting any async evaluate
        implementation. For synchronous implementations, it returns the result directly.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.
        """
        # Note: If self.evaluate is synchronous, but you need to prevent this from blocking, override this method with:
        # return await anyio.to_thread.run_sync(self.evaluate, ctx)
        output = self.evaluate(ctx)
        if inspect.iscoroutine(output):
            return await output
        else:
            return cast(EvaluatorOutput, output)

    @model_serializer(mode='plain')
    def serialize(self, info: SerializationInfo) -> Any:
        """Serialize this Evaluator to a JSON-serializable form.

        Returns:
            A JSON-serializable representation of this evaluator as an EvaluatorSpec.
        """
        return to_jsonable_python(
            self.as_spec(),
            context=info.context,
            serialize_unknown=True,
        )

    def as_spec(self) -> EvaluatorSpec:
        raw_arguments = self.build_serialization_arguments()

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            arguments = (next(iter(raw_arguments.values())),)
        else:
            arguments = raw_arguments

        return EvaluatorSpec(name=self.get_serialization_name(), arguments=arguments)

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Build the arguments for serialization.

        Evaluators are serialized for inclusion as the "source" in an `EvaluationResult`.
        If you want to modify how the evaluator is serialized for that or other purposes, you can override this method.

        Returns:
            A dictionary of arguments to be used during serialization.
        """
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # always exclude defaults:
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():  # pragma: no branch
                    continue
            raw_arguments[field.name] = value
        return raw_arguments

    __repr__ = _utils.dataclasses_no_defaults_repr
