from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Annotated, Any, Generic, TypeAlias, cast

from pydantic import Field
from typing_extensions import TypeVar

from .._utils import get_function_serialization_name, get_unwrapped_function_name, run_until_complete
from ._base import BaseEvaluator
from .context import EvaluatorContext
from .spec import EvaluatorSpec

__all__ = (
    'EvaluationReason',
    'EvaluationResult',
    'EvaluationScalar',
    'Evaluator',
    'EvaluatorFailure',
    'EvaluatorFunc',
    'EvaluatorOutput',
    'EvaluatorSpec',
    'FunctionEvaluator',
)

EvaluationScalar = bool | int | Annotated[float, Field(allow_inf_nan=False)] | str
"""The most primitive output allowed as an output from an Evaluator.

`int` and finite `float` are treated as scores, `str` as labels, and `bool` as assertions.
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


EvaluationScalarT = TypeVar('EvaluationScalarT', bound=EvaluationScalar, default=EvaluationScalar, covariant=True)
"""Type variable for the scalar result type of an evaluation."""

T = TypeVar('T', bound=EvaluationScalar)


@dataclass(kw_only=True)
class EvaluationResult(Generic[EvaluationScalarT]):
    """The details of an individual evaluation result.

    Contains the name, value, reason, and source evaluator for a single evaluation.

    Args:
        name: The name of the evaluation.
        value: The scalar result of the evaluation.
        reason: An optional explanation of the evaluation result.
        source: The spec of the evaluator that produced this result.
        evaluator_version: Optional version tag for the evaluator that produced this result
            (e.g. `'v2'`). Sourced automatically from the evaluator's
            [`get_evaluator_version`][pydantic_evals.evaluators.Evaluator.get_evaluator_version]
            method. Lets online-evaluation dashboards filter out results from retired versions
            without deleting historical rows.
    """

    name: str
    value: EvaluationScalarT
    reason: str | None
    source: EvaluatorSpec
    evaluator_version: str | None = None

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


@dataclass(kw_only=True)
class EvaluatorFailure:
    """Represents a failure raised during the execution of an evaluator."""

    name: str
    error_message: str
    error_stacktrace: str
    source: EvaluatorSpec
    evaluator_version: str | None = None
    """Optional version tag for the evaluator that raised (e.g. `'v2'`). Sourced automatically
    from the evaluator's
    [`get_evaluator_version`][pydantic_evals.evaluators.Evaluator.get_evaluator_version] method."""
    error_type: str | None = None
    """Class name of the exception that caused the failure (e.g. `'ValueError'`). Populated
    automatically when `EvaluatorFailure` is constructed from a caught exception; surfaced
    as the `error.type` attribute on emitted OTel events."""


# Evaluators are contravariant in all of its parameters.
InputsT = TypeVar('InputsT', default=Any, contravariant=True)
"""Type variable for the inputs type of the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any, contravariant=True)
"""Type variable for the output type of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)
"""Type variable for the metadata type of the task being evaluated."""


@dataclass(repr=False)
class Evaluator(BaseEvaluator, Generic[InputsT, OutputT, MetadataT]):
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

    Override [`get_default_evaluation_name`][pydantic_evals.evaluators.Evaluator.get_default_evaluation_name]
    to customize the name used in reports, and
    [`get_evaluator_version`][pydantic_evals.evaluators.Evaluator.get_evaluator_version] to tag the
    evaluator with a version that downstream sinks can filter on.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class LLMJudge(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool: ...

        def get_evaluator_version(self) -> str | None:
            return 'v2'  # bumped after prompt rewrite
    ```
    """

    def get_default_evaluation_name(self) -> str:
        """Return the default name to use in reports for the output of this evaluator.

        Defaults to the serialization name of the evaluator (which is usually the class name). Override this
        method to customize the name, e.g. using instance information.

        Note that evaluators that return a mapping of results will always use the keys of that mapping as the names
        of the associated evaluation results.
        """
        return self.get_serialization_name()

    def get_evaluator_version(self) -> str | None:
        """Return the version tag for this evaluator, or `None` if it has no version.

        Propagated to online-evaluation sinks so dashboards can filter out results produced by retired
        versions without deleting historical rows. Applies to every result the evaluator emits; bump
        whenever behavior changes in a way that invalidates prior scores. Override this method to set
        a non-`None` version.
        """
        return None

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
            return run_until_complete(output)
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


EvaluatorFunc: TypeAlias = Callable[
    [EvaluatorContext[InputsT, OutputT, MetadataT]], EvaluatorOutput | Awaitable[EvaluatorOutput]
]
"""A plain sync or async function that can be used as an evaluator.

It takes an [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] and returns the same
[`EvaluatorOutput`][pydantic_evals.evaluators.EvaluatorOutput] as
[`Evaluator.evaluate`][pydantic_evals.evaluators.Evaluator.evaluate]. Functions are accepted wherever
evaluators are, and are wrapped in a [`FunctionEvaluator`][pydantic_evals.evaluators.FunctionEvaluator].
"""


@dataclass(repr=False)
class FunctionEvaluator(Evaluator[InputsT, OutputT, MetadataT]):
    """An evaluator that calls a plain sync or async function.

    You rarely need to construct this yourself: a function passed wherever evaluators are accepted is
    wrapped automatically. Construct it explicitly to set an `evaluation_name` or `version`, or to pass a function to
    [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator].

    In a dataset file, a function evaluator is identified by the function's name, so a dataset saved with one can
    be loaded again by passing the function in `custom_evaluator_types`, see
    [Saving Function Evaluators](../../evals/evaluators/custom.md#saving-function-evaluators).

    Example:
    ```python
    from pydantic_evals.evaluators import EvaluatorContext, FunctionEvaluator


    def is_polite(ctx: EvaluatorContext[str, str]) -> bool:
        return 'please' in ctx.output.lower()


    evaluator = FunctionEvaluator(is_polite, evaluation_name='politeness', version='v2')
    ```
    """

    function: EvaluatorFunc[InputsT, OutputT, MetadataT]
    """The function to call with the evaluator context."""
    _: KW_ONLY
    evaluation_name: str | None = None
    """The name to use for the function's result in reports. Defaults to the function's name."""
    version: str | None = None
    """The version tag to record on the function's results, see
    [`get_evaluator_version`][pydantic_evals.evaluators.Evaluator.get_evaluator_version]."""

    def evaluate(
        self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:
        return self.function(ctx)

    # `EvaluatorFunc` allows any awaitable (e.g. an `asyncio.Task`), not just the coroutines the base class handles.
    def evaluate_sync(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        output = self.evaluate(ctx)
        if inspect.isawaitable(output):
            return run_until_complete(output)
        return output

    async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        output = self.evaluate(ctx)
        if inspect.isawaitable(output):
            return await output
        return output

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or self._function_name()

    def get_evaluator_version(self) -> str | None:
        return self.version

    def as_spec(self) -> EvaluatorSpec:
        # Used as the `source` of every result, so this must not raise like serialization does.
        # Like the built-in evaluators, the spec names the evaluator itself and carries any `evaluation_name` override.
        return EvaluatorSpec(
            name=self._function_name(),
            arguments={'evaluation_name': self.evaluation_name} if self.evaluation_name else None,
        )

    def _function_name(self) -> str:
        try:
            return get_unwrapped_function_name(self.function)
        except AttributeError:
            # Callables like `operator.attrgetter(...)` have no `__name__`.
            return type(self.function).__name__

    def _serialization_spec(self) -> EvaluatorSpec:
        name = get_function_serialization_name(self.function)
        if name is None:
            raise TypeError(
                f'Function evaluator {self.get_default_evaluation_name()!r} cannot be serialized, as it is identified '
                'by its name, which only a plain function defined with `def` has, not a lambda, bound method, '
                '`functools.partial` or other callable. To save a dataset to a file, use a plain function or an '
                '`Evaluator` subclass instead.'
            )
        options = {'evaluation_name': self.evaluation_name, 'version': self.version}
        arguments = {key: value for key, value in options.items() if value is not None}
        return EvaluatorSpec(name=name, arguments=arguments or None)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.get_default_evaluation_name()})'
