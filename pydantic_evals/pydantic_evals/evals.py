# TODO: Add support for dataset-wide and case-specific assertions.
# TODO: Add a score to the evaluate span that tracks the percentage of cases that pass all assertions.
from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import Awaitable
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic

import logfire_api
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from pydantic_evals.assertions import AssertionResult
from pydantic_evals.datasets import DatasetRow
from pydantic_evals.reports import EvalReport, EvalReportCase, EvalReportCaseAggregate
from pydantic_evals.scoring import ScoringContext

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

__all__ = ('Evaluation', 'increment_eval_metric')

_logfire = logfire_api.Logfire(otel_scope='pydantic-evals')


InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass(init=False)
class Evaluation(Generic[InputsT, OutputT, MetadataT]):
    """A container for evaluation cases.

    This should generally not be instantiated directly; instead, use the `evaluation` context manager.
    """

    task: Callable[[InputsT], Awaitable[OutputT]]

    name: str
    default_scoring: Callable[[ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = field(repr=False)
    span: logfire_api.LogfireSpan = field(repr=False)

    def __init__(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        scoring: Callable[[ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
        cases: list[DatasetRow[InputsT, OutputT, MetadataT]] | None = None,
    ):
        if name is None:
            name = _get_task_name(task)

        self.task = task
        self.name = name
        self.default_scoring = scoring
        self.eval_cases: list[EvalCase[InputsT, OutputT, MetadataT]] = [EvalCase(c, scoring) for c in cases or []]

        self.span = _logfire.span('evaluate {name}', name=self.name)

    def add_case(
        self,
        dataset_row: DatasetRow[InputsT, OutputT, MetadataT],
        scoring_override: Callable[[ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
    ) -> None:
        self.eval_cases.append(EvalCase(dataset_row, scoring_override or self.default_scoring))

    async def run(self, max_concurrency: int | None = None) -> EvalReport:
        with self.span:
            limiter = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()

            async def _handle_case(case: EvalCase[InputsT, OutputT, MetadataT]) -> EvalReportCase:
                async with limiter:
                    return await run_case(self.task, case)

            async_tasks: list[asyncio.Task[EvalReportCase]] = []
            async with asyncio.TaskGroup() as group:
                for eval_case in self.eval_cases:
                    async_tasks.append(group.create_task(_handle_case(eval_case), name=eval_case.dataset_row.name))

            report = EvalReport(name=self.name, cases=[x.result() for x in async_tasks])

            # TODO: This attribute will be too big in general; remove it once we can use child spans in details panel:
            self.span.set_attribute('cases', report.cases)
            # TODO: Maybe remove this 'averages' attribute if we can compute it from child spans
            self.span.set_attribute('averages', EvalReportCaseAggregate.average(report.cases))
        return report


@dataclass
class _TaskRun:
    attributes: dict[str, Any] = field(init=False, default_factory=dict)
    metrics: dict[str, int | float] = field(init=False, default_factory=dict)

    def record_metric(self, name: str, value: int | float) -> None:
        self.metrics[name] = value

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        if current_value == 0 and incremented_value == 0:
            return  # Avoid recording a metric that is always zero
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


async def _run_task(
    task: Callable[[InputsT], Awaitable[OutputT]], inputs: InputsT
) -> tuple[OutputT, _TaskRun, logfire_api.LogfireSpan]:
    task_run = _TaskRun()
    if _CURRENT_TASK_RUN.get() is not None:
        raise RuntimeError('A task run has already been entered. Task runs should not be nested')
    token = _CURRENT_TASK_RUN.set(task_run)
    try:
        with _logfire.span('execute {task}', task=_get_task_name(task)) as task_span:
            task_output = await task(inputs)
    finally:
        _CURRENT_TASK_RUN.reset(token)
    return task_output, task_run, task_span


@dataclass
class EvalCase(Generic[InputsT, OutputT, MetadataT]):
    dataset_row: DatasetRow[InputsT, OutputT, MetadataT]
    handler: Callable[[ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[None]] | None


async def run_case(
    task: Callable[[InputsT], Awaitable[OutputT]], case: EvalCase[InputsT, OutputT, MetadataT]
) -> EvalReportCase:
    dataset_row = case.dataset_row
    handler = case.handler

    with _logfire.span(
        '{task_name}: {case_name}',
        task_name=_get_task_name(task),
        case_name=dataset_row.name,
        inputs=dataset_row.inputs,
        metadata=dataset_row.metadata,
    ) as case_span:
        output, task_run, task_span = await _run_task(task, dataset_row.inputs)
        task_duration = _get_span_duration(task_span)

        case_span.set_attribute('output', output)
        case_span.set_attribute('task_duration', task_duration)
        case_span.set_attribute('metrics', task_run.metrics)
        case_span.set_attribute('attributes', task_run.attributes)

        scoring_context = ScoringContext[InputsT, OutputT, MetadataT](
            name=dataset_row.name,
            inputs=dataset_row.inputs,
            expected_output=dataset_row.expected_output,
            output=output,
            metadata=dataset_row.metadata,
            attributes=task_run.attributes,
            metrics=task_run.metrics,
        )

        assertions = []
        if dataset_row.assertions:
            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[AssertionResult]] = []
                for assertion in dataset_row.assertions:
                    tasks.append(tg.create_task(assertion.check(scoring_context)))
            assertions = [t.result() for t in tasks]
            case_span.set_attribute('assertions', assertions)

        if handler is not None:
            # TODO: Should we put a span around the handler execution?
            await handler(scoring_context)
            scores = scoring_context.scores
            labels = scoring_context.labels
            case_span.set_attribute('scores', scores)
            case_span.set_attribute('labels', labels)
        else:
            scores = {}
            labels = {}
        context = case_span.context
        assert context is not None
        trace_id = f'{context.trace_id:032x}'
        span_id = f'{context.span_id:016x}'

    jsonable_inputs = to_jsonable_python(dataset_row.inputs)
    report_inputs: dict[str, Any] = (
        jsonable_inputs if isinstance(jsonable_inputs, dict) else {'inputs': jsonable_inputs}
    )
    return EvalReportCase(
        name=dataset_row.name,
        inputs=report_inputs,
        output=output,
        metadata=dataset_row.metadata,
        expected_output=dataset_row.expected_output,
        scores=scores,
        labels=labels,
        metrics=task_run.metrics,
        attributes=task_run.attributes,
        assertions=assertions,
        task_duration=task_duration,
        total_duration=_get_span_duration(case_span),
        trace_id=trace_id,
        span_id=span_id,
    )


_CURRENT_TASK_RUN = ContextVar[_TaskRun | None]('_CURRENT_TASK_RUN', default=None)


def set_eval_attribute(name: str, value: Any) -> None:
    """Set the named attribute for the current eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.record_attribute(name, value)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_task_name(func: Callable[..., Any]) -> str:
    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        # Unwraps f, also unwrapping partials, for the sake of getting f's name
        if isinstance(f, partial):
            return _unwrap(f.func)
        return inspect.unwrap(f)

    return _unwrap(func).__qualname__


def _get_span_duration(span: logfire_api.LogfireSpan) -> float:
    end_time = span.end_time
    start_time = span.start_time
    assert isinstance(start_time, int), 'span is not started'
    assert isinstance(end_time, int), 'span is not finished'
    return (end_time - start_time) / 1_000_000_000
