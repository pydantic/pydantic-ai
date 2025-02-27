from __future__ import annotations as _annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import logfire_api
from rich.console import Console

from pydantic_evals._utils import UNSET, Unset
from pydantic_evals.reports import EvalReport, EvalReportCase, RenderNumberConfig, RenderValueConfig

__all__ = ('Eval', 'EvalCase', 'evaluation', 'increment_eval_metric')

P = ParamSpec('P')
OutputT = TypeVar('OutputT')


@contextmanager
def evaluation(name: str) -> Iterator[Eval]:
    """Context manager for starting an evaluation."""
    with logfire_api.span('evaluation', name=name) as eval_span:
        yield Eval(name, eval_span)


@dataclass
class Eval:
    """A container for evaluation cases.

    This should generally not be instantiated directly; instead, use the `evaluation` context manager.
    """

    name: str
    span: logfire_api.LogfireSpan = field(repr=False)
    cases: list[EvalCase[Any]] = field(default_factory=list)

    def as_report(self) -> EvalReport:
        return EvalReport(name=self.name, cases=[c.as_report_case() for c in self.cases])

    def print_report(
        self,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    def print_diff(
        self,
        *,
        baseline: Eval,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                baseline=baseline.as_report(),
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_removed_cases=include_removed_cases,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    @asynccontextmanager
    async def case(
        self,
        f: Callable[P, Awaitable[OutputT]],
        _case_id: str | None = None,
        *case_input_args: P.args,
        **case_input_kwargs: P.kwargs,
    ) -> AsyncIterator[EvalCase[OutputT]]:
        if _CURRENT_EVAL_CASE.get() is not None:
            raise RuntimeError('An eval case has already been entered. Evaluation cases should not be nested')

        sig = inspect.signature(f)

        bound_arguments = sig.bind(*case_input_args, **case_input_kwargs)
        task_input = dict(bound_arguments.arguments)

        bound_arguments.apply_defaults()
        task_defaults = {k: v for k, v in bound_arguments.arguments.items() if k not in task_input}
        case_id = _case_id or str(len(self.cases) + 1)

        with logfire_api.span(
            Eval.case.__name__,
            task_name=_get_task_name(f),
            case_id=case_id,
            task_input=task_input,
            task_defaults=task_defaults,
        ) as case_span:
            task_span = logfire_api.span('task execution')
            eval_case = EvalCase[OutputT](case_id, case_input=task_input, case_span=case_span, task_span=task_span)
            token = _CURRENT_EVAL_CASE.set(eval_case)

            try:
                with task_span:
                    # Note: Ideally we'd have a way to make sure that we didn't open an extra nested span here if f
                    # was already opening a span of its own
                    # It's important to have a span that exists only around the call of `f` though, otherwise
                    # if you do some slow/heavy-weight scoring after this function yields, you won't be able to
                    # distinguish between that time and the time spent in the function call itself
                    eval_case.task_span = task_span
                    case_output = await f(*case_input_args, **case_input_kwargs)

                case_span.set_attribute('case_output', case_output)
                eval_case.case_output = case_output
                self.cases.append(eval_case)

                yield eval_case
            finally:
                _CURRENT_EVAL_CASE.reset(token)


@dataclass
class EvalCase(Generic[OutputT]):
    """A container for an evaluation case."""

    case_id: str
    case_input: dict[str, Any]
    _case_output: OutputT | Unset = field(init=False, default=UNSET)

    scores: dict[str, int | float] = field(init=False, default_factory=dict)
    metrics: dict[str, int | float] = field(init=False, default_factory=dict)
    labels: dict[str, bool | str] = field(init=False, default_factory=dict)

    case_span: logfire_api.LogfireSpan = field(repr=False)
    task_span: logfire_api.LogfireSpan = field(repr=False)

    @property
    def case_output(self) -> OutputT:
        if isinstance(self._case_output, Unset):
            raise RuntimeError('case_output accessed before it was set.')
        return self._case_output

    @case_output.setter
    def case_output(self, case_output: OutputT) -> None:
        self._case_output = case_output

    def record_score(self, name: str, value: int | float) -> None:
        score_attribute = f'score.{name}'
        self.scores[name] = value

        # If we want to use span links to store scores we can do something like this:
        # with logfire.span('score {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating scores via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(score_attribute, value)

    def record_label(self, name: str, value: bool | str) -> None:
        label_attribute = f'label.{name}'
        self.labels[name] = value

        # If we want to use span links to store labels we can do something like this:
        # with logfire.span('label {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating labels via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(label_attribute, value)

    def record_metric(self, name: str, value: int | float) -> None:
        metric_attribute = f'metric.{name}'
        self.metrics[name] = value

        # If we want to use span links to store metrics we can do something like this:
        # with logfire.span('metric {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating metrics via span links, but I'm not sure if we should _only_ support that
        self.case_span.set_attribute(metric_attribute, value)

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        self.record_metric(name, incremented_value)

    def as_report_case(self) -> EvalReportCase:
        return EvalReportCase(
            case_id=self.case_id,
            case_input=self.case_input,
            case_output=self.case_output,
            scores=self.scores,
            metrics=self.metrics,
            labels=self.labels,
            task_duration=_get_span_duration(self.task_span),
            total_duration=_get_span_duration(self.case_span),
        )


_CURRENT_EVAL_CASE = ContextVar[EvalCase[Any] | None]('_CURRENT_EVAL_CASE', default=None)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current evaluation case."""
    current_case = _CURRENT_EVAL_CASE.get()
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
