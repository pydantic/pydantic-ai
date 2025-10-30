from __future__ import annotations as _annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Generic, Literal, Protocol, cast

from pydantic import BaseModel, TypeAdapter
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing_extensions import TypedDict, TypeVar

from pydantic_evals._utils import UNSET, Unset

from ..evaluators import EvaluationResult
from .render_numbers import (
    default_render_duration,
    default_render_duration_diff,
    default_render_number,
    default_render_number_diff,
    default_render_percentage,
)

__all__ = (
    'EvaluationReport',
    'EvaluationReportAdapter',
    'ReportCase',
    'ReportCaseAdapter',
    'ReportCaseFailure',
    'ReportCaseFailureAdapter',
    'EvaluationRenderer',
    'RenderValueConfig',
    'RenderNumberConfig',
    'ReportCaseAggregate',
)

from ..evaluators.evaluator import EvaluatorFailure

MISSING_VALUE_STR = '[i]<missing>[/i]'
EMPTY_CELL_STR = '-'
EMPTY_AGGREGATE_CELL_STR = ''

InputsT = TypeVar('InputsT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
MetadataT = TypeVar('MetadataT', default=Any)


@dataclass(kw_only=True)
class ReportCase(Generic[InputsT, OutputT, MetadataT]):
    """A single case in an evaluation report."""

    name: str
    """The name of the [case][pydantic_evals.Case]."""
    inputs: InputsT
    """The inputs to the task, from [`Case.inputs`][pydantic_evals.Case.inputs]."""
    metadata: MetadataT | None
    """Any metadata associated with the case, from [`Case.metadata`][pydantic_evals.Case.metadata]."""
    expected_output: OutputT | None
    """The expected output of the task, from [`Case.expected_output`][pydantic_evals.Case.expected_output]."""
    output: OutputT
    """The output of the task execution."""

    metrics: dict[str, float | int]
    attributes: dict[str, Any]

    scores: dict[str, EvaluationResult[int | float]]
    labels: dict[str, EvaluationResult[str]]
    assertions: dict[str, EvaluationResult[bool]]

    task_duration: float
    total_duration: float  # includes evaluator execution time

    trace_id: str | None = None
    """The trace ID of the case span."""
    span_id: str | None = None
    """The span ID of the case span."""

    evaluator_failures: list[EvaluatorFailure] = field(default_factory=list)


@dataclass(kw_only=True)
class ReportCaseFailure(Generic[InputsT, OutputT, MetadataT]):
    """A single case in an evaluation report that failed due to an error during task execution."""

    name: str
    """The name of the [case][pydantic_evals.Case]."""
    inputs: InputsT
    """The inputs to the task, from [`Case.inputs`][pydantic_evals.Case.inputs]."""
    metadata: MetadataT | None
    """Any metadata associated with the case, from [`Case.metadata`][pydantic_evals.Case.metadata]."""
    expected_output: OutputT | None
    """The expected output of the task, from [`Case.expected_output`][pydantic_evals.Case.expected_output]."""

    error_message: str
    """The message of the exception that caused the failure."""
    error_stacktrace: str
    """The stacktrace of the exception that caused the failure."""

    trace_id: str | None = None
    """The trace ID of the case span."""
    span_id: str | None = None
    """The span ID of the case span."""


ReportCaseAdapter = TypeAdapter(ReportCase[Any, Any, Any])
ReportCaseFailureAdapter = TypeAdapter(ReportCaseFailure[Any, Any, Any])


class ReportCaseAggregate(BaseModel):
    """A synthetic case that summarizes a set of cases."""

    name: str

    scores: dict[str, float | int]
    labels: dict[str, dict[str, float]]
    metrics: dict[str, float | int]
    assertions: float | None
    task_duration: float
    total_duration: float

    @staticmethod
    def average(cases: list[ReportCase]) -> ReportCaseAggregate:
        """Produce a synthetic "summary" case by averaging quantitative attributes."""
        num_cases = len(cases)
        if num_cases == 0:
            return ReportCaseAggregate(
                name='Averages',
                scores={},
                labels={},
                metrics={},
                assertions=None,
                task_duration=0.0,
                total_duration=0.0,
            )

        def _scores_averages(scores_by_name: list[dict[str, int | float | bool]]) -> dict[str, float]:
            counts_by_name: dict[str, int] = defaultdict(int)
            sums_by_name: dict[str, float] = defaultdict(float)
            for sbn in scores_by_name:
                for name, score in sbn.items():
                    counts_by_name[name] += 1
                    sums_by_name[name] += score
            return {name: sums_by_name[name] / counts_by_name[name] for name in sums_by_name}

        def _labels_averages(labels_by_name: list[dict[str, str]]) -> dict[str, dict[str, float]]:
            counts_by_name: dict[str, int] = defaultdict(int)
            sums_by_name: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for lbn in labels_by_name:
                for name, label in lbn.items():
                    counts_by_name[name] += 1
                    sums_by_name[name][label] += 1
            return {
                name: {value: count / counts_by_name[name] for value, count in sums_by_name[name].items()}
                for name in sums_by_name
            }

        average_task_duration = sum(case.task_duration for case in cases) / num_cases
        average_total_duration = sum(case.total_duration for case in cases) / num_cases

        # average_assertions: dict[str, float] = _scores_averages([{k: v.value for k, v in case.scores.items()} for case in cases])
        average_scores: dict[str, float] = _scores_averages(
            [{k: v.value for k, v in case.scores.items()} for case in cases]
        )
        average_labels: dict[str, dict[str, float]] = _labels_averages(
            [{k: v.value for k, v in case.labels.items()} for case in cases]
        )
        average_metrics: dict[str, float] = _scores_averages([case.metrics for case in cases])

        average_assertions: float | None = None
        n_assertions = sum(len(case.assertions) for case in cases)
        if n_assertions > 0:
            n_passing = sum(1 for case in cases for assertion in case.assertions.values() if assertion.value)
            average_assertions = n_passing / n_assertions

        return ReportCaseAggregate(
            name='Averages',
            scores=average_scores,
            labels=average_labels,
            metrics=average_metrics,
            assertions=average_assertions,
            task_duration=average_task_duration,
            total_duration=average_total_duration,
        )


@dataclass(kw_only=True)
class EvaluationReport(Generic[InputsT, OutputT, MetadataT]):
    """A report of the results of evaluating a model on a set of cases."""

    name: str
    """The name of the report."""

    cases: list[ReportCase[InputsT, OutputT, MetadataT]]
    """The cases in the report."""
    failures: list[ReportCaseFailure[InputsT, OutputT, MetadataT]] = field(default_factory=list)
    """The failures in the report. These are cases where task execution raised an exception."""

    experiment_metadata: dict[str, Any] | None = None
    """Metadata associated with the specific experiment represented by this report."""
    trace_id: str | None = None
    """The trace ID of the evaluation."""
    span_id: str | None = None
    """The span ID of the evaluation."""

    def averages(self) -> ReportCaseAggregate | None:
        if self.cases:
            return ReportCaseAggregate.average(self.cases)
        return None

    def render(
        self,
        width: int | None = None,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_errors: bool = True,
        include_error_stacktrace: bool = False,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
    ) -> str:
        """Render this report to a nicely-formatted string, optionally comparing it to a baseline report.

        If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
        """
        io_file = StringIO()
        console = Console(width=width, file=io_file)
        self.print(
            width=width,
            baseline=baseline,
            console=console,
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_errors=include_errors,
            include_error_stacktrace=include_error_stacktrace,
            include_evaluator_failures=include_evaluator_failures,
            input_config=input_config,
            metadata_config=metadata_config,
            output_config=output_config,
            score_configs=score_configs,
            label_configs=label_configs,
            metric_configs=metric_configs,
            duration_config=duration_config,
            include_reasons=include_reasons,
        )
        return io_file.getvalue()

    def print(
        self,
        width: int | None = None,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        console: Console | None = None,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_errors: bool = True,
        include_error_stacktrace: bool = False,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
    ) -> None:
        """Print this report to the console, optionally comparing it to a baseline report.

        If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
        """
        if console is None:  # pragma: no branch
            console = Console(width=width)

        metadata_panel = self._metadata_panel(baseline=baseline)
        renderable: RenderableType = self.console_table(
            baseline=baseline,
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_evaluator_failures=include_evaluator_failures,
            input_config=input_config,
            metadata_config=metadata_config,
            output_config=output_config,
            score_configs=score_configs,
            label_configs=label_configs,
            metric_configs=metric_configs,
            duration_config=duration_config,
            include_reasons=include_reasons,
            with_title=not metadata_panel,
        )
        # Wrap table with experiment metadata panel if present
        if metadata_panel:
            renderable = Group(metadata_panel, renderable)
        console.print(renderable)
        if include_errors and self.failures:  # pragma: no cover
            failures_table = self.failures_table(
                include_input=include_input,
                include_metadata=include_metadata,
                include_expected_output=include_expected_output,
                include_error_message=True,
                include_error_stacktrace=include_error_stacktrace,
                input_config=input_config,
                metadata_config=metadata_config,
            )
            console.print(failures_table, style='red')

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def console_table(
        self,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
        with_title: bool = True,
    ) -> Table:
        """Return a table containing the data from this report.

        If a baseline is provided, returns a diff between this report and the baseline report.
        Optionally include input and output details.
        """
        renderer = EvaluationRenderer(
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_error_message=False,
            include_error_stacktrace=False,
            include_evaluator_failures=include_evaluator_failures,
            input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
            metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
            output_config=output_config or _DEFAULT_VALUE_CONFIG,
            score_configs=score_configs or {},
            label_configs=label_configs or {},
            metric_configs=metric_configs or {},
            duration_config=duration_config or _DEFAULT_DURATION_CONFIG,
            include_reasons=include_reasons,
        )
        if baseline is None:
            return renderer.build_table(self, with_title=with_title)
        else:
            return renderer.build_diff_table(self, baseline, with_title=with_title)

    def _metadata_panel(
        self, baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None
    ) -> RenderableType | None:
        """Wrap a table with an experiment metadata panel if metadata exists.

        Args:
            table: The table to wrap
            baseline: Optional baseline report for diff metadata

        Returns:
            Either the table unchanged or a Group with Panel and Table
        """
        if baseline is None:
            # Single report - show metadata if present
            if self.experiment_metadata:
                metadata_text = Text()
                items = list(self.experiment_metadata.items())
                for i, (key, value) in enumerate(items):
                    metadata_text.append(f'{key}: {value}', style='dim')
                    if i < len(items) - 1:
                        metadata_text.append('\n')
                return Panel(
                    metadata_text,
                    title=f'Evaluation Summary: {self.name}',
                    title_align='left',
                    border_style='dim',
                    padding=(0, 1),
                    expand=False,
                )
        else:
            # Diff report - show metadata diff if either has metadata
            if self.experiment_metadata or baseline.experiment_metadata:
                diff_name = baseline.name if baseline.name == self.name else f'{baseline.name} → {self.name}'
                metadata_text = Text()
                lines_styles: list[tuple[str, str]] = []
                if baseline.experiment_metadata and self.experiment_metadata:
                    # Collect all keys from both
                    all_keys = sorted(set(baseline.experiment_metadata.keys()) | set(self.experiment_metadata.keys()))
                    for key in all_keys:
                        baseline_val = baseline.experiment_metadata.get(key)
                        report_val = self.experiment_metadata.get(key)
                        if baseline_val == report_val:
                            lines_styles.append((f'{key}: {report_val}', 'dim'))
                        elif baseline_val is None:
                            lines_styles.append((f'+ {key}: {report_val}', 'green'))
                        elif report_val is None:
                            lines_styles.append((f'- {key}: {baseline_val}', 'red'))
                        else:
                            lines_styles.append((f'{key}: {baseline_val} → {report_val}', 'yellow'))
                elif self.experiment_metadata:
                    lines_styles = [(f'+ {k}: {v}', 'green') for k, v in self.experiment_metadata.items()]
                else:  # baseline.experiment_metadata only
                    assert baseline.experiment_metadata is not None
                    lines_styles = [(f'- {k}: {v}', 'red') for k, v in baseline.experiment_metadata.items()]

                for i, (line, style) in enumerate(lines_styles):
                    metadata_text.append(line, style=style)
                    if i < len(lines_styles) - 1:
                        metadata_text.append('\n')

                return Panel(
                    metadata_text,
                    title=f'Evaluation Diff: {diff_name}',
                    title_align='left',
                    border_style='dim',
                    padding=(0, 1),
                    expand=False,
                )

        return None

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def failures_table(
        self,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_error_message: bool = True,
        include_error_stacktrace: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
    ) -> Table:
        """Return a table containing the failures in this report."""
        renderer = EvaluationRenderer(
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=False,
            include_durations=False,
            include_total_duration=False,
            include_removed_cases=False,
            include_averages=False,
            input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
            metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
            output_config=_DEFAULT_VALUE_CONFIG,
            score_configs={},
            label_configs={},
            metric_configs={},
            duration_config=_DEFAULT_DURATION_CONFIG,
            include_reasons=False,
            include_error_message=include_error_message,
            include_error_stacktrace=include_error_stacktrace,
            include_evaluator_failures=False,  # Not applicable for failures table
        )
        return renderer.build_failures_table(self)

    def __str__(self) -> str:  # pragma: lax no cover
        """Return a string representation of the report."""
        return self.render()


EvaluationReportAdapter = TypeAdapter(EvaluationReport[Any, Any, Any])


class RenderValueConfig(TypedDict, total=False):
    """A configuration for rendering a values in an Evaluation report."""

    value_formatter: str | Callable[[Any], str]
    diff_checker: Callable[[Any, Any], bool] | None
    diff_formatter: Callable[[Any, Any], str | None] | None
    diff_style: str


@dataclass(kw_only=True)
class _ValueRenderer:
    value_formatter: str | Callable[[Any], str] = '{}'
    diff_checker: Callable[[Any, Any], bool] | None = lambda x, y: x != y
    diff_formatter: Callable[[Any, Any], str | None] | None = None
    diff_style: str = 'magenta'

    @staticmethod
    def from_config(config: RenderValueConfig) -> _ValueRenderer:
        return _ValueRenderer(
            value_formatter=config.get('value_formatter', '{}'),
            diff_checker=config.get('diff_checker', lambda x, y: x != y),
            diff_formatter=config.get('diff_formatter'),
            diff_style=config.get('diff_style', 'magenta'),
        )

    def render_value(self, name: str | None, v: Any) -> str:
        result = self._get_value_str(v)
        if name:
            result = f'{name}: {result}'
        return result

    def render_diff(self, name: str | None, old: Any | None, new: Any | None) -> str:
        old_str = self._get_value_str(old) or MISSING_VALUE_STR
        new_str = self._get_value_str(new) or MISSING_VALUE_STR
        if old_str == new_str:
            result = old_str
        else:
            result = f'{old_str} → {new_str}'

            has_diff = self.diff_checker and self.diff_checker(old, new)
            if has_diff:  # pragma: no branch
                # If there is a diff, make the name bold and compute the diff_str
                name = name and f'[bold]{name}[/]'
                diff_str = self.diff_formatter and self.diff_formatter(old, new)
                if diff_str:  # pragma: no cover
                    result += f' ({diff_str})'
                result = f'[{self.diff_style}]{result}[/]'

        # Add the name
        if name:
            result = f'{name}: {result}'

        return result

    def _get_value_str(self, value: Any) -> str:
        if value is None:
            return MISSING_VALUE_STR
        if isinstance(self.value_formatter, str):
            return self.value_formatter.format(value)
        else:
            return self.value_formatter(value)


class RenderNumberConfig(TypedDict, total=False):
    """A configuration for rendering a particular score or metric in an Evaluation report.

    See the implementation of `_RenderNumber` for more clarity on how these parameters affect the rendering.
    """

    value_formatter: str | Callable[[float | int], str]
    """The logic to use for formatting values.

    * If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four significant figures.
    * You can also use a custom string format spec, e.g. '{:.3f}'
    * You can also use a custom function, e.g. lambda x: f'{x:.3f}'
    """
    diff_formatter: str | Callable[[float | int, float | int], str | None] | None
    """The logic to use for formatting details about the diff.

    The strings produced by the value_formatter will always be included in the reports, but the diff_formatter is
    used to produce additional text about the difference between the old and new values, such as the absolute or
    relative difference.

    * If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four
        significant figures, and will include the percentage change.
    * You can also use a custom string format spec, e.g. '{:+.3f}'
    * You can also use a custom function, e.g. lambda x: f'{x:+.3f}'.
        If this function returns None, no extra diff text will be added.
    * You can also use None to never generate extra diff text.
    """
    diff_atol: float
    """The absolute tolerance for considering a difference "significant".

    A difference is "significant" if `abs(new - old) < self.diff_atol + self.diff_rtol * abs(old)`.

    If a difference is not significant, it will not have the diff styles applied. Note that we still show
    both the rendered before and after values in the diff any time they differ, even if the difference is not
    significant. (If the rendered values are exactly the same, we only show the value once.)

    If not provided, use 1e-6.
    """
    diff_rtol: float
    """The relative tolerance for considering a difference "significant".

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use 0.001 if all values are ints, otherwise 0.05.
    """
    diff_increase_style: str
    """The style to apply to diffed values that have a significant increase.

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use green for scores and red for metrics. You can also use arbitrary `rich` styles, such as "bold red".
    """
    diff_decrease_style: str
    """The style to apply to diffed values that have significant decrease.

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use red for scores and green for metrics. You can also use arbitrary `rich` styles, such as "bold red".
    """


@dataclass(kw_only=True)
class _NumberRenderer:
    """See documentation of `RenderNumberConfig` for more details about the parameters here."""

    value_formatter: str | Callable[[float | int], str]
    diff_formatter: str | Callable[[float | int, float | int], str | None] | None
    diff_atol: float
    diff_rtol: float
    diff_increase_style: str
    diff_decrease_style: str

    def render_value(self, name: str | None, v: float | int) -> str:
        result = self._get_value_str(v)
        if name:
            result = f'{name}: {result}'
        return result

    def render_diff(self, name: str | None, old: float | int | None, new: float | int | None) -> str:
        old_str = self._get_value_str(old)
        new_str = self._get_value_str(new)
        if old_str == new_str:
            result = old_str
        else:
            result = f'{old_str} → {new_str}'

            diff_style = self._get_diff_style(old, new)
            if diff_style:
                # If there is a diff, make the name bold and compute the diff_str
                name = name and f'[bold]{name}[/]'
                diff_str = self._get_diff_str(old, new)
                if diff_str:  # pragma: no branch
                    result += f' ({diff_str})'
                result = f'[{diff_style}]{result}[/]'

        # Add the name
        if name:  # pragma: no branch
            result = f'{name}: {result}'

        return result

    @staticmethod
    def infer_from_config(
        config: RenderNumberConfig, kind: Literal['score', 'metric', 'duration'], values: list[float | int]
    ) -> _NumberRenderer:
        value_formatter = config.get('value_formatter', UNSET)
        if isinstance(value_formatter, Unset):
            value_formatter = default_render_number

        diff_formatter = config.get('diff_formatter', UNSET)
        if isinstance(diff_formatter, Unset):
            diff_formatter = default_render_number_diff

        diff_atol = config.get('diff_atol', UNSET)
        if isinstance(diff_atol, Unset):
            diff_atol = 1e-6

        diff_rtol = config.get('diff_rtol', UNSET)
        if isinstance(diff_rtol, Unset):
            values_are_ints = all(isinstance(v, int) for v in values)
            diff_rtol = 0.001 if values_are_ints else 0.05

        diff_increase_style = config.get('diff_increase_style', UNSET)
        if isinstance(diff_increase_style, Unset):
            diff_increase_style = 'green' if kind == 'score' else 'red'

        diff_decrease_style = config.get('diff_decrease_style', UNSET)
        if isinstance(diff_decrease_style, Unset):
            diff_decrease_style = 'red' if kind == 'score' else 'green'

        return _NumberRenderer(
            value_formatter=value_formatter,
            diff_formatter=diff_formatter,
            diff_rtol=diff_rtol,
            diff_atol=diff_atol,
            diff_increase_style=diff_increase_style,
            diff_decrease_style=diff_decrease_style,
        )

    def _get_value_str(self, value: float | int | None) -> str:
        if value is None:
            return MISSING_VALUE_STR
        if isinstance(self.value_formatter, str):
            return self.value_formatter.format(value)
        else:
            return self.value_formatter(value)

    def _get_diff_str(self, old: float | int | None, new: float | int | None) -> str | None:
        if old is None or new is None:  # pragma: no cover
            return None
        if isinstance(self.diff_formatter, str):  # pragma: no cover
            return self.diff_formatter.format(new - old)
        elif self.diff_formatter is None:  # pragma: no cover
            return None
        else:
            return self.diff_formatter(old, new)

    def _get_diff_style(self, old: float | int | None, new: float | int | None) -> str | None:
        # 1 means new is higher, -1 means new is lower, 0 means no change
        if old is None or new is None:
            return None

        diff = new - old
        if abs(diff) < self.diff_atol + self.diff_rtol * abs(old):
            return None
        return self.diff_increase_style if diff > 0 else self.diff_decrease_style


T_contra = TypeVar('T_contra', contravariant=True)


class _AbstractRenderer(Protocol[T_contra]):
    def render_value(self, name: str | None, v: T_contra) -> str: ...  # pragma: no branch

    def render_diff(self, name: str | None, old: T_contra | None, new: T_contra | None) -> str: ...  # pragma: no branch


_DEFAULT_NUMBER_CONFIG = RenderNumberConfig()
_DEFAULT_VALUE_CONFIG = RenderValueConfig()
_DEFAULT_DURATION_CONFIG = RenderNumberConfig(
    value_formatter=default_render_duration,
    diff_formatter=default_render_duration_diff,
    diff_atol=1e-6,  # one microsecond
    diff_rtol=0.1,
    diff_increase_style='red',
    diff_decrease_style='green',
)


T = TypeVar('T')


@dataclass(kw_only=True)
class ReportCaseRenderer:
    include_input: bool
    include_metadata: bool
    include_expected_output: bool
    include_output: bool
    include_scores: bool
    include_labels: bool
    include_metrics: bool
    include_assertions: bool
    include_reasons: bool
    include_durations: bool
    include_total_duration: bool
    include_error_message: bool
    include_error_stacktrace: bool
    include_evaluator_failures: bool

    input_renderer: _ValueRenderer
    metadata_renderer: _ValueRenderer
    output_renderer: _ValueRenderer
    score_renderers: Mapping[str, _NumberRenderer]
    label_renderers: Mapping[str, _ValueRenderer]
    metric_renderers: Mapping[str, _NumberRenderer]
    duration_renderer: _NumberRenderer

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_base_table(self, title: str) -> Table:
        """Build and return a Rich Table for the diff output."""
        table = Table(title=title, show_lines=True)
        table.add_column('Case ID', style='bold')
        if self.include_input:
            table.add_column('Inputs', overflow='fold')
        if self.include_metadata:
            table.add_column('Metadata', overflow='fold')
        if self.include_expected_output:
            table.add_column('Expected Output', overflow='fold')
        if self.include_output:
            table.add_column('Outputs', overflow='fold')
        if self.include_scores:
            table.add_column('Scores', overflow='fold')
        if self.include_labels:
            table.add_column('Labels', overflow='fold')
        if self.include_metrics:
            table.add_column('Metrics', overflow='fold')
        if self.include_assertions:
            table.add_column('Assertions', overflow='fold')
        if self.include_evaluator_failures:
            table.add_column('Evaluator Failures', overflow='fold')
        if self.include_durations:
            table.add_column('Durations' if self.include_total_duration else 'Duration', justify='right')
        return table

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_failures_table(self, title: str) -> Table:
        """Build and return a Rich Table for the failures output."""
        table = Table(title=title, show_lines=True)
        table.add_column('Case ID', style='bold')
        if self.include_input:
            table.add_column('Inputs', overflow='fold')
        if self.include_metadata:
            table.add_column('Metadata', overflow='fold')
        if self.include_expected_output:
            table.add_column('Expected Output', overflow='fold')
        if self.include_error_message:
            table.add_column('Error Message', overflow='fold')
        if self.include_error_stacktrace:
            table.add_column('Error Stacktrace', overflow='fold')
        return table

    def build_row(self, case: ReportCase) -> list[str]:
        """Build a table row for a single case."""
        row = [case.name]

        if self.include_input:
            row.append(self.input_renderer.render_value(None, case.inputs) or EMPTY_CELL_STR)

        if self.include_metadata:
            row.append(self.metadata_renderer.render_value(None, case.metadata) or EMPTY_CELL_STR)

        if self.include_expected_output:
            row.append(self.output_renderer.render_value(None, case.expected_output) or EMPTY_CELL_STR)

        if self.include_output:
            row.append(self.output_renderer.render_value(None, case.output) or EMPTY_CELL_STR)

        if self.include_scores:
            row.append(self._render_dict({k: v for k, v in case.scores.items()}, self.score_renderers))

        if self.include_labels:
            row.append(self._render_dict({k: v for k, v in case.labels.items()}, self.label_renderers))

        if self.include_metrics:
            row.append(self._render_dict(case.metrics, self.metric_renderers))

        if self.include_assertions:
            row.append(self._render_assertions(list(case.assertions.values())))

        if self.include_evaluator_failures:
            row.append(self._render_evaluator_failures(case.evaluator_failures))

        if self.include_durations:
            row.append(self._render_durations(case))

        return row

    def build_aggregate_row(self, aggregate: ReportCaseAggregate) -> list[str]:
        """Build a table row for an aggregated case."""
        row = [f'[b i]{aggregate.name}[/]']

        if self.include_input:
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_metadata:
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_expected_output:
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_output:
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_scores:
            row.append(self._render_dict(aggregate.scores, self.score_renderers))

        if self.include_labels:
            row.append(self._render_dict(aggregate.labels, self.label_renderers))

        if self.include_metrics:
            row.append(self._render_dict(aggregate.metrics, self.metric_renderers))

        if self.include_assertions:
            row.append(self._render_aggregate_assertions(aggregate.assertions))

        if self.include_evaluator_failures:
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_durations:
            row.append(self._render_durations(aggregate))

        return row

    def build_diff_row(
        self,
        new_case: ReportCase,
        baseline: ReportCase,
    ) -> list[str]:
        """Build a table row for a given case ID."""
        assert baseline.name == new_case.name, 'This should only be called for matching case IDs'
        row = [baseline.name]

        if self.include_input:  # pragma: no branch
            input_diff = self.input_renderer.render_diff(None, baseline.inputs, new_case.inputs) or EMPTY_CELL_STR
            row.append(input_diff)

        if self.include_metadata:  # pragma: no branch
            metadata_diff = (
                self.metadata_renderer.render_diff(None, baseline.metadata, new_case.metadata) or EMPTY_CELL_STR
            )
            row.append(metadata_diff)

        if self.include_expected_output:  # pragma: no branch
            expected_output_diff = (
                self.output_renderer.render_diff(None, baseline.expected_output, new_case.expected_output)
                or EMPTY_CELL_STR
            )
            row.append(expected_output_diff)

        if self.include_output:  # pragma: no branch
            output_diff = self.output_renderer.render_diff(None, baseline.output, new_case.output) or EMPTY_CELL_STR
            row.append(output_diff)

        if self.include_scores:  # pragma: no branch
            scores_diff = self._render_dicts_diff(
                {k: v.value for k, v in baseline.scores.items()},
                {k: v.value for k, v in new_case.scores.items()},
                self.score_renderers,
            )
            row.append(scores_diff)

        if self.include_labels:  # pragma: no branch
            labels_diff = self._render_dicts_diff(
                {k: v.value for k, v in baseline.labels.items()},
                {k: v.value for k, v in new_case.labels.items()},
                self.label_renderers,
            )
            row.append(labels_diff)

        if self.include_metrics:  # pragma: no branch
            metrics_diff = self._render_dicts_diff(baseline.metrics, new_case.metrics, self.metric_renderers)
            row.append(metrics_diff)

        if self.include_assertions:  # pragma: no branch
            assertions_diff = self._render_assertions_diff(
                list(baseline.assertions.values()), list(new_case.assertions.values())
            )
            row.append(assertions_diff)

        if self.include_evaluator_failures:  # pragma: no branch
            evaluator_failures_diff = self._render_evaluator_failures_diff(
                baseline.evaluator_failures, new_case.evaluator_failures
            )
            row.append(evaluator_failures_diff)

        if self.include_durations:  # pragma: no branch
            durations_diff = self._render_durations_diff(baseline, new_case)
            row.append(durations_diff)

        return row

    def build_diff_aggregate_row(
        self,
        new: ReportCaseAggregate,
        baseline: ReportCaseAggregate,
    ) -> list[str]:
        """Build a table row for a given case ID."""
        assert baseline.name == new.name, 'This should only be called for aggregates with matching names'
        row = [f'[b i]{baseline.name}[/]']

        if self.include_input:  # pragma: no branch
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_metadata:  # pragma: no branch
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_expected_output:  # pragma: no branch
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_output:  # pragma: no branch
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_scores:  # pragma: no branch
            scores_diff = self._render_dicts_diff(baseline.scores, new.scores, self.score_renderers)
            row.append(scores_diff)

        if self.include_labels:  # pragma: no branch
            labels_diff = self._render_dicts_diff(baseline.labels, new.labels, self.label_renderers)
            row.append(labels_diff)

        if self.include_metrics:  # pragma: no branch
            metrics_diff = self._render_dicts_diff(baseline.metrics, new.metrics, self.metric_renderers)
            row.append(metrics_diff)

        if self.include_assertions:  # pragma: no branch
            assertions_diff = self._render_aggregate_assertions_diff(baseline.assertions, new.assertions)
            row.append(assertions_diff)

        if self.include_evaluator_failures:  # pragma: no branch
            row.append(EMPTY_AGGREGATE_CELL_STR)

        if self.include_durations:  # pragma: no branch
            durations_diff = self._render_durations_diff(baseline, new)
            row.append(durations_diff)

        return row

    def build_failure_row(self, case: ReportCaseFailure) -> list[str]:
        """Build a table row for a single case failure."""
        row = [case.name]

        if self.include_input:
            row.append(self.input_renderer.render_value(None, case.inputs) or EMPTY_CELL_STR)

        if self.include_metadata:
            row.append(self.metadata_renderer.render_value(None, case.metadata) or EMPTY_CELL_STR)

        if self.include_expected_output:
            row.append(self.output_renderer.render_value(None, case.expected_output) or EMPTY_CELL_STR)

        if self.include_error_message:
            row.append(case.error_message or EMPTY_CELL_STR)

        if self.include_error_stacktrace:
            row.append(case.error_stacktrace or EMPTY_CELL_STR)

        return row

    def _render_durations(self, case: ReportCase | ReportCaseAggregate) -> str:
        """Build the diff string for a duration value."""
        case_durations: dict[str, float] = {'task': case.task_duration}
        if self.include_total_duration:
            case_durations['total'] = case.total_duration
        return self._render_dict(
            case_durations,
            {'task': self.duration_renderer, 'total': self.duration_renderer},
            include_names=self.include_total_duration,
        )

    def _render_durations_diff(
        self,
        base_case: ReportCase | ReportCaseAggregate,
        new_case: ReportCase | ReportCaseAggregate,
    ) -> str:
        """Build the diff string for a duration value."""
        base_case_durations: dict[str, float] = {'task': base_case.task_duration}
        new_case_durations: dict[str, float] = {'task': new_case.task_duration}
        if self.include_total_duration:  # pragma: no branch
            base_case_durations['total'] = base_case.total_duration
            new_case_durations['total'] = new_case.total_duration
        return self._render_dicts_diff(
            base_case_durations,
            new_case_durations,
            {'task': self.duration_renderer, 'total': self.duration_renderer},
            include_names=self.include_total_duration,
        )

    @staticmethod
    def _render_dicts_diff(
        baseline_dict: dict[str, T],
        new_dict: dict[str, T],
        renderers: Mapping[str, _AbstractRenderer[T]],
        *,
        include_names: bool = True,
    ) -> str:
        keys: set[str] = set()
        keys.update(baseline_dict.keys())
        keys.update(new_dict.keys())
        diff_lines: list[str] = []
        for key in sorted(keys):
            old_val = baseline_dict.get(key)
            new_val = new_dict.get(key)
            rendered = renderers[key].render_diff(key if include_names else None, old_val, new_val)
            diff_lines.append(rendered)
        return '\n'.join(diff_lines) if diff_lines else EMPTY_CELL_STR

    def _render_dict(
        self,
        case_dict: Mapping[str, EvaluationResult[T] | T],
        renderers: Mapping[str, _AbstractRenderer[T]],
        *,
        include_names: bool = True,
    ) -> str:
        diff_lines: list[str] = []
        for key, val in case_dict.items():
            value = cast(EvaluationResult[T], val).value if isinstance(val, EvaluationResult) else val
            rendered = renderers[key].render_value(key if include_names else None, value)
            if self.include_reasons and isinstance(val, EvaluationResult) and (reason := val.reason):
                rendered += f'\n  Reason: {reason}\n'
            diff_lines.append(rendered)
        return '\n'.join(diff_lines) if diff_lines else EMPTY_CELL_STR

    def _render_assertions(
        self,
        assertions: list[EvaluationResult[bool]],
    ) -> str:
        if not assertions:
            return EMPTY_CELL_STR
        lines: list[str] = []
        for a in assertions:
            line = '[green]✔[/]' if a.value else '[red]✗[/]'
            if self.include_reasons:
                line = f'{a.name}: {line}\n'
                line = f'{line}  Reason: {a.reason}\n\n' if a.reason else line
            lines.append(line)
        return ''.join(lines)

    @staticmethod
    def _render_aggregate_assertions(
        assertions: float | None,
    ) -> str:
        return (
            default_render_percentage(assertions) + ' [green]✔[/]'
            if assertions is not None
            else EMPTY_AGGREGATE_CELL_STR
        )

    @staticmethod
    def _render_assertions_diff(
        assertions: list[EvaluationResult[bool]], new_assertions: list[EvaluationResult[bool]]
    ) -> str:
        if not assertions and not new_assertions:  # pragma: no cover
            return EMPTY_CELL_STR

        old = ''.join(['[green]✔[/]' if a.value else '[red]✗[/]' for a in assertions])
        new = ''.join(['[green]✔[/]' if a.value else '[red]✗[/]' for a in new_assertions])
        return old if old == new else f'{old} → {new}'

    @staticmethod
    def _render_aggregate_assertions_diff(
        baseline: float | None,
        new: float | None,
    ) -> str:
        if baseline is None and new is None:  # pragma: no cover
            return EMPTY_AGGREGATE_CELL_STR
        rendered_baseline = (
            default_render_percentage(baseline) + ' [green]✔[/]' if baseline is not None else EMPTY_CELL_STR
        )
        rendered_new = default_render_percentage(new) + ' [green]✔[/]' if new is not None else EMPTY_CELL_STR
        return rendered_new if rendered_baseline == rendered_new else f'{rendered_baseline} → {rendered_new}'

    def _render_evaluator_failures(
        self,
        failures: list[EvaluatorFailure],
    ) -> str:
        if not failures:
            return EMPTY_CELL_STR  # pragma: no cover
        lines: list[str] = []
        for failure in failures:
            line = f'[red]{failure.name}[/]'
            if failure.error_message:
                line += f': {failure.error_message}'
            lines.append(line)
        return '\n'.join(lines)

    def _render_evaluator_failures_diff(
        self,
        baseline_failures: list[EvaluatorFailure],
        new_failures: list[EvaluatorFailure],
    ) -> str:
        baseline_str = self._render_evaluator_failures(baseline_failures)
        new_str = self._render_evaluator_failures(new_failures)
        if baseline_str == new_str:
            return baseline_str  # pragma: no cover
        return f'{baseline_str}\n→\n{new_str}'


@dataclass(kw_only=True)
class EvaluationRenderer:
    """A class for rendering an EvalReport or the diff between two EvalReports."""

    # Columns to include
    include_input: bool
    include_metadata: bool
    include_expected_output: bool
    include_output: bool
    include_durations: bool
    include_total_duration: bool

    # Rows to include
    include_removed_cases: bool
    include_averages: bool

    input_config: RenderValueConfig
    metadata_config: RenderValueConfig
    output_config: RenderValueConfig
    score_configs: dict[str, RenderNumberConfig]
    label_configs: dict[str, RenderValueConfig]
    metric_configs: dict[str, RenderNumberConfig]
    duration_config: RenderNumberConfig

    # Data to include
    include_reasons: bool  # only applies to reports, not to diffs

    include_error_message: bool
    include_error_stacktrace: bool
    include_evaluator_failures: bool

    def include_scores(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.scores for case in self._all_cases(report, baseline))

    def include_labels(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.labels for case in self._all_cases(report, baseline))

    def include_metrics(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.metrics for case in self._all_cases(report, baseline))

    def include_assertions(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.assertions for case in self._all_cases(report, baseline))

    def include_evaluator_failures_column(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return self.include_evaluator_failures and any(
            case.evaluator_failures for case in self._all_cases(report, baseline)
        )

    def _all_cases(self, report: EvaluationReport, baseline: EvaluationReport | None) -> list[ReportCase]:
        if not baseline:
            return report.cases
        else:
            return report.cases + self._baseline_cases_to_include(report, baseline)

    def _baseline_cases_to_include(self, report: EvaluationReport, baseline: EvaluationReport) -> list[ReportCase]:
        if self.include_removed_cases:
            return baseline.cases
        report_case_names = {case.name for case in report.cases}
        return [case for case in baseline.cases if case.name in report_case_names]

    def _get_case_renderer(
        self, report: EvaluationReport, baseline: EvaluationReport | None = None
    ) -> ReportCaseRenderer:
        input_renderer = _ValueRenderer.from_config(self.input_config)
        metadata_renderer = _ValueRenderer.from_config(self.metadata_config)
        output_renderer = _ValueRenderer.from_config(self.output_config)
        score_renderers = self._infer_score_renderers(report, baseline)
        label_renderers = self._infer_label_renderers(report, baseline)
        metric_renderers = self._infer_metric_renderers(report, baseline)
        duration_renderer = _NumberRenderer.infer_from_config(
            self.duration_config, 'duration', [x.task_duration for x in self._all_cases(report, baseline)]
        )

        return ReportCaseRenderer(
            include_input=self.include_input,
            include_metadata=self.include_metadata,
            include_expected_output=self.include_expected_output,
            include_output=self.include_output,
            include_scores=self.include_scores(report, baseline),
            include_labels=self.include_labels(report, baseline),
            include_metrics=self.include_metrics(report, baseline),
            include_assertions=self.include_assertions(report, baseline),
            include_reasons=self.include_reasons,
            include_durations=self.include_durations,
            include_total_duration=self.include_total_duration,
            include_error_message=self.include_error_message,
            include_error_stacktrace=self.include_error_stacktrace,
            include_evaluator_failures=self.include_evaluator_failures_column(report, baseline),
            input_renderer=input_renderer,
            metadata_renderer=metadata_renderer,
            output_renderer=output_renderer,
            score_renderers=score_renderers,
            label_renderers=label_renderers,
            metric_renderers=metric_renderers,
            duration_renderer=duration_renderer,
        )

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_table(self, report: EvaluationReport, *, with_title: bool = True) -> Table:
        """Build a table for the report.

        Args:
            report: The evaluation report to render
            with_title: Whether to include the title in the table (default True)

        Returns:
            A Rich Table object
        """
        case_renderer = self._get_case_renderer(report)

        title = f'Evaluation Summary: {report.name}' if with_title else ''
        table = case_renderer.build_base_table(title)

        for case in report.cases:
            table.add_row(*case_renderer.build_row(case))

        if self.include_averages:  # pragma: no branch
            average = report.averages()
            if average:  # pragma: no branch
                table.add_row(*case_renderer.build_aggregate_row(average))

        return table

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_diff_table(
        self, report: EvaluationReport, baseline: EvaluationReport, *, with_title: bool = True
    ) -> Table:
        """Build a diff table comparing report to baseline.

        Args:
            report: The evaluation report to compare
            baseline: The baseline report to compare against
            with_title: Whether to include the title in the table (default True)

        Returns:
            A Rich Table object
        """
        report_cases = report.cases
        baseline_cases = self._baseline_cases_to_include(report, baseline)

        report_cases_by_id = {case.name: case for case in report_cases}
        baseline_cases_by_id = {case.name: case for case in baseline_cases}

        diff_cases: list[tuple[ReportCase, ReportCase]] = []
        removed_cases: list[ReportCase] = []
        added_cases: list[ReportCase] = []

        for case_id in sorted(set(baseline_cases_by_id.keys()) | set(report_cases_by_id.keys())):
            maybe_baseline_case = baseline_cases_by_id.get(case_id)
            maybe_report_case = report_cases_by_id.get(case_id)
            if maybe_baseline_case and maybe_report_case:
                diff_cases.append((maybe_baseline_case, maybe_report_case))
            elif maybe_baseline_case:
                removed_cases.append(maybe_baseline_case)
            elif maybe_report_case:
                added_cases.append(maybe_report_case)
            else:  # pragma: no cover
                assert False, 'This should be unreachable'

        case_renderer = self._get_case_renderer(report, baseline)
        diff_name = baseline.name if baseline.name == report.name else f'{baseline.name} → {report.name}'

        title = f'Evaluation Diff: {diff_name}' if with_title else ''
        table = case_renderer.build_base_table(title)

        for baseline_case, new_case in diff_cases:
            table.add_row(*case_renderer.build_diff_row(new_case, baseline_case))
        for case in added_cases:
            row = case_renderer.build_row(case)
            row[0] = f'[green]+ Added Case[/]\n{row[0]}'
            table.add_row(*row)
        for case in removed_cases:
            row = case_renderer.build_row(case)
            row[0] = f'[red]- Removed Case[/]\n{row[0]}'
            table.add_row(*row)

        if self.include_averages:  # pragma: no branch
            report_average = ReportCaseAggregate.average(report_cases)
            baseline_average = ReportCaseAggregate.average(baseline_cases)
            table.add_row(*case_renderer.build_diff_aggregate_row(report_average, baseline_average))

        return table

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_failures_table(self, report: EvaluationReport) -> Table:
        case_renderer = self._get_case_renderer(report)
        table = case_renderer.build_failures_table('Case Failures')
        for case in report.failures:
            table.add_row(*case_renderer.build_failure_row(case))

        return table

    def _infer_score_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _NumberRenderer]:
        all_cases = self._all_cases(report, baseline)

        values_by_name: dict[str, list[float | int]] = {}
        for case in all_cases:
            for k, score in case.scores.items():
                values_by_name.setdefault(k, []).append(score.value)

        all_renderers: dict[str, _NumberRenderer] = {}
        for name, values in values_by_name.items():
            merged_config = _DEFAULT_NUMBER_CONFIG.copy()
            merged_config.update(self.score_configs.get(name, {}))
            all_renderers[name] = _NumberRenderer.infer_from_config(merged_config, 'score', values)
        return all_renderers

    def _infer_label_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _ValueRenderer]:
        all_cases = self._all_cases(report, baseline)
        all_names: set[str] = set()
        for case in all_cases:
            for k in case.labels:
                all_names.add(k)

        all_renderers: dict[str, _ValueRenderer] = {}
        for name in all_names:
            merged_config = _DEFAULT_VALUE_CONFIG.copy()
            merged_config.update(self.label_configs.get(name, {}))
            all_renderers[name] = _ValueRenderer.from_config(merged_config)
        return all_renderers

    def _infer_metric_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _NumberRenderer]:
        all_cases = self._all_cases(report, baseline)

        values_by_name: dict[str, list[float | int]] = {}
        for case in all_cases:
            for k, v in case.metrics.items():
                values_by_name.setdefault(k, []).append(v)

        all_renderers: dict[str, _NumberRenderer] = {}
        for name, values in values_by_name.items():
            merged_config = _DEFAULT_NUMBER_CONFIG.copy()
            merged_config.update(self.metric_configs.get(name, {}))
            all_renderers[name] = _NumberRenderer.infer_from_config(merged_config, 'metric', values)
        return all_renderers

    def _infer_duration_renderer(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> _NumberRenderer:  # pragma: no cover
        all_cases = self._all_cases(report, baseline)
        all_durations = [x.task_duration for x in all_cases]
        if self.include_total_duration:
            all_durations += [x.total_duration for x in all_cases]
        return _NumberRenderer.infer_from_config(self.duration_config, 'duration', all_durations)
