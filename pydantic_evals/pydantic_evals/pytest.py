"""Pytest integration for Pydantic Evals."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard
from urllib.parse import quote

import pytest
from typing_extensions import TypeVar

from .dataset import Dataset
from .reporting import EvaluationReport

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter

    from pydantic_ai.retries import RetryConfig

    from .lifecycle import CaseLifecycle

__all__ = ('assert_evaluation_passes', 'assert_evaluation_passes_sync')

InputsT = TypeVar('InputsT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
MetadataT = TypeVar('MetadataT', default=Any)

EvalTask: TypeAlias = Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT]
ReportCheck: TypeAlias = Callable[[EvaluationReport[InputsT, OutputT, MetadataT]], None]

_EVALUATE_KWARGS = {
    'name',
    'max_concurrency',
    'progress',
    'retry_task',
    'retry_evaluators',
    'task_name',
    'metadata',
    'repeat',
    'lifecycle',
}
_CONFIG_KWARGS = {
    'min_assertion_pass_rate',
    'require_assertions',
    'fail_on_task_failures',
    'fail_on_evaluator_failures',
    'fail_on_report_evaluator_failures',
    'include_input',
    'include_metadata',
    'include_output',
    'include_expected_output',
    'include_reasons',
    'include_error_stacktrace',
    'report_check',
}
_MARKER_KWARGS = _EVALUATE_KWARGS | _CONFIG_KWARGS | {'task_factory'}
_DIRECT_TASK_ATTR = '__pydantic_evals_direct_task__'
_EVAL_REPORT_OPTION = '--pydantic-evals-report'
_LOGFIRE_OPTION = '--pydantic-evals-logfire'


@dataclass(kw_only=True)
class _EvalReportEntry:
    nodeid: str
    dataset_name: str
    report: EvaluationReport[Any, Any, Any]
    test_config: _EvaluationTestConfig


_EVAL_REPORTS_KEY: pytest.StashKey[list[_EvalReportEntry]] = pytest.StashKey()
_LOGFIRE_CONFIGURED_KEY: pytest.StashKey[bool] = pytest.StashKey()


@dataclass(kw_only=True)
class _EvaluationTestConfig:
    min_assertion_pass_rate: float | None = 1.0
    require_assertions: bool = True
    fail_on_task_failures: bool = True
    fail_on_evaluator_failures: bool = True
    fail_on_report_evaluator_failures: bool = True
    include_input: bool = False
    include_metadata: bool = False
    include_output: bool = True
    include_expected_output: bool = True
    include_reasons: bool = True
    include_error_stacktrace: bool = False
    report_check: ReportCheck[Any, Any, Any] | None = None

    def __post_init__(self) -> None:
        if self.min_assertion_pass_rate is not None and not 0 <= self.min_assertion_pass_rate <= 1:
            raise ValueError('`min_assertion_pass_rate` must be between 0 and 1.')


@dataclass(kw_only=True)
class _EvalMarkConfig:
    dataset: Dataset[Any, Any, Any]
    test_config: _EvaluationTestConfig
    evaluate_kwargs: dict[str, Any]
    task_factory: bool


async def assert_evaluation_passes(
    dataset: Dataset[InputsT, OutputT, MetadataT],
    task: EvalTask[InputsT, OutputT],
    *,
    min_assertion_pass_rate: float | None = 1.0,
    require_assertions: bool = True,
    fail_on_task_failures: bool = True,
    fail_on_evaluator_failures: bool = True,
    fail_on_report_evaluator_failures: bool = True,
    include_input: bool = False,
    include_metadata: bool = False,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_error_stacktrace: bool = False,
    report_check: ReportCheck[InputsT, OutputT, MetadataT] | None = None,
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = False,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    repeat: int = 1,
    lifecycle: type[CaseLifecycle[InputsT, OutputT, MetadataT]] | None = None,
) -> EvaluationReport[InputsT, OutputT, MetadataT]:
    """Run an evaluation and fail the current pytest test if the report does not satisfy the criteria."""
    report = await dataset.evaluate(
        task,
        name=name,
        max_concurrency=max_concurrency,
        progress=progress,
        retry_task=retry_task,
        retry_evaluators=retry_evaluators,
        task_name=task_name,
        metadata=metadata,
        repeat=repeat,
        lifecycle=lifecycle,
    )
    _assert_report_passes(
        report,
        _EvaluationTestConfig(
            min_assertion_pass_rate=min_assertion_pass_rate,
            require_assertions=require_assertions,
            fail_on_task_failures=fail_on_task_failures,
            fail_on_evaluator_failures=fail_on_evaluator_failures,
            fail_on_report_evaluator_failures=fail_on_report_evaluator_failures,
            include_input=include_input,
            include_metadata=include_metadata,
            include_output=include_output,
            include_expected_output=include_expected_output,
            include_reasons=include_reasons,
            include_error_stacktrace=include_error_stacktrace,
            report_check=report_check,
        ),
    )
    return report


def assert_evaluation_passes_sync(
    dataset: Dataset[InputsT, OutputT, MetadataT],
    task: EvalTask[InputsT, OutputT],
    *,
    min_assertion_pass_rate: float | None = 1.0,
    require_assertions: bool = True,
    fail_on_task_failures: bool = True,
    fail_on_evaluator_failures: bool = True,
    fail_on_report_evaluator_failures: bool = True,
    include_input: bool = False,
    include_metadata: bool = False,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_error_stacktrace: bool = False,
    report_check: ReportCheck[InputsT, OutputT, MetadataT] | None = None,
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = False,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    repeat: int = 1,
    lifecycle: type[CaseLifecycle[InputsT, OutputT, MetadataT]] | None = None,
) -> EvaluationReport[InputsT, OutputT, MetadataT]:
    """Run an evaluation and fail the current pytest test if the report does not satisfy the criteria."""
    report = dataset.evaluate_sync(
        task,
        name=name,
        max_concurrency=max_concurrency,
        progress=progress,
        retry_task=retry_task,
        retry_evaluators=retry_evaluators,
        task_name=task_name,
        metadata=metadata,
        repeat=repeat,
        lifecycle=lifecycle,
    )
    _assert_report_passes(
        report,
        _EvaluationTestConfig(
            min_assertion_pass_rate=min_assertion_pass_rate,
            require_assertions=require_assertions,
            fail_on_task_failures=fail_on_task_failures,
            fail_on_evaluator_failures=fail_on_evaluator_failures,
            fail_on_report_evaluator_failures=fail_on_report_evaluator_failures,
            include_input=include_input,
            include_metadata=include_metadata,
            include_output=include_output,
            include_expected_output=include_expected_output,
            include_reasons=include_reasons,
            include_error_stacktrace=include_error_stacktrace,
            report_check=report_check,
        ),
    )
    return report


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('pydantic-evals')
    group.addoption(
        _EVAL_REPORT_OPTION,
        choices=('all', 'failures', 'none'),
        default='all',
        help='Control pydantic-evals report rendering in the pytest terminal summary.',
    )
    group.addoption(
        _LOGFIRE_OPTION,
        choices=('auto', 'none'),
        default='auto',
        help='Control automatic Logfire configuration for pytest eval runs.',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_EVAL_REPORTS_KEY] = []
    config.stash[_LOGFIRE_CONFIGURED_KEY] = False
    config.addinivalue_line(
        'markers',
        (
            'eval(dataset, *, task_factory=False, ...): run a pydantic-evals dataset as a pytest test. '
            'By default the marked function is used as the task and receives each case input. '
            'Use task_factory=True when the function should receive pytest fixtures and return the task callable.'
        ),
    )


def _configure_logfire_for_pytest(config: pytest.Config) -> None:
    if config.stash[_LOGFIRE_CONFIGURED_KEY] or config.getoption(_LOGFIRE_OPTION) == 'none':
        return
    config.stash[_LOGFIRE_CONFIGURED_KEY] = True
    try:
        import logfire
    except ImportError:
        _configure_otel_tracer_provider()
        return

    logfire.configure(
        send_to_logfire='if-token-present',
        service_name='pydantic-evals-pytest',
        environment='test',
    )


def _configure_otel_tracer_provider() -> None:
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import ProxyTracerProvider, get_tracer_provider, set_tracer_provider
    except ImportError:
        return

    if isinstance(get_tracer_provider(), ProxyTracerProvider):
        set_tracer_provider(TracerProvider())


def pytest_pycollect_makeitem(
    collector: pytest.Module | pytest.Class, name: str, obj: object
) -> pytest.Collector | pytest.Item | None:
    if not isinstance(obj, FunctionType):
        return None

    eval_mark = _get_eval_mark(obj)
    if eval_mark is None:
        return None

    config = _parse_eval_mark(eval_mark)
    if _has_parametrize_mark(obj):
        raise pytest.UsageError('`pytest.mark.eval` cannot be combined with `pytest.mark.parametrize`.')

    _configure_logfire_for_pytest(collector.config)
    if not config.task_factory and _is_direct_task(obj):
        setattr(obj, _DIRECT_TASK_ATTR, True)
        obj.__dict__['__signature__'] = inspect.Signature()
    return None


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    eval_mark = pyfuncitem.get_closest_marker('eval')
    if eval_mark is None:
        return None

    config = _parse_eval_mark(eval_mark)
    if getattr(pyfuncitem.obj, _DIRECT_TASK_ATTR, False):
        task = pyfuncitem.obj
    else:
        task = _call_task_factory(pyfuncitem)

    report = config.dataset.evaluate_sync(task, **config.evaluate_kwargs)
    _add_report(pyfuncitem, config.dataset.name, report, config.test_config)
    _assert_report_passes(report, config.test_config)
    return True


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: pytest.Config) -> None:
    report_mode = config.getoption(_EVAL_REPORT_OPTION)
    if report_mode == 'none':
        return

    empty_reports: list[_EvalReportEntry] = []
    reports = config.stash.get(_EVAL_REPORTS_KEY, empty_reports)
    if not reports:
        return

    entries = reports
    if report_mode == 'failures':
        entries = [entry for entry in reports if _collect_report_problems(entry.report, entry.test_config)]
    if not entries:
        return

    terminalreporter.section('Pydantic Evals')
    for entry in entries:
        terminalreporter.write_sep('-', entry.nodeid)
        terminalreporter.write_line(
            entry.report.render(
                width=120,
                include_input=entry.test_config.include_input,
                include_metadata=entry.test_config.include_metadata,
                include_output=entry.test_config.include_output,
                include_expected_output=entry.test_config.include_expected_output,
                include_reasons=entry.test_config.include_reasons,
                include_error_stacktrace=entry.test_config.include_error_stacktrace,
            )
        )
        if evals_url := _logfire_evals_url(entry.report):
            dataset_url = _logfire_dataset_eval_url(evals_url, entry.dataset_name)
            terminalreporter.write_line(
                f'Logfire dataset ({entry.dataset_name}): {_terminal_hyperlink(dataset_url, dataset_url)}'
            )
            terminalreporter.write_line('')


def _add_report(
    pyfuncitem: pytest.Function,
    dataset_name: str,
    report: EvaluationReport[Any, Any, Any],
    test_config: _EvaluationTestConfig,
) -> None:
    pyfuncitem.config.stash[_EVAL_REPORTS_KEY].append(
        _EvalReportEntry(
            nodeid=pyfuncitem.nodeid,
            dataset_name=dataset_name,
            report=report,
            test_config=test_config,
        )
    )


def _logfire_evals_url(report: EvaluationReport[Any, Any, Any]) -> str | None:
    try:
        import logfire
    except ImportError:
        return None

    url_from_eval = getattr(logfire, 'url_from_eval', None)
    if not callable(url_from_eval):
        return None

    eval_url = url_from_eval(report)
    if not isinstance(eval_url, str):
        return None
    marker = '/evals/'
    if marker not in eval_url:
        return None
    project_url, _ = eval_url.split(marker, maxsplit=1)
    return f'{project_url}/evals'


def _logfire_dataset_eval_url(evals_url: str, eval_name: str) -> str:
    return f'{evals_url.rstrip("/")}/{quote(eval_name, safe="")}'


def _terminal_hyperlink(label: str, url: str) -> str:
    return f'\033]8;;{url}\033\\\033[36m{label}\033[0m\033]8;;\033\\'


def _call_task_factory(pyfuncitem: pytest.Function) -> EvalTask[Any, Any]:
    kwargs = _task_factory_kwargs(pyfuncitem)
    task = pyfuncitem.obj(**kwargs)
    if inspect.isawaitable(task):
        from ._utils import get_event_loop

        task = get_event_loop().run_until_complete(task)
    if not callable(task):
        raise TypeError('A `pytest.mark.eval(..., task_factory=True)` test must return the task callable.')
    return task


def _task_factory_kwargs(pyfuncitem: pytest.Function) -> dict[str, Any]:
    parameters = inspect.signature(pyfuncitem.obj).parameters
    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if name in pyfuncitem.funcargs:  # pragma: no branch
            kwargs[name] = pyfuncitem.funcargs[name]
    return kwargs


def _assert_report_passes(report: EvaluationReport[InputsT, OutputT, MetadataT], config: _EvaluationTestConfig) -> None:
    problems = _collect_report_problems(report, config)
    if problems:
        pytest.fail(_format_failure_message(report, config, problems), pytrace=False)
    if config.report_check is not None:
        config.report_check(report)


def _collect_report_problems(report: EvaluationReport[Any, Any, Any], config: _EvaluationTestConfig) -> list[str]:
    problems: list[str] = []

    if config.fail_on_task_failures and report.failures:
        problems.append(_pluralize(len(report.failures), 'task failure'))

    n_assertions, n_failed_assertions = _count_assertions(report)
    averages = report.averages()
    assertion_pass_rate = averages.assertions if averages is not None else None
    if config.require_assertions and n_assertions == 0:
        problems.append('no assertions were produced')
    if config.min_assertion_pass_rate is not None:
        if assertion_pass_rate is None:
            if not config.require_assertions:
                problems.append('assertion pass rate is unavailable')
        elif assertion_pass_rate < config.min_assertion_pass_rate:
            problems.append(
                f'assertion pass rate {assertion_pass_rate:.2f} is below required {config.min_assertion_pass_rate:.2f}'
            )
            if n_failed_assertions:  # pragma: no branch
                problems.append(_pluralize(n_failed_assertions, 'failed assertion'))

    n_evaluator_failures = sum(len(case.evaluator_failures) for case in report.cases)
    if config.fail_on_evaluator_failures and n_evaluator_failures:
        problems.append(_pluralize(n_evaluator_failures, 'evaluator failure'))

    if config.fail_on_report_evaluator_failures and report.report_evaluator_failures:
        problems.append(_pluralize(len(report.report_evaluator_failures), 'report evaluator failure'))

    return problems


def _format_failure_message(
    report: EvaluationReport[Any, Any, Any], config: _EvaluationTestConfig, problems: list[str]
) -> str:
    lines = [f'Evaluation {report.name!r} failed pytest criteria:', '']
    lines.extend(f'- {problem}' for problem in problems)
    failed_assertions = _failed_assertion_lines(report)
    if failed_assertions:
        lines.append('')
        lines.append('Failed assertions:')
        lines.extend(f'- {line}' for line in failed_assertions[:10])
        if len(failed_assertions) > 10:
            lines.append(f'- ... {len(failed_assertions) - 10} more')
    lines.append('')
    lines.append(
        report.render(
            width=120,
            include_input=config.include_input,
            include_metadata=config.include_metadata,
            include_output=config.include_output,
            include_expected_output=config.include_expected_output,
            include_reasons=config.include_reasons,
            include_error_stacktrace=config.include_error_stacktrace,
        )
    )
    return '\n'.join(lines)


def _count_assertions(report: EvaluationReport[Any, Any, Any]) -> tuple[int, int]:
    n_assertions = 0
    n_failed_assertions = 0
    for case in report.cases:
        n_assertions += len(case.assertions)
        n_failed_assertions += sum(not assertion.value for assertion in case.assertions.values())
    return n_assertions, n_failed_assertions


def _failed_assertion_lines(report: EvaluationReport[Any, Any, Any]) -> list[str]:
    lines: list[str] = []
    for case in report.cases:
        for assertion in case.assertions.values():
            if assertion.value:
                continue
            line = f'{case.name}: {assertion.name}'
            if assertion.reason:
                line += f' - {assertion.reason}'
            lines.append(line)
    return lines


def _pluralize(count: int, singular: str) -> str:
    return f'{count} {singular}' if count == 1 else f'{count} {singular}s'


def _get_eval_mark(obj: object) -> pytest.Mark | None:
    marks = getattr(obj, 'pytestmark', ())
    if isinstance(marks, pytest.Mark):  # pragma: no cover
        marks = (marks,)
    eval_marks = [mark for mark in marks if isinstance(mark, pytest.Mark) and mark.name == 'eval']
    if len(eval_marks) > 1:
        raise pytest.UsageError('Only one `pytest.mark.eval` marker can be applied to a test.')
    return eval_marks[0] if eval_marks else None


def _parse_eval_mark(mark: pytest.Mark) -> _EvalMarkConfig:
    if len(mark.args) != 1:
        raise pytest.UsageError('`pytest.mark.eval` requires exactly one positional argument: the dataset.')
    raw_dataset: object = mark.args[0]
    if not _is_dataset(raw_dataset):
        raise pytest.UsageError('The first argument to `pytest.mark.eval` must be a `pydantic_evals.Dataset`.')
    dataset = raw_dataset

    unknown_kwargs = set(mark.kwargs) - _MARKER_KWARGS
    if unknown_kwargs:
        unknown = ', '.join(sorted(unknown_kwargs))
        raise pytest.UsageError(f'Unexpected `pytest.mark.eval` keyword argument(s): {unknown}.')

    config_kwargs = {key: value for key, value in mark.kwargs.items() if key in _CONFIG_KWARGS}
    evaluate_kwargs = {key: value for key, value in mark.kwargs.items() if key in _EVALUATE_KWARGS}
    evaluate_kwargs.setdefault('progress', False)
    try:
        test_config = _EvaluationTestConfig(**config_kwargs)
    except ValueError as e:
        raise pytest.UsageError(str(e)) from e
    task_factory = mark.kwargs.get('task_factory', False)
    if not isinstance(task_factory, bool):
        raise pytest.UsageError('`pytest.mark.eval` keyword argument `task_factory` must be a boolean.')

    return _EvalMarkConfig(
        dataset=dataset,
        test_config=test_config,
        evaluate_kwargs=evaluate_kwargs,
        task_factory=task_factory,
    )


def _has_parametrize_mark(obj: object) -> bool:
    marks = getattr(obj, 'pytestmark', ())
    return any(isinstance(mark, pytest.Mark) and mark.name == 'parametrize' for mark in marks)


def _is_dataset(value: object) -> TypeGuard[Dataset[Any, Any, Any]]:
    return isinstance(value, Dataset)


def _is_direct_task(obj: Callable[..., Any]) -> bool:
    signature = inspect.signature(obj)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in {'self', 'cls'}:
        parameters = parameters[1:]
    return len(parameters) == 1 and parameters[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
