from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EqualsExpected, Evaluator, EvaluatorContext
    from pydantic_evals.evaluators.evaluator import EvaluationReason
    from pydantic_evals.evaluators.report_evaluator import ReportEvaluator, ReportEvaluatorContext
    from pydantic_evals.pytest import assert_evaluation_passes, assert_evaluation_passes_sync
    from pydantic_evals.reporting import EvaluationReport
    from pydantic_evals.reporting.analyses import ReportAnalysis

    @dataclass
    class _RaisingEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
            raise RuntimeError('evaluator boom')

    @dataclass
    class _RaisingReportEvaluator(ReportEvaluator[Any, Any, Any]):
        def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> ReportAnalysis:
            raise RuntimeError('report evaluator boom')

    @dataclass
    class _FailWithReason(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluationReason:
            return EvaluationReason(value=False, reason='because reasons')


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
]


def _module_exists(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


_INLINE_PYTEST_TIMEOUT = 120
_OTEL_SDK_INSTALLED = _module_exists('opentelemetry.sdk.trace')


def _passing_dataset() -> Dataset[str, str, None]:
    return Dataset[str, str, None](
        name='uppercase',
        cases=[Case(inputs='hello', expected_output='HELLO')],
        evaluators=[EqualsExpected()],
    )


def test_assert_evaluation_passes_sync_returns_report() -> None:
    report = assert_evaluation_passes_sync(_passing_dataset(), str.upper)

    assert report.name == 'upper'


async def test_assert_evaluation_passes_returns_report() -> None:
    async def upper(text: str) -> str:
        return text.upper()

    report = await assert_evaluation_passes(_passing_dataset(), upper)

    assert report.name == 'upper'


def test_assert_evaluation_passes_sync_invokes_report_check() -> None:
    seen: list[EvaluationReport[Any, Any, Any]] = []
    assert_evaluation_passes_sync(_passing_dataset(), str.upper, report_check=seen.append)

    assert len(seen) == 1
    assert seen[0].name == 'upper'


def test_assert_evaluation_passes_sync_fails_on_task_failures() -> None:
    def failing_task(text: str) -> str:
        raise ValueError('task boom')

    with pytest.raises(pytest.fail.Exception, match='1 task failure'):
        assert_evaluation_passes_sync(_passing_dataset(), failing_task)


def test_assert_evaluation_passes_sync_fails_on_evaluator_failures() -> None:
    dataset = Dataset[str, str, None](
        name='eval-fail',
        cases=[Case(inputs='hello', expected_output='HELLO')],
        evaluators=[EqualsExpected(), _RaisingEvaluator()],
    )

    with pytest.raises(pytest.fail.Exception, match='1 evaluator failure'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_assert_evaluation_passes_sync_fails_on_report_evaluator_failures() -> None:
    dataset = Dataset[str, str, None](
        name='report-eval-fail',
        cases=[Case(inputs='hello', expected_output='HELLO')],
        evaluators=[EqualsExpected()],
        report_evaluators=[_RaisingReportEvaluator()],
    )

    with pytest.raises(pytest.fail.Exception, match='1 report evaluator failure'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_assert_evaluation_passes_sync_requires_assertions() -> None:
    dataset = Dataset[str, str, None](name='no-evaluators', cases=[Case(inputs='hello')])

    with pytest.raises(pytest.fail.Exception, match='no assertions were produced'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_assert_evaluation_passes_sync_unavailable_assertion_pass_rate() -> None:
    dataset = Dataset[str, str, None](name='no-evaluators', cases=[Case(inputs='hello')])

    with pytest.raises(pytest.fail.Exception, match='assertion pass rate is unavailable'):
        assert_evaluation_passes_sync(dataset, str.upper, require_assertions=False)


def test_assert_evaluation_passes_sync_ignores_missing_assertions_when_disabled() -> None:
    dataset = Dataset[str, str, None](name='no-evaluators', cases=[Case(inputs='hello')])

    report = assert_evaluation_passes_sync(dataset, str.upper, require_assertions=False, min_assertion_pass_rate=None)

    assert report.name == 'upper'


def test_assert_evaluation_passes_sync_reports_failed_assertion_count() -> None:
    dataset = Dataset[str, str, None](
        name='wrong',
        cases=[Case(name='basic', inputs='hello', expected_output='NOPE')],
        evaluators=[EqualsExpected()],
    )

    with pytest.raises(pytest.fail.Exception, match='1 failed assertion'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_assert_evaluation_passes_sync_includes_assertion_reason() -> None:
    dataset = Dataset[str, str, None](
        name='reasoned',
        cases=[Case(name='basic', inputs='hello')],
        evaluators=[_FailWithReason()],
    )

    with pytest.raises(pytest.fail.Exception, match='basic: _FailWithReason - because reasons'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_assert_evaluation_passes_sync_truncates_failed_assertions() -> None:
    dataset = Dataset[str, str, None](
        name='many-wrong',
        cases=[Case(name=f'case-{i}', inputs='hello', expected_output='NOPE') for i in range(12)],
        evaluators=[EqualsExpected()],
    )

    with pytest.raises(pytest.fail.Exception, match='2 more'):
        assert_evaluation_passes_sync(dataset, str.upper)


def test_invalid_min_assertion_pass_rate_raises() -> None:
    with pytest.raises(ValueError, match='`min_assertion_pass_rate` must be between 0 and 1.'):
        assert_evaluation_passes_sync(_passing_dataset(), str.upper, min_assertion_pass_rate=2.0)


def test_eval_mark_uses_function_as_task(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'Pydantic Evals' in result.stdout
    assert 'Evaluation Summary: test_uppercase' in result.stdout


def test_eval_mark_reports_failed_assertions(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='NOPE')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 1
    assert "Evaluation 'test_uppercase' failed pytest criteria:" in result.stdout
    assert 'assertion pass rate 0.00 is below required 1.00' in result.stdout
    assert 'basic: EqualsExpected' in result.stdout


def test_eval_mark_supports_task_factory_with_fixtures(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='suffix',
            cases=[Case(name='basic', inputs='hello', expected_output='hello!')],
            evaluators=[EqualsExpected()],
        )


        @pytest.fixture
        def suffix() -> str:
            return '!'


        @pytest.mark.eval(dataset, task_factory=True)
        def test_suffix(suffix: str):
            def task(text: str) -> str:
                return text + suffix

            return task
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'Pydantic Evals' in result.stdout
    assert 'Evaluation Summary: task' in result.stdout


def test_eval_mark_supports_direct_task_methods(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        class TestUppercase:
            @pytest.mark.eval(dataset)
            async def test_uppercase(self, text: str) -> str:
                return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'Evaluation Summary: test_uppercase' in result.stdout


def test_eval_report_terminal_summary_can_be_disabled(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
        '--pydantic-evals-report=none',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Pydantic Evals' not in result.stdout
    assert 'Evaluation Summary: test_uppercase' not in result.stdout


def test_logfire_is_configured_automatically_for_eval_mark(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        def configure(**kwargs):
            print(f'LOGFIRE CONFIGURED: {kwargs}')


        sys.modules['logfire'] = types.SimpleNamespace(configure=configure, url_from_eval=lambda report: None)

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
        '-s',
        '--pydantic-evals-report=none',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOGFIRE CONFIGURED: {'send_to_logfire': 'if-token-present'" in result.stdout


def test_logfire_auto_config_can_be_disabled(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        def configure(**kwargs):
            raise RuntimeError('logfire should not be configured')


        sys.modules['logfire'] = types.SimpleNamespace(configure=configure, url_from_eval=lambda report: None)

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
        '--pydantic-evals-logfire=none',
        '--pydantic-evals-report=none',
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_terminal_summary_supports_logfire_without_eval_urls(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        sys.modules['logfire'] = types.SimpleNamespace(configure=lambda **kwargs: None)

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'Logfire dataset' not in result.stdout


def test_eval_mark_configures_otel_provider_without_logfire(tmp_path: Path) -> None:
    if not _OTEL_SDK_INSTALLED:
        pytest.skip('opentelemetry-sdk is not installed')

    result = _run_inline_pytest(
        tmp_path,
        """
        import builtins

        import pytest
        from opentelemetry import trace

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import HasMatchingSpan
        from pydantic_evals.otel import SpanQuery

        real_import = builtins.__import__


        def import_without_logfire(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'logfire' or name.startswith('logfire.'):
                raise ImportError('logfire is not installed')
            return real_import(name, globals, locals, fromlist, level)


        builtins.__import__ = import_without_logfire

        dataset = Dataset(
            name='spans',
            cases=[Case(name='basic', inputs='hello')],
            evaluators=[HasMatchingSpan(SpanQuery(name='task'))],
        )


        @pytest.mark.eval(dataset)
        def test_span(text: str) -> str:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span('task'):
                return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'SpanTreeRecordingError' not in result.stdout + result.stderr


def test_eval_mark_keeps_existing_tracer_provider(tmp_path: Path) -> None:
    if not _OTEL_SDK_INSTALLED:
        pytest.skip('opentelemetry-sdk is not installed')

    result = _run_inline_pytest(
        tmp_path,
        """
        import builtins

        import pytest
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        real_import = builtins.__import__


        def import_without_logfire(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'logfire' or name.startswith('logfire.'):
                raise ImportError('logfire is not installed')
            return real_import(name, globals, locals, fromlist, level)


        builtins.__import__ = import_without_logfire

        existing = TracerProvider()
        trace.set_tracer_provider(existing)

        assert trace.get_tracer_provider() is existing

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            assert trace.get_tracer_provider() is existing
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_eval_mark_without_otel_sdk(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import builtins

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        real_import = builtins.__import__


        def import_without_otel_sdk(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'logfire' or name.startswith('logfire.'):
                raise ImportError('logfire is not installed')
            if name == 'opentelemetry.sdk.trace':
                raise ImportError('opentelemetry-sdk is not installed')
            return real_import(name, globals, locals, fromlist, level)


        builtins.__import__ = import_without_otel_sdk

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_terminal_summary_prints_logfire_eval_links(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        sys.modules['logfire'] = types.SimpleNamespace(
            configure=lambda **kwargs: None,
            url_from_eval=lambda report: 'https://logfire-us.pydantic.dev/kludex/potato/evals/compare?experiment=abc'
        )

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Logfire dataset (uppercase): ' in result.stdout
    assert '\033]8;;https://logfire-us.pydantic.dev/kludex/potato/evals/uppercase\033\\' in result.stdout
    assert '\033[36mhttps://logfire-us.pydantic.dev/kludex/potato/evals/uppercase\033[0m' in result.stdout
    assert '\033[0m\033]8;;\033\\\n\nResults' in result.stdout
    assert 'Logfire evals overview' not in result.stdout
    assert 'compare?experiment=abc' not in result.stdout


_DATASET_HEADER = """\
import pytest

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    name='uppercase',
    cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
    evaluators=[EqualsExpected()],
)
"""


def _with_dataset(body: str) -> str:
    return _DATASET_HEADER + textwrap.dedent(body)


def test_eval_mark_rejects_parametrize(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset)
            @pytest.mark.parametrize('value', [1, 2])
            async def test_uppercase(text: str, value: int) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert '`pytest.mark.eval` cannot be combined with `pytest.mark.parametrize`.' in result.stdout + result.stderr


def test_eval_mark_rejects_multiple_markers(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset)
            @pytest.mark.eval(dataset)
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert 'Only one `pytest.mark.eval` marker can be applied to a test.' in result.stdout + result.stderr


def test_eval_mark_requires_single_positional_argument(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset, dataset)
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert '`pytest.mark.eval` requires exactly one positional argument: the dataset.' in result.stdout + result.stderr


def test_eval_mark_requires_dataset_argument(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval('not-a-dataset')
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert (
        'The first argument to `pytest.mark.eval` must be a `pydantic_evals.Dataset`.' in result.stdout + result.stderr
    )


def test_eval_mark_rejects_unknown_kwargs(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset, bogus=1, nope=2)
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert 'Unexpected `pytest.mark.eval` keyword argument(s): bogus, nope.' in result.stdout + result.stderr


def test_eval_mark_rejects_invalid_min_assertion_pass_rate(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset, min_assertion_pass_rate=2.0)
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
    )

    assert result.returncode != 0
    assert '`min_assertion_pass_rate` must be between 0 and 1.' in result.stdout + result.stderr


def test_eval_mark_rejects_non_boolean_task_factory(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset, task_factory='yes')
            def test_uppercase():
                return str.upper
            """
        ),
    )

    assert result.returncode != 0
    assert '`pytest.mark.eval` keyword argument `task_factory` must be a boolean.' in result.stdout + result.stderr


def test_eval_mark_supports_async_task_factory(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.fixture
            def suffix() -> str:
                return '!'


            @pytest.mark.eval(
                Dataset(
                    name='suffix',
                    cases=[Case(name='basic', inputs='hello', expected_output='hello!')],
                    evaluators=[EqualsExpected()],
                ),
                task_factory=True,
            )
            async def test_suffix(suffix: str, *args, **kwargs):
                def task(text: str) -> str:
                    return text + suffix

                return task
            """
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_eval_mark_task_factory_must_return_callable(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset, task_factory=True)
            def test_uppercase():
                return 'not callable'
            """
        ),
    )

    assert result.returncode != 0
    assert (
        'A `pytest.mark.eval(..., task_factory=True)` test must return the task callable.'
        in result.stdout + result.stderr
    )


def test_eval_report_failures_mode_skips_passing_reports(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        _with_dataset(
            """
            @pytest.mark.eval(dataset)
            async def test_uppercase(text: str) -> str:
                return text.upper()
            """
        ),
        '--pydantic-evals-report=failures',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'Pydantic Evals' not in result.stdout


def test_eval_report_failures_mode_shows_failing_reports(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='NOPE')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
        '--pydantic-evals-report=failures',
    )

    assert result.returncode == 1
    assert 'Pydantic Evals' in result.stdout


def test_terminal_summary_ignores_non_string_logfire_url(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        sys.modules['logfire'] = types.SimpleNamespace(
            configure=lambda **kwargs: None,
            url_from_eval=lambda report: 123,
        )

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Logfire dataset' not in result.stdout


def test_terminal_summary_ignores_logfire_url_without_evals_marker(tmp_path: Path) -> None:
    result = _run_inline_pytest(
        tmp_path,
        """
        import sys
        import types

        import pytest

        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EqualsExpected

        sys.modules['logfire'] = types.SimpleNamespace(
            configure=lambda **kwargs: None,
            url_from_eval=lambda report: 'https://logfire-us.pydantic.dev/kludex/potato',
        )

        dataset = Dataset(
            name='uppercase',
            cases=[Case(name='basic', inputs='hello', expected_output='HELLO')],
            evaluators=[EqualsExpected()],
        )


        @pytest.mark.eval(dataset)
        async def test_uppercase(text: str) -> str:
            return text.upper()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Logfire dataset' not in result.stdout


def _run_inline_pytest(tmp_path: Path, source: str, *pytest_args: str) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / 'test_eval_marker.py'
    test_file.write_text(textwrap.dedent(source), encoding='utf-8')
    args = [sys.executable, '-m', 'pytest', str(test_file), '-q', *pytest_args]
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=_INLINE_PYTEST_TIMEOUT,
    )
