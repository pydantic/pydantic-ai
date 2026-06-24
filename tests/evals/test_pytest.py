from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EqualsExpected
    from pydantic_evals.pytest import assert_evaluation_passes_sync

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


def test_assert_evaluation_passes_sync_returns_report() -> None:
    dataset = Dataset[str, str, None](
        name='uppercase',
        cases=[Case(inputs='hello', expected_output='HELLO')],
        evaluators=[EqualsExpected()],
    )

    report = assert_evaluation_passes_sync(dataset, str.upper)

    assert report.name == 'upper'


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


def _run_inline_pytest(tmp_path: Path, source: str, *pytest_args: str) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / 'test_eval_marker.py'
    test_file.write_text(textwrap.dedent(source), encoding='utf-8')
    args = [sys.executable, '-m', 'pytest', str(test_file), '-q', *pytest_args]
    try:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_INLINE_PYTEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _subprocess_output(exc.stderr)
        if stderr:
            stderr = f'\n{stderr}'
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=_subprocess_output(exc.stdout),
            stderr=f'pytest timed out after {_INLINE_PYTEST_TIMEOUT}s{stderr}',
        )


def _subprocess_output(output: bytes | str | None) -> str:
    if output is None:
        return ''
    if isinstance(output, bytes):
        return output.decode(errors='replace')
    return output
