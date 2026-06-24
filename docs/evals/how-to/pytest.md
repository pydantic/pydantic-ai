# Pytest Integration

Use `pytest.mark.eval` to run a [`Dataset`][pydantic_evals.dataset.Dataset] as part of a pytest suite.

Install the optional pytest integration:

```bash
pip install 'pydantic-evals[pytest]'
```

## Mark a Task Function

The recommended pattern is to mark the task function directly. The marked function receives each
[`Case.inputs`][pydantic_evals.dataset.Case.inputs] value and returns the task output.

```python
import pytest

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    name='uppercase',
    cases=[
        Case(name='basic', inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)


@pytest.mark.eval(dataset)
async def test_uppercase(text: str) -> str:
    return text.upper()
```

Pytest collects this as one test item. The plugin runs [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate]
with `progress=False` and fails the test if the evaluation report does not meet the configured criteria.

At the end of the pytest run, the plugin renders each eval report in a `Pydantic Evals` terminal-summary section.

By default, the test fails when:

- any task case raises an exception
- no assertions are produced
- the assertion pass rate is below `1.0`
- any evaluator raises an exception
- any report evaluator raises an exception

## Configure Pass Criteria

Set marker keyword arguments to adjust the pass criteria and forward evaluation options:

```python
import pytest

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    name='uppercase',
    cases=[
        Case(name='basic', inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)


@pytest.mark.eval(
    dataset,
    min_assertion_pass_rate=0.95,
    max_concurrency=5,
    repeat=3,
)
async def test_uppercase(text: str) -> str:
    return text.upper()
```

Most [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] keyword arguments can be passed through the marker,
including `name`, `max_concurrency`, `retry_task`, `retry_evaluators`, `metadata`, `repeat`, and `lifecycle`.

Use `--pydantic-evals-report` to control terminal-summary report rendering:

```bash
pytest --pydantic-evals-report=all      # default: render every eval report
pytest --pydantic-evals-report=failures # render only reports that failed eval criteria
pytest --pydantic-evals-report=none     # disable terminal-summary eval reports
```

Use `report_check` for project-specific checks that need the full
[`EvaluationReport`][pydantic_evals.reporting.EvaluationReport]:

```python
import pytest

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected
from pydantic_evals.reporting import EvaluationReport

dataset = Dataset(
    name='uppercase',
    cases=[
        Case(name='basic', inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)


def check_report(report: EvaluationReport[str, str, None]) -> None:
    averages = report.averages()
    assert averages is not None
    assert averages.metrics.get('cost', 0) < 1


@pytest.mark.eval(dataset, report_check=check_report)
async def test_uppercase(text: str) -> str:
    return text.upper()
```

## Use Pytest Fixtures

If the eval task needs pytest fixtures, use `task_factory=True`. In this mode, the marked function receives pytest
fixtures and returns the task callable.

```python
import pytest

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    name='suffix',
    cases=[
        Case(name='basic', inputs='hello', expected_output='hello!'),
    ],
    evaluators=[EqualsExpected()],
)


@pytest.fixture
def suffix() -> str:
    return '!'


@pytest.mark.eval(dataset, task_factory=True)
def test_suffix_eval(suffix: str):
    def task(text: str) -> str:
        return text + suffix

    return task
```

Use `task_factory=True` for any marked function parameters that should be resolved as pytest fixtures. Without it,
a single function parameter is treated as the dataset input argument.

## Helper Functions

In addition to the marker, [`assert_evaluation_passes()`][pydantic_evals.pytest.assert_evaluation_passes] and
[`assert_evaluation_passes_sync()`][pydantic_evals.pytest.assert_evaluation_passes_sync] can be used from ordinary
pytest tests:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected
from pydantic_evals.pytest import assert_evaluation_passes

dataset = Dataset(
    name='uppercase',
    cases=[
        Case(name='basic', inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)


async def test_uppercase_eval():
    await assert_evaluation_passes(dataset, str.upper)
```

## Logfire

When a test uses `pytest.mark.eval`, the plugin automatically configures Logfire for the pytest run if the `logfire`
package is installed:

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    service_name='pydantic-evals-pytest',
    environment='test',
)
```

Evals appear in Logfire when a Logfire token is available, for example from a local `.logfire` project configuration or
the `LOGFIRE_TOKEN` environment variable.

When Logfire provides a URL for the generated evaluation report, the pytest terminal summary includes a visible,
clickable terminal link for the dataset eval page:

```text
Logfire dataset (uppercase): https://logfire-us.pydantic.dev/my-org/my-project/evals/uppercase
```

To disable automatic Logfire configuration for eval-marked tests, use:

```bash
pytest --pydantic-evals-logfire=none
```

If your eval task runs Pydantic AI agents and you want agent/model/tool spans inside the eval trace, enable Pydantic AI
instrumentation in your test setup:

```python
import logfire


def pytest_configure() -> None:
    logfire.instrument_pydantic_ai()
```
