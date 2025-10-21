# Retry Strategies

Handle transient failures in tasks and evaluators with automatic retry logic.

## Overview

LLM-based systems can experience transient failures:
- Rate limits
- Network timeouts
- Temporary API outages
- Context length errors

Pydantic Evals supports retry configuration for both:
- **Task execution** - The function being evaluated
- **Evaluator execution** - The evaluators themselves

## Basic Retry Configuration

Pass a retry configuration to `evaluate()` or `evaluate_sync()`:

```python
from pydantic_evals import Case, Dataset


def my_function(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

report = dataset.evaluate_sync(
    task=my_function,
    retry_task={'max_attempts': 3},
    retry_evaluators={'max_attempts': 2},
)
```

## Retry Configuration Options

Retry configurations use [Tenacity](https://tenacity.readthedocs.io/) under the hood and support the same options as Pydantic AI's [`RetryConfig`](../../retries.md):

```python
from pydantic_evals import Case, Dataset


def my_function(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

retry_config = {
    'max_attempts': 3,  # Total attempts (default: 1, no retries)
    'initial_delay': 0.5,  # Initial delay in seconds (default: 0.5)
    'max_delay': 10.0,  # Maximum delay in seconds (default: 10)
    'timeout': 60.0,  # Total timeout in seconds (optional)
}

dataset.evaluate_sync(
    task=my_function,
    retry_task=retry_config,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_attempts` | int | 1 | Total number of attempts (1 = no retries) |
| `initial_delay` | float | 0.5 | Initial retry delay in seconds |
| `max_delay` | float | 10.0 | Maximum retry delay (uses exponential backoff) |
| `timeout` | float \| None | None | Total timeout for all attempts |

## Task Retries

Retry the task function when it fails:

```python
from pydantic_evals import Case, Dataset


async def call_llm(inputs: str) -> str:
    return f'LLM response to: {inputs}'


async def flaky_llm_task(inputs: str) -> str:
    """This might hit rate limits or timeout."""
    response = await call_llm(inputs)
    return response


dataset = Dataset(cases=[Case(inputs='test')])

report = dataset.evaluate_sync(
    task=flaky_llm_task,
    retry_task={
        'max_attempts': 5,  # Try up to 5 times
        'initial_delay': 1.0,  # Wait 1s before first retry
        'max_delay': 30.0,  # Cap delays at 30s
        'timeout': 120.0,  # Give up after 2 minutes total
    },
)
```

### When Task Retries Trigger

Retries trigger when the task raises an exception:

```python
class RateLimitError(Exception):
    pass


class ValidationError(Exception):
    pass


async def call_api(inputs: str) -> str:
    return f'API response: {inputs}'


async def my_task(inputs: str) -> str:
    try:
        return await call_api(inputs)
    except RateLimitError:
        # Will trigger retry
        raise
    except ValidationError:
        # Will also trigger retry
        raise
```

### Exponential Backoff

Delays increase exponentially with jitter:

```
Attempt 1: immediate
Attempt 2: ~0.5s delay
Attempt 3: ~1.0s delay
Attempt 4: ~2.0s delay
Attempt 5: ~4.0s delay (capped at max_delay)
```

## Evaluator Retries

Retry evaluators when they fail:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        # LLMJudge might hit rate limits
        LLMJudge(rubric='Response is accurate'),
    ],
)

report = dataset.evaluate_sync(
    task=my_task,
    retry_evaluators={
        'max_attempts': 3,
        'initial_delay': 0.5,
    },
)
```

### When Evaluator Retries Trigger

Retries trigger when an evaluator raises an exception:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


async def external_api_call(output: str) -> bool:
    return len(output) > 0


@dataclass
class APIEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        # If this raises an exception, retry logic will trigger
        result = await external_api_call(ctx.output)
        return result
```

### Evaluator Failures

If an evaluator fails after all retries, it's recorded as an [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure]:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

report = dataset.evaluate_sync(task, retry_evaluators={'max_attempts': 3})

# Check for evaluator failures
for case in report.cases:
    if case.evaluator_failures:
        for failure in case.evaluator_failures:
            print(f'Evaluator {failure.name} failed: {failure.error_message}')
    #> (No output - no evaluator failures in this case)
```

View evaluator failures in reports:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])
report = dataset.evaluate_sync(task)

report.print(include_evaluator_failures=True)
"""
  Evaluation Summary:
         task
┏━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━┩
│ Case 1   │     10ms │
├──────────┼──────────┤
│ Averages │     10ms │
└──────────┴──────────┘
"""
#>
#> ✅ case_0                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (0/0)
```

## Combining Task and Evaluator Retries

You can configure both independently:

```python
from pydantic_evals import Case, Dataset


def flaky_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

report = dataset.evaluate_sync(
    task=flaky_task,
    retry_task={
        'max_attempts': 5,  # Retry task up to 5 times
        'initial_delay': 1.0,
        'timeout': 120.0,
    },
    retry_evaluators={
        'max_attempts': 3,  # Retry evaluators up to 3 times
        'initial_delay': 0.5,
        'timeout': 30.0,
    },
)
```

## Practical Examples

### Rate Limit Handling

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


async def expensive_llm_call(inputs: str) -> str:
    return f'LLM response: {inputs}'


async def llm_task(inputs: str) -> str:
    """Task that might hit rate limits."""
    return await expensive_llm_call(inputs)


dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        LLMJudge(rubric='Quality check'),  # Also might hit rate limits
    ],
)

# Generous retries for rate limits
report = dataset.evaluate_sync(
    task=llm_task,
    retry_task={
        'max_attempts': 10,  # Rate limits can take multiple retries
        'initial_delay': 2.0,
        'max_delay': 60.0,  # Wait up to 1 minute between retries
        'timeout': 300.0,  # 5 minutes total timeout
    },
    retry_evaluators={
        'max_attempts': 5,
        'initial_delay': 2.0,
        'max_delay': 30.0,
    },
)
```

### Network Timeout Handling

```python
import httpx

from pydantic_evals import Case, Dataset


async def api_task(inputs: str) -> str:
    """Task that calls external API which might timeout."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post('https://api.example.com', json={'input': inputs})
        return response.text


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

# Quick retries for network issues
report = dataset.evaluate_sync(
    task=api_task,
    retry_task={
        'max_attempts': 4,  # A few quick retries
        'initial_delay': 0.5,  # Fast retry
        'max_delay': 5.0,  # Don't wait too long
        'timeout': 60.0,  # Give up after 1 minute
    },
)
```

### Context Length Handling

```python
from pydantic_evals import Case, Dataset


class ContextLengthError(Exception):
    pass


async def llm_call(inputs: str, max_tokens: int = 8000) -> str:
    return f'LLM response: {inputs[:100]}'


async def smart_llm_task(inputs: str) -> str:
    """Task that might exceed context length."""
    try:
        return await llm_call(inputs, max_tokens=8000)
    except ContextLengthError:
        # Retry with shorter context
        truncated_inputs = inputs[:4000]
        return await llm_call(truncated_inputs, max_tokens=4000)


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

# Don't retry context length errors (handle in task)
report = dataset.evaluate_sync(
    task=smart_llm_task,
    retry_task={'max_attempts': 1},  # No retries, we handle it
)
```

## Retry vs Error Handling

**Use retries for:**
- Transient failures (rate limits, timeouts)
- Network issues
- Temporary service outages
- Recoverable errors

**Use error handling for:**
- Validation errors
- Logic errors
- Permanent failures
- Expected error conditions

```python
class RateLimitError(Exception):
    pass


async def llm_call(inputs: str) -> str:
    return f'LLM response: {inputs}'


def is_valid(result: str) -> bool:
    return len(result) > 0


async def smart_task(inputs: str) -> str:
    """Handle expected errors, let retries handle transient failures."""
    try:
        result = await llm_call(inputs)

        # Validate output (don't retry validation errors)
        if not is_valid(result):
            return 'ERROR: Invalid output format'

        return result

    except RateLimitError:
        # Let retry logic handle this
        raise

    except ValueError as e:
        # Don't retry - this is a permanent error
        return f'ERROR: {e}'
```

## Monitoring Retries

### Check Total Duration

Retries increase execution time:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

report = dataset.evaluate_sync(task, retry_task={'max_attempts': 5})

for case in report.cases:
    print(f'{case.name}: {case.task_duration:.2f}s (includes retries)')
    #> case_0: 0.00s (includes retries)
```

### View in Logfire

If you're using Logfire, retries appear as separate spans:

```python
import logfire

from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


logfire.configure(send_to_logfire='if-token-present')

dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])
report = dataset.evaluate_sync(task, retry_task={'max_attempts': 3})

# View in Logfire web UI to see retry spans
```

## Best Practices

### 1. Be Conservative with max_attempts

More retries = longer evaluation time:

```python
# Development: fail fast
retry_config = {'max_attempts': 2}

# Production: be patient
retry_config = {'max_attempts': 5}
```

### 2. Use Different Configs for Tasks and Evaluators

Tasks might need more retries than evaluators:

```python
from pydantic_evals import Case, Dataset


def expensive_llm_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

report = dataset.evaluate_sync(
    task=expensive_llm_task,
    retry_task={'max_attempts': 5},  # Task is critical
    retry_evaluators={'max_attempts': 2},  # Evaluators less critical
)
```

### 3. Set Timeouts

Prevent hanging evaluations:

```python
retry_config = {
    'max_attempts': 10,
    'timeout': 120.0,  # Give up after 2 minutes, regardless of attempts
}
```

### 4. Log Retry Attempts

```python
from dataclasses import dataclass, field

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


async def flaky_check(output: str) -> bool:
    return len(output) > 0


@dataclass
class LoggingEvaluator(Evaluator):
    attempts: int = field(default=0, init=False)

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        self.attempts += 1
        print(f'Attempt {self.attempts}')

        result = await flaky_check(ctx.output)
        return result
```

### 5. Test Retry Logic

Simulate failures to test retry behavior:

```python {test="skip"}
from pydantic_evals import Case, Dataset


def my_task(inputs: str) -> str:
    """Task that demonstrates retry configuration.

    In a real scenario, this would fail intermittently due to
    rate limits or network issues, then succeed on retry.
    """
    return f'Success: {inputs}'


# Configure retries for handling transient failures
dataset = Dataset(cases=[Case(inputs='test')])

report = dataset.evaluate_sync(
    task=my_task,
    retry_task={'max_attempts': 3},  # Will retry up to 3 times if needed
)

# Check results
print(f'Cases evaluated: {len(report.cases)}')
#> Cases evaluated: 1
if report.cases:
    print(f'Output: {report.cases[0].output}')
    #> Output: Success: test
```

## Troubleshooting

### "Still failing after retries"

Increase `max_attempts` or check if error is retriable:

```python
import logging

from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


# Add logging to see what's failing
logging.basicConfig(level=logging.DEBUG)

dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

# Tenacity logs retry attempts
report = dataset.evaluate_sync(task, retry_task={'max_attempts': 5})
```

### "Evaluations taking too long"

Reduce `max_attempts` or `max_delay`:

```python
# Faster retries
retry_config = {
    'max_attempts': 3,  # Fewer attempts
    'initial_delay': 0.1,  # Quick retries
    'max_delay': 2.0,  # Low cap
    'timeout': 30.0,  # Overall timeout
}
```

### "Hitting rate limits despite retries"

Increase delays or use `max_concurrency`:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])

# Longer delays
retry_config = {
    'max_attempts': 5,
    'initial_delay': 5.0,  # Start with 5s delay
    'max_delay': 60.0,  # Up to 1 minute
}

# Also reduce concurrency
report = dataset.evaluate_sync(
    task=task,
    retry_task=retry_config,
    max_concurrency=2,  # Only 2 concurrent tasks
)
```

## Next Steps

- **[Performance Testing](performance-testing.md)** - Optimize evaluation performance
- **[Debugging](debugging.md)** - Debug failed evaluations
- **[Logfire Integration](logfire-integration.md)** - View retries in Logfire
