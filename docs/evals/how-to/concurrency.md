# Concurrency & Performance

Control how evaluation cases are executed in parallel.

## Overview

By default, Pydantic Evals runs all cases concurrently to maximize throughput. You can control this behavior using the `max_concurrency` parameter.

## Basic Usage

```python
from pydantic_evals import Case, Dataset


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1'), Case(inputs='test2')])

# Run all cases concurrently (default)
report = dataset.evaluate_sync(my_task)

# Limit to 5 concurrent cases
report = dataset.evaluate_sync(my_task, max_concurrency=5)

# Run sequentially (one at a time)
report = dataset.evaluate_sync(my_task, max_concurrency=1)
```

## When to Limit Concurrency

### Rate Limiting

Many APIs have rate limits that restrict concurrent requests:

```python
from pydantic_evals import Case, Dataset


async def my_llm_task(inputs: str) -> str:
    return f'LLM Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1')])

# If your API allows 10 requests/second
report = dataset.evaluate_sync(
    my_llm_task,
    max_concurrency=10,
)
```

### Resource Constraints

Limit concurrency to avoid overwhelming system resources:

```python
from pydantic_evals import Case, Dataset


def heavy_computation(inputs: str) -> str:
    return f'Heavy: {inputs}'


def db_query_task(inputs: str) -> str:
    return f'DB: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1')])

# Memory-intensive operations
report = dataset.evaluate_sync(
    heavy_computation,
    max_concurrency=2,  # Only 2 at a time
)

# Database connection pool limits
report = dataset.evaluate_sync(
    db_query_task,
    max_concurrency=5,  # Match connection pool size
)
```

### Debugging

Run sequentially to see clear error traces:

```python
from pydantic_evals import Case, Dataset


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1')])

# Easier to debug
report = dataset.evaluate_sync(
    my_task,
    max_concurrency=1,
)
```

## Performance Comparison

Here's an example showing the performance difference:

```python {title="concurrency_example.py"}
import asyncio

from pydantic_evals import Case, Dataset

# Create a dataset with multiple test cases
dataset = Dataset(
    cases=[
        Case(
            name=f'case_{i}',
            inputs=i,
            expected_output=i * 2,
        )
        for i in range(10)
    ]
)


async def slow_task(input_value: int) -> int:
    """Simulates a slow operation (e.g., API call)."""
    await asyncio.sleep(0.1)  # 100ms per case
    return input_value * 2


# Unlimited concurrency: ~0.1s total (all cases run in parallel)
report = dataset.evaluate_sync(slow_task)

# Limited concurrency: ~0.5s total (2 at a time, 5 batches)
report = dataset.evaluate_sync(slow_task, max_concurrency=2)

# Sequential: ~1.0s total (one at a time, 10 cases)
report = dataset.evaluate_sync(slow_task, max_concurrency=1)
```

## Concurrency with Evaluators

Both task execution and evaluator execution happen concurrently by default:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(
    cases=[Case(inputs=f'test{i}') for i in range(100)],  # 100 cases
    evaluators=[
        LLMJudge(rubric='Quality check'),  # Makes API calls
    ],
)

# Both task and evaluator run with controlled concurrency
report = dataset.evaluate_sync(
    my_task,
    max_concurrency=10,
)
```

If your evaluators are expensive (e.g., [`LLMJudge`][pydantic_evals.evaluators.LLMJudge]), limiting concurrency helps manage:
- API rate limits
- Cost (fewer concurrent API calls)
- Memory usage

## Async vs Sync

Both sync and async evaluation support concurrency control:

### Sync API

```python
from pydantic_evals import Case, Dataset


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1')])

# Runs async operations internally with controlled concurrency
report = dataset.evaluate_sync(my_task, max_concurrency=10)
```

### Async API

```python
from pydantic_evals import Case, Dataset


async def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


async def run_evaluation():
    dataset = Dataset(cases=[Case(inputs='test1')])
    # Same behavior, but in async context
    report = await dataset.evaluate(my_task, max_concurrency=10)
    return report
```

## Monitoring Concurrency

Track execution to optimize settings:

```python {test="skip"}
import time

from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs=f'test{i}') for i in range(10)])

t0 = time.time()
report = dataset.evaluate_sync(task, max_concurrency=10)
duration = time.time() - t0

num_cases = len(report.cases) + len(report.failures)
avg_duration = duration / num_cases

print(f'Total: {duration:.2f}s')
#> Total: 0.01s
print(f'Cases: {num_cases}')
#> Cases: 10
print(f'Avg per case: {avg_duration:.2f}s')
#> Avg per case: 0.00s
print(f'Effective concurrency: ~{num_cases * avg_duration / duration:.1f}')
#> Effective concurrency: ~1.0
```

## Handling Rate Limits

If you hit rate limits, the evaluation will fail. Use retry strategies:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test1')])

# Reduce concurrency to avoid rate limits
report = dataset.evaluate_sync(
    task,
    max_concurrency=5,  # Stay under rate limit
)
```

See [Retry Strategies](retry-strategies.md) for handling transient failures.

## Next Steps

- **[Retry Strategies](retry-strategies.md)** - Handle transient failures
- **[Dataset Management](dataset-management.md)** - Work with large datasets
- **[Logfire Integration](logfire-integration.md)** - Monitor performance
