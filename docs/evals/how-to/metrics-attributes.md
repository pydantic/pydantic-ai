# Metrics & Attributes

Track custom metrics and attributes during task execution for richer evaluation insights.

## Overview

While executing evaluation tasks, you can record:

- **Metrics** - Numeric values (int/float) for quantitative measurements
- **Attributes** - Any data for qualitative information

These appear in evaluation reports and can be used by evaluators for assessment.

## Recording Metrics

Use [`increment_eval_metric`][pydantic_evals.increment_eval_metric] to track numeric values:

```python
from dataclasses import dataclass

from pydantic_evals.dataset import increment_eval_metric


@dataclass
class APIResult:
    output: str
    usage: 'Usage'


@dataclass
class Usage:
    total_tokens: int


def call_api(inputs: str) -> APIResult:
    return APIResult(output=f'Result: {inputs}', usage=Usage(total_tokens=100))


def my_task(inputs: str) -> str:
    # Track API calls
    increment_eval_metric('api_calls', 1)

    result = call_api(inputs)

    # Track tokens used
    increment_eval_metric('tokens_used', result.usage.total_tokens)

    return result.output
```

## Recording Attributes

Use [`set_eval_attribute`][pydantic_evals.set_eval_attribute] to store any data:

```python
from pydantic_evals import set_eval_attribute


def process(inputs: str) -> str:
    return f'Processed: {inputs}'


def my_task(inputs: str) -> str:
    # Record which model was used
    set_eval_attribute('model', 'gpt-5')

    # Record feature flags
    set_eval_attribute('used_cache', True)
    set_eval_attribute('retry_count', 2)

    # Record structured data
    set_eval_attribute('config', {
        'temperature': 0.7,
        'max_tokens': 100,
    })

    return process(inputs)
```

## Accessing in Evaluators

Metrics and attributes are available in the [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext]:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class EfficiencyChecker(Evaluator):
    max_api_calls: int = 5

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool]:
        # Access metrics
        api_calls = ctx.metrics.get('api_calls', 0)
        tokens_used = ctx.metrics.get('tokens_used', 0)

        # Access attributes
        used_cache = ctx.attributes.get('used_cache', False)

        return {
            'efficient_api_usage': api_calls <= self.max_api_calls,
            'used_caching': used_cache,
            'token_efficient': tokens_used < 1000,
        }
```

## Viewing in Reports

Metrics and attributes appear in report data:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])
report = dataset.evaluate_sync(task)

for case in report.cases:
    print(f'{case.name}:')
    #> Case 1:
    print(f'  Metrics: {case.metrics}')
    #>   Metrics: {}
    print(f'  Attributes: {case.attributes}')
    #>   Attributes: {}
```

You can also display them in printed reports:

```python
from pydantic_evals import Case, Dataset


def task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(cases=[Case(inputs='test')], evaluators=[])
report = dataset.evaluate_sync(task)

# Metrics and attributes are available but not shown by default
# Access them programmatically or via Logfire

for case in report.cases:
    print(f'\nCase: {case.name}')
    """
    Case: Case 1
    """
    print(f'Metrics: {case.metrics}')
    #> Metrics: {}
    print(f'Attributes: {case.attributes}')
    #> Attributes: {}
```

## Automatic Metrics

When using Pydantic AI and Logfire, some metrics are automatically tracked:

```python
import logfire

from pydantic_ai import Agent

logfire.configure(send_to_logfire='if-token-present')

agent = Agent('openai:gpt-5')


async def ai_task(inputs: str) -> str:
    result = await agent.run(inputs)
    return result.output


# Automatically tracked metrics:
# - requests: Number of LLM calls
# - input_tokens: Total input tokens
# - output_tokens: Total output tokens
# - prompt_tokens: Prompt tokens (if available)
# - completion_tokens: Completion tokens (if available)
# - cost: Estimated cost (if using genai-prices)
```

Access these in evaluators:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CostChecker(Evaluator):
    max_cost: float = 0.01  # $0.01

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        cost = ctx.metrics.get('cost', 0.0)
        return cost <= self.max_cost
```

## Practical Examples

### API Usage Tracking

```python
from dataclasses import dataclass

from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def check_cache(inputs: str) -> str | None:
    return None  # No cache hit for demo


@dataclass
class APIResult:
    text: str
    usage: 'Usage'


@dataclass
class Usage:
    total_tokens: int


async def call_api(inputs: str) -> APIResult:
    return APIResult(text=f'Result: {inputs}', usage=Usage(total_tokens=100))


def save_to_cache(inputs: str, result: str) -> None:
    pass  # Save to cache


async def smart_task(inputs: str) -> str:
    # Try cache first
    if cached := check_cache(inputs):
        set_eval_attribute('cache_hit', True)
        return cached

    set_eval_attribute('cache_hit', False)

    # Call API
    increment_eval_metric('api_calls', 1)
    result = await call_api(inputs)

    increment_eval_metric('tokens', result.usage.total_tokens)

    # Cache result
    save_to_cache(inputs, result.text)

    return result.text


# Evaluate efficiency
@dataclass
class EfficiencyEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float]:
        api_calls = ctx.metrics.get('api_calls', 0)
        cache_hit = ctx.attributes.get('cache_hit', False)

        return {
            'used_cache': cache_hit,
            'made_api_call': api_calls > 0,
            'efficiency_score': 1.0 if cache_hit else 0.5,
        }
```

### Tool Usage Tracking

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

agent = Agent('openai:gpt-5')


def search(query: str) -> str:
    return f'Search results for: {query}'


def call(endpoint: str) -> str:
    return f'API response from: {endpoint}'


@agent.tool
def search_database(ctx: RunContext, query: str) -> str:
    increment_eval_metric('db_searches', 1)
    set_eval_attribute('last_query', query)
    return search(query)


@agent.tool
def call_api(ctx: RunContext, endpoint: str) -> str:
    increment_eval_metric('api_calls', 1)
    set_eval_attribute('last_endpoint', endpoint)
    return call(endpoint)


# Evaluate tool usage
@dataclass
class ToolUsageEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | int]:
        db_searches = ctx.metrics.get('db_searches', 0)
        api_calls = ctx.metrics.get('api_calls', 0)

        return {
            'used_database': db_searches > 0,
            'used_api': api_calls > 0,
            'tool_call_count': db_searches + api_calls,
            'reasonable_tool_usage': (db_searches + api_calls) <= 5,
        }
```

### Performance Tracking

```python
import time
from dataclasses import dataclass

from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


async def retrieve_context(inputs: str) -> list[str]:
    return ['context1', 'context2']


async def generate_response(context: list[str], inputs: str) -> str:
    return f'Generated response for {inputs}'


async def monitored_task(inputs: str) -> str:
    # Track sub-operation timing
    t0 = time.perf_counter()
    context = await retrieve_context(inputs)
    retrieve_time = time.perf_counter() - t0

    increment_eval_metric('retrieve_time', retrieve_time)

    t0 = time.perf_counter()
    result = await generate_response(context, inputs)
    generate_time = time.perf_counter() - t0

    increment_eval_metric('generate_time', generate_time)

    # Record which operations were needed
    set_eval_attribute('needed_retrieval', len(context) > 0)
    set_eval_attribute('context_chunks', len(context))

    return result


# Evaluate performance
@dataclass
class PerformanceEvaluator(Evaluator):
    max_retrieve_time: float = 0.5
    max_generate_time: float = 2.0

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool]:
        retrieve_time = ctx.metrics.get('retrieve_time', 0.0)
        generate_time = ctx.metrics.get('generate_time', 0.0)

        return {
            'fast_retrieval': retrieve_time <= self.max_retrieve_time,
            'fast_generation': generate_time <= self.max_generate_time,
        }
```

### Quality Tracking

```python
from dataclasses import dataclass

from pydantic_evals import set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


async def llm_call(inputs: str) -> dict:
    return {'text': f'Response: {inputs}', 'confidence': 0.85, 'sources': ['doc1', 'doc2']}


async def quality_task(inputs: str) -> str:
    result = await llm_call(inputs)

    # Extract quality indicators
    confidence = result.get('confidence', 0.0)
    sources_used = result.get('sources', [])

    set_eval_attribute('confidence', confidence)
    set_eval_attribute('source_count', len(sources_used))
    set_eval_attribute('sources', sources_used)

    return result['text']


# Evaluate based on quality signals
@dataclass
class QualityEvaluator(Evaluator):
    min_confidence: float = 0.7

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float]:
        confidence = ctx.attributes.get('confidence', 0.0)
        source_count = ctx.attributes.get('source_count', 0)

        return {
            'high_confidence': confidence >= self.min_confidence,
            'used_sources': source_count > 0,
            'quality_score': confidence * (1.0 + 0.1 * source_count),
        }
```


## Experiment-Level Metadata

In addition to case-level metadata, you can also pass experiment-level metadata when calling [`evaluate()`][pydantic_evals.dataset.Dataset.evaluate]:

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(
    cases=[
        Case(
            inputs='test',
            metadata={'difficulty': 'easy'},  # Case-level metadata
        )
    ]
)


async def task(inputs: str) -> str:
    return f'Result: {inputs}'


# Pass experiment-level metadata
async def main():
    report = await dataset.evaluate(
        task,
        metadata={
            'model': 'gpt-4o',
            'prompt_version': 'v2.1',
            'temperature': 0.7,
        },
    )

    # Access experiment metadata in the report
    print(report.experiment_metadata)
    #> {'model': 'gpt-4o', 'prompt_version': 'v2.1', 'temperature': 0.7}
```

### When to Use Experiment Metadata

Experiment metadata is useful for tracking configuration that applies to the entire evaluation run:

- **Model configuration**: Model name, version, parameters
- **Prompt versioning**: Which prompt template was used
- **Infrastructure**: Deployment environment, region
- **Experiment context**: Developer name, feature branch, commit hash

This metadata is especially valuable when:

- Comparing multiple evaluation runs over time
- Tracking which configuration produced which results
- Reproducing evaluation results from historical data

### Viewing in Reports

Experiment metadata appears at the top of printed reports:

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(inputs='hello', expected_output='HELLO')])


async def task(text: str) -> str:
    return text.upper()

async def main():
    report = await dataset.evaluate(
        task,
        metadata={'model': 'gpt-4o', 'version': 'v1.0'},
    )

    print(report.render())
    """
    ╭─ Evaluation Summary: task ─╮
    │ model: gpt-4o              │
    │ version: v1.0              │
    ╰────────────────────────────╯
    ┏━━━━━━━━━━┳━━━━━━━━━━┓
    ┃ Case ID  ┃ Duration ┃
    ┡━━━━━━━━━━╇━━━━━━━━━━┩
    │ Case 1   │     10ms │
    ├──────────┼──────────┤
    │ Averages │     10ms │
    └──────────┴──────────┘
    """
```

## Synchronization between Tasks and Experiment Metadata

Experiment metadata is for *recording* configuration, not *configuring* the task.
The metadata dict doesn't automatically configure your task's behavior; you must ensure the values in the metadata dict match what your task actually uses.
For example, it's easy to accidentally have metadata claim `temperature: 0.7` while your task actually uses `temperature: 1.0`, leading to incorrect experiment tracking and unreproducible results.

To avoid this problem, we recommend establishing a single source of truth for configuration that both your task and metadata reference.
Below are a few suggested patterns for achieving this synchronization.

### Pattern 1: Shared Module Constants

For simpler cases, use module-level constants:

```python
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset

# Module constants as single source of truth
MODEL_NAME = 'openai:gpt-5-mini'
TEMPERATURE = 0.7
INSTRUCTIONS = 'You are a helpful assistant.'

agent = Agent(MODEL_NAME, model_settings={'temperature': TEMPERATURE}, instructions=INSTRUCTIONS)


async def task(inputs: str) -> str:
    result = await agent.run(inputs)
    return result.output


async def main():
    dataset = Dataset(cases=[Case(inputs='What is the capital of France?')])

    # Metadata references same constants
    await dataset.evaluate(
        task,
        metadata={
            'model': MODEL_NAME,
            'temperature': TEMPERATURE,
            'instructions': INSTRUCTIONS,
        },
    )
```

### Pattern 2: Configuration Object (Recommended)

Define configuration once and use it everywhere:

```python
from dataclasses import asdict, dataclass

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset


@dataclass
class TaskConfig:
    """Single source of truth for task configuration.

    Includes all variables you'd like to see in experiment metadata.
    """

    model: str
    temperature: float
    max_tokens: int
    prompt_version: str


# Define configuration once
config = TaskConfig(
    model='openai:gpt-5-mini',
    temperature=0.7,
    max_tokens=500,
    prompt_version='v2.1',
)

# Use config in task
agent = Agent(
    config.model,
    model_settings={'temperature': config.temperature, 'max_tokens': config.max_tokens},
)


async def task(inputs: str) -> str:
    """Task uses the same config that's recorded in metadata."""
    result = await agent.run(inputs)
    return result.output


# Evaluate with metadata derived from the same config
async def main():
    dataset = Dataset(cases=[Case(inputs='What is the capital of France?')])

    report = await dataset.evaluate(
        task,
        metadata=asdict(config),  # Guaranteed to match task behavior
    )

    print(report.experiment_metadata)
    """
    {
        'model': 'openai:gpt-5-mini',
        'temperature': 0.7,
        'max_tokens': 500,
        'prompt_version': 'v2.1',
    }
    """
```

If it's problematic to have a global task configuration, you can also create your `TaskConfig` object at the task
call-site and pass it to the agent via `deps` or similar, but in this case you would still need to guarantee that the
value is always the same as the value passed to `metadata` in the call to `Dataset.evaluate`.

### Anti-Pattern: Duplicate Configuration

**Avoid this common mistake**:

```python
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset

# ❌ BAD: Configuration defined in multiple places
agent = Agent('openai:gpt-5-mini', model_settings={'temperature': 0.7})


async def task(inputs: str) -> str:
    result = await agent.run(inputs)
    return result.output


async def main():
    dataset = Dataset(cases=[Case(inputs='test')])

    # ❌ BAD: Metadata manually typed - easy to get out of sync
    await dataset.evaluate(
        task,
        metadata={
            'model': 'openai:gpt-5-mini',  # Duplicated! Could diverge from agent definition
            'temperature': 0.8,  # ⚠️ WRONG! Task actually uses 0.7
        },
    )
```

In this anti-pattern, the metadata claims `temperature: 0.8` but the task uses `0.7`. This leads to:

- Incorrect experiment tracking
- Inability to reproduce results
- Confusion when comparing runs
- Wasted time debugging "why results differ"

## Metrics vs Attributes vs Metadata

Understanding the differences:

| Feature | Metrics | Attributes | Case Metadata | Experiment Metadata |
|---------|---------|------------|---------------|---------------------|
| **Set in** | Task execution | Task execution | Case definition | `evaluate()` call |
| **Type** | int, float | Any | Any | Any |
| **Purpose** | Quantitative | Qualitative | Test data | Experiment config |
| **Used for** | Aggregation | Context | Input to task | Tracking runs |
| **Available to** | Evaluators | Evaluators | Task & Evaluators | Report only |
| **Scope** | Per case | Per case | Per case | Per experiment |

```python
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute

# Case Metadata: Defined in case (before execution)
case = Case(
    inputs='question',
    metadata={'difficulty': 'hard', 'category': 'math'},  # Per-case metadata
)

dataset = Dataset(cases=[case])


# Metrics & Attributes: Recorded during execution
async def task(inputs):
    # These are recorded during execution for each case
    increment_eval_metric('tokens', 100)
    set_eval_attribute('model', 'gpt-5')
    return f'Result: {inputs}'


async def main():
    # Experiment Metadata: Defined at evaluation time
    await dataset.evaluate(
        task,
        metadata={  # Experiment-level metadata
            'prompt_version': 'v2.1',
            'temperature': 0.7,
        },
    )
```

## Troubleshooting

### "Metrics/attributes not appearing"

Ensure you're calling the functions inside the task:

```python
from pydantic_evals import increment_eval_metric


def process(inputs: str) -> str:
    return f'Processed: {inputs}'


# Bad: Called outside task
increment_eval_metric('count', 1)


def bad_task(inputs):
    return process(inputs)


# Good: Called inside task
def good_task(inputs):
    increment_eval_metric('count', 1)
    return process(inputs)
```

### "Metrics not incrementing"

Check you're using `increment_eval_metric`, not `set_eval_attribute`:

```python
from pydantic_evals import increment_eval_metric, set_eval_attribute

# Bad: This will overwrite, not increment
set_eval_attribute('count', 1)
set_eval_attribute('count', 1)  # Still 1

# Good: This increments
increment_eval_metric('count', 1)
increment_eval_metric('count', 1)  # Now 2
```

### "Too much data in attributes"

Store summaries, not raw data:

```python
from pydantic_evals import set_eval_attribute

giant_response_object = {'key' + str(i): 'value' * 100 for i in range(1000)}

# Bad: Huge object
set_eval_attribute('full_response', giant_response_object)

# Good: Summary
set_eval_attribute('response_size_kb', len(str(giant_response_object)) / 1024)
set_eval_attribute('response_keys', list(giant_response_object.keys())[:10])  # First 10 keys
```

## Next Steps

- **[Custom Evaluators](../evaluators/custom.md)** - Use metrics/attributes in evaluators
- **[Logfire Integration](logfire-integration.md)** - View metrics in Logfire
- **[Concurrency & Performance](concurrency.md)** - Optimize evaluation performance
