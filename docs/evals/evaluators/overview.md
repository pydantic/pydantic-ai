# Evaluators Overview

Evaluators are the core of Pydantic Evals. They analyze task outputs and provide scores, labels, or pass/fail assertions.

## When to Use Different Evaluators

### Deterministic Checks (Fast & Reliable)

Use deterministic evaluators when you can define exact rules:

| Evaluator | Use Case | Example |
|-----------|----------|---------|
| [`EqualsExpected`][pydantic_evals.evaluators.EqualsExpected] | Exact output match | Structured data, classification |
| [`Equals`][pydantic_evals.evaluators.Equals] | Equals specific value | Checking for sentinel values |
| [`Contains`][pydantic_evals.evaluators.Contains] | Substring/element check | Required keywords, PII detection |
| [`IsInstance`][pydantic_evals.evaluators.IsInstance] | Type validation | Format validation |
| [`MaxDuration`][pydantic_evals.evaluators.MaxDuration] | Performance threshold | SLA compliance |
| [`HasMatchingSpan`][pydantic_evals.evaluators.HasMatchingSpan] | Behavior verification | Tool calls, code paths |

**Advantages:**

- Fast execution (microseconds to milliseconds)
- Deterministic results
- No cost
- Easy to debug

**When to use:**

- Format validation (JSON structure, type checking)
- Required content checks (must contain X, must not contain Y)
- Performance requirements (latency, token counts)
- Behavioral checks (which tools were called, which code paths executed)

### LLM-as-a-Judge (Flexible & Nuanced)

Use [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] when evaluation requires understanding or judgment:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[Case(inputs='What is 2+2?', expected_output='4')],
    evaluators=[
        LLMJudge(
            rubric='Response is factually accurate based on the input',
            include_input=True,
        )
    ],
)
```

**Advantages:**

- Can evaluate subjective qualities (helpfulness, tone, creativity)
- Understands natural language
- Can follow complex rubrics
- Flexible across domains

**Disadvantages:**

- Slower (seconds per evaluation)
- Costs money
- Non-deterministic
- Can have biases

**When to use:**

- Factual accuracy
- Relevance and helpfulness
- Tone and style
- Completeness
- Following instructions
- RAG quality (groundedness, citation accuracy)

### Custom Evaluators

Custom evaluators can be useful if you want to make use of any evaluation logic we don't provide with the framework.
They are frequently useful for domain-specific logic:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ValidSQL(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        try:
            import sqlparse
            sqlparse.parse(ctx.output)
            return True
        except Exception:
            return False
```

**When to use:**

- Domain-specific validation (SQL syntax, regex patterns, business rules)
- External API calls (running generated code, checking databases)
- Complex calculations (precision/recall, BLEU scores)
- Integration checks (does API call succeed?)

## Evaluation Types

!!! info "Detailed Return Types Guide"
    For full detail about precisely what custom Evaluators may return, see [Custom Evaluator Return Types](custom.md#return-types).

Evaluators essentially return three types of results:

### 1. Assertions (bool)

Pass/fail checks that appear as ✔ or ✗ in reports:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class HasKeyword(Evaluator):
    keyword: str

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return self.keyword in ctx.output
```

**Use for:** Binary checks, quality gates, compliance requirements

### 2. Scores (int or float)

Numeric metrics:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ConfidenceScore(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        # Analyze and return score
        return 0.87  # 87% confidence
```

**Use for:** Quality metrics, ranking, A/B testing, regression tracking

### 3. Labels (str)

Categorical classifications:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class SentimentClassifier(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> str:
        if 'error' in ctx.output.lower():
            return 'error'
        elif 'success' in ctx.output.lower():
            return 'success'
        return 'neutral'
```

**Use for:** Classification, error categorization, quality buckets

### Multiple Results

You can return multiple evaluations from a single evaluator:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ComprehensiveCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        return {
            'valid_format': self._check_format(ctx.output),  # bool
            'quality_score': self._score_quality(ctx.output),  # float
            'category': self._classify(ctx.output),  # str
        }

    def _check_format(self, output: str) -> bool:
        return True

    def _score_quality(self, output: str) -> float:
        return 0.85

    def _classify(self, output: str) -> str:
        return 'good'
```

## Combining Evaluators

Mix and match evaluators to create comprehensive evaluation suites:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    IsInstance,
    LLMJudge,
    MaxDuration,
)

dataset = Dataset(
    cases=[Case(inputs='test', expected_output='result')],
    evaluators=[
        # Fast deterministic checks first
        IsInstance(type_name='str'),
        Contains(value='required_field'),
        MaxDuration(seconds=2.0),
        # Slower LLM checks after
        LLMJudge(
            rubric='Response is accurate and helpful',
            include_input=True,
        ),
    ],
)
```

## Case-specific evaluators

Case-specific evaluators are one of the most powerful features for building comprehensive evaluation suites. You can attach evaluators to individual [`Case`][pydantic_evals.Case] objects that only run for those specific cases:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='greeting_response',
            inputs='Say hello',
            evaluators=[
                # This evaluator only runs for this case
                LLMJudge(
                    rubric='Response is warm and friendly, uses casual tone',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='formal_response',
            inputs='Write a business email',
            evaluators=[
                # Different requirements for this case
                LLMJudge(
                    rubric='Response is professional and formal, uses business language',
                    include_input=True,
                ),
            ],
        ),
    ],
    evaluators=[
        # This runs for ALL cases
        IsInstance(type_name='str'),
    ],
)
```

### Why Case-Specific Evaluators Matter

Case-specific evaluators solve a fundamental problem with one-size-fits-all evaluation: **if you could write a single evaluator rubric that perfectly captured your requirements across all cases, you'd just incorporate that rubric into your agent's instructions**. (Note: this is less relevant in cases where you want to use a cheaper model in production and assess it using a more expensive model, but in many cases it makes sense to use the best model you can in production.)

The power of case-specific evaluation comes from the nuance:

- **Different cases have different requirements**: A customer support response needs empathy; a technical API response needs precision
- **Avoid "inmates running the asylum"**: If your LLMJudge rubric is generic enough to work everywhere, your agent should already be following it
- **Capture nuanced golden behavior**: Each case can specify exactly what "good" looks like for that scenario

### Building Golden Datasets with Case-Specific LLMJudge

A particularly powerful pattern is using case-specific [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] evaluators to quickly build comprehensive, maintainable evaluation suites. Instead of needing exact `expected_output` values, you can describe what you care about:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='handle_refund_request',
            inputs={'query': 'I want my money back', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Acknowledge the refund request empathetically
                    2. Ask for the reason for the refund
                    3. Mention our 30-day refund policy
                    4. NOT process the refund immediately (needs manager approval)
                    """,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='handle_shipping_question',
            inputs={'query': 'Where is my order?', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Confirm the order number
                    2. Provide tracking information
                    3. Give estimated delivery date
                    4. Be brief and factual (not overly apologetic)
                    """,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='handle_angry_customer',
            inputs={'query': 'This is completely unacceptable!', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Prioritize de-escalation with empathy
                    2. Avoid being defensive
                    3. Offer concrete next steps
                    4. Use phrases like "I understand" and "Let me help"
                    """,
                    include_input=True,
                ),
            ],
        ),
    ],
)
```

This approach lets you:

- **Build comprehensive test suites quickly**: Just describe what you want per case
- **Maintain easily**: Update rubrics as requirements change, without regenerating outputs
- **Cover edge cases naturally**: Add new cases with specific requirements as you discover them
- **Capture domain knowledge**: Each rubric documents what "good" means for that scenario

The LLM evaluator excels at understanding nuanced requirements and assessing compliance, making this a practical way to create thorough evaluation coverage without brittleness.

## Async vs Sync

Evaluators can be sync or async:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class SyncEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


async def some_async_operation() -> bool:
    return True


@dataclass
class AsyncEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        result = await some_async_operation()
        return result
```

Pydantic Evals handles both automatically. Use async when:
- Making API calls
- Running database queries
- Performing I/O operations
- Calling LLMs (like [`LLMJudge`][pydantic_evals.evaluators.LLMJudge])

## Evaluation Context

All evaluators receive an [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext]:

- `ctx.inputs` - Task inputs
- `ctx.output` - Task output (to evaluate)
- `ctx.expected_output` - Expected output (if provided)
- `ctx.metadata` - Case metadata (if provided)
- `ctx.duration` - Task execution time (seconds)
- `ctx.span_tree` - OpenTelemetry spans (if logfire configured)
- `ctx.metrics` - Custom metrics dict
- `ctx.attributes` - Custom attributes dict

This gives evaluators full context to make informed assessments.

## Error Handling

If an evaluator raises an exception, it's captured as an [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure]:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def risky_operation(output: str) -> bool:
    # This might raise an exception
    if 'error' in output:
        raise ValueError('Found error in output')
    return True


@dataclass
class RiskyEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        # If this raises an exception, it will be captured
        result = risky_operation(ctx.output)
        return result
```

Failures appear in `report.cases[i].evaluator_failures` with:

- Evaluator name
- Error message
- Full stacktrace

Use retry configuration to handle transient failures (see [Retry Strategies](../how-to/retry-strategies.md)).

## Next Steps

- **[Built-in Evaluators](built-in.md)** - Complete reference of all provided evaluators
- **[LLM Judge](llm-judge.md)** - Deep dive on LLM-as-a-Judge evaluation
- **[Custom Evaluators](custom.md)** - Write your own evaluation logic
- **[Span-Based Evaluation](span-based.md)** - Evaluate using OpenTelemetry spans
