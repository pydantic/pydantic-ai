# Custom Evaluators

Write custom evaluators for domain-specific logic, external integrations, or specialized metrics.

## Basic Custom Evaluator

All evaluators inherit from [`Evaluator`][pydantic_evals.evaluators.Evaluator] and must implement `evaluate`:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    """Check if output exactly matches expected output."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output
```

**Key Points:**

- Use `@dataclass` decorator (required)
- Inherit from `Evaluator`
- Implement `evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput`
- Return `bool`, `int`, `float`, `str`, [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason], or `dict` of these

## EvaluatorContext

The context provides all information about the case execution:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Access case data
        ctx.name              # Case name
        ctx.inputs            # Task inputs
        ctx.metadata          # Case metadata
        ctx.expected_output   # Expected output (may be None)
        ctx.output            # Actual output

        # Performance data
        ctx.duration          # Task execution time (seconds)

        # Custom metrics/attributes (see metrics guide)
        ctx.metrics           # dict[str, int | float]
        ctx.attributes        # dict[str, Any]

        # OpenTelemetry spans (if logfire configured)
        ctx.span_tree         # SpanTree for behavioral checks

        return True
```

## Evaluator Parameters

Add configurable parameters as dataclass fields:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ContainsKeyword(Evaluator):
    keyword: str
    case_sensitive: bool = True

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = ctx.output
        keyword = self.keyword

        if not self.case_sensitive:
            output = output.lower()
            keyword = keyword.lower()

        return keyword in output


# Usage
dataset = Dataset(
    cases=[Case(name='test', inputs='This is important')],
    evaluators=[
        ContainsKeyword(keyword='important', case_sensitive=False),
    ],
)
```

## Return Types

### Boolean Assertions

Simple pass/fail checks:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class IsValidJSON(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        try:
            import json
            json.loads(ctx.output)
            return True
        except Exception:
            return False
```

### Numeric Scores

Quality metrics:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class LengthScore(Evaluator):
    """Score based on output length (0.0 = too short, 1.0 = ideal)."""

    ideal_length: int = 100
    tolerance: int = 20

    def evaluate(self, ctx: EvaluatorContext) -> float:
        length = len(ctx.output)
        diff = abs(length - self.ideal_length)

        if diff <= self.tolerance:
            return 1.0
        else:
            # Decay score as we move away from ideal
            score = max(0.0, 1.0 - (diff - self.tolerance) / self.ideal_length)
            return score
```

### String Labels

Categorical classifications:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class SentimentClassifier(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> str:
        output_lower = ctx.output.lower()

        if any(word in output_lower for word in ['error', 'failed', 'wrong']):
            return 'negative'
        elif any(word in output_lower for word in ['success', 'correct', 'great']):
            return 'positive'
        else:
            return 'neutral'
```

### With Reasons

Add explanations to any result:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class SmartCheck(Evaluator):
    threshold: float = 0.8

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        score = self._calculate_score(ctx.output)

        if score >= self.threshold:
            return EvaluationReason(
                value=True,
                reason=f'Score {score:.2f} exceeds threshold {self.threshold}',
            )
        else:
            return EvaluationReason(
                value=False,
                reason=f'Score {score:.2f} below threshold {self.threshold}',
            )

    def _calculate_score(self, output: str) -> float:
        # Your scoring logic
        return 0.75
```

### Multiple Results

Return multiple evaluations from one evaluator:

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
        return output.startswith('{') and output.endswith('}')

    def _score_quality(self, output: str) -> float:
        return len(output) / 100.0

    def _classify(self, output: str) -> str:
        return 'short' if len(output) < 50 else 'long'
```

## Async Evaluators

Use `async def` for I/O-bound operations:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class APIValidator(Evaluator):
    api_url: str

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json={'output': ctx.output},
            )
            return response.json()['valid']
```

Pydantic Evals handles both sync and async evaluators automatically.

## Using Metadata

Access case metadata for context-aware evaluation:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class DifficultyAwareScore(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        # Base score
        base_score = self._score_output(ctx.output)

        # Adjust based on difficulty from metadata
        if ctx.metadata and 'difficulty' in ctx.metadata:
            difficulty = ctx.metadata['difficulty']

            if difficulty == 'easy':
                # Penalize mistakes more on easy questions
                return base_score
            elif difficulty == 'hard':
                # Be more lenient on hard questions
                return min(1.0, base_score * 1.2)

        return base_score

    def _score_output(self, output: str) -> float:
        # Your scoring logic
        return 0.8
```

## Using Metrics

Access custom metrics set during task execution:

```python
from dataclasses import dataclass

from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


# In your task
def my_task(inputs: str) -> str:
    result = f'processed: {inputs}'

    # Record metrics
    increment_eval_metric('api_calls', 3)
    set_eval_attribute('used_cache', True)

    return result


# In your evaluator
@dataclass
class EfficiencyCheck(Evaluator):
    max_api_calls: int = 5

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        api_calls = ctx.metrics.get('api_calls', 0)
        return api_calls <= self.max_api_calls
```

See [Metrics & Attributes Guide](../how-to/metrics-attributes.md) for more.

## Generic Type Parameters

Make evaluators type-safe with generics:

```python
from dataclasses import dataclass
from typing import TypeVar

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

InputsT = TypeVar('InputsT')
OutputT = TypeVar('OutputT')


@dataclass
class TypedEvaluator(Evaluator[InputsT, OutputT, dict]):
    def evaluate(self, ctx: EvaluatorContext[InputsT, OutputT, dict]) -> bool:
        # ctx.inputs and ctx.output are now properly typed
        return True
```

## Custom Evaluation Names

Control how evaluations appear in reports:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CustomNameEvaluator(Evaluator):
    check_type: str

    def get_default_evaluation_name(self) -> str:
        # Use check_type as the name instead of class name
        return f'{self.check_type}_check'

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# In reports, appears as "format_check" instead of "CustomNameEvaluator"
evaluator = CustomNameEvaluator(check_type='format')
```

Or use the `evaluation_name` field (if using the built-in pattern):

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyEvaluator(Evaluator):
    evaluation_name: str | None = None

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# Usage
MyEvaluator(evaluation_name='my_custom_name')
```

## Real-World Examples

### SQL Validation

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class ValidSQL(Evaluator):
    dialect: str = 'postgresql'

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        try:
            import sqlparse
            parsed = sqlparse.parse(ctx.output)

            if not parsed:
                return EvaluationReason(
                    value=False,
                    reason='Could not parse SQL',
                )

            # Check for dangerous operations
            sql_upper = ctx.output.upper()
            if 'DROP' in sql_upper or 'DELETE' in sql_upper:
                return EvaluationReason(
                    value=False,
                    reason='Contains dangerous operations (DROP/DELETE)',
                )

            return EvaluationReason(
                value=True,
                reason='Valid SQL syntax',
            )
        except Exception as e:
            return EvaluationReason(
                value=False,
                reason=f'SQL parsing error: {e}',
            )
```

### Code Execution

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class ExecutablePython(Evaluator):
    timeout_seconds: float = 5.0

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        import asyncio
        import os
        import tempfile

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(ctx.output)
            temp_path = f.name

        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                'python', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return EvaluationReason(
                    value=False,
                    reason=f'Execution timeout after {self.timeout_seconds}s',
                )

            if process.returncode == 0:
                return EvaluationReason(
                    value=True,
                    reason='Code executed successfully',
                )
            else:
                return EvaluationReason(
                    value=False,
                    reason=f'Execution failed: {stderr.decode()}',
                )
        finally:
            os.unlink(temp_path)
```

### External API Validation

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class APIResponseValid(Evaluator):
    api_endpoint: str
    api_key: str

    async def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float]:
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_endpoint,
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    json={'data': ctx.output},
                    timeout=10.0,
                )

                result = response.json()

                return {
                    'api_reachable': True,
                    'validation_passed': result.get('valid', False),
                    'confidence_score': result.get('confidence', 0.0),
                }
        except Exception:
            return {
                'api_reachable': False,
                'validation_passed': False,
                'confidence_score': 0.0,
            }
```

## Testing Evaluators

Test evaluators like any other Python code:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    """Check if output exactly matches expected output."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output


def test_exact_match():
    evaluator = ExactMatch()

    # Test match
    ctx = EvaluatorContext(
        name='test',
        inputs='input',
        metadata=None,
        expected_output='expected',
        output='expected',
        duration=0.1,
        _span_tree=None,
        attributes={},
        metrics={},
    )
    assert evaluator.evaluate(ctx) is True

    # Test mismatch
    ctx.output = 'different'
    assert evaluator.evaluate(ctx) is False
```

## Best Practices

### 1. Keep Evaluators Focused

Each evaluator should check one thing:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def check_format(output: str) -> bool:
    return output.startswith('{')


def check_content(output: str) -> bool:
    return len(output) > 10


def check_length(output: str) -> bool:
    return len(output) < 1000


def check_spelling(output: str) -> bool:
    return True  # Placeholder


# Bad: Doing too much
@dataclass
class EverythingChecker(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict:
        return {
            'format_valid': check_format(ctx.output),
            'content_good': check_content(ctx.output),
            'length_ok': check_length(ctx.output),
            'spelling_correct': check_spelling(ctx.output),
        }


# Good: Separate evaluators
@dataclass
class FormatValidator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return check_format(ctx.output)


@dataclass
class ContentChecker(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return check_content(ctx.output)


@dataclass
class LengthChecker(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return check_length(ctx.output)


@dataclass
class SpellingChecker(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return check_spelling(ctx.output)
```

### 2. Handle Missing Data Gracefully

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class SafeEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        if ctx.expected_output is None:
            return EvaluationReason(
                value=True,
                reason='Skipped: no expected output provided',
            )

        # Your evaluation logic
        ...
```

### 3. Provide Helpful Reasons

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class HelpfulEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        # Bad
        return EvaluationReason(value=False, reason='Failed')

        # Good
        return EvaluationReason(
            value=False,
            reason=f'Expected {ctx.expected_output!r}, got {ctx.output!r}',
        )
```

### 4. Use Timeouts for External Calls

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class APIEvaluator(Evaluator):
    timeout: float = 10.0

    async def _call_api(self, output: str) -> bool:
        # Placeholder for API call
        return True

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        import asyncio

        try:
            return await asyncio.wait_for(
                self._call_api(ctx.output),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return False
```

## Next Steps

- **[Span-Based Evaluation](span-based.md)** - Using OpenTelemetry spans
- **[Examples](../examples/simple-validation.md)** - Practical examples
