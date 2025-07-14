# Pydantic Evals

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/pydantic-evals.svg)](https://pypi.python.org/pypi/pydantic-evals)
[![python versions](https://img.shields.io/pypi/pyversions/pydantic-evals.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

This is a library for evaluating non-deterministic (or "stochastic") functions in Python. It provides a simple,
Pythonic interface for defining and running stochastic functions, and analyzing the results of running those functions.

While this library is developed as part of [Pydantic AI](https://ai.pydantic.dev), it only uses Pydantic AI for a small
subset of generative functionality internally, and it is designed to be used with arbitrary "stochastic function"
implementations. In particular, it can be used with other (non-Pydantic AI) AI libraries, agent frameworks, etc.

As with Pydantic AI, this library prioritizes type safety and use of common Python syntax over esoteric, domain-specific
use of Python syntax.

Full documentation is available at [ai.pydantic.dev/evals](https://ai.pydantic.dev/evals).

## Example

While you'd typically use Pydantic Evals with more complex functions (such as Pydantic AI agents or graphs), here's a
quick example that evaluates a simple function against a test case using both custom and built-in evaluators:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

# Define a test case with inputs and expected output
case = Case(
    name='capital_question',
    inputs='What is the capital of France?',
    expected_output='Paris',
)

# Define a custom evaluator
class MatchAnswer(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif isinstance(ctx.output, str) and ctx.expected_output.lower() in ctx.output.lower():
            return 0.8
        return 0.0

# Create a dataset with the test case and evaluators
dataset = Dataset(
    cases=[case],
    evaluators=[IsInstance(type_name='str'), MatchAnswer()],
)

# Define the function to evaluate
async def answer_question(question: str) -> str:
    return 'Paris'

# Run the evaluation
report = dataset.evaluate_sync(answer_question)
report.print(include_input=True, include_output=True)
"""
                                    Evaluation Summary: answer_question
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID          ┃ Inputs                         ┃ Outputs ┃ Scores            ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ capital_question │ What is the capital of France? │ Paris   │ MatchAnswer: 1.00 │ ✔          │     10ms │
├──────────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┼──────────┤
│ Averages         │                                │         │ MatchAnswer: 1.00 │ 100.0% ✔   │     10ms │
└──────────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┴──────────┘
"""
```

Using the library with more complex functions, such as Pydantic AI agents, is similar — all you need to do is define a
task function wrapping the function you want to evaluate, with a signature that matches the inputs and outputs of your
test cases.

## Logfire Integration

Pydantic Evals uses OpenTelemetry to record traces for each case in your evaluations.

You can send these traces to any OpenTelemetry-compatible backend. For the best experience, we recommend [Pydantic Logfire](https://logfire.pydantic.dev/docs), which includes custom views for evals:

<div style="display: flex; gap: 1rem; flex-wrap: wrap;">
  <img src="https://ai.pydantic.dev/img/logfire-evals-overview.png" alt="Logfire Evals Overview" width="48%">
  <img src="https://ai.pydantic.dev/img/logfire-evals-case.png" alt="Logfire Evals Case View" width="48%">
</div>

You'll see full details about the inputs, outputs, token usage, execution durations, etc. And you'll have access to the full trace for each case — ideal for debugging, writing path-aware evaluators, or running the similar evaluations against production traces.

Basic setup:

```python {test="skip" lint="skip" format="skip"}
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)

...

my_dataset.evaluate_sync(my_task)
```

[Read more about the Logfire integration here.](https://ai.pydantic.dev/evals/#logfire-integration)
