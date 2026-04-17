# Third-Party Integrations

Pydantic Evals does not take a hard dependency on any particular metrics framework. When a team
already uses [Ragas](https://github.com/explodinggradients/ragas),
[DeepEval](https://github.com/confident-ai/deepeval), or another scoring library, the
[`Evaluator`][pydantic_evals.evaluators.Evaluator] base class makes it straightforward to wrap the
upstream metric and run it inside any Pydantic Evals dataset. This page shows worked examples for
the common ones.

!!! tip "Prefer a native evaluator where you can"
    If a rubric-based [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] or a
    [custom evaluator](custom.md) covers your use case, that's usually simpler — zero extra
    dependencies and the scores slot into reports cleanly. Reach for the integrations below
    when you specifically want the *exact* upstream implementation (for reproducibility with
    published benchmarks, parity with an existing evaluation suite, or features we don't
    expose natively). You can mix external and native evaluators in one dataset.

## Pattern

Each framework integration follows the same pattern:

1. Subclass [`Evaluator`][pydantic_evals.evaluators.Evaluator].
2. Adapt `ctx.inputs`, `ctx.output`, `ctx.expected_output`, and metadata into whatever the
   upstream metric expects.
3. Return a `float` score, a `bool` assertion, an [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason],
   or a `dict` of these.

The rest of this page shows concrete adapters. They are intentionally compact — extend them with
whatever configuration your team needs (model selection, thresholds, per-case toggles).

## Ragas

Install with `pip install ragas` (not included in `pydantic-evals`).

This adapter wraps [`ragas.metrics.Faithfulness`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
for a single-turn sample. Each case is expected to provide the retrieved context as part of its
inputs or metadata.

```python {test="skip" lint="skip"}
from dataclasses import dataclass

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class RagasFaithfulness(Evaluator):
    """Wrap `ragas.metrics.Faithfulness` as a Pydantic Evals evaluator."""

    context_field: str = 'context'

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        metadata = ctx.metadata or {}
        retrieved_contexts = metadata.get(self.context_field, [])
        if isinstance(retrieved_contexts, str):
            retrieved_contexts = [retrieved_contexts]

        sample = SingleTurnSample(
            user_input=str(ctx.inputs),
            response=str(ctx.output),
            retrieved_contexts=retrieved_contexts,
        )
        metric = Faithfulness()
        score = await metric.single_turn_ascore(sample)
        return EvaluationReason(value=float(score), reason=f'ragas.Faithfulness = {score:.3f}')
```

Usage is the same as any built-in evaluator:

```python {test="skip" lint="skip"}
from pydantic_evals import Case, Dataset

dataset = Dataset(
    name='rag_eval',
    cases=[
        Case(
            inputs='What is the capital of France?',
            metadata={'context': ['Paris is the capital of France.']},
        ),
    ],
    evaluators=[RagasFaithfulness()],
)
```

The same pattern works for `ragas.metrics.answer_relevancy`, `context_precision`, and the other
scoring metrics: swap the metric class and (if needed) the sample fields.

## DeepEval

Install with `pip install deepeval` (not included in `pydantic-evals`).

This adapter wraps [DeepEval's `GEval` metric](https://docs.confident-ai.com/docs/metrics-llm-evals)
to score a criterion against a `LLMTestCase`. DeepEval's `measure` is synchronous, so the
evaluator is synchronous too.

```python {test="skip" lint="skip"}
from dataclasses import dataclass

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class DeepEvalGEval(Evaluator):
    """Wrap `deepeval.metrics.GEval` as a Pydantic Evals evaluator."""

    name: str
    criteria: str
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool | EvaluationReason]:
        test_case = LLMTestCase(
            input=str(ctx.inputs),
            actual_output=str(ctx.output),
            expected_output=None if ctx.expected_output is None else str(ctx.expected_output),
        )
        metric = GEval(
            name=self.name,
            criteria=self.criteria,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=self.threshold,
        )
        metric.measure(test_case)
        return {
            f'{self.name}_score': EvaluationReason(value=float(metric.score), reason=metric.reason or ''),
            f'{self.name}_pass': bool(metric.success),
        }
```

The same wrapper shape works for DeepEval's `FaithfulnessMetric`, `AnswerRelevancyMetric`,
`HallucinationMetric`, and others — swap the metric class and populate the relevant
`LLMTestCase` fields (for example `retrieval_context` for faithfulness).

## Notes on dependencies

- `ragas` and `deepeval` are optional dependencies — they are not installed with
  `pydantic-evals` and are not part of any dependency group. Install them only in projects that
  use these integrations.
- Both libraries make their own LLM calls, so be prepared for extra API usage when running a
  dataset that includes these evaluators.
