# Standard Quality Metrics

Pydantic Evals ships a small set of LLM-backed evaluators whose names and rubrics track widely-used
evaluation methods:

- [Ragas](https://github.com/explodinggradients/ragas)-style RAG metrics, under the
  [`ragas`][pydantic_evals.evaluators.ragas] namespace.
- [`GEval`][pydantic_evals.evaluators.GEval] — G-Eval chain-of-thought scoring (Liu et al., 2023).
- [`GembaScore`][pydantic_evals.evaluators.GembaScore] — GEMBA translation quality (Kocmi & Federmann, 2023).

Each one is a thin wrapper over [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] and the
[`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers — a *preset* with a recognisable name
and a sensible default rubric, not a new evaluation mechanism. They take the same `model`,
`model_settings`, `score` and `assertion` arguments as [`LLMJudge`][pydantic_evals.evaluators.LLMJudge],
so they drop into existing evaluation suites without ceremony. You can always write your own rubric
against [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] directly.

!!! note "Presets, not the upstream implementation"
    The Ragas-style evaluators approximate each metric with a single LLM-judge rubric; they do not
    reproduce Ragas's exact algorithm (for example, Ragas's `answer_relevancy` generates questions
    from the answer and compares embeddings). They are namespaced under `ragas` precisely so the
    bare metric names stay free for first-party, general-purpose alternatives we may add later. If
    you need parity with published Ragas numbers, wrap the real library as shown in
    [Third-Party Integrations](framework-integrations.md).

## Ragas-style RAG metrics

These live under the [`ragas`][pydantic_evals.evaluators.ragas] namespace and serialize as
`ragas.Faithfulness`, `ragas.AnswerRelevance`, and so on.

### Dataset shape

The Ragas-style evaluators enforce the shape of each case's `inputs` at the type level via two
[`Protocol`][typing.Protocol]s:

- [`HasQuestion`][pydantic_evals.evaluators.ragas.HasQuestion] — `question: str`. Required by
  [`AnswerRelevance`][pydantic_evals.evaluators.ragas.AnswerRelevance].
- [`QuestionWithContext`][pydantic_evals.evaluators.ragas.QuestionWithContext] — `question: str`
  plus `context: Sequence[str]`. Required by the retrieval-oriented evaluators.

Any dataclass, `TypedDict`, Pydantic model, or plain class that structurally exposes those
attributes will satisfy the corresponding protocol — no inheritance required.

!!! info "Context comes from your inputs"
    These evaluators grade against the `context` you attach to each case's `inputs` — i.e. a
    *supplied* context. They do not inspect what an agent retrieved at runtime (which would live in
    the span tree). Use them to evaluate generation/retrieval quality against a known context; if
    you need to grade against runtime retrievals, capture them into your case inputs first.

### Faithfulness

[`Faithfulness`][pydantic_evals.evaluators.ragas.Faithfulness] checks that every factual claim in
the output is grounded in the provided context. Unsupported, contradicted, or fabricated claims
cause the assertion to fail; the score is the fraction of claims that are supported.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ragas


@dataclass
class RagInputs:
    question: str
    context: list[str]


dataset = Dataset[RagInputs, str, object](
    name='faithfulness_demo',
    cases=[
        Case(
            inputs=RagInputs(
                question='Where is the Eiffel Tower?',
                context=['The Eiffel Tower is in Paris, France.'],
            ),
        ),
    ],
    evaluators=[ragas.Faithfulness()],
)
```

### Answer Relevance

[`AnswerRelevance`][pydantic_evals.evaluators.ragas.AnswerRelevance] judges whether the output
directly addresses the question in the input, without padding or tangents. It only needs a
`question` field on the inputs.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ragas


@dataclass
class QInputs:
    question: str


dataset = Dataset[QInputs, str, object](
    name='answer_relevance_demo',
    cases=[Case(inputs=QInputs(question='What is the speed of light?'))],
    evaluators=[ragas.AnswerRelevance()],
)
```

### Context Precision

[`ContextPrecision`][pydantic_evals.evaluators.ragas.ContextPrecision] estimates how much of a
retrieved context is actually relevant to the question — a low score flags retrievers that drown
the downstream model in noise.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ragas


@dataclass
class RagInputs:
    question: str
    context: list[str]


dataset = Dataset[RagInputs, str, object](
    name='context_precision_demo',
    cases=[
        Case(
            inputs=RagInputs(
                question='What is the capital of France?',
                context=[
                    'Paris is the capital of France.',
                    'The Seine flows through Paris.',
                    'Lyon is a city in France.',
                    'Bordeaux is a city in France.',
                ],
            ),
        ),
    ],
    evaluators=[ragas.ContextPrecision()],
)
```

### Context Recall

[`ContextRecall`][pydantic_evals.evaluators.ragas.ContextRecall] checks whether the retrieved
context contains enough information to produce the ground-truth answer. It reads the ground-truth
answer from each case's `expected_output`; if no `expected_output` is set the evaluator returns an
empty result rather than fabricating one.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ragas


@dataclass
class RagInputs:
    question: str
    context: list[str]


dataset = Dataset[RagInputs, str, object](
    name='context_recall_demo',
    cases=[
        Case(
            inputs=RagInputs(
                question='Who wrote Pride and Prejudice?',
                context=['Pride and Prejudice is a classic English novel.'],
            ),
            expected_output='Jane Austen wrote Pride and Prejudice in 1813.',
        ),
    ],
    evaluators=[ragas.ContextRecall()],
)
```

### If these evaluators don't fit your use case

These are **presets**, not a rigid API: a thin, opinionated layer over
[`LLMJudge`][pydantic_evals.evaluators.LLMJudge] with a fixed schema and rubric per metric. If your
dataset is shaped differently (e.g. the question field is called `prompt`, the context is a
structured object rather than a list of passages), or you want to tweak the rubric for your domain,
the intended escape hatch is to **vendor the evaluator's source** into your own project and modify
it to fit. Each class is short and deliberately readable so this remains a small patch.

For teams that would rather wrap an off-the-shelf library like Ragas or DeepEval directly, see the
[Third-Party Integrations](framework-integrations.md) page.

## G-Eval (Liu et al., 2023)

[`GEval`][pydantic_evals.evaluators.GEval] implements a chain-of-thought evaluator: you provide
the aspect being evaluated (`criteria`) and a list of explicit `evaluation_steps`, and the judge
returns a reasoning trace plus an integer score in `score_range`. Because the criteria and steps
are user-supplied, `GEval` puts no structural requirements on the inputs.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GEval

dataset = Dataset(
    name='g_eval_demo',
    cases=[Case(inputs='Explain how black holes form.')],
    evaluators=[
        GEval(
            criteria='coherence',
            evaluation_steps=[
                'Read the output carefully.',
                'Check that each sentence follows logically from the previous one.',
                'Assign a score from 1 (incoherent) to 5 (fully coherent).',
            ],
            include_input=True,
        ),
    ],
)
```

!!! note "Simplified G-Eval"
    The published G-Eval method computes a probability-weighted expectation over score tokens
    using the judge model's log-probs. Pydantic Evals asks the model for a direct integer score
    instead, trading a small amount of correlation with human judgment for provider-agnostic
    simplicity. See Liu et al., 2023, "G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment".

## GEMBA (Kocmi & Federmann, 2023)

[`GembaScore`][pydantic_evals.evaluators.GembaScore] is a machine-translation quality evaluator
using the GEMBA prompts. Two variants are supported, matching the published prompts:

- `score_type='DA'` — Direct Assessment, integer score **0-100**.
- `score_type='SQM'` — Scalar Quality Metrics, integer score **0-6**.

The source text is `ctx.inputs` (a `str`), the candidate translation is `ctx.output` (a `str`),
and the optional human reference is `ctx.expected_output`.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GembaScore

dataset = Dataset[str, str, object](
    name='gemba_demo',
    cases=[
        Case(
            inputs='Hello, world!',
            expected_output='Bonjour, le monde !',
        ),
    ],
    evaluators=[GembaScore(source_lang='English', target_lang='French')],
)
```

See Kocmi & Federmann, 2023, "Large Language Models Are State-of-the-Art Evaluators of Translation
Quality".

## Picking the right tool

| Need | Use |
| --- | --- |
| Is the output supported by my retrieved context? | [`ragas.Faithfulness`][pydantic_evals.evaluators.ragas.Faithfulness] |
| Does the answer actually address the question? | [`ragas.AnswerRelevance`][pydantic_evals.evaluators.ragas.AnswerRelevance] |
| Is my retriever surfacing relevant passages? | [`ragas.ContextPrecision`][pydantic_evals.evaluators.ragas.ContextPrecision] |
| Is my retriever returning enough information? | [`ragas.ContextRecall`][pydantic_evals.evaluators.ragas.ContextRecall] |
| Score a quality dimension on an integer scale with explicit CoT steps | [`GEval`][pydantic_evals.evaluators.GEval] |
| Score a translation against a source (and optional reference) | [`GembaScore`][pydantic_evals.evaluators.GembaScore] |
| Something bespoke | [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] with a custom rubric |

For bringing in the exact upstream implementations of these and other frameworks, see
[Third-Party Integrations](framework-integrations.md).
