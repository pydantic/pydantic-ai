# Standard Quality Metrics

Pydantic Evals ships a small curated pack of LLM-backed evaluators whose names and rubrics
track widely-used evaluation frameworks — [Ragas](https://github.com/explodinggradients/ragas),
G-Eval (Liu et al., 2023) and GEMBA (Kocmi & Federmann, 2023).

These evaluators are thin wrappers over [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] and the
[`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers. You can always write your own rubric
against `LLMJudge` — the point of this pack is to provide recognisable names and sensible defaults
for the quality dimensions users most often reach for. Each evaluator takes the same `model`,
`model_settings`, `score` and `assertion` arguments as `LLMJudge`, so they drop into existing
evaluation suites without ceremony.

## Dataset shape

The Ragas-style evaluators (`Faithfulness`, `AnswerRelevance`, `ContextPrecision`,
`ContextRecall`, `Hallucination`) enforce the shape of each case's `inputs` at the type level via
two [`Protocol`][typing.Protocol]s:

- [`HasQuestion`][pydantic_evals.evaluators.HasQuestion] — `question: str`. Required by
  [`AnswerRelevance`][pydantic_evals.evaluators.AnswerRelevance].
- [`QuestionWithContext`][pydantic_evals.evaluators.QuestionWithContext] — `question: str` plus
  `context: Sequence[str]`. Required by the retrieval-oriented evaluators.

Any dataclass, `TypedDict`, Pydantic model, or plain class that structurally exposes those
attributes will satisfy the corresponding protocol — no inheritance required. Here is the shape
most of the examples on this page use:

```python
from dataclasses import dataclass


@dataclass
class RagInputs:
    question: str
    context: list[str]
```

## If these evaluators don't fit your use case

These are **curated presets**, not a rigid API: a thin, opinionated layer over `LLMJudge` with a
fixed schema and rubric per metric. If your dataset is shaped differently (e.g. the question
field is called `prompt`, the context is a structured object rather than a list of passages), or
you want to tweak the rubric for your domain, the intended escape hatch is to **vendor the
evaluator's source** into your own project and modify it to fit. Each class is short and
deliberately readable so this remains a small patch.

For teams that would rather wrap an off-the-shelf library like Ragas or DeepEval directly, see
the [Third-Party Integrations](framework-integrations.md) page.

!!! tip "Mix and match"
    Nothing stops you from combining these curated evaluators with a custom
    [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] or your own domain-specific evaluator in the
    same dataset. Many users end up with one or two curated metrics plus case-specific rubrics.

## Faithfulness (Ragas)

Faithfulness checks that every factual claim in the output is grounded in a provided retrieval
context. Unsupported, contradicted, or fabricated claims cause the assertion to fail; the score
is the fraction of claims that are supported.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Faithfulness


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
    evaluators=[Faithfulness()],
)
```

## Answer Relevance (Ragas)

[`AnswerRelevance`][pydantic_evals.evaluators.AnswerRelevance] judges whether the output directly
addresses the question in the input, without padding or tangents. It only needs a `question`
field on the inputs.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import AnswerRelevance


@dataclass
class QInputs:
    question: str


dataset = Dataset[QInputs, str, object](
    name='answer_relevance_demo',
    cases=[Case(inputs=QInputs(question='What is the speed of light?'))],
    evaluators=[AnswerRelevance()],
)
```

## Context Precision (Ragas)

[`ContextPrecision`][pydantic_evals.evaluators.ContextPrecision] estimates how much of a retrieved
context is actually relevant to the question — a low score flags retrievers that drown the
downstream model in noise.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ContextPrecision


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
    evaluators=[ContextPrecision()],
)
```

## Context Recall (Ragas)

[`ContextRecall`][pydantic_evals.evaluators.ContextRecall] checks whether the retrieved context
contains enough information to produce the ground-truth answer. It reads the ground-truth answer
from each case's `expected_output`; if no `expected_output` is set the evaluator returns an empty
result rather than fabricating one.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ContextRecall


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
    evaluators=[ContextRecall()],
)
```

## Hallucination

[`Hallucination`][pydantic_evals.evaluators.Hallucination] is the sibling of `Faithfulness`,
framed as a detector:

- `score` is a groundedness score — **higher is better** (1.0 = fully grounded, 0.0 = fully
  hallucinated).
- The assertion fires (`pass=True`) when the detector **does** find hallucinated claims. This is
  the opposite of most assertions, and deliberate: CI can gate on `pass=False` to mean "no
  hallucination detected".

If you prefer the conventional direction where `pass=True` means "the output meets the criterion",
use [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] instead.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Hallucination


@dataclass
class RagInputs:
    question: str
    context: list[str]


dataset = Dataset[RagInputs, str, object](
    name='hallucination_demo',
    cases=[
        Case(
            inputs=RagInputs(
                question='Summarise the article.',
                context=['The article discusses Mars exploration...'],
            ),
        ),
    ],
    evaluators=[Hallucination()],
)
```

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
| Is the output supported by my retrieved context? | [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] |
| Did the model hallucinate anything? (gate on `pass=False`) | [`Hallucination`][pydantic_evals.evaluators.Hallucination] |
| Does the answer actually address the question? | [`AnswerRelevance`][pydantic_evals.evaluators.AnswerRelevance] |
| Is my retriever surfacing relevant passages? | [`ContextPrecision`][pydantic_evals.evaluators.ContextPrecision] |
| Is my retriever returning enough information? | [`ContextRecall`][pydantic_evals.evaluators.ContextRecall] |
| Score a quality dimension on an integer scale with explicit CoT steps | [`GEval`][pydantic_evals.evaluators.GEval] |
| Score a translation against a source (and optional reference) | [`GembaScore`][pydantic_evals.evaluators.GembaScore] |
| Something bespoke | [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] with a custom rubric |

For bringing in the exact upstream implementations of these and other frameworks, see
[Third-Party Integrations](framework-integrations.md).
