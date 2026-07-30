# Standard Quality Metrics

This page shows how to express widely-used LLM evaluation methods with Pydantic Evals primitives:

- [`GEval`][pydantic_evals.evaluators.GEval] — a first-class evaluator implementing G-Eval
  chain-of-thought scoring (Liu et al., 2023).
- Ready-made [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] rubrics for the RAG metrics
  popularized by [Ragas](https://github.com/explodinggradients/ragas) (faithfulness, answer
  relevance, context precision, context recall) and for GEMBA translation quality
  (Kocmi & Federmann, 2023).

The RAG and GEMBA metrics are provided as *rubric recipes* rather than evaluator classes: each is
one rubric away from `LLMJudge`, and a rubric you own adapts freely to your dataset shape and
domain — rename a field, tighten a criterion, or translate the instructions without waiting on a
library release. Copy them into your project and edit as needed.

!!! note "Rubric approximations, not the upstream implementations"
    These rubrics approximate each metric with a single LLM-judge call; they do not reproduce the
    upstream algorithms (for example, Ragas's `answer_relevancy` generates questions from the
    answer and compares embeddings). If you need parity with published numbers, wrap the real
    library as shown in [Third-Party Integrations](framework-integrations.md).

## G-Eval

[`GEval`][pydantic_evals.evaluators.GEval] implements chain-of-thought evaluation: you provide the
aspect being evaluated (`criteria`) and a list of explicit `evaluation_steps`, and the judge
returns a reasoning trace plus an integer score in `score_range` (inclusive). Because the criteria
and steps are user-supplied, `GEval` puts no structural requirements on the inputs, and it works
in serialized datasets out of the box.

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

The result is an [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] whose value is
the raw integer score — on the scale you chose via `score_range`, not normalized to `0.0`-`1.0`
like [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] scores. If the judge returns a score outside
`score_range`, the evaluation fails rather than recording a misleading value.

!!! note "Simplified G-Eval"
    The published G-Eval method computes a probability-weighted expectation over score tokens
    using the judge model's log-probs. Pydantic Evals asks the model for a direct integer score
    instead, trading a small amount of correlation with human judgment for provider-agnostic
    simplicity. See Liu et al., 2023, "G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment".

## RAG metric rubrics

These recipes assume each case's `inputs` carries the user question and the context passages the
output is supposed to rely on — a *supplied* context, not whatever an agent retrieved at runtime.
With `include_input=True`, [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] shows the judge your
full inputs object, so any shape works as long as the rubric describes it; adjust the wording if
your fields are named differently.

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

faithfulness = LLMJudge(
    rubric=(
        'Every factual claim in the Output must be directly supported by the context passages '
        'in the Input. Unsupported claims, contradictions, and fabrications constitute failure; '
        'ignore claims that are true in the real world but absent from the provided context. '
        'The score is the fraction of claims that are supported (0.0 = none, 1.0 = all); '
        'pass only if every claim is supported.'
    ),
    include_input=True,
    score={'evaluation_name': 'faithfulness'},
    assertion=False,
)

answer_relevance = LLMJudge(
    rubric=(
        'Judge whether the Output directly and completely answers the question in the Input, '
        'without padding or unrelated tangents. '
        'The score reflects how directly the Output addresses the question '
        '(0.0 = unrelated, 1.0 = a direct, on-point answer).'
    ),
    include_input=True,
    score={'evaluation_name': 'answer_relevance'},
    assertion=False,
)

context_precision = LLMJudge(
    rubric=(
        'This metric judges the retrieval, not the answer: assess the context passages in the '
        'Input against the question in the Input, and disregard the Output. '
        'The score is the fraction of the context that is relevant to answering the question '
        '(0.0 = none is relevant, 1.0 = all of it is relevant).'
    ),
    include_input=True,
    score={'evaluation_name': 'context_precision'},
    assertion=False,
)

context_recall = LLMJudge(
    rubric=(
        'This metric judges the retrieval, not the answer: determine whether the context '
        'passages in the Input contain enough information to produce the ground-truth answer '
        'in the Expected Output, and disregard the Output. '
        'The score is the fraction of the ground-truth answer that is supported by the context '
        '(0.0 = none of it, 1.0 = all of it).'
    ),
    include_input=True,
    include_expected_output=True,
    score={'evaluation_name': 'context_recall'},
    assertion=False,
)


@dataclass
class RagInputs:
    question: str
    context: list[str]


dataset = Dataset(
    name='rag_quality',
    cases=[
        Case(
            inputs=RagInputs(
                question='Where is the Eiffel Tower?',
                context=['The Eiffel Tower is in Paris, France.'],
            ),
            expected_output='The Eiffel Tower is in Paris.',
        ),
    ],
    evaluators=[faithfulness, answer_relevance, context_precision, context_recall],
)
```

Each recipe emits a `0.0`-`1.0` score named via the `score`
[`OutputConfig`][pydantic_evals.evaluators.OutputConfig]; swap `assertion=False` for an
`assertion` config (or keep both) if you also want a pass/fail column, as described in
[LLM Judge](llm-judge.md).

## GEMBA translation quality

The GEMBA Direct Assessment prompt (Kocmi & Federmann, 2023, "Large Language Models Are
State-of-the-Art Evaluators of Translation Quality") scores a translation from 0 to 100. Here the
case's `inputs` is the source text, the output is the candidate translation, and (optionally) the
`expected_output` is a human reference translation:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

gemba_da = LLMJudge(
    rubric=(
        'The Input is the English source text and the Output is its French translation '
        '(the Expected Output, if present, is a human reference translation). '
        'Score the translation on a continuous scale from 0 to 100, where 0 means '
        '"no meaning preserved" and 100 means "perfect meaning and grammar", '
        'then report it normalized to the 0.0-1.0 range by dividing by 100.'
    ),
    include_input=True,
    include_expected_output=True,
    score={'evaluation_name': 'gemba_da'},
    assertion=False,
)

dataset = Dataset(
    name='translation_quality',
    cases=[
        Case(
            inputs='Hello, world!',
            expected_output='Bonjour, le monde !',
        ),
    ],
    evaluators=[gemba_da],
)
```

Adjust the language names to your language pair. For the GEMBA-SQM variant, replace the scale
sentence with the anchored 0-6 scale from the paper (0 = no meaning preserved, 2 = some meaning
preserved, 4 = most meaning preserved with few grammar mistakes, 6 = perfect meaning and grammar).

## Picking the right tool

| Need | Use |
| --- | --- |
| Score a quality dimension on an integer scale with explicit CoT steps | [`GEval`][pydantic_evals.evaluators.GEval] |
| Grounding, relevance, retrieval quality, translation quality | The [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] recipes above |
| Something bespoke | [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] with your own rubric |
| Exact parity with an upstream framework | [Third-Party Integrations](framework-integrations.md) |
