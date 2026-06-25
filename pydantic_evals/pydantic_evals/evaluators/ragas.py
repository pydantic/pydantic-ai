"""Rubric presets modeled on [Ragas](https://github.com/explodinggradients/ragas) RAG metrics.

Each evaluator here is a thin wrapper over [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] and the
[`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers, with a rubric aligned to the
correspondingly-named Ragas metric (`faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`). They are *presets*, not the upstream implementation: they carry zero extra
dependencies and approximate each metric with a single LLM-judge rubric rather than reproducing
Ragas's exact algorithm (for example, Ragas's `answer_relevancy` generates questions from the answer
and compares embeddings). If you need parity with published Ragas numbers, wrap the real library as
shown in [Third-Party Integrations](../../docs/evals/evaluators/framework-integrations.md).

The evaluators live under the `ragas` namespace (and serialize as `ragas.Faithfulness`, etc.) so that
the bare metric names stay free for first-party, general-purpose alternatives we may add later.

Each one enforces its dataset shape at the type level via simple [`Protocol`][typing.Protocol]s
(`HasQuestion`, `QuestionWithContext`) instead of accepting string-or-callable extraction hooks. The
context they grade against comes from `ctx.inputs`, i.e. the context you attach to each case — these
evaluate generation/retrieval quality against a *supplied* context, not against whatever an agent
retrieved at runtime. If your dataset shape doesn't match, the implementations here are short and
deliberately readable, so vendoring and adapting one is straightforward.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal, Protocol

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings

from .common import OutputConfig, serialize_model_as_string, update_combined_output
from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationScalar, Evaluator, EvaluatorOutput
from .llm_as_a_judge import (
    GradingOutput,
    judge_input_output,
    judge_input_output_expected,
)

__all__ = (
    'AnswerRelevance',
    'ContextPrecision',
    'ContextRecall',
    'Faithfulness',
    'HasQuestion',
    'QuestionWithContext',
)


class HasQuestion(Protocol):
    """Structural type for dataset inputs that expose the user question as `question: str`.

    Used by [`AnswerRelevance`][pydantic_evals.evaluators.ragas.AnswerRelevance] to locate the
    question field on each case's inputs at the type level.
    """

    @property
    def question(self) -> str: ...


class QuestionWithContext(Protocol):
    """Structural type for dataset inputs that expose a question and retrieved context.

    Used by the retrieval evaluators
    ([`Faithfulness`][pydantic_evals.evaluators.ragas.Faithfulness],
    [`ContextPrecision`][pydantic_evals.evaluators.ragas.ContextPrecision],
    [`ContextRecall`][pydantic_evals.evaluators.ragas.ContextRecall]).
    """

    @property
    def question(self) -> str: ...

    @property
    def context(self) -> Sequence[str]: ...


class _RagasNamespaced:
    """Mixin that serializes Ragas-preset evaluators under the `ragas.` namespace.

    Keeps the bare metric names (`Faithfulness`, ...) free for future first-party evaluators while
    making the Ragas provenance explicit in serialized dataset specs and reports.
    """

    @classmethod
    def get_serialization_name(cls) -> str:
        return f'ragas.{cls.__name__}'


def _default_score_config() -> OutputConfig:
    return OutputConfig(include_reason=True)


def _default_assertion_config() -> OutputConfig:
    return OutputConfig(include_reason=True)


def _emit(
    evaluation_name: str,
    grading_output: GradingOutput,
    score_config: OutputConfig | Literal[False],
    assertion_config: OutputConfig | Literal[False],
) -> EvaluatorOutput:
    """Build an [`EvaluatorOutput`][pydantic_evals.evaluators.EvaluatorOutput] from a [`GradingOutput`][pydantic_evals.evaluators.llm_as_a_judge.GradingOutput].

    Mirrors the score/assertion naming logic used by [`LLMJudge`][pydantic_evals.evaluators.LLMJudge]
    so these presets produce the same shape of reports.
    """
    output: dict[str, EvaluationScalar | EvaluationReason] = {}
    include_both = score_config is not False and assertion_config is not False

    if score_config is not False:
        default_name = f'{evaluation_name}_score' if include_both else evaluation_name
        update_combined_output(output, grading_output.score, grading_output.reason, score_config, default_name)

    if assertion_config is not False:
        default_name = f'{evaluation_name}_pass' if include_both else evaluation_name
        update_combined_output(output, grading_output.pass_, grading_output.reason, assertion_config, default_name)

    return output


_FAITHFULNESS_RUBRIC = dedent(
    """\
    The Input is a JSON object with two fields: `question` (the user question) and `context`
    (the retrieved context passages that the Output is supposed to rely on).

    Every factual claim in the Output must be directly supported by `context`. Pass only if all
    claims are entailed by the context. Unsupported claims, contradictions, or fabrications
    constitute failure. Ignore information that is correct in the real world but not present in
    the provided context.

    The score should be the fraction of factual claims that are supported by the context
    (0.0 = none are supported, 1.0 = all are supported).
    """
)


@dataclass(repr=False)
class Faithfulness(_RagasNamespaced, Evaluator[QuestionWithContext, object, object]):
    """Ragas-style `faithfulness`: every factual claim in the output is supported by the provided context.

    Pass when all claims are grounded in `context`; score reflects the fraction of claims that are
    supported (higher is better). This is a single-rubric LLM-judge preset, not Ragas's exact
    statement-decomposition algorithm.

    The evaluator reads the `question` and `context` fields directly off `ctx.inputs`, which must
    structurally satisfy [`QuestionWithContext`][pydantic_evals.evaluators.ragas.QuestionWithContext].
    """

    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[QuestionWithContext, object, object]) -> EvaluatorOutput:
        judge_inputs = {'question': ctx.inputs.question, 'context': list(ctx.inputs.context)}
        grading_output = await judge_input_output(
            judge_inputs, ctx.output, _FAITHFULNESS_RUBRIC, self.model, self.model_settings
        )
        return _emit(self.get_default_evaluation_name(), grading_output, self.score, self.assertion)

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())


_ANSWER_RELEVANCE_RUBRIC = dedent(
    """\
    The Input is a JSON object with a single field `question` (the user question).

    The Output must directly answer the question.

    A strong answer:
    - Addresses the question without redundancy or padding,
    - Stays on topic (no unrelated tangents),
    - Gives the level of detail the question calls for (terse when asked terse, thorough when asked thorough).

    The score should reflect how directly and completely the Output answers the question
    (0.0 = unrelated, 1.0 = a direct, on-point answer).
    """
)


@dataclass(repr=False)
class AnswerRelevance(_RagasNamespaced, Evaluator[HasQuestion, object, object]):
    """Ragas-style `answer_relevancy`: the output directly addresses the input question.

    Ragas computes this by generating questions from the answer and comparing their embeddings to
    the original question; this preset instead scores relevance with a single LLM-judge rubric.

    The evaluator reads `question` directly off `ctx.inputs`, which must structurally satisfy
    [`HasQuestion`][pydantic_evals.evaluators.ragas.HasQuestion].
    """

    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[HasQuestion, object, object]) -> EvaluatorOutput:
        judge_inputs = {'question': ctx.inputs.question}
        grading_output = await judge_input_output(
            judge_inputs, ctx.output, _ANSWER_RELEVANCE_RUBRIC, self.model, self.model_settings
        )
        return _emit(self.get_default_evaluation_name(), grading_output, self.score, self.assertion)

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())


_CONTEXT_PRECISION_RUBRIC = dedent(
    """\
    The Input is a JSON object with fields `question` (the user question) and `context`
    (the retrieved context passages being evaluated). The Output echoes the same `context` —
    it is the thing being judged, not a generated answer.

    Identify which passages in `context` are actually relevant to answering the `question`,
    and which are irrelevant filler. A high-precision retrieval surfaces only relevant passages.

    The score should be the fraction of the context that is relevant to the question
    (0.0 = none is relevant, 1.0 = all of it is relevant). Pass when precision is high enough
    that irrelevant content would not materially distract a downstream model.
    """
)


@dataclass(repr=False)
class ContextPrecision(_RagasNamespaced, Evaluator[QuestionWithContext, object, object]):
    """Ragas-style `context_precision`: how much of the retrieved context is relevant to the question.

    The evaluator treats the retrieved context as the thing being judged. Both fields are pulled
    from `ctx.inputs`, which must structurally satisfy
    [`QuestionWithContext`][pydantic_evals.evaluators.ragas.QuestionWithContext].
    """

    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[QuestionWithContext, object, object]) -> EvaluatorOutput:
        context_passages = list(ctx.inputs.context)
        judge_inputs = {'question': ctx.inputs.question, 'context': context_passages}
        grading_output = await judge_input_output(
            judge_inputs, context_passages, _CONTEXT_PRECISION_RUBRIC, self.model, self.model_settings
        )
        return _emit(self.get_default_evaluation_name(), grading_output, self.score, self.assertion)

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())


_CONTEXT_RECALL_RUBRIC = dedent(
    """\
    The Input is a JSON object with fields `question` (the user question) and `context`
    (the retrieved context passages). The ExpectedOutput contains the ground-truth answer.

    This metric judges the retrieved `context` only. Ignore the Output entirely — it is not part
    of the assessment. Determine whether `context` contains enough information to produce the
    ground-truth answer in ExpectedOutput. A high-recall retrieval leaves no gaps: every claim in
    the ground-truth answer should be supported by `context`.

    The score should be the fraction of the ground-truth answer that is supported by `context`
    (0.0 = none of it is covered, 1.0 = all of it is covered). Pass when `context` is sufficient
    to fully produce the ground-truth answer.
    """
)


@dataclass(repr=False)
class ContextRecall(_RagasNamespaced, Evaluator[QuestionWithContext, object, object]):
    """Ragas-style `context_recall`: whether the retrieved context is sufficient to produce the ground-truth answer.

    The `question` and `context` fields come from `ctx.inputs` (which must structurally satisfy
    [`QuestionWithContext`][pydantic_evals.evaluators.ragas.QuestionWithContext]); the ground-truth
    answer comes from `ctx.expected_output`. If no `expected_output` is set on the case, the
    evaluator returns an empty result rather than fabricating one.
    """

    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[QuestionWithContext, object, object]) -> EvaluatorOutput:
        if ctx.expected_output is None:
            return {}
        judge_inputs = {'question': ctx.inputs.question, 'context': list(ctx.inputs.context)}
        grading_output = await judge_input_output_expected(
            judge_inputs,
            ctx.output,
            ctx.expected_output,
            _CONTEXT_RECALL_RUBRIC,
            self.model,
            self.model_settings,
        )
        return _emit(self.get_default_evaluation_name(), grading_output, self.score, self.assertion)

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())
