"""LLM-backed quality evaluators named after the methods they implement.

These are thin wrappers over the [`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers with
rubrics aligned to published evaluation methods:

- [`GEval`][pydantic_evals.evaluators.GEval] — G-Eval chain-of-thought scoring (Liu et al., 2023).
- [`GembaScore`][pydantic_evals.evaluators.GembaScore] — GEMBA translation quality (Kocmi & Federmann, 2023).

For RAG-style metrics modeled on Ragas (`Faithfulness`, `AnswerRelevance`, `ContextPrecision`,
`ContextRecall`), see the [`ragas`][pydantic_evals.evaluators.ragas] module. You can always reach for
[`LLMJudge`][pydantic_evals.evaluators.LLMJudge] directly with a bespoke rubric.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings

from .common import serialize_model_as_string
from .context import EvaluatorContext
from .evaluator import EvaluationReason, Evaluator, EvaluatorOutput
from .llm_as_a_judge import judge_g_eval, judge_gemba_da, judge_gemba_sqm

__all__ = (
    'GEval',
    'GembaScore',
)


@dataclass(repr=False)
class GEval(Evaluator[object, object, object]):
    """G-Eval-style chain-of-thought evaluator (Liu et al., 2023).

    The judge is shown the evaluation `criteria` and a list of explicit `evaluation_steps`,
    produces a short reasoning trace, and emits an integer score within `score_range`. Because
    the criteria and steps are user-supplied, `GEval` puts no structural requirements on
    `ctx.inputs` or `ctx.output`.

    !!! note "Simplified G-Eval"
        The paper computes a probability-weighted expectation over score tokens using log-probs.
        We ask the model for a direct integer score instead, trading some correlation with human
        judgment for provider-agnostic simplicity.
    """

    criteria: str
    evaluation_steps: list[str]
    score_range: tuple[int, int] = (1, 5)
    include_input: bool = False
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = None

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        inputs = ctx.inputs if self.include_input else None
        g_eval_output = await judge_g_eval(
            ctx.output,
            self.criteria,
            self.evaluation_steps,
            self.score_range,
            inputs=inputs,
            model=self.model,
            model_settings=self.model_settings,
        )
        return EvaluationReason(value=g_eval_output.score, reason=g_eval_output.reason)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())


@dataclass(repr=False)
class GembaScore(Evaluator[str, str, object]):
    """GEMBA translation quality evaluator (Kocmi & Federmann, 2023).

    Two variants are supported, matching the published prompts:

    - `'DA'` — Direct Assessment, integer score 0-100.
    - `'SQM'` — Scalar Quality Metrics, integer score 0-6.

    The source text comes from `ctx.inputs` (a `str`), the candidate translation from
    `ctx.output` (a `str`), and the optional human reference from `ctx.expected_output`.
    """

    source_lang: str
    target_lang: str
    score_type: Literal['DA', 'SQM'] = 'DA'
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = None

    async def evaluate(self, ctx: EvaluatorContext[str, str, object]) -> EvaluatorOutput:
        reference_text = ctx.expected_output

        if self.score_type == 'DA':
            gemba_output = await judge_gemba_da(
                ctx.inputs,
                ctx.output,
                self.source_lang,
                self.target_lang,
                reference=reference_text,
                model=self.model,
                model_settings=self.model_settings,
            )
        else:
            gemba_output = await judge_gemba_sqm(
                ctx.inputs,
                ctx.output,
                self.source_lang,
                self.target_lang,
                reference=reference_text,
                model=self.model,
                model_settings=self.model_settings,
            )
        return EvaluationReason(value=gemba_output.score, reason=gemba_output.reason)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def build_serialization_arguments(self):
        return serialize_model_as_string(super().build_serialization_arguments())
