"""Compare LLMJudge and GEval across Jev and two text-generating judges."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from typing_extensions import TypedDict

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GEval, LLMJudge, OutputConfig

sys.path.insert(0, str(Path(__file__).parent))
from cases import CASES, Area

MODELS = (
    'typesafe:jev-latest',
    'openai:gpt-5.6-sol',
    'anthropic:claude-opus-5',
)
RESULT_PATH = Path(__file__).with_name('bench_judge.txt')
CONCURRENCY = 8

URGENCY_RULE = (
    'The ticket is urgent under this rule: someone is blocked right now, money is moving wrongly right now, '
    'or the product is broken for more than the one person writing. Everything else is not urgent.'
)
AREA_RULE = (
    'billing means money already owed, charged, quoted or refunded; account means access, membership, permissions '
    'or user identity; bug means the product did something other than what it says; other means none of those.'
)


class _Labels(TypedDict):
    urgent: bool
    area: Area


class _CostTrackingModel(WrapperModel):
    """Collect costs reported on every request made by an evaluator."""

    responses: list[ModelResponse]

    def __init__(self, model: str):
        super().__init__(infer_model(model))
        self.reset_cost()

    def reset_cost(self) -> None:
        self.responses = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        # The agent fills best-effort prices after the model returns, so retain each response
        # and read its finalized usage after the dataset finishes.
        self.responses.append(response)
        return response


@dataclass
class _LLMJudgeResult:
    model: str
    urgent_agreement: int
    area_agreement: int
    both_agreement: int
    median_latency: float
    p95_latency: float
    wall_time: float
    cost: str


@dataclass
class _GEvalResult:
    model: str
    correlation: float
    median_latency: float
    p95_latency: float
    wall_time: float
    cost: str


def _identity(ticket: str) -> str:
    return ticket


def _cost_text(model: _CostTrackingModel) -> str:
    costs = [response.usage.cost for response in model.responses]
    requests_without_cost = costs.count(None)
    cost = sum((value for value in costs if value is not None), start=Decimal())
    if requests_without_cost == len(costs):
        return 'not reported'
    if requests_without_cost:
        return f'${cost:.4f} + {requests_without_cost} unpriced requests'
    return f'${cost:.4f}'


def _latency(report_cases: list[object]) -> tuple[float, float]:
    durations = sorted(case.total_duration for case in report_cases)  # type: ignore[attr-defined]
    return statistics.median(durations), durations[int(0.95 * len(durations)) - 1]


def _check_report(report: object) -> None:
    failures = report.failures  # type: ignore[attr-defined]
    evaluator_failures = [
        (case.name, failure)
        for case in report.cases  # type: ignore[attr-defined]
        for failure in case.evaluator_failures
    ]
    if failures or evaluator_failures:
        raise RuntimeError(f'Evaluation failures: task={failures!r}, evaluators={evaluator_failures!r}')


async def _run_llm_judge(model_name: str, model: _CostTrackingModel) -> _LLMJudgeResult:
    cases = []
    for index, (ticket, urgent, area) in enumerate(CASES):
        cases.append(
            Case(
                name=f'ticket-{index + 1:03}',
                inputs=ticket,
                metadata=_Labels(urgent=urgent, area=area),
                evaluators=(
                    LLMJudge(
                        rubric=URGENCY_RULE,
                        model=model,
                        assertion=OutputConfig(evaluation_name='urgent'),
                    ),
                    LLMJudge(
                        rubric=f'The ticket belongs to the {area} support area. Area definitions: {AREA_RULE}',
                        model=model,
                        assertion=OutputConfig(evaluation_name='area'),
                    ),
                ),
            )
        )
    dataset = Dataset(name='support-ticket-llm-judge', cases=cases)

    started = time.perf_counter()
    report = await dataset.evaluate(_identity, max_concurrency=CONCURRENCY, progress=False)
    wall_time = time.perf_counter() - started
    _check_report(report)

    urgent_agreement = sum(case.assertions['urgent'].value == case.metadata['urgent'] for case in report.cases)
    area_agreement = sum(case.assertions['area'].value for case in report.cases)
    both_agreement = sum(
        case.assertions['urgent'].value == case.metadata['urgent'] and case.assertions['area'].value
        for case in report.cases
    )
    median_latency, p95_latency = _latency(report.cases)
    return _LLMJudgeResult(
        model_name,
        urgent_agreement,
        area_agreement,
        both_agreement,
        median_latency,
        p95_latency,
        wall_time,
        _cost_text(model),
    )


async def _run_g_eval(model_name: str, model: _CostTrackingModel) -> _GEvalResult:
    evaluator = GEval(
        criteria='Urgency under the support-ticket rule.',
        evaluation_steps=[
            'Decide whether someone is blocked right now, money is moving wrongly right now, or the product is broken '
            'for more than the one person writing.',
            'Assign 4 when the ticket is urgent under that rule and 0 when it is not; use intermediate scores only '
            'when the evidence is ambiguous.',
        ],
        score_range=(0, 4),
        model=model,
        evaluation_name='urgency_score',
    )
    dataset = Dataset(
        name='support-ticket-g-eval',
        cases=[
            Case(name=f'ticket-{index + 1:03}', inputs=ticket, metadata=_Labels(urgent=urgent, area=area))
            for index, (ticket, urgent, area) in enumerate(CASES)
        ],
        evaluators=[evaluator],
    )

    started = time.perf_counter()
    report = await dataset.evaluate(_identity, max_concurrency=CONCURRENCY, progress=False)
    wall_time = time.perf_counter() - started
    _check_report(report)

    scores = [float(case.scores['urgency_score'].value) for case in report.cases]
    labels = [4.0 if case.metadata['urgent'] else 0.0 for case in report.cases]
    try:
        correlation = statistics.correlation(scores, labels)
    except statistics.StatisticsError:
        correlation = float('nan')
    median_latency, p95_latency = _latency(report.cases)
    return _GEvalResult(model_name, correlation, median_latency, p95_latency, wall_time, _cost_text(model))


def _render(llm_results: list[_LLMJudgeResult], g_eval_results: list[_GEvalResult]) -> str:
    lines = [
        '# Evaluation-judge benchmark',
        '',
        '120 support tickets; concurrency 8. LLMJudge latency is per ticket for the urgency and area judges together.',
        'GEval correlation is Pearson correlation against urgent → 4 and non-urgent → 0.',
        '',
        '## LLMJudge',
        '',
        '| judge | urgency agreement | area agreement | both labels | median | p95 | wall time (120) | cost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    lines.extend(
        f'| `{result.model}` | {result.urgent_agreement}/120 | {result.area_agreement}/120 | '
        f'{result.both_agreement}/120 | {result.median_latency * 1000:.0f} ms | '
        f'{result.p95_latency * 1000:.0f} ms | {result.wall_time:.1f} s | {result.cost} |'
        for result in llm_results
    )
    lines.extend(
        [
            '',
            '## GEval (urgency, 0–4)',
            '',
            '| judge | score correlation | median | p95 | wall time (120) | cost |',
            '|---|---:|---:|---:|---:|---:|',
        ]
    )
    lines.extend(
        f'| `{result.model}` | {result.correlation:.3f} | {result.median_latency * 1000:.0f} ms | '
        f'{result.p95_latency * 1000:.0f} ms | {result.wall_time:.1f} s | {result.cost} |'
        for result in g_eval_results
    )
    return '\n'.join(lines) + '\n'


async def _main() -> None:
    llm_results: list[_LLMJudgeResult] = []
    g_eval_results: list[_GEvalResult] = []
    for model_name in MODELS:
        print(f'Running LLMJudge with {model_name}...', flush=True)
        model = _CostTrackingModel(model_name)
        llm_results.append(await _run_llm_judge(model_name, model))
        RESULT_PATH.write_text(_render(llm_results, g_eval_results))
        print(_render([llm_results[-1]], []), flush=True)

        print(f'Running GEval with {model_name}...', flush=True)
        model.reset_cost()
        g_eval_results.append(await _run_g_eval(model_name, model))
        RESULT_PATH.write_text(_render(llm_results, g_eval_results))
        print(_render([], [g_eval_results[-1]]), flush=True)


if __name__ == '__main__':
    asyncio.run(_main())
