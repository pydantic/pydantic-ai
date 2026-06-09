"""easy_evals core -- the FACADE: translate friendly kwargs into real evaluators.

Every kwarg maps to an existing `pydantic_evals` evaluator or one of the proposed
new built-ins in `easy_evals.evaluators`. Cases are real `pydantic_evals.Case`s;
runs go through the real `Dataset.evaluate`; pass/fail is read from the real
`EvaluationReport`. This module contains NO evaluation logic of its own.

The facade is generic over the agent's output type `OutputT`, so `expect=`,
`passes=` and structured checks are type-checked against what the agent returns.

kwarg          -> evaluator
-----------------------------------------------
expect=        -> Contains(value, case_sensitive=False)   (forgiving text)
expect={...}   -> HasFields(...)                          (structured output)
equals=        -> Equals(value)                           (strict)
contains=      -> Contains(...)  (str or list)
excludes=      -> NotContains(...)
one_of=        -> OneOf(...)
matches=       -> Matches(pattern)
max_words=     -> MaxLength(words=...)
max_chars=     -> MaxLength(chars=...)
max_duration=  -> MaxDuration(seconds=...)
judge=         -> LLMJudge(rubric='The output ...')
calls_tool=    -> HasMatchingSpan(...)   (auto-instruments tracing)
passes=        -> a function escape hatch (the only wrapper)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol

from typing_extensions import TypeAlias, TypeVar

from pydantic_ai import Agent, models
from pydantic_ai.agent import AbstractAgent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    Equals,
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    HasMatchingSpan,
    LLMJudge,
    MaxDuration,
)
from pydantic_evals.reporting import EvaluationReport

from .evaluators import HasFields, Matches, MaxLength, NotContains, OneOf

OutputT = TypeVar('OutputT', default=str)
JudgeModel: TypeAlias = 'models.Model | models.KnownModelName | str | None'


# --- the only custom evaluator: the function escape hatch ---------------------


@dataclass(repr=False)
class _Check(Evaluator[object, OutputT, object]):
    """Escape hatch: any `(output) -> bool` function. (Not serializable -- by design.)"""

    fn: Callable[[OutputT], bool]

    def evaluate(self, ctx: EvaluatorContext[object, OutputT, object]) -> bool:
        return bool(self.fn(ctx.output))

    def get_default_evaluation_name(self) -> str:
        name = getattr(self.fn, '__name__', 'check')
        return name if isinstance(name, str) else 'check'


# --- a Case is plain data; it translates itself into real evaluators ----------


@dataclass
class case(Generic[OutputT]):  # noqa: N801 -- reads like a value/factory, not a class
    """One eval case: a prompt plus what you expect. Pure data; translates to a real Case."""

    prompt: str
    expect: str | dict[str, object] | None = None
    equals: str | None = None
    contains: str | Sequence[str] | None = None
    excludes: str | Sequence[str] | None = None
    one_of: Sequence[str] | None = None
    matches: str | None = None
    max_words: int | None = None
    max_chars: int | None = None
    max_duration: float | None = None
    judge: str | None = None
    calls_tool: str | None = None
    passes: Callable[[OutputT], bool] | None = None
    name: str | None = None

    @property
    def uses_spans(self) -> bool:
        return self.calls_tool is not None

    def _evaluators(self, judge_model: JudgeModel) -> list[Evaluator[object, OutputT, object]]:
        """Translate the friendly fields into real Evaluator instances. The whole facade."""
        out: list[Evaluator[object, OutputT, object]] = []
        if isinstance(self.expect, dict):
            out.append(HasFields(fields=self.expect, evaluation_name='expect'))
        elif self.expect is not None:
            out.append(Contains(value=self.expect, case_sensitive=False, evaluation_name='expect'))
        if self.equals is not None:
            out.append(Equals(value=self.equals, evaluation_name='equals'))
        if self.calls_tool is not None:
            # Pydantic AI's instrumentation emits a 'running tool' span carrying the OpenTelemetry
            # GenAI attribute `gen_ai.tool.name` (semconv: gen_ai.execute_tool). Matching on both keeps
            # this aligned with the spans that show up in production telemetry / Logfire.
            out.append(
                HasMatchingSpan(
                    query={'name_equals': 'running tool', 'has_attributes': {'gen_ai.tool.name': self.calls_tool}},
                    evaluation_name=f'calls_{self.calls_tool}',
                )
            )
        if self.contains is not None:
            for item in [self.contains] if isinstance(self.contains, str) else self.contains:
                out.append(Contains(value=item, case_sensitive=False))
        if self.excludes is not None:
            out.append(NotContains(value=self.excludes, evaluation_name='excludes'))
        if self.one_of is not None:
            out.append(OneOf(options=self.one_of))
        if self.matches is not None:
            out.append(Matches(pattern=self.matches))
        if self.max_words is not None:
            out.append(MaxLength(words=self.max_words))
        if self.max_chars is not None:
            out.append(MaxLength(chars=self.max_chars))
        if self.max_duration is not None:
            out.append(MaxDuration(seconds=self.max_duration))
        if self.judge is not None:
            out.append(LLMJudge(rubric=f'The output {self.judge}.', model=judge_model, assertion={'evaluation_name': 'judge'}))
        if self.passes is not None:
            out.append(_Check(self.passes))
        return out

    def to_case(self, judge_model: JudgeModel = None) -> Case[str, OutputT, None]:
        """Translate to a real pydantic_evals.Case (the escape hatch to the full library)."""
        label = self.name or (self.prompt if len(self.prompt) <= 40 else self.prompt[:37] + '...')
        return Case[str, OutputT, None](
            name=label,
            inputs=self.prompt,
            evaluators=tuple(self._evaluators(judge_model)),
        )

    def check_against(self, agent: AbstractAgent[Any, OutputT], *, judge_model: JudgeModel = None) -> None:
        """Run this case via the real engine; raise AssertionError if any matcher fails."""
        report = _evaluate([self], agent, judge_model=judge_model, quiet=True)
        _raise_for(report.cases[0])


# --- reading results -- depends only on the shape we use, not the generics ----


class _CaseResult(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def output(self) -> object: ...
    @property
    def assertions(self) -> Mapping[str, EvaluationResult[bool]]: ...


def case_failures(report_case: _CaseResult) -> list[str]:
    """Human-readable failure lines for one case (shared by check() and the pytest plugin)."""
    return [f'  - {name}: {res.reason or "failed"}' for name, res in report_case.assertions.items() if not res.value]


def _raise_for(rc: _CaseResult) -> None:
    if failures := case_failures(rc):
        raise AssertionError(f'Eval failed for {rc.name!r}\n  answer: {rc.output!r}\n' + '\n'.join(failures))


# --- running: everything goes through the real pydantic_evals engine ----------


def _dataset_for(
    agent: AbstractAgent[Any, OutputT], cases: Sequence[case[OutputT]], judge_model: JudgeModel
) -> Dataset[str, OutputT, None]:
    pyd_cases: list[Case[str, OutputT, None]] = [c.to_case(judge_model) for c in cases]
    return Dataset(name=agent.name or 'evals', cases=pyd_cases)


def _evaluate(
    cases: Sequence[case[OutputT]],
    agent: AbstractAgent[Any, OutputT],
    *,
    judge_model: JudgeModel,
    quiet: bool,
    max_concurrency: int | None = None,
    repeat: int = 1,
) -> EvaluationReport[str, OutputT, None]:
    if any(c.uses_spans for c in cases):
        _ensure_tracing()

    async def task(prompt: str) -> OutputT:
        result = await agent.run(prompt)
        return result.output

    return _dataset_for(agent, cases, judge_model).evaluate_sync(
        task, name=agent.name or 'evals', max_concurrency=max_concurrency, progress=not quiet, repeat=repeat
    )


def check(
    agent: AbstractAgent[Any, OutputT],
    prompt: str,
    *,
    judge_model: JudgeModel = None,
    expect: str | dict[str, object] | None = None,
    equals: str | None = None,
    contains: str | Sequence[str] | None = None,
    excludes: str | Sequence[str] | None = None,
    one_of: Sequence[str] | None = None,
    matches: str | None = None,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_duration: float | None = None,
    judge: str | None = None,
    calls_tool: str | None = None,
    passes: Callable[[OutputT], bool] | None = None,
) -> None:
    """Immediate single check: run the agent now and assert. Works in pytest or a script."""
    one = case[OutputT](
        prompt=prompt, name=prompt, expect=expect, equals=equals, contains=contains, excludes=excludes,
        one_of=one_of, matches=matches, max_words=max_words, max_chars=max_chars, max_duration=max_duration,
        judge=judge, calls_tool=calls_tool, passes=passes,
    )
    one.check_against(agent, judge_model=judge_model)


class EvalSuite(Generic[OutputT]):
    """A collection of eval cases colocated with one agent."""

    def __init__(
        self,
        agent: AbstractAgent[Any, OutputT],
        *,
        cases: Sequence[case[OutputT]] | None = None,
        judge_model: JudgeModel = None,
    ):
        self._agent = agent
        self._judge_model = judge_model
        self._cases: list[case[OutputT]] = list(cases) if cases else []

    def case(
        self,
        prompt: str,
        *,
        expect: str | dict[str, object] | None = None,
        equals: str | None = None,
        contains: str | Sequence[str] | None = None,
        excludes: str | Sequence[str] | None = None,
        one_of: Sequence[str] | None = None,
        matches: str | None = None,
        max_words: int | None = None,
        max_chars: int | None = None,
        max_duration: float | None = None,
        judge: str | None = None,
        calls_tool: str | None = None,
        passes: Callable[[OutputT], bool] | None = None,
        name: str | None = None,
    ) -> EvalSuite[OutputT]:
        """Add a case. Chainable."""
        self._cases.append(
            case[OutputT](
                prompt=prompt, name=name or f'{len(self._cases) + 1}. {prompt[:40]}',
                expect=expect, equals=equals, contains=contains, excludes=excludes, one_of=one_of,
                matches=matches, max_words=max_words, max_chars=max_chars, max_duration=max_duration,
                judge=judge, calls_tool=calls_tool, passes=passes,
            )
        )
        return self

    @property
    def case_names(self) -> list[str]:
        return [c.name or c.prompt for c in self._cases]

    def to_dataset(self) -> Dataset[str, OutputT, None]:
        """The real pydantic_evals.Dataset behind this suite (e.g. to push to Logfire)."""
        return _dataset_for(self._agent, self._cases, self._judge_model)

    def generate(
        self, *, n: int = 5, about: str | None = None, model: models.Model | models.KnownModelName | None = None
    ) -> EvalSuite[OutputT]:
        """Generate `n` example cases with an LLM and add them to the suite (beats the blank page).

        Each generated case becomes a prompt with a forgiving `expect`. Runs the generator
        synchronously, so call it from a normal script (not inside a running event loop).
        """
        import asyncio

        from pydantic_evals.generation import generate_dataset

        async def _gen() -> Dataset[str, str, None]:
            if model is None:
                return await generate_dataset(dataset_type=Dataset[str, str, None], n_examples=n, extra_instructions=about)
            return await generate_dataset(
                dataset_type=Dataset[str, str, None], n_examples=n, extra_instructions=about, model=model
            )

        for c in asyncio.run(_gen()).cases:
            self._cases.append(case[OutputT](prompt=c.inputs, expect=c.expected_output, name=c.name))
        return self

    def report(self, *, max_concurrency: int | None = None, repeat: int = 1) -> EvaluationReport[str, OutputT, None]:
        """Run every case once (concurrently) and return the raw report. `repeat` runs each case N×."""
        return _evaluate(
            self._cases, self._agent, judge_model=self._judge_model,
            quiet=True, max_concurrency=max_concurrency, repeat=repeat,
        )

    def run(
        self,
        *,
        max_concurrency: int | None = None,
        repeat: int = 1,
        experiment: str | None = None,
        baseline: EvaluationReport[str, OutputT, None] | None = None,
    ) -> bool:
        """Run every case, print a report, and return True if all passed.

        `repeat` runs each case N times (LLM outputs are flaky); `baseline` prints a diff
        against an earlier report (e.g. `before = suite.report(); ...; suite.run(baseline=before)`).
        """
        report = _evaluate(
            self._cases, self._agent, judge_model=self._judge_model,
            quiet=False, max_concurrency=max_concurrency, repeat=repeat,
        )
        report.print(include_input=True, include_output=True, include_durations=False, baseline=baseline)
        if experiment is not None:  # hosted: results stream to the Logfire UI
            from .hosted import experiment_url

            print(f'\nView this experiment in Logfire: {experiment_url(experiment)}')
        averages = report.averages()
        return bool(averages and averages.assertions == 1.0)


def eval_suite(
    agent: AbstractAgent[Any, OutputT],
    *,
    cases: Sequence[case[OutputT]] | None = None,
    judge_model: JudgeModel = None,
) -> EvalSuite[OutputT]:
    """Start an eval suite for an agent."""
    return EvalSuite(agent, cases=cases, judge_model=judge_model)


# --- tracing setup so span matchers (calls_tool) 'just work' ------------------


def _ensure_tracing() -> None:
    """Configure OpenTelemetry + agent instrumentation so span-based matchers work with no setup."""
    from opentelemetry import trace

    if type(trace.get_tracer_provider()).__name__ in ('ProxyTracerProvider', 'NoOpTracerProvider'):
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(TracerProvider())
    Agent.instrument_all()

