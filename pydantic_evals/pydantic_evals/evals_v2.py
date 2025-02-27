"""
Workflow:
- You bookmark a span
"""
from __future__ import annotations as _annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, Literal, ParamSpec, TypeVar

import logfire_api
from pydantic_core import to_json

from pydantic_evals._utils import UNSET, Unset

P = ParamSpec('P')
OutputT = TypeVar('OutputT')


@dataclass
class Score:
    value: float
    reason: str | None = None


@dataclass
class Label:
    value: bool | str
    reason: str | None = None


@dataclass
class Usage:
    value: bool | str


@dataclass
class PythonAssertion:
    """An assertion that can be evaluated by the Python runtime.

    The assertion is evaluated by calling the `function` with the output of the task.
    The assertion fails if the function raises an AssertionError.
    """

    function: Callable[[Any], None | Awaitable[None]]
    kind: Literal['python'] = 'python'


@dataclass
class LlmAsAJudgeAssertion:
    """An assertion that can be evaluated by the LLM as a judge.

    The LLM will be prompted to evaluate the assertion by providing the output of the task and return a boolean.
    The assertion fails if the LLM returns False.
    """

    description: str
    kind: Literal['llm'] = 'llm'


Assertion = PythonAssertion | LlmAsAJudgeAssertion


class AssertionResult:
    passed: bool
    reason: str | None


async def evaluate_assertion(assertion: LlmAsAJudgeAssertion, output: Any) -> AssertionResult:
    # Note: we could try reworking this to handle a list of assertions in a single prompt
    # I think we shouldn't bother with that until later though
    from pydantic_ai import Agent

    LlmAssertionAgent = Agent(
        result_type=AssertionResult,
        system_prompt=(
            'Please decide whether the assertion passes or fails for the provided output.'
            ' If it fails, optionally provide a brief reason.'
        ),
    )

    serialized_output = to_json(output, fallback=repr)
    user_prompt = f'Output: {serialized_output}Assertion: {assertion.description}\n\n'
    return (await LlmAssertionAgent.run(user_prompt=user_prompt)).data


@dataclass
class TestCase:
    """A test case that can be evaluated.

    The "immutable" parts are analogous to a bookmarked span in logfire.
    The "mutable" parts are intended to be user-defined metadata.
    """

    # Immutable parts
    case_id: str  # f'{trace_id}:{span_id}'
    created_at: datetime  # When the case was created
    task_name: str  # The name of the task
    inputs: dict[str, Any]  # The inputs to the task

    # Mutable parts
    name: str  # Short name of the case
    comment: str | None  # User-provided comment on the case
    labels: dict[str, Label] = field(init=False, default_factory=dict)
    metadata: dict[str, Any]  # Additional metadata about the case
    assertions: list[Assertion]

@dataclass
class RunningEvalCase(Generic[OutputT]):
    case: TestCase

    case_span: logfire_api.LogfireSpan = field(repr=False)
    task_span: logfire_api.LogfireSpan = field(repr=False)

    scores: dict[str, Score] = field(init=False, default_factory=dict)
    labels: dict[str, Label] = field(init=False, default_factory=dict)
    usage: dict[str, Usage] = field(init=False, default_factory=dict)
    metadata: dict[str, Any] = field(init=False, default_factory=dict)

    # The assertion_results should be in direct correspondence with the case.assertions:
    assertion_results: list[AssertionResult] = field(init=False, default_factory=list)

    output: OutputT | Unset = field(init=False, default=UNSET)

class FinishedEvalCase(Generic[OutputT])

@dataclass
class EvalCase(Generic[OutputT]):
    case: TestCase

    case_span: logfire_api.LogfireSpan = field(repr=False)
    task_span: logfire_api.LogfireSpan = field(repr=False)

    scores: dict[str, Score] = field(init=False, default_factory=dict)
    labels: dict[str, Label] = field(init=False, default_factory=dict)
    usage: dict[str, Usage] = field(init=False, default_factory=dict)
    metadata: dict[str, Any] = field(init=False, default_factory=dict)

    # The assertion_results should be in direct correspondence with the case.assertions:
    assertion_results: list[AssertionResult] = field(init=False, default_factory=list)

    output: OutputT | Unset = field(init=False, default=UNSET)

Dataset = list[EvalCase[Any]]


class Eval(Generic[OutputT]):
    cases: list[EvalCase[OutputT]] = field(init=False, default_factory=list)
