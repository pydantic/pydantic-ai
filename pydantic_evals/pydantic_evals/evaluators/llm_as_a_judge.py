from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import to_json

from pydantic_ai import Agent, StructuredDict, UserContent, models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import MULTI_MODAL_CONTENT_TYPES
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

__all__ = (
    'GEvalOutput',
    'GradingOutput',
    'judge_g_eval',
    'judge_input_output',
    'judge_input_output_expected',
    'judge_output',
    'judge_output_expected',
    'set_default_judge_model',
)


_default_model: models.Model | models.KnownModelName = 'openai:gpt-5.2'
_MAX_G_EVAL_SCORE_LEVELS = 20
_JUDGE_REASON_DESCRIPTION = 'A concise 1-2 sentence justification for the verdict.'


class GradingOutput(BaseModel, populate_by_name=True):
    """The output of a grading operation."""

    reason: str = Field(description=_JUDGE_REASON_DESCRIPTION)
    pass_: bool = Field(validation_alias='pass', serialization_alias='pass')
    score: float


@dataclass(frozen=True)
class _GradingResult:
    reason: str | None
    pass_: bool
    score: float


def _binary_grading_output_type(context: Sequence[str]) -> type[JsonSchemaValue]:
    """A pass/fail verdict for a judge that cannot write the reason.

    The question lives in the field description, so it names only the sections this prompt
    actually carries: a judge told to take an `<ExpectedOutput>` into account when there is
    none is being asked about something it cannot see.
    """
    considering = f', taking {" and ".join(context)} into account' if context else ''
    return StructuredDict(
        {
            'type': 'object',
            'properties': {
                'pass': {
                    'type': 'boolean',
                    'description': f'Is the statement in <Rubric> true for <Output>{considering}?',
                }
            },
            'required': ['pass'],
            'additionalProperties': False,
        },
        name='BinaryGrading',
        description='Judge an output against a rubric.',
    )


_non_text_judge_agent = Agent(name='judge_without_text')


def _resolve_judge_model(model: models.Model | models.KnownModelName | str | None) -> models.Model:
    model = model or _default_model
    if isinstance(model, models.Model):
        return model
    return models.infer_model(model)


def _model_supports_text_output(model: models.Model) -> bool:
    """Whether every model that may handle the request can produce the text judge's output shape."""
    if isinstance(model, FallbackModel):
        return all(_model_supports_text_output(candidate) for candidate in model.models)
    if isinstance(model, WrapperModel):
        return _model_supports_text_output(model.wrapped)
    return model.profile.get('supports_text_output', True)


async def _run_grading_agent(
    agent: Agent[None, GradingOutput],
    user_prompt: str | Sequence[str | UserContent],
    context: Sequence[str],
    model: models.Model | models.KnownModelName | str | None,
    model_settings: ModelSettings | None,
    *,
    allow_reasonless: bool,
) -> _GradingResult:
    resolved_model = _resolve_judge_model(model)
    if not _model_supports_text_output(resolved_model):
        if not allow_reasonless:
            raise UserError(
                'This judge model cannot generate the reason required by the `judge_*` helpers. '
                'Use the `LLMJudge` evaluator to record a reasonless verdict.'
            )
        verdict = (
            await _non_text_judge_agent.run(
                user_prompt,
                model=resolved_model,
                model_settings=model_settings,
                output_type=_binary_grading_output_type(context),
            )
        ).output.get('pass')
        if not isinstance(verdict, bool):
            raise ValueError(f'Judge returned an invalid verdict: {verdict!r}')
        return _GradingResult(reason=None, pass_=verdict, score=float(verdict))
    output = (await agent.run(user_prompt, model=resolved_model, model_settings=model_settings)).output
    return _GradingResult(reason=output.reason, pass_=output.pass_, score=output.score)


def _grading_output_with_reason(result: _GradingResult) -> GradingOutput:
    assert result.reason is not None
    return GradingOutput(reason=result.reason, pass_=result.pass_, score=result.score)


_JUDGE_REASON_INSTRUCTION = (
    '\nThe "reason" field must be a concise 1-2 sentence justification. '
    'Do not include your reasoning process, self-corrections, or re-checking in the reason. '
    'State only the final justification.'
)


_judge_output_agent = Agent(
    name='judge_output',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <Output>Hello world</Output>
        <Rubric>Content contains a greeting</Rubric>
        {"reason": "the content contains the word 'Hello'", "pass": true, "score": 1.0}

        <Output>Avast ye swabs, repel the invaders!</Output>
        <Rubric>Does not speak like a pirate</Rubric>
        {"reason": "'avast ye' is a common pirate term", "pass": false, "score": 0.0}
        """
    )
    + _JUDGE_REASON_INSTRUCTION,
    output_type=GradingOutput,
)


async def _judge_output(
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    *,
    allow_reasonless: bool,
) -> _GradingResult:
    user_prompt, context = _build_prompt(output=output, rubric=rubric)
    return await _run_grading_agent(
        _judge_output_agent, user_prompt, context, model, model_settings, allow_reasonless=allow_reasonless
    )


async def judge_output(
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-5.2',
    but this can be changed using the `set_default_judge_model` function.
    """
    result = await _judge_output(output, rubric, model, model_settings, allow_reasonless=False)
    return _grading_output_with_reason(result)


_judge_input_output_agent = Agent(
    name='judge_input_output',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided input and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <Input>Hello world</Input>
        <Output>Hello</Output>
        <Rubric>Content contains a greeting word which is present in the input</Rubric>
        {"reason": "the content contains the word 'Hello'", "pass": true, "score": 1.0}

        <Input>Pirate</Input>
        <Output>Avast ye swabs, repel the invaders!</Output>
        <Rubric>Does not speak in the style described by the input</Rubric>
        {"reason": "'avast ye' is a common pirate term", "pass": false, "score": 0.0}
        """
    )
    + _JUDGE_REASON_INSTRUCTION,
    output_type=GradingOutput,
)


async def _judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    *,
    allow_reasonless: bool,
) -> _GradingResult:
    user_prompt, context = _build_prompt(inputs=inputs, output=output, rubric=rubric)
    return await _run_grading_agent(
        _judge_input_output_agent, user_prompt, context, model, model_settings, allow_reasonless=allow_reasonless
    )


async def judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-5.2',
    but this can be changed using the `set_default_judge_model` function.
    """
    result = await _judge_input_output(inputs, output, rubric, model, model_settings, allow_reasonless=False)
    return _grading_output_with_reason(result)


_judge_input_output_expected_agent = Agent(
    name='judge_input_output_expected',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided input, expected output, and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <Input>What color is the sky?</Input>
        <Output>Cerulean</Output>
        <ExpectedOutput>Blue</ExpectedOutput>
        <Rubric>The output is consistent with the expected output but doesn't have to match exactly</Rubric>
        {"reason": "'Cerulean' is a shade of blue", "pass": true, "score": 1.0}

        <Input>How many legs does a spider have?</Input>
        <Output>Six</Output>
        <ExpectedOutput>8</ExpectedOutput>
        <Rubric>The output is factually consistent with the expected output</Rubric>
        {"reason": "Spiders have 8 legs", "pass": false, "score": 0.0}
        """
    )
    + _JUDGE_REASON_INSTRUCTION,
    output_type=GradingOutput,
)


async def _judge_input_output_expected(
    inputs: Any,
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    *,
    allow_reasonless: bool,
) -> _GradingResult:
    user_prompt, context = _build_prompt(inputs=inputs, output=output, rubric=rubric, expected_output=expected_output)
    return await _run_grading_agent(
        _judge_input_output_expected_agent,
        user_prompt,
        context,
        model,
        model_settings,
        allow_reasonless=allow_reasonless,
    )


async def judge_input_output_expected(
    inputs: Any,
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-5.2',
    but this can be changed using the `set_default_judge_model` function.
    """
    result = await _judge_input_output_expected(
        inputs, output, expected_output, rubric, model, model_settings, allow_reasonless=False
    )
    return _grading_output_with_reason(result)


_judge_output_expected_agent = Agent(
    name='judge_output_expected',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided expected output and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <Output>Cerulean</Output>
        <ExpectedOutput>Blue</ExpectedOutput>
        <Rubric>The output should be a shade of the expected output color</Rubric>
        {"reason": "'Cerulean' is a shade of blue", "pass": true, "score": 1.0}

        <Output>Six</Output>
        <ExpectedOutput>8</ExpectedOutput>
        <Rubric>The output should be a number written in words which matches the number written in digits in the expected output</Rubric>
        {"reason": "The output is 'Six' which is a different number than 8", "pass": false, "score": 0.0}
        """
    )
    + _JUDGE_REASON_INSTRUCTION,
    output_type=GradingOutput,
)


async def _judge_output_expected(
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    *,
    allow_reasonless: bool,
) -> _GradingResult:
    user_prompt, context = _build_prompt(output=output, rubric=rubric, expected_output=expected_output)
    return await _run_grading_agent(
        _judge_output_expected_agent, user_prompt, context, model, model_settings, allow_reasonless=allow_reasonless
    )


async def judge_output_expected(
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the expected output, output, and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-5.2',
    but this can be changed using the `set_default_judge_model` function.
    """
    result = await _judge_output_expected(
        output, expected_output, rubric, model, model_settings, allow_reasonless=False
    )
    return _grading_output_with_reason(result)


def set_default_judge_model(model: models.Model | models.KnownModelName) -> None:
    """Set the default model used for judging.

    This model is used if `None` is passed to the `model` argument of `judge_output` and `judge_input_output`.
    """
    global _default_model
    _default_model = model


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        # If the value can be serialized to JSON, use that.
        # If that behavior is undesirable, the user could manually call repr on the arguments to the judge_* functions
        return to_json(value).decode()
    except Exception:
        return repr(value)


def _make_section(content: Any, tag: str) -> list[str | UserContent]:
    """Create a tagged section, handling different content types, for use in the LLMJudge's prompt.

    Args:
        content (Any): content to include in the section_
        tag (str): tag name for the section

    Returns:
        list[str | UserContent]: the tagged section as a list of strings or UserContent
    """
    sections: list[str | UserContent] = []
    items: Sequence[str | UserContent] = (  # pyright: ignore[reportUnknownVariableType]
        content if isinstance(content, Sequence) and not isinstance(content, str) else [content]
    )

    sections.append(f'<{tag}>')
    for item in items:
        sections.append(item if isinstance(item, (str, *MULTI_MODAL_CONTENT_TYPES)) else _stringify(item))
    sections.append(f'</{tag}>')
    return sections


def _build_prompt(
    output: Any,
    rubric: str,
    inputs: Any | None = None,
    expected_output: Any | None = None,
) -> tuple[str | Sequence[str | UserContent], Sequence[str]]:
    """Build a prompt that includes input, output, expected output, and rubric.

    Sections are emitted in the same order the judge agents' system-prompt few-shot
    examples demonstrate — `Input → Output → ExpectedOutput → Rubric`, matching the
    `judge_input_output_expected` naming — so the runtime prompt matches the format the
    model was primed with and the rubric (the instruction) comes last, after all the
    context it applies to.

    Returns the prompt along with the optional context sections it carries, so that a judge
    which has to put its question in a field description can name what it was actually given.
    """
    sections: list[str | UserContent] = []
    context: list[str] = []
    if inputs is not None:
        sections.extend(_make_section(inputs, 'Input'))
        context.append('<Input>')

    sections.extend(_make_section(output, 'Output'))

    if expected_output is not None:
        sections.extend(_make_section(expected_output, 'ExpectedOutput'))
        context.append('<ExpectedOutput>')

    sections.extend(_make_section(rubric, 'Rubric'))
    if all(isinstance(section, str) for section in sections):
        return '\n'.join(sections), context  # type: ignore[arg-type]
    return sections, context


class GEvalOutput(BaseModel):
    """The output of a G-Eval grading operation.

    G-Eval asks the judge to emit a short chain-of-thought `reason` followed by an
    integer `score` in a user-specified range (see [`judge_g_eval`][pydantic_evals.evaluators.llm_as_a_judge.judge_g_eval]).
    """

    reason: str
    score: int


@dataclass(frozen=True)
class _GEvalResult:
    reason: str | None
    score: int


def _g_eval_output_type(score_range: tuple[int, int]) -> type[JsonSchemaValue]:
    """A normalized integer rubric for a judge that cannot write the reasoning trace."""
    minimum, maximum = score_range
    levels: list[dict[str, int | str]] = []
    for score in range(minimum, maximum + 1):
        if score == minimum:
            description = f'{score}: the worst score according to the evaluation criteria.'
        elif score == maximum:
            description = f'{score}: the best score according to the evaluation criteria.'
        else:
            description = f'{score}: an intermediate score between the worst and best.'
        levels.append({'const': score - minimum, 'description': description})
    return StructuredDict(
        {
            'type': 'object',
            'properties': {
                'score': {
                    'description': 'What score does the output earn according to the criteria and evaluation steps?',
                    'anyOf': levels,
                }
            },
            'required': ['score'],
            'additionalProperties': False,
        },
        name='GEvalScore',
        description='Grade the output using the evaluation criteria and steps.',
    )


_judge_g_eval_agent = Agent(
    name='judge_g_eval',
    system_prompt=dedent(
        """
        You are a rigorous evaluator scoring LLM outputs using the G-Eval framework.

        Follow the evaluation steps exactly, then return a JSON object with this structure:
        {"reason": string, "score": integer}

        - `reason`: a concise chain-of-thought summary of how you applied the evaluation steps.
        - `score`: a single integer within the score range specified in the prompt.

        Do not include any other keys or prose outside the JSON object.
        """
    ),
    output_type=GEvalOutput,
)


async def _judge_g_eval(
    output: Any,
    criteria: str,
    evaluation_steps: Sequence[str],
    score_range: tuple[int, int] = (1, 5),
    inputs: Any | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
    *,
    allow_reasonless: bool,
) -> _GEvalResult:
    if score_range[0] >= score_range[1]:
        raise ValueError(f'`score_range` must satisfy min < max, got {score_range!r}')
    if not evaluation_steps:
        raise ValueError('`evaluation_steps` must contain at least one step')

    numbered_steps = '\n'.join(f'{i}. {step}' for i, step in enumerate(evaluation_steps, start=1))
    rubric = '\n'.join(
        [
            f'Evaluation criteria: {criteria}',
            '',
            'Evaluation steps (apply each step in order):',
            numbered_steps,
            '',
            f'Produce a single integer score between {score_range[0]} and {score_range[1]} inclusive,',
            f'where {score_range[0]} is the worst and {score_range[1]} is the best according to the criteria.',
        ]
    )
    user_prompt, _ = _build_prompt(output=output, rubric=rubric, inputs=inputs)
    resolved_model = _resolve_judge_model(model)
    if not _model_supports_text_output(resolved_model):
        if not allow_reasonless:
            raise UserError(
                'This judge model cannot generate the reason required by `judge_g_eval`. '
                'Use the `GEval` evaluator to record a reasonless score.'
            )
        score_levels = score_range[1] - score_range[0] + 1
        if score_levels > _MAX_G_EVAL_SCORE_LEVELS:
            raise UserError(
                f'`score_range` can contain at most {_MAX_G_EVAL_SCORE_LEVELS} levels for a judge that does not '
                f'support text output; got {score_levels} in {score_range!r}.'
            )
        normalized = (
            await _non_text_judge_agent.run(
                user_prompt,
                model=resolved_model,
                model_settings=model_settings,
                output_type=_g_eval_output_type(score_range),
            )
        ).output.get('score')
        if not isinstance(normalized, int) or isinstance(normalized, bool):
            raise ValueError(f'Judge returned an invalid score: {normalized!r}')
        result = _GEvalResult(reason=None, score=normalized + score_range[0])
    else:
        output = (
            await _judge_g_eval_agent.run(user_prompt, model=resolved_model, model_settings=model_settings)
        ).output
        result = _GEvalResult(reason=output.reason, score=output.score)
    if not score_range[0] <= result.score <= score_range[1]:
        raise ValueError(f'Judge returned score {result.score}, outside the requested `score_range` {score_range!r}')
    return result


async def judge_g_eval(
    output: Any,
    criteria: str,
    evaluation_steps: Sequence[str],
    score_range: tuple[int, int] = (1, 5),
    inputs: Any | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GEvalOutput:
    """Judge an output using a G-Eval style chain-of-thought prompt.

    This is a simplified implementation of G-Eval (Liu et al., 2023, "G-Eval: NLG Evaluation using
    GPT-4 with Better Human Alignment"). The original paper computes an expectation over the
    distribution of score tokens using log-probs. We skip that step and simply ask the model for
    a direct integer score. This keeps the evaluator provider-agnostic at the cost of some
    correlation with human judgments.

    Args:
        output: The output being evaluated.
        criteria: The aspect being evaluated (e.g. "coherence", "fluency").
        evaluation_steps: Explicit chain-of-thought steps the judge should follow.
        score_range: Inclusive `(min, max)` integer score range.
        inputs: Optional inputs/context to show alongside the output.
        model: The model to use. If not specified, the default judge model is used.
        model_settings: Optional model settings.

    Returns:
        A [`GEvalOutput`][pydantic_evals.evaluators.llm_as_a_judge.GEvalOutput] containing
        the judge's reasoning and integer score.

    Raises:
        UserError: If the judge cannot generate the required reason.
        ValueError: If `score_range` is invalid, `evaluation_steps` is empty, or the judge
            returns a score outside the range.
    """
    result = await _judge_g_eval(
        output,
        criteria,
        evaluation_steps,
        score_range,
        inputs,
        model,
        model_settings,
        allow_reasonless=False,
    )
    assert result.reason is not None
    return GEvalOutput(reason=result.reason, score=result.score)
