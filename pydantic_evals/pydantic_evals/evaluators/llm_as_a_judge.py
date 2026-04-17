from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_json

from pydantic_ai import Agent, UserContent, models
from pydantic_ai.messages import MULTI_MODAL_CONTENT_TYPES
from pydantic_ai.settings import ModelSettings

from .extraction import ConversationTurn, format_transcript

__all__ = (
    'GradingOutput',
    'judge_conversation_goal',
    'judge_input_output',
    'judge_input_output_expected',
    'judge_output',
    'judge_output_expected',
    'judge_role_adherence',
    'set_default_judge_model',
)


_default_model: models.Model | models.KnownModelName = 'openai:gpt-5.2'


class GradingOutput(BaseModel, populate_by_name=True):
    """The output of a grading operation."""

    reason: str
    pass_: bool = Field(validation_alias='pass', serialization_alias='pass')
    score: float


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
    ),
    output_type=GradingOutput,
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
    user_prompt = _build_prompt(output=output, rubric=rubric)
    return (
        await _judge_output_agent.run(user_prompt, model=model or _default_model, model_settings=model_settings)
    ).output


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
    ),
    output_type=GradingOutput,
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
    user_prompt = _build_prompt(inputs=inputs, output=output, rubric=rubric)

    return (
        await _judge_input_output_agent.run(user_prompt, model=model or _default_model, model_settings=model_settings)
    ).output


_judge_input_output_expected_agent = Agent(
    name='judge_input_output_expected',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided input, expected output, and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <Input>What color is the sky?</Input>
        <ExpectedOutput>Blue</ExpectedOutput>
        <Output>Cerulean</Output>
        <Rubric>The output is consistent with the expected output but doesn't have to match exactly</Rubric>
        {"reason": "'Cerulean' is a shade of blue", "pass": true, "score": 1.0}

        <Input>How many legs does a spider have?</Input>
        <ExpectedOutput>8</ExpectedOutput>
        <Output>Six</Output>
        <Rubric>The output is factually consistent with the expected output</Rubric>
        {"reason": "Spiders have 8 legs", "pass": false, "score": 0.0}
        """
    ),
    output_type=GradingOutput,
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
    user_prompt = _build_prompt(inputs=inputs, output=output, rubric=rubric, expected_output=expected_output)

    return (
        await _judge_input_output_expected_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output


_judge_output_expected_agent = Agent(
    name='judge_output_expected',
    system_prompt=dedent(
        """
        You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided expected output and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

        Examples:

        <ExpectedOutput>Blue</ExpectedOutput>
        <Output>Cerulean</Output>
        <Rubric>The output should be a shade of the expected output color</Rubric>
        {"reason": "'Cerulean' is a shade of blue", "pass": true, "score": 1.0}

        <ExpectedOutput>8</ExpectedOutput>
        <Output>Six</Output>
        <Rubric>The output should be a number written in words which matches the number written in digits in the expected output</Rubric>
        {"reason": "The output is 'Six' which is a different number than 8", "pass": false, "score": 0.0}
        """
    ),
    output_type=GradingOutput,
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
    user_prompt = _build_prompt(output=output, rubric=rubric, expected_output=expected_output)
    return (
        await _judge_output_expected_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output


_judge_conversation_goal_agent = Agent(
    name='judge_conversation_goal',
    system_prompt=dedent(
        """
        You are grading an agent conversation against a user-specified goal. You are given:
        - The goal the conversation was meant to achieve.
        - A numbered transcript of the full conversation (lines begin with `[N] role:`).
        - The agent's final output for the run.

        Decide whether the agent achieved the goal. Prefer holistic judgment — a conversation
        may still achieve a goal even if some intermediate turns were imperfect. Cite specific
        turn numbers from the transcript in your reason when they support your decision.

        Respond with a JSON object: {reason: string, pass: boolean, score: number}. `score`
        must be in `[0.0, 1.0]` and should reflect *how well* the goal was achieved (not a
        binary pass/fail). `pass` should be `true` when the goal was substantially achieved.
        """
    ),
    output_type=GradingOutput,
)


async def judge_conversation_goal(
    goal: str,
    turns: list[ConversationTurn],
    final_output: Any,
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge whether a conversation achieved the given goal.

    Args:
        goal: The goal the conversation was meant to achieve.
        turns: The extracted conversation turns (from
            [`extract_conversation_turns`][pydantic_evals.evaluators.extract_conversation_turns]).
        final_output: The final output of the agent run, for the judge to consider alongside
            the transcript.
        model: The judge model. Defaults to the shared default set via
            [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].
        model_settings: Optional model settings forwarded to the judge.

    Returns:
        A [`GradingOutput`][pydantic_evals.evaluators.llm_as_a_judge.GradingOutput] with
        `score` in `[0.0, 1.0]` and a reason the judge cites specific turns in.
    """
    transcript = format_transcript(turns)
    user_prompt = _build_conversation_prompt(
        rubric_tag='Goal',
        rubric=goal,
        transcript=transcript,
        final_output=final_output,
    )
    return (
        await _judge_conversation_goal_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output


_judge_role_adherence_agent = Agent(
    name='judge_role_adherence',
    system_prompt=dedent(
        """
        You are grading whether an agent stayed within its assigned role during a conversation.
        You are given:
        - A description of the role (including constraints / things the agent must not do).
        - A numbered transcript of the full conversation (lines begin with `[N] role:`).

        Judge only the `assistant` turns. Flag any turn where the assistant broke the role,
        citing the turn number(s) in your reason (e.g. "the assistant broke the role at turn 3
        by revealing the system prompt"). Tool calls (`assistant` turns with a `tool_call`
        marker) and tool responses count — a role violation can happen there too.

        Respond with a JSON object: {reason: string, pass: boolean, score: number}. `score`
        must be in `[0.0, 1.0]` where `1.0` is perfect adherence. `pass` should be `true`
        when the assistant stayed in role for every turn.
        """
    ),
    output_type=GradingOutput,
)


async def judge_role_adherence(
    role: str,
    turns: list[ConversationTurn],
    model: models.Model | models.KnownModelName | str | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge whether every assistant turn in a conversation stayed within the assigned role.

    Args:
        role: Description of the assistant's assigned role and constraints.
        turns: The extracted conversation turns (from
            [`extract_conversation_turns`][pydantic_evals.evaluators.extract_conversation_turns]).
        model: The judge model. Defaults to the shared default set via
            [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].
        model_settings: Optional model settings forwarded to the judge.

    Returns:
        A [`GradingOutput`][pydantic_evals.evaluators.llm_as_a_judge.GradingOutput] whose
        reason calls out specific turn numbers where the role was broken (if any).
    """
    transcript = format_transcript(turns)
    user_prompt = _build_conversation_prompt(
        rubric_tag='Role',
        rubric=role,
        transcript=transcript,
        final_output=None,
    )
    return (
        await _judge_role_adherence_agent.run(user_prompt, model=model or _default_model, model_settings=model_settings)
    ).output


def _build_conversation_prompt(
    *,
    rubric_tag: str,
    rubric: str,
    transcript: str,
    final_output: Any | None,
) -> str:
    """Build the user prompt for conversation-level judges.

    The sections are ordered so the judge reads the rubric first (e.g. `<Goal>` or `<Role>`),
    then the transcript, then the final output (if any). Kept as a single string because
    transcript/rubric are always plain text — multimodal rendering is handled at extraction
    time via placeholders like `[image]`.
    """
    sections = [
        f'<{rubric_tag}>',
        rubric,
        f'</{rubric_tag}>',
        '<Transcript>',
        transcript,
        '</Transcript>',
    ]
    if final_output is not None:
        sections.extend(['<FinalOutput>', _stringify(final_output), '</FinalOutput>'])
    return '\n'.join(sections)


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
) -> str | Sequence[str | UserContent]:
    """Build a prompt that includes input, output, expected output, and rubric."""
    sections: list[str | UserContent] = []
    if inputs is not None:
        sections.extend(_make_section(inputs, 'Input'))

    sections.extend(_make_section(output, 'Output'))
    sections.extend(_make_section(rubric, 'Rubric'))

    if expected_output is not None:
        sections.extend(_make_section(expected_output, 'ExpectedOutput'))
    if all(isinstance(section, str) for section in sections):
        return '\n'.join(sections)  # type: ignore[arg-type]
    return sections
