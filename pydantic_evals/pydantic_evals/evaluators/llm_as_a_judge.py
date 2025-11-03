from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_json

from pydantic_ai import Agent, MultiModalContent, UserContent, models
from pydantic_ai.settings import ModelSettings

__all__ = (
    'GradingOutput',
    'judge_input_output',
    'judge_input_output_expected',
    'judge_output',
    'judge_output_expected',
    'set_default_judge_model',
)


_default_model: models.Model | models.KnownModelName = 'openai:gpt-4o'


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
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
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
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
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
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
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
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the expected output, output, and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = _build_prompt(output=output, rubric=rubric, expected_output=expected_output)
    return (
        await _judge_output_expected_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output


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


def _build_prompt(
    output: Any,
    rubric: str,
    inputs: Any | None = None,
    expected_output: Any | None = None,
) -> str | Sequence[str | UserContent]:
    """Build a prompt that includes input, output, and rubric."""
    sections: list[str | UserContent] = []

    if inputs is not None:
        if isinstance(inputs, str):
            sections.append(f'<Input>\n{inputs}\n</Input>')
        else:
            sections.append('<Input>\n')
            if isinstance(inputs, Sequence):
                for item in inputs:  # type: ignore
                    if isinstance(item, str | MultiModalContent):
                        sections.append(item)
                    else:
                        sections.append(_stringify(item))
            elif isinstance(inputs, MultiModalContent):
                sections.append(inputs)
            else:
                sections.append(_stringify(inputs))
            sections.append('</Input>')

    sections.append(f'<Output>\n{_stringify(output)}\n</Output>')
    sections.append(f'<Rubric>\n{rubric}\n</Rubric>')

    if expected_output is not None:
        sections.append(f'<ExpectedOutput>\n{_stringify(expected_output)}\n</ExpectedOutput>')

    if inputs is None or isinstance(inputs, str):
        return '\n\n'.join(sections)  # type: ignore[arg-type]
    else:
        return sections
