from __future__ import annotations

from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_json

from pydantic_ai import Agent, models

__all__ = ('GradingOutput', 'judge_input_output', 'judge_output', 'set_default_judge_model')


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
    result_type=GradingOutput,
)


async def judge_output(
    output: Any, rubric: str, model: models.Model | models.KnownModelName | None = None
) -> GradingOutput:
    """Judge the output of a model based on a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = f'<Output>\n{_stringify(output)}\n</Output>\n<Rubric>\n{rubric}\n</Rubric>'
    return (await _judge_output_agent.run(user_prompt, model=model or _default_model)).data


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
    result_type=GradingOutput,
)


async def judge_input_output(
    inputs: Any, output: Any, rubric: str, model: models.Model | models.KnownModelName | None = None
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = f'<Input>\n{_stringify(inputs)}\n</Input>\n<Output>\n{_stringify(output)}\n</Output>\n<Rubric>\n{rubric}\n</Rubric>'
    return (await _judge_input_output_agent.run(user_prompt, model=model or _default_model)).data


def set_default_judge_model(model: models.Model | models.KnownModelName) -> None:  # pragma: no cover
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
