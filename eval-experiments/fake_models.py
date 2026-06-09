"""Offline model helpers so the eval experiments run without API keys.

- `qa_agent()` -- answers a few trivia questions plausibly (full sentences).
- `tool_agent()` -- calls a `get_weather` tool, then answers.
- `person_agent()` -- a structured-output agent returning a `Person`.
- `judge_model()` -- a FunctionModel that emulates an LLM judge.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

_ANSWERS = {
    'france': 'The capital of France is Paris.',
    'japan': 'Tokyo is the capital of Japan.',
    'italy': 'That would be Rome.',
    '2 + 2': '2 + 2 = 4.',
    'love': 'positive',
    'sentiment': 'positive',
    'haiku': 'An endless blue swell\nwaves whisper against the shore\nsalt air fills my lungs',
    'limerick': 'There once was a cat from Peru\nwho dreamed he was eating his shoe.',
}


def _answer(question: str) -> str:
    q = question.lower()
    for key, value in _ANSWERS.items():
        if key in q:
            return value
    return "I'm not sure about that one."


def _user_text(messages: list[ModelMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    parts.append(part.content)
    return '\n'.join(parts)


def _qa_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=_answer(_user_text(messages[-1:])))])


def qa_agent(**kwargs: object) -> Agent[None, str]:
    """An agent that answers trivia in full sentences."""
    return Agent(FunctionModel(_qa_fn), **kwargs)  # type: ignore[arg-type]


def _has_tool_result(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts) for m in messages
    )


def _weather_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Stateless: call the tool first; once its result is in history, answer.
    if _has_tool_result(messages):
        return ModelResponse(parts=[TextPart(content='It is sunny in Paris today.')])
    return ModelResponse(parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'})])


def tool_agent() -> Agent[None, str]:
    """An agent that calls a `get_weather` tool, then answers."""
    agent = Agent(FunctionModel(_weather_fn), name='weather')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pyright: ignore[reportUnusedFunction]
        return 'sunny'

    return agent


class Person(BaseModel):
    """Structured output for `person_agent`."""

    name: str
    age: int


def _person_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'name': 'Ada', 'age': 36})])


def person_agent() -> Agent[None, Person]:
    """A structured-output agent returning a `Person`."""
    return Agent(FunctionModel(_person_fn), output_type=Person, name='people')


def _judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Emulate an LLM judge: pass unless the rubric clearly fails the output."""
    text = _user_text(messages)

    output = ''
    rubric = ''
    if m := re.search(r'<Output>(.*?)</Output>', text, re.DOTALL):
        output = m.group(1).strip()
    if m := re.search(r'<Rubric>(.*?)</Rubric>', text, re.DOTALL):
        rubric = m.group(1).strip()

    # Trivial heuristic: a "haiku" rubric passes only on 3-line output, etc.
    passed = True
    reason = 'looks good'
    if 'haiku' in rubric.lower() and len(output.splitlines()) != 3:
        passed, reason = False, 'not three lines'
    if 'limerick' in rubric.lower() and len(output.splitlines()) < 5:
        passed, reason = False, 'too short for a limerick'

    grading = {'reason': reason, 'pass': passed, 'score': 1.0 if passed else 0.0}
    return ModelResponse(parts=[TextPart(content=json.dumps(grading))])


def judge_model() -> FunctionModel:
    """A FunctionModel that emulates an LLM judge (passes unless the rubric clearly fails)."""
    return FunctionModel(_judge_fn)
