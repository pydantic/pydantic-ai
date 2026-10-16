"""A scripted `demo:tour` model for the README screenshots: canned turns, no network, no spend.

`capture.sh` copies this file into a throwaway CLAI configuration as a drop-in plugin, so every
built-in plugin still loads and every tool call in a screenshot really runs. Each turn picks a
scene by a keyword in the prompt and streams one step per model request.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_clai2.plugins import ModelProvider, Plugin

T = TypeVar('T')

CHUNK = 6
"""Characters per streamed delta: small enough that eager execution and speculation have time to act."""

PAUSE = 0.015
"""Seconds between deltas."""


@dataclass(frozen=True, kw_only=True)
class Step:
    """One model response: optional thinking and text, then optionally one tool call."""

    thinking: str = ''
    text: str = ''
    tool: str = ''
    args: dict[str, object] = field(default_factory=dict[str, object])


CLI_OLD = """\
    forecast.add_argument('--days', type=int, default=3)
    args = parser.parse_args(argv)

    for day in fetch_forecast(args.city, days=args.days):
        print(f'{day.date}  {day.summary:<12} {day.high}°/{day.low}°')
"""

CLI_NEW = """\
    forecast.add_argument('--days', type=int, default=3)
    forecast.add_argument('--units', choices=['c', 'f'], default='c', help='Celsius or Fahrenheit')
    args = parser.parse_args(argv)

    for day in fetch_forecast(args.city, days=args.days):
        high, low = (day.high, day.low) if args.units == 'c' else (to_f(day.high), to_f(day.low))
        print(f'{day.date}  {day.summary:<12} {high}°/{low}°')


def to_f(celsius):
    return round(celsius * 9 / 5 + 32)
"""

CI_WORKFLOW = """\
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python -m unittest -v
"""

SCENES: dict[str, tuple[Step, ...]] = {
    '--units flag': (
        Step(
            thinking='I will read the CLI and its tests first, so the new flag matches the existing options.',
            tool='run_code',
            args={
                'code': "cli = await read_file(path='weather/cli.py')\n"
                "tests = await grep(pattern='def test_', path='tests')\n"
                'print(tests)\n'
            },
        ),
        Step(
            tool='edit_file',
            args={'path': 'weather/cli.py', 'old_text': CLI_OLD, 'new_text': CLI_NEW},
        ),
        Step(
            tool='run_code',
            args={'code': "result = await shell(command='python3 -m unittest -q')\nprint(result)\n"},
        ),
        Step(
            text='Added **`--units f`** to `weather forecast` for Fahrenheit; Celsius stays the default. '
            'All three tests still pass.\n'
        ),
    ),
    'set up ci': (
        Step(
            thinking='There are several ways to run CI. Rather than guess, I will ask which one they use.',
            tool='run_code',
            args={
                'code': 'answer = await ask_user_question(questions=[{\n'
                "    'header': 'CI service',\n"
                "    'question': 'Where should the tests run?',\n"
                "    'options': [\n"
                "        {'label': 'GitHub Actions', 'description': 'A workflow in .github/workflows'},\n"
                "        {'label': 'GitLab CI', 'description': 'A .gitlab-ci.yml pipeline'},\n"
                "        {'label': 'CircleCI', 'description': 'A .circleci/config.yml'},\n"
                '    ],\n'
                '}])\n'
                'print(answer)\n'
            },
        ),
        Step(
            text='GitHub Actions it is. One job runs the tests on every push and pull request.\n',
            tool='write_file',
            args={'path': '.github/workflows/ci.yml', 'content': CI_WORKFLOW},
        ),
        Step(text='Added `.github/workflows/ci.yml`. Push a branch to see it run.\n'),
    ),
    'how a forecast': (
        Step(
            tool='run_code',
            args={
                'code': "tests = await shell(command='python3 -m unittest -q')\n"
                "files = await list_files(path='weather')\n"
                "cli = await read_file(path='weather/cli.py')\n"
                "source = await read_file(path='weather/forecast.py')\n"
                "calls = await grep(pattern='fetch_forecast', path='.')\n"
                "test_source = await read_file(path='tests/test_cli.py')\n"
                'print(calls, tests)\n'
            },
        ),
        Step(
            text='A forecast takes three steps:\n\n'
            '1. `weather/cli.py` parses `forecast CITY --days N`.\n'
            '2. `fetch_forecast` in `weather/forecast.py` looks the city up in `SAMPLE` and '
            'returns `Day` objects.\n'
            '3. The CLI prints one row per day.\n\n'
            'Nothing goes over the network, so the tests run offline.\n'
        ),
    ),
    'tests for the parser': (
        Step(
            text='Here is a test for `--days`, in the style of `tests/test_cli.py`:\n\n'
            '```python\ndef test_days_limits_rows(self):\n'
            "    out = run('forecast', 'lisbon', '--days', '1')\n"
            '    self.assertEqual(len(out.splitlines()), 1)\n```\n'
        ),
    ),
}

DEFAULT = (Step(text='Done.\n'),)

NAMES: dict[str, tuple[str, str, list[str]]] = {
    'tests for the parser': ('Tests for the days option', 'A unittest for --days', ['tests']),
    '--units flag': ('Fahrenheit for the forecast CLI', 'Added --units and kept tests green', ['cli']),
    'set up ci': ('Set up GitHub Actions', 'Asked which CI, wrote the workflow', ['ci', 'github']),
    'how a forecast': ('How a forecast is fetched', 'Walked through cli.py and forecast.py', ['docs']),
    'retry with backoff': ('Retry flaky HTTP calls', 'Backoff for the orders client', ['http', 'retry']),
    'add an index': ('Speed up order search', 'Added a database index', ['sql', 'perf']),
    'readme install': ('Rewrite the README', 'Shorter install and usage sections', ['docs']),
}
"""Title cards for the session browser, by a phrase in the conversation."""

SUMMARY_PROMPT = 'You are a context summarization assistant'
"""How the compaction plugin's summary request starts."""


def _prompt(messages: Sequence[ModelMessage]) -> tuple[str, int]:
    """The latest user prompt, and how many responses the model has given to it so far."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest):
            texts = [
                part.content
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            ]
            if texts:
                responses = sum(isinstance(later, ModelResponse) for later in messages[index + 1 :])
                return texts[-1], responses
    return '', 0


def _match(text: str, table: dict[str, T]) -> T | None:
    lowered = text.lower()
    return next((value for key, value in table.items() if key in lowered), None)


def _step(messages: Sequence[ModelMessage], info: AgentInfo) -> Step:
    prompt, responses = _prompt(messages)
    if info.output_tools:
        # The background session namer asks for a title card.
        title, subtitle, tags = _match(prompt, NAMES) or ('Untitled session', '', [])
        return Step(tool=info.output_tools[0].name, args={'title': title, 'subtitle': subtitle, 'tags': tags})
    if prompt.startswith(SUMMARY_PROMPT):
        return Step(text='The user is adding features to a small weather CLI; tests use unittest.\n')
    scene = _match(prompt, SCENES) or DEFAULT
    return scene[min(responses, len(scene) - 1)]


def _chunks(text: str) -> list[str]:
    return [text[start : start + CHUNK] for start in range(0, len(text), CHUNK)]


async def stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
    """Stream the scene's next step: thinking, then text, then the tool call's arguments."""
    step = _step(messages, info)
    for chunk in _chunks(step.thinking):
        yield {0: DeltaThinkingPart(content=chunk)}
        await asyncio.sleep(PAUSE)
    for chunk in _chunks(step.text):
        yield chunk
        await asyncio.sleep(PAUSE)
    if step.tool:
        yield {1: DeltaToolCall(name=step.tool, json_args='', tool_call_id=f'call_{len(messages)}')}
        for chunk in _chunks(json.dumps(step.args)):
            yield {1: DeltaToolCall(json_args=chunk)}
            await asyncio.sleep(PAUSE)


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The same step unstreamed, for requests made with `agent.run`, such as session naming."""
    step = _step(messages, info)
    parts: list[ModelResponsePart] = []
    if step.thinking:
        parts.append(ThinkingPart(content=step.thinking))
    if step.text:
        parts.append(TextPart(content=step.text))
    if step.tool:
        parts.append(ToolCallPart(tool_name=step.tool, args=step.args))
    return ModelResponse(parts=parts)


def resolve(name: str) -> Model:
    """Build the scripted model; `name` is what follows `demo:`."""
    return FunctionModel(respond, stream_function=stream, model_name=name)


class Demo(Plugin):
    """Scripted `demo:tour` model for regenerating the README screenshots."""

    def get_model_providers(self) -> Sequence[ModelProvider]:
        """Offer `demo:tour`."""
        return (ModelProvider(prefix='demo', resolve=resolve, models=('tour',)),)
