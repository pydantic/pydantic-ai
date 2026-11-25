"""What the D1 suites share: the model that makes a case's calls, how a case is built, and how a journal is read.

`tests/harness/test_durable_tool_io.py` runs every case with `calls` in a DBOS workflow, and
`tests/harness/test_durable_temporal_replay.py` in a Temporal one; each checks what the run journaled against the
case's `durable_steps`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar
from unittest.mock import patch

import pytest

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, FunctionToolset, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentStreamEvent,
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.capability_creation import CapabilityStore
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.memory import InMemoryMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore

from ..append_only import ToolCall
from ..capability_cases import BROKEN_SIGNATURES, SCENARIO_RUNS, SCENARIO_SUPPORT, Engine, Scenario
from ._contracts import CASES
from ._counting_clients import AWS_CLI_COUNT_VARIABLE, REQUESTS

PROMPT = 'go'
"""The prompt a case's run starts with, which tells its requests apart from a nested run's."""

P = ParamSpec('P')
T = TypeVar('T')

UNPROMPTED_REQUEST = 'model.unprompted_request'
"""What `REQUESTS` counts a request without `PROMPT` under: a nested run's, or one after compaction dropped it."""


def one_call_per_response(made: tuple[ToolCall, ...]) -> FunctionModel:
    """A model that makes the calls in order, one per response, then answers; call `i` gets the ID `call_{i}`.

    It doesn't stream, so a run makes one model request per call and one to answer. A request whose first message
    isn't the run's `PROMPT` gets the answer, and is counted in `REQUESTS`: it is a capability's own run on the
    run's model, such as `BrowserUse`'s browser agent, whose requests recovery must not make again.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if not any(isinstance(part, UserPromptPart) and part.content == PROMPT for part in messages[0].parts):
            REQUESTS[UNPROMPTED_REQUEST] += 1
            return ModelResponse(parts=[TextPart('done')])
        if step >= len(made):
            return ModelResponse(parts=[TextPart('done')])
        name, args = made[step]
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f'call_{step}')])

    return FunctionModel(respond)


def installed(name: str) -> bool:
    """Whether the optional dependency the case of `name` needs, if any, is installed: the test skips it if not."""
    requires = CASES[name].requires
    return requires is None or importlib.util.find_spec(requires) is not None


def build_case(name: str, directory: Path) -> list[AbstractCapability[None]]:
    """What the calls of the case of `name` run against, built in `directory` with the credentials it checks."""
    case = CASES[name]
    assert case.calls is not None
    with patch.dict(os.environ, case.env):
        built = case.calls.build(directory)
    return [built] if isinstance(built, AbstractCapability) else list(built)


def journal_breaks(
    recorded: Sequence[str], *, prefix: str, model_request: str, calls: int, expected: tuple[str, ...], engine: str
) -> list[str]:
    """How the durable units a run recorded differ from what its case's `durable_steps` lists for `engine`.

    `recorded` holds the units' names, each starting with `prefix`. The run's own model requests are left out of
    the comparison, but there must be one per call and one to answer.
    """
    units = [unit.removeprefix(prefix) for unit in recorded]
    requests = units.count(model_request)
    journaled = tuple(unit for unit in units if unit != model_request)
    counted = (
        []
        if requests == calls + 1
        else [f'the run recorded {requests} `{model_request}` units, not one per call and one to answer']
    )
    drifted = (
        []
        if journaled == expected
        else [
            f'the calls record `{engine}={journaled!r}`, not {expected!r}. A recorded name is what a workflow from an '
            'earlier release replays by: change the case only with a compatibility note '
            '(agent_docs/capability-conformance.md, "Durable execution")'
        ]
    )
    return counted + drifted


def _counted(key: str, method: Callable[P, T]) -> Callable[P, T]:
    """`method`, counting each call in `REQUESTS` under `key`: an async method's call, when its request starts."""

    def counted(*args: P.args, **kwargs: P.kwargs) -> T:
        REQUESTS[key] += 1
        return method(*args, **kwargs)

    return counted


def counting_requests(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Callable[[], Counter[str]]:
    """Count every request the cases' stand-ins, the `aws` stand-in, and the stores make, from now on.

    Both D1 suites count with it, so a request means the same thing in each. `directory` holds the `aws` count.
    """
    REQUESTS.clear()
    aws_count = directory / 'aws_cli_count'
    monkeypatch.setenv(AWS_CLI_COUNT_VARIABLE, str(aws_count))

    monkeypatch.setattr(CapabilityStore, 'write', _counted('store.write', CapabilityStore.write))
    monkeypatch.setattr(CapabilityStore, 'list_all', _counted('store.list_all', CapabilityStore.list_all))
    monkeypatch.setattr(CapabilityStore, 'disable', _counted('store.disable', CapabilityStore.disable))
    # A memory write first looks up its idempotency key; the write is what recovery must not repeat.
    monkeypatch.setattr(InMemoryMemoryStore, 'write', _counted('memory.write', InMemoryMemoryStore.write))
    # The plan tools read the store many times; a write is what recovery must not repeat.
    monkeypatch.setattr(SqlitePlanStore, 'set_items', _counted('plan.set_items', SqlitePlanStore.set_items))
    monkeypatch.setattr(SqlitePlanStore, 'add_item', _counted('plan.add_item', SqlitePlanStore.add_item))
    monkeypatch.setattr(
        SnapshotHistorySource, 'list_runs', _counted('history.list_runs', SnapshotHistorySource.list_runs)
    )

    def snapshot() -> Counter[str]:
        counts = Counter(REQUESTS)
        if aws_count.exists():
            counts['aws_cli'] = len(aws_count.read_text().splitlines())
        return counts

    return snapshot


def streaming_one_call_per_response(made: tuple[ToolCall, ...]) -> FunctionModel:
    """`one_call_per_response` that also streams, which a run with an event listener needs."""
    respond = one_call_per_response(made).function
    assert respond is not None

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        response = respond(messages, info)
        assert isinstance(response, ModelResponse)
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {0: DeltaToolCall(part.tool_name, json.dumps(part.args), tool_call_id=part.tool_call_id)}

    return FunctionModel(respond, stream_function=stream)


_EVENTS: dict[str, list[str]] = {}
"""What each scenario agent's `ScenarioRecorder` has seen in its current run, by agent name."""


@dataclass
class ScenarioRecorder(AbstractCapability[None]):
    """Records the custom and capability events and enqueued messages a scenario's run observes, in workflow code.

    It keeps them by agent name rather than on itself, as an engine may run a copy of it.
    """

    agent_name: str

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        seen = _EVENTS.setdefault(self.agent_name, [])
        if isinstance(event, CustomEvent | CapabilityEvent):
            seen.append(f'event:{getattr(event, "text", type(event).__name__)}')
        elif isinstance(event, EnqueuedMessagesEvent):
            seen.extend(f'enqueued:{content}' for content in _prompts(event.messages))


def _prompts(messages: Sequence[ModelMessage]) -> list[str]:
    """The user prompts in `messages` other than the run's own `PROMPT`: the enqueued ones."""
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and part.content != PROMPT
    ]


def scenario_agent(scenario: Scenario, durability: AbstractCapability[None]) -> Agent[None, str]:
    """The agent that runs `scenario` under `durability`, with a `ScenarioRecorder` as its first capability."""
    built = list(SCENARIO_RUNS[scenario].build())
    name = f'scenario_{scenario.partition(".")[2]}'
    return Agent(
        streaming_one_call_per_response((('work', {}),)),
        name=name,
        deps_type=type(None),
        toolsets=[part for part in built if isinstance(part, FunctionToolset)],
        capabilities=[
            ScenarioRecorder(name),
            *(part for part in built if isinstance(part, AbstractCapability)),
            durability,
        ],
    )


OBSERVED: dict[str, list[str]] = {}
"""What the last run of each scenario agent observed, by agent name: a replay's run leaves its own here."""


async def observe_run(agent: Agent[None, str]) -> list[str]:
    """Run a `scenario_agent` from workflow code, and return what it observed: events, then enqueued messages."""
    assert agent.name is not None
    _EVENTS[agent.name] = []
    result = await agent.run(PROMPT)
    observed = [*_EVENTS[agent.name], *(f'message:{prompt}' for prompt in _prompts(result.all_messages()))]
    OBSERVED[agent.name] = observed
    return observed


def scenario_breaks(scenario: Scenario, first: Sequence[str], recovered: Sequence[str]) -> list[str]:
    """How the first run and its recovery or replay break `scenario`: each item once, and none lost or added."""
    expected = list(SCENARIO_RUNS[scenario].observed)
    breaks = [] if list(first) == expected else [f'the run observed {list(first)}, not {expected}']
    lost = sorted((Counter(first) - Counter(recovered)).elements())
    added = sorted((Counter(recovered) - Counter(first)).elements())
    breaks += [f'recovery lost {lost}'] if lost else []
    breaks += [f'recovery added {added}'] if added else []
    return breaks


def assert_scenario_cell(engine: Engine, scenario: Scenario, breaks: Sequence[str]) -> None:
    """`breaks` must be what the scenario's `SCENARIO_SUPPORT` cell for `engine` says: none, a refusal or the bug."""
    cell = SCENARIO_SUPPORT[scenario][engine]
    if cell.level in ('supported', 'limited'):
        assert not breaks, '; '.join(breaks)
    elif cell.level == 'refused':
        assert cell.raises is not None
        assert any(re.search(cell.raises, line) for line in breaks), (
            f'{scenario} no longer raises {cell.raises!r} under {engine}: update its cell ({breaks})'
        )
    else:
        assert cell.level == 'broken' and cell.issue is not None, f'{scenario} is {cell.level} under {engine}'
        assert any(re.search(BROKEN_SIGNATURES[cell.issue], line) for line in breaks), (
            f'{scenario} no longer shows #{cell.issue} under {engine}: update its cell ({breaks})'
        )
