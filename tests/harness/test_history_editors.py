"""Every capability that edits the history it is sent keeps that history sound.

A capability that overrides `before_model_request` or `after_tool_execute` declares in its case's `edits_history`
whether it edits history (`tests/capability_cases.py`, `tests/harness/capability_cases.py`). Each configuration an
editor declares runs over the histories in `HISTORIES`, continued by a run whose model makes the configuration's
calls, and must keep three invariants:

1. The request being answered reaches the model: the run's prompt, then the replies to each call the model made.
2. A summary keeps what each message it replaced said, for a configuration that `summarizes`.
3. A capability load or a tool search survives the edit, or is recorded again, so the tool it revealed still runs.

A configuration that never edits any of the histories checks nothing, so it fails too. A capability that declares
`edits_history='no'` must keep every part of the history it continues.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _pytest.mark import ParameterSet

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, ToolSearch, Toolset
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from .. import capability_cases as core
from ..append_only import build_capabilities, discover_capabilities, look_up, skip_unless_installed
from ..capability_cases import CORE_CAPABILITY_PACKAGES, CapabilityCase, EditorConfig, Unbuildable
from ..conftest import legacy_retry_prompt_part
from ..function_model_prefix import COMPACTION_REWRITES_HISTORY, REQUEST_SCOPED_PROMPT_DROPPED
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES, MARKER
from .test_history_append_only import ALLOWED, CAPABILITIES as HARNESS_CAPABILITIES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}

HISTORY_HOOKS = ('before_model_request', 'after_tool_execute')
"""The hooks a capability edits history through: what is sent for a request, and a tool's result before it's kept."""

PROMPT = 'And what else? [m99]'

_FILLER = ' The forecast holds through the week, with a light westerly wind and clear skies by the evening.' * 3
"""Makes each answer long, so a summary of a few of them is shorter than they are and a compaction commits it."""


def _turns() -> list[ModelMessage]:
    """Four questions, each answered at length."""
    history: list[ModelMessage] = []
    for turn in range(4):
        history.append(ModelRequest(parts=[UserPromptPart(f'Question {turn} [m{2 * turn}]')]))
        history.append(ModelResponse(parts=[TextPart(f'Answer {turn} [m{2 * turn + 1}].{_FILLER}')]))
    return history


def _tool_turns() -> list[ModelMessage]:
    """Two questions, each answered with a lookup."""
    history: list[ModelMessage] = []
    for turn in range(2):
        call = ToolCallPart('look_up', {'topic': f'topic {turn}'}, f'h{turn}')
        history += [
            ModelRequest(parts=[UserPromptPart(f'Question {turn} [m{3 * turn}]')]),
            ModelResponse(parts=[call]),
            ModelRequest(
                parts=[ToolReturnPart('look_up', f'All quiet [m{3 * turn + 1}].{_FILLER}', call.tool_call_id)]
            ),
            ModelResponse(parts=[TextPart(f'Answer {turn} [m{3 * turn + 2}].')]),
        ]
    return history


def _legacy_retry() -> list[ModelMessage]:
    """A call the model retried, answered by a `RetryPromptPart`, as a history stored before v2 holds it (#10759)."""
    return [
        ModelRequest(parts=[UserPromptPart('Look up the news [m0]')]),
        ModelResponse(parts=[ToolCallPart('look_up', {'topic': ''}, 'a')]),
        ModelRequest(
            parts=[legacy_retry_prompt_part('The topic is empty [m1].', tool_name='look_up', tool_call_id='a')]
        ),
        ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'news'}, 'b')]),
        ModelRequest(parts=[ToolReturnPart('look_up', f'All quiet [m2].{_FILLER}', 'b')]),
        ModelResponse(parts=[TextPart('Nothing new [m3].')]),
    ]


def _file_reads() -> list[ModelMessage]:
    """The same file read twice, the first read superseded by the second."""
    history: list[ModelMessage] = []
    for turn in range(2):
        call = ToolCallPart('read_file', {'path': 'notes.md'}, f'f{turn}')
        history += [
            ModelRequest(parts=[UserPromptPart(f'Read the notes [m{3 * turn}]')]),
            ModelResponse(parts=[call]),
            ModelRequest(parts=[ToolReturnPart('read_file', f'Sunny [m{3 * turn + 1}].{_FILLER}', call.tool_call_id)]),
            ModelResponse(parts=[TextPart(f'They say sunny [m{3 * turn + 2}].')]),
        ]
    return history


HISTORIES: dict[str, Callable[[], list[ModelMessage]]] = {
    'turns': _turns,
    'tool turns': _tool_turns,
    'legacy retry': _legacy_retry,
    'file reads': _file_reads,
}
"""The histories each configuration continues, each with a marker `[m<n>]` in every message."""


def _overrides(cls: type[AbstractCapability[Any]], hooks: Sequence[str]) -> bool:
    return any(getattr(cls, hook) is not getattr(AbstractCapability, hook) for hook in hooks)


def _is_composite(cls: type[AbstractCapability[Any]]) -> bool:
    """A capability made of others, which are cases of their own."""
    return issubclass(cls, CombinedCapability | WrapperCapability)


EDITORS: dict[str, dict[str, EditorConfig]] = {
    name: dict(case.edits_history) for name, case in CASES.items() if isinstance(case.edits_history, dict)
}
CONFIGS: list[ParameterSet] = [
    pytest.param(
        name, config, marks=pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY), id=f'{name}:{label}'
    )
    for name, configs in sorted(EDITORS.items())
    for label, config in configs.items()
]


def test_every_capability_with_a_history_hook_declares_whether_it_edits_history() -> None:
    problems: list[str] = []
    for name, ref in sorted(PUBLIC.items()):
        try:
            cls = ref.load()
        except ImportError:  # pragma: lax no cover
            continue
        if _is_composite(cls) or not _overrides(cls, HISTORY_HOOKS):
            continue
        if CASES[name].edits_history is None:
            problems.append(f'{name} overrides {" or ".join(HISTORY_HOOKS)}: declare `edits_history` in its case')
    assert not problems, '\n'.join(problems)


def test_every_documented_history_rewriter_declares_its_edits() -> None:
    """A capability the append-only suite allows to rewrite history is an editor these invariants must hold."""
    rewriters = {name for name, allowed in ALLOWED.items() if not allowed.tail and name in HARNESS_CAPABILITIES}
    undeclared = sorted(name for name in rewriters if name not in EDITORS)
    assert not undeclared, f'listed in `ALLOWED` as rewriting history, but declare no `EditorConfig`: {undeclared}'


@dataclass
class _Run:
    """What one run sent the model, and what its model answered."""

    received: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])
    responses: list[ModelResponse] = field(default_factory=list[ModelResponse])

    def respond(
        self, calls: Sequence[tuple[str, dict[str, Any]]]
    ) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
        script = iter(calls)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.received.append(deepcopy(messages))
            call = next(script, None)
            if call is None:
                response = ModelResponse(parts=[TextPart('Done.')])
            else:
                name, args = call
                response = ModelResponse(parts=[ToolCallPart(name, args, f'call_{len(self.responses)}')])
            self.responses.append(response)
            return response

        return respond


def _capabilities(config: EditorConfig) -> list[AbstractCapability[None]]:
    built = config.build()
    return [built] if isinstance(built, AbstractCapability) else list(built)


async def _continue(config: EditorConfig, history: list[ModelMessage]) -> _Run:
    run = _Run()
    agent = Agent(
        FunctionModel(run.respond(config.calls)),
        deps_type=type(None),
        instructions='You are a helpful assistant.',
        toolsets=[FunctionToolset([look_up], id='look_up')],
        capabilities=_capabilities(config),
    )
    await agent.run(PROMPT, message_history=history)
    return run


def _flat(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        f'{type(part).__name__}:{getattr(part, "content", getattr(part, "args", ""))}'
        for m in messages
        for part in m.parts
    ]


def _edited(history: list[ModelMessage], run: _Run) -> bool:
    """Whether some request the model received isn't the one before it plus what came after: a part was edited."""
    expected = _flat(history)
    for received, response in zip(run.received, [*run.responses]):
        sent = _flat(received)
        if sent[: len(expected)] != expected:
            return True
        expected = sent + _flat([response])
    return False


def _answered(received: list[ModelMessage]) -> set[str]:
    request = next(message for message in reversed(received) if isinstance(message, ModelRequest))
    return {
        part.tool_call_id
        for part in request.parts
        if isinstance(part, ToolReturnPart | RetryPromptPart)  # pyright: ignore[reportDeprecated]
    }


def _newest_request_breaks(run: _Run) -> list[str]:
    """Invariant 1: each request the model received holds the run's prompt, then the replies to its last calls."""
    breaks: list[str] = []
    first = next(message for message in reversed(run.received[0]) if isinstance(message, ModelRequest))
    if not any(isinstance(part, UserPromptPart) and part.content == PROMPT for part in first.parts):
        breaks.append(f'the first request does not end with the prompt: {first.parts}')
    for index, (received, previous) in enumerate(zip(run.received[1:], run.responses), start=1):
        called = {part.tool_call_id for part in previous.parts if isinstance(part, ToolCallPart)}
        if missing := called - _answered(received):
            breaks.append(f'request {index} leaves out the replies to {sorted(missing)}')
    return breaks


def _markers(messages: Sequence[ModelMessage]) -> set[str]:
    return set(MARKER.findall(str(messages)))


def _summary_breaks(history: list[ModelMessage], run: _Run) -> list[str]:
    """Invariant 2: what each message said reaches the model, kept or in the summary that replaced it."""
    said = _markers(history) | {MARKER.findall(PROMPT)[0]}
    return [
        f'request {index} lost {sorted(lost)}'
        for index, received in enumerate(run.received)
        if (lost := said - _markers(received))
    ]


NON_EDITORS = sorted(
    name
    for name, case in CASES.items()
    if case.edits_history == 'no' and name in PUBLIC and not isinstance(case.build, Unbuildable)
)


# A capability that only adds to what is sent may add a request-scoped prompt, which the next request leaves out.
@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
@pytest.mark.parametrize('name', NON_EDITORS)
async def test_a_capability_declared_not_to_edit_keeps_the_history(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> None:
    """A capability whose case says `edits_history='no'` keeps every part of the history it continues."""
    case = CASES[name]
    skip_unless_installed(name, case.requires)
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)
    capabilities = build_capabilities(name, PUBLIC[name].load, case.fixture, tmp_path)
    run = _Run()
    respond = run.respond((('look_up', {'topic': 'sports'}),))

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        response = respond(messages, info)
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                    )
                }
            elif isinstance(part, TextPart):
                yield part.content

    # Named, as a durable engine names its units after the agent; streamed, for a capability that streams its run.
    agent = Agent(
        FunctionModel(respond, stream_function=stream),
        name='history-editors',
        deps_type=type(None),
        toolsets=[FunctionToolset([look_up], id='look_up')],
        capabilities=capabilities,
    )
    kept = (await agent.run(PROMPT, message_history=_turns())).all_messages()
    assert _flat(kept)[: len(_flat(_turns()))] == _flat(_turns()), f'{name} declares no edits, but edited the history'


@pytest.mark.parametrize(('name', 'config'), CONFIGS)
async def test_editing_keeps_the_history_sound(
    name: str, config: EditorConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> None:
    skip_unless_installed(name, CASES[name].requires)
    edited = False
    breaks: list[str] = []
    for label, build in HISTORIES.items():
        history = build()
        # Built again rather than copied: copying the legacy `RetryPromptPart` warns that it is deprecated.
        run = await _continue(config, build())
        edited = edited or _edited(history, run)
        found = _newest_request_breaks(run)
        if config.summarizes:
            found += _summary_breaks(history, run)
        breaks += [f'{label}: {item}' for item in found]
    assert not breaks, '\n'.join(breaks)
    assert edited, (
        f'{name} edits none of the histories, so the invariants check nothing: give it a configuration that does'
    )


def _forecast(city: str) -> str:
    """Get the weather forecast for a city."""
    return f'Sunny in {city}.'


def _forecasting() -> list[AbstractCapability[None]]:
    return [
        Capability(
            id='forecasting',
            description='Weather forecasts.',
            tools=[Tool(_forecast, name='forecast')],
            defer_loading=True,
        )
    ]


def _searchable_forecast() -> list[AbstractCapability[None]]:
    return [
        ToolSearch(strategy='keywords'),
        Toolset(FunctionToolset([Tool(_forecast, name='forecast', defer_loading=True)])),
    ]


DISCOVERIES: dict[str, tuple[Callable[[], list[AbstractCapability[None]]], tuple[str, dict[str, Any]]]] = {
    'load_capability': (_forecasting, ('load_capability', {'id': 'forecasting'})),
    'search_tools': (_searchable_forecast, ('search_tools', {'queries': ['forecast']})),
}
"""The ways a run reveals a tool, each recorded in history as a typed part an editor must keep or record again."""


@pytest.mark.parametrize('discovery', DISCOVERIES)
@pytest.mark.parametrize(('name', 'config'), CONFIGS)
async def test_a_revealed_tool_stays_callable_after_an_edit(
    name: str, config: EditorConfig, discovery: str, allow_model_requests: None
) -> None:
    """Invariant 3: a tool revealed in the first run runs in both, whatever the editor did in between (#11446, #11773)."""
    skip_unless_installed(name, CASES[name].requires)
    extra, reveal = DISCOVERIES[discovery]
    replies: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = next(message for message in reversed(messages) if isinstance(message, ModelRequest))
        replies.extend(
            f'{part.outcome}: {part.content}' if isinstance(part, ToolReturnPart) else f'retry: {part.content}'
            for part in request.parts
            if isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_name == 'forecast'  # pyright: ignore[reportDeprecated]
        )
        return next(script)

    def call(name: str, args: dict[str, Any], index: int) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(name, args, f'call_{index}')])

    script = iter(
        [
            call(*reveal, 0),
            call('forecast', {'city': 'Paris'}, 1),
            ModelResponse(parts=[TextPart('Sunny in Paris.')]),
            call('look_up', {'topic': 'sports'}, 2),
            call('forecast', {'city': 'Brest'}, 3),
            ModelResponse(parts=[TextPart('Sunny in Brest.')]),
        ]
    )
    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        toolsets=[FunctionToolset([look_up], id='look_up')],
        capabilities=[*_capabilities(config), *extra()],
    )
    first = await agent.run('What is the forecast for Paris? [m0]', message_history=_turns())
    await agent.run('And for Brest?', message_history=first.all_messages())
    assert replies == ['success: Sunny in Paris.', 'success: Sunny in Brest.']
