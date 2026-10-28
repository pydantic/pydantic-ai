"""Every core capability keeps a conversation's requests append-only.

See `tests/append_only.py` for the scenario and the three invariants it checks. The cases are discovered
from the package: every public capability class in `pydantic_ai.capabilities` and `pydantic_ai.models`
runs, constructed with its defaults or with a fixture from `FIXTURES`, unless `EXEMPT` says why it can't
be, so a new capability is checked without anyone adding it; one that can't be built with its defaults
fails until it gets a fixture or an exemption. Agent features that render state into the prompt are
cases too (`AGENT_FEATURES`). Durable execution engines in `pydantic_ai.durable_exec` are left to their
own suites, which check that a durable run sends the same requests as a plain one.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, ModelRetry, RunContext, TextOutput
from pydantic_ai.capabilities import (
    MCP,
    AbstractCapability,
    Capability,
    CombinedCapability,
    DynamicCapability,
    FileUnderstanding,
    HandleDeferredToolCalls,
    LocalWorkspace,
    NativeTool,
    PrefixTools,
    PrepareOutputTools,
    PrepareTools,
    ProcessEventStream,
    ProcessHistory,
    ResolveModelId,
    SelectModel,
    Thinking,
    Toolset,
    UseThreadExecutor,
    WrapModelRequestHandler,
    WrapperCapability,
    XSearch,
)
from pydantic_ai.messages import (
    CapabilityInstructionSource,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ._inline_snapshot import snapshot
from .append_only import (
    CLOCK,
    PROVIDERS,
    Allowed,
    Change,
    Clock,
    ConversationResult,
    Exempt,
    Fixture,
    Known,
    Provider,
    Scenario,
    Turn,
    build_capabilities,
    check_conversation,
    discover_capabilities,
    drain_events,
    inject_clock,
    provider_imports_successful,
    run_conversation,
    ui_imports_successful,
)
from .conftest import try_import

with try_import() as mcp_imports_successful:
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset

# The module is skipped without the provider extras, so this needs no flag of its own.
with try_import():
    from pydantic_ai.models.openai import OpenAIResponsesModelSettings

pytestmark = pytest.mark.skipif(not provider_imports_successful(), reason='anthropic or openai not installed')


def _mcp_server() -> FastMCP[None]:
    server: FastMCP[None] = FastMCP('extra', instructions='Extra server instructions.')

    @server.tool()
    def extra_tool() -> str:
        return 'extra'  # pragma: no cover

    return server


def _no_op_tool() -> str:
    """Do nothing."""
    return 'ok'  # pragma: no cover


def _keep_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages


def _keep_model(ctx: ModelSelectionContext[Any]) -> Model:
    assert ctx.model is not None
    return ctx.model


FIXTURES: dict[str, Fixture] = {
    'CombinedCapability': lambda _: CombinedCapability([Thinking()]),
    'DynamicCapability': lambda _: DynamicCapability(lambda ctx: Thinking()),
    'FileUnderstanding': lambda _: FileUnderstanding(fallback_subagent_model=TestModel()),
    'HandleDeferredToolCalls': lambda _: HandleDeferredToolCalls(lambda ctx, requests: requests.build_results()),
    'LocalWorkspace': lambda tmp_path: LocalWorkspace(tmp_path),
    'MCP': lambda _: MCP(local=MCPToolset(_mcp_server())),
    'NativeTool': lambda _: NativeTool(CodeExecutionTool()),
    'PrefixTools': lambda _: PrefixTools(Toolset(FunctionToolset([_no_op_tool])), prefix='extra'),
    'PrepareOutputTools': lambda _: PrepareOutputTools(lambda ctx, tools: tools),
    'PrepareTools': lambda _: PrepareTools(lambda ctx, tools: tools),
    'ProcessEventStream': lambda _: ProcessEventStream(drain_events),
    'ProcessHistory': lambda _: ProcessHistory(_keep_history),
    'ResolveModelId': lambda _: ResolveModelId(lambda ctx, model_id: None),
    'SelectModel': lambda _: SelectModel(_keep_model),
    'Toolset': lambda _: Toolset(FunctionToolset([_no_op_tool])),
    'UseThreadExecutor': lambda _: UseThreadExecutor(ThreadPoolExecutor(max_workers=1)),
    'WrapperCapability': lambda _: WrapperCapability(Thinking()),
    'XSearch': lambda _: XSearch(fallback_subagent_model=TestModel()),
}
"""Capabilities that need configuration, keyed by class name, built for each test in its `tmp_path`."""

EXEMPT: dict[str, Exempt] = {
    'BatchModelRequests': Exempt(
        "Sends each request through the provider's batch API, which the scenario doesn't serve. A batch item's body is "
        "built by the same code as the model's own request, so it's the unbatched request's (see the providers' "
        '`batch_encode` tests).'
    ),
    'NativeOrLocalTool': Exempt(
        'Base class that needs a subclass; `WebSearch`, `WebFetch`, `XSearch` and `MCP` are its cases.'
    ),
}

_NO_LOCAL_FALLBACK = 'The provider has no native {}, and the default has no local fallback.'

UNSUPPORTED: dict[tuple[str, Provider], str] = {
    ('ImageGeneration', 'anthropic'): _NO_LOCAL_FALLBACK.format('image generation'),
    ('ImageGeneration', 'openai-chat'): _NO_LOCAL_FALLBACK.format('image generation'),
    ('NativeTool', 'openai-chat'): 'Chat Completions has no native code execution.',
    ('WebFetch', 'openai-responses'): _NO_LOCAL_FALLBACK.format('web fetch'),
    ('WebFetch', 'openai-chat'): _NO_LOCAL_FALLBACK.format('web fetch'),
    ('WebSearch', 'openai-chat'): _NO_LOCAL_FALLBACK.format('web search'),
}
"""(capability, provider) pairs where the capability refuses the provider, so there is no conversation to check."""

KNOWN: dict[str, tuple[Known, ...]] = {
    'agent:dynamic_system_prompt': (
        Known('prefix', 'https://github.com/pydantic/pydantic-ai/issues/10521', 'system', 'system-prompt'),
    ),
}
"""Expected failures, each linked to the issue or PR that fixes it. Remove an entry when its fix lands."""

ALLOWED: dict[str, Allowed] = {
    'agent:effort_change': Allowed(
        'docs/capabilities/thinking.md: an API without a mid-conversation effort channel gets each change as the request-level '
        'effort, which is part of the cache key',
        (('settings', 'setting:reasoning_effort'),),
        max_moves=2,
        providers=('openai-chat',),
    ),
    'agent:dynamic_instructions': Allowed(
        'docs/agent.md: dynamic instructions are re-rendered for every request, after the static ones and outside '
        'the cached prefix; append mode is #8188',
        # The `@agent.instructions` part, which has no id of its own, after the agent's static instructions.
        (('system', 'instruction:anonymous@1'),),
        max_moves=4,
    ),
    **{
        f'retry:{name}+request-scoped': Allowed(
            'A request-scoped system prompt (#9711) is sent for one request, as the tail, and left out of the next '
            'one, retried steps included; Anthropic clears it natively',
            (('messages', 'system-prompt'),),
            tail=True,
            records=('request-scoped system prompt',),
            providers=('openai-responses', 'openai-chat'),
        )
        for name in ('tool-retry', 'output-validation-retry')
    },
    'Capability(defer_loading=True)': Allowed(
        'docs/capabilities/on-demand.md: Chat Completions has no deferred-tool channel, so a load adds the '
        "capability's tools to `tools`, like #9405's mid-run newcomers",
        tuple(('tools', f'tool:{name}') for name in ('write_note', 'read_note', 'delete_note')),
        providers=('openai-chat',),
    ),
    # Since #10938, a `no_output` retry leaves out the request-scoped prompt no user message would clear, so
    # Anthropic drops it from the tail like the OpenAI APIs.
    'retry:no-output-retry+request-scoped': Allowed(
        'A request-scoped system prompt (#9711) is sent for one request, as the tail, and left out of the next '
        'one, retried steps included',
        (('messages', 'system-prompt'),),
        tail=True,
        records=('request-scoped system prompt',),
    ),
}
"""Documented prefix movers, checked exactly."""


def _dynamic_system_prompt(agent: Agent[Any, Any], clock: Clock) -> None:
    @agent.system_prompt(dynamic=True)
    def now() -> str:
        return f'The time is {clock.now:%H:%M}.'


def _dynamic_instructions(agent: Agent[Any, Any], clock: Clock) -> None:
    @agent.instructions
    def now() -> str:
        return f'The time is {clock.now:%H:%M}.'


def _effort_by_step(ctx: RunContext[Any]) -> ModelSettings:
    """Raise the effort as the conversation goes on: at a tool-result step of the first run, and as the second starts."""
    responses = sum(isinstance(message, ModelResponse) for message in ctx.messages)
    return ModelSettings(thinking='low' if responses == 0 else 'medium' if responses < 3 else 'high')


AGENT_FEATURES: dict[str, Scenario] = {
    'agent:dynamic_system_prompt': Scenario(configure=_dynamic_system_prompt, check_ids=False),
    'agent:dynamic_instructions': Scenario(configure=_dynamic_instructions, check_ids=False),
    # The models whose APIs can change effort mid-conversation (#10385); Chat Completions can't.
    'agent:effort_change': Scenario(
        run_kwargs={'model_settings': _effort_by_step},
        model_names={'openai-responses': 'gpt-6.1-sol', 'openai-chat': 'gpt-6.1-sol'},
        serialize_history=True,
    ),
    # A UI endpoint that keeps the history on the server (#11372): the client sends its whole conversation back,
    # and `manage_message_history='server'` must still only ever append to the requests the stored history built.
    'ui:server_held_history_vercel': Scenario(ui_adapter='vercel', serialize_history=True),
    'ui:server_held_history_agui': Scenario(ui_adapter='agui', serialize_history=True),
}
"""Agent features, not capabilities, that render state into the prompt, run through the same checks."""


def write_note(content: str) -> str:
    """Write a note."""
    return 'Saved.'


def read_note() -> str:
    """Read the notes."""
    return 'No notes.'  # pragma: no cover


def delete_note() -> str:
    """Delete the notes."""
    return 'Deleted.'  # pragma: no cover


def look_up_archive(topic: str) -> str:
    """Look a topic up in the archive."""
    return 'Nothing archived.'  # pragma: no cover


VARIANTS: dict[str, tuple[Fixture, Scenario]] = {
    # Two deferred capabilities, the first loaded, with tools declared out of name order: the loaded tools
    # must keep the order their deferred entries were sent in, ahead of the other capability's.
    'Capability(defer_loading=True)': (
        lambda _: [
            Capability(
                id='notes', description='Notes.', tools=[write_note, read_note, delete_note], defer_loading=True
            ),
            Capability(id='archive', description='The archive.', tools=[look_up_archive], defer_loading=True),
        ],
        Scenario(state_calls=(('load_capability', {'id': 'notes'}), ('write_note', {'content': 'Sports too.'}))),
    ),
}
"""Capabilities in a configuration their defaults don't reach, with a scenario that exercises it."""


def _flaky_tool(agent: Agent[Any, Any], clock: Clock) -> None:
    attempts = 0

    @agent.tool_plain
    def check_forecast(city: str) -> str:
        """Check the forecast for a city."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry('The forecast service is busy, try again.')
        return f'Sunny in {city}.'


def _count(text: str) -> int:
    """Read the answer as a count; anything else fails validation, so the model is asked again."""
    return TypeAdapter(int).validate_python(text)


@dataclass
class _RequestScopedNudge(AbstractCapability[Any]):
    """Add a request-scoped system prompt (#9711) to every request, the way `WarnNearLimits` does."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        ctx.enqueue(SystemPromptPart('Keep it short.', scope='request'))
        return request_context


_LOOK_UP = ('look_up', {'topic': 'weather'})

RETRY_SCENARIOS: dict[str, Scenario] = {
    # A tool raises `ModelRetry`, so its result is a `ToolReturnPart(outcome='retried')` and the call is made again.
    'tool-retry': Scenario(
        configure=_flaky_tool,
        turns=(
            Turn(calls=(_LOOK_UP,)),
            Turn(calls=(('check_forecast', {'city': 'Paris'}),)),
            Turn(calls=(('check_forecast', {'city': 'Paris'}),)),
            Turn(text='Sunny in Paris.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='Here is more.'),
        ),
        serialize_history=True,
    ),
    # Text output fails validation, so `RetryFeedbackPart(cause='validation_error')` asks for it again.
    'output-validation-retry': Scenario(
        output_type=TextOutput(_count),
        turns=(
            Turn(calls=(_LOOK_UP,)),
            Turn(text='several'),
            Turn(text='3'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='a few'),
            Turn(text='4'),
        ),
        serialize_history=True,
        replay_text='5',
    ),
    # A response with nothing usable as output gets `RetryFeedbackPart(cause='no_output')`.
    'no-output-retry': Scenario(
        turns=(
            Turn(calls=(_LOOK_UP,)),
            Turn(text=''),
            Turn(text='Here is the weather.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text=''),
            Turn(text='Here is more.'),
        ),
        serialize_history=True,
    ),
}
"""Conversations that retry, continued from stored history. Each also runs with a request-scoped system prompt
on every request, which a retried step must still leave out of the next request."""


def _timekeeping() -> list[AbstractCapability[Any]]:
    capability = Capability[Any](
        id='timekeeping', defer_loading=True, description='Tell the time.', instructions='Give times in UTC.'
    )

    @capability.instructions
    def now() -> str:
        return f'The time is {CLOCK.now:%H:%M}.'

    return [capability]


LOADED_CAPABILITY_CASES: dict[str, tuple[Callable[[], list[AbstractCapability[Any]]], Scenario]] = {
    # The clock moves on every request, so every request after the load records a change to its instructions.
    'deferred:dynamic_instructions': (
        _timekeeping,
        Scenario(
            turns=(
                Turn(calls=(('load_capability', {'id': 'timekeeping'}),)),
                Turn(calls=(_LOOK_UP,)),
                Turn(text='Here is the weather.'),
                Turn(calls=(('look_up', {'topic': 'sports'}),)),
                Turn(text='Here is more.'),
            ),
            serialize_history=True,
        ),
    ),
}
"""Deferred capabilities the model loads mid-conversation, whose instructions then change."""

RETRY_CASES: dict[str, tuple[Scenario, bool]] = {
    **{f'retry:{name}': (scenario, False) for name, scenario in RETRY_SCENARIOS.items()},
    **{f'retry:{name}+request-scoped': (scenario, True) for name, scenario in RETRY_SCENARIOS.items()},
}

CAPABILITIES = discover_capabilities(('pydantic_ai.capabilities', 'pydantic_ai.models'))


def test_every_capability_has_a_case():
    """Every table entry names a capability that still exists, so renames and removals don't leave dead rows."""
    tables = FIXTURES.keys() | EXEMPT.keys() | KNOWN.keys() | ALLOWED.keys() | {name for name, _ in UNSUPPORTED}
    stale = (
        tables
        - CAPABILITIES.keys()
        - AGENT_FEATURES.keys()
        - RETRY_CASES.keys()
        - VARIANTS.keys()
        - LOADED_CAPABILITY_CASES.keys()
    )
    assert not stale, f'entries for capabilities that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize(
    'name',
    sorted(CAPABILITIES.keys() - EXEMPT.keys())
    + sorted(AGENT_FEATURES)
    + sorted(RETRY_CASES)
    + sorted(VARIANTS)
    + sorted(LOADED_CAPABILITY_CASES),
)
async def test_conversation_only_appends(
    name: str, provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    if reason := UNSUPPORTED.get((name, provider)):
        pytest.skip(reason)
    if name == 'MCP' and not mcp_imports_successful():  # pragma: lax no cover
        pytest.skip('mcp not installed')
    if name.startswith('ui:') and not ui_imports_successful():  # pragma: lax no cover
        pytest.skip('ag-ui-protocol not installed')
    if name in AGENT_FEATURES:
        capabilities, scenario = [], AGENT_FEATURES[name]
    elif name in LOADED_CAPABILITY_CASES:
        build, scenario = LOADED_CAPABILITY_CASES[name]
        capabilities = build()
    elif name in RETRY_CASES:
        scenario, nudged = RETRY_CASES[name]
        capabilities = [_RequestScopedNudge()] if nudged else []
    elif name in VARIANTS:
        fixture, scenario = VARIANTS[name]
        capabilities = build_capabilities(name, lambda: AbstractCapability, fixture, tmp_path)
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name].load, FIXTURES.get(name), tmp_path)
        scenario = Scenario()
    clock = inject_clock(monkeypatch)
    result = await run_conversation(provider, capabilities, scenario, clock)
    check_conversation(name, result, known=KNOWN.get(name, ()), allowed=ALLOWED.get(name), check_ids=scenario.check_ids)


def _chat_tool(name: str) -> dict[str, Any]:
    return {'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object', 'properties': {}}}}


@dataclass
class _RedactLatestToolResult(AbstractCapability[Any]):
    """Rewrite the latest real tool result for one request only: a tail change that is not a framework record."""

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        last = request_context.messages[-1]
        assert isinstance(last, ModelRequest)
        parts = [
            replace(part, content='[redacted]') if isinstance(part, ToolReturnPart) else part for part in last.parts
        ]
        request_context.messages[-1] = replace(last, parts=parts)
        return await handler(request_context)


@dataclass
class _RepeatFirstPrompt(AbstractCapability[Any]):
    """Repeat the first, persisted user prompt at the end of each request: a dropped tail that is real history."""

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        first, last = request_context.messages[0], request_context.messages[-1]
        assert isinstance(first, ModelRequest) and isinstance(last, ModelRequest)
        request_context.messages[-1] = replace(last, parts=[*last.parts, first.parts[0]])
        return await handler(request_context)


def _clock_instructions(ctx: RunContext[Any]) -> str:
    return f'The time is {CLOCK.now:%H:%M}.'


def _random_instructions(ctx: RunContext[Any]) -> str:
    return f'Your session is {uuid4().hex}.'


def _stamp_first_prompt(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Rewrite the first user prompt on every request, in the same requests the clock instruction changes."""
    first = messages[0]
    assert isinstance(first, ModelRequest)
    parts = [
        replace(part, content=f'{part.content} (sent at {CLOCK.now:%H:%M})')
        if isinstance(part, UserPromptPart)
        else part
        for part in first.parts
    ]
    return [replace(first, parts=parts), *messages[1:]]


async def _clock_tool_descriptions(ctx: RunContext[Any], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    return [replace(tool, description=f'{tool.description} As of {CLOCK.now:%H:%M}.') for tool in tools]


@pytest.mark.parametrize(
    ('capability', 'known', 'allowed', 'outcome', 'report'),
    [
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            None,
            'fail',
            '(instruction:capability:clock)',
            id='prefix',
        ),
        pytest.param(Capability[Any](instructions='Answer briefly.'), (), None, 'fail', 'without an `id`', id='ids'),
        pytest.param(
            Capability[Any](id='random', instructions=_random_instructions),
            (Known('prefix', 'example', 'system', 'instruction:capability:random', count=4),),
            None,
            'fail',
            'built the same request twice from the same history with different bytes',
            id='determinism',
        ),
        pytest.param(
            CombinedCapability[Any](
                [Capability(id='clock', instructions=_clock_instructions), PrepareTools(_clock_tool_descriptions)]
            ),
            (Known('prefix', 'example', 'system', 'instruction:capability:clock', count=4),),
            None,
            'fail',
            'As of',
            id='known-does-not-hide-another',
        ),
        pytest.param(
            CombinedCapability[Any](
                [Capability(id='clock', instructions=_clock_instructions), PrepareTools(_clock_tool_descriptions)]
            ),
            (),
            Allowed('example', (('system', 'instruction:capability:clock'),), max_moves=4),
            'fail',
            'is not one of the changes its `ALLOWED` entry permits: request 1 changed tools',
            id='allowed-does-not-cover-another',
        ),
        pytest.param(
            CombinedCapability[Any](
                [Capability(id='clock', instructions=_clock_instructions), ProcessHistory(_stamp_first_prompt)]
            ),
            (Known('prefix', 'example', 'system', 'instruction:capability:clock', count=4),),
            None,
            'fail',
            '(user-prompt)',
            id='known-does-not-hide-another-in-the-same-request',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (Known('prefix', 'example', 'system', 'instruction:capability:clock'),),
            None,
            'fail',
            'request 2 changed system',
            id='known-explains-only-its-count',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (Known('prefix', 'example', 'system', 'instruction:capability:clock', count=4),),
            None,
            'xfail',
            'known: [prefix] example',
            id='known',
        ),
        pytest.param(
            Capability[Any](instructions='Answer briefly.', id='static'),
            (Known('prefix', 'example', 'system', 'instruction:capability:static'),),
            None,
            'fail',
            'no longer fails these checks as expected',
            id='stale-known',
        ),
        pytest.param(
            Capability[Any](instructions='Answer briefly.', id='static'),
            (),
            Allowed('example', (('system', 'instruction:capability:static'),)),
            'fail',
            'is listed in `ALLOWED` (example) but no longer moves the prefix',
            id='stale-allowed',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            Allowed('example', (('system', 'instruction:capability:clock'),)),
            'fail',
            'moved the prefix in 4 requests, more than the 1 its `ALLOWED` entry permits',
            id='too-many-moves',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            Allowed('example', (('messages', 'user-prompt'),), max_moves=4),
            'fail',
            'is not one of the changes its `ALLOWED` entry permits: request 1 changed system',
            id='standing-prefix',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            Allowed('example', (('system', 'instruction:capability:clock'),), tail=True),
            'fail',
            'may only replace the last message of each request',
            id='deep-tail',
        ),
        pytest.param(
            _RedactLatestToolResult(),
            (),
            Allowed(
                'example',
                (('messages', 'tool-return'),),
                tail=True,
                records=('request-only user prompt',),
            ),
            'fail',
            'request 2 rewrote the last message of request 1',
            id='tail-is-only-framework-records',
        ),
        pytest.param(
            _RepeatFirstPrompt(),
            (),
            Allowed(
                'example',
                (('messages', 'tool-return'), ('messages', 'user-prompt')),
                tail=True,
                records=('request-only user prompt',),
            ),
            'fail',
            'dropped UserPromptPart (persisted history) from the last message of request 0',
            id='tail-is-not-persisted-history',
        ),
    ],
)
async def test_checks_report_what_they_catch(
    capability: AbstractCapability[Any],
    known: tuple[Known, ...],
    allowed: Allowed | None,
    outcome: Literal['fail', 'xfail'],
    report: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_model_requests: None,
):
    """Each check fails on what it's meant to catch, and the expectation tables can't go stale.

    A guard that never fires passes forever, so this runs deliberately broken capabilities through
    `check_conversation` directly: a block that renders the clock (a moving prefix), one without an `id`,
    one that renders a random value (nondeterministic projection), and table entries that no longer apply.
    """
    clock = inject_clock(monkeypatch)
    result = await run_conversation('anthropic', [capability], Scenario(), clock)
    expected = pytest.fail.Exception if outcome == 'fail' else pytest.xfail.Exception
    with pytest.raises(expected, match=re.escape(report)):
        check_conversation('example', result, known=known, allowed=allowed)


def test_a_capability_without_defaults_needs_a_fixture(tmp_path: Path):
    """A new capability that can't be built with its defaults fails with what to add, rather than a `TypeError`."""
    with pytest.raises(pytest.fail.Exception, match='Add a `FIXTURES` entry'):
        build_capabilities('NativeTool', lambda: NativeTool, None, tmp_path)


@pytest.fixture
def capability_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, str]], str]:
    """Write a throwaway package from `{module: source}` and put it on the path, returning its name."""
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])

    def write(modules: dict[str, str]) -> str:
        package = tmp_path / 'probe_capabilities'
        package.mkdir()
        for name, source in modules.items():
            (package / f'{name}.py').write_text(source)
        importlib.invalidate_caches()
        return package.name

    return write


@pytest.mark.parametrize(
    ('modules', 'expected'),
    [
        pytest.param(
            {
                '__init__': 'from ._caps import Base, Leaf\n',
                '_caps': (
                    'from typing import Any, Callable\n'
                    'from pydantic_ai.capabilities import AbstractCapability\n\n'
                    'class Base(AbstractCapability[dict[str, Callable[[int], Any]]]):\n    pass\n\n'
                    'class Leaf(Base, metaclass=type):\n    pass\n\n'
                    'class _Private(AbstractCapability[dict[str, Any]]):\n    pass\n'
                ),
            },
            {'Base': 'probe_capabilities._caps', 'Leaf': 'probe_capabilities._caps'},
            id='nested-type-arguments',
        ),
        pytest.param(
            {
                '__init__': 'from ._base import CapabilityBase\nfrom ._caps import Aliased, Qualified\n',
                '_base': 'from pydantic_ai.capabilities import AbstractCapability as CapabilityBase\n',
                '_caps': (
                    'from pydantic_ai import capabilities as caps\n'
                    'from ._base import CapabilityBase\n\n'
                    'class Aliased(\n    CapabilityBase[None],\n):\n    pass\n\n'
                    'class Qualified(caps.WrapperCapability[None]):\n    pass\n'
                ),
            },
            {'Aliased': 'probe_capabilities._caps', 'Qualified': 'probe_capabilities._caps'},
            id='aliased-and-qualified-bases',
        ),
        pytest.param(
            {
                '__init__': 'from ._caps import Leaf, Tool\nfrom ._plain import Plain\n',
                # A class named like a capability base that isn't one, in another module.
                # And a base that names nothing the package imports.
                '_lookalike': 'class AbstractCapability:\n    pass\n\nclass Base:\n    pass\n\nclass Stray(missing.Base):\n    pass\n',
                '_plain': 'from ._lookalike import AbstractCapability, Base\n\nclass Plain(Base, AbstractCapability):\n    pass\n',
                '_caps': (
                    'from pydantic_ai.capabilities import AbstractCapability\n\n'
                    'class Base(AbstractCapability[None]):\n    pass\n\n'
                    'class Leaf(Base):\n    pass\n\n'
                    'class Tool:\n    pass\n'
                ),
            },
            {'Leaf': 'probe_capabilities._caps'},
            id='same-named-classes-in-other-modules',
        ),
    ],
)
def test_discovery_resolves_bases_to_their_definitions(
    capability_package: Callable[[dict[str, str]], str], modules: dict[str, str], expected: dict[str, str]
):
    """Bases are followed to the class they name, through imports and aliases, not by their spelling.

    Discovery reads source, so this runs it over throwaway packages rather than shipped ones: a missed class is
    a capability the guard silently never checks, and a lookalike is a case that tests nothing.
    """
    found = discover_capabilities((capability_package(modules),))
    assert {name: ref.module for name, ref in found.items()} == expected


def test_only_provider_cache_markers_are_ignored():
    """A breakpoint moving is ignored, but the same key inside tool arguments is data and still counts.

    A unit test of the comparator: no capability sends `cache_control` in its tool arguments today, so no
    conversation would show the difference.
    """

    def body(marker_on: int, argument: str) -> bytes:
        messages: list[dict[str, Any]] = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]},
            {
                'role': 'assistant',
                'content': [{'type': 'tool_use', 'id': 't1', 'name': 'save', 'input': {'cache_control': argument}}],
            },
        ]
        messages[marker_on]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
        return json.dumps({'model': 'claude-opus-5-5', 'messages': messages}).encode()

    def breaks(*bodies: bytes) -> list[Change]:
        return ConversationResult('anthropic', list(bodies), (b'', b''), []).changes()

    assert breaks(body(0, 'a'), body(1, 'a')) == []
    assert [b.section for b in breaks(body(1, 'a'), body(1, 'b'))] == ['messages']


def test_an_unmarked_text_array_is_not_its_string():
    """A text part read back as a string only when it carried the cache marker that made it an array.

    A unit test of the comparator, like the one above: no adapter switches an unmarked body between the two
    forms today, so no conversation would show the difference.
    """

    def body(content: str | list[dict[str, Any]]) -> bytes:
        return json.dumps({'model': 'gpt-5.6-sol', 'messages': [{'role': 'user', 'content': content}]}).encode()

    def breaks(*bodies: bytes) -> list[Change]:
        return ConversationResult('openai-chat', list(bodies), (b'', b''), []).changes()

    marked = [{'type': 'text', 'text': 'Hi', 'prompt_cache_breakpoint': {'mode': 'explicit'}}]
    assert breaks(body(marked), body('Hi')) == []
    assert [b.section for b in breaks(body([{'type': 'text', 'text': 'Hi'}]), body('Hi'))] == ['messages']


def test_changes_are_named_by_structure():
    """Each change is named from the request's structure, which is what `KNOWN` entries match on.

    A unit test of the comparator: the conversations above don't change a setting, and a request whose bytes
    differ outside the compared sections is rare, but both need a name a `Known` entry can match.
    """

    def body(tool_choice: str, messages: list[str], max_tokens: int = 100) -> bytes:
        return json.dumps(
            {
                'model': 'gpt-5.6-sol',
                'max_tokens': max_tokens,
                'tool_choice': tool_choice,
                'messages': [{'role': 'user', 'content': text} for text in messages],
            }
        ).encode()

    def identities(*bodies: bytes) -> list[str]:
        return [change.identity for change in ConversationResult('openai-chat', list(bodies), (b'', b''), []).changes()]

    assert identities(body('auto', ['Hi']), body('none', ['Hi'])) == ['setting:tool_choice']
    # A block inserted ahead of earlier ones shows as each earlier block it displaced, by position.
    displaced = ConversationResult(
        'openai-chat',
        [body('auto', ['Hi', 'There', 'Now']), body('auto', ['Hi', 'Hey', 'There', 'Now'])],
        (b'', b''),
        [],
    ).changes()
    assert [(change.block_index, change.identity) for change in displaced] == [(1, 'user-prompt'), (2, 'user-prompt')]

    replays = (body('auto', ['Hi']), body('auto', ['Hi'], max_tokens=200))
    changes = ConversationResult('openai-chat', [], replays, []).nondeterministic_projection()
    assert [change.identity for change in changes or ()] == ['request']


def test_a_known_rewrite_does_not_cover_an_adjacent_insertion():
    """A rewrite and an insertion right after it are two changes, so an entry for one leaves the other failing.

    A unit test of the comparator on hand-built requests: index by index, the inserted message shows as the
    earlier message it displaced.
    """

    def body(messages: list[str]) -> bytes:
        return json.dumps(
            {'model': 'gpt-5.6-sol', 'messages': [{'role': 'user', 'content': m} for m in messages]}
        ).encode()

    result = ConversationResult('openai-chat', [body(['A', 'B', 'C']), body(['A', 'B2', 'X', 'C'])], (b'', b''), [])
    assert [change.block_index for change in result.changes()] == [1, 2]
    with pytest.raises(pytest.fail.Exception, match=re.escape('request 1 changed messages block 2 (user-prompt)')):
        check_conversation('example', result, known=(Known('prefix', 'example', 'messages', 'user-prompt'),))


def test_identical_anonymous_instructions_are_two_violations():
    """Instruction parts without an `id` are told apart by position, so two with the same text are two violations."""
    parameters = ModelRequestParameters(instruction_parts=[InstructionPart('Be brief.'), InstructionPart('Be brief.')])
    body = json.dumps({'model': 'gpt-5.6-sol', 'messages': [{'role': 'user', 'content': 'Hi'}]}).encode()
    result = ConversationResult('openai-chat', [body], (body, body), [parameters])
    assert list(result.unaddressable_instructions()) == ['instruction:anonymous@0', 'instruction:anonymous@1']
    with pytest.raises(pytest.fail.Exception, match=re.escape('instruction:anonymous@1: Be brief.')):
        check_conversation(
            'example', result, known=(Known('ids', 'example', 'instructions', 'instruction:anonymous@0'),)
        )


@pytest.mark.parametrize(
    ('earlier', 'later', 'expected'),
    [
        pytest.param(
            {'tools': [_chat_tool('look_up')], 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {'tools': [_chat_tool('look_up'), _chat_tool('forecast')], 'messages': [{'role': 'user', 'content': 'Hi'}]},
            [('tools', 'tool:forecast')],
            id='second-tool',
        ),
        pytest.param(
            {'messages': [{'role': 'developer', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}]},
            {
                'messages': [
                    {'role': 'developer', 'content': 'Be brief.'},
                    {'role': 'developer', 'content': 'Be kind.'},
                    {'role': 'user', 'content': 'Hi'},
                ]
            },
            [('system', 'system-prompt')],
            id='second-developer-message',
        ),
        pytest.param(
            {'messages': [{'role': 'user', 'content': 'Hi'}]},
            {'prompt_cache_key': 'conversation-2', 'messages': [{'role': 'user', 'content': 'Hi'}]},
            [('settings', 'setting:prompt_cache_key')],
            id='prompt-cache-key',
        ),
        pytest.param(
            {'tools': [_chat_tool('look_up')], 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {
                'tools': [_chat_tool('look_up')],
                'messages': [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}],
            },
            [],
            id='appended-message',
        ),
    ],
)
def test_appending_is_free_only_in_the_last_section(
    earlier: dict[str, Any], later: dict[str, Any], expected: list[tuple[str, str]]
):
    """A tool or system block appended ahead of messages already sent moves the prefix; a new message doesn't.

    A unit test of the comparator on hand-built OpenAI Chat requests: tools and system blocks serialize before
    the messages, so growing them pushes every message back even though each section on its own only grew.
    """
    bodies = [json.dumps({'model': 'gpt-5.6-sol', **body}).encode() for body in (earlier, later)]
    changes = ConversationResult('openai-chat', bodies, (b'', b''), []).changes()
    assert [(change.section, change.identity) for change in changes] == expected


def test_instruction_parts_sharing_an_id_are_told_apart():
    """A change to the first of two parts with the same id is named for that part, not lost under the second."""
    instruction_id = InstructionId(CapabilityInstructionSource('notes'))

    def parameters(first: str) -> ModelRequestParameters:
        return ModelRequestParameters(
            instruction_parts=[InstructionPart(first, id=instruction_id), InstructionPart('Second.', id=instruction_id)]
        )

    def body(first: str) -> bytes:
        system = [{'type': 'text', 'text': first}, {'type': 'text', 'text': 'Second.'}]
        return json.dumps({'model': 'claude-opus-5-5', 'system': system, 'messages': []}).encode()

    result = ConversationResult(
        'anthropic', [body('First.'), body('Changed.')], (b'', b''), [parameters('First.'), parameters('Changed.')]
    )
    assert [(change.section, change.identity) for change in result.changes()] == [
        ('system', 'instruction:capability:notes')
    ]

    result = ConversationResult(
        'anthropic', [body('First.'), body('First.')], (b'', b''), [parameters('First.'), parameters('First.')]
    )
    assert result.changes() == []


def test_a_request_scoped_prompt_left_visible_fails():
    """A request-scoped prompt still visible after its request fails the `request-scope` check.

    A unit test of the check on hand-built Anthropic requests, since #10938 fixed the one case the matrix had:
    the earlier request's prompt is marked `clear_at: next_user_message`, and no user message follows it.
    """
    nudge = {'role': 'system', 'content': [{'type': 'text', 'text': 'Keep it short.'}], 'clear_at': 'next_user_message'}
    first = [{'role': 'user', 'content': 'Hi'}, nudge]
    second = [
        *first,
        {'role': 'assistant', 'content': [{'type': 'text', 'text': ''}]},
        {'role': 'system', 'content': [{'type': 'text', 'text': 'Please return text or call a tool.'}]},
    ]
    bodies = [json.dumps({'model': 'claude-opus-5-5', 'messages': messages}).encode() for messages in (first, second)]
    result = ConversationResult('anthropic', bodies, (b'', b''), [])
    with pytest.raises(pytest.fail.Exception, match=re.escape('request 1 still shows')):
        check_conversation('example', result)


@pytest.mark.parametrize('protocol', ['vercel', 'agui'])
async def test_server_held_ui_history_keeps_openai_responses_chaining(
    protocol: Literal['vercel', 'agui'], monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    """A later turn through a server-held UI endpoint continues the OpenAI Responses chain.

    With `openai_previous_response_id='auto'`, a request chains only when the history it continues is the one
    the previous response answered (#10509, #10656, #11331). The stored history keeps each response's
    `provider_response_id` and history digest through a JSON round trip, so the second run's first request sends
    `previous_response_id` and only its new turn, though the client sent its whole conversation.
    """
    if not ui_imports_successful():  # pragma: lax no cover
        pytest.skip('ag-ui-protocol not installed')
    scenario = Scenario(
        ui_adapter=protocol,
        serialize_history=True,
        run_kwargs={'model_settings': OpenAIResponsesModelSettings(openai_previous_response_id='auto')},
    )
    result = await run_conversation('openai-responses', [], scenario, inject_clock(monkeypatch))

    # The first run makes three requests; the fourth is the second run's first.
    continued = json.loads(result.conversation[3])
    assert continued['previous_response_id'] == 'resp_3'
    assert continued['input'] == snapshot([{'role': 'user', 'content': 'And what else?'}])
