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

import re
from collections.abc import AsyncIterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    MCP,
    Capability,
    CombinedCapability,
    DynamicCapability,
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
    WrapperCapability,
    XSearch,
)
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.toolsets import FunctionToolset

from .append_only import (
    CLOCK,
    PROVIDERS,
    Allowed,
    Clock,
    Exempt,
    Fixture,
    Known,
    Provider,
    Scenario,
    build_capabilities,
    check_conversation,
    discover_capabilities,
    inject_clock,
    provider_imports_successful,
    run_conversation,
)
from .conftest import try_import

with try_import() as mcp_imports_successful:
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset

pytestmark = pytest.mark.skipif(
    not (provider_imports_successful() and mcp_imports_successful()), reason='anthropic, openai or mcp not installed'
)


def _mcp_server() -> FastMCP[None]:
    server: FastMCP[None] = FastMCP('extra', instructions='Extra server instructions.')

    @server.tool()
    def extra_tool() -> str:
        return 'extra'  # pragma: no cover

    return server


def _no_op_tool() -> str:
    """Do nothing."""
    return 'ok'  # pragma: no cover


async def _ignore_events(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


def _keep_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages


def _keep_model(ctx: ModelSelectionContext[Any]) -> Model:
    assert ctx.model is not None
    return ctx.model


FIXTURES: dict[str, Fixture] = {
    'CombinedCapability': lambda _: CombinedCapability([Thinking()]),
    'DynamicCapability': lambda _: DynamicCapability(lambda ctx: Thinking()),
    'HandleDeferredToolCalls': lambda _: HandleDeferredToolCalls(lambda ctx, requests: requests.build_results()),
    'LocalWorkspace': lambda tmp_path: LocalWorkspace(tmp_path),
    'MCP': lambda _: MCP(local=MCPToolset(_mcp_server())),
    'NativeTool': lambda _: NativeTool(CodeExecutionTool()),
    'PrefixTools': lambda _: PrefixTools(Toolset(FunctionToolset([_no_op_tool])), prefix='extra'),
    'PrepareOutputTools': lambda _: PrepareOutputTools(lambda ctx, tools: tools),
    'PrepareTools': lambda _: PrepareTools(lambda ctx, tools: tools),
    'ProcessEventStream': lambda _: ProcessEventStream(_ignore_events),
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
    'AbstractCapability': Exempt('Abstract base class; every concrete subclass is a case of its own.'),
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
        Known('prefix', 'https://github.com/pydantic/pydantic-ai/issues/10521', match='The time is'),
    ),
}
"""Expected failures, each linked to the issue or PR that fixes it. Remove an entry when its fix lands."""

ALLOWED: dict[str, Allowed] = {
    'agent:dynamic_instructions': Allowed(
        'docs/agent.md: dynamic instructions are re-rendered for every request, after the static ones and outside '
        'the cached prefix; append mode is #8188',
        max_moves=4,
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


AGENT_FEATURES: dict[str, Scenario] = {
    'agent:dynamic_system_prompt': Scenario(configure=_dynamic_system_prompt, check_ids=False),
    'agent:dynamic_instructions': Scenario(configure=_dynamic_instructions, check_ids=False),
}
"""Agent features, not capabilities, that render state into the prompt, run through the same checks."""

DISCOVERED = discover_capabilities(('pydantic_ai.capabilities', 'pydantic_ai.models'))
CAPABILITIES = DISCOVERED.capabilities


def test_every_capability_has_a_case():
    """Every table entry names a capability that still exists, so renames and removals don't leave dead rows."""
    stale = (FIXTURES.keys() | EXEMPT.keys()) - CAPABILITIES.keys() - DISCOVERED.not_installed
    assert not stale, f'`FIXTURES`/`EXEMPT` entries for capabilities that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('name', sorted(CAPABILITIES.keys() - EXEMPT.keys()) + sorted(AGENT_FEATURES))
async def test_conversation_only_appends(
    name: str, provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    if reason := UNSUPPORTED.get((name, provider)):
        pytest.skip(reason)
    if name in AGENT_FEATURES:
        capabilities, scenario = [], AGENT_FEATURES[name]
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name], FIXTURES.get(name), tmp_path)
        scenario = Scenario()
    clock = inject_clock(monkeypatch)
    result = await run_conversation(provider, capabilities, scenario, clock)
    check_conversation(name, result, known=KNOWN.get(name, ()), allowed=ALLOWED.get(name), check_ids=scenario.check_ids)


def _clock_instructions(ctx: RunContext[Any]) -> str:
    return f'The time is {CLOCK.now:%H:%M}.'


def _random_instructions(ctx: RunContext[Any]) -> str:
    return f'Your session is {uuid4().hex}.'


@pytest.mark.parametrize(
    ('capability', 'known', 'allowed', 'outcome', 'report'),
    [
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            None,
            'fail',
            'rewrote the prefix',
            id='prefix',
        ),
        pytest.param(Capability[Any](instructions='Answer briefly.'), (), None, 'fail', 'without an `id`', id='ids'),
        pytest.param(
            Capability[Any](id='random', instructions=_random_instructions),
            (Known('prefix', 'example', match='Your session is'),),
            None,
            'fail',
            'built the same request twice from the same history with different bytes',
            id='determinism',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (Known('prefix', 'example', match='The time is'),),
            None,
            'xfail',
            'known: [prefix] example',
            id='known',
        ),
        pytest.param(
            Capability[Any](instructions='Answer briefly.', id='static'),
            (Known('prefix', 'example'),),
            None,
            'fail',
            'no longer fails these checks as expected',
            id='stale-known',
        ),
        pytest.param(
            Capability[Any](instructions='Answer briefly.', id='static'),
            (),
            Allowed('example'),
            'fail',
            'is listed in `ALLOWED` (example) but no longer moves the prefix',
            id='stale-allowed',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            Allowed('example'),
            'fail',
            'moved the prefix 4 times, more than the 1 its `ALLOWED` entry permits',
            id='too-many-moves',
        ),
        pytest.param(
            Capability[Any](id='clock', instructions=_clock_instructions),
            (),
            Allowed('example', tail=True),
            'fail',
            'may only replace the last block of each request',
            id='deep-tail',
        ),
    ],
)
async def test_checks_report_what_they_catch(
    capability: Capability[Any],
    known: tuple[Known, ...],
    allowed: Allowed | None,
    outcome: str,
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
        build_capabilities('NativeTool', NativeTool, None, tmp_path)
