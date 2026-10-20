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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    MCP,
    AbstractCapability,
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
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

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
    build_capabilities,
    check_conversation,
    discover_capabilities,
    drain_events,
    inject_clock,
    provider_imports_successful,
    run_conversation,
)
from .conftest import try_import

with try_import() as mcp_imports_successful:
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset

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
        Known('prefix', 'https://github.com/pydantic/pydantic-ai/issues/10521', 'system', 'system-prompt'),
    ),
}
"""Expected failures, each linked to the issue or PR that fixes it. Remove an entry when its fix lands."""

ALLOWED: dict[str, Allowed] = {
    'agent:dynamic_instructions': Allowed(
        'docs/agent.md: dynamic instructions are re-rendered for every request, after the static ones and outside '
        'the cached prefix; append mode is #8188',
        # The `@agent.instructions` part, which has no id of its own, after the agent's static instructions.
        (('system', 'instruction:anonymous@1'),),
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

CAPABILITIES = discover_capabilities(('pydantic_ai.capabilities', 'pydantic_ai.models'))


def test_every_capability_has_a_case():
    """Every table entry names a capability that still exists, so renames and removals don't leave dead rows."""
    tables = FIXTURES.keys() | EXEMPT.keys() | KNOWN.keys() | ALLOWED.keys() | {name for name, _ in UNSUPPORTED}
    stale = tables - CAPABILITIES.keys() - AGENT_FEATURES.keys()
    assert not stale, f'entries for capabilities that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('name', sorted(CAPABILITIES.keys() - EXEMPT.keys()) + sorted(AGENT_FEATURES))
async def test_conversation_only_appends(
    name: str, provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    if reason := UNSUPPORTED.get((name, provider)):
        pytest.skip(reason)
    if name == 'MCP' and not mcp_imports_successful():  # pragma: lax no cover
        pytest.skip('mcp not installed')
    if name in AGENT_FEATURES:
        capabilities, scenario = [], AGENT_FEATURES[name]
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name].load, FIXTURES.get(name), tmp_path)
        scenario = Scenario()
    clock = inject_clock(monkeypatch)
    result = await run_conversation(provider, capabilities, scenario, clock)
    check_conversation(name, result, known=KNOWN.get(name, ()), allowed=ALLOWED.get(name), check_ids=scenario.check_ids)


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


def test_discovery_finds_capabilities_with_nested_type_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A capability whose base takes nested type arguments is still found, through an intermediate base.

    Discovery reads class statements from source, so this runs it over a throwaway package rather than a
    shipped one: a missed class is a capability the guard silently never checks.
    """
    package = tmp_path / 'nested_generic_capabilities'
    package.mkdir()
    (package / '__init__.py').write_text('from ._caps import Base, Leaf\n')
    (package / '_caps.py').write_text(
        'from typing import Any, Callable\n'
        'from pydantic_ai.capabilities import AbstractCapability\n\n'
        'class Base(AbstractCapability[dict[str, Callable[[int], Any]]]):\n    pass\n\n'
        'class Leaf(Base, metaclass=type):\n    pass\n\n'
        'class _Private(AbstractCapability[dict[str, Any]]):\n    pass\n'
    )
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])
    importlib.invalidate_caches()
    found = discover_capabilities(('nested_generic_capabilities',))
    assert sorted(found) == ['Base', 'Leaf']
    assert found['Leaf'].module == 'nested_generic_capabilities._caps'


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
