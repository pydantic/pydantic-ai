"""Every Harness capability keeps a conversation's requests append-only.

See `tests/append_only.py` for the scenario and the three invariants it checks, and
`tests/test_history_append_only.py` for the core capabilities. Every public capability class in
`pydantic_ai_harness` runs, constructed with its defaults or with a fixture from `FIXTURES`, unless
`EXEMPT` says why it can't run offline. Capabilities that keep state change it mid-conversation through
`SCENARIOS`, and the clock moves an hour between requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, Capability, LocalWorkspace, ToolSearch, Toolset
from pydantic_ai.messages import ModelMessage, TextContent, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import (
    Advisor,
    ApprovalReviewer,
    AskUser,
    BubblewrapSandbox,
    CapabilityCreation,
    ClampOversizedMessages,
    ClearToolResults,
    CodeMode,
    Coder,
    CompactOnRequest,
    DayAI,
    DeduplicateFileReads,
    FallbackCompaction,
    FileSystem,
    Grain,
    InputGuardrail,
    Macroscope,
    ManagedPrompt,
    Ordinal,
    OutputGuardrail,
    PostHog,
    Pylon,
    RelevanceCompaction,
    RepoContext,
    Researcher,
    Shell,
    Skills,
    SlidingWindowCompaction,
    SmartSearch,
    StackOne,
    SummarizingCompaction,
    SystemReminders,
    TieredCompaction,
    ToolCallJudge,
    ToolOutputLimits,
    TrajectoryJudge,
    WarnNearLimits,
)
from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
from pydantic_ai_harness.github import GitHub
from pydantic_ai_harness.linear import Linear
from pydantic_ai_harness.logfire_mcp import LogfireMCP
from pydantic_ai_harness.notion import Notion
from pydantic_ai_harness.slack import Slack
from pydantic_ai_harness.step_persistence import InMemoryStepStore
from pydantic_ai_harness.subagents import (
    DelegationReports,
    DelegationTaskEvent,
    DelegationTasks,
    SubAgent,
    SubAgents,
    is_task_report,
)
from pydantic_ai_harness.system_reminders import LLMReminder

from ..append_only import (
    CLOCK,
    PROVIDERS,
    Allowed,
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
    inject_clock,
    run_conversation,
    skip_unless_installed,
)

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


def _hosted_server() -> FastMCP:
    """An in-process stand-in for a hosted MCP service, with server instructions and one tool."""
    server = FastMCP('hosted', instructions='Hosted service instructions.')

    @server.tool()
    def search() -> str:
        return 'found'  # pragma: no cover

    return server


_HOSTED_MCP: tuple[type[AbstractCapability[Any]], ...] = (
    DayAI,
    GitHub,
    Grain,
    Linear,
    LogfireMCP,
    Notion,
    Ordinal,
    PostHog,
    Pylon,
    Slack,
)
"""Hosted MCP integrations, run against an in-process server through their `client` field."""


def _hosted(cls: Callable[..., AbstractCapability[Any]]) -> Fixture:
    return lambda _: cls(client=_hosted_server())


def _workspace(build: Callable[[], AbstractCapability[Any]]) -> Fixture:
    """The capability plus a local workspace in the test's directory, for capabilities that need one."""
    return lambda tmp_path: [build(), LocalWorkspace(tmp_path)]


def _repo_context(tmp_path: Path) -> list[AbstractCapability[Any]]:
    (tmp_path / 'AGENTS.md').write_text('Keep answers short.\n')
    return [RepoContext(), FileSystem(), LocalWorkspace(tmp_path)]


def _skills(tmp_path: Path) -> list[AbstractCapability[Any]]:
    skill = tmp_path / 'skills' / 'weather'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\ndescription: Report the weather.\n---\n\nSay sunny.\n')
    return [Skills(tmp_path / 'skills'), LocalWorkspace(tmp_path)]


def _deduplicate_file_reads(tmp_path: Path) -> list[AbstractCapability[Any]]:
    (tmp_path / 'notes.md').write_text('Sunny.\n')
    return [DeduplicateFileReads(file_key=_read_file_key), FileSystem(), LocalWorkspace(tmp_path)]


def _run_look_up(topic: str) -> tuple[str, dict[str, Any]]:
    return 'run_code', {'code': f'await look_up(topic={topic!r})'}


def _forecast(city: str) -> str:
    """Get the weather forecast for a city."""
    return f'Sunny in {city}.'


def _tides(port: str) -> str:
    """Get the tide times for a port."""
    return f'High tide at noon in {port}.'


def _deferred_forecast_and_tides() -> AbstractCapability[Any]:
    return Toolset(
        FunctionToolset(
            [Tool(_forecast, name='forecast', defer_loading=True), Tool(_tides, name='tides', defer_loading=True)]
        )
    )


def _forecasting() -> AbstractCapability[Any]:
    return Capability(
        id='forecasting', description='Weather forecasts.', tools=[Tool(_forecast, name='forecast')], defer_loading=True
    )


def _tidal() -> AbstractCapability[Any]:
    return Capability(id='tidal', description='Tide times.', tools=[Tool(_tides, name='tides')], defer_loading=True)


def _run_code(code: str) -> tuple[str, dict[str, Any]]:
    return 'run_code', {'code': code}


def _code_mode_discoveries(
    tides_discovery: tuple[str, dict[str, Any]],
    tides_call: tuple[str, dict[str, Any]],
    forecast_discovery: tuple[str, dict[str, Any]] = ('search_tools', {'queries': ['forecast']}),
) -> Scenario:
    """Reveal `forecast` and call it from code in the first run, then reveal and call `tides` in the continued one.

    The continued run starts from history that went through a JSON round trip, and is streamed.
    """
    return Scenario(
        turns=(
            Turn(calls=(forecast_discovery,)),
            Turn(calls=(_run_code("await forecast(city='Paris')"),)),
            Turn(text='Sunny in Paris.'),
            Turn(calls=(tides_discovery,)),
            Turn(calls=(tides_call,)),
            Turn(text='High tide at noon.'),
        ),
        serialize_history=True,
    )


_SEARCH_TIDES = ('search_tools', {'queries': ['tides']})
_LOAD_TIDAL = ('load_capability', {'id': 'tidal'})
_TIDES_FROM_CODE = _run_code("await tides(port='Brest')")
_CODE_MODE_SEARCHES = _code_mode_discoveries(_SEARCH_TIDES, _TIDES_FROM_CODE)
_CODE_MODE_LOADS = _code_mode_discoveries(_LOAD_TIDAL, _TIDES_FROM_CODE, ('load_capability', {'id': 'forecasting'}))


def _modal_sandbox() -> AbstractCapability[Any]:
    # Imported here, not at module level, so collecting this file never imports the Modal SDK. The test has
    # already skipped when `modal` isn't installed (`OPTIONAL_DEPENDENCIES`).
    from pydantic_ai_harness.modal_sandbox import ModalSandbox

    return ModalSandbox(warn_if_no_tools=False)


def _ipython_code_mode() -> AbstractCapability[Any]:
    # Imported here, so collecting this file never imports `jupyter_client`. The test has already skipped
    # when it isn't installed (`OPTIONAL_DEPENDENCIES`).
    from pydantic_ai_harness.ipython_code_mode import IPythonCodeMode

    return IPythonCodeMode(tools=[])


def _read_file_key(call: ToolCallPart) -> str | None:
    args = call.args_as_dict()
    return str(args['path']) if call.tool_name == 'read_file' else None


def _time_reminder(ctx: RunContext[Any]) -> str:
    return f'It is now {CLOCK.now:%H:%M} UTC.'


FIXTURES: dict[str, Fixture] = {
    **{cls.__name__: _hosted(cls) for cls in _HOSTED_MCP},
    'Advisor': lambda _: Advisor(TestModel()),
    'ApprovalReviewer': lambda _: ApprovalReviewer(TestModel(), question='Is this safe?'),
    'AskUser': lambda _: AskUser(answerer=None),
    'BubblewrapSandbox': lambda tmp_path: BubblewrapSandbox(LocalWorkspace(tmp_path)),
    'CapabilityCreation': lambda tmp_path: [CapabilityCreation(tmp_path / 'capabilities'), LocalWorkspace(tmp_path)],
    'ClampOversizedMessages': lambda _: ClampOversizedMessages(max_part_chars=100_000),
    'ClearToolResults': lambda _: ClearToolResults(max_messages=6, keep_pairs=1),
    'Coder': _workspace(Coder),
    'CompactOnRequest': lambda _: CompactOnRequest(SlidingWindowCompaction(max_messages=1, keep_messages=2)),
    'ConversationSearch': lambda _: ConversationSearch(
        SnapshotHistorySource(InMemoryStepStore()), scope='conversation'
    ),
    'DeduplicateFileReads': _deduplicate_file_reads,
    'FallbackCompaction': lambda _: FallbackCompaction(
        fallback_chain=[SlidingWindowCompaction(max_messages=6, keep_messages=2)]
    ),
    'FileSystem': _workspace(FileSystem),
    'InputGuardrail': lambda _: InputGuardrail(guard=lambda text: True),
    # Keeps the scenario's tools native, so the conversation checks the `run_code` declaration without
    # launching a kernel process for every provider.
    'IPythonCodeMode': lambda _: _ipython_code_mode(),
    'Macroscope': _workspace(Macroscope),
    'ManagedPrompt': lambda _: ManagedPrompt('assistant', default='You answer questions about the news.'),
    'ModalSandbox': lambda _: _modal_sandbox(),
    'OutputGuardrail': lambda _: OutputGuardrail(guard=lambda output: True),
    'RepoContext': _repo_context,
    'RelevanceCompaction': lambda _: RelevanceCompaction(TestModel(), max_tokens=20, keep_tokens=1),
    'Researcher': _workspace(Researcher),
    'Shell': _workspace(Shell),
    'Skills': _skills,
    'SlidingWindowCompaction': lambda _: SlidingWindowCompaction(max_messages=6, keep_messages=2),
    'SmartSearch': _workspace(lambda: SmartSearch(model=TestModel())),
    'StackOne': lambda _: StackOne('account', client=_hosted_server()),
    'SummarizingCompaction': lambda _: SummarizingCompaction(TestModel(), max_messages=6, keep_messages=2),
    'SystemReminders': lambda _: SystemReminders(dynamic_reminders=[_time_reminder]),
    'TieredCompaction': lambda _: TieredCompaction(
        tiers=[ClearToolResults(max_messages=6, keep_pairs=1)], target_tokens=10
    ),
    'ToolCallJudge': lambda _: ToolCallJudge(TestModel(), question='Is this call safe?'),
    'ToolOutputLimits': _workspace(ToolOutputLimits),
    'TrajectoryJudge': lambda _: TrajectoryJudge(model=TestModel()),
    'WarnNearLimits': lambda _: WarnNearLimits(max_iterations=2),
}
"""Capabilities that need configuration, keyed by class name, built for each test in its `tmp_path`."""

VARIANTS: dict[str, tuple[Fixture, Scenario]] = {
    # Each firing records a framework note and the generated text for good, so they must only ever be appended.
    'SystemReminders(LLMReminder)': (
        lambda _: SystemReminders(
            dynamic_reminders=[LLMReminder(TestModel(custom_output_text='Refocus.'), interval=2)]
        ),
        Scenario(),
    ),
    # Starts a background summary on almost every request but never reaches its trigger: a held summary
    # must not move the history before compaction is due.
    'FallbackCompaction(background_start=...)': (
        lambda _: FallbackCompaction(
            fallback_chain=[SummarizingCompaction(TestModel(), max_messages=1, keep_messages=1)],
            max_tokens=100_000,
            background_start=0.0001,
        ),
        Scenario(),
    ),
    'CodeMode(dynamic_catalog=True)': (
        lambda _: [
            CodeMode(dynamic_catalog=True),
            ToolSearch(strategy='keywords'),
            Toolset(FunctionToolset([Tool(_forecast, name='forecast', defer_loading=True)])),
        ],
        Scenario(
            look_up=_run_look_up,
            # Discover the deferred tool in the first run, then call it from code in the second.
            state_calls=(
                ('search_tools', {'queries': ['forecast']}),
                ('run_code', {'code': "await forecast(city='Paris')"}),
            ),
        ),
    ),
    # docs/harness/code-mode.md#tool-search-interaction: with the default `dynamic_catalog=True`, every discovery
    # below appends to the conversation, whichever way the tool is found and whether search is native or local.
    'CodeMode+ToolSearch': (
        lambda _: [CodeMode(), ToolSearch(), _deferred_forecast_and_tides()],
        _CODE_MODE_SEARCHES,
    ),
    'CodeMode+Capability(defer_loading=True)': (
        lambda _: [CodeMode(), _forecasting(), _tidal()],
        _CODE_MODE_LOADS,
    ),
    'CodeMode+ToolSearch+Capability(defer_loading=True)': (
        lambda _: [
            CodeMode(),
            ToolSearch(),
            Toolset(FunctionToolset([Tool(_forecast, name='forecast', defer_loading=True)])),
            _tidal(),
        ],
        _code_mode_discoveries(_LOAD_TIDAL, _TIDES_FROM_CODE),
    ),
    # A Tool Search corpus kept fully native: never folded into `run_code`, called directly once found.
    'CodeMode(tools=not-in-corpus)+ToolSearch': (
        lambda _: [
            CodeMode(tools=lambda ctx, td: td.with_native is None),
            ToolSearch(strategy='keywords'),
            _deferred_forecast_and_tides(),
        ],
        _code_mode_discoveries(_SEARCH_TIDES, ('tides', {'port': 'Brest'})),
    ),
    # One deferred tool sandboxed, another kept native: `search_tools` searches both, and the native search isn't sent.
    'CodeMode(tools=not-tides)+ToolSearch': (
        lambda _: [
            CodeMode(tools=lambda ctx, td: td.name != 'tides'),
            ToolSearch(strategy='keywords'),
            _deferred_forecast_and_tides(),
        ],
        _code_mode_discoveries(_SEARCH_TIDES, ('tides', {'port': 'Brest'})),
    ),
    # The same with the default strategy, which uses the provider's native tool search where there is one.
    'CodeMode(tools=not-tides)+ToolSearch()': (
        lambda _: [
            CodeMode(tools=lambda ctx, td: td.name != 'tides'),
            ToolSearch(),
            _deferred_forecast_and_tides(),
        ],
        _code_mode_discoveries(_SEARCH_TIDES, ('tides', {'port': 'Brest'})),
    ),
    # The documented contrast: the catalog lives in `run_code`'s description.
    'CodeMode(dynamic_catalog=False)+ToolSearch': (
        lambda _: [CodeMode(dynamic_catalog=False), ToolSearch(strategy='keywords'), _deferred_forecast_and_tides()],
        _CODE_MODE_SEARCHES,
    ),
    'CodeMode(dynamic_catalog=False)+Capability(defer_loading=True)': (
        lambda _: [CodeMode(dynamic_catalog=False), _forecasting(), _tidal()],
        _CODE_MODE_LOADS,
    ),
}
"""Non-default configurations worth their own case, keyed by a case name."""

SCENARIOS: dict[str, Scenario] = {
    'CodeMode': Scenario(look_up=_run_look_up),
    'CompactOnRequest': Scenario(state_calls=(('compact_context', {'focus': 'the forecast'}),)),
    'DeduplicateFileReads': Scenario(
        state_calls=(('read_file', {'path': 'notes.md'}), ('read_file', {'path': 'notes.md'}))
    ),
    'FileSystem': Scenario(
        state_calls=(
            ('write_file', {'path': 'notes.md', 'content': 'Sunny.\n'}),
            ('edit_file', {'path': 'notes.md', 'old_text': 'Sunny.', 'new_text': 'Rainy.'}),
        )
    ),
    'Memory': Scenario(
        state_calls=(
            ('write_memory', {'content': 'The user follows the weather.'}),
            ('write_memory', {'content': 'The user follows sports too.'}),
        )
    ),
    'Planning': Scenario(
        state_calls=(('add_task', {'content': 'Check the weather'}), ('add_task', {'content': 'Check the sports'}))
    ),
    'RepoContext': Scenario(
        state_calls=(
            ('edit_file', {'path': 'AGENTS.md', 'old_text': 'short', 'new_text': 'very short'}),
            ('write_file', {'path': 'docs/AGENTS.md', 'content': 'Docs are in Markdown.\n'}),
        )
    ),
}
"""Conversations that change a capability's state, or call tools the way the capability exposes them."""

EXEMPT: dict[str, Exempt] = {
    'AbsurdDurability': Exempt('Durable engine: needs a Postgres-backed Absurd queue.'),
    'AWSLambdaDurability': Exempt('Durable engine: needs the AWS Lambda durable functions runtime.'),
    'DynamicWorkflow': Exempt('Runs sub-agents a caller supplies, whose requests are their own conversations.'),
    'GoogleWorkspace': Exempt('Needs Google OAuth credentials to list its tools.'),
    'PGTaskDurability': Exempt('Durable engine: needs Postgres.'),
    'RenderWorkflows': Exempt('Durable engine: needs a Render Workflows app.'),
    'SSHWorkspace': Exempt('Needs an SSH host to attach a workspace to.'),
}

_ISSUES = 'https://github.com/pydantic/pydantic-ai/issues/'
_PULLS = 'https://github.com/pydantic/pydantic-ai/pull/'


KNOWN: dict[str, tuple[Known, ...]] = {}
"""Expected failures, each linked to the issue or PR that fixes it. Remove an entry when its fix lands."""

_COMPACTION = 'Compaction rewrites history by design, once per time its threshold is crossed'

_HISTORY = (('messages', 'user-prompt'), ('messages', 'model-response'), ('messages', 'tool-return'))
"""Every kind of history part, for compaction, which may rewrite any of them (but not the system prompt or tools)."""

ALLOWED: dict[str, Allowed] = {
    'ClearToolResults': Allowed(_COMPACTION, _HISTORY),
    'CompactOnRequest': Allowed(
        'Compaction rewrites history by design, once per accepted `compact_context` request from the model',
        _HISTORY,
    ),
    'DeduplicateFileReads': Allowed(
        'Clears superseded file reads in place, as documented, when the model reads a file again',
        (('messages', 'tool-return'),),
    ),
    # With a 20-token trigger every request is over it, so each request with an older step outside the
    # one-token tail judges and rewrites it: two in this conversation.
    'RelevanceCompaction': Allowed(_COMPACTION, _HISTORY, max_moves=2),
    'SlidingWindowCompaction': Allowed(_COMPACTION, _HISTORY),
    'SummarizingCompaction': Allowed(_COMPACTION, _HISTORY),
    # With a 10-token target every request is over it, so the tier clears each tool result as it ages out of
    # `keep_pairs`: one on the third request and one on the fifth.
    'TieredCompaction': Allowed(_COMPACTION, _HISTORY, max_moves=2),
    'Planning': Allowed(
        f'Request-only plan reminder at the end of each request, behind a cache breakpoint ({_PULLS}10257)',
        # Anthropic adds it to the last tool result message; the OpenAI APIs send it as a user message.
        (('messages', 'tool-return'), ('messages', 'user-prompt')),
        tail=True,
        records=('request-only user prompt',),
    ),
    'WarnNearLimits': Allowed(
        f'Request-scoped system prompt ({_PULLS}10561): sent for one request, then left out. Anthropic clears it '
        'natively with `clear_at`; elsewhere it is a request-only tail',
        (('messages', 'system-prompt'),),
        tail=True,
        records=('request-scoped system prompt',),
        providers=('openai-responses', 'openai-chat'),
    ),
    'SystemReminders': Allowed(
        f'Request-scoped system prompt ({_PULLS}9712): sent for one request, then left out. Anthropic clears it '
        'natively with `clear_at`; elsewhere it is a request-only tail',
        (('messages', 'system-prompt'),),
        tail=True,
        records=('request-scoped system prompt',),
        providers=('openai-responses', 'openai-chat'),
    ),
    # docs/harness/code-mode.md#tool-search-interaction: each discovery grows `run_code`'s description, and deferred
    # tools are declared to the native tool search and dropped when they move into `run_code`. Once the last one has
    # moved, the emptied corpus swaps the native search for the local `search_tools`, so the earlier native search
    # exchange is sent as a function call from then on, in the same request whose tools already changed.
    'CodeMode(dynamic_catalog=False)+ToolSearch': Allowed(
        '`dynamic_catalog=False` rewrites `run_code` on each discovery, as documented',
        (
            ('tools', 'tool:run_code'),
            ('tools', 'tool:forecast'),
            ('tools', 'tool:tides'),
            ('tools', 'tool:tool_search'),
            ('deferred-tools', 'tool:forecast'),
            ('deferred-tools', 'tool:tides'),
            ('messages', 'tool-return'),
            ('messages', 'model-response'),
        ),
        max_moves=2,
    ),
    'CodeMode(dynamic_catalog=False)+Capability(defer_loading=True)': Allowed(
        '`dynamic_catalog=False` rewrites `run_code` on each capability load, as documented',
        (('tools', 'tool:run_code'), ('deferred-tools', 'tool:forecast'), ('deferred-tools', 'tool:tides')),
        max_moves=2,
    ),
    # docs/harness/code-mode.md#tool-search-interaction: a selector that sandboxes one deferred tool and keeps
    # another native has `search_tools` search both, so neither is declared before it's found. Chat Completions
    # has no deferred-tool channel, so the native `tides` is added to `tools` when it's found (docs/tools-advanced.md).
    **{
        name: Allowed(
            'Chat Completions has no native tool search, so a native reveal adds the tool to `tools`, as documented',
            (('tools', 'tool:search_tools'), ('tools', 'tool:run_code')),
            providers=('openai-chat',),
        )
        for name in ('CodeMode(tools=not-tides)+ToolSearch', 'CodeMode(tools=not-tides)+ToolSearch()')
    },
    # docs/tools-advanced.md: the local `search_tools` fallback reveals a tool by adding it to `tools`.
    'CodeMode(tools=not-in-corpus)+ToolSearch': Allowed(
        'Chat Completions has no native tool search, so each native reveal adds the tool to `tools`, as documented',
        (('tools', 'tool:search_tools'), ('tools', 'tool:run_code')),
        max_moves=2,
        providers=('openai-chat',),
    ),
}
"""Documented prefix movers, checked exactly."""

OPTIONAL_DEPENDENCIES: dict[str, str] = {
    'BoxLiteSandbox': 'boxlite',
    'BrowserUse': 'browser_use',
    'DaytonaSandbox': 'daytona',
    'E2BSandbox': 'e2b',
    'ExaAgent': 'exa_py',
    'ExaSearch': 'exa_py',
    'IPythonCodeMode': 'jupyter_client',
    'ModalSandbox': 'modal',
    'PlaywrightBrowser': 'playwright',
    'PromptInjectionDefender': 'stackone_defender',
    'SmolSandbox': 'smol',
    'SpritesSandbox': 'sprites',
    'YouResearch': 'youdotcom',
    'YouSearch': 'youdotcom',
}
"""Capabilities whose module needs an optional dependency group, by the module that group provides."""

ENV: dict[str, dict[str, str]] = {
    'ExaAgent': {'EXA_API_KEY': 'test-key'},
    'ExaSearch': {'EXA_API_KEY': 'test-key'},
    'YouResearch': {'YDC_API_KEY': 'test-key'},
    'YouSearch': {'YDC_API_KEY': 'test-key'},
}
"""Credentials a capability checks when it's built; nothing is sent, as the scripted model never calls its tools."""

CAPABILITIES = discover_capabilities(('pydantic_ai_harness',))


_REPORT = 'The logs show one error.'


def _delegation_reports(
    tmp_path: Path,
) -> tuple[list[AbstractCapability[Any]], Scenario, Callable[[ConversationResult], None]]:
    """A background delegation that settles between the runs, and its report delivered in the second.

    The parent delegates in the background and answers; the child finishes; the second run, continued from
    stored history, gets the report when it would otherwise end and makes one more request. The check holds
    whatever shape the report takes: it only needs the child's output to reach history once.
    """
    settled = asyncio.Event()

    async def observe(update: DelegationTaskEvent) -> None:
        if update.task.status == 'finished':
            settled.set()

    async def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield _REPORT

    owner = DelegationTasks(directory=tmp_path / 'tasks', observer=observe)
    child = Agent(FunctionModel(stream_function=child_stream), deps_type=object, name='worker')

    @asynccontextmanager
    async def owning() -> AsyncGenerator[None]:
        async with owner.opened():
            with owner.bind():
                yield

    async def settle() -> None:
        with anyio.fail_after(10):
            await settled.wait()

    def delivered_once(result: ConversationResult) -> None:
        parts = [part for message in result.history for part in getattr(message, 'parts', ()) if _REPORT in str(part)]
        assert len(parts) == 1, f'expected the report in history once, found {len(parts)}'
        # Delivered as framed `<task_output>` data on the user channel, recognized by its metadata.
        (part,) = parts
        assert is_task_report(part), f'expected a task report, found {part!r}'
        assert not isinstance(part.content, str)
        (text,) = part.content
        assert isinstance(text, TextContent) and text.content.startswith('<task_output task_id=')

    scenario = Scenario(
        turns=(
            Turn(calls=(('delegate_task', {'agent_name': 'worker', 'task': 'Check the logs.', 'background': True}),)),
            Turn(text='I started a check of the logs.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='Here is more.'),
            Turn(text='The check found one error.'),
        ),
        serialize_history=True,
        run_kwargs={'conversation_id': 'parent'},
        around=owning,
        between_runs=settle,
    )
    capabilities: list[AbstractCapability[Any]] = [
        SubAgents(agents=[SubAgent(child)], agent_folders=None),
        DelegationReports(owner, conversation_id='parent'),
    ]
    return capabilities, scenario, delivered_once


CONVERSATIONS = {'DelegationReports': _delegation_reports}
"""Capabilities whose behavior needs a conversation of its own, with the check it must also pass."""


def test_every_capability_has_a_case():
    """Every table entry names a capability that still exists, so renames and removals don't leave dead rows."""
    tables = (
        FIXTURES.keys() | EXEMPT.keys() | SCENARIOS.keys() | KNOWN.keys() | ALLOWED.keys() | ENV.keys()
    ) | OPTIONAL_DEPENDENCIES.keys()
    stale = (tables | CONVERSATIONS.keys()) - CAPABILITIES.keys() - VARIANTS.keys()
    assert not stale, f'entries for capabilities that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('name', sorted(CAPABILITIES.keys() - EXEMPT.keys()) + sorted(VARIANTS))
async def test_conversation_only_appends(
    name: str, provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    skip_unless_installed(name, OPTIONAL_DEPENDENCIES)
    for key, value in ENV.get(name, {}).items():
        monkeypatch.setenv(key, value)
    check: Callable[[ConversationResult], None] | None = None
    if name in CONVERSATIONS:
        capabilities, scenario, check = CONVERSATIONS[name](tmp_path)
    elif name in VARIANTS:
        fixture, scenario = VARIANTS[name]
        capabilities = build_capabilities(name, lambda: AbstractCapability, fixture, tmp_path)
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name].load, FIXTURES.get(name), tmp_path)
        scenario = SCENARIOS.get(name, Scenario())
    clock = inject_clock(monkeypatch)
    result = await run_conversation(provider, capabilities, scenario, clock)
    if check is not None:
        check(result)
    check_conversation(name, result, known=KNOWN.get(name, ()), allowed=ALLOWED.get(name))
