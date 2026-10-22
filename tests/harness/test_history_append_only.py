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
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ToolSearch, Toolset
from pydantic_ai.messages import ModelMessage, ToolCallPart
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
    RepoContext,
    Researcher,
    Shell,
    Skills,
    SlidingWindowCompaction,
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
from pydantic_ai_harness.subagents import DelegationReports, DelegationTaskEvent, DelegationTasks, SubAgent, SubAgents

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


def _modal_sandbox() -> AbstractCapability[Any]:
    # Imported here, not at module level, so collecting this file never imports the Modal SDK. The test has
    # already skipped when `modal` isn't installed (`OPTIONAL_DEPENDENCIES`).
    from pydantic_ai_harness.modal_sandbox import ModalSandbox

    return ModalSandbox(warn_if_no_tools=False)


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
    'ConversationSearch': lambda _: ConversationSearch(
        SnapshotHistorySource(InMemoryStepStore()), scope='conversation'
    ),
    'DeduplicateFileReads': _deduplicate_file_reads,
    'FallbackCompaction': lambda _: FallbackCompaction(
        fallback_chain=[SlidingWindowCompaction(max_messages=6, keep_messages=2)]
    ),
    'FileSystem': _workspace(FileSystem),
    'InputGuardrail': lambda _: InputGuardrail(guard=lambda text: True),
    'Macroscope': _workspace(Macroscope),
    'ManagedPrompt': lambda _: ManagedPrompt('assistant', default='You answer questions about the news.'),
    'ModalSandbox': lambda _: _modal_sandbox(),
    'OutputGuardrail': lambda _: OutputGuardrail(guard=lambda output: True),
    'RepoContext': _repo_context,
    'Researcher': _workspace(Researcher),
    'Shell': _workspace(Shell),
    'Skills': _skills,
    'SlidingWindowCompaction': lambda _: SlidingWindowCompaction(max_messages=6, keep_messages=2),
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
}
"""Non-default configurations worth their own case, keyed by a case name."""

SCENARIOS: dict[str, Scenario] = {
    'CodeMode': Scenario(look_up=_run_look_up),
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


_MEMORY_OPENAI_MOVES: dict[Provider, dict[str, int]] = {
    'openai-chat': {'user-prompt': 2, 'model-response': 3, 'tool-return': 4},
    'openai-responses': {'user-prompt': 2, 'model-response': 4, 'tool-return': 4},
}
"""How many messages of each kind `Memory`'s snapshot rewrites or pushes back on the OpenAI APIs."""

KNOWN: dict[str, tuple[Known, ...]] = {
    'CodeMode(dynamic_catalog=True)': (
        Known('ids', f'{_ISSUES}8965', 'instructions', 'instruction:anonymous@1'),
        # Discovering a deferred tool rewrites the catalog block in the system prompt and folds the tool into
        # `run_code` by dropping its declaration. On Responses the native client-executed `tool_search` is also
        # swapped for the local `search_tools` function, which moves `run_code` up.
        Known('prefix', f'{_ISSUES}8965', 'system', 'instruction:anonymous@1'),
        Known('prefix', f'{_ISSUES}8965', 'deferred-tools', 'tool:forecast', providers=('anthropic',)),
        *(
            Known('prefix', f'{_ISSUES}8965', 'tools', f'tool:{name}', providers=('openai-responses',))
            for name in ('forecast', 'tool_search', 'run_code')
        ),
    ),
    # Anthropic attaches the snapshot to the turn's user prompt, so the write, the next turn and the second write
    # each rewrite one. The OpenAI APIs send it as a message after the user prompt, so each change also pushes
    # every later message back a position.
    'Memory': (
        Known('prefix', f'{_ISSUES}8996', 'messages', 'user-prompt', count=3, providers=('anthropic',)),
        *(
            Known('prefix', f'{_ISSUES}8996', 'messages', identity, count=count, providers=(provider,))
            for provider, counts in _MEMORY_OPENAI_MOVES.items()
            for identity, count in counts.items()
        ),
    ),
    # The summary goes into the standing system prompt instead of joining history: ahead of the agent's
    # instructions (which it pushes back) on Anthropic and Chat, and as a leading system message on Responses.
    'SummarizingCompaction': (
        Known('prefix', f'{_ISSUES}7289', 'system', 'system-prompt', count=2, providers=('anthropic', 'openai-chat')),
        Known('prefix', f'{_ISSUES}7289', 'system', 'system-prompt', providers=('openai-responses',)),
    ),
}
"""Expected failures, each linked to the issue or PR that fixes it. Remove an entry when its fix lands."""

_COMPACTION = 'Compaction rewrites history by design, once per time its threshold is crossed'

_HISTORY = (('messages', 'user-prompt'), ('messages', 'model-response'), ('messages', 'tool-return'))
"""Every kind of history part, for compaction, which may rewrite any of them (but not the system prompt or tools)."""

ALLOWED: dict[str, Allowed] = {
    'ClearToolResults': Allowed(_COMPACTION, _HISTORY),
    'DeduplicateFileReads': Allowed(
        'Clears superseded file reads in place, as documented, when the model reads a file again',
        (('messages', 'tool-return'),),
    ),
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
        f'Request-only reminders at the end of each request, behind a cache breakpoint; moving to '
        f'request-scoped system prompts in {_PULLS}9712',
        (('messages', 'user-prompt'), ('messages', 'tool-return')),
        tail=True,
        records=('request-only user prompt',),
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
