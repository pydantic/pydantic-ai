"""One case per Pydantic AI Harness capability: how the suites build it, and what two of it on one agent mean.

See `tests/capability_cases.py` for the case types and the core capabilities. Every capability class in
`pydantic_ai_harness`, public or not, has an entry in `CASES`, and `test_every_capability_has_a_case` fails
when one is missing. Adding a capability is therefore one entry, and a decision about what two of it mean,
taken once, here, as the case's `combine`. The answers, and what picks between them:

- `Combines` -- the class declares a default `id`, so two are one configuration stated twice and
  Pydantic AI merges them field by field. This is what lets two packaged harnesses that each carry
  a `ToolOutputLimits` compose instead of colliding.
- `Narrows` -- a default `id` and a `combine` method that narrows rather than unions, so two that
  disagree raise: the configuration is an access boundary.
- `Anonymous` -- no default `id`, and two really do coexist.
- `Collides` -- no default `id`, and two never coexist anyway, because the toolset registers fixed
  tool names. A fact rather than a preference, and pinned by
  `test_two_of_a_colliding_capability_still_raise` so it cannot drift back into an `Anonymous`
  reason describing a configuration ("one per rooted directory") that is unreachable.
- `Rejected` -- no default `id`, and something else rejects a second one, as a durable engine does.

Declaring a default `id` is the whole policy: there is no `combine` method to override unless the
merge needs something the field-by-field default cannot express, such as a budget that should take the
*smaller* value.
"""

from __future__ import annotations

import ast
import importlib.util
import tempfile
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic_ai_harness
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, RawToolArgs
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import (
    Advisor,
    AnnounceWorkingDirectory,
    ApprovalReviewer,
    AskUser,
    BackgroundTools,
    BenchmarkInstructions,
    BubblewrappedWorkspace,
    CapabilityCreation,
    ClampOversizedMessages,
    ClearToolResults,
    Coder,
    CompactOnRequest,
    ComputerUse,
    CurrentTime,
    DayAI,
    DeduplicateFileReads,
    FallbackCompaction,
    FileSystem,
    Goal,
    Grain,
    InputGuardrail,
    KeenableSearch,
    LabelCrossModelHistory,
    LocalStack,
    Macroscope,
    ManagedPrompt,
    Memory,
    ModelChoice,
    ModelRouter,
    Ordinal,
    OutputGuardrail,
    Planning,
    PostHog,
    PruneToolResults,
    Pylon,
    RelevanceCompaction,
    RepoContext,
    ReportToolErrors,
    RequireVerification,
    Researcher,
    Shell,
    SlidingWindowCompaction,
    SmartSearch,
    SpendLimits,
    SSHWorkspace,
    StackOne,
    StepPersistence,
    StreamGuardrail,
    SubAgent,
    SubAgents,
    SummarizingCompaction,
    SystemReminders,
    TieredCompaction,
    ToolBridge,
    ToolBudgets,
    ToolCallJudge,
    ToolGuardrail,
    ToolOutputLimits,
    TrajectoryJudge,
    WarnNearLimits,
)
from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse
from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
from pydantic_ai_harness.github import GitHub
from pydantic_ai_harness.linear import Linear
from pydantic_ai_harness.logfire_mcp import LogfireMCP
from pydantic_ai_harness.memory import InMemoryMemoryStore
from pydantic_ai_harness.notion import Notion
from pydantic_ai_harness.peers import InMemoryPeerNetwork, Peers
from pydantic_ai_harness.planning import SqlitePlanStore
from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.slack import Slack
from pydantic_ai_harness.step_persistence import InMemoryStepStore
from pydantic_ai_harness.sub_agents import DelegationReports, DelegationTasks
from pydantic_ai_harness.system_reminders import Reminder

from ..append_only import CLOCK, Fixture, discover_capabilities
from ..capability_cases import (
    ECHO,
    ECHO_STEPS,
    Anonymous,
    Calls,
    CapabilityCase,
    Collides,
    Combines,
    Contract,
    DurableSteps,
    Narrows,
    Rejected,
    Unbuildable,
    with_echo,
)
from ..function_model_prefix import COMPACTION_REWRITES_HISTORY
from ._counting_clients import (
    ExaAgentRuns,
    ExaClient,
    KeenableClient,
    YouClient,
    counting_aws_cli,
    counting_model,
    judge_relevant,
)
from ._mcp_server import MCPServer
from .computer_use.display_fakes import ScreenProvider

if TYPE_CHECKING:
    from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox


HARNESS_CAPABILITY_PACKAGES = ('pydantic_ai_harness',)
"""The package whose capability classes `CASES` covers."""


def hosted_server() -> MCPServer:
    """An in-process stand-in for a hosted MCP service, with server instructions and one tool."""
    server = MCPServer('hosted', instructions='Hosted service instructions.')

    @server.tool()
    def search() -> str:
        return 'found'

    return server


def _hosted(cls: Callable[..., AbstractCapability[None]]) -> Fixture:
    """A hosted MCP integration, run against an in-process server through its `client` field."""
    return lambda _: cls(client=hosted_server())


def _mcp_steps(toolset_id: str, *, instructions: bool = True) -> DurableSteps:
    """What one call to a tool of the MCP server `toolset_id` records: its tools, listed once per run and cached, the
    call, and, when the server's instructions are included, a read of them for each of the run's two requests."""
    reads = ('get_instructions',) if instructions else ()
    operations = ('get_tools', *reads, 'call_tool', *reads)
    return DurableSteps(
        dbos=tuple(f'mcp_server__{toolset_id}.{operation}' for operation in operations),
        temporal=tuple(f'mcp_server__{toolset_id}__{operation}' for operation in operations),
    )


def _hosted_calls(cls: Callable[..., AbstractCapability[None]], tool: str, toolset_id: str) -> Calls:
    """A call to the hosted stand-in's one tool, under the name `cls` gives it, through the toolset `toolset_id`."""
    return Calls(((tool, {}),), _hosted(cls), durable_steps=_mcp_steps(toolset_id))


def _workspace_steps(calls: int) -> tuple[str, ...]:
    """The DBOS steps of `calls` workspace calls, the run's `ensure` included."""
    return ('capability__workspace.call',) * calls


def _workspace(build: Callable[[], AbstractCapability[None]]) -> Fixture:
    """The capability plus a local workspace in the test's directory, for capabilities that need one."""
    return lambda tmp_path: [build(), LocalWorkspace(tmp_path)]


def _repo_context(tmp_path: Path) -> list[AbstractCapability[None]]:
    (tmp_path / 'AGENTS.md').write_text('Keep answers short.\n')
    return [RepoContext(), FileSystem(), LocalWorkspace(tmp_path)]


def _skills(tmp_path: Path) -> list[AbstractCapability[None]]:
    skill = tmp_path / 'skills' / 'weather'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\ndescription: Report the weather.\n---\n\nSay sunny.\n')
    return [Skills(tmp_path / 'skills'), LocalWorkspace(tmp_path)]


def _lowered_skills(tmp_path: Path) -> list[AbstractCapability[None]]:
    # Below `'operator'`, so the skill is held back from the instructions and `list_skills` and `load_skill` offer it.
    capabilities = _skills(tmp_path)
    return [capabilities[0], LocalWorkspace(tmp_path, authority='none')]


def _read_file_key(call: ToolCallPart) -> str | None:
    args = call.args_as_dict()
    return str(args['path']) if call.tool_name == 'read_file' else None


def _deduplicate_file_reads(tmp_path: Path) -> list[AbstractCapability[None]]:
    (tmp_path / 'notes.md').write_text('Sunny.\n')
    return [DeduplicateFileReads(file_key=_read_file_key), FileSystem(), LocalWorkspace(tmp_path)]


def _pydantic_ai_docs(tmp_path: Path) -> list[AbstractCapability[None]]:
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'hooks.md').write_text('# Hooks\n')
    return [PydanticAIDocs(local_docs_path=tmp_path / 'docs', cache=False), LocalWorkspace(tmp_path)]


def _macroscope(tmp_path: Path) -> list[AbstractCapability[None]]:
    """`Macroscope` running a stand-in CLI that reports a finished review with no issues, on stderr as it does."""
    command = tmp_path / 'macroscope'
    command.write_text("#!/bin/sh\nprintf 'review_id=review\\nissue_status=completed\\n' >&2\n")
    command.chmod(0o755)
    return [Macroscope(command=str(command), base='main'), LocalWorkspace(tmp_path)]


def _capability_creation(tmp_path: Path) -> list[AbstractCapability[None]]:
    return [CapabilityCreation(tmp_path / 'capabilities'), LocalWorkspace(tmp_path)]


_AUTHORED = """
from pydantic_ai.capabilities import AbstractCapability


class Marker(AbstractCapability):
    def get_instructions(self):
        return 'marker'
"""
"""A capability for `CapabilityCreation` to author."""


def _smart_search(tmp_path: Path) -> list[AbstractCapability[None]]:
    (tmp_path / 'config.py').write_text('def load(path):\n    # parse the config file\n    return path\n')
    return [SmartSearch(counting_model('smart_search.judge', judge_relevant), guidance=''), LocalWorkspace(tmp_path)]


class _PruneLatestReturn(AbstractCapability[None]):
    """Points a scripted `prune_tool_result` call at the previous step's result, whose ID differs by provider."""

    async def before_tool_validate(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        if call.tool_name != 'prune_tool_result':
            return args
        [latest] = [part.tool_call_id for part in ctx.messages[-2].parts if isinstance(part, ToolReturnPart)]
        return {**call.args_as_dict(), 'tool_call_id': latest}


def _modal_sandbox() -> AbstractCapability[None]:
    # Imported here, not at module level, so collecting the suites never imports the Modal SDK. The test has
    # already skipped when `modal` isn't installed (the case's `requires`).
    from pydantic_ai_harness.modal_sandbox import ModalSandbox

    return ModalSandbox(warn_if_no_tools=False)


def _kubernetes_sandbox() -> KubernetesSandbox:
    # Imported here, so collecting the suites never imports the agent-sandbox SDK. The test has already
    # skipped when it isn't installed (the case's `requires`).
    from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox

    return KubernetesSandbox(warm_pool='python-pool')


def _ipython_code_mode() -> AbstractCapability[None]:
    # Imported here, so collecting the suites never imports `jupyter_client`. The test has already skipped
    # when it isn't installed (the case's `requires`).
    from pydantic_ai_harness.ipython_code_mode import IPythonCodeMode

    return IPythonCodeMode(tools=[])


def _browser_use(_: Path) -> AbstractCapability[None]:
    # Imported here, so collecting the suites never imports browser-use, which only the `harness browser-use` CI job
    # installs. The test has already skipped when it isn't installed (the case's `requires`).
    from .browser_use._offline_browser import offline_browser_use

    return offline_browser_use()


def _time_reminder(ctx: RunContext[None]) -> str:
    return f'It is now {CLOCK.now:%H:%M} UTC.'


async def _goal_met(ctx: RunContext[None], output: object) -> None:
    return None


async def _decline(request: AskUserRequest) -> AskUserResponse:
    """Only here to construct `AskUser`; the combine tests never call a tool."""
    return AskUserResponse(cancelled=True)  # pragma: no cover


COMBINE_ROOT_A = Path(tempfile.mkdtemp(prefix='combine-a-'))
COMBINE_ROOT_B = Path(tempfile.mkdtemp(prefix='combine-b-'))
"""Two distinct roots, so a `Collides` pair differs in the way its `Anonymous` reason once claimed
mattered -- and still collides."""


def _check_skills(merged: Skills[None]) -> None:
    sources = merged._sources  # pyright: ignore[reportPrivateUsage]
    assert [source.directories for source in sources] == [('first',), ('second',)]


def _check_model_router(merged: ModelRouter[object]) -> None:
    assert (list(merged.choices), merged.default) == (['fast', 'capable'], 'capable')


def _check_planning(merged: Planning[None]) -> None:
    assert merged.inject is False


def _check_spend_limits(merged: SpendLimits[None]) -> None:
    assert merged.expose_tools is False


def _check_step_persistence(merged: StepPersistence[None]) -> None:
    assert merged.agent_name == 'second'


def _check_summarizing(merged: SummarizingCompaction[None]) -> None:
    assert merged.keep_messages == 7


def _check_relevance(merged: RelevanceCompaction[None]) -> None:
    assert merged.keep_tokens == 7


def _check_system_reminders(merged: SystemReminders[None]) -> None:
    # Additive sequences union, so every reminder either side declared still fires.
    assert [r.content for r in merged.reminders] == ['first', 'second']


def _check_compact_on_request(merged: CompactOnRequest[None]) -> None:
    assert merged.max_per_run == 2


def _check_prune_tool_results(merged: PruneToolResults[None]) -> None:
    assert merged.placeholder == '[second]'


def _check_tool_output_limits(merged: ToolOutputLimits[None]) -> None:
    assert merged.strip_ansi is True


def _check_tool_bridge(merged: ToolBridge[None]) -> None:
    assert merged.max_results == 5


def _check_advisor(merged: Advisor[None]) -> None:
    assert merged.max_tokens == 4096


def _check_background_tools(merged: BackgroundTools[None]) -> None:
    assert callable(merged.tools)


def _check_benchmark_instructions(merged: BenchmarkInstructions[None]) -> None:
    assert merged.id == 'benchmark_instructions'


def _check_working_directory(merged: AnnounceWorkingDirectory[None]) -> None:
    assert merged.id == 'announce_working_directory'


def _check_current_time(merged: CurrentTime[None]) -> None:
    assert (merged.timezone, merged.precision) == ('Asia/Tokyo', 'hour')


def _check_label_cross_model_history(merged: LabelCrossModelHistory[None]) -> None:
    assert (merged.id, merged.notice) == ('label_cross_model_history', 'second')


def _check_sub_agents(merged: SubAgents[None]) -> None:
    # Rosters union: an agent either side could reach stays reachable through one delegate tool.
    assert [entry.agent.name for entry in merged.agents] == ['alpha', 'beta']


def _check_computer_use(merged: ComputerUse[None]) -> None:
    # One display: the later configuration wins, defaults included.
    assert (merged.settle_seconds, merged.native) == (2, False)


def _check_ssh_workspace(merged: SSHWorkspace[None]) -> None:
    assert (merged.destination, merged.env, merged.read_only) == ('second', None, False)


_SANDBOX = 'two coexist; the first supplies the run workspace, as core picks the first supplier'
_DURABILITY = (
    'a durability engine is one per agent; `from_agent` rejects a second when the engine looks '
    'itself up, before any id is consulted'
)
_EXA_KEY = {'EXA_API_KEY': 'test-key'}
_YOU_KEY = {'YDC_API_KEY': 'test-key'}


CASES: dict[str, CapabilityCase] = {
    # -- One per agent: a default `id`, and `combine` says what two of them mean. --
    'Skills': CapabilityCase(
        Combines(
            'one catalog; every library either names stays reachable',
            # `skip`: the contracts run the pair, in a workspace that holds neither library.
            lambda: (
                Skills[None]('first', missing_directories='skip'),
                Skills[None]('second', missing_directories='skip'),
            ),
            _check_skills,
        ),
        build=_skills,
        calls=Calls(
            (('list_skills', {}), ('load_skill', {'name': 'weather'})),
            _lowered_skills,
            durable_steps=DurableSteps(
                dbos=_workspace_steps(22),
                # Under Temporal, the tools' own workspace reads run inside their `call_tool` activities.
                temporal=(*('capability__workspace__call',) * 8, *('toolset__skills_lowered__call_tool',) * 2),
            ),
        ),
    ),
    'Memory': CapabilityCase(
        Narrows(
            'one notebook per id, and its store and namespace are a tenant boundary; two that differ need their own ids',
            lambda cls: (cls(namespace='first'), cls(namespace='second')),
        ),
        calls=Calls(
            (
                ('write_memory', {'content': 'The user prefers tabs.', 'file': 'style.md'}),
                # A missing file asks the model to retry, which the operation records as data.
                ('read_memory', {'file': 'missing.md'}),
                ('search_memory', {'query': 'tabs'}),
                ('delete_memory', {'file': 'missing.md'}),
            ),
            lambda _: Memory(store=InMemoryMemoryStore()),
            durable_steps=DurableSteps(
                dbos=(
                    'capability__memory.load_snapshot',
                    'capability__memory.write_memory',
                    'capability__memory.read_memory',
                    'capability__memory.search_memory',
                    'capability__memory.delete_memory',
                ),
                temporal=('capability__memory__load_snapshot', *('toolset__memory__call_tool',) * 4),
            ),
        ),
    ),
    'ModelRouter': CapabilityCase(
        Combines(
            'only the last model contribution selects, so two routers become one whose menu has both sets of choices',
            lambda: (
                ModelRouter[object](
                    choices={'fast': ModelChoice('test', 'Use for lookups.')},
                    router_model='test',
                    default='fast',
                    mode='per_step',
                ),
                ModelRouter[object](
                    choices={'capable': ModelChoice('test', 'Use for hard work.')},
                    router_model='test',
                    default='capable',
                ),
            ),
            _check_model_router,
        ),
        build=Unbuildable(
            "Replaces the run's model with one from its menu, so the scenario's provider model is never requested."
        ),
    ),
    'Planning': CapabilityCase(
        Combines(
            'one plan per agent; `PlanningToolset` registers fixed tool names',
            lambda: (Planning[None](), Planning[None](inject=False)),
            _check_planning,
        ),
        calls=Calls(
            (
                ('write_plan', {'items': [{'id': 'first', 'content': 'Write the migration'}]}),
                ('add_task', {'content': 'Run the tests'}),
            ),
            lambda tmp_path: Planning(store=SqlitePlanStore(database=str(tmp_path / 'plan.db'))),
            durable_steps=DurableSteps(
                dbos=(
                    'capability__planning.read_plan',
                    'capability__planning.read_plan',
                    'capability__planning.set_items',
                    'capability__planning.read_plan',
                    'capability__planning.read_plan',
                    'capability__planning.read_plan',
                    'capability__planning.add_item',
                    'capability__planning.read_plan',
                    'capability__planning.read_plan',
                ),
                # Its run fails under Temporal (`DURABLE_TEMPORAL` exempts it, #7971), so nothing is recorded there.
            ),
        ),
    ),
    'SpendLimits': CapabilityCase(
        Combines(
            'one spend authority per agent',
            lambda: (SpendLimits[None](), SpendLimits[None](expose_tools=False)),
            _check_spend_limits,
        )
    ),
    'StepPersistence': CapabilityCase(
        Combines(
            'one run identity per agent',
            lambda: (StepPersistence[None](agent_name='first'), StepPersistence[None](agent_name='second')),
            _check_step_persistence,
        ),
        calls=Calls(
            ECHO,
            with_echo(StepPersistence),
            # Every hook that records the run is a `durable_operation` that workflow code calls.
            durable_steps=DurableSteps(
                dbos=(
                    'capability__step_persistence.registration_id',
                    'capability__step_persistence.register_run',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.record_tool_effect',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.finish_tool_effect',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.save_snapshot',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.append_event',
                    'capability__step_persistence.save_snapshot',
                    'capability__step_persistence.append_event',
                ),
                temporal=(
                    'capability__step_persistence__registration_id',
                    'capability__step_persistence__register_run',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__record_tool_effect',
                    'capability__step_persistence__append_event',
                    'toolset__echo__call_tool',
                    'capability__step_persistence__finish_tool_effect',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__save_snapshot',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__append_event',
                    'capability__step_persistence__save_snapshot',
                    'capability__step_persistence__append_event',
                ),
            ),
        ),
    ),
    'RelevanceCompaction': CapabilityCase(
        Combines(
            'two would each judge every step, the second judging what the first already compacted',
            lambda: (
                RelevanceCompaction[None](TestModel(), max_tokens=50, keep_tokens=3),
                RelevanceCompaction[None](TestModel(), max_tokens=50, keep_tokens=7),
            ),
            _check_relevance,
        ),
        build=lambda _: RelevanceCompaction(TestModel(), max_tokens=20, keep_tokens=1),
    ),
    'SummarizingCompaction': CapabilityCase(
        Combines(
            'two would each make a model call, the second summarizing the first summary',
            lambda: (
                SummarizingCompaction[None](max_messages=50, keep_messages=3),
                SummarizingCompaction[None](max_messages=50, keep_messages=7),
            ),
            _check_summarizing,
        ),
        build=lambda _: SummarizingCompaction(TestModel(), max_messages=6, keep_messages=2),
    ),
    'ComputerUse': CapabilityCase(
        Combines(
            'an agent has one display: the later configuration wins',
            # Both without the native tool, which reads the display's size, so each runs with no display.
            lambda: (ComputerUse[None](native=False, settle_seconds=1), ComputerUse[None](native=False)),
            _check_computer_use,
        ),
        # The fake screen never changes, so settling after an action would only wait out its interval.
        build=lambda _: [ScreenProvider().capability(), ComputerUse(settle_seconds=0)],
    ),
    'SystemReminders': CapabilityCase(
        Combines(
            'reminders are additive, so both sides keep firing',
            lambda: (
                SystemReminders[None](reminders=[Reminder('first', tag=None)]),
                SystemReminders[None](reminders=[Reminder('second', tag=None)]),
            ),
            _check_system_reminders,
        ),
        build=lambda _: SystemReminders(dynamic_reminders=[_time_reminder]),
    ),
    'CompactOnRequest': CapabilityCase(
        Combines(
            'one `compact_context` tool per agent, and two strategies for one request would compact twice',
            lambda: (
                CompactOnRequest[None](SlidingWindowCompaction[None](max_messages=1), max_per_run=1),
                CompactOnRequest[None](SlidingWindowCompaction[None](max_messages=1), max_per_run=2),
            ),
            _check_compact_on_request,
        ),
        build=lambda _: CompactOnRequest(SlidingWindowCompaction(max_messages=1, keep_messages=2)),
        calls=Calls(
            (('compact_context', {}),),
            lambda _: CompactOnRequest(SlidingWindowCompaction(max_messages=1, keep_messages=2)),
            # Temporal refuses its per-run toolset (`DURABLE_TEMPORAL` exempts it), so it records nothing there.
            durable_steps=DurableSteps(dbos=('capability__compact_on_request.compact',)),
            moves_cache_prefix=COMPACTION_REWRITES_HISTORY,
        ),
    ),
    'PruneToolResults': CapabilityCase(
        Combines(
            'one `prune_tool_result` tool per agent',
            lambda: (PruneToolResults[None](placeholder='[first]'), PruneToolResults[None](placeholder='[second]')),
            _check_prune_tool_results,
        ),
        build=lambda _: [PruneToolResults(), _PruneLatestReturn()],
        # The first is refused, as no previous step returned `missing`; the second prunes the first's result, which a
        # contract running one call per response gives the ID `call_0`.
        calls=Calls(
            (
                ('prune_tool_result', {'tool_call_id': 'missing', 'summary': 'Quiet.'}),
                ('prune_tool_result', {'tool_call_id': 'call_0', 'summary': 'Quiet.'}),
            ),
            lambda _: PruneToolResults(),
            durable_steps=DurableSteps(temporal=('toolset__prune_tool_results__call_tool',) * 2),
            moves_cache_prefix=COMPACTION_REWRITES_HISTORY,
        ),
    ),
    'ToolOutputLimits': CapabilityCase(
        Combines(
            'one output-limit policy; `tool_filter`/`per_tool` are how one instance varies by tool',
            lambda: (ToolOutputLimits[None](), ToolOutputLimits[None](strip_ansi=True)),
            _check_tool_output_limits,
        ),
        build=_workspace(ToolOutputLimits),
    ),
    'ToolBridge': CapabilityCase(
        Combines(
            'one set of dispatcher tools per agent; `tools` is how one instance picks what it bridges',
            lambda: (ToolBridge[None](max_results=3), ToolBridge[None](max_results=5)),
            _check_tool_bridge,
        )
    ),
    'SubAgents': CapabilityCase(
        Combines(
            'one delegate tool per agent, so two rosters become one',
            lambda: (
                SubAgents[None](agents=[SubAgent(Agent(TestModel(), name='alpha'))]),
                SubAgents[None](agents=[SubAgent(Agent(TestModel(), name='beta'))]),
            ),
            _check_sub_agents,
        )
    ),
    'Advisor': CapabilityCase(
        Combines(
            'one advisor per agent; its tool name is fixed',
            lambda: (
                Advisor[None]('anthropic:claude-fable-5', max_tokens=2048),
                Advisor[None]('anthropic:claude-fable-5', max_tokens=4096),
            ),
            _check_advisor,
        ),
        build=lambda _: Advisor(TestModel()),
        calls=Calls(
            (('advisor', {'prompt': 'q'}),),
            lambda _: Advisor(
                counting_model('advisor.model', lambda _: ModelResponse(parts=[TextPart('advice')])), native=False
            ),
            durable_steps=DurableSteps(
                dbos=('capability__advisor.consult',), temporal=('toolset__advisor__call_tool',)
            ),
        ),
    ),
    'SSHWorkspace': CapabilityCase(
        Combines(
            'the later configuration replaces the earlier one whole, so an `env` (and its secrets) never carries over',
            lambda: (
                SSHWorkspace[None]('first', env={'FIRST_SECRET': 'x'}, read_only=True),
                SSHWorkspace[None]('second'),
            ),
            _check_ssh_workspace,
        ),
        build=Unbuildable('Needs an SSH host to attach a workspace to.'),
    ),
    'BackgroundTools': CapabilityCase(
        Combines(
            'one background scheduler per agent; selectors combine without wrapping a tool twice',
            lambda: (BackgroundTools[None](tools=['first']), BackgroundTools[None](tools=['second'])),
            _check_background_tools,
        )
    ),
    'BenchmarkInstructions': CapabilityCase(
        Combines(
            'one fixed text; two would state it twice',
            lambda: (BenchmarkInstructions[None](), BenchmarkInstructions[None]()),
            _check_benchmark_instructions,
        )
    ),
    'AnnounceWorkingDirectory': CapabilityCase(
        Combines(
            'one fixed line; `Shell` and `FileSystem` each bring a copy, which must state it once',
            lambda: (AnnounceWorkingDirectory[None](), AnnounceWorkingDirectory[None]()),
            _check_working_directory,
        )
    ),
    'CurrentTime': CapabilityCase(
        Combines(
            'one statement of the time per id; the later configuration wins',
            lambda: (
                CurrentTime[None](timezone='Europe/Amsterdam'),
                CurrentTime[None](timezone='Asia/Tokyo', precision='hour'),
            ),
            _check_current_time,
        )
    ),
    'LabelCrossModelHistory': CapabilityCase(
        Combines(
            'one `notice` instruction block per agent; two would state it twice',
            lambda: (LabelCrossModelHistory[None](notice='first'), LabelCrossModelHistory[None](notice='second')),
            _check_label_cross_model_history,
        )
    ),
    # -- Several of these is the normal case, so they stay anonymous. --
    'RepairToolArguments': CapabilityCase(
        Anonymous('repairing valid arguments again is a no-op'),
        calls=Calls(ECHO, with_echo(RepairToolArguments), durable_steps=ECHO_STEPS),
    ),
    'ServingModelRecorder': CapabilityCase(Anonymous('one per sub-agent run, added by the delegating toolset')),
    '_Skill': CapabilityCase(
        Anonymous('one per skill, with the skill name as its id; `Skills` rejects two with one name')
    ),
    'RequireWorkspace': CapabilityCase(Anonymous('the same run-start check; two check the same thing twice')),
    '_BoundToolOutputs': CapabilityCase(Anonymous('Coder-local truncation composes with standalone output policies')),
    '_CoderGuidance': CapabilityCase(
        Rejected(
            'Coder-local guidance; its `coder_instructions` id is set per instance, so a second copy raises rather '
            'than merging'
        )
    ),
    '_DefaultCaching': CapabilityCase(Anonymous('Coder-local caching default; it yields to any `cache` already set')),
    '_SetToolStrict': CapabilityCase(
        Anonymous('Coder-local strict default; it only fills a `strict` still unset, so the first wins')
    ),
    '_ReportExhaustedRetries': CapabilityCase(
        Anonymous(
            'Coder-local retry reporting; whichever sees the exhausted retry turns it into a tool failure, and the '
            'other passes that on'
        )
    ),
    'Coder': CapabilityCase(
        Anonymous('a packaged harness; composing two is composing their members'),
        build=_workspace(Coder),
    ),
    'OpenSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='opensandbox'),
    'DockerContainer': CapabilityCase(Anonymous(_SANDBOX)),
    'DockerSandbox': CapabilityCase(Anonymous('deprecated alias of `DockerContainer`, and combines the same way')),
    'E2BSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='e2b'),
    'KubernetesSandbox': CapabilityCase(
        Anonymous(_SANDBOX),
        build=lambda _: _kubernetes_sandbox(),
        requires='k8s_agent_sandbox',
    ),
    'RenderSandbox': CapabilityCase(Anonymous(_SANDBOX)),
    'SpritesSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='sprites'),
    'DaytonaSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='daytona'),
    'BoxdSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='boxd'),
    'SmolSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='smol'),
    'BoxLiteSandbox': CapabilityCase(Anonymous(_SANDBOX), requires='boxlite'),
    'BubblewrappedWorkspace': CapabilityCase(
        Anonymous('a structural wrapper, applied once per wrapped workspace capability'),
        build=lambda tmp_path: BubblewrappedWorkspace(LocalWorkspace(tmp_path)),
    ),
    'ModalSandbox': CapabilityCase(
        Anonymous(_SANDBOX),
        build=lambda _: _modal_sandbox(),
        requires='modal',
    ),
    'Researcher': CapabilityCase(
        Anonymous('a packaged harness; composing two is composing their members'),
        build=_workspace(Researcher),
    ),
    'ClampOversizedMessages': CapabilityCase(
        Anonymous('clamping twice is a no-op; several thresholds compose'),
        build=lambda _: ClampOversizedMessages(max_part_chars=100_000),
    ),
    'DelegationReports': CapabilityCase(
        Anonymous('one per task owner and conversation; each delivers only its own reports, so several coexist'),
        build=lambda tmp_path: DelegationReports(DelegationTasks(directory=tmp_path / 'tasks'), conversation_id='c'),
    ),
    'TaskLink': CapabilityCase(Anonymous('the abstract base of the links below; never attached itself')),
    'LocalTaskLink': CapabilityCase(
        Anonymous('`SubAgents(task_control=TaskControl())` attaches one to each task run it starts')
    ),
    '_WorkerLink': CapabilityCase(Anonymous('`SubAgentWorker` attaches one to each task run it serves')),
    '_ForwardingWorkerLink': CapabilityCase(
        Anonymous('`SubAgentWorker` attaches one to each task run that forwards its events')
    ),
    'ClearToolResults': CapabilityCase(
        Anonymous('several form an escalation ladder, like `TieredCompaction` tiers'),
        build=lambda _: ClearToolResults(max_messages=6, keep_pairs=1),
    ),
    'DenyUnanswered': CapabilityCase(Anonymous('a second instance would deny what the first already denied')),
    'DeduplicateFileReads': CapabilityCase(
        Anonymous('file-read identification is agent-specific; one per `file_key`'),
        build=_deduplicate_file_reads,
        calls=Calls(
            (('read_file', {'path': 'notes.md'}), ('read_file', {'path': 'notes.md'})),
            _deduplicate_file_reads,
            durable_steps=DurableSteps(
                dbos=_workspace_steps(8),
                temporal=('capability__workspace__call', *('toolset__file_system__call_tool',) * 2),
            ),
            moves_cache_prefix=COMPACTION_REWRITES_HISTORY,
        ),
    ),
    'DynamicWorkflow': CapabilityCase(
        Anonymous('one per workflow definition'),
        build=Unbuildable('Runs sub-agents a caller supplies, whose requests are their own conversations.'),
    ),
    'FallbackCompaction': CapabilityCase(
        Anonymous('drives a fallback chain; several independent chains compose'),
        build=lambda _: FallbackCompaction(fallback_chain=[SlidingWindowCompaction(max_messages=6, keep_messages=2)]),
    ),
    'InputGuardrail': CapabilityCase(
        Anonymous('several guards is the design'),
        build=lambda _: InputGuardrail(guard=lambda text: True),
    ),
    'OutputGuardrail': CapabilityCase(
        Anonymous('several guards is the design'),
        build=lambda _: OutputGuardrail(guard=lambda output: True),
    ),
    'PromptInjectionDefender': CapabilityCase(
        Anonymous('one per `tool_filter`; several scopes compose'),
        requires='stackone_defender',
        # It declares durable operations and no default id, so a durable agent needs one given.
        calls=Calls(
            ECHO,
            with_echo(lambda: pydantic_ai_harness.PromptInjectionDefender(id='prompt_injection_defender')),
            durable_steps=ECHO_STEPS,
        ),
    ),
    'StreamGuardrail': CapabilityCase(
        Anonymous('several guards is the design'),
        build=lambda _: StreamGuardrail(guard=lambda part: True),
    ),
    'ToolGuardrail': CapabilityCase(
        Anonymous('several guards is the design'),
        # It declares durable operations and no default id, so a durable agent needs one given.
        calls=Calls(ECHO, with_echo(lambda: ToolGuardrail(id='tool_guardrail')), durable_steps=ECHO_STEPS),
    ),
    'ToolCallJudge': CapabilityCase(
        Anonymous('one per risk question; stacked judges must all allow a call'),
        build=lambda _: ToolCallJudge(TestModel(), question='Is this call safe?'),
    ),
    'ApprovalReviewer': CapabilityCase(
        Anonymous('one per risk question; each answers the approval requests the ones before it left'),
        build=lambda _: ApprovalReviewer(TestModel(), question='Is this safe?'),
    ),
    'Permissions': CapabilityCase(
        Anonymous('each mode holds the calls it asks about, so stacked ones ask when any of them would')
    ),
    'ManagedPrompt': CapabilityCase(
        Anonymous(
            'one per prompt name; the name derives its `id`, so two of one prompt are rejected when the agent is built'
        ),
        build=lambda _: ManagedPrompt('assistant', default='You answer questions about the news.'),
    ),
    'LogfireMCP': CapabilityCase(
        Narrows(
            'one Logfire connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(LogfireMCP),
        calls=_hosted_calls(LogfireMCP, 'logfire_search', 'logfire-mcp'),
    ),
    'RepoContext': CapabilityCase(Anonymous('one per workspace root'), build=_repo_context),
    'ReportContextUsage': CapabilityCase(Anonymous('a passive observer; several callbacks compose')),
    'ReportRemainingBudget': CapabilityCase(
        Anonymous('only a hook; each reports the run limits and its own `SpendLimits`')
    ),
    'ReportStaleFiles': CapabilityCase(Anonymous('only an event listener and a hook; two would send two notices')),
    'SlidingWindowCompaction': CapabilityCase(
        Anonymous('composes as a tier under `TieredCompaction`'),
        build=lambda _: SlidingWindowCompaction(max_messages=6, keep_messages=2),
    ),
    'GoogleWorkspace': CapabilityCase(
        Anonymous('one per set of products, and `services` is what names it'),
        build=Unbuildable('Needs Google OAuth credentials to list its tools.'),
    ),
    'StackOne': CapabilityCase(
        Anonymous('one per linked account, and `account_id` is what names it'),
        build=lambda _: StackOne('account', client=hosted_server()),
        calls=Calls(
            (('search', {}),),
            lambda _: StackOne('account', client=hosted_server()),
            durable_steps=_mcp_steps('stackone-account', instructions=False),
        ),
    ),
    'TieredCompaction': CapabilityCase(
        Anonymous('drives other strategies; one per tier list'),
        build=lambda _: TieredCompaction(tiers=[ClearToolResults(max_messages=6, keep_pairs=1)], target_tokens=10),
    ),
    'WarnNearLimits': CapabilityCase(
        Anonymous('a passive observer; several thresholds compose'),
        build=lambda _: WarnNearLimits(max_iterations=2),
    ),
    'WarnOnCacheBusts': CapabilityCase(Anonymous('a passive observer; several thresholds compose')),
    'TrajectoryJudge': CapabilityCase(
        Anonymous('each judge independently evaluates and steers the run'),
        build=lambda _: TrajectoryJudge(model=TestModel()),
    ),
    'DetectLoops': CapabilityCase(
        Anonymous('only hooks; several detectors with their own thresholds and actions compose')
    ),
    'ReportToolErrors': CapabilityCase(
        Anonymous('a wrapper toolset; the innermost (last listed) formats the error and outer ones see a ToolFailed'),
        calls=Calls(ECHO, with_echo(ReportToolErrors), durable_steps=ECHO_STEPS),
    ),
    'ToolBudgets': CapabilityCase(
        Anonymous('only a hook; each counts the calls it lets run, so stacked budgets all apply'),
        calls=Calls(ECHO, with_echo(ToolBudgets), durable_steps=ECHO_STEPS),
    ),
    'RefineOutput': CapabilityCase(
        Anonymous('only a hook; each counts its own rounds and adds its instructions to a round')
    ),
    'RequireVerification': CapabilityCase(
        Anonymous('only hooks; each keeps its own record of edits, checks, and attempts'),
        build=lambda tmp_path: [
            RequireVerification(verification_commands=[r'true\b']),
            FileSystem(),
            Shell(denied_commands=[]),
            LocalWorkspace(tmp_path),
        ],
    ),
    'WrapUpAtLimits': CapabilityCase(
        Anonymous('only hooks; several on one agent each add their prompt to the final request')
    ),
    'RenderWorkflows': CapabilityCase(
        Rejected('Only one durability capability can be attached to an agent.'),
        build=Unbuildable('Durable engine: needs a Render Workflows app.'),
    ),
    'AWSLambdaDurability': CapabilityCase(
        Rejected(_DURABILITY),
        build=Unbuildable('Durable engine: needs the AWS Lambda durable functions runtime.'),
    ),
    'AbsurdDurability': CapabilityCase(
        Rejected(_DURABILITY),
        build=Unbuildable('Durable engine: needs a Postgres-backed Absurd queue.'),
    ),
    'AzureDurableTaskDurability': CapabilityCase(
        Rejected('an agent accepts only one durable execution engine; core rejects a second before binding'),
        requires='durabletask',
    ),
    'PGTaskDurability': CapabilityCase(
        Rejected(_DURABILITY),
        build=Unbuildable('Durable engine: needs Postgres.'),
    ),
    # -- No default `id`, but two never coexist anyway: their tool names collide. --
    'FileSystem': CapabilityCase(
        Collides(
            'its toolset registers `read_file` and friends under fixed names',
            lambda cls: (cls(str(COMBINE_ROOT_A)), cls(str(COMBINE_ROOT_A.parent))),
        ),
        build=_workspace(FileSystem),
        calls=Calls(
            (
                ('write_file', {'path': 'notes.md', 'content': 'Sunny.\n'}),
                ('edit_file', {'path': 'notes.md', 'old_text': 'Sunny.', 'new_text': 'Rainy.'}),
                ('read_file', {'path': 'notes.md'}),
            ),
            _workspace(FileSystem),
            durable_steps=DurableSteps(
                dbos=_workspace_steps(17),
                # `write_file` and `edit_file` run in the workflow, so their workspace calls are activities.
                temporal=(*('capability__workspace__call',) * 14, 'toolset__file_system__call_tool'),
            ),
        ),
    ),
    'Goal': CapabilityCase(
        Narrows(
            'one goal per id; several goals each need their own id, and every completion check must accept',
            lambda cls: (cls(goal='Finish'), cls(goal='Also finish')),
        ),
        build=lambda _: Goal(goal='Report the weather.', verify=_goal_met),
    ),
    'AskUser': CapabilityCase(
        Narrows(
            'one `ask_user_question` tool per id, and two answerers is a conflict, not one configuration stated twice',
            lambda cls: (cls(answerer=_decline), cls(answerer=None)),
        ),
        build=lambda _: AskUser(answerer=None),
    ),
    'Peers': CapabilityCase(
        Narrows(
            'one peer network per id; merging two networks would widen who receives messages',
            lambda cls: (
                cls(network=InMemoryPeerNetwork().join(handle='a')),
                cls(network=InMemoryPeerNetwork().join(handle='b')),
            ),
        ),
        build=lambda _: Peers(network=InMemoryPeerNetwork().join(handle='a')),
    ),
    'Shell': CapabilityCase(
        Collides(
            'its toolset registers `run_command` and friends under fixed names',
            lambda cls: (cls(default_timeout=10.0), cls(default_timeout=20.0)),
        ),
        build=_workspace(Shell),
        calls=Calls(
            (('run_command', {'command': 'echo hi'}),),
            _workspace(Shell),
            durable_steps=DurableSteps(
                dbos=_workspace_steps(2), temporal=('capability__workspace__call', 'toolset__shell__call_tool')
            ),
        ),
    ),
    'CapabilityCreation': CapabilityCase(
        Narrows(
            'one store directory per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(str(COMBINE_ROOT_A)), cls(str(COMBINE_ROOT_B))),
        ),
        build=_capability_creation,
        calls=Calls(
            (
                ('author_capability', {'name': 'marker', 'code': _AUTHORED}),
                ('list_authored_capabilities', {}),
                ('disable_authored_capability', {'name': 'marker'}),
            ),
            _capability_creation,
            durable_steps=DurableSteps(
                dbos=(
                    'capability__workspace.call',
                    'capability__capability_creation.absolute_directory',
                    'capability__workspace.call',
                    'capability__workspace.call',
                    'capability__capability_creation.absolute_directory',
                    'capability__workspace.call',
                    'capability__workspace.call',
                    'capability__capability_creation.write',
                    'capability__capability_creation.absolute_directory',
                    'capability__workspace.call',
                    'capability__workspace.call',
                    'capability__capability_creation.list_all',
                    'capability__capability_creation.absolute_directory',
                    'capability__workspace.call',
                    'capability__workspace.call',
                    'capability__capability_creation.disable',
                ),
                # Its tools resolve the store directory and check it through the workspace in workflow code first.
                temporal=(
                    'capability__workspace__call',
                    'capability__capability_creation__absolute_directory',
                    'capability__workspace__call',
                    'capability__workspace__call',
                    'capability__capability_creation__absolute_directory',
                    'capability__workspace__call',
                    'capability__workspace__call',
                    'toolset__capability_creation__call_tool',
                    'capability__capability_creation__absolute_directory',
                    'capability__workspace__call',
                    'capability__workspace__call',
                    'toolset__capability_creation__call_tool',
                    'capability__capability_creation__absolute_directory',
                    'capability__workspace__call',
                    'capability__workspace__call',
                    'toolset__capability_creation__call_tool',
                ),
            ),
        ),
    ),
    'PydanticAIDocs': CapabilityCase(
        Narrows(
            'one `read_pyai_docs` tool per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(), cls(cache=False)),
        ),
        calls=Calls(
            (('read_pyai_docs', {'topic': 'hooks'}),),
            _pydantic_ai_docs,
            # Temporal refuses its id-less toolset (`DURABLE_TEMPORAL` exempts it), so it records nothing there.
            durable_steps=DurableSteps(dbos=_workspace_steps(2)),
        ),
    ),
    'Notion': CapabilityCase(
        Narrows(
            'one Notion connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Notion),
        calls=_hosted_calls(Notion, 'search', 'notion'),
    ),
    'PyaiDocs': CapabilityCase(
        Narrows(
            'deprecated alias of `PydanticAIDocs`, and combines the same way',
            lambda cls: (cls(), cls(cache=False)),
        )
    ),
    'Macroscope': CapabilityCase(
        Narrows(
            'one `run_macroscope_review` tool per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(timeout=10.0), cls(timeout=20.0)),
        ),
        build=_workspace(Macroscope),
        calls=Calls(
            (('run_macroscope_review', {}),),
            _macroscope,
            durable_steps=DurableSteps(
                dbos=_workspace_steps(2), temporal=('capability__workspace__call', 'toolset__macroscope__call_tool')
            ),
        ),
    ),
    'LocalStack': CapabilityCase(
        Narrows(
            'one LocalStack environment per id, and its service lists are an access boundary',
            lambda cls: (cls(allowed_services=['s3']), cls(allowed_services=['sqs'])),
        ),
        calls=Calls(
            (('aws_cli', {'command': 's3 ls'}),),
            lambda tmp_path: LocalStack(aws_cli_path=counting_aws_cli(tmp_path), include_instructions=False),
            durable_steps=DurableSteps(
                dbos=('capability__localstack.aws_cli',), temporal=('toolset__localstack__call_tool',)
            ),
        ),
    ),
    'Linear': CapabilityCase(
        Narrows(
            'one Linear connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Linear),
        calls=_hosted_calls(Linear, 'linear_search', 'linear'),
    ),
    'Ordinal': CapabilityCase(
        Narrows(
            'one Ordinal connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Ordinal),
        calls=_hosted_calls(Ordinal, 'search', 'ordinal'),
    ),
    'Grain': CapabilityCase(
        Narrows(
            'one Grain connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Grain),
        calls=_hosted_calls(Grain, 'grain_search', 'grain'),
    ),
    'DayAI': CapabilityCase(
        Narrows(
            'one DayAI connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(DayAI),
        calls=_hosted_calls(DayAI, 'day_ai_search', 'day_ai'),
    ),
    'PostHog': CapabilityCase(
        Narrows(
            'one PostHog connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(PostHog),
        calls=_hosted_calls(PostHog, 'posthog_search', 'posthog'),
    ),
    'Pylon': CapabilityCase(
        Narrows(
            'one Pylon connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Pylon),
        calls=_hosted_calls(Pylon, 'pylon_search', 'pylon'),
    ),
    'CodeMode': CapabilityCase(
        Narrows(
            'one `run_code` tool per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(max_retries=1), cls(max_retries=2)),
        )
    ),
    'IPythonCodeMode': CapabilityCase(
        Collides('`run_code` is reserved, so a second one is rejected by name'),
        # Keeps the scenario's tools native, so the conversation checks the `run_code` declaration without
        # launching a kernel process for every provider.
        build=lambda _: _ipython_code_mode(),
        requires='jupyter_client',
    ),
    'BrowserUse': CapabilityCase(
        Collides('its toolset registers its browser tools under fixed names'),
        requires='browser_use',
        # Temporal refuses its id-less toolset, and DBOS recovery repeats its browser agent's requests (`DURABLE_TEMPORAL`
        # and `DURABLE_DBOS` exempt it). Only the browser-use CI job installs
        # `browser_use`, so that job runs both durable suites for this case.
        calls=Calls((('browse_web', {'task': 'Find the price of the Pro plan.'}),), _browser_use),
    ),
    'SmartSearch': CapabilityCase(
        Narrows(
            'one `smart_search` tool per id, and its judge `model` decides which provider receives the source',
            lambda cls: (cls('test'), cls('other')),
        ),
        build=_workspace(lambda: SmartSearch(model=TestModel())),
        calls=Calls(
            (('smart_search', {'query': 'parse the config'}),),
            _smart_search,
            durable_steps=DurableSteps(
                dbos=(*_workspace_steps(6), 'capability__smart_search.judge'),
                temporal=('capability__workspace__call', 'toolset__smart_search__call_tool'),
            ),
        ),
    ),
    'PlaywrightBrowser': CapabilityCase(
        Narrows(
            'one browser per id, and its egress settings are an access boundary',
            lambda cls: (cls(), cls(allowed_domains=['example.com'])),
        ),
        requires='playwright',
    ),
    'CDPBrowser': CapabilityCase(
        Narrows(
            'one browser per id, and its commands and egress settings are an access boundary',
            lambda cls: (cls(), cls(allowed_methods=['Runtime.evaluate'])),
        ),
        build=Unbuildable('Launches Chrome when a run starts, which the offline suites have no binary for.'),
        requires='websockets',
    ),
    'ConversationSearch': CapabilityCase(
        Narrows(
            'one `search_conversation_history` tool per id, and its source and scope bound what it can read',
            lambda cls: (
                cls(SnapshotHistorySource(InMemoryStepStore()), scope='conversation'),
                cls(SnapshotHistorySource(InMemoryStepStore()), scope='all'),
            ),
        ),
        build=lambda _: ConversationSearch(SnapshotHistorySource(InMemoryStepStore()), scope='conversation'),
        calls=Calls(
            (('search_conversation_history', {'query': 'tabs'}),),
            lambda _: ConversationSearch(
                SnapshotHistorySource(InMemoryStepStore()), scope='all', add_instructions=False
            ),
            durable_steps=DurableSteps(
                dbos=('capability__conversation_search.search',),
                temporal=('toolset__conversation-search__call_tool',),
            ),
        ),
    ),
    'ExaAgent': CapabilityCase(
        Narrows(
            'one Exa agent configuration per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(effort=None), cls(system_prompt='Be thorough.')),
        ),
        requires='exa_py',
        env=_EXA_KEY,
        calls=Calls(
            (('exa_agent', {'query': 'q'}),),
            lambda _: pydantic_ai_harness.ExaAgent(runs=ExaAgentRuns()),
            durable_steps=DurableSteps(
                dbos=('capability__exa_agent.create_run', 'capability__exa_agent.resolve_run'),
                temporal=('toolset__exa_agent__call_tool', 'capability__exa_agent__resolve_run'),
            ),
        ),
    ),
    'Pixeltable': CapabilityCase(
        Narrows(
            'one Pixeltable catalog configuration per id, and its table allowlist is an access boundary',
            lambda cls: (cls(tables=['app']), cls(tables=['app.docs'])),
        ),
        build=lambda _: pydantic_ai_harness.Pixeltable(tables=['*']),
        requires='pixeltable',
    ),
    'ExaSearch': CapabilityCase(
        Narrows(
            'one Exa search configuration per id, and its domain lists are an access boundary',
            lambda cls: (cls(include_domains=['a.dev']), cls(include_domains=['b.dev'])),
        ),
        requires='exa_py',
        env=_EXA_KEY,
        calls=Calls(
            (
                ('web_search', {'query': 'q'}),
                ('get_page', {'url': 'https://a.dev'}),
                ('deep_search', {'question': 'q'}),
            ),
            lambda _: pydantic_ai_harness.ExaSearch(client=ExaClient(), include_deep_search=True),
            durable_steps=DurableSteps(
                dbos=(
                    'capability__exa_search.web_search',
                    'capability__exa_search.get_page',
                    'capability__exa_search.deep_search',
                ),
                temporal=('toolset__exa_search__call_tool',) * 3,
            ),
        ),
    ),
    'GitHub': CapabilityCase(
        Narrows(
            'one GitHub connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(GitHub),
        calls=_hosted_calls(GitHub, 'github_search', 'github'),
    ),
    'YouResearch': CapabilityCase(
        Narrows(
            'one You.com research configuration per id, and its domain lists are an access boundary',
            lambda cls: (cls(include_domains=['a.dev']), cls(include_domains=['b.dev'])),
        ),
        requires='youdotcom',
        env=_YOU_KEY,
        calls=Calls(
            (('answer', {'query': 'q'}), ('research', {'input': 'q'}), ('finance_research', {'input': 'q'})),
            lambda _: pydantic_ai_harness.YouResearch(client=YouClient()),
            durable_steps=DurableSteps(
                dbos=(
                    'capability__you_research.answer',
                    'capability__you_research.research',
                    'capability__you_research.finance_research',
                ),
                temporal=('toolset__you_research__call_tool',) * 3,
            ),
        ),
    ),
    'YouSearch': CapabilityCase(
        Narrows(
            'one You.com search configuration per id, and its domain lists are an access boundary',
            lambda cls: (cls(include_domains=['a.dev']), cls(include_domains=['b.dev'])),
        ),
        requires='youdotcom',
        env=_YOU_KEY,
        calls=Calls(
            (('web_search', {'query': 'q'}), ('get_page', {'url': 'https://a.dev'})),
            lambda _: pydantic_ai_harness.YouSearch(client=YouClient()),
            durable_steps=DurableSteps(
                dbos=('capability__you_search.web_search', 'capability__you_search.get_page'),
                temporal=('toolset__you_search__call_tool',) * 2,
            ),
        ),
    ),
    'KeenableSearch': CapabilityCase(
        Narrows(
            'one Keenable search configuration per id; two that differ need their own ids and PrefixTools',
            lambda cls: (cls(), cls(num_results=3)),
        ),
        calls=Calls(
            (('web_search', {'query': 'q'}), ('get_page', {'url': 'https://a.dev'})),
            lambda _: KeenableSearch(client=KeenableClient()),
            durable_steps=DurableSteps(
                dbos=('capability__keenable_search.web_search', 'capability__keenable_search.get_page'),
                temporal=('toolset__keenable_search__call_tool',) * 2,
            ),
        ),
    ),
    'Slack': CapabilityCase(
        Narrows(
            'one Slack connection per id; two that differ need their own ids and tool prefixes',
            lambda cls: (cls(auth='first-key'), cls(auth='second-key')),
        ),
        build=_hosted(Slack),
        calls=_hosted_calls(Slack, 'search', 'slack'),
    ),
}
"""Every Pydantic AI Harness capability, public or not, keyed by class name."""


@cache
def _hosted_mcp_capabilities() -> frozenset[str]:
    """The capabilities whose module connects through `hosted_toolset`, found by reading the source.

    Reads each module's file from the package root rather than importing it, so a module whose optional
    dependency group isn't installed is still read.
    """
    spec = importlib.util.find_spec('pydantic_ai_harness')
    assert spec is not None and spec.submodule_search_locations is not None
    root = Path(next(iter(spec.submodule_search_locations)))
    found: set[str] = set()
    for name, ref in discover_capabilities(HARNESS_CAPABILITY_PACKAGES).items():
        path = root.joinpath(*ref.module.split('.')[1:])
        source = (path / '__init__.py' if path.is_dir() else path.with_suffix('.py')).read_text(encoding='utf-8')
        # Only a module that mentions it is parsed, to keep collection cheap.
        if 'hosted_toolset' in source and any(
            isinstance(node, ast.Name) and node.id == 'hosted_toolset' for node in ast.walk(ast.parse(source))
        ):
            found.add(name)
    return frozenset(found)


HOSTED_MCP_CONNECTION = Contract('hosted_mcp_connection', applies=lambda name, _: name in _hosted_mcp_capabilities())
"""A hosted MCP integration connects through either `client` or `auth`, and names the connection by its own id.

`client` owns the connection, so passing `auth` with it raises, and `auth=''`, documented as unset, doesn't. It
applies to every capability whose module calls `hosted_toolset`, so a new integration is covered without a case
saying so.
"""

CONTRACTS = (HOSTED_MCP_CONNECTION,)
"""The contracts only Pydantic AI Harness capabilities are subject to."""
