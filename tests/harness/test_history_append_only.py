"""Every Harness capability keeps a conversation's requests append-only.

See `tests/append_only.py` for the scenario and the three invariants it checks, and
`tests/test_history_append_only.py` for the core capabilities. Every public capability class in
`pydantic_ai_harness` runs, constructed with its defaults or with the `build` of its case in
`tests/harness/capability_cases.py`, unless that `build` is `Unbuildable`, saying why it can't run offline.
Capabilities that keep state change it mid-conversation through `SCENARIOS`, and the clock moves an hour
between requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    HandleDeferredToolCalls,
    LocalWorkspace,
    ToolSearch,
    Toolset,
)
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import (
    CodeMode,
    FallbackCompaction,
    FileSystem,
    Peers,
    ReportStaleFiles,
    SummarizingCompaction,
    SystemReminders,
    UntrustedContentGate,
)
from pydantic_ai_harness.peers import InMemoryPeerNetwork, LocalPeerInbox, read_peer_message
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.report_stale_files import StaleFilesPart
from pydantic_ai_harness.sub_agents import (
    DelegationReports,
    DelegationTaskEvent,
    DelegationTasks,
    SubAgent,
    SubAgents,
    TaskOutputPart,
)
from pydantic_ai_harness.system_reminders import LLMReminder, Reminder
from pydantic_ai_harness.untrusted_content_gate import untrusted_content_reason

from ..append_only import (
    PROVIDERS,
    Allowed,
    ConversationResult,
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
from ..capability_cases import Unbuildable
from .capability_cases import CASES, HARNESS_CAPABILITY_PACKAGES


def _run_look_up(topic: str) -> tuple[str, dict[str, Any]]:
    return 'run_code', {'code': f'await look_up(topic={topic!r})'}


def _bridge_look_up(topic: str) -> tuple[str, dict[str, str]]:
    return 'tool_call', {'name': 'look_up', 'args_json': f'{{"topic": "{topic}"}}'}


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


def _open_forecasting(ctx: RunContext[Any]) -> str:
    """Open the forecasting capability."""
    ctx.load_capability('forecasting')
    return 'Opened.'


def _summarized_load(
    load: tuple[str, dict[str, Any]], *extra: AbstractCapability[Any]
) -> Callable[[Path], tuple[list[AbstractCapability[Any]], Scenario, Callable[[ConversationResult], None]]]:
    """Load `forecasting` and call `forecast` in the first run; the continued run compacts, then calls it again.

    A load the summary replaced is recorded again right after it as a `CapabilityLoadDeltaPart` and the reveal of its
    tools (#11770), whether the model or code loaded it, so the requests after the compaction only append, and the
    model's call to the tool the load revealed still runs.
    """
    scenario = Scenario(
        turns=(
            Turn(calls=(load,)),
            Turn(calls=(('forecast', {'city': 'Paris'}),)),
            Turn(text='Sunny in Paris.'),
            # The second run's second request compacts, and the model then calls the tool the load revealed.
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(calls=(('forecast', {'city': 'Brest'}),)),
            Turn(text='Sunny in Brest.'),
        ),
    )

    def still_loaded(result: ConversationResult) -> None:
        returns = [
            part.content
            for message in result.history
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'forecast'
        ]
        assert returns[-1] == 'Sunny in Brest.', f'the call after the compaction was refused: {returns[-1]!r}'

    def conversation(
        tmp_path: Path,
    ) -> tuple[list[AbstractCapability[Any]], Scenario, Callable[[ConversationResult], None]]:
        compaction = SummarizingCompaction(TestModel(), max_messages=8, keep_messages=2)
        return [compaction, _forecasting(), *extra], scenario, still_loaded

    return conversation


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


VARIANTS: dict[str, tuple[Fixture, Scenario]] = {
    # A lasting reminder is recorded on each firing and sent again on every later request, so it only ever appends.
    "SystemReminders(scope='conversation')": (
        lambda _: SystemReminders(
            reminders=[Reminder('Cite your sources.', interval=2, tag=None, scope='conversation')]
        ),
        Scenario(),
    ),
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
    # A discovery the summary replaced is recorded again as a reveal right after it (#11493), so the tool stays
    # callable without another search, and the requests after the compaction only append.
    'SummarizingCompaction+ToolSearch': (
        lambda _: [
            SummarizingCompaction(TestModel(), max_messages=8, keep_messages=2),
            ToolSearch(strategy='keywords'),
            _deferred_forecast_and_tides(),
        ],
        Scenario(
            turns=(
                Turn(calls=(('search_tools', {'queries': ['forecast']}),)),
                Turn(calls=(('forecast', {'city': 'Paris'}),)),
                Turn(text='Sunny in Paris.'),
                # The second run's second request compacts, and the model then calls the tool it found in the first.
                Turn(calls=(('look_up', {'topic': 'sports'}),)),
                Turn(calls=(('forecast', {'city': 'Brest'}),)),
                Turn(text='Sunny in Brest.'),
            ),
        ),
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
    'ToolBridge': Scenario(look_up=_bridge_look_up),
    'CompactOnRequest': Scenario(state_calls=(('compact_context', {'focus': 'the forecast'}),)),
    # A button pressed in one step stays held into the run's next step, which releases it.
    'ComputerUse': Scenario(
        turns=(
            Turn(calls=(('computer', {'actions': [{'type': 'mouse_down'}]}),)),
            Turn(calls=(('computer', {'actions': [{'type': 'mouse_up'}, {'type': 'screenshot'}]}),)),
            Turn(text='Clicked.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='Here is more.'),
        )
    ),
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
    # The first run's second step prunes the first step's result.
    'PruneToolResults': Scenario(state_calls=(('prune_tool_result', {'tool_call_id': '', 'summary': 'Quiet.'}),)),
    # Each run ends with a review round, which takes a turn of its own.
    'RefineOutput': Scenario(
        turns=(
            Turn(calls=(('look_up', {'topic': 'weather'}),)),
            Turn(calls=(('look_up', {'topic': 'news'}),)),
            Turn(text='Here is the weather and the news.'),
            Turn(text='Here is the weather and the news, checked.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='Here is more.'),
            Turn(text='Here is more, checked.'),
        )
    ),
    # The first run edits, is sent back to verify, and passes a check; the second makes no edit.
    'RequireVerification': Scenario(
        turns=(
            Turn(calls=(('write_file', {'path': 'weather.py', 'content': 'FORECAST = "sunny"\n'}),)),
            Turn(text='Saved the forecast.'),
            Turn(calls=(('run_command', {'command': 'true'}),)),
            Turn(text='Saved and verified the forecast.'),
            Turn(calls=(('look_up', {'topic': 'sports'}),)),
            Turn(text='Here is more.'),
        )
    ),
    'RepoContext': Scenario(
        state_calls=(
            ('edit_file', {'path': 'AGENTS.md', 'old_text': 'short', 'new_text': 'very short'}),
            ('write_file', {'path': 'docs/AGENTS.md', 'content': 'Docs are in Markdown.\n'}),
        )
    ),
}
"""Conversations that change a capability's state, or call tools the way the capability exposes them."""

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
    'PruneToolResults': Allowed(
        'Replaces the tool result the model pruned, once per accepted `prune_tool_result` call',
        (('messages', 'tool-return'),),
    ),
    # With a 20-token trigger every request is over it, so each request with an older step outside the
    # one-token tail judges and rewrites it: two in this conversation.
    'RelevanceCompaction': Allowed(_COMPACTION, _HISTORY, max_moves=2),
    'SlidingWindowCompaction': Allowed(_COMPACTION, _HISTORY),
    # The compaction, which removes the load's reveal of `forecast` (sent as a system message) with the rest, and on
    # Chat Completions, which has no deferred-tool channel, the first run's reveal adding `forecast` to `tools`
    # (docs/tools-advanced.md). The load recorded again after the summary, and the statement of the capability's
    # instructions that follows it, must move nothing more. Only Chat Completions needs the second move, but
    # `Allowed` can't vary `max_moves` by provider.
    **{
        name: Allowed(
            f'{_COMPACTION}; on Chat Completions a load also adds its tools to `tools`, as documented',
            (*_HISTORY, ('messages', 'system-prompt'), ('tools', 'tool:forecast')),
            max_moves=2,
        )
        for name in (
            'SummarizingCompaction+Capability(defer_loading=True)',
            'SummarizingCompaction+RunContext.load_capability',
        )
    },
    'SummarizingCompaction': Allowed(_COMPACTION, _HISTORY),
    # The compaction, and on Chat Completions, which has no deferred-tool channel, the first run's reveal adding
    # `forecast` to `tools` (docs/tools-advanced.md). The reveal recorded after the summary must move nothing more.
    # Only Chat Completions needs the second move: Anthropic and OpenAI Responses move once, for the compaction,
    # but `Allowed` can't vary `max_moves` by provider.
    'SummarizingCompaction+ToolSearch': Allowed(
        f'{_COMPACTION}; on Chat Completions a local reveal also adds the tool to `tools`, as documented',
        (*_HISTORY, ('tools', 'tool:search_tools')),
        max_moves=2,
    ),
    # With a 10-token target every request is over it, so the tier clears each tool result as it ages out of
    # `keep_pairs`: one on the third request and one on the fifth.
    'TieredCompaction': Allowed(_COMPACTION, _HISTORY, max_moves=2),
    'WarnNearLimits': Allowed(
        f'Request-scoped system prompt ({_PULLS}10561): sent for one request, then left out. Anthropic clears it '
        'natively with `clear_at`; elsewhere it is a request-only tail',
        (('messages', 'system-prompt'),),
        tail=True,
        records=('request-scoped system prompt',),
        providers=('openai-responses', 'openai-chat'),
    ),
    'ReportRemainingBudget': Allowed(
        f'Request-scoped system prompt ({_ISSUES}9019): sent for one request, then left out. Anthropic clears it '
        'natively with `clear_at`; elsewhere it is a request-only tail',
        (('messages', 'system-prompt'),),
        tail=True,
        records=('request-scoped system prompt',),
        providers=('openai-responses', 'openai-chat'),
    ),
    'ReportStaleFiles': Allowed(
        'Request-scoped stale-file notice: sent for one request, then left out. Anthropic clears it natively with '
        '`clear_at`; elsewhere it is a request-only tail',
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

CAPABILITIES = discover_capabilities(HARNESS_CAPABILITY_PACKAGES)


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
        # Delivered as a typed `<task_output>` capability part, which survives serialization as one.
        (part,) = parts
        assert isinstance(part, TaskOutputPart), f'expected a task report, found {part!r}'
        assert part.content == _REPORT

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


_PEER_NOTE = 'The deploy finished.'


def _peer_message(
    tmp_path: Path,
) -> tuple[list[AbstractCapability[Any]], Scenario, Callable[[ConversationResult], None]]:
    """A peer message that arrives between the runs enters the second run's history once, as peer data."""
    network = InMemoryPeerNetwork()
    inbox = LocalPeerInbox(path=tmp_path / 'inbox.json')
    view = network.join(handle='self', inbox=inbox)
    sender = network.join(handle='ops')

    async def send() -> None:
        (recipient,) = await sender.peers()
        await sender.send(recipient.id, _PEER_NOTE, key='ops:1')

    def delivered_once(result: ConversationResult) -> None:
        parts = [
            part for message in result.history for part in getattr(message, 'parts', ()) if _PEER_NOTE in str(part)
        ]
        assert len(parts) == 1, f'expected the peer message in history once, found {len(parts)}'
        assert read_peer_message(parts[0]) is not None, f'expected a peer message, found {parts[0]!r}'

    return [Peers(network=view, inbox=inbox)], Scenario(between_runs=send), delivered_once


def _stale_files(
    tmp_path: Path,
) -> tuple[list[AbstractCapability[None]], Scenario, Callable[[ConversationResult], None]]:
    """A file the model read and edited changes from outside between the runs, so the second run gets notices."""
    notes = tmp_path / 'notes.md'
    notes.write_text('Sunny.\n')

    async def change() -> None:
        notes.write_text('Snowy.\n')

    def notified(result: ConversationResult) -> None:
        parts = [
            part
            for message in result.history
            for part in getattr(message, 'parts', ())
            if isinstance(part, StaleFilesPart)
        ]
        assert parts, 'expected a stale-file notice in the second run'

    scenario = Scenario(
        state_calls=(
            ('read_file', {'path': 'notes.md'}),
            ('edit_file', {'path': 'notes.md', 'old_text': 'Sunny.', 'new_text': 'Rainy.'}),
        ),
        between_runs=change,
    )
    return [ReportStaleFiles(), FileSystem(), LocalWorkspace(tmp_path)], scenario, notified


def _gated_and_approved(
    _: Path,
) -> tuple[list[AbstractCapability[None]], Scenario, Callable[[ConversationResult], None]]:
    """Once a `look_up` result taints the run, the gate holds the next `look_up`, and a handler approves it in the run."""
    asked: list[str | None] = []

    def approve(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        asked.extend(untrusted_content_reason(requests.metadata.get(call.tool_call_id)) for call in requests.approvals)
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def held_and_approved(result: ConversationResult) -> None:
        assert asked, 'expected the gate to hold a call after the first `look_up` result'
        assert all(asked), f"expected every held call to carry the gate's reason, found {asked}"

    return [UntrustedContentGate(), HandleDeferredToolCalls(approve)], Scenario(), held_and_approved


CONVERSATIONS = {
    'Peers': _peer_message,
    'ReportStaleFiles': _stale_files,
    'DelegationReports': _delegation_reports,
    'UntrustedContentGate': _gated_and_approved,
    'SummarizingCompaction+Capability(defer_loading=True)': _summarized_load(
        ('load_capability', {'id': 'forecasting'})
    ),
    'SummarizingCompaction+RunContext.load_capability': _summarized_load(
        ('open_forecasting', {}), Toolset(FunctionToolset([Tool(_open_forecasting, name='open_forecasting')]))
    ),
}
"""Capabilities whose behavior needs a conversation of its own, with the check it must also pass."""


UNBUILDABLE = {name for name, case in CASES.items() if isinstance(case.build, Unbuildable)}


def test_every_table_entry_names_a_capability():
    """Every table entry names a capability that still exists, so renames and removals don't leave dead rows."""
    tables = SCENARIOS.keys() | KNOWN.keys() | ALLOWED.keys()
    stale = tables - CAPABILITIES.keys() - VARIANTS.keys() - CONVERSATIONS.keys()
    assert not stale, f'entries for capabilities that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('name', sorted(CAPABILITIES.keys() - UNBUILDABLE))
def test_only_coder_repairs_tool_arguments_by_default(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`Coder` is the one capability that turns `RepairToolArguments` on; every other one leaves arguments as sent."""
    case = CASES[name]
    skip_unless_installed(name, case.requires)
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)
    if name in CONVERSATIONS:
        capabilities = CONVERSATIONS[name](tmp_path)[0]
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name].load, CASES[name].fixture, tmp_path)
    leaves = [leaf for capability in capabilities for leaf in leaf_capabilities(capability)]
    repairs = any(isinstance(leaf, RepairToolArguments) for leaf in leaves)
    assert repairs == (name in {'Coder', 'RepairToolArguments'})


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize(
    'name',
    sorted(CAPABILITIES.keys() - UNBUILDABLE) + sorted(VARIANTS.keys() | CONVERSATIONS.keys() - CAPABILITIES.keys()),
)
async def test_conversation_only_appends(
    name: str, provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    if (case := CASES.get(name)) is not None:
        skip_unless_installed(name, case.requires)
        for key, value in case.env.items():
            monkeypatch.setenv(key, value)
    check: Callable[[ConversationResult], None] | None = None
    if name in CONVERSATIONS:
        capabilities, scenario, check = CONVERSATIONS[name](tmp_path)
    elif name in VARIANTS:
        fixture, scenario = VARIANTS[name]
        capabilities = build_capabilities(name, lambda: AbstractCapability, fixture, tmp_path)
    else:
        capabilities = build_capabilities(name, CAPABILITIES[name].load, CASES[name].fixture, tmp_path)
        scenario = SCENARIOS.get(name, Scenario())
    clock = inject_clock(monkeypatch)
    result = await run_conversation(provider, capabilities, scenario, clock)
    if check is not None:
        check(result)
    check_conversation(name, result, known=KNOWN.get(name, ()), allowed=ALLOWED.get(name))
