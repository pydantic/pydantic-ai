"""Capabilities whose tools make requests record them under durable execution, instead of repeating them.

DBOS runs function tools in workflow code, so a capability's tool I/O is only recorded when the
capability routes it through a `durable_operation`. Each scenario runs one capability's tools in a
DBOS workflow, then re-executes the workflow from after its last recorded step, the way recovering a
committed run does, and checks that no request was made again. Planning's tools also run with an
event stream handler, whose steps for the events a tool emits must replay in their recorded order.
Shell's `shell` tool names its job in workflow code, so a recovered workflow must report the job it started.

The other engines run function tools in their own durable unit, which needs the toolset's `id`; the
construction tests below check every capability supplies one.
"""

from __future__ import annotations

import asyncio
import json
import stat
import tempfile
import uuid
from collections import Counter
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, AgentStreamEvent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ValidatedToolArgs, durable_operation
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.capability_creation import CapabilityCreation, CapabilityStore
from pydantic_ai_harness.localstack import LocalStack
from pydantic_ai_harness.memory import InMemoryStore, Memory
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanCompletedEvent,
    PlanItem,
    Planning,
    PlanStatusChangedEvent,
    SqlitePlanStore,
)
from pydantic_ai_harness.shell import Shell
from tests.conftest import detach_dbos_logging
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED

# Module scope builds the agents and registers their DBOS workflows, which DBOS requires before launch.
try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from exa_py.agent.types import AgentEffort, AgentOutput, AgentRun
    from exa_py.api import (
        ContentsOptions,
        DeepOutputSchema,
        DeepSearchOutput,
        Result,
        SearchResponse,
        SearchType,
        TextContentsOptions,
    )
    from youdotcom import models

    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai_harness.exa import ExaAgent, ExaSearch
    from pydantic_ai_harness.youdotcom import YouResearch, YouSearch
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos, prefect, temporalio, exa-py or youdotcom not installed', allow_module_level=True)

_requests: Counter[str] = Counter()
"""Requests each scenario's fakes received, by request name."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_tool_io',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


def _call_tools(*calls: tuple[str, dict[str, Any]]) -> FunctionModel:
    """A model that calls every tool in `calls` in one response, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if len([message for message in messages if isinstance(message, ModelRequest)]) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=name) for name, args in calls])

    return FunctionModel(respond)


class _ExaClient:
    async def search(
        self,
        query: str,
        *,
        contents: ContentsOptions | Literal[False],
        num_results: int | None = None,
        type: SearchType | None = None,
        output_schema: DeepOutputSchema | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse[Result]:
        _requests[f'exa.search.{type}'] += 1
        output = DeepSearchOutput(content='deep answer', grounding=[]) if type == 'deep' else None
        result = Result(url='https://a.dev', id='a', title='A', highlights=['excerpt'])
        return SearchResponse(results=[result], resolved_search_type=None, auto_date=None, output=output)

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        _requests['exa.get_contents'] += 1
        result = Result(url=urls, id=urls, title='A', text='page text')
        return SearchResponse(results=[result], resolved_search_type=None, auto_date=None)


class _ExaAgentRuns:
    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | type[BaseModel] | None = None,
        effort: AgentEffort | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun | Any:
        _requests['exa_agent.create'] += 1
        return AgentRun(id='run_1', status='queued')

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        _requests['exa_agent.poll'] += 1
        return AgentRun(id=run_id, status='completed', output=AgentOutput(text='Done.'))


class _YouClient:
    async def search_async(
        self,
        *,
        query: str,
        count: int | None = None,
        freshness: str | None = None,
        country: str | None = None,
        extraction: models.Extraction | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.SearchResponse:
        _requests['you.search'] += 1
        return models.SearchResponse()

    async def contents_async(
        self,
        *,
        urls: Sequence[str] | None = None,
        formats: Sequence[models.ContentsFormats] | None = None,
    ) -> list[models.ContentsResponse]:
        _requests['you.contents'] += 1
        return [models.ContentsResponse(url='https://a.dev', title='A', markdown='body')]

    async def answer_async(
        self,
        *,
        query: str,
        freshness: str | None = None,
        country: str | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.AnswerResponse:
        _requests['you.answer'] += 1
        return models.AnswerResponse(answer='the answer')

    async def research_async(
        self,
        *,
        input: str,
        research_effort: models.ResearchEffort | None = None,
        background: bool | None = None,
        source_control: models.SourceControl | Mapping[str, object] | None = None,
        output_schema: Mapping[str, object] | None = None,
    ) -> models.ResearchResult:
        _requests['you.research'] += 1
        # An empty answer makes the tool ask the model to retry, which the operation records as data.
        return models.ResearchResponse(
            output=models.Output(content='', content_type=models.ContentType.TEXT, sources=[]), warnings=None
        )

    async def finance_research_async(
        self,
        *,
        input: str,
        research_effort: models.FinanceResearchEffort | None = None,
    ) -> models.FinanceResearchResponse:
        _requests['you.finance_research'] += 1
        return models.FinanceResearchResponse(
            output=models.FinanceResearchOutput(
                content='finance answer', content_type=models.FinanceResearchContentType.TEXT, sources=[]
            )
        )


def _counting_aws_cli() -> str:
    """An `aws` stand-in that counts its runs in the file named by `HARNESS_AWS_CLI_COUNT`."""
    directory = Path(tempfile.mkdtemp(prefix='harness_durable_tool_io_'))
    script = directory / 'aws'
    script.write_text('#!/bin/sh\necho run >> "$HARNESS_AWS_CLI_COUNT"\necho ok\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


_CREATION_DIRECTORY = Path(tempfile.mkdtemp(prefix='harness_durable_tool_io_creation_'))
_PLAN_DATABASE = Path(tempfile.mkdtemp(prefix='harness_durable_tool_io_plan_')) / 'plan.db'
_AUTHORED = """
from pydantic_ai.capabilities import AbstractCapability


class Marker(AbstractCapability):
    def get_instructions(self):
        return 'marker'
"""


def _agent(name: str, capability: AbstractCapability[None], *calls: tuple[str, dict[str, Any]]) -> Agent[None, str]:
    extra: list[AbstractCapability[None]] = []
    if isinstance(capability, CapabilityCreation):
        extra.append(LocalWorkspace[None](str(_CREATION_DIRECTORY)))
    return Agent(
        _call_tools(*calls),
        name=name,
        deps_type=type(None),
        capabilities=[*extra, capability, DBOSDurability[None]()],
    )


_AGENTS: dict[str, Agent[None, str]] = {
    'exa_search': _agent(
        'exa_search_agent',
        ExaSearch[None](client=_ExaClient(), include_deep_search=True),
        ('web_search', {'query': 'q'}),
        ('get_page', {'url': 'https://a.dev'}),
        ('deep_search', {'question': 'q'}),
    ),
    'exa_agent': _agent('exa_agent_agent', ExaAgent[None](runs=_ExaAgentRuns()), ('exa_agent', {'query': 'q'})),
    'you_search': _agent(
        'you_search_agent',
        YouSearch[None](client=_YouClient()),
        ('web_search', {'query': 'q'}),
        ('get_page', {'url': 'https://a.dev'}),
    ),
    'you_research': _agent(
        'you_research_agent',
        YouResearch[None](client=_YouClient()),
        ('answer', {'query': 'q'}),
        ('research', {'input': 'q'}),
        ('finance_research', {'input': 'q'}),
    ),
    'localstack': _agent(
        'localstack_agent',
        LocalStack[None](aws_cli_path=_counting_aws_cli(), include_instructions=False),
        ('aws_cli', {'command': 's3 ls'}),
    ),
    'capability_creation': _agent(
        'capability_creation_agent',
        CapabilityCreation[None](directory=_CREATION_DIRECTORY / 'authored'),
        ('author_capability', {'name': 'marker', 'code': _AUTHORED}),
        ('list_authored_capabilities', {}),
        ('disable_authored_capability', {'name': 'marker'}),
    ),
    'memory': _agent(
        'memory_agent',
        Memory[None](store=InMemoryStore()),
        ('write_memory', {'content': 'The user prefers tabs.', 'file': 'style.md'}),
        # A missing file asks the model to retry, which the operation records as data.
        ('read_memory', {'file': 'missing.md'}),
        ('search_memory', {'query': 'tabs'}),
        ('delete_memory', {'file': 'missing.md'}),
    ),
    'planning': _agent(
        'planning_agent',
        Planning[None](store=SqlitePlanStore(database=str(_PLAN_DATABASE))),
        ('write_plan', {'items': [{'id': 'first', 'content': 'Write the migration'}]}),
        ('add_task', {'content': 'Run the tests'}),
    ),
}


def _workflow(agent: Agent[None, str]) -> Callable[[], Awaitable[str]]:
    @DBOS.workflow(name=f'{agent.name}_workflow')
    async def run() -> str:
        return (await agent.run('go')).output

    return run


_WORKFLOWS = {key: _workflow(agent) for key, agent in _AGENTS.items()}


@pytest.fixture
def count_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], Counter[str]]:
    """Count every request the scenarios' fakes, the `aws` stand-in, and the authoring store make."""
    _requests.clear()
    aws_count = tmp_path / 'aws_cli_count'
    monkeypatch.setenv('HARNESS_AWS_CLI_COUNT', str(aws_count))

    for method in ('write', 'list_all', 'disable'):
        original = getattr(CapabilityStore, method)

        def counted(self: CapabilityStore, *args: Any, _original: Any = original, _method: str = method) -> Any:
            _requests[f'store.{_method}'] += 1
            return _original(self, *args)

        monkeypatch.setattr(CapabilityStore, method, counted)

    # A memory write first looks up its idempotency key; the write is what recovery must not repeat.
    original_memory_write = InMemoryStore.write

    async def counted_memory_write(self: InMemoryStore, *args: Any, **kwargs: Any) -> Any:
        _requests['memory.write'] += 1
        return await original_memory_write(self, *args, **kwargs)

    monkeypatch.setattr(InMemoryStore, 'write', counted_memory_write)

    # The plan tools read the store many times; a write is what recovery must not repeat.
    for method in ('set_items', 'add_item'):
        original = getattr(SqlitePlanStore, method)

        async def counted_plan(
            self: SqlitePlanStore, *args: Any, _original: Any = original, _method: str = method
        ) -> Any:
            _requests[f'plan.{_method}'] += 1
            return await _original(self, *args)

        monkeypatch.setattr(SqlitePlanStore, method, counted_plan)

    def snapshot() -> Counter[str]:
        counts = Counter(_requests)
        if aws_count.exists():
            counts['aws_cli'] = len(aws_count.read_text().splitlines())
        return counts

    return snapshot


_EXPECTED_STEPS: dict[str, Collection[str]] = {
    'exa_search': ['exa_search.web_search', 'exa_search.get_page', 'exa_search.deep_search'],
    'exa_agent': ['exa_agent.create_run', 'exa_agent.resolve_run'],
    'you_search': ['you_search.web_search', 'you_search.get_page'],
    'you_research': ['you_research.answer', 'you_research.research', 'you_research.finance_research'],
    'localstack': ['localstack.aws_cli'],
    'capability_creation': [
        'capability_creation.write',
        'capability_creation.list_all',
        'capability_creation.disable',
    ],
    'memory': ['memory.write_memory', 'memory.read_memory', 'memory.search_memory', 'memory.delete_memory'],
    'planning': ['planning.set_items', 'planning.add_item'],
}


@pytest.mark.parametrize('scenario', sorted(_AGENTS))
async def test_dbos_recovery_does_not_repeat_tool_requests(
    scenario: str, dbos: DBOS, count_requests: Callable[[], Counter[str]]
) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await _WORKFLOWS[scenario]() == 'done'
    first_run = count_requests()
    assert first_run, 'the scenario made no requests to count'
    assert set(first_run.values()) == {1}

    steps = await dbos.list_workflow_steps_async(workflow_id)
    # Re-execute the workflow function from after its last step, the way recovery does. Calling it
    # again under the same ID is no replay: DBOS returns the stored result without running it.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    # The default 1s result poll would add up to a second on top of the internal queue's own poll.
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'
    assert count_requests() == first_run, 'recovery made a request again'

    step_names = {step['function_name'] for step in steps}
    agent_name = _AGENTS[scenario].name
    for operation in _EXPECTED_STEPS[scenario]:
        capability_id, name = operation.split('.')
        assert f'{agent_name}__capability__{capability_id}.{name}' in step_names


_SHELL_DIRECTORY = Path(tempfile.mkdtemp(prefix='harness_durable_tool_io_shell_'))
_SHELL_AGENT = Agent(
    _call_tools(('shell', {'command': 'echo one >> side-effects.txt', 'mode': 'background'})),
    name='shell_agent',
    deps_type=type(None),
    capabilities=[LocalWorkspace[None](str(_SHELL_DIRECTORY)), Shell[None](tools=['shell']), DBOSDurability[None]()],
)
_shell_returns: list[str] = []


class _ProcessCrash(BaseException):
    """Stands in for the process dying: DBOS leaves the workflow `PENDING` for recovery."""


@DBOS.workflow(name='shell_agent_workflow')
async def _shell_workflow() -> str:
    result = await _SHELL_AGENT.run('go')
    _shell_returns.extend(
        str(part.content)
        for message in result.new_messages()
        for part in message.parts
        if part.part_kind == 'tool-return'
    )
    if len(_shell_returns) == 1:
        raise _ProcessCrash
    return result.output


async def test_dbos_recovery_reports_the_same_shell_job(dbos: DBOS) -> None:
    """The `shell` tool runs in workflow code, so a recovered workflow must name the job its replayed launch started."""
    _shell_returns.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_ProcessCrash):
        await _shell_workflow()

    # Resuming the pending workflow re-executes it under the same ID, the way recovery does.
    handle = await DBOS.resume_workflow_async(workflow_id)
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'
    first, recovered = _shell_returns
    assert recovered == first
    status = next(line for line in first.splitlines() if line.startswith('Status: '))
    assert Path(status.removeprefix('Status: ')).exists()
    assert (_SHELL_DIRECTORY / 'side-effects.txt').read_text().splitlines() == ['one']


_READINESS_WAIT_TIMEOUT = 5.0


class _SlowReadPlanStore(InMemoryPlanStore):
    """A plan store whose `get_item` waits, once a plan tool has emitted `last_event`, until it's handled.

    It stands in for a read slow enough that the event stream handler's steps for the tool's events
    all start while it is in flight. A replay returns the recorded read without calling the store.
    The status tools read before emitting, so only a regression makes the wait.
    """

    def __init__(self, last_event: tuple[type[PlanStatusChangedEvent | PlanCompletedEvent], str]) -> None:
        super().__init__()
        self.last_event = last_event
        self.released = asyncio.Event()
        """Cleared when the tool emits `last_event` and set again once the handler has handled it."""
        self.released.set()

    def is_last_event(self, event: object) -> bool:
        event_type, task_id = self.last_event
        return isinstance(event, event_type) and event.item.id == task_id

    async def get_item(self, item_id: str) -> PlanItem | None:
        with anyio.fail_after(_READINESS_WAIT_TIMEOUT):
            await self.released.wait()
        return await super().get_item(item_id)


class _AuditToolCalls(AbstractCapability[None]):
    """Makes one durable call after each tool call, as an auditing capability would."""

    id = 'audit'

    @durable_operation('record')
    async def record(self, tool_name: str) -> None:
        pass

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        await self.record(call.tool_name)
        return result


def _plan_event_agent(
    name: str,
    update: tuple[str, dict[str, Any]],
    last_event: tuple[type[PlanStatusChangedEvent | PlanCompletedEvent], str],
    *extra: AbstractCapability[None],
) -> tuple[Agent[None, str], _SlowReadPlanStore]:
    """An agent that writes a two-step plan, makes the `update` plan tool call, then answers."""
    calls = [('write_plan', {'items': [{'id': 'a', 'content': 'Survey'}, {'id': 'b', 'content': 'Draft'}]}), update]

    # An `event_stream_handler` makes the run stream its model responses.
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        del info
        turn = len([message for message in messages if isinstance(message, ModelRequest)]) - 1
        if turn < len(calls):
            tool_name, args = calls[turn]
            yield {0: DeltaToolCall(name=tool_name, json_args=json.dumps(args), tool_call_id=tool_name)}
        else:
            yield 'done'

    store = _SlowReadPlanStore(last_event)

    async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if store.is_last_event(event):
                store.released.set()

    agent = Agent(
        FunctionModel(stream_function=stream),
        name=name,
        deps_type=type(None),
        capabilities=[Planning[None](store=store), *extra, DBOSDurability[None](event_stream_handler=handle)],
    )
    return agent, store


_PLAN_EVENT_AGENTS: dict[str, tuple[Agent[None, str], _SlowReadPlanStore]] = {
    'update_task_statuses': _plan_event_agent(
        'plan_statuses_agent',
        (
            'update_task_statuses',
            {'updates': [{'task_id': 'a', 'status': 'completed'}, {'task_id': 'b', 'status': 'in_progress'}]},
        ),
        (PlanStatusChangedEvent, 'b'),
    ),
    # A single update reads one step back, so another capability's hook supplies the second durable
    # call that could follow the tool's events.
    'update_task_status': _plan_event_agent(
        'plan_status_agent',
        ('update_task_status', {'task_id': 'a', 'status': 'completed'}),
        (PlanCompletedEvent, 'a'),
        _AuditToolCalls(),
    ),
}
_PLAN_EVENT_WORKFLOWS = {key: _workflow(agent) for key, (agent, _) in _PLAN_EVENT_AGENTS.items()}


@pytest.mark.parametrize('scenario', sorted(_PLAN_EVENT_AGENTS))
@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_dbos_recovery_replays_plan_tool_events_in_recorded_order(
    scenario: str, dbos: DBOS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan tool emits its events after its last store call, so their handler steps keep their order.

    DBOS runs a function tool in workflow code and starts a handler step for each event the tool
    emits while the tool is still running. Both draw step numbers from one counter. The first
    durable call the tool call makes after emitting is numbered before any handler step, but a
    second one is numbered after however many handler steps fit into the first one's I/O: a store
    round trip live, a journal lookup on replay.
    """
    _, store = _PLAN_EVENT_AGENTS[scenario]
    # A replay re-emits `last_event` without rerunning the handler, so a previous run can leave it closed.
    store.released.set()
    emit = RunContext.emit
    last_event_emits = 0

    async def emit_marking_last_event(self: RunContext[None], event: CapabilityEvent, /) -> CapabilityEvent:
        nonlocal last_event_emits
        if store.is_last_event(event):
            last_event_emits += 1
            store.released.clear()
        return await emit(self, event)

    monkeypatch.setattr(RunContext, 'emit', emit_marking_last_event)
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await _PLAN_EVENT_WORKFLOWS[scenario]() == 'done'
    # The status tools read before emitting, so the wait never engages: pin both sides of the
    # scaffolding before the fork's replay re-emits `last_event` without rerunning the handler.
    assert last_event_emits == 1, 'the tool should emit `last_event` once through the patched `emit`'
    assert store.released.is_set(), 'the handler never handled `last_event`'

    steps = await dbos.list_workflow_steps_async(workflow_id)
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'


@pytest.mark.parametrize('durability', [TemporalDurability, PrefectDurability])
@pytest.mark.parametrize(
    'capability',
    [
        pytest.param(lambda: ExaSearch[None](client=_ExaClient()), id='exa_search'),
        pytest.param(lambda: ExaAgent[None](runs=_ExaAgentRuns()), id='exa_agent'),
        pytest.param(lambda: YouSearch[None](client=_YouClient()), id='you_search'),
        pytest.param(lambda: YouResearch[None](client=_YouClient()), id='you_research'),
        pytest.param(lambda: LocalStack[None](), id='localstack'),
        pytest.param(lambda: CapabilityCreation[None](directory=_CREATION_DIRECTORY), id='capability_creation'),
    ],
)
def test_toolset_can_be_wrapped_by_engines_that_wrap_function_tools(
    capability: Callable[[], AbstractCapability[None]], durability: type[TemporalDurability | PrefectDurability]
) -> None:
    """Temporal and Prefect run each function tool in its own unit, named after the toolset's `id`."""
    Agent(
        _call_tools(),
        name=f'wrapped_{uuid.uuid4().hex}',
        deps_type=type(None),
        capabilities=[capability(), durability()],
    )
