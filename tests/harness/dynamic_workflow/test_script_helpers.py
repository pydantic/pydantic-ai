"""Tests for the helpers every workflow script gets: `agent`, `parallel`, `pipeline`, `log`, `phase`, `budget`."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic_monty import AsyncMonty

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness._monty_exec import MontyExecutor
from pydantic_ai_harness.dynamic_workflow import (
    DynamicWorkflow,
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowLogEvent,
    WorkflowPhaseEvent,
)

from .conftest import call_workflow_tool, echo_agent, recording_tracer, run_ctx, scripted_model, spans_named


async def _run(ts: DynamicWorkflowToolset[object], code: str) -> Any:
    return await call_workflow_tool(ts, {'code': code})


async def _description(ts: DynamicWorkflowToolset[object]) -> str:
    description = (await ts.get_tools(run_ctx()))['run_workflow'].tool_def.description
    assert description is not None
    return description


def _toolset(*names: str, **options: Any) -> DynamicWorkflowToolset[object]:
    return DynamicWorkflowToolset[object](agents=[WorkflowAgent(echo_agent(name)) for name in names], **options)


# --- agent() ---------------------------------------------------------------


async def test_agent_defaults_to_the_only_sub_agent() -> None:
    assert await _run(_toolset('solo'), "await agent('hi')") == 'solo:hi'


async def test_agent_picks_a_sub_agent_by_name() -> None:
    assert await _run(_toolset('a', 'b'), "await agent('hi', name='b')") == 'b:hi'


async def test_agent_uses_default_agent() -> None:
    assert await _run(_toolset('a', 'b', default_agent='b'), "await agent('hi')") == 'b:hi'


def test_default_agent_must_be_a_sub_agent() -> None:
    with pytest.raises(UserError, match="`default_agent` 'c' is not a sub-agent; choose one of: a, b"):
        _toolset('a', 'b', default_agent='c')


@pytest.mark.parametrize(
    ('code', 'error'),
    [
        ("await agent('hi')", 'agent() needs name=, one of: a, b'),
        ("await agent('hi', name='c')", "agent() got unknown name 'c'; available: a, b"),
        ("import json\nawait agent(json.loads('1'), name='a')", 'agent() task must be a string, got int'),
        ("import json\nawait agent('hi', name=json.loads('1'))", 'agent() name must be a string, got int'),
        (
            "await agent('hi', name='a', schema={'type': 'string'})",
            'agent() schema must be a JSON schema of type "object"',
        ),
    ],
)
async def test_agent_rejects_bad_calls_before_spending_budget(code: str, error: str) -> None:
    ts = _toolset('a', 'b')
    with pytest.raises(ModelRetry, match=r'Runtime error in workflow') as exc_info:
        await _run(ts, code)
    assert error in str(exc_info.value)
    assert ts._call_count == 0  # pyright: ignore[reportPrivateUsage]


async def test_agent_schema_sets_a_per_call_output_type() -> None:
    sub = Agent(TestModel(custom_output_args={'verdict': 'pass', 'score': 3}), name='judge')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(sub)])
    schema = "{'type': 'object', 'properties': {'verdict': {'type': 'string'}, 'score': {'type': 'integer'}}}"
    out = await _run(ts, f"r = await agent('judge this', schema={schema})\n[r['verdict'], r['score']]")
    assert out == ['pass', 3]


async def test_agent_model_overrides_the_sub_agent_model() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the constructed model must not run')  # pragma: no cover

    sub = Agent(FunctionModel(broken), name='sub')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(sub)], inherit_model=True)
    assert await _run(ts, "await agent('hi', model='test')") == 'success (no tool calls)'


# --- parallel() and pipeline() --------------------------------------------------


async def test_parallel_runs_awaitables_and_thunks_and_turns_failures_into_none() -> None:
    code = """\
def plain():
    return 'value'

def boom():
    raise ValueError('nope')

await parallel([agent('x'), lambda: agent('y'), plain, boom, agent('z', name='missing')])
"""
    assert await _run(_toolset('a'), code) == ['a:x', 'a:y', 'value', None, None]


async def test_parallel_of_nothing_is_empty() -> None:
    assert await _run(_toolset('a'), 'await parallel([])') == []


async def test_pipeline_threads_each_item_through_every_stage() -> None:
    code = """\
async def review(prev, item, index):
    return await agent(f'{index}:{prev}')

def verify(prev, item, index):
    return agent(prev + '+' + item)

def tidy(prev, item, index):
    return prev.upper()

await pipeline(['p', 'q'], review, verify, tidy)
"""
    assert await _run(_toolset('a'), code) == ['A:A:0:P+P', 'A:A:1:Q+Q']


async def test_pipeline_ends_an_item_on_error_or_none() -> None:
    code = """\
def first(prev, item, index):
    if item == 'raise':
        raise ValueError('nope')
    if item == 'none':
        return None
    return item

def second(prev, item, index):
    return prev + '!'

await pipeline(['raise', 'none', 'ok'], first, second)
"""
    assert await _run(_toolset('a'), code) == [None, None, 'ok!']


@pytest.mark.parametrize('helper', ['parallel', 'pipeline'])
async def test_fan_out_helpers_cap_their_items(helper: str) -> None:
    ts = _toolset('a', max_items_per_call=2)
    with pytest.raises(ModelRetry) as exc_info:
        await _run(ts, f'await {helper}([1, 2, 3])')
    assert f'{helper}() got 3 items; the limit is 2 per call' in str(exc_info.value)


async def test_max_concurrent_agents_caps_sub_agents_in_flight() -> None:
    in_flight = 0
    peak = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ModelResponse(parts=[TextPart('ok')])

    ts = DynamicWorkflowToolset[object](
        agents=[WorkflowAgent(Agent(FunctionModel(respond), name='sub'))], max_concurrent_agents=2
    )
    out = await _run(ts, 'await parallel([agent(str(i)) for i in range(6)])')
    assert out == ['ok'] * 6
    assert peak == 2


@pytest.mark.parametrize('option', ['max_concurrent_agents', 'max_workflow_depth', 'max_items_per_call'])
def test_limits_must_be_positive(option: str) -> None:
    with pytest.raises(UserError, match=f'`{option}` must be at least 1'):
        _toolset('a', **{option: 0})


# --- budget(), log() and phase() ----------------------------------------------------


async def test_budget_reports_the_shared_call_budget() -> None:
    ts = _toolset('a', max_agent_calls=5)
    assert await _run(ts, "await agent('x')\nbudget()") == {'max': 5, 'used': 1, 'remaining': 4}


async def test_budget_exhaustion_records_one_span() -> None:
    tracer, exporter = recording_tracer()
    ts = _toolset('a', max_agent_calls=1)
    out = await call_workflow_tool(
        ts, {'code': "await parallel([agent('x'), agent('y'), agent('z')])"}, run_ctx(tracer=tracer)
    )
    assert 'exhausted its sub-agent call budget (1)' in out['error']
    [span] = spans_named(exporter, 'dynamic_workflow.budget_exhausted')
    assert dict(span.attributes or {}) == {'dynamic_workflow.max_agent_calls': 1}


class _EventLog(AbstractCapability[object]):
    def __init__(self) -> None:
        self.events: list[WorkflowLogEvent | WorkflowPhaseEvent] = []

    @on_event(WorkflowLogEvent)
    async def _log(self, ctx: RunContext[object], event: WorkflowLogEvent) -> None:
        self.events.append(event)

    @on_event(WorkflowPhaseEvent)
    async def _phase(self, ctx: RunContext[object], event: WorkflowPhaseEvent) -> None:
        self.events.append(event)


async def test_log_and_phase_emit_events_into_the_run() -> None:
    code = "phase('Find')\nlog('starting')\nawait agent('x', phase='Verify')\nlog(42)"
    observer = _EventLog()
    agent = Agent[object, str](
        scripted_model([('run_workflow', {'code': code})]),
        capabilities=[DynamicWorkflow[object](agents=[echo_agent('a')]), observer],
    )
    await agent.run('go')
    assert [
        (type(event).__name__, event.title if isinstance(event, WorkflowPhaseEvent) else event.message)
        for event in observer.events
    ] == snapshot(
        [
            ('WorkflowPhaseEvent', 'Find'),
            ('WorkflowLogEvent', 'starting'),
            ('WorkflowPhaseEvent', 'Verify'),
            ('WorkflowLogEvent', '42'),
        ]
    )
    assert {(event.workflow, event.tool_name) for event in observer.events} == {(None, 'run_workflow')}


# --- Names and description -------------------------------------------------------


async def test_a_sub_agent_named_like_a_helper_keeps_its_name() -> None:
    ts = _toolset('log', 'a')
    assert await _run(ts, "[await log(task='x'), budget()['used']]") == ['log:x', 1]
    assert '`log(message)`' not in await _description(ts)


async def test_description_documents_the_helpers() -> None:
    description = await _description(_toolset('a', 'b'))
    helpers = description[description.index('These helpers') : description.index('`asyncio.gather` also works')]
    assert helpers == snapshot("""\
These helpers are already defined:
- `await agent(task, *, name=None, schema=None, model=None, phase=None)`: run one sub-agent (`name` picks it). `schema` is a JSON schema (`{"type": "object", ...}`) the output must follow for this call, so the result is a dict. `model` names a model (such as `"openai:gpt-5"`) to run it with. `phase` also calls `phase(phase)` first.
- `await parallel(tasks)`: run awaitables (such as `agent(...)` calls) or zero-argument functions concurrently and return their results in order. A failed item becomes `None`; `parallel` never raises.
- `await pipeline(items, *stages)`: run every item through each stage in turn, without waiting for the other items between stages. A stage is called as `stage(prev, item, index)`, where `prev` is the previous stage's result (the item itself for the first stage), and may be `async` or return an awaitable. A stage that raises or returns `None` ends that item with `None`.
- `log(message)`: report progress to the user.
- `phase(title)`: start a named phase of progress for the user.
- `budget()`: `{"max": ..., "used": ..., "remaining": ...}` sub-agent calls for this run.

""")


async def test_helper_errors_report_the_script_line() -> None:
    with pytest.raises(ModelRetry) as exc_info:
        await _run(_toolset('a'), "x = 1\nraise ValueError('line two')")
    assert 'line 2, in <module>' in str(exc_info.value)


async def test_helpers_are_type_checked() -> None:
    with pytest.raises(ModelRetry, match='Type error in workflow') as exc_info:
        await _run(_toolset('a'), 'await agent(1)')
    assert 'Expected `str`' in str(exc_info.value)


# --- MontyExecutor inline names --------------------------------------------------------


async def test_inline_names_do_not_wait_for_pending_calls() -> None:
    released = asyncio.Event()

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        if name == 'release':
            released.set()
            return None
        await released.wait()
        return 'slow'

    async with AsyncMonty() as pool, pool.checkout() as session:
        executor = MontyExecutor(dispatch=dispatch, valid_names={'slow', 'release'}, inline_names={'release'})
        code = 'import asyncio\ntask = slow()\nrelease()\nawait task'
        complete = await asyncio.wait_for(executor.run(partial(session.feed_start, code)), timeout=10)
    assert complete.output == 'slow'


async def test_script_inputs_are_not_bound_when_saved_workflows_are_off() -> None:
    with pytest.raises(ModelRetry, match='Type error in workflow'):
        await _run(_toolset('a'), 'args')
