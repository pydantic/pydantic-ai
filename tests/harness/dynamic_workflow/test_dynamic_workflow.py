"""Tests for the DynamicWorkflow capability."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability, PrefixTools
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_core import core_schema

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.dynamic_workflow import (
    DynamicWorkflow,
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowResourceLimits,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (the shared Monty loop uses asyncio)."""
    return 'asyncio'


def _sub_agent(text: str = 'ok', name: str | None = 'sub', description: str | None = None) -> Agent[object, str]:
    return Agent(TestModel(custom_output_text=text), name=name, description=description)


def _wf_agent(text: str = 'ok', name: str | None = 'sub', description: str | None = None) -> WorkflowAgent[object]:
    """Build a `WorkflowAgent` wrapping a trivial `TestModel` sub-agent."""
    return WorkflowAgent(_sub_agent(text, name), description=description)


class Review(BaseModel):
    score: int = Field(description='Score from 0 to 10.')
    note: str


class ScoreValue(BaseModel):
    model_config = ConfigDict(title='Score')

    value: int


class ScoreLabel(BaseModel):
    model_config = ConfigDict(title='Score')

    label: str


def _review_agent(text: dict[str, object] | None = None, name: str | None = 'reviewer') -> Agent[object, Review]:
    return Agent(
        TestModel(custom_output_args=text if text is not None else {'score': 9, 'note': 'great'}),
        name=name,
        output_type=Review,
    )


def _ctx() -> RunContext[object]:
    return RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)


def _ctx_with_queue() -> RunContext[object]:
    """A `RunContext` with a live pending-message queue, so `enqueue` (used by reveal) works."""
    return RunContext[object](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1, pending_messages=[]
    )


def _workflow_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    """All `run_workflow` tool returns in `messages`."""
    return [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_workflow'
    ]


def _user_prompt_text(messages: list[ModelMessage]) -> str:
    """Join the string user-prompt parts of `messages`."""
    return '\n'.join(
        p.content
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    )


def _enqueued_text(ctx: RunContext[object]) -> str:
    """Join the user-prompt text of every message enqueued on `ctx` (reveal announcements)."""
    return '\n'.join(
        part.content
        for pending in ctx.pending_messages or []
        for message in pending.messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


async def _run_script(ts: DynamicWorkflowToolset[object], code: str, ctx: RunContext[object] | None = None) -> Any:
    ctx = ctx or _ctx()
    tools = await ts.get_tools(ctx)
    tool = tools[ts.tool_name]
    return await ts.call_tool(ts.tool_name, {'code': code}, ctx, tool)


class _HostInternalOutput:
    pass


class _HostInternalValidated:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

    @classmethod
    def _validate(cls, _value: str) -> _HostInternalOutput:
        return _HostInternalOutput()


# --- Construction / wiring -------------------------------------------------


def test_capability_provides_toolset_with_propagated_config() -> None:
    reviewer = _sub_agent(name='reviewer')
    usage_limits = UsageLimits(request_limit=2)
    resource_limits: WorkflowResourceLimits = {'max_memory': 1024}
    cap = DynamicWorkflow[object](
        agents=[reviewer],
        tool_name='orchestrate',
        max_agent_calls=7,
        max_retries=1,
        forward_usage=False,
        inherit_model=True,
        sub_agent_usage_limits=usage_limits,
        resource_limits=resource_limits,
        id='wf',
    )
    toolset = cap.get_toolset()
    assert toolset.agents == [WorkflowAgent(reviewer)]
    assert toolset.tool_name == 'orchestrate'
    assert toolset.max_agent_calls == 7
    assert toolset.max_retries == 1
    assert toolset.forward_usage is False
    assert toolset.inherit_model is True
    assert toolset.sub_agent_usage_limits == usage_limits
    assert toolset.resource_limits == resource_limits
    # The toolset id derives from the capability id for durable execution.
    assert toolset.id == 'wf'


def test_toolset_id_defaults_to_tool_name() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    assert ts.id == 'run_workflow'


def test_not_spec_serializable() -> None:
    # `agents` holds live Agent objects, so the capability opts out of spec construction.
    assert DynamicWorkflow.get_serialization_name() is None


# --- Name resolution -------------------------------------------------------


def test_sandbox_name_falls_back_to_agent_name() -> None:
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=_sub_agent(name='reviewer'))])
    assert set(ts._by_name) == {'reviewer'}  # pyright: ignore[reportPrivateUsage]


def test_explicit_name_overrides_agent_name() -> None:
    # The sandbox handle can differ from the agent's own `name`.
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=_sub_agent(name='reviewer'), name='check')])
    assert set(ts._by_name) == {'check'}  # pyright: ignore[reportPrivateUsage]


def test_workflow_agent_resolved_name() -> None:
    assert WorkflowAgent(agent=_sub_agent(name='reviewer')).resolved_name == 'reviewer'
    assert WorkflowAgent(agent=_sub_agent(name='reviewer'), name='check').resolved_name == 'check'
    assert WorkflowAgent(agent=_sub_agent(name=None)).resolved_name is None


def test_workflow_agent_resolved_description() -> None:
    reviewer = _sub_agent(name='reviewer', description='Agent fallback description.')
    assert WorkflowAgent(reviewer).resolved_description == 'Agent fallback description.'
    assert WorkflowAgent(reviewer, description='Workflow-specific description.').resolved_description == (
        'Workflow-specific description.'
    )
    assert WorkflowAgent(_sub_agent(name='plain')).resolved_description is None


def test_workflow_agent_accepts_positional_agent() -> None:
    reviewer = _sub_agent(name='reviewer')
    entry = WorkflowAgent(reviewer, description='Reviews code.')
    assert entry.agent is reviewer
    assert entry.resolved_name == 'reviewer'
    assert entry.resolved_description == 'Reviews code.'


# --- Validation ------------------------------------------------------------


def test_invalid_identifier_name_raises() -> None:
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=_sub_agent(), name='bad-name')])


def test_keyword_name_raises() -> None:
    # `'class'.isidentifier()` is True, but a Python keyword can't be a sandbox function name --
    # the model could never call it (`await class(...)` is a syntax error). Reject it up front.
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=_sub_agent(), name='class')])


def test_empty_agents_raises() -> None:
    with pytest.raises(UserError, match='at least one sub-agent'):
        DynamicWorkflowToolset[object](agents=[])


def test_missing_name_raises() -> None:
    # No explicit name and the agent has no `name` either: nothing to expose as a function.
    with pytest.raises(UserError, match='has no `name`'):
        DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=_sub_agent(name=None))])


def test_duplicate_name_raises() -> None:
    with pytest.raises(UserError, match='must be unique'):
        DynamicWorkflowToolset[object](agents=[_wf_agent(name='dup'), _wf_agent(name='dup')])


def test_dynamic_workflow_constructor_rejects_invalid_identifier() -> None:
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflow[object](agents=[_sub_agent(name='bad-name')])


def test_dynamic_workflow_constructor_rejects_keyword_name() -> None:
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        DynamicWorkflow[object](agents=[_sub_agent(name='class')])


def test_dynamic_workflow_constructor_rejects_missing_name() -> None:
    with pytest.raises(UserError, match='has no `name`'):
        DynamicWorkflow[object](agents=[_sub_agent(name=None)])


def test_dynamic_workflow_constructor_rejects_duplicate_names() -> None:
    with pytest.raises(UserError, match='must be unique'):
        DynamicWorkflow[object](agents=[_sub_agent(name='dup'), WorkflowAgent(_sub_agent(name='dup'))])


# --- Catalog rendering / discovery surface ---------------------------------


async def test_raw_agents_in_dynamic_workflow_use_agent_metadata() -> None:
    reviewer = _sub_agent(name='reviewer', description='Reviews code for bugs.')
    cap = DynamicWorkflow[object](agents=[reviewer])
    desc = (await cap.get_toolset().get_tools(_ctx()))['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def reviewer(*, task: str) -> str:' in desc
    assert '"""Reviews code for bugs."""' in desc


async def test_mixed_raw_agents_and_wrappers_in_dynamic_workflow_catalog() -> None:
    reviewer = _sub_agent(name='reviewer', description='Reviews code for bugs.')
    summarizer = _sub_agent(name='summary_agent', description='Agent-level summary description.')
    cap = DynamicWorkflow[object](
        agents=[
            reviewer,
            WorkflowAgent(summarizer, name='summarizer', description='Summarizes findings for this workflow.'),
        ]
    )
    desc = (await cap.get_toolset().get_tools(_ctx()))['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def reviewer(*, task: str) -> str:' in desc
    assert '"""Reviews code for bugs."""' in desc
    assert 'async def summarizer(*, task: str) -> str:' in desc
    assert '"""Summarizes findings for this workflow."""' in desc


async def test_wrapper_overrides_agent_name_and_description_in_dynamic_workflow_catalog() -> None:
    reviewer = _sub_agent(name='reviewer', description='Agent-level description.')
    cap = DynamicWorkflow[object](
        agents=[WorkflowAgent(reviewer, name='check', description='Workflow-specific description.')]
    )
    desc = (await cap.get_toolset().get_tools(_ctx()))['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def check(*, task: str) -> str:' in desc
    assert 'async def reviewer' not in desc
    assert '"""Workflow-specific description."""' in desc
    assert 'Agent-level description.' not in desc


async def test_resolved_description_fallback_visible_in_tool_description() -> None:
    reviewer = _sub_agent(name='reviewer', description='Agent fallback description.')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(reviewer)])
    desc = (await ts.get_tools(_ctx()))['run_workflow'].tool_def.description
    assert desc is not None
    assert '"""Agent fallback description."""' in desc


async def test_mutating_original_agents_list_after_construction_has_no_effect() -> None:
    agents = [_sub_agent('base-out', 'base')]
    cap = DynamicWorkflow[object](agents=agents)
    agents.append(_sub_agent('extra-out', 'extra'))

    ts = cap.get_toolset()
    desc = (await ts.get_tools(_ctx_with_queue()))['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def base(*, task: str) -> str:' in desc
    assert 'async def extra' not in desc
    assert await _run_script(ts, "await base(task='x')", _ctx_with_queue()) == 'base-out'


async def test_description_lists_agents_as_functions() -> None:
    ts = DynamicWorkflowToolset[object](
        agents=[
            WorkflowAgent(agent=_review_agent(name='reviewer'), description='Reviews code for bugs.'),
            _wf_agent(name='summarizer', description='Summarizes findings.'),
        ],
        max_agent_calls=7,
    )
    tools = await ts.get_tools(_ctx())
    desc = tools['run_workflow'].tool_def.description
    assert desc is not None
    assert desc == snapshot(
        """\
Write and run a Python orchestration script in a sandbox to coordinate multiple sub-agents.

Use this to break a task across specialized sub-agents and combine their results in a single step --
fan work out in parallel, chain one agent's output into the next, vote across several, or loop until
done -- instead of delegating to one sub-agent at a time.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes** and **no third-party libraries**.
- **Useful standard-library modules**: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use
  at the top of the script. Other modules are unavailable or stubbed -- don't rely on them.
- **No wall-clock or timing primitives** (`asyncio.sleep`, `datetime.now()`, the `time` module).

Each sub-agent below is an async function. Call it with the `task` keyword argument -- write
`reviewer(task="...")`, not `reviewer("...")`; all parameters are keyword-only. A sub-agent returns
that agent's output: a string by default, or -- if it has a structured `output_type` -- a dict, whose
fields you read by subscript (`r["field"]`), not attribute (`r.field`). Each sub-agent call is an
independent run with no memory of earlier calls; include all needed context in `task`. Run several
at once with `asyncio.gather` rather than awaiting each sequentially:

```python
import asyncio
reviews = await asyncio.gather(reviewer(task="check auth"), reviewer(task="check parsing"))
```

`asyncio.gather` does **not** support `return_exceptions=True`, and a sub-agent that raises cannot be
caught inside the script: one failure aborts the whole script and you retry it. Design the script so
sub-agents don't depend on catching each other's errors.

The last expression's value is captured as the result -- you do **not** need to `print()` it, and
printing produces a string representation, not structured data. Use `print()` only for debug logging.
Return shapes: no print returns the last expression value (or `{}` if it is `None`); print plus a
non-`None` value returns `{"output": "<printed text>", "result": <last expression>}`; print plus
`None` returns `{"output": "<printed text>"}`. If a script fails after some sub-agent calls complete,
those completed results are reported back so a retry can reuse them.

This run can make at most 7 sub-agent calls in total -- one budget shared across every `run_workflow` call in the run, not per script; plan fan-out width accordingly.

Available sub-agents:

```python
class Review(TypedDict):
    score: int
    \"\"\"Score from 0 to 10.\"\"\"
    note: str

async def reviewer(*, task: str) -> Review:
    \"\"\"Reviews code for bugs.\"\"\"
    ...

async def summarizer(*, task: str) -> str:
    \"\"\"Summarizes findings.\"\"\"
    ...
```"""
    )
    assert 'async def reviewer(*, task: str) -> Review:' in desc
    assert 'async def summarizer(*, task: str) -> str:' in desc
    assert 'score: int' in desc


async def test_descriptions_override() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent(name='reviewer', description='Reviews code for bugs.')])
    tools = await ts.get_tools(_ctx())
    desc = tools['run_workflow'].tool_def.description
    assert desc is not None
    assert '"""Reviews code for bugs."""' in desc
    assert 'async def reviewer(*, task: str) -> str:' in desc


async def test_missing_output_schema_falls_back_to_any(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = _sub_agent(name='quirky')

    def broken_output_json_schema() -> dict[str, Any]:
        raise RuntimeError('schema unavailable')

    monkeypatch.setattr(sub, 'output_json_schema', broken_output_json_schema)
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=sub)])
    tools = await ts.get_tools(_ctx())
    desc = tools['run_workflow'].tool_def.description
    assert desc is not None
    assert 'async def quirky(*, task: str) -> Any:' in desc


async def test_tool_def_is_sequential_with_code_metadata() -> None:
    # `sequential=True` is what serializes two `run_workflow` calls in one model response;
    # dropping it would let them race the shared per-run budget counter.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], max_retries=5)
    tool = (await ts.get_tools(_ctx()))['run_workflow']
    assert tool.tool_def.sequential is True
    assert tool.tool_def.metadata == {'code_arg_name': 'code', 'code_arg_language': 'python'}
    # The configured retry budget must reach the served tool, not just sit on the toolset.
    assert tool.max_retries == 5


async def test_custom_tool_name() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], tool_name='orchestrate')
    tools = await ts.get_tools(_ctx())
    assert set(tools) == {'orchestrate'}


# --- Execution -------------------------------------------------------------


async def test_single_sub_agent_call() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('looks good', 'reviewer')])
    out = await _run_script(ts, "await reviewer(task='check')")
    assert out == 'looks good'


async def test_parallel_fan_out() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('r', 'reviewer'), _wf_agent('s', 'summarizer')])
    code = "import asyncio\nawait asyncio.gather(reviewer(task='a'), reviewer(task='b'), summarizer(task='c'))"
    out = await _run_script(ts, code)
    assert out == ['r', 'r', 's']


async def test_chaining() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('done', 'sub')])
    code = "a = await sub(task='one')\nb = await sub(task='two: ' + a)\n[a, b]"
    out = await _run_script(ts, code)
    assert out == ['done', 'done']


async def test_structured_output_arrives_as_dict() -> None:
    reviewer = _review_agent({'score': 9, 'note': 'great'}, name='reviewer')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=reviewer)])
    out = await _run_script(ts, "r = await reviewer(task='x')\nr['score']")
    assert out == 9


async def test_via_agent_run_end_to_end() -> None:
    observed_returns: list[Any] = []
    seen_tools: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append([td.name for td in info.function_tools])
        ret = next(iter(_workflow_returns(messages)), None)
        if ret is not None:
            observed_returns.append(ret.content)
            return ModelResponse(parts=[TextPart(f'done: {ret.content}')])
        code = "import asyncio\nresults = await asyncio.gather(reviewer(task='a'), reviewer(task='b'))\nresults"
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': code})])

    reviewer = _sub_agent('reviewed', 'reviewer')
    agent: Agent[object, str] = Agent(
        FunctionModel(model_fn), capabilities=[DynamicWorkflow[object](agents=[reviewer])]
    )
    result = await agent.run('please review')
    # Model is shown only the orchestration tool, not the sub-agents directly.
    assert seen_tools[0] == ['run_workflow']
    assert observed_returns == [['reviewed', 'reviewed']]
    assert result.output == "done: ['reviewed', 'reviewed']"


async def test_budget_terminal_result_arrives_as_tool_return_end_to_end() -> None:
    observed_returns: list[ToolReturnPart] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        ret = next(iter(_workflow_returns(messages)), None)
        if ret is not None:
            observed_returns.append(ret)
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='run_workflow',
                    args={'code': "await counted(task='first')\nawait counted(task='second')"},
                )
            ]
        )

    counted = _sub_agent('counted-result', 'counted')
    agent: Agent[object, str] = Agent(
        FunctionModel(model_fn),
        capabilities=[DynamicWorkflow[object](agents=[counted], max_agent_calls=1)],
    )

    result = await agent.run('run over budget')
    assert result.output == 'done'
    assert len(observed_returns) == 1
    terminal = observed_returns[0]
    assert isinstance(terminal, ToolReturnPart)

    class TerminalResult(BaseModel):
        error: str
        completed: list[str]

    terminal_payload: object = terminal.content
    terminal_result = TerminalResult.model_validate(terminal_payload)
    assert 'budget' in terminal_result.error
    assert terminal_result.completed == ['counted(task="first") -> "counted-result"']
    retry_parts = [
        p
        for m in result.all_messages()
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, RetryPromptPart) and p.tool_name == 'run_workflow'
    ]
    assert retry_parts == []


# --- Budget and guards -----------------------------------------------------


async def test_max_agent_calls_enforced_exactly() -> None:
    runs: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        runs.append('x')
        return ModelResponse(parts=[TextPart('ok')])

    counted: Agent[object, str] = Agent(FunctionModel(model_fn), name='counted')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=counted)], max_agent_calls=2)
    code = 'for i in range(5):\n    await counted(task=str(i))'
    # Budget exhaustion returns a terminal result (not a retry that can never succeed).
    out = await _run_script(ts, code)
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    assert out['completed'] == ['counted(task="0") -> "ok"', 'counted(task="1") -> "ok"']
    # Exactly the budget ran before the next call was refused.
    assert len(runs) == 2


async def test_max_agent_calls_exact_under_concurrent_fan_out() -> None:
    runs: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        runs.append('x')
        return ModelResponse(parts=[TextPart('ok')])

    counted: Agent[object, str] = Agent(FunctionModel(model_fn), name='counted')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=counted)], max_agent_calls=3)
    # Fan eight calls out concurrently: the await-free check-then-increment must admit exactly the
    # budget (3), not race past it. A regression inserting an `await` before the increment fails here.
    code = 'import asyncio\nawait asyncio.gather(*[counted(task=str(i)) for i in range(8)])'
    out = await _run_script(ts, code)
    # Read `completed` before the `isinstance` narrow: it keeps the value typed rather than widening
    # every dict subscript to an unknown element type under pyright strict.
    completed: list[str] = out['completed']
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    # Which three of the eight concurrent calls win the budget race, and the order they finish in,
    # are both scheduler-dependent (3.14 admits them in a different order than 3.13). The invariant
    # under test is the count: exactly the budget ran, each exactly once, before the rest were refused.
    assert len(completed) == 3
    assert len(set(completed)) == 3
    assert all(re.fullmatch(r'counted\(task="\d+"\) -> "ok"', entry) for entry in completed)
    assert len(runs) == 3


async def test_budget_terminal_result_keeps_surfaced_error_detail() -> None:
    # When the budget is exhausted in the same batch as an unrelated sub-agent failure, the
    # terminal result leads with the budget message and demotes the surfaced error to
    # `last_error`. Which of the batch's failures surfaces first inside the sandbox is
    # scheduler-dependent (the unrelated ValueError or the budget refusal), so assert the
    # invariant shape, not the specific failure.
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('kaboom')

    bad: Agent[object, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('ok', 'good'), WorkflowAgent(agent=bad)], max_agent_calls=2)
    code = "import asyncio\nawait asyncio.gather(good(task='x'), bad(task='y'), good(task='z'))"
    out = await _run_script(ts, code)
    assert isinstance(out, dict)
    assert out['error'] == (
        'This run exhausted its sub-agent call budget (2). Conclude using the results already gathered; '
        'further sub-agent calls in this run will be refused.'
    )
    assert 'sub-agent' in out['last_error']
    assert out['completed'] == ['good(task="x") -> "ok"']


def test_max_agent_calls_must_be_positive() -> None:
    with pytest.raises(UserError, match='max_agent_calls'):
        DynamicWorkflowToolset[object](agents=[_wf_agent()], max_agent_calls=0)


async def test_budget_shared_across_run_workflow_calls_in_one_run() -> None:
    # The description's headline claim: one budget shared across every `run_workflow` call in
    # the run, not per script. A second script on the same per-run instance sees the remainder.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], max_agent_calls=2)
    ctx = _ctx()
    assert await _run_script(ts, "await sub(task='a')", ctx) == 'ok'
    out = await _run_script(ts, "import asyncio\nawait asyncio.gather(sub(task='b'), sub(task='c'))", ctx)
    # Read `completed` before the `isinstance` narrow so pyright keeps the element type.
    completed: list[str] = out['completed']
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    # Only one of the second script's two calls fit in the remaining budget.
    assert len(completed) == 1


async def test_nested_workflow_refused_end_to_end() -> None:
    # The real claim behind the contextvar: a sub-agent that itself carries a DynamicWorkflow
    # capability -- dispatched through the executor's `ensure_future` (the same context-copy
    # path `asyncio.gather` uses) -- is refused when it tries to run its own workflow.
    leaf = _sub_agent('leaf-done', 'leaf')
    inner_returns: list[Any] = []

    def inner_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = _workflow_returns(messages)
        if returns:
            inner_returns.append(returns[-1].content)
            return ModelResponse(parts=[TextPart('inner saw terminal')])
        retries = [
            p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, RetryPromptPart)
        ]
        assert retries == []
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': "await leaf(task='x')"})])

    inner: Agent[object, str] = Agent(
        FunctionModel(inner_fn),
        name='inner',
        capabilities=[DynamicWorkflow[object](agents=[leaf])],
    )
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=inner)])
    out = await _run_script(ts, "await inner(task='go')")
    assert out == 'inner saw terminal'
    assert inner_returns == [
        {
            'error': (
                'Workflows do not nest: this sub-agent was invoked from a workflow and cannot start '
                'its own. Return your result to the orchestrating workflow instead.'
            )
        }
    ]


async def test_unknown_agent_rejected_by_type_check() -> None:
    # An unresolved name can't evade the static check, so it never reaches the sandbox.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='Type error in workflow'):
        await _run_script(ts, "await nonexistent(task='x')")


async def test_missing_task_kwarg() -> None:
    # `**` through `json.loads` (typed `Any`) evades the static check; the runtime guard
    # must still reject the call before the budget is touched.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='missing required keyword argument'):
        await _run_script(ts, "import json\nawait sub(**json.loads('{}'))")


async def test_positional_sub_agent_call_is_rejected() -> None:
    # Keyword-only `task` is enforced statically, before any sandbox execution.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='Type error in workflow'):
        await _run_script(ts, "await sub('x')")


async def test_extra_kwargs_rejected() -> None:
    # An extra kwarg must not be silently dropped -- the model needs to know it was ignored.
    # Smuggled through `**json.loads(...)` so the static check can't catch it first.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='unexpected keyword argument'):
        await _run_script(ts, "import json\nawait sub(task='x', **json.loads('{\"foo\": 1}'))")


async def test_non_str_task_rejected() -> None:
    # A non-string task (a dict/list would otherwise be silently smeared into message parts).
    # `json.loads` returns `Any`, so only the runtime guard can catch this.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='task must be a string'):
        await _run_script(ts, 'import json\nawait sub(task=json.loads(\'{"k": "v"}\'))')


async def test_type_error_caught_before_budget_is_spent() -> None:
    # A statically-detectable mistake costs a retry but no sub-agent budget: the full
    # budget is still available to a corrected script on the same toolset instance.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], max_agent_calls=1)
    with pytest.raises(ModelRetry, match='Type error in workflow'):
        await _run_script(ts, 'await sub(task=123)')
    assert await _run_script(ts, "await sub(task='x')") == 'ok'


async def test_forward_usage_true_shares_parent_usage() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], forward_usage=True)
    ctx = _ctx()
    await _run_script(ts, "await sub(task='x')", ctx)
    # Exactly the sub-agent's one model request lands on the shared counter -- no double-counting.
    assert ctx.usage.requests == 1


async def test_forward_usage_false_isolates_usage() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], forward_usage=False)
    ctx = _ctx()
    await _run_script(ts, "await sub(task='x')", ctx)
    assert ctx.usage.requests == 0


async def test_inherit_model_runs_sub_agents_on_parent_run_model() -> None:
    parent_model_calls: list[str] = []

    def parent_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parent_model_calls.append(_user_prompt_text(messages))
        return ModelResponse(parts=[TextPart('FROM_PARENT')])

    sub: Agent[object, str] = Agent(TestModel(custom_output_text='OWN'), name='sub')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=sub)], inherit_model=True)
    ctx = RunContext[object](
        deps=None,
        model=FunctionModel(parent_model_fn),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=1,
    )
    assert await _run_script(ts, "await sub(task='x')", ctx) == 'FROM_PARENT'
    assert parent_model_calls == ['x']


async def test_inherit_model_off_keeps_sub_agent_bound_model() -> None:
    parent_model_calls: list[str] = []

    def parent_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        # Never invoked: with inherit_model off the sub-agent keeps its own bound model.
        parent_model_calls.append(_user_prompt_text(messages))
        return ModelResponse(parts=[TextPart('FROM_PARENT')])

    sub: Agent[object, str] = Agent(TestModel(custom_output_text='OWN'), name='sub')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=sub)])
    ctx = RunContext[object](
        deps=None,
        model=FunctionModel(parent_model_fn),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=1,
    )
    assert await _run_script(ts, "await sub(task='x')", ctx) == 'OWN'
    assert parent_model_calls == []


async def test_deps_forwarded_to_sub_agents() -> None:
    # The parent run's `deps` must reach the sub-agent run (a headline README claim).
    seen: list[str] = []
    sub = Agent(TestModel(custom_output_text='ok'), name='sub', deps_type=str)

    @sub.instructions
    def record_deps(ctx: RunContext[str]) -> str:
        seen.append(ctx.deps)
        return 'noop'

    ts = DynamicWorkflowToolset[str](agents=[WorkflowAgent(agent=sub)])
    ctx = RunContext[str](deps='parent-deps', model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)
    tools = await ts.get_tools(ctx)
    await ts.call_tool(ts.tool_name, {'code': "await sub(task='x')"}, ctx, tools[ts.tool_name])
    assert seen == ['parent-deps']


async def test_sub_agent_does_not_see_parent_messages() -> None:
    # "Each sub-agent runs in isolation: its own message history, never the parent conversation."
    seen_texts: list[str] = []

    def spy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_texts.append(_user_prompt_text(messages))
        return ModelResponse(parts=[TextPart('ok')])

    sub: Agent[object, str] = Agent(FunctionModel(spy), name='sub')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=sub)])
    ctx = RunContext[object](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt='PARENT_PROMPT_MARKER',
        messages=[ModelRequest(parts=[UserPromptPart('PARENT_PROMPT_MARKER')])],
        run_step=1,
    )
    await _run_script(ts, "await sub(task='the task only')", ctx)
    assert seen_texts == ['the task only']  # only the task; no parent conversation leaked


async def test_sub_agent_usage_limits_checked_against_shared_counter() -> None:
    # With `forward_usage=True`, `sub_agent_usage_limits` is checked against the *shared* counter:
    # a parent run that already spent the request budget trips the sub-agent immediately, even
    # though the sub-agent itself has made no requests.
    sub: Agent[object, str] = Agent(TestModel(custom_output_text='ok'), name='sub')
    ts = DynamicWorkflowToolset[object](
        agents=[WorkflowAgent(agent=sub)],
        forward_usage=True,
        sub_agent_usage_limits=UsageLimits(request_limit=3),
    )
    ctx = _ctx()
    ctx.usage.requests = 3  # parent has already consumed the whole shared budget
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await sub(task='x')", ctx)
    msg = str(exc_info.value)
    assert 'UsageLimitExceeded' in msg
    assert 'The next request would exceed the request_limit of 3' in msg


async def test_sub_agent_usage_limits_enforced() -> None:
    sub: Agent[object, str] = Agent(TestModel(), name='limited')

    @sub.tool_plain
    def helper() -> str:
        return 'used'

    # A tool-using TestModel agent needs two model requests (call the tool, then answer), so a
    # request_limit of 1 must trip the second -- proving the limit reaches the sub-agent run.
    ts = DynamicWorkflowToolset[object](
        agents=[WorkflowAgent(agent=sub)],
        forward_usage=False,
        sub_agent_usage_limits=UsageLimits(request_limit=1),
    )
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await limited(task='x')")
    msg = str(exc_info.value)
    assert 'limited' in msg
    assert 'UsageLimitExceeded' in msg
    assert 'The next request would exceed the request_limit of 1' in msg


async def test_sub_agent_usage_limits_generous_allows_run() -> None:
    sub: Agent[object, str] = Agent(TestModel(custom_output_text='done'), name='roomy')

    @sub.tool_plain
    def helper() -> str:
        return 'used'

    # The same two-request agent runs fine when the limit is high enough, so the cap is the
    # configured value rather than a blanket rejection.
    ts = DynamicWorkflowToolset[object](
        agents=[WorkflowAgent(agent=sub)],
        forward_usage=False,
        sub_agent_usage_limits=UsageLimits(request_limit=5),
    )
    out = await _run_script(ts, "await roomy(task='x')")
    assert out == 'done'


# --- Errors / output shapes ------------------------------------------------


async def test_syntax_error() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='Syntax error'):
        await _run_script(ts, 'x = (')


async def test_retry_exhaustion_for_invalid_code_end_to_end() -> None:
    model_calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': 'x = ('})])

    agent: Agent[object, str] = Agent(
        FunctionModel(model_fn),
        capabilities=[DynamicWorkflow[object](agents=[_wf_agent()], max_retries=2)],
    )

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match="Tool 'run_workflow' exceeded max retries count of 2"):
            await agent.run('keep trying invalid code')

    retry_parts = [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, RetryPromptPart) and p.tool_name == 'run_workflow'
    ]
    assert model_calls == 3
    assert len(retry_parts) == 2
    assert all(isinstance(p.content, str) and p.content.startswith('Syntax error in workflow:') for p in retry_parts)


async def test_runtime_error() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, 'x = 1 / 0')
    msg = str(exc_info.value)
    assert 'Runtime error' in msg
    assert 'Completed sub-agent results' not in msg


async def test_runtime_error_includes_completed_sub_agent_results() -> None:
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('kaboom')

    bad: Agent[object, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('good-result', 'good'), WorkflowAgent(agent=bad)])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await good(task='use this')\nawait bad(task='fails')")
    msg = str(exc_info.value)
    assert msg.startswith('Runtime error in workflow:\n')
    assert "'bad'" in msg
    assert 'Completed sub-agent results from the failed script' in msg
    assert 'good(task="use this") -> "good-result"' in msg


async def test_completed_sub_agent_results_are_truncated_and_capped() -> None:
    long_task = 'task-' + ('x' * 180)
    long_result = 'result-' + ('y' * 500)

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('kaboom')

    bad: Agent[object, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent(long_result, 'good'), WorkflowAgent(agent=bad)])
    lines = [f"await good(task='{long_task}-{i}')" for i in range(21)]
    code = '\n'.join([*lines, "await bad(task='fails')"])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, code)
    msg = str(exc_info.value)
    assert '... 1 earlier completed result(s) omitted ...' in msg
    assert msg.count('good(task=') == 20
    assert ' ... [truncated]' in msg


async def test_duplicate_future_in_gather_is_retryable() -> None:
    # Awaiting the same sub-agent call twice in one gather makes the Monty VM panic; that panic
    # must surface as a retry, not tear down the whole agent run.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    code = 'import asyncio\nf = sub(task="x")\nawait asyncio.gather(f, f)'
    with pytest.raises(ModelRetry, match='aborted inside the sandbox') as exc_info:
        await _run_script(ts, code)
    msg = str(exc_info.value)
    assert 'Completed sub-agent results from the failed script' in msg
    assert 'sub(task="x") -> "ok"' in msg


async def test_sandbox_panic_after_budget_exhaustion_returns_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PanicException(BaseException):
        pass

    class FakeExecutor:
        dispatch: Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]

        def __init__(
            self, dispatch: Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]], valid_names: object
        ) -> None:
            _ = valid_names
            self.dispatch = dispatch

        async def run(self, _state: object) -> object:
            await self.dispatch('counted', {'task': 'first'})
            with pytest.raises(RuntimeError, match='budget'):
                await self.dispatch('counted', {'task': 'second'})
            raise PanicException('sandbox panic')

    monkeypatch.setattr('pydantic_ai_harness.dynamic_workflow._toolset.MontyExecutor', FakeExecutor)
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('counted-result', 'counted')], max_agent_calls=1)
    out = await _run_script(ts, '1 + 1')
    assert isinstance(out, dict)
    assert out['error'] == (
        'This run exhausted its sub-agent call budget (1). Conclude using the results already gathered; '
        'further sub-agent calls in this run will be refused.'
    )
    assert out['last_error'] == (
        'The workflow script aborted inside the sandbox after exhausting the sub-agent budget.'
    )
    assert out['completed'] == ['counted(task="first") -> "counted-result"']


async def test_non_panic_base_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # The panic guard catches BaseException but must re-raise anything that is not a VM panic.
    class _Boom(BaseException):
        pass

    async def _boom(self: Any, state: Any) -> Any:
        raise _Boom('boom')

    monkeypatch.setattr('pydantic_ai_harness._monty_exec.MontyExecutor.run', _boom)
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(_Boom):
        await _run_script(ts, "await sub(task='x')")


async def test_print_only_returns_output_dict() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    out = await _run_script(ts, "print('hello')")
    assert out == {'output': 'hello\n'}


async def test_print_with_result_returns_both() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    out = await _run_script(ts, "print('log')\n42")
    assert out == {'output': 'log\n', 'result': 42}


async def test_no_result_returns_empty_dict() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    out = await _run_script(ts, 'x = 1')
    assert out == {}


@pytest.mark.parametrize(('code', 'expected'), [('0', 0), ("''", ''), ('False', False)])
async def test_falsy_last_expression_returns_as_is(code: str, expected: object) -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    out = await _run_script(ts, code)
    assert out == expected


async def test_runtime_error_includes_prints() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()])
    with pytest.raises(ModelRetry, match='stdout before error'):
        await _run_script(ts, "print('before crash')\n1 / 0")


async def test_sub_agent_failure_inside_gather_aborts_script() -> None:
    # One failing sub-agent in a gather aborts the whole script; the retry names the failing
    # agent and its error type, and the agent run itself is not torn down.
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('kaboom')

    bad: Agent[object, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent('ok', 'good'), WorkflowAgent(agent=bad)])
    code = "import asyncio\nawait asyncio.gather(good(task='a'), bad(task='b'))"
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, code)
    msg = str(exc_info.value)
    assert "'bad'" in msg
    assert 'ValueError' in msg


async def test_host_errors_cannot_be_caught_in_sandbox() -> None:
    # On the deferred-future path this capability uses, host-raised exceptions abort the script
    # even under a matching `except RuntimeError`. Monty's inline resume path can differ.
    counted: Agent[object, str] = Agent(TestModel(custom_output_text='ok'), name='counted')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=counted)], max_agent_calls=1)
    code = "await counted(task='a')\ntry:\n    await counted(task='b')\nexcept RuntimeError:\n    pass\n1 / 0"
    out = await _run_script(ts, code)
    # The script aborted at the second call (`1 / 0` never ran) and the terminal result surfaced.
    assert isinstance(out, dict)
    assert 'budget' in out['error']
    assert 'sub-agent call budget (1) exhausted' in out['last_error']
    assert out['completed'] == ['counted(task="a") -> "ok"']


async def test_cancellation_awaits_inflight_sub_agents() -> None:
    # Cancelling the workflow tool call must not return until in-flight sub-agent runs have
    # fully unwound -- `task.cancel()` only schedules the CancelledError; the executor awaits
    # the cancelled tasks before propagating.
    started = asyncio.Event()
    unwound = asyncio.Event()

    async def blocking(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwound.set()
            raise
        return ModelResponse(parts=[TextPart('never')])  # pragma: no cover

    slow: Agent[object, str] = Agent(FunctionModel(blocking), name='slow')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=slow)])
    run = asyncio.ensure_future(_run_script(ts, "import asyncio\nawait asyncio.gather(slow(task='x'))"))
    await started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert unwound.is_set()


async def test_sub_agent_error_does_not_leak_host_internals() -> None:
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('SECRET_TOKEN_sk_abc123 at /Users/victim/secret.py')

    bad: Agent[object, str] = Agent(FunctionModel(boom), name='bad')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=bad)])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await bad(task='x')")
    msg = str(exc_info.value)
    assert 'SECRET_TOKEN' not in msg
    assert '/Users/victim' not in msg
    assert 'bad' in msg  # the failing agent is named


async def test_sub_agent_output_serialization_error_is_wrapped() -> None:
    leaky = Agent(TestModel(custom_output_args='x'), name='leaky', output_type=_HostInternalValidated)
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=leaky)])
    with pytest.raises(ModelRetry) as exc_info:
        await _run_script(ts, "await leaky(task='x')")
    msg = str(exc_info.value)
    assert "'leaky'" in msg
    assert 'PydanticSerializationError' in msg
    assert '_HostInternalOutput' not in msg


# --- Sandbox resource limits -----------------------------------------------


async def test_runaway_loop_stopped_by_duration_cap() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], resource_limits={'max_duration_secs': 0.2})
    with pytest.raises(ModelRetry, match='Runtime error'):
        await _run_script(ts, 'while True:\n    x = 1')


async def test_awaiting_sub_agents_does_not_count_against_duration_cap() -> None:
    # `max_duration_secs` bounds in-sandbox execution time, not wall-clock: time the script spends
    # awaiting sub-agents is spent suspended on the host, so it must not accrue against the cap.
    # Here three sub-agents each sleep 0.2s on the host under a 0.1s cap; the workflow still
    # completes. Guards the documented behavior that the timer excludes sub-agent latency.
    async def slow_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(0.2)
        return ModelResponse(parts=[TextPart('done')])

    sub: Agent[object, str] = Agent(FunctionModel(slow_model), name='sub')
    ts = DynamicWorkflowToolset[object](agents=[WorkflowAgent(agent=sub)], resource_limits={'max_duration_secs': 0.1})
    code = "import asyncio\nawait asyncio.gather(sub(task='a'), sub(task='b'), sub(task='c'))"
    assert await _run_script(ts, code) == ['done', 'done', 'done']


async def test_resource_limit_override_is_enforced_in_the_sandbox() -> None:
    # The configured cap reaches the sandbox: a script comfortably under the default backstop
    # trips a small explicit `max_memory`, proving a partial dict is applied (merged, not dropped).
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], resource_limits={'max_memory': 4096})
    with pytest.raises(ModelRetry, match='Runtime error'):
        await _run_script(ts, "x = ['data'] * 100000\nlen(x)")


async def test_unlimited_runs_without_a_backstop() -> None:
    # `'unlimited'` resolves to no limits; a trivial script still completes normally.
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], resource_limits='unlimited')
    out = await _run_script(ts, '1 + 1')
    assert out == 2


def test_unknown_resource_limit_key_raises_at_construction() -> None:
    # A typo'd key (e.g. plural `max_durations_secs`) must not be silently dropped -- that would
    # quietly disable the duration cap it was meant to set. Validated eagerly, not at the first call.
    with pytest.raises(UserError, match='Unknown `resource_limits` key'):
        DynamicWorkflowToolset[object](
            agents=[_wf_agent()],
            resource_limits={'max_durations_secs': 5},  # pyright: ignore[reportArgumentType]
        )


# --- Lifecycle -------------------------------------------------------------


async def test_for_run_resets_budget() -> None:
    ts = DynamicWorkflowToolset[object](agents=[_wf_agent()], max_agent_calls=1)
    await _run_script(ts, "await sub(task='x')")
    assert ts._call_count == 1  # pyright: ignore[reportPrivateUsage]
    fresh = await ts.for_run(_ctx())
    assert fresh._call_count == 0  # pyright: ignore[reportPrivateUsage]


async def test_for_run_raises_on_invalid_direct_mutation() -> None:
    agents = [_wf_agent('b', 'base')]
    ts = DynamicWorkflowToolset[object](agents=agents)
    agents.append(WorkflowAgent(agent=_sub_agent(name=None)))
    with pytest.raises(UserError, match='has no `name`'):
        await ts.for_run(_ctx())


async def test_for_run_raises_on_duplicate_direct_mutation() -> None:
    agents = [_wf_agent('b', 'base')]
    ts = DynamicWorkflowToolset[object](agents=agents)
    agents.append(_wf_agent('shadow', 'base'))
    with pytest.raises(UserError, match='must be unique'):
        await ts.for_run(_ctx())


async def test_for_run_clones_share_reveals_and_isolate_budget() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('base-out', 'base')], max_agent_calls=2)
    ts = workflow.get_toolset()
    clone_a = await ts.for_run(_ctx())
    clone_b = await ts.for_run(_ctx())

    workflow.reveal(_sub_agent('extra-out', 'extra'))
    ctx_a = _ctx_with_queue()
    ctx_b = _ctx_with_queue()

    await clone_a.get_tools(ctx_a)
    await clone_b.get_tools(ctx_b)

    assert set(clone_a._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert set(clone_b._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert await _run_script(clone_a, "await extra(task='x')", ctx_a) == 'extra-out'
    assert clone_a._call_count == 1  # pyright: ignore[reportPrivateUsage]
    assert clone_b._call_count == 0  # pyright: ignore[reportPrivateUsage]
    assert ts._call_count == 0  # pyright: ignore[reportPrivateUsage]
    assert await _run_script(clone_b, "await extra(task='x')", ctx_b) == 'extra-out'


# --- Executor cancellation cleanup ------------------------------------------


async def test_cancellation_closes_unscheduled_coroutines() -> None:
    # In global-sequential mode (durable backends) deferred calls are kept as bare, unscheduled
    # coroutines. On cancellation those must be `close()`d, not cancelled -- covers the
    # coroutine branch of the executor's cleanup, unreachable through DynamicWorkflowToolset.
    from pydantic_monty import MontyRepl

    from pydantic_ai_harness._monty_exec import MontyExecutor

    started = asyncio.Event()

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        started.set()
        await asyncio.Event().wait()  # block forever; only cancellation ends this

    repl = MontyRepl()
    state = repl.feed_start("import asyncio\nawait asyncio.gather(sub(task='a'), sub(task='b'))")
    executor = MontyExecutor(dispatch=dispatch, valid_names={'sub'}, global_sequential=True)
    # The first call is awaited inline and blocks; the second is still a bare coroutine.
    task = asyncio.ensure_future(executor.run(state))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- Runtime reveal --------------------------------------------------------


async def test_runtime_reveal_announces_and_makes_callable() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('base-out', 'base')])
    ts = workflow.get_toolset()
    ctx = _ctx_with_queue()

    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base'}  # pyright: ignore[reportPrivateUsage]
    assert _enqueued_text(ctx) == ''  # nothing revealed yet

    workflow.reveal(_review_agent({'score': 4, 'note': 'extra-out'}, name='extra'))
    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    announced = _enqueued_text(ctx)
    assert 'class Review(TypedDict):' in announced
    assert 'score: int' in announced
    assert 'async def extra(*, task: str) -> Review:' in announced
    assert '`run_workflow`' in announced  # the announcement names the actual tool

    out = await _run_script(ts, "r = await extra(task='x')\nr['score']", ctx)
    assert out == 4


async def test_runtime_reveal_disambiguates_type_name_conflicts_against_catalog() -> None:
    baseline: Agent[object, ScoreValue] = Agent(
        TestModel(custom_output_args={'value': 1}),
        name='baseline',
        output_type=ScoreValue,
    )
    revealed: Agent[object, ScoreLabel] = Agent(
        TestModel(custom_output_args={'label': 'different'}),
        name='extra',
        output_type=ScoreLabel,
    )
    workflow = DynamicWorkflow[object](agents=[baseline])
    ts = workflow.get_toolset()
    ctx = _ctx_with_queue()

    baseline_description = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    assert baseline_description is not None
    assert 'class Score(TypedDict):' in baseline_description
    assert 'value: int' in baseline_description

    workflow.reveal(revealed)
    after_reveal_description = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    announced = _enqueued_text(ctx)

    assert after_reveal_description == baseline_description
    assert 'class Score(TypedDict):' not in announced
    assert 'value: int' not in announced
    assert 'class extra_Score(TypedDict):' in announced
    assert 'label: str' in announced
    assert 'async def extra(*, task: str) -> extra_Score:' in announced


async def test_runtime_reveal_without_pending_queue_warns_and_remains_callable() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('base-out', 'base')])
    ts = workflow.get_toolset()
    ctx = _ctx()

    await ts.get_tools(ctx)
    workflow.reveal(_sub_agent('extra-out', 'extra'))
    with pytest.warns(UserWarning, match='could not enqueue its announcement'):
        await ts.get_tools(ctx)

    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert ctx.pending_messages is None
    assert await _run_script(ts, "await extra(task='x')", ctx) == 'extra-out'


async def test_reveal_is_idempotent_across_steps() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    ts = workflow.get_toolset()
    ctx = _ctx_with_queue()

    workflow.reveal(_sub_agent('e', 'extra'))
    await ts.get_tools(ctx)
    await ts.get_tools(ctx)  # re-resolving tools must not re-announce an already-revealed agent
    assert _enqueued_text(ctx).count('async def extra') == 1


async def test_reveal_keeps_tool_description_frozen() -> None:
    # The cached prompt prefix must not change when an agent is revealed.
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    ts = workflow.get_toolset()
    ctx = _ctx_with_queue()

    before = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    workflow.reveal(_sub_agent('e', 'extra'))
    after = (await ts.get_tools(ctx))['run_workflow'].tool_def.description
    assert before == after
    assert after is not None and 'extra' not in after  # the reveal never enters the description


def test_reveal_duplicate_name_raises_immediately() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    with pytest.raises(UserError, match='must be unique'):
        workflow.reveal(_sub_agent('shadow', 'base'))


def test_reveal_invalid_identifier_raises_immediately() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        workflow.reveal(_sub_agent(name='bad-name'))


def test_reveal_keyword_name_raises_immediately() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    with pytest.raises(UserError, match='cannot be exposed as a sandbox function'):
        workflow.reveal(_sub_agent(name='class'))


def test_reveal_missing_name_raises_immediately() -> None:
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    with pytest.raises(UserError, match='has no `name`'):
        workflow.reveal(_sub_agent(name=None))


async def test_direct_toolset_mutation_invalid_entry_raises_on_get_tools() -> None:
    agents = [_wf_agent('b', 'base')]
    ts = DynamicWorkflowToolset[object](agents=agents)
    ctx = _ctx_with_queue()
    await ts.get_tools(ctx)

    agents.append(WorkflowAgent(agent=_sub_agent(name=None)))
    with pytest.raises(UserError, match='has no `name`'):
        await ts.get_tools(ctx)


def _registry_ctx(capabilities: dict[str, AbstractCapability[object]], *, loaded: set[str]) -> RunContext[object]:
    """A `RunContext` with a given capability registry and loaded-capability-id set."""
    return RunContext[object](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=1,
        pending_messages=[],
        capabilities=capabilities,
        loaded_capability_ids=loaded,
    )


def _deferred_ctx(workflow: AbstractCapability[object], *, loaded: bool) -> RunContext[object]:
    """A `RunContext` whose capability registry holds `workflow` under id `wf`, loaded or not."""
    return _registry_ctx({'wf': workflow}, loaded={'wf'} if loaded else set())


async def test_reveal_on_unloaded_deferred_capability_is_held_back() -> None:
    # While a deferred capability is unloaded, its catalog is hidden from the model -- announcing
    # a reveal then would leak the sub-agent signature into the conversation. The reveal must stay
    # pending until the model loads the capability.
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')], id='wf', defer_loading=True)
    ts = workflow.get_toolset()

    unloaded = _deferred_ctx(workflow, loaded=False)
    await ts.get_tools(unloaded)
    workflow.reveal(_sub_agent('e', 'extra'))
    await ts.get_tools(unloaded)
    assert set(ts._by_name) == {'base'}  # pyright: ignore[reportPrivateUsage]
    assert _enqueued_text(unloaded) == ''

    loaded = _deferred_ctx(workflow, loaded=True)
    await ts.get_tools(loaded)
    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert 'async def extra(*, task: str) -> str:' in _enqueued_text(loaded)


async def test_reveal_announces_when_capability_in_registry_is_not_deferred() -> None:
    # A capability registered without `defer_loading` is always visible, so reveals announce
    # immediately even though the registry holds an entry under this toolset's id.
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')], id='wf')
    ts = workflow.get_toolset()
    ctx = _deferred_ctx(workflow, loaded=False)

    await ts.get_tools(ctx)
    workflow.reveal(_sub_agent('e', 'extra'))
    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert 'async def extra(*, task: str) -> str:' in _enqueued_text(ctx)


async def test_reveal_ignores_unrelated_deferred_capability_with_colliding_id() -> None:
    # An id-less workflow's toolset id falls back to the tool name (`run_workflow`). An unrelated
    # deferred capability registered under exactly that id must not suppress this workflow's
    # reveals -- ownership is resolved by identity, not by id.
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    unrelated = DynamicWorkflow[object](agents=[_sub_agent('u', 'unrelated')], id='run_workflow', defer_loading=True)
    ts = workflow.get_toolset()
    ctx = _registry_ctx({'run_workflow': unrelated, 'dynamic_workflow': workflow}, loaded=set())

    await ts.get_tools(ctx)
    workflow.reveal(_sub_agent('e', 'extra'))
    await ts.get_tools(ctx)
    assert set(ts._by_name) == {'base', 'extra'}  # pyright: ignore[reportPrivateUsage]
    assert 'async def extra(*, task: str) -> str:' in _enqueued_text(ctx)


async def test_reveal_held_back_while_deferred_wrapper_capability_is_unloaded() -> None:
    # A wrapper capability is registered in place of what it wraps, so a deferred wrapper hides
    # a non-deferred inner workflow until loaded -- reveals must follow the wrapper's state.
    workflow = DynamicWorkflow[object](agents=[_sub_agent('b', 'base')])
    wrapper = PrefixTools[object](wrapped=workflow, prefix='team', id='wf', defer_loading=True)
    # In a run, core resolves tools through the wrapper's prefixed toolset, which delegates to
    # this one; the registry holds only the wrapper, never the inner workflow.
    ts = workflow.get_toolset()

    unloaded = _deferred_ctx(wrapper, loaded=False)
    await ts.get_tools(unloaded)
    workflow.reveal(_sub_agent('e', 'extra'))
    await ts.get_tools(unloaded)
    assert _enqueued_text(unloaded) == ''

    loaded = _deferred_ctx(wrapper, loaded=True)
    await ts.get_tools(loaded)
    assert 'async def extra(*, task: str) -> str:' in _enqueued_text(loaded)


async def test_reveal_on_deferred_capability_end_to_end_via_agent_run() -> None:
    # Regression: a `reveal()` made while the deferred capability is still unloaded must not
    # leak its announcement into the prompt; it arrives only after `load_capability`.
    base = _sub_agent('base-done', 'base')
    extra = _sub_agent('extra-done', 'extra')
    workflow = DynamicWorkflow[object](agents=[base], id='wf', defer_loading=True)
    announcement_seen: list[bool] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(1 for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart))
        if step == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_extra', args={})])
        announcement_seen.append('async def extra' in _user_prompt_text(messages))
        if step == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'wf'})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[workflow])

    @agent.tool_plain
    def reveal_extra() -> str:
        workflow.reveal(extra)
        return 'revealed'

    result = await agent.run('start')
    assert result.output == 'done'
    # Step after the reveal (capability still unloaded): no announcement leaked.
    # Step after load_capability: the announcement arrived.
    assert announcement_seen == [False, True]


async def test_reveal_end_to_end_via_agent_run() -> None:
    base = _sub_agent('base-done', 'base')
    extra = _sub_agent('extra-done', 'extra')
    # The host keeps a reference to the capability (here a closure; in practice often via `deps`).
    workflow = DynamicWorkflow[object](agents=[base])
    saw_announcement: list[bool] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = _workflow_returns(messages)
        if len(returns) == 0:
            # First step: reveal `extra`, and run `base` now to force a second step.
            workflow.reveal(extra)
            return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': "await base(task='go')"})])
        if len(returns) == 1:
            # Second step: the announcement for `extra` has arrived and it is now callable.
            saw_announcement.append('async def extra(*, task: str) -> str:' in _user_prompt_text(messages))
            return ModelResponse(
                parts=[ToolCallPart(tool_name='run_workflow', args={'code': "await extra(task='go')"})]
            )
        return ModelResponse(parts=[TextPart(f'final: {returns[-1].content}')])

    agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[workflow])
    result = await agent.run('start')
    assert saw_announcement == [True]  # the model saw the reveal announcement, mid-run
    assert result.output == 'final: extra-done'  # and the revealed sub-agent actually ran


# --- Coexistence with CodeMode -----------------------------------------------
# DynamicWorkflow and CodeMode are independent. `run_workflow` is its own code-execution
# sandbox, so CodeMode leaves it native (a peer of `run_code`) rather than folding it in.


async def test_run_workflow_stays_native_alongside_code_mode() -> None:
    # Both capabilities on one agent expose two separate tools; `run_workflow` is not folded into
    # `run_code`, and its sub-agent catalog is not leaked into `run_code`'s description.
    model = TestModel(call_tools=[])
    agent: Agent[object, str] = Agent(
        model,
        capabilities=[CodeMode[object](), DynamicWorkflow[object](agents=[_sub_agent('ok', 'reviewer')])],
    )
    await agent.run('hi')
    params = model.last_model_request_parameters
    assert params is not None
    assert sorted(td.name for td in params.function_tools) == ['run_code', 'run_workflow']
    run_code = next(td for td in params.function_tools if td.name == 'run_code')
    assert 'run_workflow' not in (run_code.description or '')


async def test_run_workflow_orchestrates_alongside_code_mode_end_to_end() -> None:
    # The native `run_workflow` still drives its own sub-agent sandbox while CodeMode is present.
    reviewer = _sub_agent('LGTM', 'reviewer')

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='run_workflow', args={'code': 'await reviewer(task="x")'})]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[object, str] = Agent(
        FunctionModel(model_fn),
        capabilities=[CodeMode[object](), DynamicWorkflow[object](agents=[reviewer])],
    )
    result = await agent.run('go')
    assert result.output == 'done'
    returns = [
        p
        for m in result.all_messages()
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_workflow'
    ]
    assert returns[-1].content == 'LGTM'
