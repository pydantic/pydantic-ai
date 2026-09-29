"""Tests for saved workflows: the file format, the library, running by name, nesting, and saving."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic_monty import AsyncMonty

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ModelRequest, SystemPromptPart, ToolReturnPart
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceTimeoutError
from pydantic_ai_harness.dynamic_workflow import (
    DynamicWorkflow,
    DynamicWorkflowToolset,
    SavedWorkflow,
    WorkflowAgent,
    WorkflowLibrary,
)
from pydantic_ai_harness.dynamic_workflow._run import Frame, WorkflowRun

from .conftest import call_workflow_tool, echo_agent, recording_tracer, run_ctx, scripted_model, spans_named

TRIAGE = """\
\"\"\"Review each file, then summarize.\"\"\"

meta = {
    'name': 'triage',
    'description': 'Review each file and summarize.',
    'when_to_use': 'When asked to review several files.',
    'args': {'type': 'object', 'properties': {'files': {'type': 'array'}}, 'required': ['files']},
    'agents': ['reviewer'],
    'returns': 'A summary string.',
}

reviews = await parallel([agent(f'review {path}', name='reviewer') for path in args['files']])
' | '.join(reviews)
"""


def _workflow(name: str, body: str, **meta: Any) -> SavedWorkflow:
    return SavedWorkflow.create(name=name, description=f'The {name} workflow.', code=body, **meta)


def _toolset(*workflows: SavedWorkflow, **options: Any) -> DynamicWorkflowToolset[object]:
    return DynamicWorkflowToolset[object](
        agents=[WorkflowAgent(echo_agent('reviewer'))],
        library=WorkflowLibrary(workflows={workflow.name: workflow for workflow in workflows}),
        **options,
    )


def _write(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(source)
    return path


def _workspace(root: Path) -> Workspace:
    return Workspace(LocalWorkspaceBackend(root))


# --- The file format ---------------------------------------------------------------------


def test_parse_reads_meta_without_running_the_script() -> None:
    workflow = SavedWorkflow.parse(TRIAGE, path='workflows/triage.py')
    assert workflow == SavedWorkflow(
        name='triage',
        description='Review each file and summarize.',
        source=TRIAGE,
        when_to_use='When asked to review several files.',
        args={'type': 'object', 'properties': {'files': {'type': 'array'}}, 'required': ['files']},
        agents=('reviewer',),
        returns='A summary string.',
        path='workflows/triage.py',
    )
    assert workflow.missing_args({}) == ['files']
    assert workflow.missing_args({'files': []}) == []


@pytest.mark.parametrize(
    ('source', 'error'),
    [
        ('meta = {', 'syntax error on line 1'),
        ('x = 1', 'expected exactly one top-level `meta = {...}` assignment'),
        ("meta = {'name': 'a', 'description': 'd'}\nmeta = {}", 'expected exactly one'),
        ("meta = dict(name='a')", '`meta` must be a dict literal'),
        ("meta = {'name': 'a', 'description': 'd', 'colour': 'red'}", 'invalid `meta`: colour: Extra inputs'),
        ("meta = {'name': 'a'}", 'invalid `meta`: description: Field required'),
        ("meta = ['a']", 'invalid `meta`: meta: Input should be a valid dictionary'),
        ("meta = {'name': 'Bad Name', 'description': 'd'}", "invalid name 'Bad Name'"),
    ],
)
def test_parse_rejects_invalid_files(source: str, error: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        SavedWorkflow.parse(source)
    assert error in str(exc_info.value)


def test_create_renders_a_file_that_parses_back() -> None:
    workflow = _workflow('echo', "await agent(args['text'])", args={'type': 'object'}, agents=['reviewer'])
    assert workflow.source == snapshot("""\
meta = {'name': 'echo',
 'description': 'The echo workflow.',
 'args': {'type': 'object'},
 'agents': ['reviewer']}

await agent(args['text'])
""")
    assert SavedWorkflow.parse(workflow.source) == workflow
    assert workflow.missing_args({}) == []


# --- The library -----------------------------------------------------------------------


async def test_library_reads_valid_files_and_records_the_rest(tmp_path: Path) -> None:
    library_dir = tmp_path / 'workflows'
    _write(library_dir, 'triage.py', TRIAGE)
    _write(library_dir, 'notes.md', 'not a workflow')
    (library_dir / 'nested').mkdir()
    _write(library_dir, 'misnamed.py', "meta = {'name': 'other', 'description': 'd'}")
    _write(library_dir, 'ghost.py', "meta = {'name': 'ghost', 'description': 'd', 'agents': ['ghost']}")
    (library_dir / 'binary.py').write_bytes(b'\xff\xfe')
    _write(tmp_path / 'more', 'triage.py', TRIAGE)

    with pytest.warns(UserWarning, match='Skipping saved workflow') as warned:
        library = await WorkflowLibrary.load(
            _workspace(tmp_path), ['workflows', 'missing', 'more'], agent_names={'reviewer'}
        )
    assert list(library.workflows) == ['triage']
    root = str(tmp_path.resolve())
    assert {Path(path).name: error.replace(root, '<tmp>') for path, error in library.errors.items()} == snapshot(
        {
            'binary.py': "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
            'ghost.py': 'names sub-agents missing from the catalog: ghost',
            'misnamed.py': "`meta[\"name\"]` is 'other' but the file is named 'misnamed'",
            'triage.py': 'repeats the name of <tmp>/workflows/triage.py',
        }
    )
    assert len(warned) == 4


async def test_library_renders_an_available_workflows_block() -> None:
    library = WorkflowLibrary(workflows={'triage': SavedWorkflow.parse(TRIAGE), 'bare': _workflow('bare', '1')})
    assert library.render() == snapshot("""\
<available_workflows>
Saved workflows: run one with the `run_workflow` tool by `name` (plus `args`), or from inside a script with `await workflow(name, args)`.
<workflow>
name: triage
description: Review each file and summarize.
when_to_use: When asked to review several files.
args: {"properties": {"files": {"type": "array"}}, "required": ["files"], "type": "object"}
returns: A summary string.
</workflow>
<workflow>
name: bare
description: The bare workflow.
</workflow>
</available_workflows>\
""")
    assert WorkflowLibrary().render() is None


# --- The capability --------------------------------------------------------------------------


def _instructions(result_messages: list[Any]) -> str:
    return '\n'.join(
        part.content
        for message in result_messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ) + '\n'.join(message.instructions or '' for message in result_messages if isinstance(message, ModelRequest))


async def test_capability_lists_saved_workflows_and_runs_one_by_name(tmp_path: Path) -> None:
    _write(tmp_path / 'workflows', 'triage.py', TRIAGE)
    agent = Agent[object, str](
        scripted_model([('run_workflow', {'name': 'triage', 'args': {'files': ['a.py', 'b.py']}})]),
        capabilities=[DynamicWorkflow[object](agents=[echo_agent('reviewer')])],
    )
    result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
    assert result.output == 'reviewer:review a.py | reviewer:review b.py'
    assert '<available_workflows>' in _instructions(result.all_messages())

    capability = DynamicWorkflow[object](agents=[echo_agent('reviewer')], workflows=[Path('elsewhere'), 'workflows'])
    agent = Agent[object, str](
        scripted_model([('run_workflow', {'name': 'triage', 'args': {'files': ['c.py']}})]), capabilities=[capability]
    )
    result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
    assert result.output == 'reviewer:review c.py'
    assert '<available_workflows>' in _instructions(result.all_messages())


async def test_capability_without_workflow_files_lists_nothing(tmp_path: Path) -> None:
    for workflows, workspace in [('workflows', LocalWorkspaceBackend(tmp_path)), (None, None), ('workflows', None)]:
        with capture_run_messages() as messages:
            agent = Agent[object, str](
                scripted_model([('run_workflow', {'code': "await reviewer(task='x')"})]),
                capabilities=[DynamicWorkflow[object](agents=[echo_agent('reviewer')], workflows=workflows)],
            )
            result = await agent.run('go', workspace=workspace)
        assert result.output == 'reviewer:x'
        assert '<available_workflows>' not in _instructions(messages)


async def test_tool_schema_offers_name_and_args_only_with_a_library() -> None:
    plain = DynamicWorkflowToolset[object](agents=[WorkflowAgent(echo_agent('reviewer'))])
    with_library = _toolset()
    plain_schema = (await plain.get_tools(run_ctx()))['run_workflow'].tool_def.parameters_json_schema
    library_schema = (await with_library.get_tools(run_ctx()))['run_workflow'].tool_def.parameters_json_schema
    assert list(plain_schema['properties']) == ['code']
    assert plain_schema['required'] == ['code']
    assert list(library_schema['properties']) == ['code', 'name', 'args']
    assert 'required' not in library_schema


# --- Running by name ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ('tool_args', 'error'),
    [
        ({}, 'Pass exactly one of `code`'),
        ({'code': '1', 'name': 'triage'}, 'Pass exactly one of `code`'),
        ({'name': 'nope'}, "unknown saved workflow 'nope'; available: triage"),
        ({'name': 'triage'}, "workflow 'triage' is missing required args: files"),
    ],
)
async def test_run_by_name_validates_before_spending_budget(tool_args: dict[str, Any], error: str) -> None:
    ts = _toolset(SavedWorkflow.parse(TRIAGE))
    with pytest.raises(ModelRetry) as exc_info:
        await call_workflow_tool(ts, tool_args)
    assert error in str(exc_info.value)
    assert ts._call_count == 0  # pyright: ignore[reportPrivateUsage]


async def test_run_by_name_records_a_span(tmp_path: Path) -> None:
    tracer, exporter = recording_tracer()
    ts = _toolset(SavedWorkflow.parse(TRIAGE))
    ctx = run_ctx(tracer=tracer, trace_include_content=True)
    await call_workflow_tool(ts, {'name': 'triage', 'args': {'files': ['a.py']}}, ctx)
    [span] = spans_named(exporter, 'dynamic_workflow.workflow')
    assert dict(span.attributes or {}) == {
        'dynamic_workflow.name': 'triage',
        'dynamic_workflow.depth': 1,
        'dynamic_workflow.args': '{"files": ["a.py"]}',
        'dynamic_workflow.outcome': 'ok',
    }


async def test_saved_workflow_errors_are_retries() -> None:
    ts = _toolset(_workflow('broken', "raise ValueError('boom')"))
    with pytest.raises(ModelRetry, match='Runtime error in workflow') as exc_info:
        await call_workflow_tool(ts, {'name': 'broken'})
    assert 'boom' in str(exc_info.value)


# --- Nesting ---------------------------------------------------------------------------------


async def test_workflow_runs_a_saved_workflow_inline_with_shared_budget() -> None:
    tracer, exporter = recording_tracer()
    ts = _toolset(SavedWorkflow.parse(TRIAGE), max_agent_calls=10)
    code = "summary = await workflow('triage', {'files': ['a.py', 'b.py']})\n[summary, budget()['used']]"
    out = await call_workflow_tool(ts, {'code': code}, run_ctx(tracer=tracer))
    assert out == ['reviewer:review a.py | reviewer:review b.py', 2]
    [span] = spans_named(exporter, 'dynamic_workflow.workflow')
    assert dict(span.attributes or {}) == {
        'dynamic_workflow.name': 'triage',
        'dynamic_workflow.depth': 1,
        'dynamic_workflow.outcome': 'ok',
    }


async def test_nested_workflows_can_run_in_parallel() -> None:
    echo = _workflow('echo', "await agent(args['text'])")
    ts = _toolset(echo, max_concurrent_agents=1)
    out = await call_workflow_tool(ts, {'code': "await parallel([workflow('echo', {'text': t}) for t in 'abc'])"})
    assert out == ['reviewer:a', 'reviewer:b', 'reviewer:c']


@pytest.mark.parametrize(
    ('code', 'error'),
    [
        ("await workflow('nope')", "unknown saved workflow 'nope'"),
        ("import json\nawait workflow(json.loads('1'))", 'workflow name must be a string, got int'),
        ("await workflow('echo', {'x': {1, 2}})", "workflow 'echo' args must be a JSON object"),
        ("await workflow('loop')", "workflow 'loop' would call itself: loop -> loop"),
        ("await workflow('deep', {'n': 0})", "workflow 'deep3' would nest deeper than 2 saved workflows"),
        ("await workflow('broken')", "workflow 'broken' failed:"),
        ("await workflow('mistyped')", "workflow 'mistyped' failed:"),
    ],
)
async def test_nested_workflow_failures_raise_runtime_error(code: str, error: str) -> None:
    ts = _toolset(
        _workflow('echo', "await agent(args['text'])"),
        _workflow('loop', "await workflow('loop')"),
        _workflow('deep', "await workflow('deep2')"),
        _workflow('deep2', "await workflow('deep3')"),
        _workflow('deep3', '1'),
        _workflow('broken', "raise ValueError('boom')"),
        _workflow('mistyped', 'await agent(1)'),
        max_workflow_depth=2,
    )
    guarded = 'try:\n' + ''.join(f'    {line}\n' for line in code.splitlines())
    caught = await call_workflow_tool(ts, {'code': guarded + 'except Exception as e:\n    r = str(e)\nr'})
    assert error in caught


async def test_budget_exhausted_in_a_nested_workflow_ends_the_call() -> None:
    ts = _toolset(SavedWorkflow.parse(TRIAGE), max_agent_calls=1)
    out = await call_workflow_tool(ts, {'code': "await workflow('triage', {'files': ['a.py', 'b.py', 'c.py']})"})
    assert 'exhausted its sub-agent call budget (1)' in out['error']


# --- Saving ------------------------------------------------------------------------------------


async def _saving_toolset(
    root: Path, *, workspace: Workspace | None = None
) -> tuple[DynamicWorkflowToolset[object], Any]:
    ts = _toolset(save_directory='workflows')
    ctx = run_ctx(workspace=workspace or _workspace(root))
    return await ts.for_run(ctx), ctx


async def test_save_workflow_writes_a_file_the_run_can_use(tmp_path: Path) -> None:
    ts, ctx = await _saving_toolset(tmp_path)
    save = {
        'tool_name': 'save_workflow',
        'name': 'echo',
        'description': 'Echo the text.',
        'code': "await agent(args['text'])",
        'args': {'type': 'object', 'required': ['text']},
        'agents': ['reviewer'],
    }
    assert (await ts.get_tools(ctx))['save_workflow'].tool_def.sequential
    message = await call_workflow_tool(ts, dict(save), ctx)
    assert message.replace(str(tmp_path.resolve()), '<tmp>') == snapshot(
        "Saved workflow 'echo' to <tmp>/workflows/echo.py. Run it with `run_workflow` by name, or from a script with `await workflow('echo', args)`."
    )
    saved = SavedWorkflow.parse((tmp_path / 'workflows' / 'echo.py').read_text())
    assert (saved.name, saved.agents) == ('echo', ('reviewer',))
    assert await call_workflow_tool(ts, {'name': 'echo', 'args': {'text': 'hi'}}, ctx) == 'reviewer:hi'

    with pytest.raises(ModelRetry, match="A workflow named 'echo' is already saved"):
        await call_workflow_tool(ts, dict(save), ctx)
    await call_workflow_tool(ts, {**save, 'description': 'Echo, again.', 'overwrite': True}, ctx)
    assert SavedWorkflow.parse((tmp_path / 'workflows' / 'echo.py').read_text()).description == 'Echo, again.'


@pytest.mark.parametrize(
    ('overrides', 'error'),
    [
        ({'agents': ['ghost']}, 'unknown sub-agents ghost; available: reviewer'),
        ({'name': 'Bad Name'}, "invalid name 'Bad Name'"),
        ({'code': "meta = {'name': 'x', 'description': 'y'}"}, 'expected exactly one top-level'),
    ],
)
async def test_save_workflow_rejects_invalid_workflows(tmp_path: Path, overrides: dict[str, Any], error: str) -> None:
    ts, ctx = await _saving_toolset(tmp_path)
    save = {'tool_name': 'save_workflow', 'name': 'echo', 'description': 'Echo.', 'code': '1', **overrides}
    with pytest.raises(ModelRetry) as exc_info:
        await call_workflow_tool(ts, save, ctx)
    assert error in str(exc_info.value)
    assert not (tmp_path / 'workflows').exists()


async def test_save_workflow_reports_workspace_failures(tmp_path: Path) -> None:
    (tmp_path / 'workflows').write_text('a file, not a directory')
    ts, ctx = await _saving_toolset(tmp_path)
    save = {'tool_name': 'save_workflow', 'name': 'echo', 'description': 'Echo.', 'code': '1'}
    with pytest.raises(ToolFailed, match=r'Cannot save workflow: .*workflows'):
        await call_workflow_tool(ts, save, ctx)


async def test_save_workflow_reports_workspace_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ts, ctx = await _saving_toolset(tmp_path)

    async def time_out(path: str, content: str) -> None:
        raise WorkspaceTimeoutError('The write timed out.')

    monkeypatch.setattr(ctx.workspace, 'write_text', time_out)
    save = {'tool_name': 'save_workflow', 'name': 'echo', 'description': 'Echo.', 'code': '1'}
    with pytest.raises(ToolFailed, match=r'The write timed out\.'):
        await call_workflow_tool(ts, save, ctx)


async def test_save_workflow_needs_a_writable_workspace(tmp_path: Path) -> None:
    read_only, _ = await _saving_toolset(tmp_path, workspace=ReadOnlyWorkspace(_workspace(tmp_path)))
    unattached = await _toolset(save_directory='workflows').for_run(run_ctx())
    not_configured = await _toolset().for_run(run_ctx(workspace=_workspace(tmp_path)))
    for ts in (read_only, unattached, not_configured):
        assert list(await ts.get_tools(run_ctx())) == ['run_workflow']


async def test_capability_offers_save_workflow_and_lists_the_result_next_run(tmp_path: Path) -> None:
    save = {'name': 'echo', 'description': 'Echo the text.', 'code': "await agent(args['text'])"}
    capability = DynamicWorkflow[object](agents=[echo_agent('reviewer')])
    agent = Agent[object, str](scripted_model([('save_workflow', save)]), capabilities=[capability])
    await agent.run('save it', workspace=LocalWorkspaceBackend(tmp_path))
    assert (tmp_path / 'workflows' / 'echo.py').exists()

    agent = Agent[object, str](
        scripted_model([('run_workflow', {'name': 'echo', 'args': {'text': 'hi'}})]), capabilities=[capability]
    )
    result = await agent.run('run it', workspace=LocalWorkspaceBackend(tmp_path))
    assert result.output == 'reviewer:hi'
    assert 'name: echo' in _instructions(result.all_messages())
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    assert [part.tool_name for part in returns] == ['run_workflow']


async def test_nested_workflow_crash_is_a_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tiny `request_timeout` plus an infinite loop crashes the nested worker for real.
    monkeypatch.setattr(
        'pydantic_ai_harness._monty_exec.AsyncMonty', functools.partial(AsyncMonty, request_timeout=0.5)
    )
    ts = _toolset(_workflow('spin', 'while True:\n    pass'))
    code = "try:\n    await workflow('spin')\nexcept RuntimeError as e:\n    r = str(e)\nr"
    assert await call_workflow_tool(ts, {'code': code}) == "workflow 'spin' crashed the sandbox worker"


async def test_nested_workflow_panic_is_a_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class PanicException(BaseException):
        """Named to match the pyo3 panic class `is_sandbox_panic` recognizes."""

    execute = WorkflowRun.execute

    async def panic_when_nested(self: WorkflowRun[object], session: Any, code: str, *, frame: Frame, args: Any) -> Any:
        if frame.stack:
            raise PanicException('sandbox panic')
        return await execute(self, session, code, frame=frame, args=args)

    monkeypatch.setattr(WorkflowRun, 'execute', panic_when_nested)
    ts = _toolset(_workflow('echo', '1'))
    code = "try:\n    await workflow('echo')\nexcept RuntimeError as e:\n    r = str(e)\nr"
    assert await call_workflow_tool(ts, {'code': code}) == "workflow 'echo' aborted inside the sandbox"


async def test_nested_workflow_base_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stop(BaseException):
        pass

    execute = WorkflowRun.execute

    async def cancel_when_nested(self: WorkflowRun[object], session: Any, code: str, *, frame: Frame, args: Any) -> Any:
        if frame.stack:
            raise Stop
        return await execute(self, session, code, frame=frame, args=args)

    monkeypatch.setattr(WorkflowRun, 'execute', cancel_when_nested)
    with pytest.raises(Stop):
        await call_workflow_tool(_toolset(_workflow('echo', '1')), {'code': "await workflow('echo')"})


def test_workflows_needs_a_directory_or_none() -> None:
    no_directories: list[str] = []
    with pytest.raises(UserError, match='`workflows` needs at least one directory; pass `None`'):
        DynamicWorkflow[object](agents=[echo_agent('reviewer')], workflows=no_directories)


async def test_an_empty_library_says_so() -> None:
    with pytest.raises(ModelRetry, match="unknown saved workflow 'nope'; available: none are saved"):
        await call_workflow_tool(_toolset(), {'name': 'nope'})


async def test_workflows_none_turns_saved_workflows_off(tmp_path: Path) -> None:
    capability = DynamicWorkflow[object](agents=[echo_agent('reviewer')], workflows=None)
    ctx = run_ctx(workspace=_workspace(tmp_path))
    assert await capability.for_run(ctx) is capability
    assert capability.get_instructions() is None
    toolset = capability.get_toolset()
    assert isinstance(toolset, DynamicWorkflowToolset)
    tools = await (await toolset.for_run(ctx)).get_tools(ctx)
    assert list(tools) == ['run_workflow']
    assert list(tools['run_workflow'].tool_def.parameters_json_schema['properties']) == ['code']
