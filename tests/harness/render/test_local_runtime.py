"""Manual integration coverage against the keyless local Render runtime.

Install Render CLI 2.28.0 or later and run this module with
`PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME=1 uv run pytest tests/harness/render/test_local_runtime.py`.
These tests are skipped by default, including in CI. They exercise nested agents, JSON dependencies,
task lineage, distinct worker processes, usage and event propagation, tool retries, and worker tracing.
They do not establish hosted storage sharing, failure recovery, or performance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import TypeAdapter

from .conftest import LocalRenderRuntime, LocalTaskRun
from .runtime_app import RootTaskResult, TracingTaskResult

ROOT_TASK = 'run-local-runtime-agent'
PARENT_MODEL_TASK = 'runtime-parent__model.request'
CHILD_MODEL_TASK = 'runtime-child__model.request_stream'
CHILD_TOOL_TASK = 'runtime-child__function_toolset__<agent>.call_tool'
GRANDCHILD_MODEL_TASK = 'runtime-grandchild__model.request'
GRANDCHILD_TOOL_TASK = 'runtime-grandchild__function_toolset__<agent>.call_tool'
OPERATION_TASKS = {
    PARENT_MODEL_TASK,
    CHILD_MODEL_TASK,
    CHILD_TOOL_TASK,
    GRANDCHILD_MODEL_TASK,
    GRANDCHILD_TOOL_TASK,
}

_ROOT_RESULTS = TypeAdapter(list[RootTaskResult])


def _runs_for_root(runtime: LocalRenderRuntime, task_name: str, root_id: str) -> list[LocalTaskRun]:
    return [run for run in runtime.list_runs(task_name) if run.root_task_run_id == root_id]


def test_nested_agents_run_as_lineaged_local_render_operations(
    local_render_runtime: LocalRenderRuntime,
) -> None:
    runtime = local_render_runtime
    registered_names = {task.name for task in runtime.list_tasks()}
    assert {ROOT_TASK, *OPERATION_TASKS} <= registered_names

    controller_pid = os.getpid()
    started = runtime.start_task(
        ROOT_TASK,
        f'["exercise nested process isolation", {{"prefix": "serializable-deps", "controller_pid": {controller_pid}}}]',
    )
    completed = runtime.wait_for_run(started.id)
    assert completed.status == 'completed', runtime.logs()
    results = _ROOT_RESULTS.validate_python(completed.results)
    assert len(results) == 1
    result = results[0]
    assert result.controller_pid == controller_pid
    assert result.deps_prefix == 'serializable-deps'
    sibling_tools = result.output.child.sibling_tools
    assert [tool.value for tool in sibling_tools] == ['serializable-deps:alpha', 'serializable-deps:beta']
    assert all(tool.retry_count == 0 for tool in sibling_tools)
    assert result.output.child.grandchild.tool.value == 'serializable-deps:grandchild'
    assert result.output.child.grandchild.tool.retry_count == 1
    assert result.usage_markers == {
        'runtime_remote_marker': 7,
        'alpha_marker': 2,
        'beta_marker': 5,
    }
    assert [(event.label, event.sequence) for event in result.events if event.label == 'alpha'] == [
        ('alpha', 1),
        ('alpha', 2),
    ]
    assert [(event.label, event.sequence) for event in result.events if event.label == 'beta'] == [
        ('beta', 1),
        ('beta', 2),
    ]
    assert len(result.events) == 4

    process_ids = {
        controller_pid,
        result.root_pid,
        result.output.model_pid,
        result.output.child.model_pid,
        *(tool.pid for tool in sibling_tools),
        result.output.child.grandchild.model_pid,
        result.output.child.grandchild.tool.pid,
    }
    assert len(process_ids) == 8, process_ids

    runs_by_task = {task_name: _runs_for_root(runtime, task_name, started.id) for task_name in OPERATION_TASKS}
    assert len(runs_by_task[PARENT_MODEL_TASK]) == 2
    assert len(runs_by_task[CHILD_MODEL_TASK]) == 3
    assert len(runs_by_task[CHILD_TOOL_TASK]) == 2
    assert len(runs_by_task[GRANDCHILD_MODEL_TASK]) == 3
    assert len(runs_by_task[GRANDCHILD_TOOL_TASK]) == 2

    operation_runs = [run for runs in runs_by_task.values() for run in runs]
    assert all(run.status == 'completed' for run in operation_runs)
    assert all(run.root_task_run_id == started.id for run in operation_runs)
    assert all(run.parent_task_run_id == started.id for run in operation_runs)
    for name in registered_names:
        if '__function_toolset__sub_agents' in name:
            assert _runs_for_root(runtime, name, started.id) == []


def test_tracer_works_in_a_separate_render_process(local_render_runtime: LocalRenderRuntime) -> None:
    runtime = local_render_runtime
    started = runtime.start_task('run-local-tracing-agent', '[]')
    completed = runtime.wait_for_run(started.id)
    assert completed.status == 'completed', runtime.logs()
    results = TypeAdapter(list[TracingTaskResult]).validate_python(completed.results)
    assert len(results) == 1
    evidence = results[0]
    assert evidence.span_exported is True
    assert len({os.getpid(), evidence.root_pid, evidence.tool_pid}) == 3
    tool_calls = _runs_for_root(runtime, 'runtime-tracing__function_toolset__<agent>.call_tool', started.id)
    assert len(tool_calls) == 1
    assert tool_calls[0].status == 'completed'


def test_memory_persists_between_separate_workers_with_configured_limits(
    local_render_runtime: LocalRenderRuntime,
    tmp_path: Path,
) -> None:
    """SQLite is shared across local processes; hosted workers require an external store."""
    runtime = local_render_runtime
    database = str(tmp_path / 'memory.sqlite')

    def run(tenant: str, prompt: str, limit: int = 64) -> str:
        started = runtime.start_task('run-local-memory-agent', json.dumps([database, tenant, prompt, limit]))
        completed = runtime.wait_for_run(started.id)
        assert completed.status == 'completed', runtime.logs()
        assert _runs_for_root(runtime, f'runtime-memory-{limit}__function_toolset__memory.call_tool', started.id)
        return TypeAdapter(list[str]).validate_python(completed.results)[0]

    assert run('alice', 'write:abcdefghij') == 'abcdefghij\n'
    assert run('bob', 'write:other tenant') == 'other tenant\n'
    for limit in (4, 8):
        assert run('alice', 'read', limit).startswith('abcdefghij'[:limit] + '\n\n[Truncated:')
    assert run('bob', 'read') == 'other tenant\n'
