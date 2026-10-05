"""Render task identity without mutating capability-owned toolsets."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
from render.workflows import TaskContext, Workflows

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext, RecordingWorkflows, renderless_environment, run_agent_in_task

_TASK_NAME_PROBE = """
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import RenderWorkflows, ToolOutputLimits
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from tests.harness.render.conftest import RecordingWorkflows

app = RecordingWorkflows()

worker = Agent(TestModel(), name='worker', description='Does the work')
async def explicit_tool():
    return 'ok'
Agent(
    TestModel(),
    name='support',
    deps_type=type(None),
    toolsets=[FunctionToolset([explicit_tool], id='explicit-tools')],
    capabilities=[
        SubAgents(agents=[SubAgent(worker)], agent_folders=None),
        ToolOutputLimits(),
        RenderWorkflows(
            app, deps_type=type(None),
            resolve_tool_options=lambda op, tool, name: False if name in {'delegate_task', 'read_tool_result'} else None,
        ),
    ],
)
print('\\n'.join(sorted({name for name in app.registered_task_names if '__function_toolset__' in name and '<agent>' not in name})))
"""


def test_importing_render_workflows_does_not_add_an_optional_mcp_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; '
                'from pydantic_ai.agent import AbstractAgent; '
                'before = "pydantic_ai.mcp" in sys.modules; '
                'from pydantic_ai_harness import RenderWorkflows; '
                'assert AbstractAgent and RenderWorkflows; '
                'assert ("pydantic_ai.mcp" in sys.modules) == before'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=renderless_environment(),
    )

    assert completed.returncode == 0, completed.stderr


class Notes(AbstractCapability[None]):
    """Retain a capability-owned toolset so its public ID can be observed."""

    def __init__(self, *, id: str | None = 'notes', toolset_id: str | None = None) -> None:
        self.id = id

        async def note(text: str) -> str:
            return text

        self.toolset = FunctionToolset[None]([note], id=toolset_id)

    def get_toolset(self) -> AbstractToolset[None]:
        return self.toolset


def build_agent(
    capability: AbstractCapability[None],
    *,
    app: Workflows | None = None,
    toolsets: list[AbstractToolset[None]] | None = None,
    call_tools: list[str] | None = None,
) -> tuple[Agent[None, str], RenderWorkflows[None]]:
    render_workflows = RenderWorkflows[None](app if app is not None else Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel() if call_tools is None else TestModel(call_tools=call_tools),
        name='support',
        deps_type=type(None),
        toolsets=toolsets,
        capabilities=[capability, render_workflows],
    )
    return agent, render_workflows


async def recorded_task_names(
    agent: Agent[None, str],
    render_workflows: RenderWorkflows[None],
    prompt: str,
) -> list[str]:
    context = RecordingTaskContext()
    await run_agent_in_task(agent, render_workflows, context, prompt=prompt)
    return context.task_names


def probe_task_names() -> list[str]:
    """Return the function-toolset task names a fresh interpreter registers."""
    completed = subprocess.run(
        [sys.executable, '-c', _TASK_NAME_PROBE],
        check=False,
        capture_output=True,
        text=True,
        env=renderless_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


async def test_a_toolset_that_brought_its_own_id_registers_its_tasks_under_it() -> None:
    agent, render_workflows = build_agent(Notes(toolset_id='handwritten'), call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    # Task names are persisted journal data, so the name an `id` produces is pinned here:
    # a rename strands in-flight workflows recorded against the old one. A toolset that
    # was named keeps its own name; the capability's id never overrides it.
    assert 'support__function_toolset__handwritten.call_tool' in names


def test_an_unnamed_capability_toolset_is_rejected_before_registration() -> None:
    notes = Notes(id='web_search')
    app = RecordingWorkflows()
    with pytest.raises(UserError, match='unique `id`'):
        build_agent(notes, app=app)
    assert notes.toolset.id is None
    assert app.registered_task_names == []


async def test_explicit_capability_and_user_toolset_ids_register_separately() -> None:

    async def recall(topic: str) -> str:
        return topic

    agent, render_workflows = build_agent(
        Notes(id='notes', toolset_id='capability-notes'),
        toolsets=[FunctionToolset[None]([recall], id='notes')],
        call_tools=['note', 'recall'],
    )

    names = await recorded_task_names(agent, render_workflows, 'remember this')

    assert 'support__function_toolset__capability-notes.call_tool' in names
    assert 'support__function_toolset__notes.call_tool' in names


class ExternallyAnsweredNotes(Notes):
    """Contribute one unnamed function leaf and one externally answered leaf."""

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        self.toolset = FunctionToolset[None]([note])
        return CombinedToolset([self.toolset, ExternalToolset[None]([ToolDefinition(name='answered_elsewhere')])])


def test_an_external_leaf_does_not_hide_an_unnamed_function_toolset() -> None:
    app = RecordingWorkflows()
    with pytest.raises(UserError, match='unique `id`'):
        build_agent(ExternallyAnsweredNotes(), app=app)
    assert app.registered_task_names == []


class TwoToolsetNotes(Notes):
    """Retain two unnamed leaves contributed by one capability."""

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        async def recall(topic: str) -> str:
            return topic

        self.leaves = [FunctionToolset[None]([note]), FunctionToolset[None]([recall])]
        return CombinedToolset(self.leaves)


def test_two_unnamed_capability_leaves_are_rejected_before_registration() -> None:
    app = RecordingWorkflows()
    with pytest.raises(UserError, match='unique `id`'):
        build_agent(TwoToolsetNotes(), app=app)
    assert app.registered_task_names == []


def test_a_toolset_the_user_attached_without_an_id_is_refused_on_its_own_terms() -> None:
    async def recall(topic: str) -> str:
        return topic

    app = RecordingWorkflows()

    with pytest.raises(UserError, match='unique `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall])])

    assert app.registered_task_names == []


def test_unnamed_leaf_under_unnamed_capability_is_rejected_before_registration() -> None:
    app = RecordingWorkflows()
    with pytest.raises(UserError, match='unique `id`'):
        build_agent(Notes(id=None), app=app)
    assert app.registered_task_names == []


def test_two_toolsets_that_already_share_an_id_are_still_a_collision() -> None:
    async def recall(topic: str) -> str:
        return topic

    app = RecordingWorkflows()

    with pytest.raises(UserError, match='Two toolsets have the same `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall], id='notes')])

    assert app.registered_task_names == []


def test_two_independent_agents_register_the_same_explicit_task_names() -> None:
    build_agent(Notes(id='web_search', toolset_id='web-search'), app=(first_app := RecordingWorkflows()))
    build_agent(Notes(id='web_search', toolset_id='web-search'), app=(second_app := RecordingWorkflows()))

    assert 'support__function_toolset__web-search.call_tool' in first_app.registered_task_names
    assert first_app.registered_task_names == second_app.registered_task_names


def test_explicit_names_are_process_stable_with_helper_opt_outs() -> None:
    names = probe_task_names()

    assert names == probe_task_names()
    assert names == [
        'support__function_toolset__explicit-tools.call_tool',
        'support__function_toolset__explicit-tools.validate_args',
    ]


async def test_matching_id_does_not_admit_a_runtime_toolset() -> None:
    def lookup() -> str:
        return 'registered'

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        TestModel(call_tools=[]),
        name='static-toolset',
        deps_type=type(None),
        toolsets=[FunctionToolset([lookup], id='lookup')],
        capabilities=[runtime],
    )

    @runtime.task
    async def run_agent(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('read', toolsets=[FunctionToolset([lookup], id='lookup')])).output

    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    with pytest.raises(UserError, match='cannot be added at runtime'):
        await pending
    assert context.task_names == []


@pytest.mark.parametrize('capability_owned', [False, True])
def test_tool_opt_out_does_not_bypass_core_toolset_identity_checks(capability_owned: bool) -> None:
    app = RecordingWorkflows()
    notes = Notes()
    runtime = RenderWorkflows[None](app, resolve_tool_options=lambda operation, tool, name: False)
    with pytest.raises(UserError, match='unique `id`'):
        Agent(
            TestModel(),
            name='unnamed-opt-out',
            deps_type=type(None),
            toolsets=[] if capability_owned else [notes.toolset],
            capabilities=[notes, runtime] if capability_owned else [runtime],
        )
    assert app.registered_task_names == []


class ResourceListingToolset(ExternalToolset[None]):
    """A non-MCP toolset whose resource API must not change core's classification."""

    async def list_resources(self) -> list[str]:
        return []


def test_list_resources_does_not_make_a_toolset_an_mcp_toolset() -> None:
    app = RecordingWorkflows()
    Agent(
        TestModel(),
        name='external-resources',
        deps_type=type(None),
        toolsets=[ResourceListingToolset([ToolDefinition(name='external')])],
        capabilities=[RenderWorkflows[None](app)],
    )
    assert 'external-resources__model.request' in app.registered_task_names
    assert not [name for name in app.registered_task_names if '__mcp_server__' in name]
