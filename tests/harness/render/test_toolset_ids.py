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

from .conftest import RecordingTaskContext
from .test_render_workflows import RegistrationRecordingWorkflows

_TASK_NAME_PROBE = """
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models.test import TestModel
from render.workflows import Workflows

from pydantic_ai_harness import RenderWorkflows, ToolOutputLimits
from pydantic_ai_harness.subagents import SubAgent, SubAgents

names = []


class RecordingWorkflows(Workflows):
    def task(self, func=None, *, name=None, retry=None, timeout_seconds=None, plan=None):
        decorator = super().task(name=name, retry=retry, timeout_seconds=timeout_seconds, plan=plan)

        def record(target):
            definition = decorator(target)
            names.append(definition.name)
            return definition

        return record if func is None else record(func)


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
        RenderWorkflows(RecordingWorkflows(), deps_type=type(None)),
    ],
)
print('\\n'.join(sorted({name for name in names if '__function_toolset__' in name and '<agent>' not in name})))
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
    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending_result = run_agent.func(context, prompt)
    assert inspect.isawaitable(pending_result)
    await pending_result
    return context.task_names


def probe_task_names() -> list[str]:
    """Return the function-toolset task names a fresh interpreter registers."""
    completed = subprocess.run(
        [sys.executable, '-c', _TASK_NAME_PROBE],
        check=False,
        capture_output=True,
        text=True,
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


async def test_an_unnamed_capability_toolset_stays_unnamed_and_runs_inline() -> None:
    notes = Notes(id='web_search')
    agent, render_workflows = build_agent(notes, call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert notes.toolset.id is None
    assert not [name for name in names if '__function_toolset__' in name]


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


async def test_unnamed_capability_leaf_stays_inline_alongside_external_leaf() -> None:
    notes = ExternallyAnsweredNotes(id='notes')
    agent, render_workflows = build_agent(notes, call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert notes.toolset.id is None
    assert not [name for name in names if '__function_toolset__notes' in name]


class TwoToolsetNotes(Notes):
    """Retain two unnamed leaves contributed by one capability."""

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        async def recall(topic: str) -> str:
            return topic

        self.leaves = [FunctionToolset[None]([note]), FunctionToolset[None]([recall])]
        return CombinedToolset(self.leaves)


async def test_two_unnamed_capability_leaves_stay_inline_and_unnamed() -> None:
    notes = TwoToolsetNotes(id='notes')
    agent, render_workflows = build_agent(notes, call_tools=['note', 'recall'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert [leaf.id for leaf in notes.leaves] == [None, None]
    assert not [name for name in names if '__function_toolset__' in name]


def test_a_toolset_the_user_attached_without_an_id_is_refused_on_its_own_terms() -> None:
    """Nothing derives a name for a toolset the user holds: they can pass one.

    The capability-derived name exists because the toolset it names is unreachable. This
    one is reachable, so the refusal says to name it rather than inventing a name for it.
    """

    async def recall(topic: str) -> str:
        return topic

    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match='needs a unique `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall])])

    assert app.registered_task_names == []


async def test_unnamed_leaf_under_unnamed_capability_stays_inline() -> None:
    notes = Notes(id=None)
    agent, render_workflows = build_agent(notes, call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert notes.toolset.id is None
    assert not [name for name in names if '__function_toolset__' in name]


def test_two_toolsets_that_already_share_an_id_are_still_a_collision() -> None:
    async def recall(topic: str) -> str:
        return topic

    # Both of these were named by whoever owns them, so there is nothing to derive and
    # nothing to disambiguate: a genuine clash is Pydantic AI's answer, given earlier.
    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match='Two toolsets have the same `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall], id='notes')])

    assert app.registered_task_names == []


def test_two_independent_agents_register_the_same_explicit_task_names() -> None:
    first, _ = build_agent(
        Notes(id='web_search', toolset_id='web-search'), app=(first_app := RegistrationRecordingWorkflows())
    )
    second, _ = build_agent(
        Notes(id='web_search', toolset_id='web-search'), app=(second_app := RegistrationRecordingWorkflows())
    )

    assert first is not second
    assert 'support__function_toolset__web-search.call_tool' in first_app.registered_task_names
    assert first_app.registered_task_names == second_app.registered_task_names


def test_explicit_names_are_process_stable_and_shared_capabilities_stay_inline() -> None:
    names = probe_task_names()

    assert names == probe_task_names()
    assert names == [
        'support__function_toolset__explicit-tools.call_tool',
        'support__function_toolset__explicit-tools.validate_args',
    ]
