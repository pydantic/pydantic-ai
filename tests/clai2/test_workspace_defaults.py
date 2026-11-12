"""CLAI's local default coexists with static and per-run workspace providers."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal, TypeGuard

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    Capability,
    CapabilityFunc,
    CombinedCapability,
    DynamicCapability,
    LocalWorkspace,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.skills import Skills
from pydantic_clai2 import Session
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.runtime.session import WorkspacelessCapability


def is_workspaceless(capability: object) -> TypeGuard[WorkspacelessCapability[None]]:
    return isinstance(capability, WorkspacelessCapability)


def per_run(capability: object) -> CapabilityFunc[None]:
    """The function a built-in plugin's capability resolves each run."""
    assert is_workspaceless(capability), 'a per-run plugin capability supplies no workspace'
    return capability.capability_func


@pytest.mark.parametrize('stock', [False, True])
@pytest.mark.parametrize('contribution', ['none', 'instructions', 'workspace', 'group workspace'])
@pytest.mark.parametrize('on_agent', [False, True])
async def test_dynamic_plugins_do_not_suppress_the_local_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stock: bool,
    contribution: Literal['none', 'instructions', 'workspace', 'group workspace'],
    on_agent: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()
    working_dirs: list[str] = []
    factory_calls = 0

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    class SandboxGroup(CombinedCapability[None]):
        # The group supplies the workspace itself, not through any of its members.
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = LocalWorkspaceBackend(sandbox)
            return backend if ref is None or ref == backend.ref else None

    def dynamic(ctx: RunContext[None]) -> AbstractCapability[None] | None:
        nonlocal factory_calls
        factory_calls += 1
        if contribution == 'workspace':
            return LocalWorkspace(sandbox, read_only=True)
        if contribution == 'group workspace':
            return SandboxGroup([Capability(instructions='Sandboxed.')])
        if contribution == 'instructions':
            return Capability(instructions='Keep working in the configured workspace.')
        return None

    model = TestModel(call_tools=[], custom_output_text='done')
    plugins: list[AgentCapability[None]] = [Coder(repo_context=False, sub_agents=stock), Probe()]
    if stock:
        agent = create_stock_agent(model)
        if on_agent:
            agent = agent.with_plugins([dynamic])
        else:
            plugins.append(dynamic)
    else:
        agent = Agent(model, deps_type=type(None), capabilities=[dynamic] if on_agent else [])
        if not on_agent:
            plugins.append(dynamic)
    session = Session(agent, deps=None, plugins=plugins)
    await session.prompt('first')
    result = await session.prompt('second')
    expected_dir = str((sandbox if contribution.endswith('workspace') else tmp_path).resolve())
    assert working_dirs == [expected_dir, expected_dir]
    assert factory_calls == 2
    assert result.response.workspace_ref == WorkspaceRef(provider='local', id=expected_dir)


@pytest.mark.parametrize('contribution', ['none', 'instructions'])
async def test_a_workspaceless_capability_function_leaves_the_local_default_to_for_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contribution: Literal['none', 'instructions']
) -> None:
    # `Skills` reads its libraries in `for_run`, before a capability function's workspace could be chosen.
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / '.agents' / 'skills' / 'review'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\ndescription: Review the change.\n---\n\nReview it.\n', encoding='utf-8')
    instructions: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        instructions.append(request.instructions or '')
        yield 'done'

    def dynamic(ctx: RunContext[None]) -> AbstractCapability[None] | None:
        return Capability(instructions='From the plugin.') if contribution == 'instructions' else None

    plugins: list[AgentCapability[None]] = [
        Coder(repo_context=False, sub_agents=False),
        Skills[None]('.agents/skills'),
        WorkspacelessCapability[None](dynamic),
    ]
    agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None))
    result = await Session(agent, deps=None, plugins=plugins).prompt('go')
    assert result.response.workspace_ref == WorkspaceRef(provider='local', id=str(tmp_path.resolve()))
    [seen] = instructions
    assert '- review: Review the change.' in seen
    assert ('From the plugin.' in seen) == (contribution == 'instructions')


async def test_a_workspaceless_capability_function_that_returns_a_workspace_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    def dynamic(ctx: RunContext[None]) -> AbstractCapability[None]:
        return LocalWorkspace(sandbox, id='sandbox')

    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    session = Session(agent, deps=None, plugins=[WorkspacelessCapability[None](dynamic)])
    with pytest.raises(UserError, match="A capability's `for_run` changed the workspace"):
        await session.prompt('go')


async def test_static_default_is_available_during_for_run(tmp_path: Path) -> None:
    working_dirs: list[str] = []

    class Probe(AbstractCapability[None]):
        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            working_dirs.append(await ctx.workspace.working_dir())
            return self

    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None), capabilities=[Probe()])
    await Session(agent, deps=None, workspace=tmp_path).prompt('go')
    assert working_dirs == [str(tmp_path.resolve())]


async def test_wrapped_dynamic_plugin_gets_the_local_default(tmp_path: Path) -> None:
    def dynamic(ctx: RunContext[None]) -> None:
        return None

    agent = Agent(
        TestModel(call_tools=[], custom_output_text='done'),
        deps_type=type(None),
        capabilities=[Coder(), DynamicCapability(dynamic).prefix_tools('plugin_')],
    )
    result = await Session(agent, deps=None, workspace=tmp_path).prompt('go')
    assert result.response.workspace_ref == WorkspaceRef(provider='local', id=str(tmp_path.resolve()))
