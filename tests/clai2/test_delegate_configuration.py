"""Configured model menus give Review an independent model and policy workspace."""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace
from pydantic_ai_harness.bubblewrap_sandbox import BubblewrapWorkspace
from pydantic_ai_harness.subagents import DelegationTask, DelegationTasks, DelegationWorkspaceRequest
from pydantic_clai2.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.chains import settings_model
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import ModelProvider, PluginHost
from pydantic_clai2.runtime.tasks import review_workspace
from pydantic_clai2.ui.menus.task_menu import TaskMenu


@pytest.mark.parametrize('reference', ['fixture@work:light', 'chain:review-light'])
async def test_review_resolves_models_settings_and_policy_directory(tmp_path: Path, reference: str) -> None:
    candidate, policy = tmp_path / 'candidate', tmp_path / 'policy'
    candidate.mkdir()
    policy.mkdir()
    (candidate / 'AGENTS.md').write_text('CANDIDATE_POLICY_MUST_NOT_REACH_REVIEW')
    (policy / 'AGENTS.md').write_text('AUTHORITATIVE_POLICY')
    (policy / 'evidence.txt').write_text('immutable evidence')
    calls: list[tuple[str, str]] = []
    child_messages: list[ModelMessage] = []

    async def review(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        child_messages[:] = messages
        assert info.instructions is not None and 'AUTHORITATIVE_POLICY' in info.instructions
        assert 'CANDIDATE_POLICY_MUST_NOT_REACH_REVIEW' not in info.instructions
        tools = {tool.name for tool in info.function_tools}
        assert 'read_file' in tools
        assert not tools & {'shell', 'unsafe', 'delegate_task', 'run_code', 'ask_parent', 'write_file', 'edit_file'}
        assert info.model_settings is not None
        assert info.model_request_parameters.thinking == 'high'
        assert info.model_settings.get('temperature') == 0.25
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield {0: DeltaToolCall(name='read_file', json_args='{"path":"evidence.txt"}')}
        else:
            yield 'review complete'

    child = FunctionModel(stream_function=review, profile=ModelProfile(supports_thinking=True))
    store = SettingsStore(tmp_path / 'config.db')
    store.save_chain(name='review-light', models=['fixture@work:light'])
    store.save_model_settings(reference, {'thinking': 'medium', 'temperature': 0.25})
    console = Console(file=io.StringIO())

    def on_account(name: str, profile: str) -> FunctionModel:
        calls.append((name, profile))
        return child

    provider = ModelProvider(prefix='fixture', resolve=lambda name: child, resolve_profile=on_account)
    resolver = ModelResolver(console=console, store=store, plugins=lambda: {'fixture': provider})
    from pydantic_clai2.cli.command_context import CommandContext

    context = CommandContext(
        settings=Settings(),
        store=store,
        apply_setting=lambda key, value: None,
        settings_model=lambda name: settings_model(store, name),
    )
    settings: dict[str, JsonValue] = {
        'repo_context': False,
        'skill_folders': [],
        'review_agent': True,
        'delegate_models': {'review-light': {'model': reference, 'settings': {'thinking': 'high'}}},
    }
    host = PluginHost[object](
        name='coder',
        console=console,
        settings=settings,
        workspace=lambda: candidate,
        resolve_model=resolver.resolve,
        model_settings=context.model_settings,
    )
    plugin = CoderPlugin[object].from_host(host)

    def parent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task',
                        {
                            'agent_name': 'Review',
                            'task': 'review supplied evidence',
                            'model': 'review-light',
                            'working_directory': '../policy',
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('parent finished')])

    agent = Agent(FunctionModel(parent), deps_type=object, capabilities=plugin.get_capabilities())

    @agent.tool_plain
    def unsafe() -> str:
        raise AssertionError('Review must not inherit this tool')  # pragma: no cover

    async with DelegationTasks().opened() as owner:
        with owner.bind():
            result = await agent.run('PRIVATE_PARENT_PROMPT', workspace=Workspace(LocalWorkspaceBackend(candidate)))
        (record,) = owner.records.values()
        assert record.working_directory == str(policy)
        assert record.resumable is False and record.model == 'review-light'
        assert record.output == 'review complete'
    assert result.output == 'parent finished'
    assert calls and set(calls) == {('light', 'work')}
    assert all('PRIVATE_PARENT_PROMPT' not in str(message.parts) for message in child_messages)
    returns = [part for message in child_messages for part in message.parts if isinstance(part, ToolReturnPart)]
    assert 'immutable evidence' in str(returns[-1].content)


@pytest.mark.parametrize('name', ['self', 'Explore', 'Plan', 'disk-agent'])
async def test_directory_override_is_review_only(tmp_path: Path, name: str) -> None:
    ctx = RunContext(
        deps=object(), model=TestModel(), usage=RunUsage(), workspace=Workspace(LocalWorkspaceBackend(tmp_path))
    )
    with pytest.raises(ModelRetry, match='Only Review'):
        await review_workspace(ctx, DelegationWorkspaceRequest(agent_name=name, directory=str(tmp_path)))


@pytest.mark.parametrize('directory', ['', 'missing', 'file', 'bad\x00path'])
async def test_review_rejects_invalid_directories(tmp_path: Path, directory: str) -> None:
    (tmp_path / 'file').write_text('not a directory')
    ctx = RunContext(
        deps=object(), model=TestModel(), usage=RunUsage(), workspace=Workspace(LocalWorkspaceBackend(tmp_path))
    )
    with pytest.raises(ModelRetry, match='directory'):
        await review_workspace(ctx, DelegationWorkspaceRequest(agent_name='Review', directory=directory))


async def test_review_rejects_missing_parent_workspace_and_keeps_read_only(tmp_path: Path) -> None:
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage())
    with pytest.raises(ModelRetry, match='local parent workspace'):
        await review_workspace(ctx, DelegationWorkspaceRequest(agent_name='Review', directory=str(tmp_path)))
    ctx.workspace = ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
    assert await (
        await review_workspace(ctx, DelegationWorkspaceRequest(agent_name='Review', directory='.'))
    ).working_dir() == str(tmp_path)


@pytest.mark.parametrize('read_only', [False, True])
async def test_review_rejects_sandbox_wrappers_over_local_backends(tmp_path: Path, read_only: bool) -> None:
    project, hidden = tmp_path / 'project', tmp_path / 'hidden'
    project.mkdir()
    hidden.mkdir()
    parent: Workspace = BubblewrapWorkspace(
        Workspace(LocalWorkspaceBackend(project)), bwrap_args=['--tmpfs', str(hidden)]
    )
    if read_only:
        parent = ReadOnlyWorkspace(parent)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), workspace=parent)
    with pytest.raises(ModelRetry, match='policy wrappers'):
        await review_workspace(ctx, DelegationWorkspaceRequest(agent_name='Review', directory=str(hidden)))


@pytest.mark.parametrize('sub_agents', [False, True])
@pytest.mark.parametrize('menu', [False, True])
@pytest.mark.parametrize('review_agent', [False, True])
async def test_model_menu_and_review_are_independent(
    tmp_path: Path, sub_agents: bool, menu: bool, review_agent: bool
) -> None:
    settings: dict[str, JsonValue] = {
        'repo_context': False,
        'skill_folders': [],
        'sub_agents': sub_agents,
        'review_agent': review_agent,
        'delegate_models': {'custom': {'model': 'fixture:model'}} if menu else {},
    }
    host = PluginHost[object](name='coder', console=Console(file=io.StringIO()), settings=settings)

    def inspect(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        delegation = [tool for tool in info.function_tools if tool.name == 'delegate_task']
        assert bool(delegation) is sub_agents
        if delegation:
            properties = delegation[0].parameters_json_schema['properties']
            assert ('model' in properties) is menu
            assert ('working_directory' in properties) is review_agent
            assert info.instructions is not None
            assert ('- Review:' in info.instructions) is review_agent
        return ModelResponse(parts=[TextPart('checked')])

    agent = Agent(
        FunctionModel(inspect), deps_type=object, capabilities=CoderPlugin[object].from_host(host).get_capabilities()
    )
    await agent.run('inspect', workspace=Workspace(LocalWorkspaceBackend(tmp_path)))


def test_delegate_menu_settings_validate_save_and_reset() -> None:
    host = PluginHost[object](name='coder', console=Console(file=io.StringIO()), settings={})
    source = CoderSource(host)
    row = next(row for row in source.rows() if row.key == 'delegate_models')
    assert source.current(row) == '{}'
    assert source.problem(row, '{"review-light":{"model":"chain:review-light","settings":{"thinking":"high"}}}') is None
    source.apply(row, '{"review-light":{"model":"chain:review-light","settings":{"thinking":"high"}}}')
    assert host.settings(CoderSettings).delegate_models['review-light'].model == 'chain:review-light'
    review = next(row for row in source.rows() if row.key == 'review_agent')
    assert source.current(review) == 'false'
    source.apply(review, 'true')
    source.reset(row)
    assert host.settings(CoderSettings).delegate_models == {}
    assert host.settings(CoderSettings).review_agent is True
    source.reset(review)
    assert host.settings(CoderSettings).review_agent is False
    for value in (
        '{"":{"model":"fixture:light"}}',
        '{"light":{"model":"unqualified"}}',
        '{"light":{"model":"fixture:"}}',
        '{"light":{"model":"fixture:light","settings":{"thinking":"invalid"}}}',
    ):
        assert source.problem(row, value) is not None
    with pytest.raises(ValidationError):
        CoderSettings.model_validate({'delegate_models': {'light': {'model': 'fixture:light', 'unexpected': True}}})


async def test_plugin_host_model_defaults_without_session() -> None:
    host = PluginHost[object](name='standalone', console=Console(file=io.StringIO()), settings={})
    assert await host.resolve_model('test') == 'test'
    assert host.model_settings('test') is None


def test_task_inspector_shows_the_selected_directory() -> None:
    record = DelegationTask(
        id='review', agent_name='Review', prompt='inspect', conversation_id='parent', working_directory='/policy'
    )
    menu = TaskMenu(snapshot=lambda: [record], action=lambda action, task: None)
    assert 'Directory: /policy' in menu.preview(menu.items()[0])


async def test_two_sessions_keep_their_delegate_models(tmp_path: Path) -> None:
    import anyio

    def parent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task',
                        {
                            'agent_name': 'Review',
                            'task': 'inspect',
                            'model': 'review-light',
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    async def run(label: str) -> None:
        selected = TestModel(call_tools=[], custom_output_text=label)

        async def resolve(name: str) -> TestModel:
            assert name == 'fixture:light'
            return selected

        settings: dict[str, JsonValue] = {
            'repo_context': False,
            'skill_folders': [],
            'review_agent': True,
            'delegate_models': {'review-light': {'model': 'fixture:light'}},
        }
        host = PluginHost[object](
            name='coder', console=Console(file=io.StringIO()), settings=settings, resolve_model=resolve
        )
        agent = Agent(
            FunctionModel(parent), deps_type=object, capabilities=CoderPlugin[object].from_host(host).get_capabilities()
        )
        await agent.run('go', conversation_id=label, workspace=Workspace(LocalWorkspaceBackend(tmp_path)))

    async with DelegationTasks().opened() as owner:
        with owner.bind():
            async with anyio.create_task_group() as group:
                group.start_soon(run, 'first')
                group.start_soon(run, 'second')
        assert {record.conversation_id: record.output for record in owner.records.values()} == {
            'first': 'first',
            'second': 'second',
        }
