"""Tests for disk-loaded sub-agents, effort floor, and model inheritance."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import AbstractModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.subagents import (
    MINIMUM_EFFORT_FLOOR,
    AgentOverride,
    SubAgent,
    SubAgents,
    clamp_effort,
)
from pydantic_ai_harness.subagents._disk import ParsedAgent, parse_agent_markdown


def _delegate_then_finish(agent_name: str) -> FunctionModel:
    """A parent model that delegates to `agent_name` once, then replies with text."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': agent_name, 'task': 'do it'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


def _delegate_returns(result: Any) -> list[str]:
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
    ]


def _write_agent(folder: Path, filename: str, content: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding='utf-8')


class TestClampEffort:
    def test_none_becomes_floor(self) -> None:
        assert clamp_effort(None) == MINIMUM_EFFORT_FLOOR

    def test_false_becomes_floor(self) -> None:
        assert clamp_effort(False) == MINIMUM_EFFORT_FLOOR

    def test_true_unchanged(self) -> None:
        assert clamp_effort(True) is True

    def test_below_floor_raised(self) -> None:
        assert clamp_effort('minimal') == 'low'

    def test_at_floor_unchanged(self) -> None:
        assert clamp_effort('low') == 'low'

    def test_above_floor_unchanged(self) -> None:
        assert clamp_effort('high') == 'high'

    def test_custom_floor(self) -> None:
        assert clamp_effort('low', floor='high') == 'high'
        assert clamp_effort('xhigh', floor='high') == 'xhigh'


class TestParseAgentMarkdown:
    def test_full_frontmatter(self) -> None:
        text = '---\nname: researcher\ndescription: Researches topics\ntools: Read, Grep\ncolor: blue\n---\nBody here.'
        parsed = parse_agent_markdown(text)
        assert parsed == ParsedAgent(
            name='researcher', description='Researches topics', tools=('Read', 'Grep'), body='Body here.'
        )

    def test_no_frontmatter_uses_whole_text_as_body(self) -> None:
        parsed = parse_agent_markdown('Just a body, no frontmatter.\n')
        assert parsed == ParsedAgent(None, None, (), 'Just a body, no frontmatter.')

    def test_empty_text(self) -> None:
        assert parse_agent_markdown('') == ParsedAgent(None, None, (), '')

    def test_unclosed_frontmatter_is_all_body(self) -> None:
        parsed = parse_agent_markdown('---\nname: x\nstill no close')
        assert parsed == ParsedAgent(None, None, (), '---\nname: x\nstill no close')

    def test_block_list_tools(self) -> None:
        # Includes a blank line (skipped) and a `- ` item with an empty value (dropped).
        text = '---\nname: a\n\ntools:\n  - Read\n  - Edit\n  - \n---\nBody'
        parsed = parse_agent_markdown(text)
        assert parsed.tools == ('Read', 'Edit')

    def test_allowed_tools_key(self) -> None:
        parsed = parse_agent_markdown('---\nname: a\nallowed-tools: Bash(git:*), Read\n---\nB')
        assert parsed.tools == ('Bash(git:*)', 'Read')

    def test_quoted_scalar_values(self) -> None:
        parsed = parse_agent_markdown('---\nname: "quoted"\ndescription: \'single\'\n---\nB')
        assert parsed.name == 'quoted'
        assert parsed.description == 'single'

    def test_non_scalar_name_falls_back_to_none(self) -> None:
        # `name:` with an empty value parses as a (empty) list, which is not a str.
        parsed = parse_agent_markdown('---\nname:\ndescription: d\n---\nB')
        assert parsed.name is None
        assert parsed.description == 'd'

    def test_stray_dash_line_without_list_key_is_ignored(self) -> None:
        # A `- item` line with no preceding list key has no colon, so it resets and is skipped.
        parsed = parse_agent_markdown('---\n- orphan\nname: a\n---\nB')
        assert parsed.name == 'a'
        assert parsed.tools == ()

    def test_no_tools_key(self) -> None:
        assert parse_agent_markdown('---\nname: a\n---\nB').tools == ()


async def _listing(cap: SubAgents[object], workspace: LocalWorkspaceBackend | None) -> str | None:
    """The sub-agent listing a run sees on its first step."""
    seen: list[tuple[str | None, list[str]]] = []
    parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[cap])
    await parent.run('go', workspace=workspace)
    return seen[0][0]


def _built(cap: SubAgents[object]) -> dict[str, Agent[object, Any]]:
    """The disk delegates built so far, by name."""
    return {definition.name: sub_agent.agent for definition, sub_agent in cap._built.items()}  # pyright: ignore[reportReturnType,reportPrivateUsage]


class TestDiskLoading:
    def test_nothing_is_read_at_construction(self) -> None:
        # The home root is not a source, and folders are read per run, not when the capability is built.
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', '---\nname: planner\n---\nPlan.')
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', '---\nname: planner\n---\nPlan.')
        cap: SubAgents[object] = SubAgents()
        assert cap._by_name == {}  # pyright: ignore[reportPrivateUsage]
        assert cap.get_toolset() is None

    async def test_none_disables_loading(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'planner.md', 'Plan.')
        assert await _listing(SubAgents(agent_folders=None), LocalWorkspaceBackend(tmp_path)) is None

    async def test_loads_folders_from_the_workspace_in_order(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / 'project', 'worker.md', '---\nname: worker\ndescription: project\n---\nB')
        _write_agent(tmp_path / 'shared', 'worker.md', '---\nname: worker\ndescription: shared\n---\nB')
        _write_agent(tmp_path / 'shared', 'planner.md', 'No frontmatter, just a body.')
        # A relative path and a `Path` naming the same folder are read once; a missing folder is skipped.
        cap: SubAgents[object] = SubAgents(
            agent_folders=['project', 'shared', Path('shared'), str(tmp_path / 'does-not-exist')]
        )
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed") as record:
            listing = await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1
        assert listing is not None
        assert '- worker: project' in listing and 'shared' not in listing
        assert '- planner' in listing, 'the name falls back to the file stem'
        assert _built(cap)['worker'].model is None, 'a disk agent inherits the parent model at delegation'

    async def test_own_workspace_replaces_the_runs(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / 'app' / '.agents' / 'agents', 'packaged.md', 'Ship it.')
        _write_agent(tmp_path / 'run' / '.agents' / 'agents', 'project.md', 'Project work.')
        cap: SubAgents[object] = SubAgents(workspace=LocalWorkspaceBackend(tmp_path / 'app'))
        for workspace in (LocalWorkspaceBackend(tmp_path / 'run'), None):
            listing = await _listing(cap, workspace)
            assert listing is not None and '- packaged' in listing and 'project' not in listing

    def test_workspace_capability_is_refused(self) -> None:
        with pytest.raises(TypeError, match='takes a workspace backend'):
            SubAgents(workspace=LocalWorkspace('.'))  # pyright: ignore[reportArgumentType]

    async def test_explicit_folders_without_a_workspace_fail_the_run(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'worker.md', 'Work.')
        cap: SubAgents[object] = SubAgents(agent_folders=[tmp_path])
        with pytest.raises(UserError, match='`SubAgents` needs a workspace'):
            await _listing(cap, None)

    async def test_undecodable_file_is_skipped_with_warning(self, tmp_path: Path) -> None:
        # A non-UTF-8 `.md` file must not abort loading: every valid definition in the folder still loads.
        (tmp_path / 'broken.md').write_bytes(b'---\nname: broken\n---\n\xff\xfe not utf-8')
        _write_agent(tmp_path, 'valid.md', '---\nname: valid\n---\nWork.')
        with pytest.warns(UserWarning, match='Skipping unreadable disk sub-agent file'):
            listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- valid' in listing and 'broken' not in listing


class TestOverrides:
    async def test_model_and_effort_override(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        model = TestModel()
        cap: SubAgents[object] = SubAgents(
            agent_folders=['.'],
            agent_overrides={'w': AgentOverride(model=model, effort='high')},
        )
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert _built(cap)['w'].model is model

    async def test_effort_floored_without_override(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders=['.'])
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert _built(cap)['w'].model_settings == {'thinking': MINIMUM_EFFORT_FLOOR}


class TestToolResolver:
    async def test_resolver_attaches_tools(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: search\n---\nB')

        toolset: FunctionToolset[object] = FunctionToolset()

        def resolver(name: str) -> Sequence[AgentToolset[object]] | None:
            return [toolset] if name == 'search' else None

        cap: SubAgents[object] = SubAgents(agent_folders=['.'], tool_resolver=resolver)
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert toolset in _built(cap)['w'].toolsets

    async def test_unknown_tool_warns_and_skips(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: mystery\n---\nB')

        def resolver(name: str) -> Sequence[AgentToolset[object]] | None:
            return None

        with pytest.warns(UserWarning, match="Unknown tool 'mystery'"):
            await _listing(SubAgents(agent_folders=['.'], tool_resolver=resolver), LocalWorkspaceBackend(tmp_path))

    async def test_no_resolver_ignores_frontmatter_tools(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\ntools: Read, Edit\n---\nB')
        # Without a resolver, no warning and no own tools -- inheritance is the path.
        listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- w' in listing


class TestModelInheritance:
    async def test_disk_agent_inherits_parent_model(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'worker.md', '---\nname: worker\n---\nDo the work.')
        cap: SubAgents[object] = SubAgents(agent_folders=['.'])
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        disk_agent = _built(cap)['worker']

        # `RunContext.model` is an `AbstractModel`; the inherited model captured here is the
        # parent's request-response model, asserted by identity below.
        captured: dict[str, AbstractModel] = {}

        @disk_agent.instructions
        def _capture(ctx: RunContext[object]) -> str:
            captured['model'] = ctx.model
            return ''

        parent_model = _delegate_then_finish('worker')
        parent: Agent[object, str] = Agent(parent_model, capabilities=[cap])
        result = await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert result.output == 'all done'
        # The model-less disk agent ran on the parent's resolved model.
        assert captured['model'] is parent_model
        assert _delegate_returns(result) == ['all done']


def _recording_model(seen: list[tuple[str | None, list[str]]], delegate_to: str | None = None) -> FunctionModel:
    """A parent model that records each step's instructions and tool names, delegating once if asked."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.append((request.instructions, [tool.name for tool in info.function_tools]))
        if delegate_to is not None and len(seen) == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': delegate_to, 'task': 'do it'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


class TestWorkspaceDiscovery:
    """The project folder is read through `ctx.workspace` at the start of each run."""

    async def test_project_agents_come_from_the_workspace(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\ndescription: Works\n---\nWork.')
        # Neither a non-markdown file nor a directory is a definition.
        _write_agent(tmp_path / '.agents' / 'agents', 'notes.txt', 'not an agent')
        (tmp_path / '.agents' / 'agents' / 'nested.md').mkdir()
        cap: SubAgents[object] = SubAgents()
        assert cap._by_name == {}, 'nothing is read from the workspace before a run'  # pyright: ignore[reportPrivateUsage]

        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen, delegate_to='worker'), capabilities=[cap])
        result = await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert _delegate_returns(result) == ['all done']
        instructions, tools = seen[0]
        assert instructions is not None and '- worker: Works' in instructions
        assert tools == ['delegate_task']
        # The disk child inherits this model, so it records the middle step; the parent's
        # later step sees the same listing and tool as its first.
        assert seen[2] == seen[0]

    async def test_claude_fallback_and_stem_name(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan things.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- planner' in seen[0][0]

    async def test_agents_root_without_the_leaf_folder(self, tmp_path: Path) -> None:
        # `.agents/` exists, so `.claude/` is not consulted even though only it has the leaf.
        (tmp_path / '.agents').mkdir()
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents()])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0] == (None, [])

    async def test_no_workspace_skips_convention_discovery(self) -> None:
        # Neither the home root nor the host's cwd stands in for a missing workspace, but a folder
        # earlier releases read is reported once per instance, not dropped without a word.
        _write_agent(Path.home() / '.agents' / 'agents', 'reviewer.md', 'Review.')
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', 'Plan.')
        cap: SubAgents[object] = SubAgents()
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions') as record:
            assert await _listing(cap, None) is None
            assert await _listing(cap, None) is None
        assert len(record) == 1
        assert f'`SubAgents(workspace=LocalWorkspaceBackend({str(Path.cwd())!r}))`' in str(record[0].message)

    async def test_no_workspace_warns_about_the_home_folder(self) -> None:
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions') as record:
            assert await _listing(SubAgents(), None) is None
        assert f'`SubAgents(workspace=LocalWorkspaceBackend({str(Path.home())!r}))`' in str(record[0].message)

        # The fix the warning names reads the home folder with the default `agent_folders`.
        listing = await _listing(SubAgents(workspace=LocalWorkspaceBackend(Path.home())), None)
        assert listing is not None and '- planner' in listing

    async def test_workspace_run_warns_about_the_home_folder(self, tmp_path: Path) -> None:
        # The run's workspace holds the project, not this machine's home folder.
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions'):
            assert await _listing(SubAgents(), LocalWorkspaceBackend(tmp_path)) is None

        # A workspace at the home directory reads the folder, so there is nothing to report.
        listing = await _listing(SubAgents(), LocalWorkspaceBackend(Path.home()))
        assert listing is not None and '- planner' in listing

    async def test_no_workspace_warns_about_the_claude_fallback(self) -> None:
        _write_agent(Path.cwd() / '.claude' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'\.claude') as record:
            assert await _listing(SubAgents(), None) is None
        assert f'LocalWorkspaceBackend({str(Path.cwd())!r})' in str(record[0].message)

    async def test_no_workspace_and_no_host_folder_is_silent(self) -> None:
        # Nothing was there to lose. `filterwarnings = error` turns any warning into a failure.
        assert await _listing(SubAgents(), None) is None

    async def test_no_workspace_with_discovery_off_is_silent(self) -> None:
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', 'Plan.')
        assert await _listing(SubAgents(agent_folders=None), None) is None

    async def test_runs_over_unchanged_files_are_identical(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents()
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[cap])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0] == seen[1]
        # The delegate is built once and reused, so its agent does not change between runs.
        assert len(cap._built) == 1  # pyright: ignore[reportPrivateUsage]
        assert cap._by_name == {}, 'per-run discovery does not leak into the shared instance'  # pyright: ignore[reportPrivateUsage]

    async def test_explicit_shadows_project(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\ndescription: disk\n---\nB')
        explicit = Agent(TestModel(), name='worker', description='code')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(
            _recording_model(seen), capabilities=[SubAgents(agents=[SubAgent(explicit)])]
        )
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- worker: code' in seen[0][0]

    async def test_merged_per_run_copies_keep_both_rosters(self, tmp_path: Path) -> None:
        # Run-level capabilities under the shared id are combined after `for_run`, on the per-run copies.
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\n---\nB')
        first: SubAgents[object] = SubAgents(agents=[SubAgent(Agent(TestModel(), name='alpha'))])
        second: SubAgents[object] = SubAgents(agents=[SubAgent(Agent(TestModel(), name='beta'))])
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen))
        await parent.run('go', capabilities=[first, second], workspace=LocalWorkspaceBackend(tmp_path))
        instructions = seen[0][0]
        assert instructions is not None
        assert all(f'- {name}' in instructions for name in ('alpha', 'beta', 'worker'))
