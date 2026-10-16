"""Tests for disk-loaded sub-agents, effort floor, and model inheritance."""

from __future__ import annotations

import warnings
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
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import HarnessDeprecationWarning, subagents
from pydantic_ai_harness.subagents import AgentOverride, SubAgent, SubAgents
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
    def test_deprecated_exports_still_apply_the_old_floor(self) -> None:
        with pytest.warns(HarnessDeprecationWarning, match='no longer imposes a minimum thinking effort') as record:
            floor = subagents.MINIMUM_EFFORT_FLOOR
            clamp = subagents.clamp_effort
        assert len(record) == 2
        assert all('AgentOverride(effort=' in str(warning.message) for warning in record)
        assert clamp(None) == floor
        assert clamp(False) == floor
        assert clamp(True) is True
        assert clamp('minimal') == 'low'
        assert clamp('low') == 'low'
        assert clamp('high') == 'high'
        assert clamp('low', floor='high') == 'high'
        assert clamp('xhigh', floor='high') == 'xhigh'
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            subagents.__getattr__('missing')


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

    async def test_default_does_not_load_from_the_workspace(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'planner.md', 'Plan.')
        cap: SubAgents[object] = SubAgents()
        assert cap.agent_folders is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert await _listing(cap, LocalWorkspaceBackend(tmp_path)) is None

    async def test_loads_the_conventional_folder_when_requested(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'planner.md', 'Plan.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- planner' in listing

    async def test_none_disables_loading(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'planner.md', 'Plan.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
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
        cap: SubAgents[object] = SubAgents(agent_folders='agents', workspace=LocalWorkspaceBackend(tmp_path / 'app'))
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

    @pytest.mark.parametrize('suffix', ['.md', '.toml'])
    async def test_undecodable_file_is_skipped_with_warning(self, tmp_path: Path, suffix: str) -> None:
        # A non-UTF-8 definition must not abort loading: every valid definition in the folder still loads.
        (tmp_path / f'broken{suffix}').write_bytes(b'---\nname: broken\n---\n\xff\xfe not utf-8')
        _write_agent(tmp_path, 'valid.md', '---\nname: valid\n---\nWork.')
        with pytest.warns(UserWarning, match='Skipping unreadable disk sub-agent file'):
            listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- valid' in listing and 'broken' not in listing


class TestCodexDiskLoading:
    async def test_standalone_toml_instructions_and_tools(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path / '.codex' / 'workers',
            'different-stem.toml',
            'name = " researcher "\ndescription = " Researches topics "\n'
            'developer_instructions = """\nResearch carefully.\nReport sources.\n"""\n'
            'allowed-tools = [" search ", "Read"]\n',
        )
        toolset: FunctionToolset[object] = FunctionToolset()
        resolved: list[str] = []

        def resolver(name: str) -> Sequence[AgentToolset[object]]:
            resolved.append(name)
            return [toolset]

        cap: SubAgents[object] = SubAgents(agent_folders='workers', tool_resolver=resolver)
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen, delegate_to='researcher'), capabilities=[cap])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- researcher: Researches topics' in seen[0][0]
        assert seen[1][0] == 'Research carefully.\nReport sources.'
        assert resolved == ['search', 'Read']
        assert toolset in _built(cap)['researcher'].toolsets
        assert _built(cap)['researcher'].model is None

    @pytest.mark.parametrize(
        'tools', ['', 'tools = []', 'tools = ["Read"]', 'tools = " Read , Grep "', 'allowed-tools = "Read"']
    )
    async def test_optional_tools(self, tmp_path: Path, tools: str) -> None:
        _write_agent(
            tmp_path,
            'worker.toml',
            'name = "worker"\ndescription = "Works"\ndeveloper_instructions = "Work."\n' + tools,
        )
        resolved: list[str] = []

        def resolver(name: str) -> Sequence[AgentToolset[object]]:
            resolved.append(name)
            return []

        listing = await _listing(
            SubAgents(agent_folders=['.'], tool_resolver=resolver), LocalWorkspaceBackend(tmp_path)
        )
        assert listing is not None and '- worker: Works' in listing
        assert resolved == ([] if tools in ('', 'tools = []') else ['Read', 'Grep'] if 'Grep' in tools else ['Read'])

    @pytest.mark.parametrize('field', ['name', 'description', 'developer_instructions'])
    @pytest.mark.parametrize('value', [None, '""', '"  "', '42', 'true', '[]', '{}'])
    async def test_invalid_required_fields_skip_only_one_file(
        self, tmp_path: Path, field: str, value: str | None
    ) -> None:
        fields = {'name': '"broken"', 'description': '"Works"', 'developer_instructions': '"Work."'}
        if value is None:
            del fields[field]
        else:
            fields[field] = value
        text = '\n'.join(f'{key} = {item}' for key, item in fields.items())
        _write_agent(tmp_path, 'broken.toml', text)
        _write_agent(tmp_path, 'valid.md', 'Valid.')
        with pytest.warns(UserWarning, match=f'`{field}` must be a nonempty string') as record:
            listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1 and 'broken.toml' in str(record[0].message)
        assert listing is not None and '- valid' in listing and 'broken' not in listing

    @pytest.mark.parametrize(
        'tools',
        [
            'tools = 42',
            'allowed-tools = true',
            'tools = {}',
            'tools = ["Read", 1]',
            'tools = [" "]',
            'tools = ""',
            'tools = "Read,,Grep"',
            'tools = ["Read"]\nallowed-tools = ["Grep"]',
        ],
    )
    async def test_invalid_tools_skip_file(self, tmp_path: Path, tools: str) -> None:
        _write_agent(
            tmp_path,
            'broken.toml',
            'name = "broken"\ndescription = "Works"\ndeveloper_instructions = "Work."\n' + tools,
        )
        with pytest.warns(UserWarning, match='Skipping invalid disk sub-agent file'):
            assert await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path)) is None

    @pytest.mark.parametrize(
        'text', ['name =', 'name = "first"\nname = "second"', '[agents.worker]\nconfig_file = "worker.toml"']
    )
    async def test_malformed_and_legacy_toml_skip_only_one_file(self, tmp_path: Path, text: str) -> None:
        _write_agent(tmp_path, 'broken.toml', text)
        _write_agent(tmp_path, 'valid.toml', 'name = "valid"\ndescription = "Works"\ndeveloper_instructions = "Work."')
        with pytest.warns(UserWarning, match='Skipping invalid disk sub-agent file') as record:
            listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1 and 'broken.toml' in str(record[0].message)
        assert listing is not None and '- valid' in listing and 'broken' not in listing

    @pytest.mark.parametrize(
        'setting',
        [
            'sandbox_mode = "read-only"',
            'approval_policy = "never"',
            'permissions = "restricted"',
            'unknown_future_security_setting = true',
            '[sandbox_workspace_write]\nnetwork_access = false',
            '[mcp_servers.example]\ncommand = "never-execute-this"',
        ],
    )
    async def test_unsupported_settings_fail_closed(self, tmp_path: Path, setting: str) -> None:
        _write_agent(
            tmp_path,
            'broken.toml',
            'name = "broken"\ndescription = "Works"\ndeveloper_instructions = "Work."\n' + setting,
        )
        with pytest.warns(UserWarning, match='unsupported TOML settings'):
            assert await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path)) is None

    async def test_nonsecurity_settings_warn_and_do_not_change_model_or_effort(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path,
            'worker.toml',
            'name = "worker"\ndescription = "Works"\ndeveloper_instructions = "Work."\n'
            'model = "not-a-model"\neffort = "high"\nmodel_reasoning_effort = "high"\ncolor = "blue"',
        )
        cap: SubAgents[object] = SubAgents(agent_folders=['.'])
        with pytest.warns(UserWarning, match='Ignoring TOML disk sub-agent settings') as record:
            listing = await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1 and 'agent_overrides' in str(record[0].message)
        assert listing is not None and '- worker' in listing
        assert _built(cap)['worker'].model is None
        assert _built(cap)['worker'].model_settings is None

    @pytest.mark.parametrize('toml_first', [False, True])
    async def test_mixed_formats_are_sorted_and_first_definition_wins(self, tmp_path: Path, toml_first: bool) -> None:
        _write_agent(
            tmp_path,
            'a.toml' if toml_first else 'b.toml',
            'name = "worker"\ndescription = "toml"\ndeveloper_instructions = "Work."',
        )
        _write_agent(tmp_path, 'b.md' if toml_first else 'a.md', '---\nname: worker\ndescription: markdown\n---\nWork.')
        (tmp_path / 'directory.toml').mkdir()
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed") as record:
            listing = await _listing(SubAgents(agent_folders=['.']), LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1
        assert listing is not None
        assert f'- worker: {"toml" if toml_first else "markdown"}' in listing
        assert f': {"markdown" if toml_first else "toml"}' not in listing

    async def test_conventional_precedence(self, tmp_path: Path) -> None:
        for root in ('.agents', '.claude', '.codex'):
            _write_agent(
                tmp_path / root / 'agents',
                'worker.toml',
                f'name = "worker"\ndescription = "{root}"\ndeveloper_instructions = "Work."',
            )
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed") as record:
            listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path))
        assert len(record) == 2
        assert listing is not None and '- worker: .agents' in listing
        assert '.claude' not in listing and '.codex' not in listing
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            listing = await _listing(
                SubAgents(agent_folders=['.claude/agents', '.codex/agents']), LocalWorkspaceBackend(tmp_path)
            )
        assert listing is not None and '- worker: .claude' in listing and '.codex' not in listing

    async def test_codex_symlink_is_deduplicated(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path / '.agents' / 'agents',
            'worker.toml',
            'name = "worker"\ndescription = "Works"\ndeveloper_instructions = "Work."',
        )
        (tmp_path / '.codex').symlink_to(tmp_path / '.agents', target_is_directory=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- worker' in listing

    async def test_codex_discovery_stays_off_by_default(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path / '.codex' / 'agents',
            'worker.toml',
            'name = "worker"\ndescription = "Works"\ndeveloper_instructions = "Work."',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert await _listing(SubAgents(), LocalWorkspaceBackend(tmp_path)) is None
            assert await _listing(SubAgents(agent_folders=None), LocalWorkspaceBackend(tmp_path)) is None


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
        assert _built(cap)['w'].model_settings == {'thinking': 'high'}

    async def test_no_effort_override_leaves_thinking_unset(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders=['.'])
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert _built(cap)['w'].model_settings is None

    @pytest.mark.parametrize('effort', ['minimal', False])
    async def test_effort_override_is_not_clamped(self, tmp_path: Path, effort: ThinkingLevel) -> None:
        _write_agent(tmp_path, 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents(
            agent_folders=['.'],
            agent_overrides={'w': AgentOverride(effort=effort)},
        )
        await _listing(cap, LocalWorkspaceBackend(tmp_path))
        assert _built(cap)['w'].model_settings == {'thinking': effort}


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


class TestInheritToolsDeprecation:
    def test_true_warns(self) -> None:
        worker = Agent(TestModel(), name='worker')
        with pytest.warns(
            HarnessDeprecationWarning,
            match=r'inherit_tools=True\)` is deprecated.*Bind required tools directly.*include_self=True',
        ):
            cap: SubAgents[object] = SubAgents(agents=[SubAgent(worker)], inherit_tools=True)
        assert cap.inherit_tools is True

    def test_false_is_silent(self) -> None:
        worker = Agent(TestModel(), name='worker')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SubAgents(agents=[SubAgent(worker)], inherit_tools=False)


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
        cap: SubAgents[object] = SubAgents(agent_folders='agents')
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

    async def test_claude_folder_and_stem_name(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan things.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents(agent_folders='agents')])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- planner' in seen[0][0]

    async def test_claude_folder_loads_when_agents_root_exists(self, tmp_path: Path) -> None:
        # A workspace that uses `.agents/` for something else still loads agents from `.claude/`.
        (tmp_path / '.agents').mkdir()
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan.')
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen), capabilities=[SubAgents(agent_folders='agents')])
        await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- planner' in seen[0][0]

    async def test_agents_folder_shadows_claude_folder(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\ndescription: agents\n---\nWork.')
        _write_agent(tmp_path / '.claude' / 'agents', 'worker.md', '---\nname: worker\ndescription: claude\n---\nWork.')
        _write_agent(tmp_path / '.claude' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed") as record:
            listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path))
        assert len(record) == 1
        assert listing is not None
        assert '- worker: agents' in listing and 'claude' not in listing
        assert '- planner' in listing

    async def test_claude_symlinked_to_agents_loads_once_without_warning(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'planner.md', 'Plan.')
        (tmp_path / '.claude').symlink_to(tmp_path / '.agents', target_is_directory=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path))
        assert listing is not None and '- planner' in listing

    async def test_explicit_folder_resolves_parent_segments_before_symlinks(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / 'agents', 'expected.md', 'Expected.')
        (tmp_path / 'target' / 'child').mkdir(parents=True)
        _write_agent(tmp_path / 'target' / 'agents', 'wrong.md', 'Wrong.')
        (tmp_path / 'link').symlink_to(tmp_path / 'target' / 'child', target_is_directory=True)

        listing = await _listing(SubAgents(agent_folders=['link/../agents']), LocalWorkspaceBackend(tmp_path))

        assert listing is not None and '- expected' in listing and 'wrong' not in listing

    async def test_no_workspace_skips_convention_discovery(self) -> None:
        # Neither the home root nor the host's cwd stands in for a missing workspace, but a folder
        # earlier releases read is reported once per instance, not dropped without a word.
        _write_agent(Path.home() / '.agents' / 'agents', 'reviewer.md', 'Review.')
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', 'Plan.')
        cap: SubAgents[object] = SubAgents(agent_folders='agents')
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions') as record:
            assert await _listing(cap, None) is None
            assert await _listing(cap, None) is None
        assert len(record) == 1
        assert f'`SubAgents(workspace=LocalWorkspaceBackend({str(Path.cwd())!r}))`' in str(record[0].message)

    async def test_no_workspace_warns_about_the_home_folder(self) -> None:
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions') as record:
            assert await _listing(SubAgents(agent_folders='agents'), None) is None
        assert f'`SubAgents(workspace=LocalWorkspaceBackend({str(Path.home())!r}))`' in str(record[0].message)

        # The fix the warning names reads the home folder with convention discovery enabled.
        listing = await _listing(SubAgents(agent_folders='agents', workspace=LocalWorkspaceBackend(Path.home())), None)
        assert listing is not None and '- planner' in listing

    async def test_workspace_run_warns_about_the_home_folder(self, tmp_path: Path) -> None:
        # The run's workspace holds the project, not this machine's home folder.
        _write_agent(Path.home() / '.agents' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'did not load the agent definitions'):
            assert await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(tmp_path)) is None

        # A workspace at the home directory reads the folder, so there is nothing to report.
        listing = await _listing(SubAgents(agent_folders='agents'), LocalWorkspaceBackend(Path.home()))
        assert listing is not None and '- planner' in listing

    async def test_no_workspace_warns_about_the_claude_folder(self) -> None:
        _write_agent(Path.cwd() / '.claude' / 'agents', 'planner.md', 'Plan.')
        with pytest.warns(HarnessDeprecationWarning, match=r'\.claude') as record:
            assert await _listing(SubAgents(agent_folders='agents'), None) is None
        assert f'LocalWorkspaceBackend({str(Path.cwd())!r})' in str(record[0].message)

    async def test_no_workspace_and_no_host_folder_is_silent(self) -> None:
        # Nothing was there to lose. `filterwarnings = error` turns any warning into a failure.
        assert await _listing(SubAgents(agent_folders='agents'), None) is None

    async def test_no_workspace_with_discovery_off_is_silent(self) -> None:
        _write_agent(Path.cwd() / '.agents' / 'agents', 'planner.md', 'Plan.')
        assert await _listing(SubAgents(agent_folders=None), None) is None

    async def test_runs_over_unchanged_files_are_identical(self, tmp_path: Path) -> None:
        _write_agent(tmp_path / '.agents' / 'agents', 'w.md', '---\nname: w\n---\nB')
        cap: SubAgents[object] = SubAgents(agent_folders='agents')
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
            _recording_model(seen),
            capabilities=[SubAgents(agents=[SubAgent(explicit)], agent_folders='agents')],
        )
        with pytest.warns(UserWarning, match="Disk sub-agent 'worker' is shadowed"):
            await parent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert seen[0][0] is not None and '- worker: code' in seen[0][0]

    async def test_merged_per_run_copies_keep_both_rosters(self, tmp_path: Path) -> None:
        # Run-level capabilities under the shared id are combined after `for_run`, on the per-run copies.
        _write_agent(tmp_path / '.agents' / 'agents', 'worker.md', '---\nname: worker\n---\nB')
        first: SubAgents[object] = SubAgents(
            agents=[SubAgent(Agent(TestModel(), name='alpha'))], agent_folders='agents'
        )
        second: SubAgents[object] = SubAgents(
            agents=[SubAgent(Agent(TestModel(), name='beta'))], agent_folders='agents'
        )
        seen: list[tuple[str | None, list[str]]] = []
        parent: Agent[object, str] = Agent(_recording_model(seen))
        await parent.run('go', capabilities=[first, second], workspace=LocalWorkspaceBackend(tmp_path))
        instructions = seen[0][0]
        assert instructions is not None
        assert all(f'- {name}' in instructions for name in ('alpha', 'beta', 'worker'))
