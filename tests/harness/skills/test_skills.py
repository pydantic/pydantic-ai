from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio.to_thread
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.skills import Skills


def _write_skill(
    library: Path,
    name: str,
    *,
    description: str = 'Help with the task.',
    body: str = 'Follow these directions.',
    frontmatter: str | None = None,
    files: Mapping[str, str] | None = None,
) -> None:
    directory = library / name
    directory.mkdir(parents=True)
    metadata = frontmatter if frontmatter is not None else f'description: {description}'
    (directory / 'SKILL.md').write_text(f'---\n{metadata}\n---\n\n{body}\n', encoding='utf-8')
    for relative, content in (files or {}).items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')


@dataclass
class _Run:
    """What the model saw on its first step, and the instructions `load_capability` returned."""

    instructions: str | None = None
    tools: list[str] = field(default_factory=list[str])
    loaded: str | None = None


def _model(run: _Run, load: str | None) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if len(messages) == 1:
            run.instructions = request.instructions
            run.tools = [tool.name for tool in info.function_tools]
            if load is not None:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': load}, tool_call_id='load')])
        for part in request.parts:
            if isinstance(part, LoadCapabilityReturnPart):
                run.loaded = part.content.get('instructions')
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


async def _run(skills: Skills[Any], workspace: Path, *, load: str | None = None) -> _Run:
    run = _Run()
    agent: Agent[None, str] = Agent(_model(run, load), capabilities=[skills])
    await agent.run('go', workspace=LocalWorkspaceBackend(workspace))
    return run


def _catalog(*entries: str) -> str:
    return (
        'The following capabilities are deferred and can be loaded using the `load_capability` tool. '
        "A capability's tools stay hidden until it is loaded:\n" + '\n'.join(entries)
    )


class TestSkills:
    def test_public_constructor_only_exposes_skill_library_configuration(self) -> None:
        assert tuple(inspect.signature(Skills).parameters) == ('directories', 'include', 'exclude', 'workspace')

    def test_repr_only_exposes_skill_library_configuration(self) -> None:
        assert repr(Skills('skills')) == "Skills(directories=('skills',), include=None, exclude=frozenset())"

    def test_construction_reads_nothing(self, tmp_path: Path) -> None:
        # Libraries live in the run's workspace, so a missing host path is not an error here.
        Skills(tmp_path / 'missing')

    async def test_skills_are_deferred_capabilities_listed_by_name_and_description(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'beta', description='Beta help.')
        _write_skill(tmp_path / 'skills', 'alpha', description='Alpha help.')

        run = await _run(Skills('skills'), tmp_path)

        assert run.instructions == _catalog('- alpha: Alpha help.', '- beta: Beta help.')
        assert run.tools == ['load_capability']

    async def test_relative_directories_resolve_in_the_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_skill(tmp_path / 'workspace' / 'skills', 'alpha')
        _write_skill(tmp_path / 'host' / 'skills', 'host-only')
        monkeypatch.chdir(tmp_path / 'host')

        run = await _run(Skills('skills'), tmp_path / 'workspace')

        assert run.instructions == _catalog('- alpha: Help with the task.')

    async def test_local_workspace_capability_supplies_the_libraries(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha', body='Alpha directions.')
        run = _Run()
        agent: Agent[None, str] = Agent(
            _model(run, 'alpha'), capabilities=[Skills('skills'), LocalWorkspace(str(tmp_path))]
        )

        await agent.run('go')

        assert run.loaded == '# Skill: alpha\n\nAlpha directions.'

    async def test_no_workspace_fails_the_run(self) -> None:
        agent: Agent[None, str] = Agent(_model(_Run(), None), capabilities=[Skills('skills')])

        with pytest.raises(UserError, match='`Skills` needs a workspace'):
            await agent.run('go')

    async def test_own_workspace_supplies_the_libraries(self, tmp_path: Path) -> None:
        # Skills shipped with the code, while the run has no workspace (or a sandbox) of its own.
        _write_skill(tmp_path / 'app' / 'skills', 'alpha', body='Alpha directions.')
        run = _Run()
        skills: Skills[Any] = Skills('skills', workspace=LocalWorkspaceBackend(tmp_path / 'app'))
        agent: Agent[None, str] = Agent(_model(run, 'alpha'), capabilities=[skills])

        await agent.run('go')

        assert run.loaded == '# Skill: alpha\n\nAlpha directions.'

    def test_own_workspace_must_be_a_backend(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match=r'takes a workspace backend.*LocalWorkspaceBackend\('):
            Skills('skills', workspace=LocalWorkspace(tmp_path))  # pyright: ignore[reportArgumentType]

    async def test_empty_selection_offers_no_tool(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha')

        run = await _run(Skills('skills', include=[]), tmp_path)

        assert (run.instructions, run.tools) == (None, [])

    async def test_runs_over_unchanged_files_are_identical_and_rescan(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'first')
        skills: Skills[Any] = Skills('skills')

        first = await _run(skills, tmp_path)
        again = await _run(skills, tmp_path)
        _write_skill(tmp_path / 'skills', 'later')
        later = await _run(skills, tmp_path)

        assert (first.instructions, first.tools) == (again.instructions, again.tools)
        assert later.instructions == _catalog('- first: Help with the task.', '- later: Help with the task.')

    async def test_two_skills_combine_into_one_catalog(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'first', 'alpha', description='Alpha help.')
        _write_skill(tmp_path / 'second', 'beta', description='Beta help.')
        _write_skill(tmp_path / 'second', 'gamma', description='Gamma help.')
        run = _Run()
        agent: Agent[None, str] = Agent(
            _model(run, 'beta'),
            # The same `SKILL.md` selected twice is listed once.
            capabilities=[Skills('first'), Skills('second', exclude=['gamma']), Skills('first', include=['alpha'])],
        )

        await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))

        assert run.instructions == _catalog('- alpha: Alpha help.', '- beta: Beta help.')
        assert run.tools == ['load_capability']
        assert run.loaded == '# Skill: beta\n\nFollow these directions.'

    async def test_run_level_skills_combine_with_the_agents(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'first', 'alpha')
        _write_skill(tmp_path / 'second', 'beta')
        run = _Run()
        agent: Agent[None, str] = Agent(_model(run, None))

        await agent.run(
            'go', capabilities=[Skills('first'), Skills('second')], workspace=LocalWorkspaceBackend(tmp_path)
        )

        assert run.instructions == _catalog('- alpha: Help with the task.', '- beta: Help with the task.')

    async def test_two_libraries_sharing_a_skill_name_collide(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'first', 'shared')
        _write_skill(tmp_path / 'second', 'shared')

        with pytest.raises(ValueError, match="Duplicate skill name 'shared'"):
            await _run(_combined(Skills('first'), Skills('second')), tmp_path)

    async def test_include_exposes_only_selected_skills(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha')
        _write_skill(tmp_path / 'skills', 'beta')

        run = await _run(Skills('skills', include=['beta']), tmp_path)

        assert run.instructions == _catalog('- beta: Help with the task.')

    async def test_exclude_hides_selected_skills(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha')
        _write_skill(tmp_path / 'skills', 'beta')

        run = await _run(Skills('skills', exclude=['alpha']), tmp_path)

        assert run.instructions == _catalog('- beta: Help with the task.')

    @pytest.mark.parametrize(
        ('body', 'loaded'),
        [
            ('Answer from this embedded guidance.', '# Skill: knowledge\n\nAnswer from this embedded guidance.'),
            ('', '# Skill: knowledge'),
            ('    print("hello")', '# Skill: knowledge\n\n    print("hello")'),
            ('Do the task.\n\n', '# Skill: knowledge\n\nDo the task.'),
            ('Read ${CLAUDE_SKILL_DIR}/guide.md.', '# Skill: knowledge\n\nRead ${CLAUDE_SKILL_DIR}/guide.md.'),
        ],
        ids=['body', 'empty', 'indentation', 'trailing-blank-lines', 'placeholder-unresolved'],
    )
    async def test_loaded_instructions_contain_only_heading_and_body(
        self, tmp_path: Path, body: str, loaded: str
    ) -> None:
        _write_skill(tmp_path / 'skills', 'knowledge', body=body)

        run = await _run(Skills('skills'), tmp_path, load='knowledge')

        assert run.loaded == loaded

    async def test_bundled_files_do_not_change_loaded_instructions(self, tmp_path: Path) -> None:
        _write_skill(
            tmp_path / 'skills',
            'portable',
            body='Follow the portable workflow.',
            files={'references/guide.md': 'Use the documented workflow.'},
        )

        run = await _run(Skills('skills'), tmp_path, load='portable')

        assert run.loaded == '# Skill: portable\n\nFollow the portable workflow.'

    @pytest.mark.parametrize(
        ('suffix', 'spec_text'),
        [
            (
                '.yaml',
                'capabilities:\n  - Skills:\n      directories: skills\n      include:\n        - from-spec\n',
            ),
            (
                '.json',
                '{"capabilities": [{"Skills": {"directories": "skills", "include": ["from-spec"]}}]}',
            ),
        ],
    )
    async def test_agent_spec_constructs_skills(self, tmp_path: Path, suffix: str, spec_text: str) -> None:
        _write_skill(tmp_path / 'skills', 'from-spec')
        _write_skill(tmp_path / 'skills', 'not-selected')
        spec = tmp_path / f'agent{suffix}'
        spec.write_text(spec_text, encoding='utf-8')
        run = _Run()

        # `from_file` reads the spec synchronously; keep it off the event loop.
        agent = await anyio.to_thread.run_sync(
            lambda: Agent.from_file(spec, custom_capability_types=[Skills], model=_model(run, None))
        )
        await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))

        assert run.instructions == _catalog('- from-spec: Help with the task.')


def _combined(*skills: Skills[Any]) -> Skills[Any]:
    merged = Skills.combine(list(skills))
    assert isinstance(merged, Skills)
    return merged


async def _names(skills: Skills[Any], workspace: Path) -> list[str]:
    """The skill names a run over `workspace` lists, in catalog order."""
    run = await _run(skills, workspace)
    if run.instructions is None:
        return []
    return [line[2:].split(':', 1)[0] for line in run.instructions.splitlines()[1:] if line.startswith('- ')]


async def _descriptions(skills: Skills[Any], workspace: Path) -> str:
    run = await _run(skills, workspace)
    assert run.instructions is not None
    return run.instructions.split('\n', 1)[1]


class TestSkillValidation:
    @pytest.mark.parametrize(('include', 'exclude'), [(['alpha'], ['alpha']), ([], [])])
    def test_runtime_rejects_include_and_exclude_together(self, include: list[str], exclude: list[str]) -> None:
        with pytest.raises(ValueError, match='include and exclude cannot be used together'):
            Skills.from_spec('skills', include=include, exclude=exclude)

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    async def test_unknown_selected_skill_is_rejected(self, tmp_path: Path, selector: str) -> None:
        _write_skill(tmp_path / 'skills', 'available')
        skills = (
            Skills('skills', include=['missing']) if selector == 'include' else Skills('skills', exclude=['missing'])
        )

        with pytest.raises(ValueError, match=rf'Unknown skill in {selector}: missing.*Available skills: available'):
            await _run(skills, tmp_path)

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    def test_selector_must_not_be_a_string(self, selector: str) -> None:
        with pytest.raises(TypeError, match=f'{selector} must be a collection of skill names'):
            Skills.from_spec('skills', **{selector: 'alpha'})

    def test_selector_entries_must_be_strings(self) -> None:
        with pytest.raises(TypeError, match='include must contain only skill names as strings'):
            Skills.from_spec('skills', include=[1])

    async def test_multiple_unknown_skills_report_an_empty_library(self, tmp_path: Path) -> None:
        (tmp_path / 'skills').mkdir()

        with pytest.raises(
            ValueError,
            match=r'Unknown skills in include: first, second\. Available skills: \(none\)\.',
        ):
            await _run(Skills('skills', include=['second', 'first']), tmp_path)

    async def test_selection_happens_before_frontmatter_parsing(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'selected')
        _write_skill(tmp_path / 'skills', 'ignored', frontmatter='not: [valid')

        assert await _names(Skills('skills', include=['selected']), tmp_path) == ['selected']

    async def test_empty_exclude_exposes_all_skills(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha')

        assert await _names(Skills('skills', exclude=[]), tmp_path) == ['alpha']

    @pytest.mark.parametrize('name', ['123', 'on', 'true', 'null', '技能', 'мой-навык'])
    async def test_yaml_like_and_unicode_names_are_preserved_as_text(self, tmp_path: Path, name: str) -> None:
        _write_skill(tmp_path / 'skills', name, frontmatter=f'name: {name}\ndescription: Help')

        assert await _names(Skills('skills'), tmp_path) == [name]

    async def test_skill_name_uses_nfkc_normalization(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'café', frontmatter='name: café\ndescription: Help')

        assert await _names(Skills('skills', include=['café']), tmp_path) == ['café']

    @pytest.mark.parametrize('description', ['123', 'yes', 'null'])
    async def test_yaml_like_description_is_preserved_as_text(self, tmp_path: Path, description: str) -> None:
        _write_skill(tmp_path / 'skills', 'alpha', frontmatter=f'name: alpha\ndescription: {description}')

        assert await _descriptions(Skills('skills'), tmp_path) == f'- alpha: {description}'

    async def test_explicit_name_must_match_directory(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'actual', frontmatter='name: different\ndescription: Help')

        with pytest.warns(UserWarning, match='must match its parent directory'):
            await _run(Skills('skills'), tmp_path)

    @pytest.mark.parametrize('name', ['', ' alpha '])
    async def test_invalid_explicit_name_is_not_normalized(self, tmp_path: Path, name: str) -> None:
        _write_skill(tmp_path / 'skills', 'alpha', frontmatter=f'name: "{name}"\ndescription: Help')

        with pytest.warns(UserWarning, match='Invalid skill name'):
            await _run(Skills('skills'), tmp_path)

    @pytest.mark.parametrize('name', ['Uppercase', '-leading', 'trailing-', 'two--hyphens', 'under_score', 'a' * 65])
    async def test_invalid_derived_name_is_rejected(self, tmp_path: Path, name: str) -> None:
        _write_skill(tmp_path / 'skills', name)

        with pytest.warns(UserWarning, match='Invalid skill name.*at most 64'):
            await _run(Skills('skills'), tmp_path)

    @pytest.mark.parametrize(
        ('text', 'error'),
        [
            ('description: no delimiters', 'must start with YAML frontmatter'),
            ('  ---\ndescription: indented opening\n---', 'must start with YAML frontmatter'),
            ('---\ndescription: unclosed', 'unclosed YAML frontmatter'),
            ('---\ndescription: [invalid\n---', 'Invalid YAML frontmatter'),
            ('---\n? [complex, key]\n: value\n---', 'Invalid YAML frontmatter'),
            ('---\n- description\n---', 'must be a mapping'),
            ('---\nname: okay\n---', 'Invalid Agent Skill frontmatter'),
            ('---\ndescription: "   "\n---', 'must not be empty'),
            ('---\ndescription: First\ndescription: Second\n---', "found duplicate key 'description'"),
        ],
    )
    async def test_invalid_frontmatter_is_rejected(self, tmp_path: Path, text: str, error: str) -> None:
        directory = tmp_path / 'skills' / 'invalid'
        directory.mkdir(parents=True)
        (directory / 'SKILL.md').write_text(text, encoding='utf-8')

        with pytest.warns(UserWarning, match=error):
            await _run(Skills('skills'), tmp_path)

    async def test_non_utf8_skill_names_its_file(self, tmp_path: Path) -> None:
        directory = tmp_path / 'skills' / 'latin'
        directory.mkdir(parents=True)
        (directory / 'SKILL.md').write_bytes(b'---\ndescription: \xe9\n---\n')
        with pytest.warns(UserWarning, match=r'SKILL.md is not valid UTF-8'):
            await _run(Skills('skills'), tmp_path)

    async def test_invalid_skill_does_not_hide_valid_skills(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'good', frontmatter='description: A valid skill.')
        bad = tmp_path / 'skills' / 'bad'
        bad.mkdir()
        (bad / 'SKILL.md').write_text('not frontmatter')
        with pytest.warns(UserWarning, match='SKILL.md'):
            assert '- good: A valid skill.' in await _descriptions(Skills('skills'), tmp_path)

    async def test_multiline_description_continues_on_indented_lines(self, tmp_path: Path) -> None:
        # An indented `---` inside a block scalar is not a frontmatter delimiter.
        _write_skill(tmp_path / 'skills', 'multiline', frontmatter='description: |\n  First line.\n  ---\n  Last line.')

        assert await _descriptions(Skills('skills'), tmp_path) == '- multiline: First line.\n  ---\n  Last line.'

    async def test_overlong_description_warns_and_is_preserved(self, tmp_path: Path) -> None:
        description = 'x' * 1025
        _write_skill(tmp_path / 'skills', 'verbose', description=description)

        with pytest.warns(UserWarning, match=r'verbose \(1,025 characters\)'):
            listed = await _descriptions(Skills('skills'), tmp_path)

        assert listed == f'- verbose: {description}'

    async def test_duplicate_names_across_roots_are_rejected(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'first', 'duplicate')
        _write_skill(tmp_path / 'second', 'duplicate')

        with pytest.raises(ValueError, match="Duplicate skill name 'duplicate'"):
            await _run(Skills(['first', 'second']), tmp_path)

    async def test_duplicate_root_is_scanned_once(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'once')

        assert await _names(Skills(['skills', tmp_path / 'skills', 'other/../skills']), tmp_path) == ['once']

    def test_at_least_one_library_is_required(self) -> None:
        with pytest.raises(ValueError, match='requires at least one skill-library directory'):
            Skills([])

    async def test_skill_package_path_is_rejected(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'alpha')

        with pytest.raises(ValueError, match=r'points to a skill package.*Pass its parent directory'):
            await _run(Skills('skills/alpha'), tmp_path)

    async def test_missing_root_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='does not exist in the workspace: missing'):
            await _run(Skills('missing'), tmp_path)

    async def test_file_root_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / 'skills').write_text('not a directory', encoding='utf-8')

        with pytest.raises(ValueError, match='is not a directory'):
            await _run(Skills('skills'), tmp_path)

    async def test_non_skill_children_are_ignored(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()
        (library / 'README.md').write_text('ordinary file', encoding='utf-8')
        (library / 'not-a-skill').mkdir()
        (library / 'dir-named-skill' / 'SKILL.md').mkdir(parents=True)

        assert await _names(Skills('skills'), tmp_path) == []

    async def test_nested_skill_md_is_not_a_skill(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / 'skills', 'outer', files={'references/SKILL.md': 'reference'})

        assert await _names(Skills('skills'), tmp_path) == ['outer']

    async def test_behavioral_fields_are_ignored_with_one_warning(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'first', frontmatter='description: First\nallowed-tools: Read\nmodel: sonnet')
        _write_skill(library, 'second', frontmatter='description: Second\ndisable-model-invocation: true')

        with pytest.warns(UserWarning) as caught:
            await _run(Skills('skills'), tmp_path)

        assert [str(warning.message) for warning in caught] == [
            'Ignoring unsupported Agent Skill behavioral frontmatter fields: '
            'first: allowed-tools, model; second: disable-model-invocation'
        ]

    async def test_standard_non_behavioral_fields_are_accepted(self, tmp_path: Path) -> None:
        _write_skill(
            tmp_path / 'skills',
            'standard',
            frontmatter='description: Standard\nlicense: Apache-2.0\ncompatibility: Python\nmetadata:\n  owner: pydantic',
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await _run(Skills('skills'), tmp_path)
