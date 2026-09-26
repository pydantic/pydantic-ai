from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, LoadCapabilityReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness.skills import Skills

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


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


def _leaves(skills: Skills[object]) -> list[AbstractCapability[object]]:
    leaves: list[AbstractCapability[object]] = []
    skills.apply(leaves.append)
    return leaves


async def _load_skill(skills: Skills[object], skill_name: str) -> str:
    class LoadSkillModel(TestModel):
        def gen_tool_args(self, tool_def: ToolDefinition) -> dict[str, str]:
            return {'id': skill_name}

    agent: Agent[object, str] = Agent(LoadSkillModel(call_tools=['load_capability']), capabilities=[skills])
    result = await agent.run(f'use the {skill_name} skill')
    returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    return returns[-1].content.get('instructions', '')


class TestSkills:
    def test_public_constructor_only_exposes_skill_library_configuration(self) -> None:
        assert tuple(inspect.signature(Skills).parameters) == ('directories', 'include', 'exclude')

    def test_repr_only_exposes_skill_library_configuration(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        assert repr(Skills(library)) == (f'Skills(directories=({library!r},), include=None, exclude=frozenset())')

    def test_directory_name_is_used_when_frontmatter_omits_name(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')

        leaves = _leaves(Skills(library))

        assert [leaf.id for leaf in leaves] == ['alpha']

    def test_frontmatter_description_is_used_for_the_capability(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')

        leaves = _leaves(Skills(library))

        assert [leaf.description for leaf in leaves] == ['Alpha help.']

    def test_selected_skills_are_deferred_capability_leaves(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        leaves = _leaves(Skills(library))

        assert [leaf.defer_loading for leaf in leaves] == [True]

    async def test_two_libraries_both_stay_reachable(self, tmp_path: Path) -> None:
        """`Skills` needs no default `id`, because the leaves it makes already carry one.

        It is a factory rather than a configuration: each selected skill becomes its own deferred
        capability named after the skill. Two libraries therefore contribute disjoint sets that
        both stay loadable, where merging the two `Skills` capabilities would drop a library.
        """
        first, second = tmp_path / 'first', tmp_path / 'second'
        _write_skill(first, 'alpha', body='Alpha directions.')
        _write_skill(second, 'beta', body='Beta directions.')

        agent = Agent(TestModel(), capabilities=[Skills(first), Skills(second)])
        loadable = {
            leaf.id
            for leaf in leaf_capabilities(agent._root_capability)
            if leaf.defer_loading  # pyright: ignore[reportPrivateUsage]
        }
        assert {'alpha', 'beta'} <= loadable

    def test_two_libraries_sharing_a_skill_name_collide(self, tmp_path: Path) -> None:
        """Two skills claiming one name is ambiguous, and says so rather than picking one."""
        first, second = tmp_path / 'first', tmp_path / 'second'
        _write_skill(first, 'shared')
        _write_skill(second, 'shared')

        with pytest.raises(UserError, match="Capability id 'shared' is used by multiple capabilities"):
            Agent(TestModel(), capabilities=[Skills(first), Skills(second)])

    def test_include_exposes_only_selected_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')
        _write_skill(library, 'beta')

        leaves = _leaves(Skills(library, include=['beta']))

        assert [leaf.id for leaf in leaves] == ['beta']

    def test_exclude_hides_selected_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')
        _write_skill(library, 'beta')

        leaves = _leaves(Skills(library, exclude=['alpha']))

        assert [leaf.id for leaf in leaves] == ['beta']

    async def test_loaded_instructions_contain_only_heading_and_body(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'knowledge', body='Answer from this embedded guidance.')

        loaded = await _load_skill(Skills(library), 'knowledge')

        assert loaded == '# Skill: knowledge\n\nAnswer from this embedded guidance.'

    async def test_empty_skill_body_loads_only_the_skill_heading(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'empty', body='')

        loaded = await _load_skill(Skills(library), 'empty')

        assert loaded == '# Skill: empty'

    async def test_skill_body_preserves_markdown_indentation(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'code-example', body='    print("hello")')

        loaded = await _load_skill(Skills(library), 'code-example')

        assert loaded == '# Skill: code-example\n\n    print("hello")'

    async def test_skill_body_omits_trailing_blank_lines(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'concise', body='Do the task.\n\n')

        loaded = await _load_skill(Skills(library), 'concise')

        assert loaded == '# Skill: concise\n\nDo the task.'

    async def test_skill_directory_placeholder_is_not_resolved_to_a_host_path(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'portable', body='Read ${CLAUDE_SKILL_DIR}/guide.md.')

        loaded = await _load_skill(Skills(library), 'portable')

        assert loaded == '# Skill: portable\n\nRead ${CLAUDE_SKILL_DIR}/guide.md.'

    async def test_bundled_files_do_not_change_loaded_instructions(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'portable',
            body='Follow the portable workflow.',
            files={'references/guide.md': 'Use the documented workflow.'},
        )

        loaded = await _load_skill(Skills(library), 'portable')

        assert loaded == '# Skill: portable\n\nFollow the portable workflow.'

    def test_construction_is_a_snapshot(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'first')
        skills = Skills(library)
        _write_skill(library, 'later')

        leaves = _leaves(skills)

        assert [leaf.id for leaf in leaves] == ['first']

    @pytest.mark.parametrize(
        ('suffix', 'spec_text'),
        [
            (
                '.yaml',
                'capabilities:\n  - Skills:\n      directories: {library}\n      include:\n        - from-spec\n',
            ),
            (
                '.json',
                '{{"capabilities": [{{"Skills": {{"directories": "{library}", "include": ["from-spec"]}}}}]}}',
            ),
        ],
    )
    async def test_agent_spec_constructs_skills(self, tmp_path: Path, suffix: str, spec_text: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'from-spec')
        _write_skill(library, 'not-selected')
        spec = tmp_path / f'agent{suffix}'
        spec.write_text(spec_text.format(library=library), encoding='utf-8')

        model = TestModel(call_tools=[])
        agent = Agent.from_file(spec, custom_capability_types=[Skills], model=model)
        await agent.run('go')
        assert model.last_model_request_parameters is not None
        instructions = InstructionPart.join(model.last_model_request_parameters.instruction_parts or [])
        assert 'from-spec' in (instructions or '')
        assert 'not-selected' not in (instructions or '')


class TestSkillValidation:
    @pytest.mark.parametrize(
        ('include', 'exclude'),
        [
            (['alpha'], ['alpha']),
            ([], []),
        ],
    )
    def test_runtime_rejects_include_and_exclude_together(
        self,
        tmp_path: Path,
        include: list[str],
        exclude: list[str],
    ) -> None:
        library = tmp_path / 'skills'
        library.mkdir()

        with pytest.raises(ValueError, match='include and exclude cannot be used together'):
            Skills.from_spec(library, include=include, exclude=exclude)

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    def test_unknown_selected_skill_is_rejected(self, tmp_path: Path, selector: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'available')

        with pytest.raises(ValueError, match=rf'Unknown skill in {selector}: missing.*Available skills: available'):
            if selector == 'include':
                Skills(library, include=['missing'])
            else:
                Skills(library, exclude=['missing'])

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    def test_selector_must_not_be_a_string(self, tmp_path: Path, selector: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        with pytest.raises(TypeError, match=f'{selector} must be a collection of skill names'):
            if selector == 'include':
                Skills.from_spec(library, include='alpha')
            else:
                Skills.from_spec(library, exclude='alpha')

    def test_selector_entries_must_be_strings(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        with pytest.raises(TypeError, match='include must contain only skill names as strings'):
            Skills.from_spec(library, include=[1])

    def test_multiple_unknown_skills_report_an_empty_library(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()

        with pytest.raises(
            ValueError,
            match=r'Unknown skills in include: first, second\. Available skills: \(none\)\.',
        ):
            Skills(library, include=['second', 'first'])

    def test_selection_happens_before_frontmatter_parsing(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'selected')
        _write_skill(library, 'ignored', frontmatter='not: [valid')

        assert [leaf.id for leaf in _leaves(Skills(library, include=['selected']))] == ['selected']

    def test_empty_include_exposes_no_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        assert _leaves(Skills(library, include=[])) == []

    def test_empty_exclude_exposes_all_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        leaves = _leaves(Skills(library, exclude=[]))

        assert [leaf.id for leaf in leaves] == ['alpha']

    @pytest.mark.parametrize('name', ['123', 'on', 'true', 'null'])
    def test_yaml_like_skill_name_is_preserved_as_text(self, tmp_path: Path, name: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, name, frontmatter=f'name: {name}\ndescription: Help')

        assert [leaf.id for leaf in _leaves(Skills(library))] == [name]

    @pytest.mark.parametrize('name', ['技能', 'мой-навык'])
    def test_unicode_skill_name_is_supported(self, tmp_path: Path, name: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, name, frontmatter=f'name: {name}\ndescription: Help')

        assert [leaf.id for leaf in _leaves(Skills(library))] == [name]

    def test_skill_name_uses_nfkc_normalization(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        composed_name = 'café'
        decomposed_name = 'cafe\u0301'
        _write_skill(
            library,
            composed_name,
            frontmatter=f'name: {decomposed_name}\ndescription: Help',
        )

        leaves = _leaves(Skills(library, include=[decomposed_name]))

        assert [leaf.id for leaf in leaves] == [composed_name]

    @pytest.mark.parametrize('description', ['123', 'yes', 'null'])
    def test_yaml_like_description_is_preserved_as_text(self, tmp_path: Path, description: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', frontmatter=f'name: alpha\ndescription: {description}')

        assert [leaf.description for leaf in _leaves(Skills(library))] == [description]

    def test_explicit_name_must_match_directory(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'actual', frontmatter='name: different\ndescription: Help')

        with pytest.raises(ValueError, match='must match its parent directory'):
            Skills(library)

    @pytest.mark.parametrize('name', ['', ' alpha '])
    def test_invalid_explicit_name_is_not_normalized(self, tmp_path: Path, name: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', frontmatter=f'name: "{name}"\ndescription: Help')

        with pytest.raises(ValueError, match='Invalid skill name'):
            Skills(library)

    @pytest.mark.parametrize('name', ['Uppercase', '-leading', 'trailing-', 'two--hyphens', 'under_score'])
    def test_invalid_derived_name_is_rejected(self, tmp_path: Path, name: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, name)

        with pytest.raises(ValueError, match='Invalid skill name'):
            Skills(library)

    def test_name_length_is_limited(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'a' * 65)

        with pytest.raises(ValueError, match='at most 64'):
            Skills(library)

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
        ],
    )
    def test_invalid_frontmatter_is_rejected(self, tmp_path: Path, text: str, error: str) -> None:
        directory = tmp_path / 'skills' / 'invalid'
        directory.mkdir(parents=True)
        (directory / 'SKILL.md').write_text(text, encoding='utf-8')

        with pytest.raises(ValueError, match=error):
            Skills(tmp_path / 'skills')

    def test_indented_separator_is_not_a_frontmatter_delimiter(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'multiline',
            frontmatter='description: |\n  First line.\n  ---\n  Last line.',
        )

        leaves = _leaves(Skills(library))

        assert [leaf.description for leaf in leaves] == ['First line.\n---\nLast line.']

    def test_duplicate_frontmatter_keys_are_rejected(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'duplicate',
            frontmatter='description: First\ndescription: Second',
        )

        with pytest.raises(ValueError, match="found duplicate key 'description'"):
            Skills(library)

    def test_overlong_description_warns_and_is_preserved(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        description = 'x' * 1025
        _write_skill(library, 'verbose', description=description)

        with pytest.warns(UserWarning, match=r'verbose \(1,025 characters\)'):
            leaves = _leaves(Skills(library))

        assert [leaf.description for leaf in leaves] == [description]

    def test_duplicate_names_across_roots_are_rejected(self, tmp_path: Path) -> None:
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        _write_skill(first, 'duplicate')
        _write_skill(second, 'duplicate')

        with pytest.raises(ValueError, match="Duplicate skill name 'duplicate'"):
            Skills([first, second])

    def test_duplicate_root_is_scanned_once(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'once')

        assert [leaf.id for leaf in _leaves(Skills([library, library.resolve()]))] == ['once']

    def test_missing_duplicate_root_is_rejected(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'once')

        with pytest.raises(ValueError, match='does not exist'):
            Skills([library, library / 'missing' / '..'])

    def test_at_least_one_library_is_required(self) -> None:
        with pytest.raises(ValueError, match='requires at least one skill-library directory'):
            Skills([])

    def test_skill_package_path_is_rejected(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        with pytest.raises(ValueError, match='points to a skill package.*Pass its parent directory'):
            Skills(library / 'alpha')

    def test_missing_root_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='does not exist'):
            Skills(tmp_path / 'missing')

    def test_file_root_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'skills'
        path.write_text('not a directory', encoding='utf-8')

        with pytest.raises(ValueError, match='is not a directory'):
            Skills(path)

    def test_non_skill_children_are_ignored(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()
        (library / 'README.md').write_text('ordinary file', encoding='utf-8')
        (library / 'not-a-skill').mkdir()

        assert _leaves(Skills(library)) == []

    def test_nested_skill_md_is_not_a_skill(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'outer', files={'references/SKILL.md': 'reference'})

        leaves = _leaves(Skills(library))

        assert [leaf.id for leaf in leaves] == ['outer']

    def test_behavioral_fields_are_ignored_with_one_warning(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'first',
            frontmatter='description: First\nallowed-tools: Read\nmodel: sonnet',
        )
        _write_skill(
            library,
            'second',
            frontmatter='description: Second\ndisable-model-invocation: true',
        )

        with pytest.warns(UserWarning) as caught:
            Skills(library)

        assert [str(warning.message) for warning in caught] == [
            'Ignoring unsupported Agent Skill behavioral frontmatter fields: '
            'first: allowed-tools, model; second: disable-model-invocation'
        ]

    def test_standard_non_behavioral_fields_are_accepted(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'standard',
            frontmatter='description: Standard\nlicense: Apache-2.0\ncompatibility: Python\nmetadata:\n  owner: pydantic',
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Skills(library)
