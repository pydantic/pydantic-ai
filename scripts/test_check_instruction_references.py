from __future__ import annotations

import pytest
from check_instruction_references import INSTRUCTION_FILE, find_problems, named_paths


def check(files: dict[str, str], ignored: frozenset[str] = frozenset()) -> list[str]:
    instructions = [path for path in sorted(files) if INSTRUCTION_FILE.search(path)]
    return find_problems(sorted(files), instructions, files.__getitem__, lambda paths: set(paths) & ignored)


@pytest.mark.parametrize(
    ('text', 'paths'),
    [
        pytest.param('See `tests/AGENTS.md` and `docs/`.', ['tests/AGENTS.md', 'docs/'], id='file-and-directory'),
        pytest.param('```\n`tests/AGENTS.md`\n```', [], id='fenced-block'),
        pytest.param('[`tests/x.md`](https://example.com/tests/x.md)', [], id='url-link-text'),
        pytest.param('[`tests/x.md`](../tests/x.md)', ['tests/x.md'], id='relative-link-text'),
        pytest.param('`tests/x.py::test_y` `<path>/x.md` `src/**/*.py` `a b/c.md`', [], id='not-a-path'),
        pytest.param('`gh pr view` `pydantic-ai` `area:agent`', [], id='no-slash'),
    ],
)
def test_named_paths(text: str, paths: list[str]):
    assert [path for _, path in named_paths(text)] == paths


def test_a_path_resolves_from_the_root_an_ancestor_or_its_own_top_level_directory():
    package = {'src/pkg/docs/guide.md': '', 'tests/pkg/test_x.py': '', 'pydantic_graph/pydantic_graph/graph.py': ''}
    guide = '`docs/guide.md` `tests/pkg/test_x.py` `pydantic_graph/graph.py`'
    assert check({**package, 'src/pkg/AGENTS.md': guide}) == []
    # A path elsewhere in the repository with the same tail does not count.
    assert check({'src/pkg/tests/x.py': '', 'tests/y.py': '', 'AGENTS.md': '`tests/x.py`'}) == [
        'AGENTS.md:1: `tests/x.py` does not exist'
    ]


def test_a_moved_file_leaves_a_missing_path():
    # https://github.com/pydantic/pydantic-ai/pull/13885 moved the pax model catalog into `models/`.
    files = {
        'src/pydantic_ai_pax/pydantic_ai_pax/models/model_catalog.py': '',
        '.agents/skills/add-new-model/SKILL.md': '`CODEX_MODELS` in `src/pydantic_ai_pax/pydantic_ai_pax/model_catalog.py`',
        '.claude/skills/add-new-model': '',
    }
    assert check(files) == [
        '.agents/skills/add-new-model/SKILL.md:1: `src/pydantic_ai_pax/pydantic_ai_pax/model_catalog.py` does not exist'
    ]


def test_a_git_ignored_path_is_runtime_state():
    files = {'AGENTS.md': 'Write `.claude/skills/branch-context/issue-brief.md`.', '.claude/x': ''}
    assert check(files) == ['AGENTS.md:1: `.claude/skills/branch-context/issue-brief.md` does not exist']
    assert check(files, frozenset({'.claude/skills/branch-context/issue-brief.md'})) == []


def test_a_shared_skill_needs_its_claude_code_entry():
    # https://github.com/pydantic/pydantic-ai/pull/9073 added the two links this skill set lacked.
    skill = {'.agents/skills/add-new-model/SKILL.md': ''}
    assert check(skill) == [
        '.agents/skills/add-new-model: add the `.claude/skills/add-new-model` symlink so Claude Code loads the skill'
    ]
    assert check({**skill, '.claude/skills/add-new-model': ''}) == []


@pytest.mark.parametrize(
    ('text', 'fails'),
    [
        # https://github.com/pydantic/pydantic-ai/pull/8858 and https://github.com/pydantic/pydantic-ai/pull/9651
        ('[version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)', True),
        ('import `pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main`', True),
        ('[#415](https://github.com/pydantic/pydantic-ai-harness/issues/415)', False),
        ('[harness PR #870](https://github.com/pydantic/pydantic-ai-harness/pull/870)', False),
    ],
)
def test_the_retired_repository_appears_only_in_issue_and_pull_request_links(text: str, fails: bool):
    shipped = 'src/pydantic_ai_harness/pydantic_ai_harness/.agents/skills/harness/SKILL.md'
    assert bool(check({shipped: text})) is fails
