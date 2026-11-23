from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_enforcer_tags
from _ratchet import Change
from check_enforcer_tags import findings, units

# The paragraph https://github.com/pydantic/pydantic-ai/pull/12670 added a hard rule to, in
# `.agents/skills/pushing-commits-to-the-repo/SKILL.md`. The added line restates `package-label.yml`.
_SKILL = '.agents/skills/pushing-commits-to-the-repo/SKILL.md'
_BEFORE_12670 = """\
Never fetch labels individually or pass raw label API responses.

Keep the category label aligned with the PR's purpose: `bug`, `feature`, `docs`, `chore` or `dependency`. A refactor also carries `refactor`, alongside `chore`.
Add existing topic labels for every subject materially covered by the final title, body, and diff.
"""
_RULE_12670 = (
    'A PR labelled `test-suite` or `github-actions` takes `chore`, never `bug`; '
    '`.github/workflows/package-label.yml` fails it otherwise.'
)


def _added(old: str | None, new: str, path: str = 'agent_docs/x.md') -> list[tuple[int, str]]:
    return [(f.line, f.message) for f in findings([Change(path if old is not None else None, path, old, new)])]


def test_the_rule_12670_added_needs_a_tag():
    after = _BEFORE_12670.replace('`chore`.\n', f'`chore`.\n{_RULE_12670}\n')
    assert _added(_BEFORE_12670, after, _SKILL) == [
        (4, 'end it with [gate: `<path>`], naming the check that fails a change breaking it, or `[review]`')
    ]
    tagged = after.replace(_RULE_12670, f'{_RULE_12670} [gate: `.github/workflows/package-label.yml`]')
    assert _added(_BEFORE_12670, tagged, _SKILL) == []


@pytest.mark.parametrize(
    ('source', 'texts'),
    [
        pytest.param('- one\n  more\n- two\n', ['- one\n  more', '- two'], id='list-items'),
        pytest.param('Wrapped\nsentence. Next.\nOwn line.\n', ['Wrapped\nsentence. Next.', 'Own line.'], id='prose'),
        pytest.param('---\nname: x\n---\nBody.\n', ['Body.'], id='frontmatter'),
        pytest.param('# Never\n| never |\n<!-- never -->\n```\nnever\n```\nText.\n', ['Text.'], id='skipped'),
    ],
)
def test_units(source: str, texts: list[str]):
    assert [rule.text for rule in units(source)] == texts


@pytest.mark.parametrize(
    ('line', 'problem'),
    [
        pytest.param('- Never do X.', 'end it with', id='untagged'),
        pytest.param("- Don't do X. [review]", None, id='review'),
        pytest.param('- Never do X. [gate: `scripts/_ratchet.py`]', None, id='gate'),
        pytest.param(
            '- Never do X. [gate: `scripts/missing.py`]', 'the gate `scripts/missing.py` does not exist', id='gone'
        ),
        pytest.param('- Never do X. [gate: scripts/_ratchet.py]', 'write the tag as', id='malformed'),
        pytest.param('- Prefer X; see `never_do`.', None, id='soft'),
    ],
)
def test_a_new_hard_rule_needs_a_tag_that_resolves(line: str, problem: str | None):
    found = [message for _, message in _added('', line + '\n')]
    if problem is None:
        assert found == []
    else:
        assert [message[: len(problem)] for message in found] == [problem]


def test_existing_and_moved_rules_keep_their_standing():
    rule = '- You must do X.\n'
    assert _added(rule, rule + '- Prefer Y.\n') == []
    assert _added(rule, '- You must do X and Y.\n') != []
    assert _added('- Never edit `src/a.py`.\n', '- Never edit `src/b/a.py`.\n') == []
    moved = findings(
        [Change('agent_docs/a.md', 'agent_docs/a.md', rule, ''), Change(None, 'agent_docs/b.md', None, rule)]
    )
    assert moved == []


def test_only_the_skill_file_states_rules_under_a_skill_root():
    assert _added(None, '- Never do X.\n', '.agents/skills/x/template.md') == []
    assert _added(None, '- Never do X.\n', '.agents/skills/x/SKILL.md') != []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True)


def test_main_compares_instruction_files_with_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'agent_docs').mkdir()
    (tmp_path / 'agent_docs/x.md').write_text('- Prefer X.\n')
    _git(tmp_path, 'init', '--quiet')
    _git(tmp_path, 'add', '--all')
    _git(tmp_path, '-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '--quiet', '-m', 'base')
    (tmp_path / 'agent_docs/x.md').write_text('- Prefer X.\n- Never do Y.\n')
    (tmp_path / 'notes.md').write_text('- Never do Z.\n')
    assert check_enforcer_tags.main(['--base', 'HEAD']) == 1
    out = capsys.readouterr().out
    assert out.startswith('agent_docs/x.md:2: hard rule without an enforcer tag: end it with')
    assert 'notes.md' not in out
    _git(tmp_path, 'add', '--all')
    assert check_enforcer_tags.main(['--staged', '--format', 'github']) == 1
    assert capsys.readouterr().out.startswith('::error file=agent_docs/x.md,line=2,title=Enforcer tag::')
