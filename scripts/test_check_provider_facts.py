from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_provider_facts
from check_provider_facts import FACTS, REGISTERED, ROOTS

_REPO = Path(__file__).parents[1]

_MISTRAL = 'pydantic_ai_slim/pydantic_ai/providers/mistral.py'
_GROK = 'pydantic_ai_slim/pydantic_ai/profiles/grok.py'

_SETS = """\
_ADJUSTABLE_REASONING_MODELS = frozenset(
    {
        'mistral-small-latest',  # fact: known
        'mistral-medium-latest',
    }
)
_LARGE_4_MODELS = frozenset({'mistral-large-4'})
_UNREGISTERED = frozenset({'mistral-tiny'})
"""

_REGISTRY = """\
FACTS = (
    ProviderFact(id='known', claim='', where='', probe='', expect='rejects'),
)
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, _MISTRAL, _SETS)
    _write(tmp_path, _GROK, "_GROK_47_REASONING_MODELS = frozenset({'grok-4.7'})\n")
    _write(tmp_path, FACTS, _REGISTRY)
    _write(tmp_path, 'outside/sets.py', "_LARGE_4_MODELS = frozenset({'a'})\n")
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, name: str, source: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _commit(repo: Path) -> None:
    _git(repo, 'add', '--all')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '--quiet', '-m', 'commit')


def _check(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = check_provider_facts.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def _flagged(out: str) -> list[str]:
    """The `path:line: member` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_untagged_member_fails_and_says_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _SETS.replace("'mistral-medium-latest',\n", "'mistral-medium-latest',\n        'mistral-medium-4',\n")

    code, out = _staged(repo, capsys, _MISTRAL, source)

    assert code == 1
    assert f'{_MISTRAL}: untagged 2 -> 3' in out
    assert _flagged(out) == [f"{_MISTRAL}:5: 'mistral-medium-4' in _ADJUSTABLE_REASONING_MODELS"]
    assert f'add a `ProviderFact` row naming it to `{FACTS}`' in out
    assert '`# fact: <id>`' in out


@pytest.mark.parametrize(
    'member',
    [
        "        'mistral-medium-4',  # fact: known\n",
        "        # fact: known\n        'mistral-medium-4',\n",
    ],
)
def test_a_new_member_tagged_with_a_row_passes(repo: Path, capsys: pytest.CaptureFixture[str], member: str):
    source = _SETS.replace("'mistral-medium-latest',\n", f"'mistral-medium-latest',\n{member}")

    assert _staged(repo, capsys, _MISTRAL, source) == (0, '')


def test_a_tag_that_names_no_row_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _SETS.replace(
        "'mistral-medium-latest',\n", "'mistral-medium-latest',\n        'mistral-medium-4',  # fact: nope\n"
    )

    code, out = _staged(repo, capsys, _MISTRAL, source)

    assert code == 1
    assert f'{_MISTRAL}:5: `# fact: nope` names no `ProviderFact` row in {FACTS}' in out
    assert f'{_MISTRAL}: untagged 2 -> 3' in out


def test_a_row_added_with_its_first_tag_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    row = "    ProviderFact(id='new', claim='', where='', probe='', expect='rejects'),\n"
    _write(repo, FACTS, _REGISTRY.replace('\n)', f'\n{row})'))
    source = _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'mistral-large-4-1'}  # fact: new")

    assert _staged(repo, capsys, _MISTRAL, source) == (0, '')


def test_debt_that_holds_or_shrinks_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, _MISTRAL, _SETS.replace("        'mistral-medium-latest',\n", '')) == (0, '')
    assert _staged(repo, capsys, _MISTRAL, _SETS.replace('mistral-medium-latest', 'mistral-medium-3')) == (0, '')


def test_merging_a_set_into_a_registered_one_adds_nothing(repo: Path, capsys: pytest.CaptureFixture[str]):
    merged = _SETS.replace(
        "{'mistral-large-4'})\n_UNREGISTERED = frozenset({'mistral-tiny'})", "{'mistral-large-4', 'mistral-tiny'})"
    )

    code, out = _staged(repo, capsys, _MISTRAL, merged)

    assert (code, out) == (0, '')


def test_a_name_the_file_had_once_counts_when_a_second_set_gains_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'mistral-tiny'}")

    code, out = _staged(repo, capsys, _MISTRAL, source)

    assert code == 1
    assert _flagged(out) == [f"{_MISTRAL}:7: 'mistral-tiny' in _LARGE_4_MODELS"]


def test_moving_a_member_between_files_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _MISTRAL, _SETS.replace("{'mistral-large-4'}", 'set()'))

    assert _staged(repo, capsys, _GROK, "_GROK_47_REASONING_MODELS = frozenset({'grok-4.7', 'mistral-large-4'})\n") == (
        0,
        '',
    )


def test_files_outside_the_roots_and_files_that_do_not_parse_are_skipped(
    repo: Path, capsys: pytest.CaptureFixture[str]
):
    _write(repo, 'outside/sets.py', "_LARGE_4_MODELS = frozenset({'a', 'b'})\n")

    broken = _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'x'}") + 'def (:\n'

    assert _staged(repo, capsys, _MISTRAL, broken) == (0, '')


def test_staged_mode_reads_the_index_not_the_working_tree(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _MISTRAL, _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'mistral-large-5'}"))

    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD')[0] == 1


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as the Mistral sets on another branch, then start merging it into this one."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, _MISTRAL, theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'pydantic_ai_slim/pydantic_ai/models/mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


_THEIRS = _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'mistral-large-5'}")


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _THEIRS)

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _THEIRS)

    code, out = _staged(
        repo, capsys, _MISTRAL, _THEIRS.replace("'mistral-large-5'", "'mistral-large-5', 'mistral-large-6'")
    )

    assert code == 1
    assert f'{_MISTRAL}: untagged 3 -> 4' in out


def test_github_format_annotates_each_added_member(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _MISTRAL, _SETS.replace("{'mistral-large-4'}", "{'mistral-large-4', 'mistral-large-5'}"))

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith(f'::error file={_MISTRAL},line=7,title=Provider facts%3A untagged 2 -> 3::')


def test_every_registered_set_exists_and_has_members():
    for path, symbols in REGISTERED.items():
        assert path.startswith(ROOTS), path
        tree = ast.parse((_REPO / path).read_text(encoding='utf-8'))
        found = {m.symbol for m in check_provider_facts._members(tree, symbols)}
        assert found == symbols, (path, symbols - found)


def test_every_fact_tag_names_a_row():
    facts = check_provider_facts._fact_ids((_REPO / FACTS).read_text(encoding='utf-8'))
    unknown = [
        f'{path.relative_to(_REPO)}:{tag.line}: {tag.fact}'
        for root in ROOTS
        for path in sorted((_REPO / root).rglob('*.py'))
        for tag in check_provider_facts._tags(path.read_text(encoding='utf-8'))
        if tag.fact not in facts
    ]
    assert unknown == []
