"""The `TID251` `per-file-ignores` in `pyproject.toml` exempt files from the `asyncio` bans in `banned-api` only.

A `per-file-ignores` entry turns off a rule for the whole file, so a `TID251` entry also turns off the other
`banned-api` entries there, such as `typing.TypedDict`. These tests run ruff once with only those other entries and no
`per-file-ignores`, and once with only the `asyncio` entries, to check that the other bans still hold in every
exempted file, and that each entry still exempts a file that uses an `asyncio` primitive.

`.github/scripts/ruff.toml` repeats the root bans and adds bans on starting a process, which only `_commands.py` may
do there, and on reading an OpenCode answer other than through `parse_answer`. These tests also check that it holds
every root ban, and that each of its `TID251` exemptions still exempts a file that starts a process.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter
from ruff.__main__ import find_ruff_bin

ROOT = Path(__file__).parents[2]
SCRIPTS = '.github/scripts/'

# The `banned-api` entries a `TID251` exemption is for: `asyncio` primitives with an `anyio` replacement, which code
# that only runs on asyncio may use. Every other entry holds in every file.
EXEMPTIBLE = frozenset(
    {
        'asyncio.sleep',
        'asyncio.Event',
        'asyncio.to_thread',
        'asyncio.wait_for',
        'asyncio.timeout',
        'asyncio.Queue',
        'asyncio.Semaphore',
        'asyncio.Condition',
    }
)
# The ban `.github/scripts/ruff.toml` adds besides the process bans.
ANSWER_BAN = 'douwebot_opencode_result.final_text'


class _Ban(BaseModel):
    msg: str


class _TidyImports(BaseModel):
    banned_api: dict[str, _Ban] = Field(alias='banned-api')


class _Lint(BaseModel):
    tidy_imports: _TidyImports = Field(alias='flake8-tidy-imports')
    per_file_ignores: dict[str, list[str]] = Field(alias='per-file-ignores')


class _ScriptsLint(BaseModel):
    tidy_imports: _TidyImports = Field(alias='flake8-tidy-imports')
    extend_per_file_ignores: dict[str, list[str]] = Field(alias='extend-per-file-ignores')


class _Diagnostic(BaseModel):
    filename: Path
    message: str


@cache
def _lint() -> _Lint:
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    return _Lint.model_validate(config['tool']['ruff']['lint'])


@cache
def _scripts_lint() -> _ScriptsLint:
    config = tomllib.loads((ROOT / SCRIPTS / 'ruff.toml').read_text(encoding='utf-8'))
    return _ScriptsLint.model_validate(config['lint'])


def _banned_api() -> dict[str, str]:
    return {name: ban.msg for name, ban in _lint().tidy_imports.banned_api.items()}


def _scripts_banned_api() -> dict[str, str]:
    return {name: ban.msg for name, ban in _scripts_lint().tidy_imports.banned_api.items()}


def _process_bans() -> frozenset[str]:
    return frozenset(_scripts_banned_api().keys() - _banned_api().keys() - {ANSWER_BAN})


def _scripts_exemptions() -> list[str]:
    return [
        SCRIPTS + pattern for pattern, codes in _scripts_lint().extend_per_file_ignores.items() if 'TID251' in codes
    ]


def _exemptions() -> list[str]:
    return [pattern for pattern, codes in _lint().per_file_ignores.items() if 'TID251' in codes]


@cache
def _banned_uses(names: frozenset[str]) -> tuple[tuple[str, str], ...]:
    """Each `(path, message)` that `TID251` reports with only `names` banned and no `per-file-ignores`."""
    bans = _banned_api() | _scripts_banned_api()
    table = ', '.join(
        f'{json.dumps(name)} = {{ msg = {json.dumps(msg)} }}' for name, msg in bans.items() if name in names
    )
    result = subprocess.run(
        [
            find_ruff_bin(),
            'check',
            '--select',
            'TID251',
            # A configuration file named here applies to every file, so `.github/scripts/ruff.toml` and its
            # `extend-per-file-ignores`, which no `--config` override can clear, do not apply.
            '--config',
            str(ROOT / 'pyproject.toml'),
            '--config',
            'lint.per-file-ignores = {}',
            '--config',
            f'lint.flake8-tidy-imports.banned-api = {{ {table} }}',
            '--output-format',
            'json',
            '--exit-zero',
            '.',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    found = TypeAdapter(list[_Diagnostic]).validate_json(result.stdout)
    return tuple((d.filename.relative_to(ROOT).as_posix(), d.message) for d in found)


def _matches(pattern: str, path: str) -> bool:
    """Whether `path` matches a `per-file-ignores` glob."""
    parts = re.split(r'(\*\*/|\*\*|\*|\?)', pattern)
    wildcards = {'**/': '(?:.*/)?', '**': '.*', '*': '[^/]*', '?': '[^/]'}
    return re.fullmatch(''.join(wildcards.get(part, re.escape(part)) for part in parts), path) is not None


def test_the_exemptible_bans_are_in_banned_api():
    assert EXEMPTIBLE <= _banned_api().keys()


def test_every_other_ban_holds_in_exempted_files():
    held = frozenset(_banned_api().keys() - EXEMPTIBLE)
    assert held
    assert _banned_uses(held) == ()


def test_every_exemption_still_exempts_an_asyncio_use():
    uses = _banned_uses(EXEMPTIBLE)
    stale = [pattern for pattern in _exemptions() if not any(_matches(pattern, path) for path, _ in uses)]
    assert _exemptions()
    assert stale == [], 'remove the `TID251` `per-file-ignores` entry: no file it matches uses an `asyncio` primitive'


def test_the_gate_scripts_hold_every_root_ban():
    assert {name: _scripts_banned_api().get(name) for name in _banned_api()} == _banned_api()


def test_the_gate_scripts_add_only_bans_on_starting_a_process_and_on_reading_an_answer():
    assert ANSWER_BAN in _scripts_banned_api()
    assert _process_bans()
    assert all(name.startswith(('subprocess.', 'os.', 'asyncio.create_subprocess_')) for name in _process_bans())


def test_every_gate_script_exemption_still_starts_a_process():
    uses = [path for path, _ in _banned_uses(_process_bans()) if path.startswith(SCRIPTS)]
    stale = [pattern for pattern in _scripts_exemptions() if not any(_matches(pattern, path) for path in uses)]
    assert _scripts_exemptions()
    assert stale == [], (
        'remove the `TID251` entry from `.github/scripts/ruff.toml`: no file it matches starts a process'
    )


def test_a_gate_script_exemption_exempts_no_asyncio_use():
    # Tests may use `asyncio`, as under `tests/`; the seam and the probe may not.
    exempted = [
        path
        for path, _ in _banned_uses(EXEMPTIBLE)
        if any(_matches(pattern, path) for pattern in _scripts_exemptions())
        and not _matches(SCRIPTS + 'test_*.py', path)
        and not any(_matches(pattern, path) for pattern in _exemptions())
    ]
    assert exempted == []


def test_globs_match_like_ruff():
    assert _matches('tests/**/*.py', 'tests/test_x.py')
    assert _matches('tests/**/*.py', 'tests/a/b/test_x.py')
    assert not _matches('tests/*.py', 'tests/a/test_x.py')
    assert not _matches('src/pkg/**/*.py', 'src/pkg_other/x.py')
