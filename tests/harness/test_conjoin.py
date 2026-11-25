"""How two `FileSystem`s or two `Shell`s sharing an id combine: every restriction either states applies.

`tests/harness/test_capability_combine.py` checks the promise over probes (`Conjoins`); these pin each field's
rule, the fields that can't combine, and that a field left at its default takes no part.
"""

from __future__ import annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness._conjoin import is_stated, smallest, state_defaults
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell


def _shell(*shells: Shell[None]) -> Shell[None]:
    merged = Shell.combine(list(shells))
    assert isinstance(merged, Shell)
    return merged


def _file_system(*file_systems: FileSystem[None]) -> FileSystem[None]:
    merged = FileSystem.combine(list(file_systems))
    assert isinstance(merged, FileSystem)
    return merged


def test_both_declare_a_default_id() -> None:
    assert (Shell().id, FileSystem().id) == ('shell', 'file_system')


def test_a_default_is_unstated_until_stated() -> None:
    shell = Shell()
    assert not is_stated(shell.default_timeout) and not is_stated(shell.tools) and not is_stated(shell.denied_commands)
    assert shell.default_timeout == 30.0 and shell.max_output_chars == 50_000
    stated = state_defaults(shell)
    assert all(is_stated(value) for value in (stated.default_timeout, stated.max_output_chars, stated.tools))
    assert (stated.default_timeout, stated.max_output_chars, stated.denied_commands) == (
        shell.default_timeout,
        shell.max_output_chars,
        shell.denied_commands,
    )


def test_shell_fields_left_at_their_default_take_no_part() -> None:
    merged = _shell(Shell(default_timeout=300.0, tools=['shell']), Shell(allowed_commands=['pytest']))
    assert (merged.default_timeout, list(merged.tools), list(merged.allowed_commands)) == (300.0, ['shell'], ['pytest'])
    assert not is_stated(merged.max_output_chars) and not is_stated(merged.max_log_bytes)

    plain = _shell(Shell(), Shell())
    assert not is_stated(plain.tools) and not is_stated(plain.denied_commands) and not is_stated(plain.default_timeout)


def test_shell_lists_and_limits_conjoin() -> None:
    merged = _shell(
        Shell(
            denied_commands=['git'],
            denied_operators=['>'],
            max_output_chars=1_000,
            max_log_bytes=None,
            max_file_bytes=10_000,
            env={'A': '1'},
            denied_env_patterns=['OPENAI_*'],
            secret_env={'TOKEN': 'x'},
        ),
        Shell(
            denied_commands=['rm'],
            denied_operators=['|'],
            max_output_chars=5_000,
            max_file_bytes=2_000,
            persist_cwd=True,
            allow_interactive=True,
            env={'A': '1', 'B': '2'},
            denied_env_patterns=['GEMINI_*'],
        ),
    )
    assert (list(merged.denied_commands), list(merged.denied_operators)) == (['git', 'rm'], ['>', '|'])
    assert (merged.max_output_chars, merged.max_log_bytes, merged.max_file_bytes) == (1_000, None, 2_000)
    assert (merged.persist_cwd, merged.allow_interactive) == (True, False)
    assert (merged.env, merged.secret_env, list(merged.denied_env_patterns)) == (
        {'A': '1', 'B': '2'},
        {'TOKEN': 'x'},
        ['OPENAI_*', 'GEMINI_*'],
    )


def test_a_denied_command_leaves_the_combined_allowlist() -> None:
    merged = _shell(Shell(allowed_commands=['git', 'ls']), Shell(denied_commands=['git']))
    assert (list(merged.allowed_commands), list(merged.denied_commands)) == (['ls'], [])


@pytest.mark.parametrize(
    ('first', 'second', 'match'),
    [
        (Shell(allowed_commands=['git']), Shell(allowed_commands=['ls']), 'no entry is in all of them'),
        (Shell(allowed_commands=['git']), Shell(denied_commands=['git']), 'every command one allows another denies'),
        (Shell(tools=['shell']), Shell(tools=['run_command']), 'no entry is in all of them'),
        (Shell(env={'A': '1'}), Shell(env={'A': '2'}), "give `env` 'A' different values"),
        (Shell(tools=['shell']), Shell(max_file_bytes=1_000), 'does not support `max_file_bytes`'),
    ],
)
def test_shells_that_cannot_conjoin_raise(first: Shell[None], second: Shell[None], match: str) -> None:
    with pytest.raises(UserError, match=match):
        _shell(first, second)


def test_file_system_fields_conjoin() -> None:
    merged = _file_system(
        FileSystem(
            allowed_patterns=['src/**'],
            denied_patterns=['a'],
            read_only_patterns=['r1'],
            max_read_lines=100,
            max_read_chars=None,
            max_image_side=512,
            max_retries=3,
            tools=['read_file', 'write_file'],
        ),
        FileSystem(
            allowed_patterns=['src/**'],
            denied_patterns=['b'],
            read_only_patterns=['r2'],
            max_search_results=10,
            read_only=True,
            content_hashes=False,
            tools=['read_file', 'grep'],
        ),
    )
    assert (list(merged.allowed_patterns), list(merged.denied_patterns), list(merged.read_only_patterns)) == (
        ['src/**'],
        ['a', 'b'],
        ['r1', 'r2'],
    )
    assert (merged.max_read_lines, merged.max_read_chars, merged.max_image_side, merged.max_search_results) == (
        100,
        None,
        512,
        10,
    )
    assert (merged.read_only, merged.content_hashes, merged.max_retries, list(merged.tools)) == (
        True,
        False,
        3,
        ['read_file'],
    )
    plain = _file_system(FileSystem(), FileSystem())
    assert (
        not is_stated(plain.read_only_patterns) and not is_stated(plain.tools) and not is_stated(plain.max_list_results)
    )


@pytest.mark.parametrize(
    ('roots', 'innermost'),
    [
        ((None, '/'), None),
        (('/', '.'), '.'),
        (('../..', '..'), '..'),
        (('/', '/srv/app', '/srv'), '/srv/app'),
        (('C:/', 'C:/repo'), 'C:/repo'),
    ],
)
def test_the_innermost_root_dir_wins(roots: tuple[str | None, ...], innermost: str | None) -> None:
    assert _file_system(*(FileSystem(root_dir=root) for root in roots)).root_dir == innermost


@pytest.mark.parametrize(
    ('first', 'second', 'match'),
    [
        (FileSystem(allowed_patterns=['src/**']), FileSystem(allowed_patterns=['docs/**']), 'holds one list'),
        (FileSystem(root_dir='/srv/a'), FileSystem(root_dir='/srv/b'), 'not inside one another'),
        (FileSystem(root_dir='/srv'), FileSystem(root_dir='..'), 'some are relative and some absolute'),
        (FileSystem(tools=['read_file']), FileSystem(tools=['write_file']), 'no entry is in all of them'),
        (FileSystem(max_retries=1), FileSystem(max_retries=2), 'disagree on `max_retries`'),
    ],
)
def test_file_systems_that_cannot_conjoin_raise(first: FileSystem[None], second: FileSystem[None], match: str) -> None:
    with pytest.raises(UserError, match=match):
        _file_system(first, second)


def test_smallest_with_only_unbounded_stated_is_unbounded() -> None:
    assert smallest([None, None], default=5) is None
