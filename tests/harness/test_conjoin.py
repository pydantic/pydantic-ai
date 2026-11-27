"""How two `FileSystem`s or two `Shell`s sharing an id combine: every restriction either states applies.

`tests/harness/test_capability_combine.py` checks the promise over probes (`Conjoins`); these pin each field's
rule, the fields that can't combine, and that a field left at its default takes no part. Combining happens when
the `Agent` is built, before any model request, so there is nothing for a cassette to record.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import Coder, HarnessDeprecationWarning
from pydantic_ai_harness.file_system import DEFAULT_TOOL_NAMES, FileSystem
from pydantic_ai_harness.shell import Shell


def _leaves(*capabilities: AbstractCapability[None]) -> list[AbstractCapability[None]]:
    """The leaves of an agent built with `capabilities`, which combines those sharing an id."""
    return leaf_capabilities(Agent(TestModel(), deps_type=type(None), capabilities=list(capabilities)).root_capability)


def _shell(*capabilities: AbstractCapability[None]) -> Shell[None]:
    [merged] = [leaf for leaf in _leaves(*capabilities) if isinstance(leaf, Shell)]
    return merged


def _file_system(*capabilities: AbstractCapability[None]) -> FileSystem[None]:
    [merged] = [leaf for leaf in _leaves(*capabilities) if isinstance(leaf, FileSystem)]
    return merged


def _coders(
    capability_type: type[Shell[None]] | type[FileSystem[None]], coder: Coder[None]
) -> AbstractCapability[None]:
    """`Coder`'s own leaf of `capability_type`, every field of which counts as stated."""
    [leaf] = [leaf for leaf in leaf_capabilities(coder) if isinstance(leaf, capability_type)]
    return leaf


def test_both_declare_a_default_id() -> None:
    assert (Shell().id, FileSystem().id) == ('shell', 'file_system')


def test_defaults_keep_their_plain_types() -> None:
    shell, file_system = Shell(), FileSystem()
    assert (shell.denied_commands, shell.tools, file_system.tools) == (
        ['rm', 'rmdir', 'mkfs', 'dd', 'format', 'shutdown', 'reboot', 'halt', 'poweroff', 'init'],
        ('run_command', 'start_command', 'check_command', 'stop_command'),
        DEFAULT_TOOL_NAMES,
    )


def test_a_field_is_stated_when_the_constructor_is_passed_it_even_at_its_default() -> None:
    """`Shell(default_timeout=30.0)` is the class default, and still narrows `Coder`'s 270 seconds."""
    assert _shell(_coders(Shell, Coder()), Shell(default_timeout=30.0)).default_timeout == 30.0
    assert _shell(_coders(Shell, Coder()), Shell()).default_timeout == 270.0
    assert _shell(_coders(Shell, Coder()), Shell(None, ['pytest'])).allowed_commands == ['pytest']


def test_every_construction_path_keeps_what_was_stated() -> None:
    """`copy` and `deepcopy` keep the record; `from_spec` records the spec's arguments; `replace()` states all."""
    assert _shell(_coders(Shell, Coder()), copy.copy(Shell())).default_timeout == 270.0
    assert _shell(_coders(Shell, Coder()), copy.copy(Shell(default_timeout=10.0))).default_timeout == 10.0
    assert _shell(_coders(Shell, Coder()), copy.deepcopy(Shell(default_timeout=10.0))).default_timeout == 10.0
    from_spec = Shell.from_spec(allowed_commands=['pytest'])
    assert isinstance(from_spec, Shell)
    merged = _shell(_coders(Shell, Coder()), from_spec)
    assert (merged.allowed_commands, merged.default_timeout) == (['pytest'], 270.0)
    assert _shell(_coders(Shell, Coder()), replace(Shell(tools=['shell']))).default_timeout == 30.0


def test_a_field_assigned_after_construction_is_stated() -> None:
    """An assigned limit narrows `Coder`'s, which only a field recorded as stated does."""
    shell = Shell()
    shell.default_timeout = 10.0
    assert _shell(_coders(Shell, Coder()), shell).default_timeout == 10.0


def test_a_file_system_left_at_its_default_keeps_its_read_only_protection() -> None:
    """Lists that add up count as each instance applies them, defaults included."""
    merged = _file_system(FileSystem(), FileSystem(read_only_patterns=['docs/**']))
    assert {'**/.git', '**/.env', 'docs/**'} <= set(merged.read_only_patterns)


def test_a_shell_left_at_its_default_keeps_its_denylist() -> None:
    merged = _shell(Shell(), Shell(denied_commands=['git']))
    assert {'rm', 'dd', 'git'} <= set(merged.denied_commands)
    # With an allowlist, a default denylist takes its commands out of it instead.
    assert list(_shell(Shell(), Shell(allowed_commands=['rm', 'ls'])).allowed_commands) == ['ls']


def test_the_deprecated_read_only_name_is_stated_as_the_new_one() -> None:
    """Patterns stated through `protected_patterns` stay on their own root, as `read_only_patterns` do."""
    with pytest.warns(HarnessDeprecationWarning):
        old = FileSystem[None](protected_patterns=['docs/**'], root_dir='..')
    with pytest.raises(UserError, match='relative to their own `root_dir`'):
        _file_system(FileSystem(), old)


def test_shell_fields_left_at_their_default_take_no_part() -> None:
    merged = _shell(Shell(default_timeout=200.0, tools=['shell']), Shell(allowed_commands=['pytest']))
    assert (merged.default_timeout, list(merged.tools), list(merged.allowed_commands)) == (200.0, ['shell'], ['pytest'])
    assert (merged.max_output_chars, merged.max_log_bytes) == (50_000, 100_000_000)

    plain = _shell(Shell(), Shell())
    assert (plain.tools, plain.denied_commands, plain.default_timeout) == (
        Shell().tools,
        Shell().denied_commands,
        30.0,
    )


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
            allow_interactive=True,
            env={'A': '1', 'B': '2'},
            denied_env_patterns=['GEMINI_*'],
        ),
    )
    assert (list(merged.denied_commands), list(merged.denied_operators)) == (['git', 'rm'], ['>', '|'])
    assert (merged.max_output_chars, merged.max_log_bytes, merged.max_file_bytes) == (1_000, None, 2_000)
    assert merged.allow_interactive is False
    assert _shell(Shell(persist_cwd=True), Shell()).persist_cwd is True
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
    assert (list(plain.read_only_patterns), plain.tools, plain.max_list_results) == (
        list(FileSystem().read_only_patterns),
        FileSystem().tools,
        1000,
    )


@pytest.mark.parametrize(
    ('roots', 'innermost'),
    [
        ((None, '/'), None),
        (('/', '.'), '.'),
        (('../..', '..', None), None),
        (('/', '/srv/app', '/srv/app/'), '/srv/app'),
    ],
)
def test_the_innermost_root_dir_wins(roots: tuple[str | None, ...], innermost: str | None) -> None:
    assert _file_system(*(FileSystem[None](root_dir=root) for root in roots)).root_dir == innermost


@pytest.mark.parametrize(
    ('first', 'second', 'match'),
    [
        (FileSystem(allowed_patterns=['src/**']), FileSystem(allowed_patterns=['docs/**']), 'holds one list'),
        # Absolute roots compare only once the workspace resolves them, symlinks included.
        (FileSystem(root_dir='/srv/outer'), FileSystem(root_dir='/srv/outer/parent'), 'only be compared'),
        (FileSystem(), FileSystem(root_dir='/other'), 'only be compared'),
        (FileSystem(root_dir='/srv'), FileSystem(root_dir='..'), 'only be compared'),
        (FileSystem(root_dir='/', denied_patterns=['etc/**']), FileSystem(), 'relative to their own `root_dir`'),
        (FileSystem(tools=['read_file']), FileSystem(tools=['write_file']), 'no entry is in all of them'),
        (FileSystem(max_retries=1), FileSystem(max_retries=2), 'disagree on `max_retries`'),
    ],
)
def test_file_systems_that_cannot_conjoin_raise(first: FileSystem[None], second: FileSystem[None], match: str) -> None:
    with pytest.raises(UserError, match=match):
        _file_system(first, second)


def test_only_unbounded_limits_stated_stay_unbounded() -> None:
    assert _shell(Shell(max_log_bytes=None), Shell(max_log_bytes=None)).max_log_bytes is None
