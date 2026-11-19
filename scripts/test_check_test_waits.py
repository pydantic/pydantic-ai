from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_test_waits
from check_test_waits import GUIDE

_REPO = Path(__file__).parents[1]

_DEBT = """\
import anyio


async def test_debt():
    await anyio.sleep(0.1)
    with anyio.fail_after(5):
        pass
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, 'tests/test_debt.py', _DEBT)
    _write(tmp_path, 'outside/test_loose.py', 'import time\ntime.sleep(1)\n')
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
    code = check_test_waits.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def _flagged(out: str) -> list[str]:
    """The `path:line: source` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_literal_wait_fails_and_says_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(
        repo, capsys, 'tests/test_new.py', 'import asyncio\n\n\nasync def t():\n    await asyncio.sleep(1)\n'
    )

    assert code == 1
    assert 'tests/test_new.py: literal-wait 0 -> 1' in out
    assert _flagged(out) == ['tests/test_new.py:5: await asyncio.sleep(1)']
    assert '`READINESS_WAIT_TIMEOUT` from `tests/_timeouts.py`' in out
    assert '`# real-wait: <reason>`' in out
    assert f'See {GUIDE}.' in out


@pytest.mark.parametrize(
    'call',
    [
        'time.sleep(0.5)',
        'anyio.sleep(delay=2)',
        'anyio.fail_after(10)',
        'anyio.move_on_after(0.2)',
        'asyncio.timeout(3)',
        'asyncio.wait_for(f, 5)',
        'asyncio.wait_for(f, timeout=5)',
        'asyncio.wait(tasks, timeout=1)',
        'event.wait(2)',
        'event.wait(timeout=2)',
        'thread.join(10)',
        'running.result(timeout=10)',
        'process.wait(timeout=SHORT)',
        'trio.sleep(1)',
        'sleep(1)',
        'aio.sleep(1)',
    ],
)
def test_each_waiter_with_a_written_delay_counts(repo: Path, capsys: pytest.CaptureFixture[str], call: str):
    source = (
        'import asyncio\nimport asyncio as aio\nimport time\n\nimport anyio\nimport trio\nfrom anyio import sleep\n\n'
        f'SHORT = 1\n\n\ndef t():\n    {call}\n'
    )

    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 1
    assert 'literal-wait 0 -> 1' in out


@pytest.mark.parametrize(
    'call',
    [
        'anyio.sleep(0)',
        'anyio.fail_after(READINESS_WAIT_TIMEOUT)',
        'anyio.fail_after(timeout)',
        'running.result()',
        'running.result(5)',
        'thread.join()',
        'clock.sleep(5)',
        "', '.join(parts)",
        'asyncio.wait_for(f, None)',
    ],
)
def test_a_wait_without_a_written_delay_does_not_count(repo: Path, capsys: pytest.CaptureFixture[str], call: str):
    source = (
        'import asyncio\n\nimport anyio\n\nfrom tests._timeouts import READINESS_WAIT_TIMEOUT\n\n\n'
        f'def t(timeout):\n    {call}\n'
    )

    assert _staged(repo, capsys, 'tests/test_new.py', source) == (0, '')


def test_a_loop_that_sleeps_counts_once_with_its_sleeps(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'import anyio\n\n\nasync def t(ready):\n    while not ready():\n        await anyio.sleep(0.01)\n'

    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 1
    assert 'tests/test_new.py: sleep-loop 0 -> 1' in out
    assert 'literal-wait' not in out.split('\n\n')[0]
    assert _flagged(out) == ['tests/test_new.py:5: while not ready():']


def test_a_sleep_in_a_nested_loop_counts_only_the_inner_loop(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'import time\n\n\ndef t(items):\n    for item in items:\n        while item:\n            time.sleep(0)\n'

    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 1
    assert _flagged(out) == ['tests/test_new.py:6: while item:']


def test_debt_that_holds_or_shrinks_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    shrunk = _DEBT.replace('    await anyio.sleep(0.1)\n', '')

    assert _staged(repo, capsys, 'tests/test_debt.py', shrunk.replace('fail_after(5)', 'fail_after(6)')) == (0, '')


def test_debt_that_grows_lists_only_what_the_change_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(repo, capsys, 'tests/test_debt.py', _DEBT + '    await anyio.sleep(0.2)\n')

    assert code == 1
    assert 'tests/test_debt.py: literal-wait 2 -> 3' in out
    assert _flagged(out) == ['tests/test_debt.py:8: await anyio.sleep(0.2)']


def test_moving_a_file_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(repo, 'mv', 'tests/test_debt.py', 'tests/test_moved.py')

    assert _check(capsys, '--staged') == (0, '')


def test_moving_a_wait_between_files_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/test_debt.py', _DEBT.replace('    await anyio.sleep(0.1)\n', ''))

    assert _staged(
        repo, capsys, 'tests/test_other.py', 'import anyio\n\n\nasync def t():\n    await anyio.sleep(0.1)\n'
    ) == (
        0,
        '',
    )


def test_removing_a_probe_elsewhere_does_not_excuse_a_new_one(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/test_old.py', _RETURNED)
    _commit(repo)
    # Emptied, not deleted, so git does not pair the two files as a rename.
    _write(repo, 'tests/test_old.py', '')

    code, out = _staged(repo, capsys, 'tests/test_new.py', _RETURNED.replace('free_port', 'other_port'))

    assert code == 1
    assert 'tests/test_new.py: probe-port 0 -> 1' in out


def test_a_loop_of_checkpoints_counts_as_a_sleep_loop(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'import anyio\n\n\nasync def t():\n    for _ in range(3):\n        await anyio.sleep(0)\n'

    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 1
    assert 'tests/test_new.py: sleep-loop 0 -> 1' in out


def test_files_outside_tests_and_files_that_do_not_parse_are_skipped(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'outside/test_loose.py', 'import time\ntime.sleep(1)\ntime.sleep(2)\n')

    assert _staged(repo, capsys, 'tests/test_broken.py', 'import time\ntime.sleep(1)\ndef (:\n') == (0, '')


def test_staged_mode_reads_the_index_not_the_working_tree(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/test_new.py', 'import time\n\ntime.sleep(1)\n')

    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD')[0] == 1


@pytest.mark.parametrize(
    'source',
    [
        'import time\n\ntime.sleep(1)  # real-wait: measures the wall clock\n',
        'import time\n\n# real-wait: measures the wall clock\ntime.sleep(1)\n',
        'import time\n\nwhile True:  # real-wait: polls a real process\n    time.sleep(1)\n',
    ],
)
def test_a_real_wait_with_a_reason_passes_and_is_reported(repo: Path, capsys: pytest.CaptureFixture[str], source: str):
    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 0
    assert re.fullmatch(
        rf'tests/test_new\.py:\d: real wait: (measures the wall clock|polls a real process), see {GUIDE}\n', out
    )


def test_a_real_wait_without_a_reason_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(repo, capsys, 'tests/test_new.py', 'import time\n\ntime.sleep(1)  # real-wait:\n')

    assert code == 1
    assert out == f'tests/test_new.py:3: give the real wait a reason, see {GUIDE}\n'


def test_a_real_wait_that_was_already_there_is_not_reported(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/test_new.py', 'import time\n\ntime.sleep(1)  # real-wait: measures the wall clock\n')
    _commit(repo)

    assert _staged(repo, capsys, 'tests/test_other.py', '') == (0, '')


def test_a_real_wait_does_not_excuse_a_readiness_constant(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(repo, capsys, 'tests/test_new.py', 'READINESS_WAIT_TIMEOUT = 5  # real-wait: fast\n')

    assert code == 1
    assert 'readiness-constant 0 -> 1' in out


@pytest.mark.parametrize('name', ['READINESS_WAIT_TIMEOUT', '_READINESS_WAIT_TIMEOUT'])
def test_a_private_readiness_constant_fails(repo: Path, capsys: pytest.CaptureFixture[str], name: str):
    code, out = _staged(repo, capsys, 'tests/test_new.py', f'{name}: float = 5.0\n')

    assert code == 1
    assert 'tests/test_new.py: readiness-constant 0 -> 1' in out
    assert 'import `READINESS_WAIT_TIMEOUT` from `tests/_timeouts.py`' in out


def test_the_shared_readiness_constant_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'tests/_timeouts.py', 'READINESS_WAIT_TIMEOUT = 300\n') == (0, '')


_RETURNED = """\
import socket


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
"""

_READ_AFTER = """\
from socket import socket


def test_server(start):
    with socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    start(port=port)
"""

_READ_BEFORE_REBIND = """\
import socket


def test_server(start, cond):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    if cond:
        port = 8123
    start(port)
    port = 8124
"""

_READ_IN_A_REBIND = """\
import socket


def test_server(start):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    port = max(
        port,
        1,
    )
    start(port)
"""


_RETURNED_BY_NAME = """\
import socket


def free_url():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        return f'http://127.0.0.1:{port}'
"""

_RETURNED_AFTER = """\
import socket


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return port
"""

_YIELDED_AFTER = """\
import socket


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    yield port
"""

_ANNOTATED_READ_AFTER = """\
import socket


def test_server(start):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port: int = probe.getsockname()[1]
    start(port=port)
"""

_UNPACKED_READ_AFTER = """\
import socket


def test_server(start):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        _, port = probe.getsockname()
    start(port=port)
"""


@pytest.mark.parametrize(
    'source',
    [
        _RETURNED,
        _RETURNED_BY_NAME,
        _RETURNED_AFTER,
        _YIELDED_AFTER,
        _READ_AFTER,
        _ANNOTATED_READ_AFTER,
        _UNPACKED_READ_AFTER,
        _READ_BEFORE_REBIND,
        _READ_IN_A_REBIND,
    ],
)
def test_a_port_used_after_its_socket_closes_fails(repo: Path, capsys: pytest.CaptureFixture[str], source: str):
    code, out = _staged(repo, capsys, 'tests/test_new.py', source)

    assert code == 1
    assert 'tests/test_new.py: probe-port 0 -> 1' in out
    assert _flagged(out) == ['tests/test_new.py:6: ' + source.splitlines()[5].strip()]
    assert '`reserve_loopback_port()` in `tests/_ports.py`' in out


_KEPT_OPEN = """\
import socket


def test_taken(serve):
    with socket.socket() as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        port = taken.getsockname()[1]
        serve(port)
    assert f'port {port}' in 'text'


def fixture():
    with socket.socket() as sock:
        sock.bind(('localhost', 0))
        yield sock.getsockname()[1]


def handed_over(serve):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    serve(sock)


def served_before_return(serve, done):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        if done:
            return
        return serve(sock)


def rebound(run):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    port = 8123
    run(port)


def rebound_before_return():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    port = 8123
    return port


def rebound_with_annotation(run):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    port, other = 8123, 8124
    run(port)
    port: int = 8125
    run(port)


def fixed():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 8080))
        return sock.getsockname()[1]
"""


def test_a_port_whose_socket_stays_open_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'tests/test_new.py', _KEPT_OPEN) == (0, '')


def test_the_shared_port_reservation_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'tests/_ports.py', _RETURNED) == (0, '')


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as `tests/test_debt.py` on another branch, then start merging it into this one."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, 'tests/test_debt.py', theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'tests/test_mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + '    await anyio.sleep(0.2)\n')

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + '    await anyio.sleep(0.2)\n')

    code, out = _staged(
        repo, capsys, 'tests/test_debt.py', _DEBT + '    await anyio.sleep(0.2)\n    await anyio.sleep(0.3)\n'
    )

    assert code == 1
    assert 'tests/test_debt.py: literal-wait 3 -> 4' in out


def test_github_format_annotates_each_added_finding(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/test_new.py', 'import time\n\ntime.sleep(1)\n')

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith('::error file=tests/test_new.py,line=3,title=Test waits%3A literal-wait 0 -> 1::')


def test_the_guide_the_report_links_has_the_section():
    path, anchor = GUIDE.split('#')
    headings = re.findall(r'^#+ (.+)$', (_REPO / path).read_text(encoding='utf-8'), re.MULTILINE)
    assert anchor in {re.sub(r'[^a-z0-9 -]', '', h.lower()).replace(' ', '-') for h in headings}


def test_the_exempt_modules_and_roots_exist():
    for path in [*check_test_waits.ROOTS, 'tests/_timeouts.py', 'tests/_ports.py']:
        assert (_REPO / path).exists(), path
