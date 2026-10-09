"""Guards that keep the cost of individual tests visible and bounded.

Registered by `tests/conftest.py`; see "Test cost" in `tests/AGENTS.md` for the patterns these guards push
towards. The plugin:

- fails a test that launches a Python interpreter unless it is marked `@pytest.mark.subprocess(reason=...)`,
  and fails the session when a `subprocess` marker no longer covers any test that launches one;
- when `PYTEST_TEST_BUDGET_SECONDS` is set, fails a test whose setup and call together take longer than that
  unless it is marked `@pytest.mark.slow(reason=...)`;
"""

from __future__ import annotations as _annotations

import multiprocessing.util
import os
import re
import shlex
import subprocess
import sys
import sysconfig
import time
import traceback
from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import pytest

PLUGIN_NAME = 'pydantic_ai_cost_guards'
BUDGET_ENV_VAR = 'PYTEST_TEST_BUDGET_SECONDS'

_THIS_FILE = __file__
_TESTS_DIR = os.path.dirname(__file__) + os.sep
_PYTHON_PROGRAM = re.compile(r'(python|pypy)(\d+(\.\d+)*)?t?(\.exe)?', re.IGNORECASE)
_SHELLS = frozenset({'sh', 'bash', 'dash', 'zsh'})

_GUIDANCE_DOC = 'the "Test cost" section of tests/AGENTS.md'


# --- Detecting Python interpreter launches ----------------------------------------------------------------------------


def _python_entry_points() -> frozenset[str]:
    """`sys.executable` and this environment's console scripts (`clai`, `pytest`, ...), by name and by path.

    Computed once, before any test runs: the check itself runs inside `Popen`, often on an event loop where the
    blocking-call detector would flag any filesystem access.
    """
    entry_points = {sys.executable, os.path.realpath(sys.executable)}
    scripts_dir = sysconfig.get_path('scripts')
    with suppress(OSError):
        for entry in os.scandir(scripts_dir):
            with suppress(OSError), open(entry.path, 'rb') as script:
                first_line = script.readline(256)
                if first_line.startswith(b'#!') and b'python' in first_line:
                    entry_points.update((entry.name, entry.path, os.path.realpath(entry.path)))
    return frozenset(entry_points)


_python_programs: frozenset[str] = frozenset()


def _is_python_program(program: str) -> bool:
    return program in _python_programs or _PYTHON_PROGRAM.fullmatch(os.path.basename(program)) is not None


def launches_python(argv: Sequence[str]) -> bool:
    """Whether a process started with `argv` is a Python interpreter.

    Looks through `sh -c '<command>'`, `env [VAR=value ...] <command>` and `uv run <command>`.
    """
    if not argv:
        return False
    program, *rest = argv
    name = os.path.basename(program)
    if name in _SHELLS and len(rest) >= 2 and rest[0] == '-c':
        try:
            return launches_python(shlex.split(rest[1]))
        except ValueError:
            return False
    if name == 'env':
        while rest and (rest[0].startswith('-') or '=' in rest[0]):
            rest.pop(0)
        return launches_python(rest)
    if name == 'uv' and 'run' in rest:
        return any(_is_python_program(arg) for arg in rest[rest.index('run') + 1 :] if not arg.startswith('-'))
    return _is_python_program(program)


def _popen_argv(args: Any, executable: Any, shell: bool) -> list[str]:
    if isinstance(args, str | bytes | os.PathLike):
        argv = [os.fsdecode(args)]  # pyright: ignore[reportUnknownArgumentType]
    else:
        argv = [os.fsdecode(arg) for arg in args]
    if shell:
        return ['/bin/sh', '-c', *argv]
    if executable is not None:
        argv[0] = os.fsdecode(executable)
    return argv


# --- Per-test state ---------------------------------------------------------------------------------------------------


@dataclass
class _Spawn:
    argv: list[str]
    location: str | None


@dataclass
class _TestState:
    pending_spawns: list[_Spawn] = field(default_factory=list[_Spawn])
    spawned: bool = False
    shared_setup_seconds: float = 0.0
    shared_fixture_depth: int = 0
    setup_seconds: float = 0.0
    eligible_for_stale_check: bool = True


_STATE_KEY = pytest.StashKey[_TestState]()
_BUDGET_KEY = pytest.StashKey[float]()
_DESELECTED_KEY = pytest.StashKey[set[str]]()
_MARKER_SCOPES_KEY = pytest.StashKey[dict[str, tuple[int, bool]]]()
_AGGREGATE_KEY = pytest.StashKey['_SessionAggregate']()
# The sessions whose terminal this process owns, innermost last; `pytest-xdist` workers own none.
_aggregates: list[_SessionAggregate] = []
# Innermost last: `pytester` runs a session inside a test, and its tests must not report to the outer one.
_running: list[pytest.Item] = []


def _spawn_site() -> str | None:
    """The innermost frame under `tests/` that led to the spawn, so the failure points at the test's own code.

    Walks frames rather than `traceback.extract_stack()`, which reads source lines from disk.
    """
    for frame, lineno in traceback.walk_stack(None):
        filename = frame.f_code.co_filename
        if filename != _THIS_FILE and filename.startswith(_TESTS_DIR):
            return f'tests/{filename[len(_TESTS_DIR) :]}:{lineno}'
    return None


def _record_spawn(argv: list[str]) -> None:
    if not _running or not launches_python(argv):
        return
    item = _running[-1]
    state = item.stash[_STATE_KEY]
    if state.shared_fixture_depth:
        # A fixture shared beyond one test pays for its interpreter once per worker, not once per test: that is the
        # pattern for expensive servers, so it needs no marker.
        return
    marked = item.get_closest_marker('subprocess') is not None
    state.pending_spawns.append(_Spawn(argv, None if marked else _spawn_site()))


_spawn_hooks_installed = False


def _install_spawn_hooks() -> None:
    """Patch the two points every Python process launch in the standard library goes through.

    `subprocess.Popen._execute_child` covers `subprocess.run`/`check_output`/`call`, `asyncio.create_subprocess_*`,
    `anyio.run_process`/`open_process` and Trio; `multiprocessing.util.spawnv_passfds` covers the `spawn` and
    `forkserver` start methods. Both are only reached when a process is started, so the guard costs nothing otherwise.
    """
    global _spawn_hooks_installed
    if _spawn_hooks_installed:
        return
    _spawn_hooks_installed = True
    # Private and untyped in typeshed.
    execute_child: Callable[..., Any] = getattr(subprocess.Popen, '_execute_child')
    spawnv_passfds = multiprocessing.util.spawnv_passfds

    def guarded_execute_child(self: subprocess.Popen[Any], args: Any, executable: Any, *rest: Any) -> Any:
        # `rest[7]` is `shell`, passed positionally on every platform since Python 3.11.
        _record_spawn(_popen_argv(args, executable, bool(rest[7])))
        return execute_child(self, args, executable, *rest)

    def guarded_spawnv_passfds(path: Any, args: Sequence[Any], passfds: Any) -> Any:
        _record_spawn([os.fsdecode(path), *(os.fsdecode(arg) for arg in args[1:])])
        return spawnv_passfds(path, args, passfds)

    setattr(subprocess.Popen, '_execute_child', guarded_execute_child)
    multiprocessing.util.spawnv_passfds = guarded_spawnv_passfds


# --- Configuration and marker validation ------------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    global _python_programs
    _python_programs = _python_programs or _python_entry_points()
    _install_spawn_hooks()
    config.stash[_BUDGET_KEY] = float(os.environ.get(BUDGET_ENV_VAR) or 0)
    if not hasattr(config, 'workerinput'):
        config.stash[_AGGREGATE_KEY] = aggregate = _SessionAggregate()
        _aggregates.append(aggregate)


def pytest_unconfigure(config: pytest.Config) -> None:
    if (aggregate := config.stash.get(_AGGREGATE_KEY, None)) is not None:
        _aggregates.remove(aggregate)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    for name in ('subprocess', 'slow'):
        marker = item.get_closest_marker(name)
        if marker is None:
            continue
        reason = marker.kwargs.get('reason')
        if marker.args or not (isinstance(reason, str) and reason.strip()):
            pytest.fail(
                f'`@pytest.mark.{name}` requires `reason=...`, a non-empty string saying why this test '
                f'needs the exemption; see {_GUIDANCE_DOC}',
                pytrace=False,
            )


# --- Running a test ---------------------------------------------------------------------------------------------------


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    item.stash[_STATE_KEY] = _TestState()
    _running.append(item)
    try:
        return (yield)
    finally:
        _running.pop()


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[Any]) -> Generator[None, object, object]:
    """Track fixtures shared beyond one test, so their setup is not charged to whichever test happens to run first."""
    if fixturedef.scope == 'function' or not _running:
        return (yield)
    state = _running[-1].stash[_STATE_KEY]
    state.shared_fixture_depth += 1
    start = time.perf_counter()
    try:
        return (yield)
    finally:
        state.shared_fixture_depth -= 1
        if not state.shared_fixture_depth:
            state.shared_setup_seconds += time.perf_counter() - start


def _fail_report(report: pytest.TestReport, message: str) -> None:
    if report.passed:
        report.outcome = 'failed'
        report.longrepr = message
    else:
        report.sections.append(('cost guard', message))


def _unmarked_spawn_message(item: pytest.Item, when: str, spawns: list[_Spawn]) -> str:
    lines = [f'`{item.nodeid}` launched a Python interpreter during {when}:']
    for spawn in spawns:
        command = shlex.join(spawn.argv)
        lines.append(f'  {command[:200]}{"..." if len(command) > 200 else ""}')
        if spawn.location:
            lines.append(f'    from {spawn.location}')
    lines.append(
        'Each interpreter imports the project again, which makes the suite materially slower. Call the entry '
        f'point in-process instead; see {_GUIDANCE_DOC}.'
    )
    lines.append(
        'If the test is about the process boundary itself (CLI startup, import isolation, signal handling), '
        "mark it `@pytest.mark.subprocess(reason='...')`."
    )
    return '\n'.join(lines)


def _over_budget_message(item: pytest.Item, spent: float, budget: float, state: _TestState) -> str:
    shared = (
        f', not counting {state.shared_setup_seconds:.2f}s of shared fixture setup'
        if state.shared_setup_seconds
        else ''
    )
    return (
        f'`{item.nodeid}` took {spent:.2f}s in setup and call{shared}, over the {budget:g}s per-test budget '
        f'({BUDGET_ENV_VAR}).\n'
        'Make it cheaper: inject or monkeypatch timeouts and poll intervals instead of waiting them out, prove '
        'complexity bounds with growth ratios at small sizes instead of one huge input, share expensive servers '
        f'through session-scoped fixtures, and call entry points in-process; see {_GUIDANCE_DOC}.\n'
        "If the cost is inherent to what the test proves, mark it `@pytest.mark.slow(reason='...')`."
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, Any]:
    report: pytest.TestReport = yield
    state = item.stash.get(_STATE_KEY, None)
    if state is None:  # pragma: no cover - `pytest_runtest_protocol` always runs first
        return report

    spawns, state.pending_spawns = state.pending_spawns, []
    if spawns:
        state.spawned = True
        if item.get_closest_marker('subprocess') is None:
            _fail_report(report, _unmarked_spawn_message(item, report.when, spawns))

    if report.when == 'setup':
        state.setup_seconds = report.duration
    elif report.when == 'call':
        budget = item.config.stash[_BUDGET_KEY]
        spent = max(state.setup_seconds - state.shared_setup_seconds, 0) + report.duration
        if budget and spent > budget and report.passed and item.get_closest_marker('slow') is None:
            _fail_report(report, _over_budget_message(item, spent, budget, state))

    if not report.passed:
        state.eligible_for_stale_check = False
    if report.when == 'teardown' and (marker_info := _subprocess_marker_info(item, state)) is not None:
        report.python_subprocess_marker = marker_info  # pyright: ignore[reportAttributeAccessIssue]
    return report


# --- Stale `subprocess` markers ---------------------------------------------------------------------------------------
#
# A marker is stale when no test it covers launched Python. Each test reports what it did on its teardown report,
# which `pytest-xdist` forwards to the controller, and the process that owns the terminal decides once every test
# has reported. A marker is only judged when the run covered all of its tests, all of them passed, and at least one
# ran, so `-k`, `-m`, `--deselect`, `pytest-split` shards, node-id selection, skips and failures never produce a
# false "stale" verdict: they leave the marker unjudged in that run.


def _marker_scope(item: pytest.Item) -> tuple[str, str] | None:
    """The key and display location of the node a test's `subprocess` marker was declared on."""
    for node, marker in item.iter_markers_with_node('subprocess'):
        if marker.kwargs.get('conditional'):
            return None
        if node is item and isinstance(item, pytest.Function):
            # Every parametrization of one function shares the decorator, so they are judged together.
            return f'{item.parent.nodeid if item.parent else ""}::{item.originalname}', item.nodeid.split('[', 1)[0]
        return node.nodeid, node.nodeid
    return None


@pytest.hookimpl(trylast=True)
def pytest_deselected(items: Sequence[pytest.Item]) -> None:
    for item in items:
        if (scope := _marker_scope(item)) is not None:
            item.config.stash.setdefault(_DESELECTED_KEY, set()).add(scope[0])


def pytest_collection_finish(session: pytest.Session) -> None:
    config = session.config
    deselected = config.stash.get(_DESELECTED_KEY, set[str]())
    # A `path::name` argument selects part of a file, so a marker declared on that file's module or one of its
    # classes may cover tests that were never collected.
    narrowed_files = {
        (config.invocation_params.dir / arg.split('::', 1)[0]).resolve() for arg in config.args if '::' in arg
    }
    scoped = [(item, scope[0]) for item in session.items if (scope := _marker_scope(item)) is not None]
    sizes: dict[str, int] = {}
    incomplete = set(deselected)
    for item, key in scoped:
        sizes[key] = sizes.get(key, 0) + 1
        if narrowed_files and item.path.resolve() in narrowed_files:
            incomplete.add(key)
    config.stash[_MARKER_SCOPES_KEY] = {key: (size, key not in incomplete) for key, size in sizes.items()}


def _subprocess_marker_info(item: pytest.Item, state: _TestState) -> dict[str, Any] | None:
    if (scope := _marker_scope(item)) is None:
        return None
    key, location = scope
    size, complete = item.config.stash.get(_MARKER_SCOPES_KEY, dict[str, tuple[int, bool]]()).get(key, (0, False))
    return {
        'key': key,
        'location': location,
        'size': size,
        'complete': complete,
        'eligible': state.eligible_for_stale_check,
        'spawned': state.spawned,
    }


@dataclass
class _MarkerAggregate:
    location: str
    size: int
    complete: bool = True
    reported: set[str] = field(default_factory=set[str])
    eligible: bool = True
    spawned: bool = False


@dataclass
class _SessionAggregate:
    markers: dict[str, _MarkerAggregate] = field(default_factory=dict[str, _MarkerAggregate])
    stale: list[str] = field(default_factory=list[str])

    def stale_markers(self) -> list[str]:
        return sorted(
            marker.location
            for marker in self.markers.values()
            if marker.complete and marker.eligible and len(marker.reported) == marker.size and not marker.spawned
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if not _aggregates:
        return
    aggregate = _aggregates[-1]
    info: dict[str, Any] | None = getattr(report, 'python_subprocess_marker', None)
    if info is None:
        return
    marker = aggregate.markers.setdefault(info['key'], _MarkerAggregate(info['location'], info['size']))
    marker.complete &= info['complete']
    marker.eligible &= info['eligible']
    marker.reported.add(report.nodeid)
    marker.spawned |= info['spawned']


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    aggregate = session.config.stash.get(_AGGREGATE_KEY, None)
    if aggregate is None or session.shouldstop or session.shouldfail:
        return
    if exitstatus not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        return
    aggregate.stale = aggregate.stale_markers()
    if aggregate.stale:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    aggregate = config.stash.get(_AGGREGATE_KEY, None) or _SessionAggregate()
    if aggregate.stale:
        terminalreporter.section('stale subprocess markers', red=True)
        for location in aggregate.stale:
            terminalreporter.write_line(
                f'{location}: no test covered by its `@pytest.mark.subprocess` launched a Python interpreter; '
                'remove the stale marker'
            )
