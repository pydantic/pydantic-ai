"""Workflow-side core code makes no new nondeterministic calls.

A durable engine replays workflow code to rebuild a run's state, so a call that answers differently on
replay (a clock, a random id, a sleep the engine does not journal) makes the replayed run diverge from
the one that was recorded. Temporal's sandbox does not catch these in Pydantic AI's own code:
`PydanticAIPlugin` passes `pydantic_ai` through the sandbox, so core modules run against the real
`time`, `datetime`, `random` and `uuid`. This test checks the direct calls: it matches each call in
the workflow-side modules against `_DENYLIST` by the dotted name its imports resolve to. It reads
source only, so it is not a VCR test.

**What it does not see**, and a workflow-level test has to cover:

- a call made through another name or object;
- a callee `_DENYLIST` does not name, such as file-system I/O through `os`, `shutil`, `pathlib` or
  `io`, which no workflow-side module makes today;
- an environment read that is not a call, such as `os.environ['NAME']` or `'NAME' in os.environ`;
- a clock read or random id drawn by a dataclass default factory (`default_factory=now_utc`,
  `default_factory=lambda: str(uuid7())`), or by a helper in a module outside the set;
- a mutation of module or class state, through `global` or an attribute assignment such as
  `module.attr = ...`;
- a branch on state that differs between the recorded run and the replay.

TODO: https://github.com/pydantic/pydantic-ai/issues/11168

**The workflow-side module set.** Under Temporal, everything `TemporalAgent.run` and
`TemporalDurability.wrap_run` execute outside an activity is workflow code. Running the Temporal
durability, agent, toolset and tool-kind suites with a `sys.monitoring` hook that records each
function first entered on Temporal's workflow event loop shows which core modules outside
`durable_exec/` execute there. Most are data types and helpers (`messages`, `usage`, `_utils`, the
model profiles), whose calls are only workflow-side when an orchestrating module makes them. The set
below is the modules that own the run's control flow -- where a nondeterministic call decides what
the workflow does next -- each of which the trace shows running on the workflow loop:

- `_agent_graph.py`: the graph nodes, prompt assembly, request preparation and response handling.
- `agent/__init__.py`: `Agent.iter`, `_prepare_run` and `_PreparedAgentRun.open`.
- `run.py`: `AgentRun`, which drives the graph.
- `tool_manager.py` and `_tool_execution.py`: tool discovery, validation and dispatch.
- `_output.py`: output processing and validation.
- `_run_context.py`: the run context and its event emission.
- `capabilities/combined.py`: hook dispatch across the run's capabilities.
- `durable_exec/*.py` (the top level) and `durable_exec/temporal/` (recursively): every module, so a
  new one is checked without being listed. Their activity-side calls are accepted in `_ACCEPTED`.

Expand the core modules only with the same evidence. The set was derived under Temporal. DBOS and
Prefect run the same core modules in their workflow and flow bodies, but no trace has confirmed the
set for them, and their engine modules and the Harness engines are not checked. TODO:
https://github.com/pydantic/pydantic-ai/issues/11176

**What is denied** is in `_DENYLIST`, each with why replay breaks it. `asyncio.to_thread` and
`loop.run_in_executor` are left out on purpose: they already raise inside a Temporal workflow, whose
event loop implements no executor, and `_utils.run_in_executor` runs inline under
`_utils.disable_threads` instead. `anyio.to_thread` is denied: it starts its own worker thread, which
runs the function and then cannot hand the result back to that loop, so the run hangs.

A key under `pydantic_ai.` names a core function, so the test also checks that it still exists.

**Existing call sites** are in `_ACCEPTED`, keyed by module, enclosing qualname and callee so the
entries survive unrelated edits, with the number of such calls each key accepts. A new call is
accepted only with a pragma comment naming the reason, `# workflow-nondeterminism-ok: <reason>`, on
any line the call spans. A pragma explains one call, so give each call its own line. Prefer a
replay-safe alternative (an id or clock the durability capability supplies; `_agent_graph_sleep`
for a delay) when one exists.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import io
import re
import textwrap
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from inline_snapshot import snapshot

REPO_ROOT = Path(__file__).parents[2]
CORE = 'pydantic_ai_slim/pydantic_ai'

_CLOCK = 'reads a clock that has moved by the time the run replays'
_RANDOM = 'draws randomness that a replay does not reproduce'
_SLEEP = 'sleeps on the event loop, which only Temporal journals as a timer, so other engines do not replay the delay; await `_agent_graph_sleep`, which an engine routes through `set_agent_graph_sleep`'
_THREAD = 'finishes on another thread in an order that a replay does not reproduce'
_COMPLETION_ORDER = 'hands back finished tasks in an order a replay need not reproduce, so Temporal supplies `workflow.wait` and `workflow.as_completed` instead'
_IO = 'performs I/O whose result can differ on replay; do it in an activity'
_ENV = "reads the worker process's environment, which can differ on the worker that replays"

# Callee -> why replay breaks it. A key ending in `.` denies everything under that module.
_DENYLIST: dict[str, str] = {
    'time.time': _CLOCK,
    'time.time_ns': _CLOCK,
    'time.monotonic': _CLOCK,
    'time.monotonic_ns': _CLOCK,
    'time.perf_counter': _CLOCK,
    'time.perf_counter_ns': _CLOCK,
    'time.process_time': _CLOCK,
    'time.clock_gettime': _CLOCK,
    'time.clock_gettime_ns': _CLOCK,
    'time.localtime': _CLOCK,
    'time.gmtime': _CLOCK,
    'time.ctime': _CLOCK,
    'time.process_time_ns': _CLOCK,
    'time.thread_time': _CLOCK,
    'time.thread_time_ns': _CLOCK,
    'datetime.datetime.now': _CLOCK,
    'datetime.datetime.utcnow': _CLOCK,
    'datetime.datetime.today': _CLOCK,
    'datetime.date.today': _CLOCK,
    'pydantic_ai._utils.now_utc': _CLOCK,
    'anyio.current_time': 'reads the event-loop clock, which Temporal replays and other engines do not',
    'random.': _RANDOM,
    'secrets.': _RANDOM,
    'os.urandom': _RANDOM,
    'uuid.uuid1': _RANDOM,
    'uuid.uuid4': _RANDOM,
    'uuid.uuid6': _RANDOM,
    'uuid.uuid7': _RANDOM,
    'uuid.uuid8': _RANDOM,
    'pydantic_ai._uuid.uuid7': _RANDOM,
    'pydantic_ai._utils.generate_tool_call_id': _RANDOM,
    'asyncio.sleep': _SLEEP,
    'anyio.sleep': _SLEEP,
    'anyio.sleep_until': _SLEEP,
    'time.sleep': _SLEEP,
    'threading.Thread': _THREAD,
    'threading.Timer': _THREAD,
    'anyio.to_thread.': (
        'runs the function on a worker thread that cannot hand the result back to the workflow loop, so the run hangs'
    ),
    'concurrent.futures.': _THREAD,
    'asyncio.wait': _COMPLETION_ORDER,
    'asyncio.as_completed': _COMPLETION_ORDER,
    'multiprocessing.': _THREAD,
    'open': _IO,
    'httpx.Client': _IO,
    'httpx.AsyncClient': _IO,
    'httpx.request': _IO,
    'httpx.stream': _IO,
    'httpx.get': _IO,
    'httpx.post': _IO,
    'httpx.put': _IO,
    'httpx.delete': _IO,
    'httpx.head': _IO,
    'httpx.patch': _IO,
    'httpx.options': _IO,
    'httpx2.Client': _IO,
    'httpx2.AsyncClient': _IO,
    'httpx2.request': _IO,
    'httpx2.stream': _IO,
    'httpx2.get': _IO,
    'httpx2.post': _IO,
    'httpx2.put': _IO,
    'httpx2.delete': _IO,
    'httpx2.head': _IO,
    'httpx2.patch': _IO,
    'httpx2.options': _IO,
    'httpx2.query': _IO,
    'httpx2.websocket': _IO,
    'pydantic_ai._http.create_async_httpx2_client': _IO,
    'subprocess.': _IO,
    'os.system': _IO,
    'os.popen': _IO,
    'http.client.': _IO,
    'os.getenv': _ENV,
    'os.environ.': _ENV,
    'os.getpid': _ENV,
    'platform.': _ENV,
    'getpass.': _ENV,
    'socket.': _IO,
    'urllib.request.': _IO,
}

_PRAGMA = re.compile(r'#\s*workflow-nondeterminism-ok:(?P<reason>.*)$')

_RUN_ID = 'fallback id when the run has none; `TemporalDurability` supplies one from `workflow.uuid4()`'
_WAIT_ORDER = (
    'whether replay can reorder the finished tasks is tracked in https://github.com/pydantic/pydantic-ai/issues/11182'
)
_TIMESTAMP = 'stamps the `timestamp` of a message the run appends; the run reads it back only to match the message against a copy of itself (`_is_same_request`) or to carry it into a merged request'

# (module, enclosing qualname, callee) -> how many such calls are accepted, and why.
_ACCEPTED: dict[tuple[str, str, str], tuple[int, str]] = {
    (f'{CORE}/_agent_graph.py', 'resolve_conversation_id', 'pydantic_ai._uuid.uuid7'): (1, _RUN_ID),
    (f'{CORE}/_agent_graph.py', 'resolve_run_id', 'pydantic_ai._uuid.uuid7'): (1, _RUN_ID),
    (f'{CORE}/_agent_graph.py', 'GraphAgentState', 'pydantic_ai._uuid.uuid7'): (
        2,
        'dataclass defaults for a state built without resolved ids; the run passes the resolved ones',
    ),
    (f'{CORE}/_agent_graph.py', '_agent_graph_sleep', 'anyio.sleep'): (
        1,
        'the default of the `set_agent_graph_sleep` indirection, used only when no engine registered a sleep',
    ),
    (f'{CORE}/_agent_graph.py', 'ModelRequestNode._prepare_request', 'pydantic_ai._utils.now_utc'): (1, _TIMESTAMP),
    (f'{CORE}/_agent_graph.py', 'ModelRequestNode._stream._streaming_handler', 'time.perf_counter'): (
        1,
        'time-to-first-chunk telemetry; nothing in the run reads it',
    ),
    (f'{CORE}/_agent_graph.py', 'CallToolsNode._handle_tool_calls', 'pydantic_ai._utils.now_utc'): (1, _TIMESTAMP),
    (f'{CORE}/_agent_graph.py', 'CallToolsNode._handle_final_result', 'pydantic_ai._utils.now_utc'): (1, _TIMESTAMP),
    (f'{CORE}/agent/__init__.py', '_synthetic_capability_id', 'uuid.uuid4'): (
        1,
        'a run-local key for an unnamed capability; every path that persists a capability key requires an explicit `id`',
    ),
    (f'{CORE}/durable_exec/temporal/_toolset.py', 'heartbeating.beat', 'anyio.sleep'): (
        1,
        'the heartbeat loop of a running activity, which is not workflow code',
    ),
    (f'{CORE}/_agent_graph.py', 'ModelRequestNode._stream', 'asyncio.wait'): (1, _WAIT_ORDER),
    (f'{CORE}/_tool_execution.py', '_iter_completed_or_buffered', 'asyncio.wait'): (2, _WAIT_ORDER),
    (f'{CORE}/agent/__init__.py', '_run_lifecycle_hooks', 'asyncio.wait'): (1, _WAIT_ORDER),
    (f'{CORE}/durable_exec/temporal/_activity_execution.py', '_await_with_single_cancel', 'asyncio.wait'): (
        1,
        'waits on a single activity or child-workflow handle, so there is no completion order to reproduce',
    ),
}


@dataclass(frozen=True)
class DeniedCall:
    """A denylisted call in one module."""

    qualname: str
    callee: str
    denial: str
    """Why the callee is denied."""
    pragma: str | None
    """The reason the call's pragma gives, `''` if it gives none, or `None` without a pragma."""


def _denylist_key(callee: str) -> str | None:
    """The `_DENYLIST` key that denies `callee`: the callee itself, or a key ending in `.` it falls under."""
    if callee in _DENYLIST:
        return callee
    return next((key for key in _DENYLIST if key.endswith('.') and callee.startswith(key)), None)


class _CallFinder(ast.NodeVisitor):
    def __init__(self, source: str, module: str, is_package: bool) -> None:
        self.comments = {
            token.start[0]: token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }
        self.scopes: list[str] = []
        self.calls: list[DeniedCall] = []
        self.claimed_pragmas: set[int] = set()
        self.package = module if is_package else module.rpartition('.')[0]
        tree = ast.parse(source)
        self.aliases = [self.scope_aliases(tree.body)]
        self.visit(tree)

    def scope_aliases(self, body: list[ast.stmt]) -> dict[str, str]:
        """The names the imports of one scope bind, without those of the functions and classes nested in it."""
        aliases: dict[str, str] = {}
        pending: list[ast.AST] = list(body)
        while pending:
            match pending.pop(0):
                case ast.Import(names=names):
                    for alias in names:
                        bound = alias.asname or alias.name.split('.')[0]
                        aliases[bound] = alias.name if alias.asname else bound
                case ast.ImportFrom(module=imported, level=0, names=names):
                    for alias in names:
                        aliases[alias.asname or alias.name] = f'{imported}.{alias.name}'
                case ast.ImportFrom(module=imported, level=level, names=names):
                    base = importlib.util.resolve_name('.' * level + (imported or ''), self.package)
                    for alias in names:
                        aliases[alias.asname or alias.name] = f'{base}.{alias.name}'
                case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                    pass
                case node:
                    pending.extend(ast.iter_child_nodes(node))
        return aliases

    def alias(self, name: str) -> str | None:
        return next((scope[name] for scope in reversed(self.aliases) if name in scope), None)

    def dotted(self, expr: ast.expr) -> str | None:
        match expr:
            case ast.Name(id='open'):
                return self.alias('open') or 'open'
            case ast.Name(id=name):
                return self.alias(name)
            case ast.Attribute(value=value, attr=attr):
                base = self.dotted(value)
                return f'{base}.{attr}' if base is not None else None
            case _:
                return None

    def visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scopes.append(node.name)
        self.aliases.append(self.scope_aliases(node.body))
        self.generic_visit(node)
        self.aliases.pop()
        self.scopes.pop()

    visit_FunctionDef = visit_scope
    visit_AsyncFunctionDef = visit_scope
    visit_ClassDef = visit_scope

    def visit_Call(self, node: ast.Call) -> None:
        callee = self.dotted(node.func)
        if callee is not None and (key := _denylist_key(callee)) is not None:
            reason = None
            for lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                if lineno not in self.claimed_pragmas and (pragma := _PRAGMA.search(self.comments.get(lineno, ''))):
                    self.claimed_pragmas.add(lineno)
                    reason = pragma['reason'].strip()
                    break
            self.calls.append(DeniedCall('.'.join(self.scopes) or '<module>', callee, _DENYLIST[key], reason))
        self.generic_visit(node)


def nondeterministic_calls(source: str, module: str, *, is_package: bool = False) -> list[DeniedCall]:
    """The denylisted calls `source`, the code of `module`, makes, in source order."""
    return _CallFinder(source, module, is_package).calls


def check_calls(relative: str, calls: list[DeniedCall]) -> tuple[list[str], Counter[tuple[str, str, str]]]:
    """The problems with the calls in the module at `relative`, and how many calls without a pragma each key has.

    A call is explained by a pragma with a reason, or by `_ACCEPTED` up to the number of calls that entry accepts.
    """
    problems = [
        f'{relative}::{call.qualname} has a `workflow-nondeterminism-ok` pragma with no reason'
        for call in calls
        if call.pragma == ''
    ]
    without_pragma = Counter((relative, call.qualname, call.callee) for call in calls if call.pragma is None)
    denials = {(relative, call.qualname, call.callee): call.denial for call in calls}
    for key, count in without_pragma.items():
        accepted = _ACCEPTED.get(key, (0, ''))[0]
        if count > accepted:
            _, qualname, callee = key
            if accepted:
                problems.append(
                    f'{relative}::{qualname} calls {callee!r} {count} times and `_ACCEPTED` accepts {accepted}; '
                    'give the new call a pragma with a reason'
                )
            else:
                problems.append(f'{relative}::{qualname} calls {callee!r}, which {denials[key]}')
    return problems, without_pragma


def test_workflow_side_code_makes_no_new_nondeterministic_calls():
    engine_globs = ('durable_exec/*.py', 'durable_exec/temporal/**/*.py')
    engine_modules = {pattern: sorted((REPO_ROOT / CORE).glob(pattern)) for pattern in engine_globs}
    assert [pattern for pattern, paths in engine_modules.items() if not paths] == []
    modules: list[str] = [
        f'{CORE}/_agent_graph.py',
        f'{CORE}/_output.py',
        f'{CORE}/_run_context.py',
        f'{CORE}/_tool_execution.py',
        f'{CORE}/agent/__init__.py',
        f'{CORE}/capabilities/combined.py',
        f'{CORE}/run.py',
        f'{CORE}/tool_manager.py',
        *sorted(path.relative_to(REPO_ROOT).as_posix() for paths in engine_modules.values() for path in paths),
    ]
    found: Counter[tuple[str, str, str]] = Counter()
    problems: list[str] = []
    for relative in modules:
        path = REPO_ROOT / relative
        parts = path.relative_to(REPO_ROOT / 'pydantic_ai_slim').with_suffix('').parts
        is_package = parts[-1] == '__init__'
        module = '.'.join(parts[:-1] if is_package else parts)
        calls = nondeterministic_calls(path.read_text(encoding='utf-8'), module, is_package=is_package)
        module_problems, counts = check_calls(relative, calls)
        problems.extend(module_problems)
        found.update(counts)
    assert problems == [], "nondeterministic calls in workflow-side code; see this module's docstring:\n" + '\n'.join(
        problems
    )
    stale = sorted(key for key, (accepted, _) in _ACCEPTED.items() if found[key] < accepted)
    assert stale == [], f'`_ACCEPTED` entries that accept more calls than remain, lower or remove them: {stale}'


def test_nondeterministic_calls_resolves_imports_and_pragmas():
    """Every import form resolves to the callee it names, and a pragma on any line of a call explains that call only."""
    source = """
    import asyncio
    import time
    import uuid as uuid_module
    import concurrent.futures
    from datetime import datetime
    from .. import _uuid as ids, _utils
    from .._utils import now_utc as stamp

    time.perf_counter()


    class Node:
        def run(self, handlers):
            datetime.now()  # workflow-nondeterminism-ok: telemetry only
            stamp()  # workflow-nondeterminism-ok:
            _utils.now_utc()
            concurrent.futures.ThreadPoolExecutor()
            time.time() + time.monotonic()  # workflow-nondeterminism-ok: telemetry only
            asyncio.sleep(
                1,
            )  # workflow-nondeterminism-ok: replay-safe delay
            note = time.time(), '# workflow-nondeterminism-ok: a string, not a comment'

            async def nested():
                uuid_module.uuid4()
                ids.uuid7()

            open('file')
            self.clock.time()
            handlers[0]()
            len(handlers)


    def local_imports():
        import datetime

        datetime.date.today()
    """
    calls = nondeterministic_calls(textwrap.dedent(source), 'pydantic_ai.agent', is_package=True)

    assert [(call.qualname, call.callee, call.pragma) for call in calls] == snapshot(
        [
            ('<module>', 'time.perf_counter', None),
            ('Node.run', 'datetime.datetime.now', 'telemetry only'),
            ('Node.run', 'pydantic_ai._utils.now_utc', ''),
            ('Node.run', 'pydantic_ai._utils.now_utc', None),
            ('Node.run', 'concurrent.futures.ThreadPoolExecutor', None),
            ('Node.run', 'time.time', 'telemetry only'),
            ('Node.run', 'time.monotonic', None),
            ('Node.run', 'asyncio.sleep', 'replay-safe delay'),
            ('Node.run', 'time.time', None),
            ('Node.run.nested', 'uuid.uuid4', None),
            ('Node.run.nested', 'pydantic_ai._uuid.uuid7', None),
            ('Node.run', 'open', None),
            ('local_imports', 'datetime.date.today', None),
        ]
    )
    assert check_calls('fixture.py', calls)[0] == snapshot(
        [
            'fixture.py::Node.run has a `workflow-nondeterminism-ok` pragma with no reason',
            "fixture.py::<module> calls 'time.perf_counter', which reads a clock that has moved by the time the run replays",
            "fixture.py::Node.run calls 'pydantic_ai._utils.now_utc', which reads a clock that has moved by the time the run replays",
            "fixture.py::Node.run calls 'concurrent.futures.ThreadPoolExecutor', which finishes on another thread in an order that a replay does not reproduce",
            "fixture.py::Node.run calls 'time.monotonic', which reads a clock that has moved by the time the run replays",
            "fixture.py::Node.run calls 'time.time', which reads a clock that has moved by the time the run replays",
            "fixture.py::Node.run.nested calls 'uuid.uuid4', which draws randomness that a replay does not reproduce",
            "fixture.py::Node.run.nested calls 'pydantic_ai._uuid.uuid7', which draws randomness that a replay does not reproduce",
            "fixture.py::Node.run calls 'open', which performs I/O whose result can differ on replay; do it in an activity",
            "fixture.py::local_imports calls 'datetime.date.today', which reads a clock that has moved by the time the run replays",
        ]
    )


def test_check_calls_counts_calls_against_accepted_entries():
    """An `_ACCEPTED` entry explains as many calls as it counts, and a further call in the same function fails."""
    call = DeniedCall('resolve_run_id', 'pydantic_ai._uuid.uuid7', _RANDOM, None)
    module = f'{CORE}/_agent_graph.py'

    assert check_calls(module, [call])[0] == []
    assert check_calls(module, [call, call])[0] == snapshot(
        [
            "pydantic_ai_slim/pydantic_ai/_agent_graph.py::resolve_run_id calls 'pydantic_ai._uuid.uuid7' 2 times and `_ACCEPTED` accepts 1; give the new call a pragma with a reason"
        ]
    )


def test_every_denylist_key_matches_the_call_it_names():
    """Each `_DENYLIST` key catches the real call it is meant for, and each core key names a function that exists."""
    source = textwrap.dedent("""
    import anyio
    import asyncio
    import concurrent.futures
    import datetime
    import getpass
    import http.client
    import httpx
    import httpx2
    import multiprocessing
    import os
    import platform
    import random
    import secrets
    import socket
    import subprocess
    import threading
    import time
    import urllib.request
    import uuid
    from pydantic_ai import _http, _utils, _uuid

    time.time()
    time.time_ns()
    time.monotonic()
    time.monotonic_ns()
    time.perf_counter()
    time.perf_counter_ns()
    time.process_time()
    time.clock_gettime(0)
    time.clock_gettime_ns(0)
    time.localtime()
    time.gmtime()
    time.ctime()
    time.process_time_ns()
    time.thread_time()
    time.thread_time_ns()
    datetime.datetime.now()
    datetime.datetime.utcnow()
    datetime.datetime.today()
    datetime.date.today()
    _utils.now_utc()
    anyio.current_time()
    random.random()
    secrets.token_hex()
    os.urandom(8)
    uuid.uuid1()
    uuid.uuid4()
    uuid.uuid6()
    uuid.uuid7()
    uuid.uuid8()
    _uuid.uuid7()
    _utils.generate_tool_call_id()
    asyncio.sleep(1)
    asyncio.wait([])
    asyncio.as_completed([])
    anyio.sleep(1)
    anyio.sleep_until(1)
    time.sleep(1)
    threading.Thread()
    threading.Timer(1, print)
    anyio.to_thread.run_sync(print)
    concurrent.futures.ThreadPoolExecutor()
    open('file')
    httpx.Client()
    httpx.AsyncClient()
    httpx.request('GET', 'url')
    httpx.stream('GET', 'url')
    httpx.get('url')
    httpx.post('url')
    httpx.put('url')
    httpx.delete('url')
    httpx.head('url')
    httpx.patch('url')
    httpx.options('url')
    httpx2.Client()
    httpx2.AsyncClient()
    httpx2.request('GET', 'url')
    httpx2.stream('GET', 'url')
    httpx2.get('url')
    httpx2.post('url')
    httpx2.put('url')
    httpx2.delete('url')
    httpx2.head('url')
    httpx2.patch('url')
    httpx2.options('url')
    httpx2.query('url')
    httpx2.websocket('url')
    _http.create_async_httpx2_client()
    socket.socket()
    urllib.request.urlopen('url')
    subprocess.run([])
    os.system('')
    os.popen('')
    http.client.HTTPConnection('host')
    multiprocessing.Process()
    os.getenv('NAME')
    os.environ.get('NAME')
    os.getpid()
    platform.system()
    getpass.getuser()
    """)
    callees = [call.callee for call in nondeterministic_calls(source, 'pydantic_ai.agent', is_package=True)]

    assert len(callees) == sum('(' in line for line in source.splitlines())
    assert set(_DENYLIST) - {_denylist_key(callee) for callee in callees} == set()
    removed = [
        key
        for key in _DENYLIST
        if key.startswith('pydantic_ai.')
        and key.rpartition('.')[2] not in vars(importlib.import_module(key.rpartition('.')[0]))
    ]
    assert removed == [], 'these core functions no longer exist; deny what replaced them'


def test_workflow_side_code_rejects_stale_accepted_entries(monkeypatch: pytest.MonkeyPatch):
    """An `_ACCEPTED` entry that accepts more calls than the code still makes fails the check."""
    monkeypatch.setitem(_ACCEPTED, (f'{CORE}/run.py', 'AgentRun.next', 'time.time'), (1, 'a call that was removed'))

    with pytest.raises(AssertionError, match='accept more calls than remain'):
        test_workflow_side_code_makes_no_new_nondeterministic_calls()
