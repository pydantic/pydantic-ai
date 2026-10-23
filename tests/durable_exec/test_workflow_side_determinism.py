"""Workflow-side core code makes no new nondeterministic calls.

A durable engine replays workflow code to rebuild a run's state, so a call that answers differently on
replay (a clock, a random id, a sleep the engine does not journal) makes the replayed run diverge from
the one that was recorded. Temporal's sandbox does not catch these in Pydantic AI's own code:
`PydanticAIPlugin` passes `pydantic_ai` through the sandbox, so core modules run against the real
`time`, `datetime`, `random` and `uuid`. This test is the static check for them.

**The workflow-side module set.** Under Temporal, everything `TemporalAgent.run` and
`TemporalDurability.wrap_run` execute outside an activity is workflow code. Running the Temporal
durability, agent, toolset and tool-kind suites with a `sys.monitoring` hook that records each
function first entered on Temporal's workflow event loop shows 92 core modules outside
`durable_exec/` executing there. Most are data types and helpers (`messages`, `usage`, `_utils`,
the model profiles), whose calls are only workflow-side when an orchestrating module makes them. The set below is the modules
that own the run's control flow -- where a nondeterministic call decides what the workflow does
next -- each of which the trace shows running on the workflow loop:

- `_agent_graph.py`: the graph nodes, prompt assembly, request preparation and response handling.
- `agent/__init__.py`: `Agent.iter`, `_prepare_run` and `_PreparedAgentRun.open`.
- `run.py`: `AgentRun`, which drives the graph.
- `tool_manager.py` and `_tool_execution.py`: tool discovery, validation and dispatch.
- `_output.py`: output processing and validation.
- `_run_context.py`: the run context and its event emission.
- `capabilities/combined.py`: hook dispatch across the run's capabilities.
- `durable_exec/` and `durable_exec/temporal/`: every module the trace shows on the workflow loop.

Expand it only with the same evidence.

**What is denied** is in `_DENYLIST`, each with why replay breaks it. Thread offload
(`anyio.to_thread.run_sync`, `asyncio.to_thread`, `loop.run_in_executor`) is left out on purpose: it
already raises inside a Temporal workflow, whose event loop implements no executor, and
`_utils.run_in_executor` runs inline under `_utils.disable_threads` instead.

**Existing call sites** are in `_ACCEPTED`, keyed by module, enclosing qualname and callee so the
entries survive unrelated edits, with the number of such calls each key accepts. A new call is
accepted only with a pragma on its line naming the reason, `# workflow-nondeterminism-ok: <reason>`;
prefer a replay-safe alternative (an id or clock
the durability capability supplies; `_agent_graph_sleep` for a delay) when one exists.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from inline_snapshot import snapshot

REPO_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = REPO_ROOT / 'pydantic_ai_slim'
CORE = 'pydantic_ai_slim/pydantic_ai'

WORKFLOW_SIDE_MODULES: tuple[str, ...] = (
    f'{CORE}/_agent_graph.py',
    f'{CORE}/_output.py',
    f'{CORE}/_run_context.py',
    f'{CORE}/_tool_execution.py',
    f'{CORE}/agent/__init__.py',
    f'{CORE}/capabilities/combined.py',
    f'{CORE}/run.py',
    f'{CORE}/tool_manager.py',
    f'{CORE}/durable_exec/_base.py',
    f'{CORE}/durable_exec/_capability_operation.py',
    f'{CORE}/durable_exec/_operation.py',
    f'{CORE}/durable_exec/_operation_backend.py',
    f'{CORE}/durable_exec/_runtime_toolsets.py',
    f'{CORE}/durable_exec/_toolset.py',
    f'{CORE}/durable_exec/_utils.py',
    f'{CORE}/durable_exec/_workspace.py',
    f'{CORE}/durable_exec/temporal/_activity_execution.py',
    f'{CORE}/durable_exec/temporal/_agent.py',
    f'{CORE}/durable_exec/temporal/_durability.py',
    f'{CORE}/durable_exec/temporal/_dynamic_toolset.py',
    f'{CORE}/durable_exec/temporal/_function_toolset.py',
    f'{CORE}/durable_exec/temporal/_mcp_toolset.py',
    f'{CORE}/durable_exec/temporal/_model.py',
    f'{CORE}/durable_exec/temporal/_model_errors.py',
    f'{CORE}/durable_exec/temporal/_operation_backend.py',
    f'{CORE}/durable_exec/temporal/_replay_safe_tracer_provider.py',
    f'{CORE}/durable_exec/temporal/_run_context.py',
    f'{CORE}/durable_exec/temporal/_toolset.py',
    f'{CORE}/durable_exec/temporal/_transports.py',
)

_CLOCK = 'reads a clock that has moved by the time the run replays'
_RANDOM = 'draws randomness that a replay does not reproduce'
_SLEEP = 'sleeps outside the engine, so the engine does not journal the delay; await `_agent_graph_sleep`, which `set_agent_graph_sleep` routes to the engine'
_THREAD = 'finishes on another thread in an order that a replay does not reproduce'
_IO = 'performs I/O whose result can differ on replay; do it in an activity'

# Callee -> why replay breaks it. A key ending in `.` denies everything under that module.
_DENYLIST: dict[str, str] = {
    'time.time': _CLOCK,
    'time.time_ns': _CLOCK,
    'time.monotonic': _CLOCK,
    'time.monotonic_ns': _CLOCK,
    'time.perf_counter': _CLOCK,
    'time.perf_counter_ns': _CLOCK,
    'datetime.datetime.now': _CLOCK,
    'datetime.datetime.utcnow': _CLOCK,
    'datetime.datetime.today': _CLOCK,
    'datetime.date.today': _CLOCK,
    'pydantic_ai._utils.now_utc': _CLOCK,
    'random.': _RANDOM,
    'secrets.': _RANDOM,
    'os.urandom': _RANDOM,
    'uuid.uuid1': _RANDOM,
    'uuid.uuid4': _RANDOM,
    'pydantic_ai._uuid.uuid7': _RANDOM,
    'pydantic_ai._utils.generate_tool_call_id': _RANDOM,
    'asyncio.sleep': _SLEEP,
    'anyio.sleep': _SLEEP,
    'threading.Thread': _THREAD,
    'threading.Timer': _THREAD,
    'concurrent.futures.': _THREAD,
    'open': _IO,
    'httpx.Client': _IO,
    'httpx.AsyncClient': _IO,
    'socket.': _IO,
    'urllib.request.': _IO,
}

_PRAGMA = re.compile(r'#\s*workflow-nondeterminism-ok:(?P<reason>.*)$')

_RUN_ID = 'fallback id when the run has none; `TemporalDurability` supplies one from `workflow.uuid4()`'
_TIMESTAMP = 'stamps the `timestamp` metadata of a message the run appends; no branch reads it'

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
}


@dataclass(frozen=True)
class Call:
    """A denylisted call in one module."""

    qualname: str
    callee: str
    denial: str
    """Why the callee is denied."""
    pragma: str | None
    """The reason the line's pragma gives, `''` if it gives none, or `None` without a pragma."""


def _denial(callee: str) -> str | None:
    """Why `callee` is denied, or `None` if it is not."""
    if callee in _DENYLIST:
        return _DENYLIST[callee]
    return next((reason for key, reason in _DENYLIST.items() if key.endswith('.') and callee.startswith(key)), None)


class _CallFinder(ast.NodeVisitor):
    def __init__(self, source: str, module: str, is_package: bool) -> None:
        self.lines = source.splitlines()
        self.scopes: list[str] = []
        self.calls: list[Call] = []
        self.package = module.split('.') if is_package else module.split('.')[:-1]
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
                    base = [*self.package[: len(self.package) - level + 1], *([imported] if imported else [])]
                    for alias in names:
                        aliases[alias.asname or alias.name] = '.'.join([*base, alias.name])
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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append(node.name)
        self.aliases.append(self.scope_aliases(node.body))
        self.generic_visit(node)
        self.aliases.pop()
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(node.name)
        self.aliases.append(self.scope_aliases(node.body))
        self.generic_visit(node)
        self.aliases.pop()
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        callee = self.dotted(node.func)
        if callee is not None and (denial := _denial(callee)) is not None:
            pragma = _PRAGMA.search(self.lines[node.lineno - 1])
            reason = pragma['reason'].strip() if pragma else None
            self.calls.append(Call('.'.join(self.scopes) or '<module>', callee, denial, reason))
        self.generic_visit(node)


def nondeterministic_calls(source: str, module: str, *, is_package: bool = False) -> list[Call]:
    """The denylisted calls `source`, the code of `module`, makes, in source order."""
    return _CallFinder(source, module, is_package).calls


def unexplained_calls(relative: str, calls: list[Call]) -> tuple[list[str], Counter[tuple[str, str, str]]]:
    """The problems with the calls in the module at `relative`, and how many calls without a pragma each key has.

    A call is explained by a pragma with a reason, or by `_ACCEPTED` up to the number of calls that entry accepts.
    """
    problems: list[str] = []
    without_pragma = Counter((relative, call.qualname, call.callee) for call in calls if call.pragma is None)
    for call in calls:
        if call.pragma == '':
            problems.append(f'{relative}::{call.qualname} has a `workflow-nondeterminism-ok` pragma with no reason')
    denials = {(relative, call.qualname, call.callee): call.denial for call in calls}
    for key, count in without_pragma.items():
        accepted = _ACCEPTED.get(key, (0, ''))[0]
        if count > accepted:
            _, qualname, callee = key
            if accepted:
                problems.append(
                    f'{relative}::{qualname} calls `{callee}` {count} times and `_ACCEPTED` accepts {accepted}; '
                    'give the new call a pragma with a reason'
                )
            else:
                problems.append(f'{relative}::{qualname} calls `{callee}`, which {denials[key]}')
    return problems, without_pragma


def test_workflow_side_code_makes_no_new_nondeterministic_calls():
    found: Counter[tuple[str, str, str]] = Counter()
    problems: list[str] = []
    for relative in WORKFLOW_SIDE_MODULES:
        path = REPO_ROOT / relative
        parts = path.relative_to(SOURCE_ROOT).with_suffix('').parts
        is_package = parts[-1] == '__init__'
        module = '.'.join(parts[:-1] if is_package else parts)
        calls = nondeterministic_calls(path.read_text(encoding='utf-8'), module, is_package=is_package)
        module_problems, counts = unexplained_calls(relative, calls)
        problems.extend(module_problems)
        found.update(counts)
    assert problems == [], "nondeterministic calls in workflow-side code; see this module's docstring:\n" + '\n'.join(
        problems
    )
    stale = sorted(key for key, (accepted, _) in _ACCEPTED.items() if found[key] < accepted)
    assert stale == [], f'`_ACCEPTED` entries that accept more calls than remain, lower or remove them: {stale}'


_FIXTURE = """
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


def test_nondeterministic_calls_resolves_imports_and_pragmas():
    """Every import form resolves to the callee it names, and a pragma needs a reason to explain a call."""
    calls = nondeterministic_calls(_FIXTURE, 'pydantic_ai.agent', is_package=True)

    assert [(call.qualname, call.callee, call.pragma) for call in calls] == snapshot(
        [
            ('<module>', 'time.perf_counter', None),
            ('Node.run', 'datetime.datetime.now', 'telemetry only'),
            ('Node.run', 'pydantic_ai._utils.now_utc', ''),
            ('Node.run', 'pydantic_ai._utils.now_utc', None),
            ('Node.run', 'concurrent.futures.ThreadPoolExecutor', None),
            ('Node.run.nested', 'uuid.uuid4', None),
            ('Node.run.nested', 'pydantic_ai._uuid.uuid7', None),
            ('Node.run', 'open', None),
            ('local_imports', 'datetime.date.today', None),
        ]
    )
    assert unexplained_calls('fixture.py', calls)[0] == snapshot(
        [
            'fixture.py::Node.run has a `workflow-nondeterminism-ok` pragma with no reason',
            'fixture.py::<module> calls `time.perf_counter`, which reads a clock that has moved by the time the run replays',
            'fixture.py::Node.run calls `pydantic_ai._utils.now_utc`, which reads a clock that has moved by the time the run replays',
            'fixture.py::Node.run calls `concurrent.futures.ThreadPoolExecutor`, which finishes on another thread in an order that a replay does not reproduce',
            'fixture.py::Node.run.nested calls `uuid.uuid4`, which draws randomness that a replay does not reproduce',
            'fixture.py::Node.run.nested calls `pydantic_ai._uuid.uuid7`, which draws randomness that a replay does not reproduce',
            'fixture.py::Node.run calls `open`, which performs I/O whose result can differ on replay; do it in an activity',
            'fixture.py::local_imports calls `datetime.date.today`, which reads a clock that has moved by the time the run replays',
        ]
    )


def test_unexplained_calls_counts_calls_against_accepted_entries():
    """An `_ACCEPTED` entry explains as many calls as it counts, and a further call in the same function fails."""
    call = Call('resolve_run_id', 'pydantic_ai._uuid.uuid7', _RANDOM, None)
    module = f'{CORE}/_agent_graph.py'

    assert unexplained_calls(module, [call])[0] == []
    assert unexplained_calls(module, [call, call])[0] == snapshot(
        [
            'pydantic_ai_slim/pydantic_ai/_agent_graph.py::resolve_run_id calls `pydantic_ai._uuid.uuid7` 2 times and `_ACCEPTED` accepts 1; give the new call a pragma with a reason'
        ]
    )
