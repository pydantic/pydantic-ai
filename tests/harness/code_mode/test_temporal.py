"""Temporal integration tests for CodeMode.

Verifies that the snapshot-based execution loop (`feed_start`/`resume`)
works with Temporal's workflow sandbox and history replay.

Monty executes snippets in subprocess workers. The first `run_code` call lazily creates
the run's worker pool and initial checked-out REPL session. That session is reused across
calls until it is reset or invalidated.

Durability is attached via the `TemporalDurability` capability; pydantic-ai
2.14 deprecated the `TemporalAgent` wrapper in its favor
(pydantic/pydantic-ai#4977). The workflow calls the plain `Agent` directly and
`AgentPlugin` finds the bound capability to register its activities on the
worker. Pydantic AI resolves CodeMode outside TemporalDurability through
CodeMode's capability ordering metadata.

These tests start a local Temporal dev server via
`WorkflowEnvironment.start_local()` -- the Temporal SDK downloads and
runs `temporalite` automatically.
"""

from __future__ import annotations

import base64
import contextvars
import csv
import dataclasses
import enum
import json
import pickle
import re
import subprocess
import sys
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, TypeGuard
from uuid import uuid4

import pytest
from anyio.from_thread import start_blocking_portal
from pydantic import BaseModel
from pydantic_monty import AsyncMontyRemote, ExternalModule, MontyRefusedError, MountDir

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError, ApplicationError, RetryState
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker, WorkflowRunner
    from temporalio.worker.workflow_sandbox import (
        RestrictedWorkflowAccessError,
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

import pydantic_ai_harness
from pydantic_ai import Agent, Tool, ToolDefinition
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai.exceptions import PydanticAIDeprecationWarning, ToolFailed, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness._monty_turns import from_wire, to_wire
from tests.harness._temporal import ignore_source_reads_left_open
from tests.harness.code_mode.grpc_relay import serve_monty_relay
from tests.harness.code_mode.stored_sessions import SERVER, stored_session_pool
from tests.temporal_utils import temporal_dev_server_cache_dir

# Its own group, on its own port: it is the largest harness Temporal module, and in the shared group it
# would run after all the others, past the durable-exec job's timeout on the slower Python versions.
pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal-code-mode'),
    ignore_source_reads_left_open,
]

TEMPORAL_PORT = 7244  # avoid conflict with other test suites
# Fixed because the agent below is built at import time, before any fixture runs.
MONTY_RELAY_PORT = 7275  # not 7246: logfire_mcp's Temporal server, which can run beside this module
TASK_QUEUE = 'pydantic-ai-harness-code-mode-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


def _workflow_runner(*, reimport_this_module: bool = False) -> SandboxedWorkflowRunner:
    """A sandbox that imports this module once, since building its agents again for every run and replay is slow.

    `reimport_this_module` makes the sandbox import it for each run, as it would a user's workflow module, so
    the `subprocess` this module imported is the sandbox's restricted one.
    """
    this_module = () if reimport_this_module else (__name__,)
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            *this_module,
            # Coverage imports parser modules lazily while tracing workflow code.
            'coverage',
            # `PydanticAIPlugin` registers `pydantic_graph`'s `UnsupportedEventLoopError` as a
            # workflow failure exception type but doesn't pass `pydantic_graph` through, so the
            # sandbox imports it for real and trips over the `os.environ.get` that
            # `opentelemetry.context` runs at import time. Drop this once
            # pydantic/pydantic-ai#6986 adds it to the plugin's own passthrough list.
            'pydantic_graph',
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=[
            '--dynamic-config-value',
            'frontend.enableServerVersionCheck=false',
        ],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


@pytest.fixture
async def temporal_monty_relay() -> AsyncIterator[None]:  # pragma: lax no cover -- only the skipped test uses it
    """Serve remote Monty workers on the port `remote_code_mode_agent` is configured with."""
    async with serve_monty_relay(MONTY_RELAY_PORT):
        yield


@pytest.fixture(autouse=True)
def fresh_server() -> None:
    """Start each test with an empty store, no planned failures, and no pool calls counted."""
    SERVER.reset()
    refusing_pool_calls.clear()
    forgetful_pool_calls.clear()


@pytest.fixture
def workflow_id() -> str:
    """An ID no other workflow in the module's Temporal server uses."""
    return f'test_code_mode_temporal_{uuid4().hex}'


# ---------------------------------------------------------------------------
# Tools and agents (module-level -- Temporal requirement)
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


_captured_tool_defs: list[list[ToolDefinition]] = []
_captured_instructions: list[str | None] = []


def _run_code_returns(messages: list[ModelRequest | ModelResponse]) -> list[object]:
    """What each `run_code` call in `messages` returned."""
    return [
        part.content
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]


# FunctionModel that emits a run_code tool call for the given code snippet.
def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that uses two REPL feeds, then returns the second result as text."""
    _captured_tool_defs.append(info.function_tools)
    _captured_instructions.append(info.instructions)

    returns = _run_code_returns(messages)
    if len(returns) == 1:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='run_code', args={'code': 'result * 10'}, tool_call_id='test_tc_2')]
        )
    if len(returns) == 2:
        return ModelResponse(parts=[TextPart(content=f'done: {returns[-1]}')])

    code = 'result = await add(a=3, b=4)\nresult'
    if any(
        isinstance(part, UserPromptPart) and part.content == 'Gather two additions'
        for message in messages
        for part in message.parts
    ):
        code = 'import asyncio\nawait asyncio.gather(add(a=3, b=4), add(a=5, b=6))'
    # First call -- emit run_code.
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': code},
                tool_call_id='test_tc_1',
            )
        ]
    )


code_mode_agent = Agent(
    FunctionModel(_code_mode_model),
    name='code_mode_temporal_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


# Set by the workflow and read by `os_access`, which runs on the workflow's own thread.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='unset')


# Only the skipped relay test calls this `os_access` (#14674).
def _workflow_os(
    *, name: str, args: tuple[object, ...], kwargs: dict[str, object], **_: object
) -> object:  # pragma: lax no cover
    if name == 'datetime.now':
        # Raises "Not in workflow event loop" anywhere but the workflow's own thread.
        return workflow.now()
    return _request_id.get()


def _remote_code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that adds with a tool, sleeps, and reads the workflow's contextvar through `os_access`."""
    returns = _run_code_returns(messages)
    if returns:
        return ModelResponse(parts=[TextPart(content=f'done: {returns[-1]}')])  # pragma: lax no cover
    # The 31 s sleep is one second past the default allowance, so it is refused without waiting.
    code = (
        'import asyncio, datetime, os\ntotal = await add(a=3, b=4)\nawait asyncio.sleep(0.1)\n'
        'try:\n    await asyncio.sleep(31)\n    capped = "uncapped"\nexcept TimeoutError:\n    capped = "capped"\n'
        'f\'{total} {os.getenv("REQUEST_ID")} {capped} {datetime.datetime.now().year > 2000}\''
    )
    return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='remote_tc_1')])


remote_code_mode_agent = Agent(
    FunctionModel(_remote_code_mode_model),
    name='code_mode_temporal_remote_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[
        CodeMode(monty_sandbox=f'http://127.0.0.1:{MONTY_RELAY_PORT}', os_access=_workflow_os),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


def _code_mode_agent(
    name: str,
    code_mode: CodeMode[object] | DynamicCapability[object],
    activity_config: ActivityConfig = BASE_ACTIVITY_CONFIG,
) -> Agent[object, str]:
    """An agent that runs `_code_mode_model`'s two snippets through `code_mode`."""
    return Agent(
        FunctionModel(_code_mode_model),
        name=f'code_mode_temporal_{name}_agent',
        toolsets=[FunctionToolset(tools=[add], id='math')],
        capabilities=[code_mode, TemporalDurability(activity_config=activity_config)],
    )


# No `id`: the class default is what binds the turn activity.
storing_code_mode_agent = _code_mode_agent('storing', CodeMode(monty_sandbox=stored_session_pool))
not_storing_code_mode_agent = _code_mode_agent(
    'not_storing', CodeMode(id='code', monty_sandbox=partial(stored_session_pool, stores=False))
)

# Runs that keep the workflow-side path although the server stores sessions.
workflow_side_code_mode_agents = {
    # One overlay mount, the default mode, in a list beside a read-only one is enough. Monty needs their
    # host directories disjoint, so the read-only one is the installed package's.
    'overlay': _code_mode_agent(
        'overlay',
        CodeMode(
            id='code',
            monty_sandbox=stored_session_pool,
            mount=[
                MountDir(
                    virtual_path='/data', host_path=str(Path(pydantic_ai_harness.__file__).parent), mode='read-only'
                ),
                MountDir(virtual_path='/tests', host_path=str(Path(__file__).parent)),
            ],
        ),
    ),
    'os_access': _code_mode_agent(
        'os_access', CodeMode(id='code', monty_sandbox=stored_session_pool, os_access=_workflow_os)
    ),
    # Monty restarts a mount's write count when a stored session loads.
    'write_limit': _code_mode_agent(
        'write_limit',
        CodeMode(
            id='code',
            monty_sandbox=stored_session_pool,
            mount=MountDir(
                virtual_path='/tests',
                host_path=str(Path(__file__).parent),
                mode='read-write',
                write_bytes_limit=6,
            ),
        ),
    ),
    # A `CodeMode` a `DynamicCapability` returns has no turn activity bound.
    'dynamic': _code_mode_agent(
        'dynamic',
        DynamicCapability(lambda ctx: CodeMode(id='code', monty_sandbox=stored_session_pool), id='dynamic_code'),
    ),
    # Runs turns as activities in a workflow that started after this version of Code Mode.
    'unpatched': storing_code_mode_agent,
}


def double(n: int) -> int:
    """Double a number."""
    return n * 2


def echo(value: object) -> object:
    """Return `value`."""
    return value


def blob() -> dict[str, object]:
    """Return values plain JSON does not carry: bytes that are not UTF-8, nested too."""
    return {
        'raw': b'\xff\x00',
        'nested': {'a': {'b': b'\xff', 'c': 2}, 't': [[1, b'x']]},
    }


_CALL_SITE_SNIPPETS = [
    # Only a fresh session's first snippet is type-checked: this one fails and resets the session,
    # the next one passes, and the third runs unchecked.
    'add(3, 4)',
    # The print cap, which the test lowers to 200 bytes, holds across the turns sleeps split a snippet
    # into. Going over it resets the session, as running out of memory does.
    'import time\nfor _ in range(3):\n    print("x" * 80)\n    time.sleep(0)',
    'n = await add(a=1, b=1)\nn',
    # A sequential tool, an error raised at a call site, an undefined name, and a mounted file read
    # each resume the paused call directly instead of deferring it.
    'from pathlib import Path\n'
    'try:\n    add(3, 4)\nexcept TypeError as e:\n    error = str(e)\n'
    'try:\n    missing_name\nexcept NameError:\n    error += " / NameError"\n'
    'line = Path("/tests/stored_sessions.py").read_text().splitlines()[0]\n'
    'f"{double(n=n)} | {error} | {line}"',
    'def broken(:',
    # An argument with no JSON form raises at the call site; a result with none comes back as its `repr`.
    'from dataclasses import dataclass\n'
    '@dataclass\nclass P:\n    x: int\n'
    'try:\n    await add(a=P(1), b=2)\nexcept TypeError as e:\n    arg_error = str(e)\n'
    '[P(1), arg_error]',
    # Values plain JSON would change, in a tool result and in the snippet's result.
    'v = await blob()\n'
    'import datetime\n'
    'e = await echo(value=[(1, b"x"), {2: (b"y",)}, float("nan")])\n'
    '[type(v["raw"]).__name__, v["raw"], v["nested"], e, (1, ...),\n'
    ' {(1, 2): {"k": (b"a",)}}, datetime.date(2026, 1, 1)]',
    # Bytes that are not UTF-8 reach the tool's call as on the workflow-side path, which rejects them.
    'await echo(value=b"\\xff")',
    # So does `max_suspensions`, 16 here, which counts every pause in the session, earlier snippets'
    # included. Last, as the session can then pause no more.
    'import time\nfor _ in range(20):\n    time.sleep(0)',
]


def _call_site_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that runs each of `_CALL_SITE_SNIPPETS`, then returns what they returned."""
    returns = _run_code_returns(messages)
    if len(returns) == len(_CALL_SITE_SNIPPETS):
        return ModelResponse(parts=[TextPart(content=json.dumps(returns, default=repr))])
    code = _CALL_SITE_SNIPPETS[len(returns)]
    return ModelResponse(
        parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id=f'site_{len(returns)}')]
    )


def _call_site_agent(*, turn_activities: bool) -> Agent[object, str]:
    """Without an `id`, `CodeMode` has no turn operation, so every turn stays in the workflow."""
    return Agent(
        FunctionModel(_call_site_model),
        name=f'code_mode_temporal_call_site_{"turns" if turn_activities else "workflow_side"}_agent',
        toolsets=[FunctionToolset(tools=[add, Tool(double, sequential=True), blob, echo], id='math')],
        capabilities=[
            CodeMode(
                id='code' if turn_activities else None,
                monty_sandbox=stored_session_pool,
                mount=MountDir(virtual_path='/tests', host_path=str(Path(__file__).parent), mode='read-only'),
                resource_limits={'max_suspensions': 16},
            ),
            TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
        ],
    )


call_site_agents = {True: _call_site_agent(turn_activities=True), False: _call_site_agent(turn_activities=False)}


refusing_pool_calls: list[None] = []


def _refusing_pool() -> AsyncMontyRemote:
    refusing_pool_calls.append(None)
    raise ConnectionRefusedError('the Monty server refused the connection')


refusing_code_mode_agent = _code_mode_agent(
    'refusing',
    CodeMode(id='code', monty_sandbox=_refusing_pool),
    ActivityConfig(
        start_to_close_timeout=timedelta(seconds=60),
        retry_policy=RetryPolicy(
            maximum_attempts=3,
            initial_interval=timedelta(milliseconds=1),
            non_retryable_error_types=['ConnectionRefusedError'],
        ),
    ),
)


def _session_loss_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that sets `x`, lets the server lose the session as the prompt says, then reads `x` back."""
    results = _run_code_returns(messages)
    if not results:
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'x = 1\nx'}, tool_call_id='set')])
    if len(results) == 1:
        # The model request is an activity in this process, between the two `run_code` calls.
        prompt = next(part.content for part in messages[0].parts if isinstance(part, UserPromptPart))
        if prompt == 'expire the record':
            SERVER.records.clear()
        else:
            SERVER.drops_next_feed = ConnectionResetError('wss://monty:secret-token@sandbox.example closed')
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'x + 1'}, tool_call_id='read')])
    return ModelResponse(parts=[TextPart(content=json.dumps(results))])


session_loss_code_mode_agent = Agent(
    FunctionModel(_session_loss_model),
    name='code_mode_temporal_session_loss_agent',
    capabilities=[
        # `'unlimited'` too: a turn then has no `max_suspensions` for earlier turns to use up.
        CodeMode(
            id='code',
            monty_sandbox=stored_session_pool,
            resource_limits='unlimited',
        ),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


async def frozen() -> None:
    """Raise the error a frozen dataclass raises on assignment."""
    raise dataclasses.FrozenInstanceError('cannot assign to field')


async def raw_bytes() -> object:
    """Return a `bytearray`, which the sandbox has no type for."""
    return bytearray(b'hi')


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def test_turn_values_keep_what_temporal_json_cannot() -> None:
    """Deep nesting, huge integers and shared values come back as they went in, as one short string.

    Called directly: an agent run cannot carry these values, since `run_code` metadata in the message
    history must also cross as JSON.
    """
    deep: object = 0
    # Past the 200 levels Temporal's Pydantic JSON decoder takes, within what Python 3.11's pickler can write.
    for _ in range(250):
        deep = [deep]
    shared = [[1, 2]] * 2
    values = (deep, 10**5000, shared, (1j, ..., {1}, frozenset({2}), b'\xff', bytearray(b'x')))
    data = to_wire(values)
    assert isinstance(data, str)
    back = from_wire(data)
    assert back == values
    assert _is_object_tuple(back)
    back_shared = back[2]
    assert _is_object_list(back_shared)
    assert back_shared[0] is back_shared[1]


class _Pair(NamedTuple):
    first: int
    second: int


class _Items(list[int]):
    pass


class _Fields(dict[str, int]):
    pass


class _Members(set[int]):
    pass


class _FrozenMembers(frozenset[int]):
    pass


class _Ratio(float):
    pass


class _Blob(bytes):
    pass


def test_turn_values_send_subclasses_as_their_base_types() -> None:
    """A subclass of a type a live session takes reaches the sandbox as that base type, as from a live session.

    Called directly: the tools that return these run in workflow code, whose lines coverage does not record.
    """
    values = (
        _Pair(1, 2),
        _Items([3]),
        _Fields(a=4),
        _Members({5}),
        _FrozenMembers({6}),
        _Ratio(7.5),
        _Blob(b'8'),
        MappingProxyType({'b': 9}),
    )
    back = from_wire(to_wire(values))
    assert back == ((1, 2), [3], {'a': 4}, {5}, frozenset({6}), 7.5, b'8', {'b': 9})
    assert _is_object_tuple(back)
    assert [type(value) for value in back] == [tuple, list, dict, set, frozenset, float, bytes, dict]


def test_turn_values_refuse_a_value_nested_too_deeply_to_send() -> None:
    """A value nested past what the pickler can write fails as a `TypeError`, as other unsendable values do."""
    deep: object = 0
    # Deep enough for each supported Python's pickler, some of which write tens of thousands of levels.
    for _ in range(100_000):
        deep = [deep]
    with pytest.raises(TypeError, match='the value is nested too deeply to send'):
        to_wire(deep)


def test_turn_values_load_only_wire_types() -> None:
    """A recorded turn value that names any other class is refused instead of loaded.

    Called directly: only a forged history reaches this refusal, and it is what keeps loading one safe.
    """
    data = base64.b64encode(pickle.dumps(subprocess.run)).decode()
    with pytest.raises(pickle.UnpicklingError, match=r'subprocess\.run is not a wire type'):
        from_wire(data)


class Colour(enum.StrEnum):
    RED = 'red'


class Level(enum.IntEnum):
    HIGH = 3


async def enum_members() -> list[object]:
    """Return an `StrEnum` and an `IntEnum` member, which a live session takes as their plain values."""
    return [Colour.RED, Level.HIGH]


forgetful_pool_calls: list[None] = []


def _forgetful_pool() -> AsyncMontyRemote:
    """A pool whose server stores the first turn's session, and whose later checkouts land on one that stores none."""
    forgetful_pool_calls.append(None)
    return stored_session_pool(stores=len(forgetful_pool_calls) == 1)


def repeat(text: str, times: int) -> str:
    """Return `text` repeated `times` times."""
    return text * times


def give_up() -> str:
    """Fail without asking the model to retry."""
    raise ToolFailed('the tool gave up')


async def type_names(values: list[object]) -> list[str]:
    """Name each value's type. It runs in the workflow, so values reach it as the turn delivers them."""
    # Coverage does not record this line, which runs in workflow code; the `host_types` case runs it.
    return [type(value).__name__ for value in values]  # pragma: lax no cover


async def same(first: object, second: object) -> bool:
    """Whether `first` equals `second`."""
    # The test refuses the call's arguments before the tool runs.
    return first == second  # pragma: lax no cover


async def echo_here(value: object) -> object:
    """Return `value`. It runs in the workflow, so the value crosses only the turn boundary."""
    return value


async def bad_csv() -> str:
    """Raise `csv.Error`. It runs in the workflow, so its error reaches the sandbox as itself."""
    raise csv.Error('bad row')


NEXT_SNIPPET = '\n# next snippet\n'


def _snippet_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that runs each `NEXT_SNIPPET`-separated part of the prompt as a snippet, then returns the last result.

    Only a run's first snippet is type-checked, so a later one can make calls the checker would reject.
    """
    returns = _run_code_returns(messages)
    prompt = next(part.content for part in messages[0].parts if isinstance(part, UserPromptPart))
    assert isinstance(prompt, str)
    codes = prompt.split(NEXT_SNIPPET)
    if len(returns) == len(codes):
        return ModelResponse(parts=[TextPart(content=str(returns[-1]))])
    code = codes[len(returns)]
    return ModelResponse(
        parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id=f'snippet{len(returns) or ""}')]
    )


def _snippet_agent(
    name: str,
    monty_sandbox: Callable[[], AsyncMontyRemote] = stored_session_pool,
    resource_limits: Literal['unlimited'] | None = None,
    workflow_tools: tuple[Callable[..., Awaitable[object]], ...] = (bad_csv,),
) -> Agent[object, str]:
    """An agent whose `workflow_tools` run in the workflow, so their values cross only the turn boundary."""
    in_workflow = [Tool(function, metadata={'temporal': False}) for function in workflow_tools]
    return Agent(
        FunctionModel(_snippet_model),
        name=f'code_mode_temporal_snippet_{name}_agent',
        toolsets=[FunctionToolset(tools=[echo, repeat, give_up, *in_workflow], id='echo')],
        capabilities=[
            CodeMode(id='code', monty_sandbox=monty_sandbox, resource_limits=resource_limits),
            TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
        ],
    )


def _refusing_factory() -> AsyncMontyRemote:
    raise UserError('secret-sentinel: the pool factory could not read its settings')


snippet_agents = {
    'default': _snippet_agent('default'),
    'unlimited': _snippet_agent('unlimited', resource_limits='unlimited'),
    'factory_error': _snippet_agent('factory_error', monty_sandbox=_refusing_factory),
    'types': _snippet_agent('types', workflow_tools=(type_names, echo_here, raw_bytes, frozen, same, enum_members)),
    'forgetful': _snippet_agent('forgetful', monty_sandbox=_forgetful_pool),
}


def lookup(key: str) -> str:
    """Read `key` from a fixed table: a module function runs in the workflow, so it must return the same on replay."""
    return {'region': 'eu', 'tier': 'gold'}[key]


async def scaled(n: int) -> int:
    """Scale `n`. It is async, so the sandbox can gather it with a tool call."""
    return n * 100


def _module_agent(name: str, monty_sandbox: Callable[[], AsyncMontyRemote] | None) -> Agent[object, str]:
    """An agent whose sandbox imports the `catalog` host module, on the local pool or in remote turns."""
    catalog = ExternalModule(
        {'lookup': lookup, 'scaled': scaled},
        stubs='def lookup(key: str) -> str: ...\nasync def scaled(n: int) -> int: ...',
    )
    return Agent(
        FunctionModel(_snippet_model),
        name=f'code_mode_temporal_modules_{name}_agent',
        toolsets=[FunctionToolset(tools=[echo], id='echo')],
        capabilities=[
            CodeMode(id='code', monty_sandbox=monty_sandbox, external_modules={'catalog': catalog}),
            TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
        ],
    )


module_agents = {'local': _module_agent('local', None), 'turns': _module_agent('turns', stored_session_pool)}

AGENTS = {
    agent.name: agent
    for agent in [
        storing_code_mode_agent,
        not_storing_code_mode_agent,
        *workflow_side_code_mode_agents.values(),
        *call_site_agents.values(),
        refusing_code_mode_agent,
        session_loss_code_mode_agent,
        *snippet_agents.values(),
        *module_agents.values(),
    ]
}
"""The agents `AgentWorkflow` runs, by name. Built at import, so the workflow sandbox's own import of this module
has them too."""


@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, agent: str, prompt: str) -> str:
        result = await AGENTS[agent].run(prompt)
        return str(result.output)


@workflow.defn
class CodeModeWorkflow:
    @workflow.run
    async def run(self, prompt: str, tool_calls_limit: int | None = None) -> dict[str, Any]:
        result = await code_mode_agent.run(prompt, usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit))
        return {
            'output': str(result.output),
            'messages': result.all_messages_json().decode(),
        }


@workflow.defn
class SandboxRestrictionWorkflow:
    """Probe that passing Monty through does not allow Python subprocess calls."""

    @workflow.run
    async def run(self) -> str:
        try:
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
        except RestrictedWorkflowAccessError as e:
            return e.qualified_name
        return 'subprocess was allowed'  # pragma: no cover


@workflow.defn
class RemoteCodeModeWorkflow:
    """`CodeModeWorkflow` against remote workers reached over a `monty_sandbox` URL."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        _request_id.set('req-42')  # pragma: no cover
        result = await remote_code_mode_agent.run(prompt)  # pragma: no cover
        return str(result.output)  # pragma: no cover


class Total(BaseModel):
    total: int


def _submit_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    code = 'total = await add(a=40, b=2)\nfinal_result(total=total)'
    return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='submit')])


submit_agent = Agent(
    FunctionModel(_submit_model),
    name='code_mode_temporal_submit_agent',
    output_type=Total,
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode(allow_final_output=True), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class SubmitWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, int]:
        result = await submit_agent.run(prompt)
        return {'total': result.output.total, 'requests': result.usage.requests}


@workflow.defn
class RemoteCodeModeRefusalWorkflow:
    """Return the original sandbox refusal without admitting a remote session."""

    @workflow.run
    async def run(self) -> str | None:
        try:
            result = await remote_code_mode_agent.run('Calculate 3 + 4')
        except MontyRefusedError as error:
            return error.cause
        return str(result.output)  # pragma: no cover


@workflow.defn
class UncaughtRemoteCodeModeRefusalWorkflow:
    """Let an uncaught sandbox refusal reach Temporal's workflow-task failure policy."""

    @workflow.run
    async def run(self) -> None:
        await remote_code_mode_agent.run('Calculate 3 + 4')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_code_mode_runs_in_temporal_workflow(client: Client) -> None:
    """CodeMode runs workflow-side, nested tools use activities, and the history replays."""
    _captured_tool_defs.clear()
    _captured_instructions.clear()
    workflow_id = 'test_code_mode_temporal_1'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CodeModeWorkflow, SandboxRestrictionWorkflow],
        plugins=[AgentPlugin(code_mode_agent)],
        workflow_runner=_workflow_runner(reimport_this_module=True),
    ):
        result = await client.execute_workflow(
            CodeModeWorkflow.run,
            args=['Calculate 3 + 4'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        sandbox_result = await client.execute_workflow(
            SandboxRestrictionWorkflow.run,
            id='test_code_mode_temporal_sandbox_restrictions',
            task_queue=TASK_QUEUE,
        )

    assert result['output'] == 'done: 70'
    assert sandbox_result == 'subprocess.run.__call__'

    messages = json.loads(result['messages'])
    assert len(messages) == 6

    # 1. User prompt
    assert messages[0]['kind'] == 'request'
    assert messages[0]['parts'][0]['part_kind'] == 'user-prompt'
    assert messages[0]['parts'][0]['content'] == 'Calculate 3 + 4'

    # 2. Model response -- run_code tool call
    assert messages[1]['kind'] == 'response'
    tc = messages[1]['parts'][0]
    assert tc['part_kind'] == 'tool-call'
    assert tc['tool_name'] == 'run_code'
    assert tc['args'] == {'code': 'result = await add(a=3, b=4)\nresult'}
    assert tc['tool_call_id'] == 'test_tc_1'

    # 3. Tool return with nested tool call metadata
    assert messages[2]['kind'] == 'request'
    tr = messages[2]['parts'][0]
    assert tr['part_kind'] == 'tool-return'
    assert tr['tool_name'] == 'run_code'
    assert tr['content'] == 7
    assert tr['tool_call_id'] == 'test_tc_1'

    # Verify nested tool call/return metadata
    metadata = tr['metadata']
    assert metadata is not None
    assert metadata['code_mode'] is True
    nested_calls = metadata['tool_calls']
    nested_returns = metadata['tool_returns']
    assert len(nested_calls) == 1
    assert len(nested_returns) == 1

    nested_call = next(iter(nested_calls.values()))
    assert nested_call['tool_name'] == 'add'
    assert nested_call['args'] == {'a': 3, 'b': 4}

    nested_return = next(iter(nested_returns.values()))
    assert nested_return['tool_name'] == 'add'
    assert nested_return['content'] == 7
    assert nested_return['tool_call_id'] == nested_call['tool_call_id']

    # 4-5. A second feed consumes the variable assigned by the first feed.
    assert messages[3]['kind'] == 'response'
    second_call = messages[3]['parts'][0]
    assert second_call['part_kind'] == 'tool-call'
    assert second_call['args'] == {'code': 'result * 10'}
    assert messages[4]['kind'] == 'request'
    second_return = messages[4]['parts'][0]
    assert second_return['part_kind'] == 'tool-return'
    assert second_return['content'] == 70

    # 6. Final text response
    assert messages[5]['kind'] == 'response'
    assert messages[5]['parts'][0]['part_kind'] == 'text'
    assert messages[5]['parts'][0]['content'] == 'done: 70'

    # 5. Verify tool definitions sent to the model
    assert len(_captured_tool_defs) == 3
    for tool_defs, instructions in zip(_captured_tool_defs, _captured_instructions, strict=True):
        tool_names = [td.name for td in tool_defs]
        # CodeMode wraps `add` into `run_code` -- the model should only see `run_code`
        assert 'run_code' in tool_names
        assert 'add' not in tool_names

        # The catalog crosses the model request in the instructions, not in `run_code`'s description.
        assert instructions is not None
        assert 'async def add' in instructions
        run_code_td = next(td for td in tool_defs if td.name == 'run_code')
        assert run_code_td.description is not None
        assert 'async def add' not in run_code_td.description
        assert run_code_td.parameters_json_schema['properties']['code']['type'] == 'string'

    history = await _history(client, workflow_id)
    # A local sandbox keeps every turn in the workflow.
    assert 'monty_turn' not in _activity_types(history)


@pytest.mark.parametrize(
    'prompt,limit', [('Calculate 3 + 4', 1), ('Gather two additions', 2)], ids=['sequential', 'gather']
)
async def test_tool_calls_limit_fails_and_replays(client: Client, prompt: str, limit: int) -> None:
    workflow_id = f'test_code_mode_temporal_tool_calls_limit_{limit}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CodeModeWorkflow],
        plugins=[AgentPlugin(code_mode_agent)],
        workflow_runner=_workflow_runner(),
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CodeModeWorkflow.run,
                args=[prompt, limit],
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )

    cause = exc_info.value.__cause__
    assert isinstance(cause, ApplicationError)
    assert cause.type == UsageLimitExceeded.__name__
    assert f'tool_calls_limit of {limit} (tool_calls={limit + 1})' in cause.message

    await _history(client, workflow_id)


async def test_uncaught_remote_setup_error_fails_workflow_task(client: Client) -> None:
    """Unclassified setup errors fail a task, not the workflow or a model retry, under Temporal's default policy."""
    with (
        start_blocking_portal() as portal,
        portal.wrap_async_context_manager(serve_monty_relay(MONTY_RELAY_PORT, refusal_causes=('unauthenticated',))),
    ):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[UncaughtRemoteCodeModeRefusalWorkflow],
            plugins=[AgentPlugin(remote_code_mode_agent)],
            workflow_runner=_workflow_runner(),
        ):
            handle = await client.start_workflow(
                UncaughtRemoteCodeModeRefusalWorkflow.run,
                id='test_code_mode_temporal_uncaught_remote_setup',
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )
            try:
                failed_task = await anext(
                    event
                    async for event in handle.fetch_history_events(wait_new_event=True)
                    if event.HasField('workflow_task_failed_event_attributes')
                )
                failure = failed_task.workflow_task_failed_event_attributes.failure
                assert failure.application_failure_info.type == 'MontyRefusedError'
                assert 'UNAUTHENTICATED' in failure.message
                assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING
                history = await handle.fetch_history()
                assert not any(event.HasField('workflow_execution_failed_event_attributes') for event in history.events)
                assert sum(event.HasField('activity_task_scheduled_event_attributes') for event in history.events) == 1
            finally:
                await handle.terminate('test verified the workflow-task failure')


async def test_remote_checkout_backoff_replays_without_workflow_timer(client: Client) -> None:
    """Network-dependent checkout backoff must not create a durable workflow timer.

    The original run retries a capacity refusal before returning an authentication refusal.
    Replay immediately encounters the authentication refusal, so recording the original backoff
    as a workflow timer would make its command history nondeterministic. No session is admitted,
    avoiding the remote execution bridge while exercising real gRPC refusals through CodeMode.
    The relay uses its own event loop because Temporal debug mode runs workflows on the test loop.
    """
    workflow_id = 'test_code_mode_temporal_remote_checkout_backoff'
    with (
        start_blocking_portal() as portal,
        portal.wrap_async_context_manager(
            serve_monty_relay(MONTY_RELAY_PORT, refusal_causes=('capacity', 'unauthenticated'))
        ) as relay,
    ):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[RemoteCodeModeRefusalWorkflow],
            plugins=[AgentPlugin(remote_code_mode_agent)],
            workflow_runner=_workflow_runner(),
        ):
            output = await client.execute_workflow(
                RemoteCodeModeRefusalWorkflow.run,
                id=workflow_id,
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )
        assert output == 'unauthenticated'
        assert len(relay.metadata) == 2

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    with (
        start_blocking_portal() as portal,
        portal.wrap_async_context_manager(
            serve_monty_relay(MONTY_RELAY_PORT, refusal_causes=('unauthenticated',))
        ) as relay,
    ):
        replay_result = await Replayer(
            workflows=[RemoteCodeModeRefusalWorkflow],
            plugins=[PydanticAIPlugin()],
            workflow_runner=_workflow_runner(),
        ).replay_workflow(history)
        assert replay_result.replay_failure is None
        assert len(relay.metadata) == 1
    assert not any(event.HasField('timer_started_event_attributes') for event in history.events)


@pytest.mark.skip(reason='Hangs in CI under xdist: https://github.com/pydantic/pydantic-ai/issues/14674')
@pytest.mark.usefixtures('temporal_monty_relay')
async def test_code_mode_runs_remotely_in_temporal_workflow(client: Client) -> None:
    """Remote workers run and replay in a workflow like local ones do.

    Every Monty call inside a workflow goes through the blocking portal, so this covers that
    path for both bindings, including `os_access` seeing the workflow's contextvars, and a
    sandbox sleep becoming a durable timer rather than blocking the workflow, still capped by the
    `max_duration_seconds` allowance even though the execution-time limit is off in a workflow.
    """
    workflow_id = 'test_code_mode_temporal_remote_1'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RemoteCodeModeWorkflow],
        plugins=[AgentPlugin(remote_code_mode_agent)],
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            RemoteCodeModeWorkflow.run,
            args=['Calculate 3 + 4'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    assert output == 'done: 7 req-42 capped True'

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    assert any(e.HasField('timer_started_event_attributes') for e in history.events)
    replay_result = await Replayer(
        workflows=[RemoteCodeModeWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None


async def test_code_mode_submits_final_output_in_temporal_workflow(client: Client) -> None:
    """`run_code` runs in the workflow under Temporal, so the code can submit the run's output, and the history replays."""
    workflow_id = 'test_code_mode_temporal_submit'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SubmitWorkflow],
        plugins=[AgentPlugin(submit_agent)],
        workflow_runner=_workflow_runner(),
    ):
        result = await client.execute_workflow(
            SubmitWorkflow.run,
            args=['Add 40 and 2'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    assert result == {'total': 42, 'requests': 1}

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay_result = await Replayer(
        workflows=[SubmitWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None


async def _run(
    client: Client,
    workflow_id: str,
    agent: Agent[object, str],
    prompt: str = 'Calculate 3 + 4',
    runner: WorkflowRunner | None = None,
) -> str:
    """Run `agent` on `prompt` in `AgentWorkflow`, on a worker for that agent alone."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        plugins=[AgentPlugin(agent)],
        workflow_runner=runner or _workflow_runner(),
    ):
        return await client.execute_workflow(
            AgentWorkflow.run,
            args=[agent.name, prompt],
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )


async def _history(client: Client, workflow_id: str) -> WorkflowHistory:
    """The workflow's history, once it has replayed without a failure."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay_result = await Replayer(
        workflows=[AgentWorkflow, CodeModeWorkflow, LegacyWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None
    return history


def _activity_types(history: WorkflowHistory) -> list[str]:
    """The activities a workflow scheduled, by the last part of their name (`monty_turn`, `call_tool`)."""
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name.rpartition('__')[2]
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


@pytest.mark.parametrize(
    'agent,prompt,expected_output,expected_activities',
    [
        pytest.param(
            storing_code_mode_agent,
            'Calculate 3 + 4',
            'done: 70',
            # Each nested call is a turn that pauses at it, then one that defers it and waits on it.
            ['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn']
            + ['model_request', 'monty_turn', 'model_request'],
            id='sequential',
        ),
        pytest.param(
            storing_code_mode_agent,
            'Gather two additions',
            # The second snippet reads `result`, which this one never set: a failed turn.
            'done: Runtime error:\nTraceback (most recent call last):\n  File "<python-input-1>", line 1, in <module>\n'
            "    result * 10\n    ~~~~~~\nNameError: name 'result' is not defined",
            ['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn', 'call_tool', 'monty_turn']
            + ['model_request', 'monty_turn', 'model_request'],
            id='gather',
        ),
        pytest.param(
            not_storing_code_mode_agent,
            'Calculate 3 + 4',
            'done: 70',
            # The first turn finds the server stores no sessions before it runs any code.
            ['model_request', 'monty_turn', 'call_tool', 'model_request', 'model_request'],
            id='no_storage',
        ),
    ],
)
async def test_remote_turns_run_as_activities(
    client: Client,
    workflow_id: str,
    agent: Agent[object, str],
    prompt: str,
    expected_output: str,
    expected_activities: list[str],
) -> None:
    """With a server that stores sessions, each remote turn is an activity and the workflow only waits on it.

    A turn that computes for longer than Temporal's 2 second workflow-task limit then stays out of the
    workflow task. In the sequential case the second `run_code` reads `result` from the first, so the
    REPL carries across turns that each start a fresh session from the stored record. With a server that
    stores no sessions, the snippet runs once in the workflow, as before turns were activities.
    """
    output = await _run(client, workflow_id, agent, prompt)
    assert output == expected_output
    history = await _history(client, workflow_id)
    assert _activity_types(history) == expected_activities


@pytest.mark.parametrize(
    'sandbox,expected_activities',
    [
        pytest.param('local', ['model_request', 'call_tool', 'model_request'], id='local'),
        # The turns answer the import, then pause at each call: the module calls run in the workflow
        # between turns, and only the tool call is scheduled as an activity.
        pytest.param(
            'turns',
            ['model_request', *['monty_turn'] * 4, 'call_tool', 'monty_turn', 'model_request'],
            id='turns',
        ),
    ],
)
async def test_external_modules_run_in_the_workflow(
    client: Client, workflow_id: str, sandbox: str, expected_activities: list[str]
) -> None:
    """Module functions run in the workflow and replay with it: only the tool call is an activity."""
    code = (
        'import asyncio, catalog\nfrom catalog import lookup\n'
        '[lookup("region"), await asyncio.gather(catalog.scaled(n=2), echo(value="tool"))]'
    )
    assert await _run(client, workflow_id, module_agents[sandbox], code) == "['eu', [200, 'tool']]"
    assert _activity_types(await _history(client, workflow_id)) == expected_activities


async def test_remote_turns_fail_when_the_server_refuses_to_store(client: Client, workflow_id: str) -> None:
    """A server that refuses to store a session, for example with a full snapshot quota, fails the run.

    A retry would meet the same refusal, so the turn reports it and the run raises `UserError` with
    CodeMode's own text, never the server's.
    """
    SERVER.dumps_left = 0
    with pytest.raises(WorkflowFailureError) as exc_info:
        await _run(client, workflow_id, storing_code_mode_agent)
    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == 'UserError'
    assert cause.message.startswith("The Monty server could not store the session's state")
    history = await _history(client, workflow_id)
    assert _activity_types(history) == ['model_request', 'monty_turn']


@pytest.mark.parametrize('reason', ['overlay', 'write_limit', 'os_access', 'dynamic', 'unpatched'])
async def test_remote_turns_stay_in_the_workflow(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """An `'overlay'` mount (the default mode), a mount with `write_bytes_limit`, `os_access`, a `CodeMode`
    a `DynamicCapability` returns, or a workflow that started before this version of Code Mode keeps
    every turn in the workflow.

    Unsandboxed for `'unpatched'`, so `workflow.patched` can be faked.
    """
    runner: WorkflowRunner | None = None
    if reason == 'unpatched':

        def unpatched(patch_id: str) -> bool:
            return False

        monkeypatch.setattr(workflow, 'patched', unpatched)
        runner = UnsandboxedWorkflowRunner()
    output = await _run(client, workflow_id, workflow_side_code_mode_agents[reason], runner=runner)
    assert output == 'done: 70'
    history = await _history(client, workflow_id)
    assert 'monty_turn' not in _activity_types(history)


async def test_remote_turns_resume_calls_in_place(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calls resolved at the call site resume the stored snapshot with their result, errors included.

    Every snippet returns what it returns on the workflow-side path, which a `CodeMode` without `id`
    takes, except for an argument with no JSON form, which raises `TypeError` at the call site.
    """
    monkeypatch.setattr('pydantic_ai_harness._monty_exec._PRINT_LIMIT_BYTES', 200)
    returns = json.loads(await _run(client, workflow_id, call_site_agents[True]))
    workflow_side_id = f'{workflow_id}_workflow_side'
    workflow_side = json.loads(await _run(client, workflow_side_id, call_site_agents[False]))
    assert 'monty_turn' not in _activity_types(await _history(client, workflow_side_id))

    type_error, prints, added, call_sites, syntax_error, no_json_form, values, not_utf8, sleeps = returns
    assert isinstance(type_error, str) and isinstance(prints, str) and isinstance(sleeps, str)
    assert isinstance(not_utf8, str)
    assert type_error.startswith('Type error in code:\nerror[missing-argument]')
    assert added == 2
    first_line = (Path(__file__).parent / 'stored_sessions.py').read_text().splitlines()[0]
    assert (
        call_sites
        == f'4 | add() does not accept positional arguments; use keyword arguments / NameError | {first_line}'
    )
    assert syntax_error == (
        'Syntax error in code:\nTraceback (most recent call last):\n  File "<python-input-2>", line 1\n'
        '    def broken(:\n               ~\nSyntaxError: Expected a parameter or the end of the parameter list'
    )
    assert no_json_form == [
        "MontyClassProxy(name='P', attributes={'x': 1})",
        'add() got arguments that cannot be sent to the tool: they need types the sandbox can hand the host and at most 524288 bytes',
    ]
    # On the workflow-side path that argument fails the tool's validation, which fails the snippet.
    assert [type_error, added, call_sites, syntax_error, values, not_utf8] == [
        workflow_side[i] for i in (0, 2, 3, 4, 6, 7)
    ]
    # The print cap and `max_suspensions` stop both paths at the same point. The numbers in Monty's
    # message differ: a turn's limits are what the earlier turns left.
    for limited, limited_in_workflow in [(prints, workflow_side[1]), (sleeps, workflow_side[8])]:
        assert isinstance(limited_in_workflow, str)
        assert re.sub(r'\d+', 'N', limited) == re.sub(r'\d+', 'N', limited_in_workflow)
    assert 'MemoryError: memory limit exceeded' in prints
    assert 'RuntimeError: suspension limit 1 exceeded' in sleeps
    assert not_utf8.endswith('ValueError: Error serializing to JSON: invalid utf-8 sequence of 1 bytes from index 0')

    history = await _history(client, workflow_id)
    # The type-checked snippet that fails and the syntax error are one turn each. The mounted file read
    # and the undefined name are answered inside a turn; only `double`, `add`, `blob` and `echo` leave it.
    assert _activity_types(history) == [
        *['model_request', 'monty_turn'],
        # Each sleep ends a turn; the third turn's print goes over the cap.
        *['model_request', 'monty_turn', 'monty_turn', 'monty_turn'],
        *['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn'],
        *['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn'],
        *['model_request', 'monty_turn'],
        *['model_request', 'monty_turn'],
        *['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn'],
        *['monty_turn', 'call_tool', 'monty_turn'],
        # The non-UTF-8 argument fails before the tool's activity starts.
        *['model_request', 'monty_turn', 'monty_turn', 'monty_turn'],
        # Three sleeps use the session's last pauses, and the fourth goes over `max_suspensions`.
        *['model_request', 'monty_turn', 'monty_turn', 'monty_turn', 'monty_turn', 'model_request'],
    ]


async def test_failed_turn_activity_fails_the_workflow(client: Client, workflow_id: str) -> None:
    """A turn activity that fails after its retries fails the run like any other activity, rather than
    raising an error Temporal would treat as a workflow bug and retry forever.

    The recorded error keeps the original type's name, so the retry policy's
    `non_retryable_error_types` still stops retries.
    """
    with pytest.raises(WorkflowFailureError) as exc_info:
        await _run(client, workflow_id, refusing_code_mode_agent)
    activity_error = exc_info.value.cause
    assert isinstance(activity_error, ActivityError)
    assert activity_error.activity_type.endswith('__monty_turn')
    assert isinstance(activity_error.cause, ApplicationError)
    # Only the error's type is recorded: a connection error's text can carry the pool's URL or credentials.
    assert activity_error.cause.type == 'ConnectionRefusedError'
    assert activity_error.cause.message == 'ConnectionRefusedError'
    assert activity_error.cause.non_retryable is False
    assert len(refusing_pool_calls) == 1
    assert activity_error.cause.cause is None
    history = await _history(client, workflow_id)
    assert 'refused the connection' not in history.to_json()


@pytest.mark.parametrize(
    ('prompt', 'reported'),
    [
        ('expire the record', 'The sandbox connection was lost and the session was reset.'),
        ('drop the connection', 'Code execution failed and the session was reset.'),
    ],
)
async def test_remote_turns_report_a_lost_session(client: Client, workflow_id: str, prompt: str, reported: str) -> None:
    """A run whose stored state the server no longer has resets the session and tells the model.

    It does not move to the workflow-side path, which would start an empty REPL without saying so, or
    rerun nested calls the snippet already made. Either way the second `run_code` reads `x` from
    state that is gone. A dropped connection's error text, which can hold credentials, stays out of
    the workflow history, as it stays out of what the model sees.
    """
    output = await _run(client, workflow_id, session_loss_code_mode_agent, prompt)
    first, second = json.loads(output)
    assert first == 1
    assert second.startswith(reported)
    history = await _history(client, workflow_id)
    assert 'secret-token' not in history.to_json()


_NOT_KEPT = 'The sandbox could not keep the session after this error, so it was reset.'


@pytest.mark.parametrize(
    ('dumps_left', 'code', 'expected_activities'),
    [
        pytest.param(0, '1 / 0', ['model_request', 'monty_turn', 'model_request'], id='snippet_error'),
        # The turns before the error are stored; the one the error ends is not.
        pytest.param(
            2,
            'await give_up()',
            ['model_request', 'monty_turn', 'monty_turn', 'call_tool', 'monty_turn', 'model_request'],
            id='tool_failed',
        ),
    ],
)
async def test_remote_turns_reset_a_session_the_server_will_not_store(
    client: Client, workflow_id: str, dumps_left: int, code: str, expected_activities: list[str]
) -> None:
    """A server that refuses to store the session a snippet error or a nested `ToolFailed` left resets it,
    rather than retrying the turn.

    Logfire Agent Sandboxes refused such a dump in a live run, and the retries outlasted the workflow.
    """
    SERVER.dumps_left = dumps_left
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert _NOT_KEPT in output
    history = await _history(client, workflow_id)
    assert _activity_types(history) == expected_activities


async def test_remote_turns_keep_the_session_after_a_syntax_error_the_server_will_not_store(
    client: Client, workflow_id: str
) -> None:
    """A syntax error ran nothing, so the session keeps its stored state even when storing it again fails."""
    SERVER.dumps_left = 1
    # Only a run's first snippet is type-checked, so the second reaches the sandbox's parser.
    output = await _run(client, workflow_id, snippet_agents['default'], f'x = 1{NEXT_SNIPPET}1 +')
    assert output.startswith('Syntax error in code:')
    assert _NOT_KEPT not in output


_CUT = '[the rest is not shown: a turn returns at most {} bytes]'


@pytest.mark.parametrize(
    ('limit', 'code', 'expected'),
    [
        pytest.param(1000, 'print("p" * 3000)\n"r" * 3000', [_CUT.format(1000)] * 2, id='prints_and_result'),
        # An error crosses once as the report and once alone.
        pytest.param(1000, 'raise ValueError("e" * 3000)', ['ValueError', _CUT.format(1000)], id='error'),
        # A result is rendered as text once and that text is cut, so text `repr` escapes and huge integers fit.
        pytest.param(2000, '"\\\\\\n" * 3000', [_CUT.format(2000)], id='escaped_text'),
        pytest.param(2000, '10**5000', ['[a int too large to show]'], id='huge_int'),
    ],
)
async def test_remote_turns_cut_output_to_fit_a_payload(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch, limit: int, code: str, expected: list[str]
) -> None:
    """Prints, a result or an error larger than a turn may return are cut with a note, rather than failing the upload.

    The limit is lowered from 1 MiB so the snippet stays small.
    """
    monkeypatch.setattr('pydantic_ai_harness._monty_turns.TURN_PAYLOAD_BYTES', limit)
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert [text for text in expected if text in output] == expected
    assert output.count(_CUT.format(limit)) == expected.count(_CUT.format(limit))
    assert len(output) < 3000


async def test_remote_turns_charge_cut_prints_to_the_print_cap(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prints a turn cut to fit its payload still count toward the feed's print cap, as in workflow code."""
    monkeypatch.setattr('pydantic_ai_harness._monty_turns.TURN_PAYLOAD_BYTES', 1200)
    monkeypatch.setattr('pydantic_ai_harness._monty_exec._PRINT_LIMIT_BYTES', 2500)
    code = 'import time\nprint("a" * 1500)\ntime.sleep(0)\nprint("b" * 1500)'
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert 'MemoryError' in output


async def test_remote_turns_record_a_pool_user_error_by_type(client: Client, workflow_id: str) -> None:
    """A `UserError` from the `monty_sandbox` factory fails the run unretried, recorded by type only.

    Its text could carry the pool's URL or credentials, and Temporal keeps an activity's failure in history.
    """
    with pytest.raises(WorkflowFailureError) as exc_info:
        await _run(client, workflow_id, snippet_agents['factory_error'], '1')
    activity_error = exc_info.value.cause
    assert isinstance(activity_error, ActivityError)
    assert activity_error.retry_state == RetryState.NON_RETRYABLE_FAILURE
    assert isinstance(activity_error.cause, ApplicationError)
    assert (activity_error.cause.type, activity_error.cause.message) == ('UserError', 'UserError')
    history = await _history(client, workflow_id)
    assert 'secret-sentinel' not in history.to_json()


async def test_remote_turns_count_a_shared_argument_as_it_is_sent(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value passed both positionally and by keyword is sent twice, so the argument limit counts it twice.

    Counted once, the call passed the limit and then made the turn too large to return after the snippet ran.
    """
    monkeypatch.setattr('pydantic_ai_harness._monty_turns.TURN_PAYLOAD_BYTES', 2000)
    code = f'0{NEXT_SNIPPET}s = "x" * 450\ntry:\n    await same(s, second=s)\nexcept TypeError as e:\n    error = str(e)\nerror'
    output = await _run(client, workflow_id, snippet_agents['types'], code)
    assert output == (
        'same() got arguments that cannot be sent to the tool: they need types the sandbox can hand'
        ' the host and at most 1000 bytes'
    )


async def test_remote_turns_keep_the_default_suspension_cap(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `'unlimited'`, Monty's default `max_suspensions` still counts across a session's turns.

    The default is lowered from 1000 so a few sleeps reach it.
    """
    monkeypatch.setattr('pydantic_ai_harness._monty_exec._MONTY_DEFAULT_MAX_SUSPENSIONS', 3)
    code = 'import time\nfor _ in range(5):\n    time.sleep(0)'
    output = await _run(client, workflow_id, snippet_agents['unlimited'], code)
    assert 'suspension limit' in output


with warnings.catch_warnings():
    warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
    legacy_code_mode_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(
            FunctionModel(_code_mode_model),
            name='code_mode_temporal_legacy_agent',
            toolsets=[FunctionToolset(tools=[add], id='math')],
            capabilities=[CodeMode(id='code', monty_sandbox=stored_session_pool)],
        ),
        activity_config=BASE_ACTIVITY_CONFIG,
    )


@workflow.defn
class LegacyWorkflow:
    @workflow.run
    async def run(self) -> str:
        result = await legacy_code_mode_agent.run('Calculate 3 + 4')
        return result.output


async def test_remote_turns_stay_in_the_workflow_under_temporal_agent(client: Client, workflow_id: str) -> None:
    """The deprecated `TemporalAgent` binds no capability operations, so turns stay in the workflow.

    Run as an operation there, a turn would reach the server from the workflow's event loop.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LegacyWorkflow],
        activities=legacy_code_mode_agent.temporal_activities,
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            LegacyWorkflow.run, id=workflow_id, task_queue=TASK_QUEUE, execution_timeout=timedelta(seconds=30)
        )
    assert output == 'done: 70'
    history = await _history(client, workflow_id)
    assert 'monty_turn' not in _activity_types(history)


async def test_remote_turns_refuse_results_too_large_for_one_turn(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Results the sandbox awaits together that exceed a turn reset the session and say how large they were."""
    monkeypatch.setattr('pydantic_ai_harness._monty_turns.TURN_PAYLOAD_BYTES', 1000)
    code = 'import asyncio\nawait asyncio.gather(repeat(text="x", times=600), repeat(text="y", times=600))'
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert 'more than the 1000 it can take' in output
    assert 'so it was reset' in output


async def test_remote_turns_keep_the_session_when_a_snippet_is_too_large_to_send(
    client: Client, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snippet too large to send fails without running, and the session keeps what earlier snippets defined."""
    monkeypatch.setattr('pydantic_ai_harness._monty_turns.TURN_PAYLOAD_BYTES', 3000)
    code = f'x = 41{NEXT_SNIPPET}pad = "{"z" * 4000}"{NEXT_SNIPPET}x + 1'
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert output == '42'


@pytest.mark.parametrize(
    ('code', 'expected'),
    [
        # A retried activity would find the connection back and return `1`.
        pytest.param('1', ['The sandbox connection was lost and the session was reset.'], id='result'),
        # A retried activity would find the connection back and store the session.
        pytest.param('raise ValueError("boom")', ['ValueError: boom', _NOT_KEPT], id='error'),
    ],
)
async def test_remote_turns_do_not_rerun_a_snippet_whose_save_dropped(
    client: Client, workflow_id: str, code: str, expected: list[str]
) -> None:
    """A connection lost while saving a finished turn, or the session a snippet error left, resets the session.

    Retrying the turn instead would run the snippet again.
    """
    SERVER.drops_next_dump = ConnectionResetError('the connection dropped')
    output = await _run(client, workflow_id, snippet_agents['default'], code)
    assert [text for text in expected if text in output] == expected


async def test_remote_turns_report_storage_lost_after_the_first_turn(client: Client, workflow_id: str) -> None:
    """A later turn on a server without storage reports a dropped session instead of rerunning in the workflow."""
    output = await _run(client, workflow_id, snippet_agents['forgetful'], 'await echo(value=1)')
    # Falling back would run the snippet again in workflow code and return its result.
    assert output.startswith('The sandbox connection was lost and the session was reset.')
    assert len(forgetful_pool_calls) == 2


_TYPES_CODE = (
    'import datetime, decimal\n'
    'values = [datetime.date(2026, 11, 19), datetime.datetime(2026, 11, 19, 10, 30, tzinfo=datetime.timezone.utc),\n'
    '    datetime.time(10, 30), datetime.timedelta(days=1, microseconds=5), decimal.Decimal("1.50"), {1}, frozenset({2})]\n'
    'names = await type_names(values=values)\n'
    'back = await echo_here(value=values)\n'
    'f"{names} {[type(v).__name__ for v in back]}"'
)
_VALUES_CODE = (
    'import dataclasses\n'
    'names = await type_names(values=[1j, {1}])\n'
    'try:\n    await frozen()\n    caught = "no"\nexcept dataclasses.FrozenInstanceError:\n    caught = "FrozenInstanceError"\n'
    'members = await enum_members()\n'
    '[names, ..., 1j, caught, members, [type(m).__name__ for m in members]]'
)


@pytest.mark.parametrize(
    ('agent', 'code', 'expected'),
    [
        # Dates, times, durations, decimals and sets reach a tool as themselves, so a `date | str` parameter
        # still validates to `date`. A tool's result reaches the sandbox in its JSON form on both paths.
        pytest.param(
            'types',
            _TYPES_CODE,
            "['date', 'datetime', 'time', 'timedelta', 'Decimal', 'set', 'frozenset'] "
            "['str', 'str', 'str', 'str', 'str', 'list', 'list']",
            id='host_types',
        ),
        # Values JSON cannot carry cross both ways, a tool's error keeps its `dataclasses` type, and enum
        # members reach the sandbox as their plain values, as from a live session.
        pytest.param(
            'types',
            _VALUES_CODE,
            "[['complex', 'set'], 'Ellipsis', '1j', 'FrozenInstanceError', ['red', 3], ['str', 'int']]",
            id='values_json_cannot_carry',
        ),
        # A tool's `csv.Error` reaches the sandbox as `csv.Error`, not as `binascii.Error` or `Exception`.
        pytest.param(
            'default',
            'import csv\ntry:\n    await bad_csv()\n    caught = "no"\nexcept csv.Error:\n    caught = "csv.Error"\ncaught',
            'csv.Error',
            id='module_exception_type',
        ),
        # A `bytearray` result fails as on a live session, which raises it on the host, so the snippet
        # cannot catch it.
        pytest.param(
            'types',
            'try:\n    r = await raw_bytes()\n    got = type(r).__name__\nexcept TypeError as e:\n    got = "TypeError"\ngot',
            'Runtime error:\nTypeError: Cannot convert builtins.bytearray to Monty value'
            ' — wrap class instances in pydantic_monty.ClassInstance(...)',
            id='bytearray_tool_result',
        ),
        # A result too deep to convert returns as a note, rather than failing the turn after the snippet ran.
        pytest.param(
            'default',
            'value = 0\nfor i in range(1050):\n    value = [value]\nvalue',
            '[the result is nested too deeply to show]',
            id='nested_too_deeply',
        ),
    ],
)
async def test_remote_turns_carry_values_as_a_live_session_does(
    client: Client, workflow_id: str, agent: str, code: str, expected: str
) -> None:
    """Values crossing a turn reach tools and the snippet as they do from a live session."""
    output = await _run(client, workflow_id, snippet_agents[agent], code)
    assert output == expected
    history = await _history(client, workflow_id)
    assert 'monty_turn' in _activity_types(history)
