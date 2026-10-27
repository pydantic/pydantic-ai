"""Code mode toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

import inspect
import math
import re
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, TypeGuard
from urllib.parse import urlsplit

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError, to_json, to_jsonable_python
from typing_extensions import TypedDict, TypeIs

from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import (
    AgentStreamEvent,
    InstructionPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnContent,
    ToolReturnPart,
    UserContent,
    is_multi_modal_content,
)
from pydantic_ai.tool_manager import ParallelExecutionMode, ToolManager
from pydantic_ai.tools import AgentDepsT, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool

try:
    from pydantic_monty import (
        AbstractOS,
        AsyncMontyWebsocket,
        MontyCrashedError,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        MountDir,
        OsFunction,
        OsHandler,
        ResourceLimits,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for CodeMode. Install it with: pip install "pydantic-ai-harness[code-mode]"'
    ) from _import_error
from pydantic_ai_harness._code_tools import (
    NestedCallOutcome,
    can_be_sandboxed,
    contains_multimodal,
    partition_callable_tools,
    record_nested_call,
    run_nested_call,
    sanitize_tool_name,
    signatures_and_conflicts,
)
from pydantic_ai_harness._monty_exec import (
    MontyExecutor,
    MontyRunState,
    PrintCapture,
    in_temporal_workflow,
    is_sandbox_panic,
)
from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_argument_renamed
from pydantic_ai_harness.code_mode._parts import RUN_CODE_KIND, RUN_CODE_TOOL_NAME, RunCodeArgs
from pydantic_ai_harness.code_mode._repl_store import CodeModeReplStore, PersistentRepl

if TYPE_CHECKING:
    from pydantic_ai_harness.code_mode._speculation import SpeculationCoordinator

# Deprecated: a positional `(name, args, kwargs)` OS callback. Pass Monty's keyword-only `OsHandler`
# instead; this form is still accepted and will be removed in the next breaking release.
CodeModeOSCallback = Callable[[OsFunction, tuple[object, ...], dict[str, object]], object]
# Accepted by `CodeMode.os_access`: a ready-made OS implementation or a handler that decides each call.
CodeModeOS = AbstractOS | OsHandler | CodeModeOSCallback
# Accepted by `CodeMode.mount`: one or more host-directory mounts.
CodeModeMount = MountDir | list[MountDir]

# Bounds for the nested-call summary appended to a budget-exhaustion retry. Monty caps printed
# output at 10 MiB by raising, which suits a stream the model asked for but not a summary the
# host adds to an error, so these are separate and much smaller: the summary exists to identify
# calls, not to redeliver their payloads.
_RETRY_VALUE_PREVIEW_CHARS = 120
_RETRY_PREVIEW_ITEMS = 5
_RETRY_SUMMARY_MAX_CHARS = 2000


# One entry per limit `CodeModeResourceLimits` exposes, mapped to the wording Monty reports when
# it trips. Monty offers no typed marker for either, so its phrasing is load-bearing here and
# nowhere else. `test_every_resource_limit_reports_started_calls_when_exhausted` exhausts each
# option the type declares and checks the summary survives, so a limit added without an entry here
# fails there rather than silently losing its summary, and a Monty reword fails it rather than
# quietly disabling recognition.
_SANDBOX_LIMIT_MARKERS = {
    'max_duration_secs': 'time limit exceeded',
    'max_memory': 'memory limit exceeded',
    'max_suspensions': 'suspension limit ',
}


def _check_monty_sandbox_url(url: str) -> None:
    """Reject URLs Monty's WebSocket client cannot dial, before the first `run_code` call fails on them."""
    scheme = urlsplit(url).scheme
    if scheme not in ('ws', 'wss'):
        raise UserError(f'`monty_sandbox` must be a `ws://` or `wss://` URL, not scheme {scheme!r}.')


def monty_sandbox_from_url(
    monty_sandbox: str | Callable[[], AsyncMontyWebsocket | Awaitable[AsyncMontyWebsocket]] | None,
    monty_sandbox_url: str | None,
    owner: str,
) -> str | Callable[[], AsyncMontyWebsocket | Awaitable[AsyncMontyWebsocket]] | None:
    """The `monty_sandbox` to use, moving the deprecated `monty_sandbox_url` into it."""
    if monty_sandbox_url is None:
        return monty_sandbox
    if monty_sandbox is not None:
        raise UserError('Pass `monty_sandbox` only; `monty_sandbox_url` is its deprecated form.')
    warn_argument_renamed(owner, 'monty_sandbox_url', 'monty_sandbox', stacklevel=5)
    return monty_sandbox_url


def _is_os_handler(os_access: CodeModeOS) -> TypeIs[AbstractOS | OsHandler]:
    """Whether `os_access` takes Monty's keyword call, rather than the deprecated positional one.

    Only a callable that takes `(name, args, kwargs)` positionally and cannot take Monty's keyword
    call counts as positional. Anything else is passed to Monty unchanged, so a malformed handler
    gets Monty's own error rather than a deprecation warning.
    """
    if isinstance(os_access, AbstractOS):
        return True
    try:
        signature = inspect.signature(os_access)
    except (TypeError, ValueError):  # pragma: no cover - builtins and some C callables have no signature
        return True
    return _binds(signature, name='os.getenv', args=(), kwargs={}, is_async=False) or not _binds(
        signature, 'os.getenv', (), {}
    )


def _binds(signature: inspect.Signature, *args: object, **kwargs: object) -> bool:
    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def as_os_handler(os_access: CodeModeOS | None) -> AbstractOS | OsHandler | None:
    """Return `os_access` in the keyword-only shape Monty calls, warning once for the positional form.

    Called from the `__post_init__` of the dataclasses that take `os_access`, which store the result:
    their per-run copies re-run `__post_init__` and then see a handler that needs no warning.
    """
    if os_access is None or _is_os_handler(os_access):
        return os_access
    warnings.warn(
        'A positional `os_access(name, args, kwargs)` callback is deprecated. Accept keyword arguments '
        'instead, as `pydantic_monty.OsHandler` does: `def handler(*, name, args, kwargs, **_): ...`. '
        'The positional form will be removed in the next breaking release.',
        category=HarnessDeprecationWarning,
        stacklevel=4,  # this function, `__post_init__`, the dataclass `__init__`, then the caller
    )
    callback: Callable[..., object] = os_access

    def handler(*, name: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any], **_: Any) -> object:
        return callback(name, args, kwargs)

    return handler


def in_durable_execution(ctx: RunContext[object]) -> bool:
    """Whether a durable executor is active, where streamed execution tiers must stay disabled."""
    return any(
        isinstance(capability, BaseDurabilityCapability) and capability.in_durable_context
        for capability in ctx.capabilities.values()
    )


def _exhausted_sandbox_limit(error: MontyRuntimeError) -> str | None:
    """Which `CodeModeResourceLimits` limit this runtime error reports, or `None` for anything else.

    Derived from `_SANDBOX_LIMIT_MARKERS` rather than from a check per limit, so recognising a new
    limit is a table entry and forgetting one is a test failure.

    This gates the started-call summary, so it deliberately errs toward inclusion and matches on
    Monty's wording alone. A nested tool that fails with one of these phrases in its own message is
    misread, and that costs nothing: the summary only states which calls really started, which is
    true regardless of why the snippet ended. The session reset cannot afford the same looseness
    and uses `_is_duration_exhausted` instead.
    """
    message = error.display(format='msg')
    for limit, marker in _SANDBOX_LIMIT_MARKERS.items():
        if marker in message:
            return limit
    return None


def _is_duration_exhausted(error: MontyRuntimeError) -> bool:
    """Whether this runtime error is Monty stopping the snippet at `max_duration_secs`.

    Stricter than `_exhausted_sandbox_limit` because it gates a session reset, and a wrong reset
    discards REPL state the session could still use.

    The empty traceback is the structural signal: the duration limit interrupts execution rather
    than failing at a particular operation, and Monty attaches no frame to it, measured at top
    level and three calls deep alike. Failures that happen at a sandbox site carry at least the
    module frame, including an exception a nested tool raised that Monty re-raised at the call
    site, which keeps the tool's own message. Wording alone would therefore misread a tool that
    failed with `'time limit exceeded'` in its message.

    Exceeding `max_memory` is excluded by either signal, since it reports different wording and
    carries a frame from the allocation that tripped it. Keeping both means neither has to be
    sound alone.

    Callers must read `False` as "not known to be a timeout". A miss keeps the session and the
    ordinary runtime-error message.
    """
    return not error.traceback() and _exhausted_sandbox_limit(error) == 'max_duration_secs'


def _elided(count: int, shown: int, unit: str) -> str:
    """Note how much a preview left out, or nothing when it left out nothing."""
    return f' ... ({count} {unit} total)' if count > shown else ''


def _preview(value: Any, *, nested: bool = False) -> str:
    """Render a value for an error message, cutting it before rendering rather than after.

    Rendering first and slicing after would copy the whole payload to produce 120 characters: a
    20 MB tool result cost 40 MB of allocation that way, spent while the host is already handling
    a resource failure. So each shape is cut at the source instead.

    Only the shapes a tool result or argument can take are rendered: text, bytes, and containers
    of those, one level deep. A nested container is reported by size rather than expanded, which
    bounds the work without recursing to arbitrary depth. Anything else is named by type, since
    calling `repr` on it is the unbounded allocation this exists to avoid -- a `BinaryContent`
    result would otherwise render its entire payload.
    """
    limit = _RETRY_VALUE_PREVIEW_CHARS
    items = _RETRY_PREVIEW_ITEMS
    # `isinstance` on a bare `list`/`dict` narrows to an unparameterized generic, which reads as
    # partially unknown under strict typing. Keeping an unnarrowed alias lets the checks stay
    # `isinstance`, so subclasses still match, while the element access stays typed.
    raw: Any = value
    if isinstance(value, str):
        return repr(value[:limit]) + _elided(len(value), limit, 'chars')
    if isinstance(value, (bytes, bytearray)):
        return repr(bytes(value[:limit])) + _elided(len(value), limit, 'bytes')
    if isinstance(value, (list, tuple)):
        if nested:
            return f'[{len(raw)} items]'
        rendered = ', '.join(_preview(item, nested=True) for item in raw[:items])
        return f'[{rendered}]' + _elided(len(raw), items, 'items')
    if isinstance(value, dict):
        if nested:
            return f'{{{len(raw)} items}}'
        rendered = ', '.join(
            f'{_preview(key, nested=True)}: {_preview(item, nested=True)}' for key, item in islice(raw.items(), items)
        )
        return '{' + rendered + '}' + _elided(len(raw), items, 'items')
    if value is None or isinstance(value, (int, float)):
        return repr(value)
    return f'<{type(value).__name__}>'


def _describe_started_calls(calls: dict[str, ToolCallPart], returns: dict[str, ToolReturnPart]) -> str:
    """Report how many nested calls started, with per-call detail inside a size cap.

    Keyed off `calls` rather than `returns` because a call that raised has no recorded return, and
    a tool can commit a side effect before raising. Those are the calls most likely to have left
    partial state, so omitting them by kind would hide exactly what the model needs to check.

    The per-call lines are bounded, so a long run drops the tail and says how many it dropped. The
    total is always exact: it is the part that survives truncation, and it is what tells the model
    the list it can see is incomplete.
    """
    lines: list[str] = []
    used = 0
    for call_id, call in calls.items():
        result = returns.get(call_id)
        # `ToolReturnPart.outcome` also allows 'failed', 'interrupted' and 'retried', but this
        # function only ever sees parts built above, which set 'denied' or leave the default
        # 'success'. Record any further outcome here rather than letting it fall through and read
        # as a return.
        if result is None:
            outcome = 'did not finish, so it may have applied a partial change'
        elif result.outcome == 'denied':
            outcome = 'was denied and did not run'
        else:
            outcome = f'returned {_preview(result.content)}'
        line = f'- {call.tool_name}({_preview(call.args)}) {outcome}'
        if used + len(line) > _RETRY_SUMMARY_MAX_CHARS:
            lines.append(f'- ... and {len(calls) - len(lines)} more not shown')
            break
        lines.append(line)
        used += len(line)
    return (
        f'{len(calls)} nested tool calls started before execution stopped:\n'
        + '\n'.join(lines)
        + f'\nAccount for all {len(calls)} before retrying; repeating a call repeats whatever it already did.'
    )


class CodeModeResourceLimits(TypedDict, total=False):
    """Caps on the sandbox code executed by `run_code`."""

    max_duration_secs: float
    """Sandbox execution time allowed to each `run_code` snippet; time awaiting tools does not count.

    Sleeping is not execution time, so each snippet may also sleep for up to this long in total.
    """
    max_memory: int
    max_suspensions: int
    """Cumulative host-interaction budget per session, not a per-snippet tool-call count.

    External calls, OS callbacks, name lookups and future resolutions consume this budget.
    Omission keeps Monty's finite default of 1,000; it cannot be disabled.
    """


def _resolve_resource_limits(limits: CodeModeResourceLimits | Literal['unlimited'] | None) -> ResourceLimits:
    """Merge caller overrides onto the `run_code` backstop."""
    if limits == 'unlimited':
        return {}
    if limits is not None:
        unknown = set(limits) - set(CodeModeResourceLimits.__annotations__)
        if unknown:
            raise UserError(
                f'Unknown `resource_limits` key(s): {sorted(unknown)}. '
                f'Valid keys are {sorted(CodeModeResourceLimits.__annotations__)}.'
            )
    # Read as `object` so values from untyped code are checked rather than trusted.
    max_duration_secs: object = 30 if limits is None else limits.get('max_duration_secs', 30)
    max_memory: object = 256 * 1024 * 1024 if limits is None else limits.get('max_memory', 256 * 1024 * 1024)
    max_suspensions: object = 1000 if limits is None else limits.get('max_suspensions', 1000)
    # Monty stores each limit as an unsigned 64-bit integer, or seconds for the duration; anything else fails
    # every snippet. An explicit `None` for memory or duration keeps reaching Monty, which reads it as no limit.
    if not (type(max_suspensions) is int and 1 <= max_suspensions < 2**64):
        raise UserError('`max_suspensions` must be an integer of at least 1 and below 2**64')
    if not (
        max_duration_secs is None
        or (
            (type(max_duration_secs) is int or isinstance(max_duration_secs, float)) and 0 < max_duration_secs < 2.0**64
        )
    ):
        raise UserError('`max_duration_secs` must be a positive number of seconds below 2**64')
    if not (max_memory is None or (type(max_memory) is int and 0 < max_memory < 2**64)):
        raise UserError('`max_memory` must be a positive integer number of bytes below 2**64')
    return {
        'max_feed_duration_secs': max_duration_secs,
        'max_memory': max_memory,
        'max_suspensions': max_suspensions,
    }


class _RunCodeArguments(RunCodeArgs):
    # Keeps the schema title the model has always been sent; the fields come from `RunCodeArgs`.
    pass


_CODE_MODE_TOOLSET_ID = 'code_mode'
_CATALOG_INSTRUCTION_NAME = 'catalog'
_TOOL_SEARCH_NATIVE_ID = 'tool_search'
"""The `with_native` value that marks a member of the native tool search's corpus (`ToolSearchTool.kind`)."""


_Placement: TypeAlias = Literal['native', 'sandboxed', 'hidden']
"""Where `CodeModeToolset.get_tools` puts a wrapped tool: on the wire, inside `run_code`, or nowhere yet."""


def _place_hidden_tools(placements: dict[str, _Placement], tools: Mapping[str, ToolsetTool[Any]]) -> set[str]:
    """Decide which not-yet-revealed tools `run_code` will take over stay off the wire until they're revealed.

    With `dynamic_catalog`, a deferred tool that will be sandboxed once revealed is never declared natively:
    it's found through the local `search_tools` function or by loading its capability, and from then on
    appears only in the catalog, whose change is appended to history. Declaring it natively first and
    dropping the declaration on reveal would change the request's tools prefix.

    Without a local `search_tools` (the `'bm25'` and `'regex'` strategies), only the native tool search can
    find a corpus member, so a hidden one is declared for it.

    Returns the corpus members that stay native but are searched locally instead: when the selector sandboxes
    some of the Tool Search corpus and keeps the rest native, the native search would take `search_tools` off
    the wire and could only find the tools it declares. Taking the native ones out of its corpus drops the
    native search instead, so `search_tools` finds every corpus member, and a native one is revealed by
    appending it like any locally found tool. The answer depends only on the selector, not on what's been
    revealed, so it holds for the whole run.
    """
    corpus = [name for name, tool in tools.items() if tool.tool_def.with_native == _TOOL_SEARCH_NATIVE_ID]
    local_search = any(
        tool.tool_def.tool_kind == 'tool-search' and placements[name] == 'native' for name, tool in tools.items()
    )
    if not local_search:
        placements.update({name: 'native' for name in corpus if placements[name] == 'hidden'})
        return set()
    native = {name for name in corpus if placements[name] == 'native'}
    sandboxes_corpus = any(placements[name] != 'native' for name in corpus)
    return native if sandboxes_corpus else set()


_RUN_CODE_ADAPTER = TypeAdapter(_RunCodeArguments)
_RUN_CODE_JSON_SCHEMA = _RUN_CODE_ADAPTER.json_schema()
_RUN_CODE_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_CODE_ADAPTER.validator  # pyright: ignore[reportAssignmentType]
# Used to serialize tool return values before sending into Monty (dump_python)
# and to reconstruct multimodal types (e.g. BinaryContent) from Monty results (validate_python).
_TOOL_RETURN_CONTENT_TA: TypeAdapter[Any] = TypeAdapter(ToolReturnContent)

# Values Monty holds as-is. `bytes` is here because Monty carries binary payloads
# natively and JSON would utf-8 decode them, which arbitrary bytes fail. `Ellipsis`
# is here because Monty holds it and JSON has no form for it at all.
_SANDBOX_NATIVE_SCALARS = (str, bytes, bytearray, bool, int, float, type(None), type(Ellipsis))


def _jsonable_key(key: Any) -> Any:
    """Render one mapping key the way `to_jsonable_python` renders JSON object keys.

    JSON object keys are always strings, so `_build_type_check_stubs` declares every
    mapping as `dict[str, ...]` whatever the Python key type is. Left alone, a `Decimal`
    key is rejected by Monty and an `int` key silently contradicts that stub, so a
    snippet indexing with the declared `str` raises `KeyError` at runtime.
    """
    (jsonable_key,) = to_jsonable_python({key: None})
    return jsonable_key


def _jsonable_for_sandbox(value: Any) -> Any:
    """Render the leaves Monty cannot hold as the JSON values the stubs describe.

    `_build_type_check_stubs` derives each stub from the tool's JSON schema, so a
    `Decimal`, `UUID` or `datetime` field is declared `str` there. A Python-mode dump
    keeps the original objects instead: Monty rejects `Decimal` and `UUID` outright,
    and a `datetime` arrives where the stub promised a `str`, so the type check passes
    and the snippet fails at runtime.

    `bytes` and `bytearray` are the deliberate exception: they cross as themselves
    rather than as the `str` their stub declares, because Monty carries binary natively
    and encoding it would change what every binary payload looks like inside the sandbox.
    """
    if isinstance(value, _SANDBOX_NATIVE_SCALARS):
        return value
    if isinstance(value, Mapping):
        jsonable: dict[Any, Any] = {}
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            jsonable_key = _jsonable_key(key)
            if jsonable_key in jsonable:
                raise UserError(
                    f'A tool returned a mapping where key {key!r} renders as the JSON key '
                    f'{jsonable_key!r}, which an earlier key already produced. The sandbox holds '
                    'one entry per JSON key, so one of the two values would be dropped.'
                )
            jsonable[jsonable_key] = _jsonable_for_sandbox(item)
        return jsonable
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable_for_sandbox(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return to_jsonable_python(value)


_RUN_CODE_DESCRIPTION_HEAD = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No third-party libraries**: only the standard library modules listed below can be used
- **Importable standard library modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `unicodedata`, `datetime`, `time`, `random`, `os`, `pathlib`, `collections`, `itertools`, `functools`, `dataclasses`, `copy`, `base64`, `binascii`, `bisect`, `csv`, `decimal`, `hashlib`, `os.path`, `tomllib`. These must be imported before use, just like in regular Python."""

# Timing/OS restriction line, swapped depending on what host access the agent
# configured. Three states, because `mount` and `os` enable different things:
# a `mount` only exposes filesystem paths, while environment and clock calls
# require an `os` handler.
_NO_OS_RESTRICTION = (
    '- **No filesystem, environment, or clock**: `pathlib.Path` I/O, `os.getenv`/`os.environ`, '
    '`datetime.datetime.now()`, `datetime.date.today()`, and `time.time()` are unavailable here '
    '(no filesystem mount or OS handler is configured). `os` and `pathlib` import successfully, but '
    'their I/O operations are not supported in this configuration. `time.sleep` and `asyncio.sleep` really wait.'
)
_MOUNT_ONLY_NOTE = (
    '- **Mounted filesystem access**: `pathlib.Path` operations under the configured mount '
    'point(s) are routed to the host. `os.getenv`/`os.environ`, `datetime.datetime.now()`, '
    '`datetime.date.today()`, and `time.time()` remain unavailable. `time.sleep` and `asyncio.sleep` really wait.'
)
_OS_ENABLED_NOTE = (
    '- **Configured OS access**: `pathlib.Path` operations, `os.getenv`/`os.environ`, '
    '`datetime.datetime.now()`, `datetime.date.today()`, and `time.time()` are routed to the OS '
    'handler configured for this agent (availability depends on that configuration). '
    '`time.sleep` and `asyncio.sleep` really wait.'
)
_MOUNT_LIFETIME_NOTE = (
    "- **Mount write lifetime**: writes through a `mode='overlay'` mount last only for the current "
    "`run_code` call. Use `mode='read-write'` when later calls need to read those writes."
)

_RUN_CODE_DESCRIPTION_TAIL = """\
- **No `import *`**: wildcard imports are not supported

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

The last expression's value is automatically captured as the return value -- you do **not** need to \
`print()` it. End the snippet with the value to return as a bare expression. A final assignment stores \
the value but does not return it. A final expression that evaluates to `None` is treated as no result. \
Without a non-`None` final expression or print output, `run_code` returns `{}`. For example:

```python
result = some_expression
result
```

Avoid `print()` for return values as it produces Python string representations, not structured data. \
Use `print()` only for supplementary logging or debug output.

Returns a non-`None` last expression's value directly when nothing is printed. With `print()` output \
and no non-`None` final expression, returns `{"output": "<printed text>"}`. With `print()` output and a \
plain, non-`None` final expression, returns \
`{"output": "<printed text>", "result": <last expression>}`. With `print()` output and a multimodal \
final expression, returns a list with the printed text followed by the native content.\
"""


def _base_description(*, has_os: bool, has_mount: bool) -> str:
    """Assemble the `run_code` base description with the right OS-access restriction line.

    `os` routes environment, clock, and filesystem calls; a `mount` alone only
    exposes filesystem paths, so a mount-only sandbox must not advertise env or
    clock access (the model would generate calls that fail and burn retries).
    """
    if has_os:
        restriction = _OS_ENABLED_NOTE
    elif has_mount:
        restriction = _MOUNT_ONLY_NOTE
    else:
        restriction = _NO_OS_RESTRICTION
    if has_mount:
        restriction = f'{restriction}\n{_MOUNT_LIFETIME_NOTE}'
    return f'{_RUN_CODE_DESCRIPTION_HEAD}\n{restriction}\n{_RUN_CODE_DESCRIPTION_TAIL}'


def _functions_header(*, has_sync: bool, has_async: bool) -> str:
    """Build the functions-header paragraph for the `run_code` tool description."""
    if has_sync and not has_async:
        return (
            '\nThe following functions are available inside the sandbox. Call them directly '
            '(do **not** redefine or import them). All parameters are keyword-only.'
            ' All tool functions are synchronous: call them directly, e.g. `tool_name(arg=value)`.'
        )
    base = (
        '\nThe following functions are available inside the sandbox. These functions are already defined'
        ' in the sandbox; call them by name without redefining or importing them.'
        ' All parameters are keyword-only.'
    )
    if has_async and not has_sync:
        return base + (
            ' All tool functions are async: invoke them with `await`,'
            ' e.g. `await tool_name(arg=value)`.'
            ' Calling without `await` returns an unresolved future, not the value.'
            # A separate paragraph that leads with the import: next to "without importing them",
            # models drop `import asyncio` and the first snippet fails the type check.
            '\n\nTo run calls concurrently, first `import asyncio`, then use `await asyncio.gather(...)`'
            ' with positional awaitables.'
            ' Monty does not support `asyncio.gather` keyword arguments or other task creation'
            ' and wait APIs.'
        )
    return base + (
        ' Async functions (`async def`) must be invoked with `await`,'
        ' e.g. `await tool_name(arg=value)`.'
        ' Sync functions (`def`) are called directly, e.g. `tool_name(arg=value)`.'
        '\n\nTo run async calls concurrently, first `import asyncio`, then use `await asyncio.gather(...)`'
        ' with positional awaitables.'
        ' Monty does not support `asyncio.gather` keyword arguments or other task creation'
        ' and wait APIs.'
    )


_SEARCH_TOOLS_MODIFIER = (
    ' Note: discovered tools become callable as functions inside the run_code sandbox in subsequent invocations.'
)


def _tool_search_addendum(search_tool_name: str) -> str:
    return (
        f'\n\nNot all functions may be available initially.'
        f' Use the `{search_tool_name}` tool to discover additional functions'
        f' that will become callable in subsequent `run_code` invocations.'
    )


class CodeModeReturnSchemaWarning(UserWarning):
    """A sandboxed tool has no return schema, so its generated signature shows `-> Any`.

    The model then writes code against a result shape it has to guess. A function tool gets a
    return schema from its return annotation; an MCP tool gets one when its server declares an
    `outputSchema`. When the tools come from a server you do not control, silence this category
    alone with `warnings.filterwarnings('ignore', category=CodeModeReturnSchemaWarning)`.
    """


def _warn_missing_return_schemas(names: Sequence[str]) -> None:
    """Warn once for every tool whose sandbox signature will show `-> Any`.

    MCP servers commonly omit output schemas, so the tools are named in one warning rather
    than one warning each.
    """
    if not names:
        return
    if len(names) == 1:
        message = f'CodeMode: tool {names[0]!r} has no return schema; its signature will show `-> Any`'
    else:
        listed = ', '.join(repr(name) for name in names)
        message = f'CodeMode: {len(names)} tools have no return schema ({listed}); their signatures will show `-> Any`'
    warnings.warn(
        f'{message}, which may reduce code mode effectiveness. Add a return annotation to a function tool, '
        'or an `outputSchema` to an MCP tool; to silence this, filter `CodeModeReturnSchemaWarning`.',
        CodeModeReturnSchemaWarning,
        stacklevel=3,
    )


def global_mode_is_sequential(get_mode: Callable[[], ParallelExecutionMode]) -> bool:
    """Whether the run-scoped execution mode forces sandbox tool calls to run sequentially.

    Only `'sequential'` does. `'parallel_ordered_events'` runs top-level tool calls concurrently and
    orders only their result events, which nested sandbox calls do not emit, so it dispatches them
    concurrently like `'parallel'`. `DBOSDurability` defaults to that mode, so serializing there
    would make every `asyncio.gather` in a DBOS run cost the sum of its calls' latencies.
    """
    return get_mode() == 'sequential'


@dataclass(kw_only=True)
class _RunCodeTool(ToolsetTool[AgentDepsT]):
    """ToolsetTool subclass that caches data computed during `get_tools`.

    Avoids a redundant `get_tools` call in `call_tool` by storing the
    callable tool definitions and name mapping on the tool instance itself.
    Follows the same pattern as `_SearchTool` in pydantic-ai's
    `ToolSearchToolset`.
    """

    callable_defs: dict[str, ToolDefinition]
    """Tool definitions callable from inside the sandbox, keyed by (possibly sanitized) name."""

    sanitized_to_original: dict[str, str]
    """Maps sanitized Python-safe names back to original tool names (only for renamed tools)."""

    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    """The wrapped toolset's tools, keyed by original name."""


@dataclass
class RunCodeExecution:
    """Mutable state for one model-visible `run_code` call.

    Normal execution uses one feed. Eager execution shares this object across several feeds so
    nested-call IDs, budgets, output, and metadata do not reset between statements.
    """

    parent_tool_call_id: str
    capture: PrintCapture = field(default_factory=PrintCapture)
    call_count: int = 0
    budget_exhausted: bool = False
    nested_calls: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    nested_returns: dict[str, ToolReturnPart] = field(default_factory=dict[str, ToolReturnPart])
    # Nested `ToolReturn.content` never enters the sandbox; it reaches the model on the `run_code` return.
    # Each call reserves its slot when it starts, so concurrent calls keep call order, as core does.
    user_content: dict[str, list[UserContent]] = field(default_factory=dict[str, list[UserContent]])
    # Snippet code can catch the error at the call site, so the feed re-raises it once it stops.
    usage_limit_exceeded: UsageLimitExceeded | None = None

    def next_tool_call_id(self, *, max_tool_calls: int) -> str:
        """Reserve nested-call budget and return the next stable child call ID."""
        if self.call_count >= max_tool_calls:
            self.budget_exhausted = True
            raise RuntimeError(
                f'Code mode allows {max_tool_calls} nested tool calls per `run_code` call '
                'and this snippet asked for more. Call fewer tools, for example by filtering '
                'the inputs first, or split the work across several `run_code` calls.'
            )
        self.call_count += 1
        tool_call_id = f'{self.parent_tool_call_id}__{self.call_count}'
        self.user_content[tool_call_id] = []
        return tool_call_id

    def finish(self, call_part: ToolCallPart, outcome: NestedCallOutcome) -> Any:
        """Record a nested call's outcome in history, then hand the sandbox its value or its error."""
        record_nested_call(
            call_part, outcome, calls=self.nested_calls, returns=self.nested_returns, user_content=self.user_content
        )
        if outcome.error is not None:
            if outcome.usage_limit_exceeded is not None:
                self.usage_limit_exceeded = outcome.usage_limit_exceeded
            raise outcome.error
        # Serialize to JSON-compatible form so Monty receives only plain data.
        return _jsonable_for_sandbox(_TOOL_RETURN_CONTENT_TA.dump_python(outcome.content, warnings=False))

    def build_tool_return(self, result: Any) -> ToolReturn[Any]:
        """Build the single public result for the logical `run_code` call."""
        output = self.capture.joined
        if not output:
            return_value: Any = result if result is not None else {}
        elif result is None:
            return_value = {'output': output}
        elif contains_multimodal(result):
            return_value = [output, *result] if isinstance(result, list) else [output, result]
        else:
            return_value = {'output': output, 'result': result}

        return ToolReturn(
            return_value=return_value,
            content=[item for items in self.user_content.values() for item in items] or None,
            metadata={
                'code_mode': True,
                'tool_calls': self.nested_calls,
                'tool_returns': self.nested_returns,
            },
        )


@dataclass
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the `CodeMode` capability.

    Exposes a single `run_code` tool alongside any native (non-sandboxed) tools.
    Tools selected by `tool_selector` are presented to the model as Python
    function signatures inside the `run_code` tool description and become
    callable from the sandbox at runtime. Non-selected tools remain visible
    to the model as normal tool calls.

    Some tools always stay native rather than being sandboxed:

    - Framework control tools (`tool_kind` of `'tool-search'` or `'capability-load'`).
    - `defer_loading=True` tools, until tool search or capability loading reveals them.
    - `unless_native` tools, so `Model.prepare_request` can drop them when the
      provider supports the native tool.

    To keep a Tool Search corpus native even after discovery (e.g. for prompt-cache
    stability), pass a `tool_selector` that excludes tools with `with_native` set.
    """

    tool_selector: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to sandbox inside `run_code`. Non-matching tools
    are exposed as native tools."""

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    # Keyword-only: `os_access`, `mount`, and `dynamic_catalog` shipped as positional parameters,
    # so inserting these into the positional sequence would silently rebind existing callers'
    # arguments (an `OSAccess` passed fourth would land in `max_tool_calls`).
    max_tool_calls: int = field(default=100, kw_only=True)
    """Maximum nested tool calls dispatched by one `run_code` invocation.

    Budget is reserved before each call is scheduled, so a snippet cannot allocate host tasks
    beyond this many. Calls past the budget are refused at the sandbox call site.
    """

    resource_limits: CodeModeResourceLimits | Literal['unlimited'] | None = field(default=None, kw_only=True)
    """Sandbox execution limits.

    `None` applies a 30-second execution and 256 MiB heap backstop. `max_duration_secs` is per
    snippet: no single `run_code` snippet runs longer than it, and it is not a run-wide budget.
    `'unlimited'` removes the time and memory caps, but Monty's finite suspension budget still
    applies. Set `max_suspensions` to bound cumulative host interactions across consecutive snippets.
    """

    os_access: CodeModeOS | None = None
    """Give sandboxed code environment variables, the clock, and file I/O through a handler you provide; unset, they are unavailable."""

    mount: CodeModeMount | None = None
    """Host directories to expose to sandboxed `pathlib` code; each mount's `mode` controls whether writes reach the host."""

    monty_sandbox: str | Callable[[], AsyncMontyWebsocket | Awaitable[AsyncMontyWebsocket]] | None = field(
        default=None, kw_only=True, repr=False
    )
    """Where sandboxed code runs: a `ws://` or `wss://` URL, which connects without a key, or a sync or async function that returns a `pydantic_monty.AsyncMontyWebsocket`."""

    monty_sandbox_url: str | None = field(default=None, kw_only=True, repr=False)
    """Deprecated: pass `monty_sandbox=url` instead."""

    dynamic_catalog: bool = False
    """Move the sandboxed-tool catalog out of `run_code.description` and into instructions.

    When `False` (default), every sandboxed tool's signature is rendered into the
    `run_code` description, which lives in the prompt-cache-keyed tool-definitions block.
    When `True`, the description keeps only the static base prose and the catalog is
    surfaced as a dynamic [`InstructionPart`][pydantic_ai.messages.InstructionPart] via
    [`get_instructions`][pydantic_ai_harness.code_mode.CodeModeToolset.get_instructions],
    with the id `toolset:code_mode:catalog` and `on_change='append'`, so a change to it is
    appended to the conversation instead of rewriting the instructions. Deferred tools the
    selector sandboxes are then kept off the wire until they're revealed, and from then on are
    callable only inside `run_code`, so a discovery doesn't change the request's tools either.

    The [`CodeMode`][pydantic_ai_harness.CodeMode] capability defaults to `True`; this toolset
    deliberately keeps `False` for direct construction, so a hand-built toolset keeps its behavior.
    """

    repl_store: CodeModeReplStore[AgentDepsT] | None = field(default=None, kw_only=True, repr=False)
    """Keep the `run_code` REPL across agent runs in this store; unset, each run starts a fresh REPL.

    Not used under durable execution.
    """

    capability: AbstractCapability[AgentDepsT] | None = field(default=None, kw_only=True, repr=False)
    """The run's `CodeMode` instance, when this toolset was built by one.

    `run_code` is attributed to it the way a capability's own toolset tools are, so capability
    events emitted from inside the sandbox dispatch carry the right owner.
    """

    speculation: SpeculationCoordinator[AgentDepsT] | None = field(default=None, kw_only=True, repr=False)
    """Per-run claim store for calls launched while `run_code` arguments stream.

    `CodeMode` creates one per run when `speculate` is set. The stream watcher launches calls
    into it and the dispatch path below claims them; `for_run_step` copies share it by reference.
    """

    # Shared by `for_run_step` copies so they use the same REPL session and the original entered
    # instance can close it. `for_run` leaves this unset, giving concurrent runs isolated state.
    _run_state: MontyRunState | None = field(default=None, init=False, repr=False, compare=False)

    # Shared with `_run_state`: created on entry when `repl_store` is set, saved on exit.
    _repl: PersistentRepl[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)

    # Catalog string stashed during `get_tools` (when `dynamic_catalog`) and read back by
    # `get_instructions` in the same step. Empty when there's nothing to surface.
    _last_catalog: str = field(default='', init=False, repr=False)

    # Tools already named in a `CodeModeReturnSchemaWarning`, so the warning is not repeated every
    # step. Reset on `for_run` because each run gets a fresh instance.
    _warned_missing_return_schema: set[str] = field(default_factory=set[str], init=False, repr=False)

    def __post_init__(self) -> None:
        # Converted once here, so the copies `for_run` and `for_run_step` make do not warn again.
        self.os_access = as_os_handler(self.os_access)
        self.monty_sandbox = monty_sandbox_from_url(self.monty_sandbox, self.monty_sandbox_url, 'CodeModeToolset')
        self.monty_sandbox_url = None

    @property
    def id(self) -> str | None:
        """`'code_mode'` with `dynamic_catalog`, which qualifies the catalog block into `toolset:code_mode:catalog`, else `None`."""
        return _CODE_MODE_TOOLSET_ID if self.dynamic_catalog else None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance with isolated REPL state for this agent run."""
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving REPL state."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        new_self = replace(self, wrapped=new_wrapped)
        new_self._run_state = self._run_state
        new_self._repl = self._repl
        new_self._warned_missing_return_schema = self._warned_missing_return_schema
        new_self._last_catalog = self._last_catalog
        return new_self

    async def __aenter__(self) -> Self:
        """Enter the wrapped toolset and prepare lazy Monty resources for this run."""
        # Reject misconfiguration when the run starts rather than at the first `run_code` call,
        # which may be many model steps later. The resolved value is recomputed at checkout.
        _resolve_resource_limits(self.resource_limits)
        if self.max_tool_calls < 1:
            raise UserError('`max_tool_calls` must be at least 1')
        sandbox = self.monty_sandbox
        if isinstance(sandbox, str):
            _check_monty_sandbox_url(sandbox)
            run_state = MontyRunState(monty_sandbox_url=sandbox)
        elif sandbox is None or callable(sandbox):
            run_state = MontyRunState(pool_factory=sandbox)
        else:
            raise UserError('`monty_sandbox` takes a URL or a function that returns the pool, not the pool itself.')
        await self.wrapped.__aenter__()
        self._run_state = run_state
        self._repl = (
            None
            if self.repl_store is None
            else PersistentRepl(self.repl_store, limits=_resolve_resource_limits(self.resource_limits))
        )
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the wrapped toolset, then tear down the worker pool."""
        run_state = self._run_state
        assert run_state is not None
        repl, self._repl = self._repl, None
        self._run_state = None
        try:
            if self.speculation is not None:
                await self.speculation.close()
            return await self.wrapped.__aexit__(*args)
        finally:
            try:
                if repl is not None:
                    await repl.save(run_state)
            finally:
                await run_state.close()

    @classmethod
    def from_run_context(cls, ctx: RunContext[AgentDepsT]) -> Self | None:
        """Return the active step's toolset of this class, or `None` when the run does not use one."""
        tool_manager = ctx.tool_manager
        if tool_manager is None or tool_manager.tools is None:
            return None
        tool = tool_manager.tools.get(RUN_CODE_TOOL_NAME)
        if tool is None:
            return None  # pragma: no cover - `CodeMode` always contributes `run_code`
        return tool.toolset if isinstance(tool.toolset, cls) else None

    async def _functions_revealed_by(self, ctx: RunContext[AgentDepsT], names: Sequence[str]) -> dict[str, str]:
        """Map each of `names` that revealing it adds to `run_code` to the function name code calls it by.

        Called while this step's tools are current, for tools a search just found: the next step sandboxes
        the ones the selector picks, and where two names sanitize to the same function, the one listed
        first keeps it, exactly as `get_tools` will. Names already callable, kept native, or hidden behind
        another name map to nothing.
        """
        tool_manager = ctx.tool_manager
        run_code = (
            None if tool_manager is None or tool_manager.tools is None else tool_manager.tools.get(RUN_CODE_TOOL_NAME)
        )
        if not isinstance(run_code, _RunCodeTool):
            return {}  # pragma: no cover - the agent installs this step's tools before any tool runs
        current = {run_code.sanitized_to_original.get(function, function) for function in run_code.callable_defs}
        revealed = set(names) - current
        owners: dict[str, str] = {}
        for name, tool in run_code.wrapped_tools.items():
            if name in current or (
                name in revealed
                and can_be_sandboxed(tool.tool_def)
                and await matches_tool_selector(self.tool_selector, ctx, tool.tool_def)
            ):
                owners.setdefault(sanitize_tool_name(name), name)
        return {name: function for function, name in owners.items() if name in revealed}

    async def observe_stream_event(self, event: AgentStreamEvent, ctx: RunContext[AgentDepsT]) -> None:
        """Feed one model stream event to the streamed execution tiers this toolset runs."""
        if self.speculation is not None:
            await self.speculation.observe(event, ctx)

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        """Surface the tool catalog as a dynamic instruction when `dynamic_catalog` is set.

        The catalog is stashed by `get_tools` earlier in the same step. It's named `'catalog'`,
        which this toolset's id qualifies into `toolset:code_mode:catalog`, and uses
        `on_change='append'`, so a discovery appends the updated catalog to the message history
        instead of rewriting the instructions. It's returned even while empty, which renders as
        nothing, so the run's first instructions already hold it and the first function to appear
        is appended too. When `dynamic_catalog` is off we defer entirely to the wrapped toolset.
        """
        # Through `super()`, so the wrapped toolsets' parts are relayed under their own ids, not this toolset's.
        upstream = await super().get_instructions(ctx)
        catalog_part = self._catalog_instruction()
        if catalog_part is None:
            return upstream
        if upstream is None:
            return catalog_part
        assert not isinstance(upstream, str | InstructionPart), '`WrapperToolset.get_instructions` returns a list'
        return [*upstream, catalog_part]

    def _catalog_instruction(self) -> InstructionPart | None:
        """The catalog `get_tools` stashed this step, as the instruction block that carries it, possibly empty."""
        if not self.dynamic_catalog:
            return None
        return InstructionPart(
            content=self._last_catalog, dynamic=True, name=_CATALOG_INSTRUCTION_NAME, on_change='append'
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool plus any native (non-sandboxed) tools."""
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Split tools into sandboxed vs native based on the selector.
        placements: dict[str, _Placement] = {}
        for name, tool in wrapped_tools.items():
            tool_def = tool.tool_def
            if not can_be_sandboxed(tool_def):
                placements[name] = 'native'
            elif not ctx.is_tool_available(tool_def):
                # Use the run's public availability predicate so Tool Search and deferred
                # capability reveals share the same wire-side semantics. A hidden tool stays
                # native until revealed and is sandboxed from then on, unless
                # `_place_hidden_tools` keeps it off the wire altogether.
                hidden = self.dynamic_catalog and await matches_tool_selector(self.tool_selector, ctx, tool_def)
                placements[name] = 'hidden' if hidden else 'native'
            elif await matches_tool_selector(self.tool_selector, ctx, tool_def):
                placements[name] = 'sandboxed'
            else:
                placements[name] = 'native'
        searched_locally = _place_hidden_tools(placements, wrapped_tools) if self.dynamic_catalog else set[str]()
        sandboxed_tools = {name: wrapped_tools[name] for name, where in placements.items() if where == 'sandboxed'}
        native_tools = {
            name: replace(tool, tool_def=replace(tool.tool_def, with_native=None)) if name in searched_locally else tool
            for name, tool in ((name, wrapped_tools[name]) for name, where in placements.items() if where == 'native')
        }

        callable_defs, sanitized_to_original = self._partition_callable_tools(sandboxed_tools)

        if self.speculation is not None:
            self.speculation.stash_step(
                wrapped=self.wrapped,
                wrapped_tools=wrapped_tools,
                sanitized_to_original=sanitized_to_original,
                callable_defs=callable_defs,
            )

        # `dynamic_catalog` keeps the catalog out of `run_code.description` (cache-stable
        # tool-defs block) and surfaces it via `get_instructions` instead. Stash it for the
        # `get_instructions` call later this step; empty string means "nothing to surface".
        # The base prose stays host-aware in both modes -- its OS/mount restriction line is
        # static (it doesn't change per discovery), so it belongs in the cached description.
        has_os = self.os_access is not None
        has_mount = self.mount is not None
        if self.dynamic_catalog:
            description = _base_description(has_os=has_os, has_mount=has_mount)
            self._last_catalog = self._render_catalog(callable_defs)
        else:
            description = self._build_description(callable_defs, has_os=has_os, has_mount=has_mount)
            self._last_catalog = ''

        if RUN_CODE_TOOL_NAME in native_tools:
            raise UserError(
                f"Tool name '{RUN_CODE_TOOL_NAME}' is reserved for code mode. Rename your tool to avoid conflicts."
            )

        # When the tool search tool is present, append context about run_code to its
        # description and add a discovery note to the run_code description. It is found by its
        # `tool_kind`, since its name can be prefixed.
        search_tool_name = next(
            (name for name, tool in native_tools.items() if tool.tool_def.tool_kind == 'tool-search'), None
        )
        if search_tool_name is not None:
            search_tool = native_tools[search_tool_name]
            native_tools[search_tool_name] = replace(
                search_tool,
                tool_def=replace(
                    search_tool.tool_def,
                    description=(search_tool.tool_def.description or '') + _SEARCH_TOOLS_MODIFIER,
                ),
            )
            description += _tool_search_addendum(search_tool_name)

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native_tools)
        result[RUN_CODE_TOOL_NAME] = _RunCodeTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=RUN_CODE_TOOL_NAME,
                description=description,
                parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                sequential=True,
                tool_kind=RUN_CODE_KIND,
                capability_id=self._capability_id(ctx),
            ),
            max_retries=self.max_retries,
            args_validator=_RUN_CODE_ARGS_VALIDATOR,
            callable_defs=callable_defs,
            sanitized_to_original=sanitized_to_original,
            wrapped_tools=wrapped_tools,
        )
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code in the sandbox, or pass through to a native tool."""
        run_code_tool = self._as_run_code_tool(tool)
        if run_code_tool is None:
            # Native (non-sandboxed) tool -- pass through to the wrapped toolset.
            with self.speculation.unsafe_call() if self.speculation is not None else nullcontext():
                return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        code = tool_args['code']
        restart = tool_args.get('restart', False)

        run_state = self._run_state
        assert run_state is not None, '`CodeModeToolset` must be entered before calling `run_code`'

        if restart:
            repl = self._persistent_repl(ctx)
            if repl is not None:
                repl.restart(ctx)
            await run_state.reset()

        execution = RunCodeExecution(parent_tool_call_id=ctx.tool_call_id or 'pyd_ai_code_mode')
        result = await self._execute_code(code, ctx, run_code_tool, execution)
        return await self._complete(ctx, execution, result)

    async def _complete(self, ctx: RunContext[AgentDepsT], execution: RunCodeExecution, result: Any) -> ToolReturn[Any]:
        """Build the public result for a `run_code` call that ran to completion.

        Speculative launches the snippet never claimed are evicted only here, on success: a
        snippet that failed into a retry keeps them, so the retry can adopt them under its fresh
        tool call id. What speculation bought goes on the return's history-only metadata, where
        traces and UIs can read it without spending the model's tokens on it every turn.
        """
        tool_return = execution.build_tool_return(result)
        if self.speculation is not None:
            summary = await self.speculation.evict_part(ctx, execution.parent_tool_call_id)
            if summary is not None:
                metadata: dict[str, Any] | None = tool_return.metadata
                assert metadata is not None, '`build_tool_return` always attaches metadata'
                metadata['speculation'] = summary
        return tool_return

    @staticmethod
    def _as_run_code_tool(tool: ToolsetTool[AgentDepsT]) -> _RunCodeTool[AgentDepsT] | None:
        return tool if isinstance(tool, _RunCodeTool) else None

    def _persistent_repl(self, ctx: RunContext[AgentDepsT]) -> PersistentRepl[AgentDepsT] | None:
        """The run's link to `repl_store`, or `None` without one or under durable execution.

        A durable workflow replays its code, so store I/O there would need to be deterministic.
        """
        return None if in_durable_execution(ctx) else self._repl

    def _capability_id(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """The id `capability` is registered under this run, so `run_code` is attributed to it."""
        if self.capability is None:
            return None
        return next((run_id for run_id, cap in ctx.capabilities.items() if cap is self.capability), None)

    async def _execute_code(  # noqa: C901
        self,
        code: str,
        ctx: RunContext[AgentDepsT],
        tool: _RunCodeTool[AgentDepsT],
        execution: RunCodeExecution,
    ) -> Any:
        """Execute one REPL feed and accumulate it into a logical `run_code` call."""
        run_state = self._run_state
        assert run_state is not None, '`CodeModeToolset` must be entered before calling `run_code`'

        repl = self._persistent_repl(ctx)
        if repl is not None:
            await repl.begin(ctx)
        fresh_repl = not run_state.has_executed_feed

        callable_defs = tool.callable_defs
        sanitized_to_original = tool.sanitized_to_original

        # Build a ToolManager for the sandbox's inner tools so that sandboxed
        # tool calls go through the standard validation/execution path. We
        # inherit `root_capability` from the agent's ToolManager (for capability
        # hooks) but use the *wrapped* toolset and its tools.
        # See https://github.com/pydantic/pydantic-ai/pull/4307
        parent_tm = ctx.tool_manager
        assert parent_tm is not None, 'CodeModeToolset requires ctx.tool_manager to be set'
        tool_manager = ToolManager(
            toolset=self.wrapped,
            root_capability=parent_tm.root_capability,
            ctx=ctx,
            tools=tool.wrapped_tools,
        )

        # Determine execution mode for sandbox tool calls:
        # - global_sequential: selected through the parallel execution mode context var.
        # - sequential_tools: per-tool `sequential` flags on ToolDefinition.
        #   These tools are rendered as `def` (sync) and resolved inline.
        global_sequential = global_mode_is_sequential(tool_manager.get_parallel_execution_mode)
        sequential_tools = {name for name, td in callable_defs.items() if td.sequential}

        speculation = self.speculation
        if speculation is not None and not in_durable_execution(ctx):
            # The code is complete here, so every literal eligible call not already in flight
            # launches now; the snippet's sequential awaits then collect from tasks that are all
            # already running instead of blocking one another.
            await speculation.prelaunch_for_execution(ctx, execution, code)

        def dispatch_tool_call(sandbox_name: str, kwargs: dict[str, Any]) -> Coroutine[Any, Any, Any]:
            """Reserve nested-call budget, then build the coroutine that runs the call.

            The reservation is synchronous because the executor turns each deferred call into an
            `asyncio.Task` as soon as this returns, without yielding to the event loop in between.
            Counting inside the coroutine would let one `asyncio.gather` over many calls allocate a
            host task per call before the first check ran, which is the cost the budget bounds.
            Refusing here means no task is created; the executor hands the error to the sandbox at
            the call site, so calls that already completed keep their recorded results.
            """
            original_name = sanitized_to_original.get(sandbox_name, sandbox_name)
            tool_call_id = execution.next_tool_call_id(max_tool_calls=self.max_tool_calls)
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            if speculation is not None:
                claimed = speculation.claim(execution.parent_tool_call_id, sandbox_name, kwargs)
                if claimed is not None:
                    return speculation.adopt(ctx, execution, claimed, call_part)
            return run_tool_call(sandbox_name, call_part)

        async def run_tool_call(sandbox_name: str, call_part: ToolCallPart) -> Any:
            """Run a single tool call dispatched from inside the sandbox.

            Returns the serialized tool result on success. On failure, the exception propagates:
            the execution loop passes it back into Monty via `ExternalException` so the sandbox
            sees it at the `await` site.
            """
            execution.nested_calls[call_part.tool_call_id] = call_part
            guard: AbstractContextManager[None] = nullcontext()
            if speculation is not None and speculation.eligible(sandbox_name):
                await speculation.report_miss(ctx, execution, sandbox_name, call_part)
            elif speculation is not None:
                guard = speculation.unsafe_call()
            with guard:
                return execution.finish(call_part, await run_nested_call(tool_manager, call_part))

        # Type-check only the first executed snippet. Monty's checker can reject valid later
        # snippets that reuse imports or pass a runtime-validated dict to a TypedDict parameter.
        type_check = fresh_repl and bool(callable_defs)
        type_check_stubs = self._build_type_check_stubs(callable_defs) if type_check else None

        # One collector is reused across eager feeds. This keeps the same output cap and error
        # behavior as a normal call while presenting one combined result to the model.
        capture = execution.capture

        def started_calls() -> str:
            # A failure that resets the session has no traceback saying what already ran, so the
            # model would otherwise repeat the side effects of calls that started.
            if not execution.nested_calls:
                return ''
            return f'\n\n{_describe_started_calls(execution.nested_calls, execution.nested_returns)}'

        in_workflow = in_temporal_workflow()
        configured = _resolve_resource_limits(self.resource_limits)
        # `run_code` executes in workflow code and Temporal replays it. An elapsed timer may make the
        # original run and replay take different branches, which Temporal cannot record.
        limits = configured | {'max_feed_duration_secs': None} if in_workflow else configured
        try:
            if repl is not None and await repl.restore(run_state, limits=limits, in_temporal_workflow=in_workflow):
                # The type checker would reject the names the restored state defines.
                fresh_repl = type_check = False
            session = await run_state.get_session(
                type_check=type_check,
                type_check_stubs=type_check_stubs if type_check else None,
                limits=limits,
                in_temporal_workflow=in_workflow,
            )
            try:
                # Already converted in `__post_init__`; this narrows the field's type.
                os_handler = as_os_handler(self.os_access)
                completed = await MontyExecutor(
                    dispatch=dispatch_tool_call,
                    valid_names=callable_defs,
                    sequential_names=sequential_tools,
                    global_sequential=global_sequential,
                    portal=run_state.portal,
                    # The configured limit, kept inside a Temporal workflow too: Monty's elapsed-time
                    # check is dropped there for replay, but sleeps are charged what they request.
                    max_sleep_secs=configured.get('max_feed_duration_secs'),
                    os_handler=os_handler,
                ).run(
                    partial(
                        session.feed_start,
                        code,
                        print_callback=capture.callback,
                        os=os_handler,
                        mount=self.mount,
                        skip_type_check=not type_check,
                    )
                )
            except MontyRuntimeError:
                # The session is idle again and keeps assignments made before the failing line.
                run_state.has_executed_feed = True
                raise
            run_state.has_executed_feed = True
        except MontySyntaxError as e:
            if fresh_repl:
                # No code ran, so discard the checkout-time type stubs. A later step may expose
                # a different tool catalog (for example after Tool Search discovers a tool).
                await run_state.reset()
            raise ModelRetry(f'Syntax error in code:\n{capture.prepend_to(e.display())}') from e
        except MontyTypingError as e:
            # Typing errors can only come from the fresh-feed check above.
            await run_state.reset()
            raise ModelRetry(f'Type error in code:\n{capture.prepend_to(e.display())}') from e
        except MontyRuntimeError as e:
            # Exceptions raised inside dispatch_tool_call (e.g. UserError from
            # ApprovalRequired, or ModelRetry from a wrapped tool) are passed
            # back into Monty via ExternalException. Monty re-raises them at the
            # await site; if the sandbox code doesn't catch them, they bubble up
            # as MontyRuntimeError. The original exception message is preserved
            # in the display string, so the model sees a useful error. This means
            # ModelRetry from a wrapped tool gets double-wrapped
            # (ModelRetry → MontyRuntimeError → ModelRetry), but the retry
            # semantics are the same -- the model gets another chance.
            message = f'Runtime error:\n{capture.prepend_to(e.display())}'
            duration_spent = _is_duration_exhausted(e)
            if execution.budget_exhausted or _exhausted_sandbox_limit(e) is not None:
                # A retry is the only record the model gets of an uncaught failure, and these
                # calls already started. Without them the model reruns their side effects when
                # it retries. Asking which limit tripped, rather than testing one flag per limit,
                # is what keeps a newly added limit from quietly losing this. It matters most on
                # the duration path, which resets the session and with it the REPL state the model
                # would otherwise reconstruct from.
                message += started_calls()
            if duration_spent:
                # The limit stops the sandbox mid-operation, and Monty makes no promise about the
                # heap it leaves behind, so the session is discarded rather than fed again.
                await run_state.reset()
                message += (
                    '\n\nThe code ran longer than `max_duration_secs` and was stopped, so the '
                    'session was reset. Re-run any imports, recreate any state you need, and make '
                    'the code do less work per call.'
                )
            if isinstance(e.exception(), RuntimeError) and re.fullmatch(
                r'suspension limit [0-9]+ exceeded', e.display(format='msg')
            ):
                # Monty has no typed marker for this limit, and unlike a timeout it has a
                # traceback. Keep the advice conditional: a tool could raise the same text.
                message += (
                    "\n\nIf this reports the sandbox session's `max_suspensions` limit, "
                    'its cumulative host-interaction budget is exhausted. Further tool calls, '
                    'OS callbacks, name lookups and future resolutions need a fresh session; '
                    'revising the snippet does not replenish this budget. Pure Python using '
                    'existing state may still work. Pass `restart: true` to start a fresh '
                    'session; that discards all REPL variables, imports and definitions. '
                    'Check the calls already started before continuing, and do not replay '
                    'completed side effects.'
                )
            if execution.usage_limit_exceeded is not None:
                raise execution.usage_limit_exceeded
            raise ModelRetry(message) from e
        except MontyCrashedError as e:
            # The worker died mid-feed (e.g. the code exhausted its memory or hit the
            # request timeout) and the REPL state died with it; the pool replaces the
            # worker transparently. Reset so the retry starts from a fresh,
            # type-checked session.
            await run_state.reset()
            if execution.usage_limit_exceeded is not None:
                raise execution.usage_limit_exceeded
            raise ModelRetry(
                'The code crashed the sandbox worker and the session was reset. Revise the code and try again.'
                f'{started_calls()}'
            ) from e
        except Exception as e:
            # The session may have been invalidated by a host-side binding or protocol failure.
            # Make the reset visible so the model can rebuild state on its next attempt. Include
            # the error text: there is no Monty `display()` for host-side failures, and the cause
            # chain is dropped once the retry becomes a prompt part, so this message is the only
            # record of what failed for both the model and the transcript.
            await run_state.reset()
            if execution.usage_limit_exceeded is not None:
                raise execution.usage_limit_exceeded
            error_text = f'{type(e).__name__}: {e}'
            if isinstance(self.monty_sandbox, str):
                # A dial failure quotes the configured URL, which may carry credentials in its
                # userinfo, path, or query; keep it out of the transcript-bound retry message.
                error_text = error_text.replace(self.monty_sandbox, '<monty_sandbox>')
            raise ModelRetry(
                'Code execution failed and the session was reset. Re-run any imports, recreate '
                f'any state you need, and try again.\n{capture.prepend_to(error_text)}{started_calls()}'
            ) from e
        except BaseException as e:
            # Convert a sandbox panic to a retry (see `is_sandbox_panic`);
            # interruptions re-raise unchanged after dropping the suspended session.
            if not is_sandbox_panic(e):
                await run_state.reset()
                raise
            # The panic aborts the VM mid-execution, so the REPL's accumulated state cannot
            # be trusted; drop it so the retry starts from a fresh, type-checked session.
            await run_state.reset()
            if execution.usage_limit_exceeded is not None:
                raise execution.usage_limit_exceeded
            raise ModelRetry(
                'The code aborted inside the sandbox and the session was reset. Revise the code and try again.'
                f'{started_calls()}'
            ) from e

        if execution.usage_limit_exceeded is not None:
            raise execution.usage_limit_exceeded
        result = completed.output
        # Validate result to reconstruct multimodal types (e.g. BinaryContent from
        # serialized dicts) so they flow through to the model natively.
        if result is not None:
            result = _model_safe_result(_TOOL_RETURN_CONTENT_TA.validate_python(result))

        return result

    def _partition_callable_tools(
        self, wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> tuple[dict[str, ToolDefinition], dict[str, str]]:
        """Return tool definitions that can be called from inside the sandbox.

        Tool names that are not valid Python identifiers (e.g. MCP tools with
        hyphens or dots like `get-weather`, `api.call`) are sanitized to
        underscored forms and mapped back to their original names for dispatch.

        Returns:
            A tuple of `(callable_defs, sanitized_to_original)`.
        """
        callable_defs, sanitized_to_original = partition_callable_tools(
            wrapped_tools, owner='CodeMode', reserved_name=RUN_CODE_TOOL_NAME
        )
        missing_return_schema: list[str] = []
        for safe_name, td in callable_defs.items():
            name = sanitized_to_original.get(safe_name, safe_name)
            if td.return_schema is None and name not in self._warned_missing_return_schema:
                missing_return_schema.append(name)
        _warn_missing_return_schemas(missing_return_schema)
        # Recorded only once warned, so a warning escalated to an error is raised again next time.
        self._warned_missing_return_schema.update(missing_return_schema)
        return callable_defs, sanitized_to_original

    @staticmethod
    def _build_description(callable_defs: dict[str, ToolDefinition], *, has_os: bool, has_mount: bool) -> str:
        """Render the `run_code` description: base prose + TypedDicts + function signatures."""
        base = _base_description(has_os=has_os, has_mount=has_mount)
        catalog = CodeModeToolset._render_catalog(callable_defs)
        if not catalog:
            return base
        return base + '\n\n' + catalog

    @staticmethod
    def _render_catalog(callable_defs: dict[str, ToolDefinition]) -> str:
        """Render the functions-header + TypedDict + function-signature blocks, or `''` if no defs.

        Excludes the `run_code` base prose; the catalog is the discovery-driven portion that's
        cache-hostile when carried in `run_code.description`. Used by `_build_description`
        (default static-description path) and by `get_instructions` (the `dynamic_catalog`
        path, which moves it into instructions instead).
        """
        if not callable_defs:
            return ''

        sigs, conflicting = signatures_and_conflicts(callable_defs)
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        function_blocks = [
            td.render_signature('...', is_async=not td.sequential, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        ]

        has_sync = any(td.sequential for td in callable_defs.values())
        has_async = any(not td.sequential for td in callable_defs.values())
        sections = [_functions_header(has_sync=has_sync, has_async=has_async)]
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
        return '\n\n'.join(sections)

    @staticmethod
    def _build_type_check_stubs(callable_defs: dict[str, ToolDefinition]) -> str:
        """Build Python stubs for Monty's static type checker."""
        sigs, conflicting = signatures_and_conflicts(callable_defs)
        parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        parts.extend(type_blocks)
        parts.extend(
            td.render_signature(
                'raise NotImplementedError()', is_async=not td.sequential, conflicting_type_names=conflicting
            )
            for td in callable_defs.values()
        )
        return '\n\n'.join(parts)


# Scalars every tool-return serializer handles. Unlike `_SANDBOX_NATIVE_SCALARS` this leaves out
# `Ellipsis`, which Monty holds but JSON has no form for.
_MODEL_NATIVE_SCALARS = (str, bytes, bytearray, bool, int, float, type(None))


def _model_safe_result(value: object) -> object:
    """Render the parts of a snippet's result that have no JSON form as their `repr`.

    Monty hands some sandbox values back as host objects: `type(x)` and `ValueError` arrive
    as `type` objects, `len` as a `MontyStdTypeProxy`, a bare exception instance as itself.
    None of them serialize, so the tool return would abort the run in whichever layer
    renders it first (`ToolOutputLimits`, or pydantic-ai building the model request). The
    `repr` is what the snippet's author would have seen in a Python REPL. Non-finite floats
    do serialize, but as `null`, which would hide the result, so they render as `repr` too.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, _MODEL_NATIVE_SCALARS) or is_multi_modal_content(value):
        return value
    if _is_mapping(value):
        rendered = {
            key if _serializes({key: None}) else repr(key): _model_safe_result(item) for key, item in value.items()
        }
        # A rendered key can equal an existing one (`{int: 1, "<class 'int'>": 2}`), and
        # the dict would silently drop an entry; the whole mapping's `repr` loses nothing.
        return rendered if len(rendered) == len(value) else repr(value)
    if _is_list_or_tuple(value):
        items = [_model_safe_result(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value if _serializes(value) else repr(value)


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_list_or_tuple(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    return isinstance(value, (list, tuple))


def _serializes(value: object) -> bool:
    try:
        to_json(value)
    except PydanticSerializationError:
        return False
    return True
