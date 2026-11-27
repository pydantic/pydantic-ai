"""One case per capability: how the suite builds it, and what two of it on one agent mean.

Every capability class a package ships has an entry in its package's `CASES`: this module's for
`pydantic_ai.capabilities` and `pydantic_ai.models`, `tests/harness/capability_cases.py` for Pydantic AI
Harness. `test_every_capability_has_a_case` fails until a new one gets an entry, so adding a capability is
one entry that says what two of it mean (`combine`), and, when its defaults can't build it, how to (`build`)
or why it can't run offline (`Unbuildable`). The suites that run over every capability read their cases from
here: `tests/test_history_append_only.py` and `tests/test_capability_combine.py`, and their Harness twins. A case
whose `calls` run its tools is also run by the `CONTRACTS` at the end of this module, which
`tests/harness/test_capability_contracts.py` checks, and under each durable engine, by
`tests/harness/test_durable_tool_io.py` (DBOS) and `tests/harness/test_durable_temporal_replay.py` (Temporal),
which check the units its calls record against its `durable_steps`. A case whose capability contributes a toolset
or a tool hook declares `calls`, or an exemption from `DECLARES_CALLS` with its reason. A public capability's calls
are defined once, in `pydantic_ai.capabilities.conformance.CALLS` (`pydantic_ai_harness.conformance.CALLS` for
Pydantic AI Harness), which a third-party durable engine's suite runs, and its case refers to them; only a case
whose calls need a stand-in from the tests defines its calls here.
`agent_docs/capability-conformance.md` lists what each area's contracts require of a capability.

This module imports nothing from Pydantic AI Harness: `tests/conftest.py` leaves `tests/harness` out of a
lane that doesn't install it, and the core suites import this module on every lane.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, TypeGuard, get_args
from unittest.mock import patch

from pydantic_ai.capabilities import (
    MCP,
    Caching,
    Citations,
    CombinedCapability,
    DynamicCapability,
    FileUnderstanding,
    HandleDeferredToolCalls,
    ImageGeneration,
    Instrumentation,
    LocalWorkspace,
    ModelFallback,
    NativeTool,
    PrefixTools,
    PrepareOutputTools,
    PrepareTools,
    ProcessEventStream,
    ProcessHistory,
    RaiseContentFilterError,
    ReinjectSystemPrompt,
    ResolveModelId,
    SelectModel,
    Thinking,
    ToolSearch,
    Toolset,
    UseThreadExecutor,
    WebFetch,
    WebSearch,
    WindowsWorkspace,
    WrapperCapability,
    XSearch,
    conformance,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.conformance import (
    CALLS,
    _InUnit,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.durable_exec.conformance import Recovery, Support
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool, WebFetchTool, WebSearchTool, XSearchTool
from pydantic_ai.toolsets import FunctionToolset

from .append_only import Fixture, drain_events
from .conftest import try_import
from .manifests import Exempt

with try_import():
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset


CORE_CAPABILITY_PACKAGES = ('pydantic_ai.capabilities', 'pydantic_ai.models')
"""The packages whose capability classes `CASES` covers."""


@dataclass
class Anonymous:
    """No default `id`, and two of them really do coexist: two different things, both active.

    The run derives a distinct id per occurrence instead. A user who gives two the same `id`
    explicitly gets the base `combine`, which raises -- that is a mistake, not a composition.
    """

    reason: str


@dataclass
class Rejected:
    """No default `id`, and two never coexist because something else rejects them.

    Distinct from `Collides`, which is about tool names. A durability capability is refused by the
    engine's own `from_agent` lookup, which runs before an `id` would matter -- so the reason it
    cannot repeat has nothing to do with this table's subject, and saying "tool names" would be
    wrong.
    """

    reason: str


@dataclass
class Collides:
    """No default `id`, and two of them never coexist: their toolsets register the same tool names.

    A fact about the capability, not a preference. It is recorded here because the alternative --
    an `Anonymous` reason like "one per rooted directory" -- describes a configuration the fixed
    tool names make unreachable, and nothing would catch the drift.

    Whether these should instead declare a default `id`, so that two merge, is a decision for each
    capability's owner: merging `FileSystem` would union allow-lists, which widens access. This
    table records where the question applies rather than answering it silently.
    """

    reason: str
    make: (
        Callable[[Callable[..., AbstractCapability[None]]], tuple[AbstractCapability[None], AbstractCapability[None]]]
        | None
    ) = None
    """Builds two from the discovered class, when two can be built without credentials or a network.

    Takes the class rather than closing over an import: these capabilities live in optional groups,
    and a module-level import would break the whole file on a lane that installs none of them.
    """


@dataclass
class Combines:
    """A default `id`: two of these are one configuration stated twice, and `combine` resolves them."""

    reason: str
    make: Callable[[], tuple[AbstractCapability[None], AbstractCapability[None]]]
    """Builds two instances that state *different* configuration, so a merge is observable."""
    # `Any`: each check takes its own capability type, which no single parameter type admits.
    check: Callable[[Any], None]
    """Asserts what survived. Reads derived state too, not just the declared fields."""


@dataclass
class Narrows:
    """A default `id` and a `combine` that narrows rather than unions: two that disagree raise.

    The access-boundary case in "Deciding What Two Of It Mean"
    (`src/pydantic_ai_harness/agent_docs/capability-authoring.md`), for connections to a provider: merging
    two field by field could send one account's credential to another's server or drop `read_only`. `make`
    takes the class, like `Collides`, because these live in optional groups that a module-level import would
    break.
    """

    reason: str
    make: Callable[[Callable[..., AbstractCapability[None]]], tuple[AbstractCapability[None], AbstractCapability[None]]]
    """Builds two that differ, from the discovered class, without credentials or a network."""


Policy = Anonymous | Collides | Combines | Narrows | Rejected


@dataclass(frozen=True)
class Unbuildable:
    """A capability the suites can't construct offline, with the reason."""

    reason: str


@dataclass(frozen=True)
class DurableSteps:
    """The durable units a case's calls record under each engine, in the order they record them.

    The names are persisted: a workflow recorded by one release replays by them in the next, so `DURABLE_DBOS` and
    `DURABLE_TEMPORAL` fail when a case records anything else. The run's own model requests are left out, as
    every case makes one per call, one per follow-up its calls' `follow_ups` counts, and one to answer. A case
    whose tools do no I/O through a durable unit records nothing, which is checked too.
    """

    dbos: tuple[str, ...] = ()
    """DBOS step names, without the `<agent name>__` prefix: `capability__<id>.<operation>` for a
    `durable_operation` or a workspace call, `mcp_server__<toolset id>.<operation>` for an MCP server."""
    temporal: tuple[str, ...] = ()
    """Temporal activity names, without the `agent__<agent name>__` prefix: `toolset__<toolset id>__call_tool` for a
    function tool, which runs in its activity, `capability__<id>__<operation>` for a `durable_operation` that
    workflow code calls, and `mcp_server__<toolset id>__<operation>` for an MCP server."""
    streams: bool = False
    """Whether the run's model requests are streamed units (`model.request_stream`, `model_request_stream`), as they
    are when a capability listens for events. The unit's name is persisted like the others, so a case whose model
    unit flips between the two fails."""


@dataclass(frozen=True, kw_only=True)
class Calls(conformance.Calls):
    """The public `Calls`, with why running them moves the prompt-cache prefix.

    `moves_cache_prefix` belongs on `CapabilityCase`, beside `durable_steps`, but #14376 (the C1 route matrix) reads
    it as `CASES[name].calls.moves_cache_prefix`. Whichever of the two lands second moves it there, and makes a
    public case's `calls` the public `Calls` itself instead of a copy.
    """

    moves_cache_prefix: str | None = None
    """Why running the calls rewrites earlier history by design, which moves the prompt-cache prefix: a suite
    that runs them one per response marks the run `moves_cache_prefix` with it."""

    @classmethod
    def of(cls, calls: conformance.Calls, *, moves_cache_prefix: str | None = None) -> Calls:
        """The public `calls` of a case, for its `CapabilityCase`."""
        return cls(calls.made, build=calls.build, follow_ups=calls.follow_ups, moves_cache_prefix=moves_cache_prefix)


@dataclass(frozen=True)
class CapabilityCase:
    """One capability: what two of it on one agent mean, and how the suites build it."""

    combine: Policy
    """What two of it on one agent mean, checked by `test_capability_combine_policy_holds`."""
    build: Fixture | Unbuildable | None = None
    """How to build it in a test's `tmp_path`: `None` for its defaults, or why it can't run offline."""
    requires: str | None = None
    """The module its optional dependency group provides, when its module needs one; a test skips it without."""
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    """Credentials it checks when it's built; nothing is sent, as the scripted model never calls its tools."""
    calls: Calls | None = None
    """Calls that run its tools, for the contracts and suites that call them: `Calls.of` the public ones, from
    `CALLS`, unless they need a stand-in from the tests."""
    durable_steps: DurableSteps = DurableSteps()
    """What the calls record under each durable engine, run one per response."""
    launches_python: str | None = None
    """Why running the calls launches a Python interpreter, such as a kernel process: a suite that runs them marks
    the test `subprocess` with it, as `conditional`, since a durable engine may refuse the case before it launches."""

    @property
    def reason(self) -> str:
        """Why two of it mean what `combine` says, which is what classifies the capability."""
        return self.combine.reason

    @property
    def fixture(self) -> Fixture | None:
        """`build`, for a suite that already left out the unbuildable cases."""
        assert not isinstance(self.build, Unbuildable), f'an unbuildable case was built: {self.build.reason}'
        return self.build


def _is_capability_class(obj: object) -> TypeGuard[type[AbstractCapability[None]]]:
    """Whether `obj` is a capability class, and not something that merely looks like one."""
    if not isinstance(obj, type):
        return False
    try:
        return issubclass(obj, AbstractCapability)
    except TypeError:  # pragma: lax no cover
        return False


def caused_by_missing_dependency(exc: BaseException) -> bool:
    """Whether `exc` is an uninstalled optional group rather than a broken capability.

    Follows the chain rather than matching a type, because the same condition reaches this walk in
    three shapes: `ModuleNotFoundError` from the import itself, a plain `ImportError` from a module
    that re-raises install guidance, and a `UserError` a capability raises while constructing
    something at import time. The last two are raised inside an `except ImportError` block, so the
    original is on `__cause__` or `__context__` either way.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ImportError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def shipped_capability_types(packages: Sequence[str]) -> tuple[dict[str, type[AbstractCapability[None]]], list[str]]:
    """Every capability class in `packages`, public or not, and the modules the walk skipped.

    Walks the packages rather than reading an export list, so a capability that is never re-exported
    is still covered. Modules whose optional dependency group is not installed are skipped -- the
    import error means the capability could not have been used either.
    """
    found: dict[str, type[AbstractCapability[None]]] = {}
    classes: set[type[AbstractCapability[None]]] = set()
    skipped: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for package_name in packages:
            package = importlib.import_module(package_name)
            owner = f'{package_name.split(".")[0]}.'
            for module_info in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
                try:
                    module = importlib.import_module(module_info.name)
                except Exception as exc:
                    if not caused_by_missing_dependency(exc):
                        # A real failure -- a broken module, a bad constant -- has to surface.
                        # Swallowing it drops the capability from `found`, and the exhaustiveness check
                        # then passes because the capability it should have caught is invisible
                        # rather than absent. That is not hypothetical: it is how `AWSLambdaDurability`
                        # reached this branch with no policy entry while that check was green locally.
                        raise
                    skipped.append(module_info.name)
                    continue
                for obj in vars(module).values():
                    if _is_capability_class(obj) and obj is not AbstractCapability and obj.__module__.startswith(owner):
                        found[obj.__name__] = obj
                        classes.add(obj)
    # `CASES` is keyed by name, so two classes sharing one would collapse into a single entry and let
    # whichever lost ship with no case at all -- the exact gap this guards.
    assert len(found) == len(classes), (
        f'two capability classes share a name: {sorted(cls.__module__ + "." + cls.__name__ for cls in classes)}'
    )
    return found, skipped


def _mcp_server() -> FastMCP[None]:
    server: FastMCP[None] = FastMCP('extra', instructions='Extra server instructions.')

    @server.tool()
    def extra_tool() -> str:
        return 'extra'

    return server


def _no_op_tool() -> str:
    """Do nothing."""
    return 'ok'  # pragma: no cover


ECHO_STEPS = DurableSteps(temporal=('toolset__echo__call_tool',))
"""What a call to the `echo` tool records under each durable engine: `echo` does no I/O, so only Temporal's activity
for it."""


def _keep_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages


def _keep_model(ctx: ModelSelectionContext[None]) -> Model:
    assert ctx.model is not None
    return ctx.model


def _check_caching(merged: Caching) -> None:
    assert merged.retention == '1h', 'a scalar takes the later value'


def _check_thinking(merged: Thinking) -> None:
    assert merged.effort == 'high', 'a scalar takes the later value'


def _check_web_search(merged: WebSearch) -> None:
    assert merged.allowed_domains == ['a.com', 'b.com'], 'allow-lists are unioned, not replaced'
    # The native tool is what reaches the provider, so the merge has to reach it too.
    assert isinstance(merged.native, WebSearchTool)
    assert merged.native.allowed_domains == ['a.com', 'b.com'], (
        'the merged allow-list must reach the native tool, or the request goes out unrestricted'
    )


def _check_web_fetch(merged: WebFetch) -> None:
    assert merged.allowed_domains == ['a.com', 'b.com']
    assert isinstance(merged.native, WebFetchTool)
    assert merged.native.allowed_domains == ['a.com', 'b.com']


def _check_reinject(merged: ReinjectSystemPrompt) -> None:
    assert merged.replace_existing is True


def _check_content_filter(merged: RaiseContentFilterError) -> None:
    assert merged.id == 'raise_content_filter_error'


def _check_citations(merged: Citations) -> None:
    assert merged.id == 'citations'


def _check_x_search(merged: XSearch) -> None:
    assert merged.allowed_x_handles == ['a', 'b']
    assert isinstance(merged.native, XSearchTool)
    assert merged.native.allowed_x_handles == ['a', 'b']


def _check_image_generation(merged: ImageGeneration) -> None:
    assert merged.quality == 'high'


def _check_file_understanding(merged: FileUnderstanding) -> None:
    assert merged.instructions == 'One sentence.', 'a scalar takes the later value'


def _check_instrumentation(merged: Instrumentation) -> None:
    assert merged.settings is not None
    assert merged.settings.include_content is False, 'a scalar takes the later value'


_FIRST_EXECUTOR = ThreadPoolExecutor(1, 'first')
_SECOND_EXECUTOR = ThreadPoolExecutor(1, 'second')


def _check_local_workspace(merged: LocalWorkspace[None]) -> None:
    # One environment: the later workspace replaces the earlier whole, so the first one's `env` never leaks.
    assert (merged.working_dir, merged.env, merged.read_only) == ('/second', None, False)


def _windows_workspaces() -> tuple[WindowsWorkspace[None], WindowsWorkspace[None]]:
    # Built as if on Windows, which they need; combining them touches no backend.
    with patch('pydantic_ai.workspaces.windows._on_windows', return_value=True):
        return WindowsWorkspace('/first', env={'FIRST_SECRET': 'x'}, read_only=True), WindowsWorkspace('/second')


def _check_tool_search(merged: ToolSearch) -> None:
    assert merged.max_results == 20, 'a scalar takes the later value'


def _check_in_unit(merged: _InUnit) -> None:
    assert (merged.emit, merged.enqueue) == (False, True), 'a scalar takes the later value'


def _check_thread_executor(merged: UseThreadExecutor) -> None:
    assert merged.executor is _SECOND_EXECUTOR, 'the executor that would have shadowed the other'


CASES: dict[str, CapabilityCase] = {
    # -- One per agent: a default `id`, and `combine` says what two of them mean. --
    'Caching': CapabilityCase(
        Combines(
            'an agent has one caching configuration',
            lambda: (Caching('5m'), Caching('1h')),
            _check_caching,
        )
    ),
    'Thinking': CapabilityCase(
        Combines(
            'an agent has one thinking configuration',
            lambda: (Thinking(effort='low'), Thinking(effort='high')),
            _check_thinking,
        )
    ),
    'WebSearch': CapabilityCase(
        Combines(
            'one web search configuration, but its allow-list must not be silently widened',
            lambda: (WebSearch(allowed_domains=['a.com']), WebSearch(allowed_domains=['b.com'])),
            _check_web_search,
        )
    ),
    'WebFetch': CapabilityCase(
        Combines(
            'one web fetch configuration, same allow-list concern as `WebSearch`',
            lambda: (WebFetch(allowed_domains=['a.com']), WebFetch(allowed_domains=['b.com'])),
            _check_web_fetch,
        )
    ),
    'XSearch': CapabilityCase(
        Combines(
            'one X search configuration',
            lambda: (
                XSearch(fallback_subagent_model='xai:grok-4.3', allowed_x_handles=['a']),
                XSearch(fallback_subagent_model='xai:grok-4.3', allowed_x_handles=['b']),
            ),
            _check_x_search,
        ),
        build=lambda _: XSearch(fallback_subagent_model=TestModel()),
    ),
    'ImageGeneration': CapabilityCase(
        Combines(
            'one image generation configuration',
            lambda: (
                ImageGeneration(fallback_subagent_model='openai-responses:gpt-5.4', quality='low'),
                ImageGeneration(fallback_subagent_model='openai-responses:gpt-5.4', quality='high'),
            ),
            _check_image_generation,
        )
    ),
    'FileUnderstanding': CapabilityCase(
        Combines(
            'an agent describes files one way',
            lambda: (
                FileUnderstanding(fallback_subagent_model='openai:gpt-5.6-sol'),
                FileUnderstanding(fallback_subagent_model='openai:gpt-5.6-sol', instructions='One sentence.'),
            ),
            _check_file_understanding,
        ),
        build=lambda _: FileUnderstanding(fallback_subagent_model=TestModel()),
    ),
    'Instrumentation': CapabilityCase(
        Combines(
            'an agent is instrumented one way',
            lambda: (
                Instrumentation(settings=InstrumentationSettings(include_content=True)),
                Instrumentation(settings=InstrumentationSettings(include_content=False)),
            ),
            _check_instrumentation,
        ),
        calls=Calls.of(CALLS['Instrumentation']),
        durable_steps=ECHO_STEPS,
    ),
    'ReinjectSystemPrompt': CapabilityCase(
        Combines(
            'one reinjection policy per agent',
            lambda: (ReinjectSystemPrompt(), ReinjectSystemPrompt(replace_existing=True)),
            _check_reinject,
        )
    ),
    'Citations': CapabilityCase(
        Combines(
            'carries no configuration at all, so two are interchangeable',
            lambda: (Citations(), Citations()),
            _check_citations,
        )
    ),
    'RaiseContentFilterError': CapabilityCase(
        Combines(
            'carries no configuration at all, so two are interchangeable',
            lambda: (RaiseContentFilterError(), RaiseContentFilterError()),
            _check_content_filter,
        )
    ),
    'LocalWorkspace': CapabilityCase(
        Combines(
            'the later configuration replaces the earlier one whole',
            lambda: (LocalWorkspace('/first', env={'FIRST_SECRET': 'x'}, read_only=True), LocalWorkspace('/second')),
            _check_local_workspace,
        ),
        build=lambda tmp_path: LocalWorkspace(tmp_path),
    ),
    'WindowsWorkspace': CapabilityCase(
        Combines(
            'the later configuration replaces the earlier one whole, as with `LocalWorkspace`',
            _windows_workspaces,
            _check_local_workspace,
        ),
        build=Unbuildable(
            "Builds only on Windows; it's `LocalWorkspace` with another backend, which adds nothing to the conversation."
        ),
    ),
    'ToolSearch': CapabilityCase(
        Combines(
            'one tool-discovery configuration per agent',
            lambda: (ToolSearch(max_results=5), ToolSearch(max_results=20)),
            _check_tool_search,
        )
    ),
    'UseThreadExecutor': CapabilityCase(
        Combines(
            'exactly one executor is in effect; nesting already made this last-wins implicitly',
            lambda: (UseThreadExecutor(_FIRST_EXECUTOR), UseThreadExecutor(_SECOND_EXECUTOR)),
            _check_thread_executor,
        ),
        build=lambda _: UseThreadExecutor(ThreadPoolExecutor(max_workers=1)),
    ),
    # -- Several of these is the normal case, so they stay anonymous. --
    'Capability': CapabilityCase(Anonymous('a generic bundle; several per agent is the usual shape')),
    'CombinedCapability': CapabilityCase(
        Anonymous('structural container; nesting is the semantic'),
        build=lambda _: CombinedCapability([Thinking()]),
    ),
    'WrapperCapability': CapabilityCase(
        Anonymous('structural wrapper; nesting is the semantic'),
        build=lambda _: WrapperCapability(Thinking()),
    ),
    'PrefixTools': CapabilityCase(
        Anonymous('structural wrapper, applied once per wrapped capability'),
        build=lambda _: PrefixTools(Toolset(FunctionToolset([_no_op_tool])), prefix='extra'),
        calls=Calls.of(CALLS['PrefixTools']),
        durable_steps=ECHO_STEPS,
    ),
    'DynamicCapability': CapabilityCase(
        Anonymous('one per capability function'),
        build=lambda _: DynamicCapability(lambda ctx: Thinking()),
    ),
    'ResolvedDynamicCapability': CapabilityCase(Anonymous('the resolved form of a `DynamicCapability`')),
    'NativeTool': CapabilityCase(
        Anonymous('one per native tool'),
        build=lambda _: NativeTool(CodeExecutionTool()),
    ),
    'NativeOrLocalTool': CapabilityCase(
        Anonymous('used directly it is parameterized by the tools passed to it'),
        build=Unbuildable(
            'Base class that needs a subclass; `WebSearch`, `WebFetch`, `XSearch` and `MCP` are its cases.'
        ),
    ),
    'MCP': CapabilityCase(
        Anonymous(
            'several servers per agent is the normal case; the URL derives its *toolset* id, not a capability id'
        ),
        build=lambda _: MCP(local=MCPToolset(_mcp_server())),
        requires='fastmcp',
        calls=Calls.of(CALLS['MCP']),
        # The server's instructions are left out, as `include_instructions` is off by default.
        durable_steps=DurableSteps(
            dbos=('mcp_server__extra.get_tools', 'mcp_server__extra.call_tool'),
            temporal=('mcp_server__extra__get_tools', 'mcp_server__extra__call_tool'),
        ),
    ),
    'Toolset': CapabilityCase(
        Anonymous('one per toolset'),
        build=lambda _: Toolset(FunctionToolset([_no_op_tool])),
        calls=Calls.of(CALLS['Toolset']),
        durable_steps=ECHO_STEPS,
    ),
    'Hooks': CapabilityCase(Anonymous('several hook bundles compose')),
    'HandleDeferredToolCalls': CapabilityCase(
        Anonymous('`CombinedCapability` chains handlers via `remaining`'),
        build=lambda _: HandleDeferredToolCalls(lambda ctx, requests: requests.build_results()),
    ),
    'RequestInput': CapabilityCase(
        Anonymous('deprecated alias of `HandleDeferredToolCalls`, chained the same way'),
        build=Unbuildable(
            "Deprecated alias of `HandleDeferredToolCalls` with a handler that takes a resume token, which that class's "
            'case covers; constructing it warns.'
        ),
    ),
    'ResolveModelId': CapabilityCase(
        Anonymous('returns `None` to let a later capability resolve; chaining is the feature'),
        build=lambda _: ResolveModelId(lambda ctx, model_id: None),
    ),
    'SelectModel': CapabilityCase(
        Anonymous('receives the lower-precedence model; chaining is designed'),
        build=lambda _: SelectModel(_keep_model),
    ),
    'ModelFallback': CapabilityCase(
        Anonymous('each instance walks its own chain; the innermost handles a failure first'),
        # The step's model answers, so the fallback model is never sent anything.
        build=lambda _: ModelFallback(TestModel()),
    ),
    'RetryOnModelErrors': CapabilityCase(
        Anonymous(
            'each instance retries its own `retry_on`; all count the same attempts at a model against their `max_attempts`'
        )
    ),
    'BatchModelRequests': CapabilityCase(
        Anonymous('each instance is its own batching pool; a model already batched is not batched again'),
        build=Unbuildable(
            "Sends each request through the provider's batch API, which the scenario doesn't serve. A batch item's "
            "body is built by the same code as the model's own request, so it's the unbatched request's (see the "
            "providers' `batch_encode` tests)."
        ),
    ),
    'ProcessHistory': CapabilityCase(
        Anonymous('history processors stack'),
        build=lambda _: ProcessHistory(_keep_history),
    ),
    'ProcessEventStream': CapabilityCase(
        Anonymous('event-stream processors stack'),
        build=lambda _: ProcessEventStream(drain_events),
    ),
    'PrepareTools': CapabilityCase(
        Anonymous('tool preparers stack'),
        build=lambda _: PrepareTools(lambda ctx, tools: tools),
    ),
    'PrepareOutputTools': CapabilityCase(
        Anonymous('output-tool preparers stack'),
        build=lambda _: PrepareOutputTools(lambda ctx, tools: tools),
    ),
    'SetToolMetadata': CapabilityCase(
        Anonymous('one per `ToolSelector`; several selectors compose'),
        calls=Calls.of(CALLS['SetToolMetadata']),
        durable_steps=ECHO_STEPS,
    ),
    'IncludeToolReturnSchemas': CapabilityCase(
        Anonymous('one per `ToolSelector`; several selectors compose'),
        calls=Calls.of(CALLS['IncludeToolReturnSchemas']),
        durable_steps=ECHO_STEPS,
    ),
    'DeferredCapabilityLoader': CapabilityCase(Anonymous('auto-injected only when absent')),
    'PendingMessageDrainCapability': CapabilityCase(Anonymous('auto-injected only when absent')),
    '_GroupMember': CapabilityCase(Anonymous('internal: dispatches a deferred group member, never registered')),
    '_InUnit': CapabilityCase(
        Combines(
            'internal: a scenario stand-in, one per agent, which emits or enqueues inside a durable unit',
            lambda: (_InUnit(emit=True), _InUnit(enqueue=True)),
            _check_in_unit,
        )
    ),
    '_FromHook': CapabilityCase(Anonymous('internal: a scenario stand-in, which emits and enqueues from a hook')),
    'AnthropicCompaction': CapabilityCase(
        Anonymous('each appends its own compaction edit to `anthropic_context_management`, so two send both edits')
    ),
    'OpenAICompaction': CapabilityCase(
        Anonymous(
            'stateful, each defers to an `openai_context_management` already in the model settings; stateless, each '
            'runs its own trigger before the request'
        )
    ),
}
"""Every core capability, keyed by class name."""


@dataclass(frozen=True)
class Contract:
    """A property every case it applies to must have, checked by a test of its own over those cases.

    A case it applies to is checked unless `exempt` names it with a reason, which is where a capability that
    breaks the property on main is recorded, linked to the issue that tracks it.
    """

    name: str
    applies: Callable[[str, CapabilityCase], bool]
    """Whether it applies to the case of the capability with this class name."""
    exempt: Mapping[str, Exempt] = field(default_factory=dict[str, Exempt])


RENAMED = Contract(
    'renamed',
    applies=lambda _, case: case.calls is not None,
    exempt={
        'DeduplicateFileReads': Exempt(
            'its documented `file_key` example matches `read_file` by name, so under a prefix it never '
            'deduplicates: https://github.com/pydantic/pydantic-ai/issues/13967'
        ),
        **dict.fromkeys(
            ('CodeMode', 'IPythonCodeMode', 'ToolBridge'),
            Exempt(
                'its tools come from its wrapper toolset, which `PrefixTools` leaves unprefixed: '
                'https://github.com/pydantic/pydantic-ai/issues/14537'
            ),
        ),
    },
)
"""A capability's tools behave the same under `prefix_tools()`: a tool, hook or event found by name breaks it."""


def _buildable(_: str, case: CapabilityCase) -> bool:
    return not isinstance(case.build, Unbuildable)


_ISSUES = 'https://github.com/pydantic/pydantic-ai/issues/'
_UNSET_ID_KEEPS_TOOLSET_ID = (
    f'with `id=None` the capability is anonymous, but its toolset keeps the default id: {_ISSUES}13968'
)
_SEPARATE_TOOLSET_ID = (
    'its toolset id is kept apart from the capability id, so durable operation names do not change (#10904): '
    f'{_ISSUES}13968'
)
_SKILLS_ID = (
    'its `__init__` takes no `id`, since two `Skills` combine into one; its `list_skills` and `load_skill` toolset '
    "has an id of its own, so a durable engine names their activities apart from `read_skill_file`'s"
)
_NO_DEFAULT_ID = f'it declares no default id, but its toolset has a fixed one: {_ISSUES}9233'
_TOOLSET_WITHOUT_ID = f'its toolset has no id, so a durable engine refuses it: {_ISSUES}9233'
_USERS_TOOLSET = 'its toolset is one the user built, which keeps the id the user gave it'
_WRAPPED_TOOLSET = "it wraps another capability, whose toolset keeps that capability's id"
_NO_DURABLE_TOOLSET_ID = (
    'it declares a default id, but its toolset has none; it refuses durable agents, so no durable unit is '
    f'named after it: {_ISSUES}13968'
)

UNSET_ID = Contract(
    'unset_id',
    applies=_buildable,
    exempt={
        **dict.fromkeys(
            (
                'CompactOnRequest',
                'DayAI',
                'GitHub',
                'Linear',
                'Macroscope',
                'Memory',
                'Notion',
                'Ordinal',
                'Pixeltable',
                'PostHog',
                'PruneToolResults',
                'Pylon',
                'Slack',
                'SmartSearch',
                'ToolOutputLimits',
            ),
            Exempt(_UNSET_ID_KEEPS_TOOLSET_ID),
        ),
        'Planning': Exempt(f'{_UNSET_ID_KEEPS_TOOLSET_ID}; it hardcodes the toolset id (#10904)'),
        'ConversationSearch': Exempt(_SEPARATE_TOOLSET_ID),
        'PlaywrightBrowser': Exempt(_SEPARATE_TOOLSET_ID),
        'FileSystem': Exempt(_NO_DEFAULT_ID),
        'Shell': Exempt(_NO_DEFAULT_ID),
        'PydanticAIDocs': Exempt(_TOOLSET_WITHOUT_ID),
        'Peers': Exempt(_NO_DURABLE_TOOLSET_ID),
        'Skills': Exempt(_SKILLS_ID),
    },
)
"""Built with its defaults, and with `id=None`, a capability's id is the id of every toolset it builds.

This is what both answers to what an unset `id` means (#13968) share: the default everywhere, or no id
anywhere. A capability made of others is checked through its parts, which are cases of their own.
"""

CUSTOM_ID = Contract(
    'custom_id',
    applies=_buildable,
    exempt={
        'ConversationSearch': Exempt(_SEPARATE_TOOLSET_ID),
        'PlaywrightBrowser': Exempt(_SEPARATE_TOOLSET_ID),
        'Planning': Exempt(f'its toolset hardcodes the id `planning`: {_ISSUES}13968'),
        'PydanticAIDocs': Exempt(_TOOLSET_WITHOUT_ID),
        'BrowserUse': Exempt(_TOOLSET_WITHOUT_ID),
        'Peers': Exempt(_NO_DURABLE_TOOLSET_ID),
        'MCP': Exempt(_USERS_TOOLSET),
        'Toolset': Exempt(_USERS_TOOLSET),
        'PrefixTools': Exempt(_WRAPPED_TOOLSET),
        'Skills': Exempt(_SKILLS_ID),
    },
)
"""A capability built with a custom `id` gives it to every toolset it builds, so two with distinct ids never
share a toolset id, which a durable engine names its units after."""

COMBINE_FIXED_POINT = Contract(
    'combine_fixed_point',
    applies=lambda name, case: isinstance(case.combine, Combines) and _buildable(name, case),
    exempt={
        'SubAgents': Exempt(f'stating one roster twice raises on the repeated sub-agent name: {_ISSUES}13966'),
    },
)
"""A configuration stated twice is the same as stating it once, for one of `make`'s pair and for what the pair
combines into: the same instructions, tool definitions, model settings and native tools reach the model."""

FORWARDS_USAGE = Contract('forwards_usage', applies=lambda _, case: case.calls is not None)
"""Every request a capability's calls send, nested runs included, counts toward the run's usage and stays within
its `request_limit`, which a nested run that leaves out the run's usage or limits breaks."""

WORKSPACE_FROM_FUNCTION = Contract(
    'workspace_from_function',
    applies=_buildable,
    exempt={
        'Skills': Exempt(
            'it reads its catalog in `for_run`, before the run selects a workspace a capability function supplies: '
            'https://github.com/pydantic/pydantic-ai/pull/10403'
        ),
    },
)
"""A capability that uses the run's workspace works the same when a capability function supplies it, which the run
selects only after every `for_run`."""

PENDING_CALLS = 'pending: C0 backfill'
"""How an exemption from `declares_calls` starts when the case can have `calls`, but nobody has written them yet."""

_ROUTE = f'{PENDING_CALLS}; other tools run through it, so its calls run a tool of a case beside it'

DECLARES_CALLS = Contract(
    'declares_calls',
    applies=_buildable,
    exempt={
        'Capability': Exempt(
            'a bundle of what the user passes it; each toolset and hook it holds is a case of its own'
        ),
        'DynamicCapability': Exempt('the capability its function returns runs the tools, and is a case of its own'),
        'WrapperCapability': Exempt('forwards every hook to the capability it wraps, which is a case of its own'),
        'Hooks': Exempt(
            "runs the user's hook functions and registers none by default; `tests/test_capability_hooks.py` covers each"
        ),
        'AzureDurableTaskDurability': Exempt(
            'a durable engine: its wrapper toolset runs tools only inside a Durable Task worker'
        ),
        'BubblewrappedWorkspace': Exempt('needs a sandbox backend: the `bwrap` binary and user namespaces'),
        'PlaywrightBrowser': Exempt('needs a browser that Playwright launches'),
        **dict.fromkeys(('ToolSearch',), Exempt(_ROUTE)),
        **dict.fromkeys(
            (
                'AskUser',
                'ComputerUse',
                'Peers',
                'Pixeltable',
                'RequireVerification',
                'ToolCallJudge',
                'ToolOutputLimits',
                'XSearch',
            ),
            Exempt(PENDING_CALLS),
        ),
    },
)
"""A capability that contributes a toolset, a wrapper toolset or a tool hook declares `calls` that run its tools.

Every contract that runs a capability's tools applies only to a case with `calls`, so without them it passes
vacuously. An exemption says why the tools can't run offline; one whose reason starts with `PENDING_CALLS` is a
case still to be backfilled, which `test_pending_calls_backfill` counts.
"""

Engine = Literal['dbos', 'temporal', 'prefect', 'aws_lambda', 'absurd', 'pgtask', 'azure_durable_task', 'render']
"""A durable execution engine, core or Harness."""

ENGINES: tuple[Engine, ...] = get_args(Engine)

ENGINE_CAPABILITIES: Mapping[Engine, str] = {
    'dbos': 'DBOSDurability',
    'temporal': 'TemporalDurability',
    'prefect': 'PrefectDurability',
    'aws_lambda': 'AWSLambdaDurability',
    'absurd': 'AbsurdDurability',
    'pgtask': 'PGTaskDurability',
    'azure_durable_task': 'AzureDurableTaskDurability',
    'render': 'RenderWorkflows',
}
"""The durability capability of each engine: every public `BaseDurabilityCapability` must be one of them."""

ENGINE_SUPPORT_MODULES: Mapping[Engine, str] = {
    'dbos': 'pydantic_ai.durable_exec.dbos._conformance',
    'temporal': 'pydantic_ai.durable_exec.temporal._conformance',
    'prefect': 'pydantic_ai.durable_exec.prefect._conformance',
    'aws_lambda': 'pydantic_ai_harness.aws_lambda._conformance',
    'absurd': 'pydantic_ai_harness.absurd._conformance',
    'pgtask': 'pydantic_ai_harness.pgtask._conformance',
    'azure_durable_task': 'pydantic_ai_harness.azure_durable_task._conformance',
    'render': 'pydantic_ai_harness.render._conformance',
}
"""The support table of each engine: a module with its `SUPPORT` and `RECOVERY`.

The tables live in the engines' own packages, as plain data: the `DurableEngineSuite` subclasses and the docs
generator (`scripts/durable_exec_support.py`) read them. An engine that doesn't run the suite yet lists only what is
measured elsewhere. Importing a table imports its engine's SDK, through its package.
"""

PENDING_ENGINES: tuple[Engine, ...] = ('prefect', 'aws_lambda', 'absurd', 'pgtask', 'azure_durable_task', 'render')
"""The engines that don't run `DurableEngineSuite` yet, whose tables list only what is measured elsewhere.

It may only shrink, which `test_every_durability_capability_has_a_suite` pins: a new engine joins with a suite.
"""

UNTESTED_ISSUE = 12291
"""Tracks running `DurableEngineSuite` under every engine."""

ID_LESS_TOOLSETS = ('BrowserUse', 'PydanticAIDocs')
"""The cases whose toolset has no id, which Temporal and Prefect refuse."""


Scenario = Literal[
    'BackgroundTools.runs_in_background',
    'BackgroundTools.survives_approval_wait',
    'RunContext.emit_from_tool',
    'RunContext.enqueue_from_tool',
    'RunContext.emit_in_unit',
    'RunContext.enqueue_in_unit',
    'RunContext.emit_enqueue_from_hook',
]
"""A scenario a capability, or `RunContext`, must handle under every engine, named `<owner>.<scenario>`.

A case is D1's bar on a capability's `calls`, run one per response; a scenario is a run of its own, such as a
background call kept running across a mid-run approval, which its capability's tests script. An engine's support
table has a cell for each that it doesn't support. What each scenario checks:

- `BackgroundTools.runs_in_background`: a tool marked `background` replies that it is running in the background,
  and its result arrives later as a follow-up message.
- `BackgroundTools.survives_approval_wait`: while that call is still running, the model calls a tool that needs
  approval, which a `HandleDeferredToolCalls` handler that takes a resume token answers mid-run. The call keeps
  running across the wait, without the run suspending, and its result arrives.
- `RunContext.*`: `ctx.emit` and `ctx.enqueue` from a function tool, from inside a durable unit (a
  `@durable_operation`), and from a hook. What the run observes, its custom and capability events and enqueued
  messages, must be the same after recovery or replay, each once. The public `SCENARIO_RUNS`
  (`pydantic_ai.capabilities.conformance`) scripts them, and every `DurableEngineSuite` subclass runs them.
"""

SCENARIOS: tuple[Scenario, ...] = get_args(Scenario)


@dataclass(frozen=True)
class EngineTable:
    """An engine's support table, read from its `_conformance.py`."""

    support: Mapping[str, Support]
    recovery: Recovery


def engine_table(engine: Engine) -> EngineTable:
    """The support table of `engine`, from `ENGINE_SUPPORT_MODULES`. Importing it imports the engine's SDK."""
    module = importlib.import_module(ENGINE_SUPPORT_MODULES[engine])
    support: Mapping[str, Support] = module.SUPPORT
    recovery: Recovery = module.RECOVERY
    return EngineTable(support, recovery)


class _EngineExemptions(Mapping[str, Exempt]):
    """The cases an engine's support table marks `refused` or `broken`, which its D1 contract exempts.

    It reads the table on first use, as importing it imports the engine's SDK: without the SDK, which a lane
    without the engine's suite lacks, it exempts nothing.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine: Engine = engine

    @cached_property
    def _cells(self) -> dict[str, Exempt]:
        try:
            cells = engine_table(self._engine).support
        except ImportError:  # pragma: lax no cover
            return {}
        return {
            name: Exempt(cell.reason if cell.issue is None else f'{cell.reason}: {_ISSUES}{cell.issue}')
            for name, cell in cells.items()
            if name not in SCENARIOS and cell.level in ('refused', 'broken')
        }

    def __getitem__(self, key: str) -> Exempt:
        return self._cells[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)


DURABLE_DBOS = Contract(
    'durable_dbos', applies=lambda _, case: case.calls is not None, exempt=_EngineExemptions('dbos')
)
"""D1: in a DBOS workflow, a case's calls record the steps its `durable_steps.dbos` lists, and recovering the
workflow, from after its last step or from part of its record, makes none of their requests again.

DBOS runs a function tool in workflow code, so a request the tool makes outside a `durable_operation` or a
workspace call is made again on every recovery. `TestDBOS` (`tests/harness/test_durable_tool_io.py`) checks it.
Its exemptions are the cases DBOS's support table marks `refused` or `broken`.
"""

DURABLE_TEMPORAL = Contract(
    'durable_temporal', applies=lambda _, case: case.calls is not None, exempt=_EngineExemptions('temporal')
)
"""D1: in a Temporal workflow, a case's calls schedule the activities its `durable_steps.temporal` lists, and a
`Replayer` replays the recorded history equal, without a request. `TestTemporal`
(`tests/harness/test_durable_temporal_replay.py`) checks it. Its exemptions are the cases Temporal's support table
marks `refused` or `broken`.

Temporal runs a function tool in an activity named after its toolset's `id`, so an agent with an id-less toolset is
refused, and what replays is the workflow code around it: hooks, instructions and workflow-side tools.
"""

CONTRACTS = (
    RENAMED,
    UNSET_ID,
    CUSTOM_ID,
    COMBINE_FIXED_POINT,
    FORWARDS_USAGE,
    WORKSPACE_FROM_FUNCTION,
    DECLARES_CALLS,
    DURABLE_DBOS,
    DURABLE_TEMPORAL,
)
"""Every contract, so a meta-test can check each one's exemptions."""
