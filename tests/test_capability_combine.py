"""How two capabilities that resolve to the same `id` compose.

Every capability Pydantic AI ships is listed in `COMBINE_POLICY`, and
`test_every_capability_declares_a_combine_policy` fails when one is missing. Adding a capability is
therefore a decision about what two of it mean, taken once, here -- not something that defaults
quietly to whatever `AbstractCapability` happens to do.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, TypeGuard, cast

import pytest

import pydantic_ai.capabilities as capabilities_package
from pydantic_ai import Agent, FunctionToolset, RunContext, Tool
from pydantic_ai.agent import find_capability
from pydantic_ai.capabilities import (
    Capability,
    CapabilityOrdering,
    ImageGeneration,
    Instrumentation,
    RaiseContentFilterError,
    ReinjectSystemPrompt,
    Thinking,
    ToolSearch,
    UseThreadExecutor,
    WebFetch,
    WebSearch,
    XSearch,
)
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    combine_duplicate_capabilities,
    leaf_capabilities,
    merge_capability_fields,
)
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool, XSearchTool
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

pytestmark = pytest.mark.anyio


@dataclass
class Anonymous:
    """No default `id`: two of these are two different things, so `combine` is never reached.

    The run derives a distinct id per occurrence instead. A user who gives two the same `id`
    explicitly gets the base `combine`, which raises -- that is a mistake, not a composition.
    """

    reason: str


@dataclass
class Combines:
    """A default `id`: two of these are one configuration stated twice, and `combine` resolves them."""

    reason: str
    make: Callable[[], tuple[AbstractCapability[Any], AbstractCapability[Any]]]
    """Builds two instances that state *different* configuration, so a merge is observable."""
    check: Callable[[Any], None]
    """Asserts what survived. Reads derived state too, not just the declared fields."""


Policy = Anonymous | Combines


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


def _check_x_search(merged: XSearch) -> None:
    assert merged.allowed_x_handles == ['a', 'b']
    assert isinstance(merged.native, XSearchTool)
    assert merged.native.allowed_x_handles == ['a', 'b']


def _check_image_generation(merged: ImageGeneration) -> None:
    assert merged.quality == 'high'


def _check_instrumentation(merged: Instrumentation) -> None:
    assert merged.settings is not None


_FIRST_EXECUTOR = ThreadPoolExecutor(1, 'first')
_SECOND_EXECUTOR = ThreadPoolExecutor(1, 'second')


def _check_tool_search(merged: ToolSearch) -> None:
    assert merged.max_results == 20, 'a scalar takes the later value'


def _check_thread_executor(merged: UseThreadExecutor) -> None:
    assert merged.executor is _SECOND_EXECUTOR, 'the executor that would have shadowed the other'


COMBINE_POLICY: dict[str, Policy] = {
    # -- One per agent: a default `id`, and `combine` says what two of them mean. --
    'Thinking': Combines(
        'an agent has one thinking configuration',
        lambda: (Thinking(effort='low'), Thinking(effort='high')),
        _check_thinking,
    ),
    'WebSearch': Combines(
        'one web search configuration, but its allow-list must not be silently widened',
        lambda: (WebSearch(allowed_domains=['a.com']), WebSearch(allowed_domains=['b.com'])),
        _check_web_search,
    ),
    'WebFetch': Combines(
        'one web fetch configuration, same allow-list concern as `WebSearch`',
        lambda: (WebFetch(allowed_domains=['a.com']), WebFetch(allowed_domains=['b.com'])),
        _check_web_fetch,
    ),
    'XSearch': Combines(
        'one X search configuration',
        lambda: (
            XSearch(fallback_model='xai:grok-4.3', allowed_x_handles=['a']),
            XSearch(fallback_model='xai:grok-4.3', allowed_x_handles=['b']),
        ),
        _check_x_search,
    ),
    'ImageGeneration': Combines(
        'one image generation configuration',
        lambda: (
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', quality='low'),
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', quality='high'),
        ),
        _check_image_generation,
    ),
    'Instrumentation': Combines(
        'an agent is instrumented one way',
        lambda: (Instrumentation(), Instrumentation()),
        _check_instrumentation,
    ),
    'ReinjectSystemPrompt': Combines(
        'one reinjection policy per agent',
        lambda: (ReinjectSystemPrompt(), ReinjectSystemPrompt(replace_existing=True)),
        _check_reinject,
    ),
    'RaiseContentFilterError': Combines(
        'carries no configuration at all, so two are interchangeable',
        lambda: (RaiseContentFilterError(), RaiseContentFilterError()),
        _check_content_filter,
    ),
    'ToolSearch': Combines(
        'one tool-discovery configuration per agent',
        lambda: (ToolSearch(max_results=5), ToolSearch(max_results=20)),
        _check_tool_search,
    ),
    'UseThreadExecutor': Combines(
        'exactly one executor is in effect; nesting already made this last-wins implicitly',
        lambda: (UseThreadExecutor(_FIRST_EXECUTOR), UseThreadExecutor(_SECOND_EXECUTOR)),
        _check_thread_executor,
    ),
    # -- Several of these is the normal case, so they stay anonymous. --
    'Capability': Anonymous('a generic bundle; several per agent is the usual shape'),
    'CombinedCapability': Anonymous('structural container; nesting is the semantic'),
    'WrapperCapability': Anonymous('structural wrapper; nesting is the semantic'),
    'PrefixTools': Anonymous('structural wrapper, applied once per wrapped capability'),
    'DynamicCapability': Anonymous('one per capability function'),
    'ResolvedDynamicCapability': Anonymous('the resolved form of a `DynamicCapability`'),
    'NativeTool': Anonymous('one per native tool'),
    'NativeOrLocalTool': Anonymous('used directly it is parameterized by the tools passed to it'),
    'MCP': Anonymous(
        'several servers per agent is the normal case; the URL derives its *toolset* id, not a capability id'
    ),
    'Toolset': Anonymous('one per toolset'),
    'Hooks': Anonymous('several hook bundles compose'),
    'HandleDeferredToolCalls': Anonymous('`CombinedCapability` chains handlers via `remaining`'),
    'ResolveModelId': Anonymous('returns `None` to let a later capability resolve; chaining is the feature'),
    'SelectModel': Anonymous('receives the lower-precedence model; chaining is designed'),
    'ProcessHistory': Anonymous('history processors stack'),
    'ProcessEventStream': Anonymous('event-stream processors stack'),
    'PrepareTools': Anonymous('tool preparers stack'),
    'PrepareOutputTools': Anonymous('output-tool preparers stack'),
    'SetToolMetadata': Anonymous('one per `ToolSelector`; several selectors compose'),
    'IncludeToolReturnSchemas': Anonymous('one per `ToolSelector`; several selectors compose'),
    'DeferredCapabilityLoader': Anonymous('auto-injected only when absent'),
    'PendingMessageDrainCapability': Anonymous('auto-injected only when absent'),
}


def _is_capability_class(obj: object) -> TypeGuard[type[AbstractCapability[Any]]]:
    """Whether `obj` is a capability class, and not something that merely looks like one.

    A module's namespace holds type aliases and parameterized generics beside its classes, and on
    Python 3.10 some of those satisfy `inspect.isclass` while `issubclass` then raises on them.
    """
    if not isinstance(obj, type):
        return False
    try:
        return issubclass(obj, AbstractCapability)
    except TypeError:  # pragma: no cover
        return False


def _shipped_capability_types() -> dict[str, type[AbstractCapability[Any]]]:
    """Every capability class in `pydantic_ai.capabilities`, public or not."""
    found: dict[str, type[AbstractCapability[Any]]] = {}
    for module_info in pkgutil.walk_packages(capabilities_package.__path__, f'{capabilities_package.__name__}.'):
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if (
                _is_capability_class(obj)
                and obj is not AbstractCapability
                and obj.__module__.startswith('pydantic_ai.')
            ):
                found[obj.__name__] = obj
    return found


def test_every_capability_declares_a_combine_policy() -> None:
    """A new capability must say what two of it mean before it can ship.

    Without this the answer defaults to whatever the base class does, which is the one outcome
    nobody chose. Add an entry to `COMBINE_POLICY` -- `Anonymous` when several per agent is normal,
    `Combines` when it carries a default `id`.
    """
    shipped = set(_shipped_capability_types())
    declared = set(COMBINE_POLICY)
    assert not (shipped - declared), (
        f'capabilities with no `COMBINE_POLICY` entry: {sorted(shipped - declared)}. '
        'Decide what two of them mean and add an entry.'
    )
    assert not (declared - shipped), (
        f'`COMBINE_POLICY` names capabilities that no longer exist: {sorted(declared - shipped)}.'
    )


@pytest.mark.parametrize('name', sorted(COMBINE_POLICY))
def test_capability_combine_policy_holds(name: str) -> None:
    """Each capability composes -- or refuses to -- the way its policy says."""
    policy = COMBINE_POLICY[name]
    capability_type = _shipped_capability_types()[name]

    if isinstance(policy, Anonymous):
        # Anonymous capabilities carry no default id, so two never meet under one key.
        assert capability_type.id is None, (
            f'{name} is declared `Anonymous` but carries a default id {capability_type.id!r}'
        )
        return

    first, second = policy.make()
    assert first.id is not None and first.id == second.id, (
        f'{name} is declared `Combines` but two instances do not share an id'
    )
    policy.check(type(first).combine([first, second]))


def test_base_combine_rejects_duplicates() -> None:
    """A capability that has not said how it composes refuses to guess."""

    @dataclass
    class Custom(AbstractCapability[Any]):
        pass

    with pytest.raises(UserError, match='is used by'):
        Custom.combine([Custom(id='same'), Custom(id='same')])


async def test_one_instance_registered_twice_survives_once() -> None:
    """The same object on the agent and passed again for the run keeps exactly one occurrence.

    Keyed by object rather than occurrence, every occurrence would be handed the same replacement
    and the survivor would stay in the tree as many times as it went in -- contributing its tools
    and firing its hooks twice.
    """
    shared = Thinking(effort='low')
    tree = CombinedCapability[Any]([shared, shared])
    assert len(leaf_capabilities(tree)) == 2

    combined = combine_duplicate_capabilities(tree)

    leaves = leaf_capabilities(combined)
    assert [(type(leaf).__name__, leaf.id) for leaf in leaves] == [('Thinking', 'thinking')]


def test_merging_into_a_contradictory_configuration_is_rejected() -> None:
    """A merge can reach a combination no constructor would accept, and must fail the same way.

    `replace_no_init` skips `__post_init__`, so without re-running it the merged capability
    contributes neither the native tool (`native=False`) nor a local fallback (suppressed because
    native-only constraints are set), and does so silently.
    """
    with pytest.raises(UserError, match='constraint fields require the native tool'):
        WebSearch.combine([WebSearch(allowed_domains=['a.com']), WebSearch(native=False, local='duckduckgo')])


def _dyn_toolset(ctx: RunContext[Any]) -> FunctionToolset[Any]:  # pragma: no cover
    """A `toolsets=` callable, resolved per run."""
    return FunctionToolset([_a_tool])


def _a_tool() -> str:  # pragma: no cover
    """A tool."""
    return 'x'


def test_capability_id_reaches_a_callable_toolset() -> None:
    """An explicit `Capability(id=...)` names every leaf it contributes, not just the function one.

    Durable execution identifies a leaf toolset by `id`, so a `toolsets=` callable left anonymous
    made a capability the user *had* named unusable there (#7274). One capability can contribute
    several leaves, so the position within its own arguments keeps them apart.
    """
    capability = Capability[Any](id='mycap', tools=[_a_tool], toolsets=[_dyn_toolset])
    toolset = cast('AbstractToolset[Any]', capability.get_toolset())
    leaves: list[tuple[str, str | None]] = []

    def record(ts: AbstractToolset[Any]) -> None:
        leaves.append((type(ts).__name__, ts.id))

    toolset.apply(record)
    assert leaves == [
        ('FunctionToolset', 'mycap'),
        ('DynamicToolset', 'mycap_1'),
    ]


def test_anonymous_capability_leaves_its_toolsets_anonymous() -> None:
    """`id=None` states nothing to pass down, so the contributed toolsets stay unnamed."""
    capability = Capability[Any](toolsets=[_dyn_toolset])
    toolset = capability.get_toolset()
    assert isinstance(toolset, DynamicToolset)
    assert toolset.id is None


def test_merged_local_fallback_carries_the_merged_configuration() -> None:
    """The local tool enforces the merged domains too, not only the last capability's.

    On a provider without native fetch the local fallback is what runs, and it carries its own copy
    of the domain lists. Rebuilding only the native tool left the fallback enforcing whatever the
    last capability declared -- a merged `blocked_domains` that the fallback never applied.
    """
    merged = WebFetch.combine(
        [WebFetch(local=True, allowed_domains=['a.com']), WebFetch(local=True, allowed_domains=['b.com'])]
    )
    assert isinstance(merged, WebFetch)
    local = merged.local
    assert isinstance(local, Tool)
    # The fallback is a bound method of the fetcher, which carries its own copy of the domain lists.
    fetcher = cast('Any', local).function.__self__
    assert fetcher.allowed_domains == ['a.com', 'b.com']


async def test_a_later_layer_wins_even_when_it_sorts_first() -> None:
    """Application order decides which duplicate is later, not the ordering-sorted tree.

    `CombinedCapability` sorts leaves into ordering tiers, so a capability supplied for the run but
    positioned `'outermost'` moves ahead of the agent-level one. Reading "last" off the tree then
    picks the agent-level capability and the run's override silently loses.
    """
    seen: dict[str, AbstractCapability[Any]] = {}

    @dataclass
    class Probe(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            seen.update(ctx.capabilities)

    agent = Agent(TestModel(), capabilities=[_Positioned(id='m', tag='agent'), Probe()])
    await agent.run('hi', capabilities=[_Positioned(id='m', tag='run', outermost=True)])

    assert isinstance(seen['m'], _Positioned)
    assert seen['m'].tag == 'run'


@dataclass
class _Positioned(AbstractCapability[Any]):
    """A capability whose ordering tier can differ per instance, so the two sorts can disagree."""

    tag: str = ''
    outermost: bool = False

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost') if self.outermost else CapabilityOrdering()

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return merge_capability_fields(capabilities)


@dataclass
class _Collections(AbstractCapability[Any]):
    """A capability whose configuration is collections, to pin how the merge unions them."""

    tags: set[str] = field(default_factory=set[str])
    labels: dict[str, str] = field(default_factory=dict[str, str])

    _: KW_ONLY

    id: str | None = 'collections'

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return merge_capability_fields(capabilities)


def test_collections_merge_as_unions() -> None:
    """Sets union and mappings merge, with a key stated on both sides taking the later value."""
    merged = _Collections.combine(
        [
            _Collections(tags={'a'}, labels={'shared': 'first', 'only-first': 'x'}),
            _Collections(tags={'b'}, labels={'shared': 'second', 'only-second': 'y'}),
        ]
    )
    assert isinstance(merged, _Collections)
    assert merged.tags == {'a', 'b'}
    assert merged.labels == {'shared': 'second', 'only-first': 'x', 'only-second': 'y'}


def test_find_capability_returns_the_first_match_in_the_tree() -> None:
    """`find_capability` searches leaves in tree order, which is not the same question `combine` asks.

    It answers "is one of these present", so it stops at the first match. Anything that needs the
    capability a run will actually use has to read the combined tree instead.
    """
    first, second = Thinking(effort='low', id=None), Thinking(effort='high', id=None)
    tree = CombinedCapability[Any]([Capability[Any](), first, second])

    assert find_capability([tree], Thinking) is first
    assert find_capability([tree], WebSearch) is None
