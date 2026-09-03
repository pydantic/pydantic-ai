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
from typing import Any, NamedTuple, TypeGuard, cast

import pytest
from inline_snapshot import snapshot

import pydantic_ai.capabilities as capabilities_package
from pydantic_ai import Agent, FunctionToolset, RunContext, Tool
from pydantic_ai.capabilities import (
    MCP,
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
from pydantic_ai.capabilities._merge import merge_capability_fields
from pydantic_ai.capabilities._ordering import find_capability
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    _combine_duplicate_capabilities,  # pyright: ignore[reportPrivateUsage]
    _declares_default_id,  # pyright: ignore[reportPrivateUsage]
    leaf_capabilities,
)
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
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
    assert merged.settings.include_content is False, 'a scalar takes the later value'


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
        lambda: (
            Instrumentation(settings=InstrumentationSettings(include_content=True)),
            Instrumentation(settings=InstrumentationSettings(include_content=False)),
        ),
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
    except TypeError:
        return False


def _shipped_capability_types() -> dict[str, type[AbstractCapability[Any]]]:
    """Every capability class in `pydantic_ai.capabilities`, public or not."""
    found: dict[str, type[AbstractCapability[Any]]] = {}
    classes: set[type[AbstractCapability[Any]]] = set()
    for module_info in pkgutil.walk_packages(capabilities_package.__path__, f'{capabilities_package.__name__}.'):
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if (
                _is_capability_class(obj)
                and obj is not AbstractCapability
                and obj.__module__.startswith('pydantic_ai.')
            ):
                found[obj.__name__] = obj
                classes.add(obj)
    # `COMBINE_POLICY` is keyed by name, so two classes sharing one would collapse into a single
    # entry and let whichever lost ship with no policy at all -- the exact gap this guards.
    assert len(found) == len(classes), (
        f'two capability classes share a name: {sorted(cls.__module__ + "." + cls.__name__ for cls in classes)}'
    )
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
        # Anonymous capabilities declare no default id, so two never meet under one key. Read
        # through `_declares_default_id` rather than the class attribute directly, so this test
        # asks the same question the resolver does.
        assert not _declares_default_id(capability_type), (
            f'{name} is declared `Anonymous` but its instances carry a default id'
        )
        return

    assert _declares_default_id(capability_type), (
        f'{name} is declared `Combines` but declares no default id, so two never meet'
    )

    first, second = policy.make()
    assert first.id is not None and first.id == second.id, (
        f'{name} is declared `Combines` but two instances do not share an id'
    )
    policy.check(type(first).combine([first, second]))


def test_an_id_the_user_chose_twice_is_a_collision_not_a_repeat() -> None:
    """A class that declares no default `id` has not said an agent has one of it.

    So an `id` on one of its instances exists only because the user passed it, and passing the same
    one twice names two capabilities the same rather than stating one configuration twice. Merging
    would paper over the typo; the class is never asked.
    """

    @dataclass
    class Custom(AbstractCapability[Any]):
        pass

    with pytest.raises(UserError, match="Capability id 'same' is used by multiple capabilities"):
        Agent(TestModel(), capabilities=[Custom(id='same'), Custom(id='same')])


def test_a_declared_id_merges_without_an_override() -> None:
    """Declaring a default `id` is the statement that an agent has one, so a repeat merges."""

    @dataclass
    class Settings(AbstractCapability[Any]):
        effort: str | None = None
        budget: int | None = None

        _: KW_ONLY

        id: str | None = 'settings'

    merged = Settings.combine([Settings(effort='low'), Settings(budget=10)])

    assert isinstance(merged, Settings)
    assert (merged.effort, merged.budget) == ('low', 10)


async def test_one_instance_registered_twice_survives_once() -> None:
    """The same object on the agent and passed again for the run keeps exactly one occurrence.

    Keyed by object rather than occurrence, every occurrence would be handed the same replacement
    and the survivor would stay in the tree as many times as it went in -- contributing its tools
    and firing its hooks twice.
    """
    shared = Thinking(effort='low')
    tree = CombinedCapability[Any]([shared, shared])
    assert len(leaf_capabilities(tree)) == 2

    combined = _combine_duplicate_capabilities(tree, [[shared, shared]])

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
    assert _leaf_ids(capability) == [
        ('FunctionToolset', 'mycap'),
        ('DynamicToolset', 'mycap_0'),
    ]


def test_callable_toolset_id_survives_a_late_tool_registration() -> None:
    """The number is the callable's place in `toolsets=`, not its place in the composed result.

    Durable execution registers what the first call returned, so an id that moved once a `@tool`
    landed would leave it holding a name nothing answers to any more.
    """
    capability = Capability[Any](id='mycap', toolsets=[_dyn_toolset])
    assert _leaf_ids(capability) == [('DynamicToolset', 'mycap_0')]

    capability.tool_plain(_a_tool)

    assert _leaf_ids(capability) == [
        ('FunctionToolset', 'mycap'),
        ('DynamicToolset', 'mycap_0'),
    ]


def _leaf_ids(capability: Capability[Any]) -> list[tuple[str, str | None]]:
    toolset = cast('AbstractToolset[Any]', capability.get_toolset())
    leaves: list[tuple[str, str | None]] = []

    def record(ts: AbstractToolset[Any]) -> None:
        leaves.append((type(ts).__name__, ts.id))

    toolset.apply(record)
    return leaves


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
    """Which layer a duplicate came from decides, not where the tree sorted it.

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


@dataclass(eq=False)
class _Uncomparable:
    """A value whose `__eq__` raises, the way an array-like refuses elementwise comparison."""

    def __eq__(self, other: object) -> bool:
        raise ValueError('comparison is not supported')


@dataclass
class _CarriesUncomparable(AbstractCapability[Any]):
    """A capability one of whose fields holds values the merge cannot compare."""

    value: _Uncomparable | None = None

    _: KW_ONLY

    id: str | None = 'uncomparable'

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return merge_capability_fields(capabilities)


def test_a_plain_class_capability_cannot_silently_lose_its_configuration() -> None:
    """A merge keeps a value only one side states, and cannot do that for an undeclared attribute.

    A plain class keeps its configuration in plain attributes. `replace_no_init` copies the last
    instance, so such an attribute does get *a* value -- the last one -- but not the one the table
    promises, where a value only an earlier instance stated survives. Raising turns that silent
    difference into a decision.
    """

    class Retries(AbstractCapability[Any]):
        id: str | None = 'retries'

        def __init__(self, limit: int) -> None:
            self.limit = limit

    with pytest.raises(UserError, match='sets limit outside its dataclass fields'):
        Retries.combine([Retries(1), Retries(9)])


def test_the_undeclared_attribute_error_names_the_underscore_way_out() -> None:
    """The message has to name every way out, because two of the three are usually wrong.

    Internal state a `__post_init__` derives is the common case, not configuration -- declaring it
    as a field or writing a `combine` for it are both the wrong advice there, and renaming it is
    the whole fix. An error that only says "declare it as a field" sends that user the wrong way.
    """

    @dataclass
    class Derived(AbstractCapability[Any]):
        limit: int = 3
        _: KW_ONLY
        id: str | None = 'derived'

        def __post_init__(self) -> None:
            self.doubled = self.limit * 2

    with pytest.raises(UserError) as exc_info:
        Derived.combine([Derived(limit=1), Derived(limit=2)])

    message = str(exc_info.value)
    assert 'dataclass fields' in message, 'declaring it is the fix for real configuration'
    assert '`combine`' in message, 'recomputing it is the fix for state derived from merged fields'
    assert 'only silences this check' in message, 'an underscore is not a way to get derived state merged'


def test_an_underscore_silences_the_check_without_recomputing_derived_state() -> None:
    """The underscore exemption is not a fix for derived state, and the message must not imply it.

    Nothing here re-runs `__post_init__` -- `replace_no_init` exists to skip it -- so an underscored
    attribute keeps the *last* instance's value. Where the merge changed the field it derives from,
    that value is stale. A scalar hides this, because last-wins gives the same answer either way;
    only a union shows it, which is why this test unions.
    """

    @dataclass
    class Underscored(AbstractCapability[Any]):
        domains: list[str] = field(default_factory=list[str])
        _: KW_ONLY
        id: str | None = 'underscored'

        def __post_init__(self) -> None:
            self._count = len(self.domains)

        @property
        def count(self) -> int:
            """How the capability itself would read the derived value."""
            return self._count

    merged = Underscored.combine([Underscored(domains=['a']), Underscored(domains=['b'])])
    assert isinstance(merged, Underscored)
    assert merged.domains == ['a', 'b'], 'the declared field unions, as the table promises'
    assert merged.count == 1, "and the derived attribute is the last instance's, now stale"

    # Recomputing in `combine` is what actually fixes it, which is what the message says.
    @dataclass
    class Recomputes(Underscored):
        id: str | None = 'recomputes'

        @classmethod
        def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
            merged = merge_capability_fields(capabilities)
            assert isinstance(merged, Recomputes)
            merged.__post_init__()
            return merged

    recomputed = Recomputes.combine([Recomputes(domains=['a']), Recomputes(domains=['b'])])
    assert isinstance(recomputed, Recomputes)
    assert recomputed.count == 2


def test_a_field_whose_equality_raises_takes_the_later_value() -> None:
    """Values that cannot be compared are not mergeable, so the later one wins.

    `_same_value` treats an `__eq__` that raises as "different" rather than crashing the merge --
    the same answer two stores or two clients already get.
    """
    first, second = _Uncomparable(), _Uncomparable()

    merged = _CarriesUncomparable.combine([_CarriesUncomparable(value=first), _CarriesUncomparable(value=second)])

    assert isinstance(merged, _CarriesUncomparable)
    assert merged.value is second


def test_find_capability_returns_the_first_match_in_the_tree() -> None:
    """`find_capability` searches leaves in tree order, which is not the same question `combine` asks.

    It answers "is one of these present", so it stops at the first match. Anything that needs the
    capability a run will actually use has to read the combined tree instead.
    """
    first, second = Thinking(effort='low', id=None), Thinking(effort='high', id=None)
    tree = CombinedCapability[Any]([Capability[Any](), first, second])

    assert find_capability([tree], Thinking) is first
    assert find_capability([tree], WebSearch) is None


@dataclass
class _Note(AbstractCapability[Any]):
    """A capability that contributes instructions, so a repeat shows up in the prompt."""

    text: str = ''

    _: KW_ONLY

    id: str | None = 'note'

    def get_instructions(self) -> str:
        return self.text

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return capabilities[-1]


async def test_one_object_listed_twice_contributes_its_instructions_once() -> None:
    """Combining keeps one occurrence, and the composition view has to drop the others with it.

    A replacement decision keyed by object identity alone collapses the occurrences of one object
    into a single answer, so every one of them takes the surviving decision and says its piece
    again -- the capability runs once, but its instructions land twice.
    """
    note = _Note('Be brief.')
    agent = Agent(TestModel(call_tools=[]), capabilities=[note, note])

    result = await agent.run('hi')

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'Be brief.'


def test_an_id_two_classes_claim_is_rejected_whichever_one_comes_first() -> None:
    """Whether an id may repeat is a property of the pair, not of the capability that came second.

    No class can be handed another's instances to combine, so a class-crossing id is unresolvable
    either way round -- and reading the answer off the later capability alone let one order build
    an agent that only failed once it ran.
    """
    message = r"Capability id 'shared' is used by capabilities of different types \(Thinking, _Note\)"

    with pytest.raises(UserError, match=message):
        Agent(TestModel(), capabilities=[Thinking(id='shared'), _Note(id='shared')])

    with pytest.raises(UserError, match=message):
        Agent(TestModel(), capabilities=[_Note(id='shared'), Thinking(id='shared')])


async def test_two_native_capabilities_on_one_agent_merge_rather_than_collide() -> None:
    """Duplicates the agent was constructed with are resolved before anything reads them.

    Native tools are keyed by the tool's own id, not the capability's, so two `WebSearch`
    capabilities on one agent looked to `_validate_native_tool_ids` like one id with two
    definitions and were rejected at `Agent(...)` -- while the same two supplied one per layer
    merged as designed. Combining the agent's own list as it is assembled is what makes the rule
    the docs state ("two on the agent") true of the case they name.
    """
    seen: list[Sequence[Any]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[WebSearch(search_context_size='low'), WebSearch(max_uses=3)],
    )

    await agent.run('hi')

    assert seen == snapshot([[WebSearchTool(search_context_size='low', max_uses=3)]])


def test_a_chain_of_wrappers_walks_its_subtree_once_per_level() -> None:
    """A wrapper asks what its subtree registers and then registers it, from one walk, not two.

    `apply` used to walk the subtree once to ask what it registers and then walk that same
    subtree again to visit it. Two walks per level is `2 ** depth` traversals for a chain of
    wrappers, so a stack of `prefix_tools()` calls over a container
    stopped resolving in any reasonable time.
    """
    walks = 0

    class _CountingContainer(CombinedCapability[Any]):
        def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
            nonlocal walks
            walks += 1
            super().apply(visitor)

    capability: AbstractCapability[Any] = _CountingContainer([Capability[Any](id='a'), Capability[Any](id='b')])
    for _ in range(6):
        capability = WrapperCapability[Any](wrapped=capability)

    # Building the container walked it once, to sort its leaves into ordering tiers.
    walks = 0
    seen: list[AbstractCapability[Any]] = []
    capability.apply(seen.append)

    assert walks == 1
    assert len(seen) == 8


@pytest.mark.parametrize('capability_type', [ImageGeneration, XSearch])
def test_a_merge_cannot_reach_a_combination_the_constructor_rejects(
    capability_type: type[ImageGeneration[Any]] | type[XSearch[Any]],
) -> None:
    """`fallback_model` and `local` are alternatives, and merging two instances must not pair them.

    Each states one half of a combination `__init__` refuses, so the merged capability would carry
    both -- and the local tool would take effect while `fallback_model` was silently ignored. The
    invariant lives in `__post_init__`, which `combine` re-runs, rather than in `__init__`, which
    it cannot.
    """
    with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
        capability_type.combine(
            [
                capability_type(fallback_model=TestModel()),
                capability_type(local=_a_local_tool),
            ]
        )


def _a_local_tool(prompt: str) -> str:  # pragma: no cover
    """A local fallback."""
    return 'x'


def test_a_merged_collection_keeps_the_type_the_field_declared() -> None:
    """A union is computed in a plain `list`/`set`, but the field keeps the type it was annotated.

    `replace_no_init` skips `__post_init__`, so a `tuple[str, ...]` field handed a `list` would
    survive as one and fail somewhere downstream instead of here.
    """

    @dataclass
    class Collections(AbstractCapability[Any]):
        ordered: tuple[str, ...] = ()
        unique: frozenset[str] = frozenset()
        _: KW_ONLY
        id: str | None = 'collections'

    merged = Collections.combine(
        [Collections(ordered=('a',), unique=frozenset({'a'})), Collections(ordered=('b',), unique=frozenset({'b'}))]
    )
    assert isinstance(merged, Collections)
    assert merged.ordered == ('a', 'b')
    assert type(merged.ordered) is tuple
    assert merged.unique == frozenset({'a', 'b'})
    assert type(merged.unique) is frozenset


def test_a_collection_that_cannot_be_rebuilt_keeps_the_plain_merge() -> None:
    """A `NamedTuple` takes its fields positionally, so rebuilding it from a list raises.

    Merging keeps the plain value rather than turning a type mismatch into a `TypeError`. Two of
    these are not really a union anyway -- a `NamedTuple` is a record, not a collection.
    """

    class Pair(NamedTuple):
        left: str
        right: str

    @dataclass
    class Record(AbstractCapability[Any]):
        pair: Pair = Pair('a', 'b')
        _: KW_ONLY
        id: str | None = 'record'

    merged = Record.combine([Record(pair=Pair('a', 'b')), Record(pair=Pair('c', 'd'))])
    assert isinstance(merged, Record)
    assert merged.pair == ['a', 'b', 'c', 'd']


async def test_a_second_local_search_tool_replaces_the_first() -> None:
    """`local` names one fallback, so two under one id resolve like any other scalar: later wins.

    Two independent local search tools was never a configuration worth keeping -- an agent searches
    one way -- so this is the dictionary rule applied to a key, not a special case. A user who does
    want two search tools names them apart with an explicit `id=`, and both survive.
    """
    offered: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append(sorted(tool.name for tool in info.function_tools))
        return ModelResponse(parts=[TextPart('done')])

    def alpha(query: str) -> str:  # pragma: no cover
        """Search alpha."""
        return query

    def beta(query: str) -> str:  # pragma: no cover
        """Search beta."""
        return query

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            WebSearch(native=False, local=Tool(alpha, name='alpha')),
            WebSearch(native=False, local=Tool(beta, name='beta')),
        ],
    )
    await agent.run('hi')
    assert offered == [['beta']]

    named_apart = Agent(
        FunctionModel(model_fn),
        capabilities=[
            WebSearch(native=False, local=Tool(alpha, name='alpha')),
            WebSearch(native=False, local=Tool(beta, name='beta'), id='second_search'),
        ],
    )
    offered.clear()
    await named_apart.run('hi')
    assert offered == [['alpha', 'beta']]


async def test_a_run_level_capability_replaces_the_agent_level_one_whole() -> None:
    """Across layers the later one overrides; `combine` is not consulted.

    A run states what *this* run does. Merging would let an agent-level allow-list widen the very
    restriction the run was passed to impose, and would leave agent-level settings in place that
    the run's configuration replaced.
    """
    seen: list[Sequence[Any]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[WebSearch(allowed_domains=['agent.example'], max_uses=5)],
    )

    await agent.run('hi', capabilities=[WebSearch(allowed_domains=['run.example'])])

    assert seen == snapshot([[WebSearchTool(allowed_domains=['run.example'])]])


async def test_a_session_level_instrumentation_supersedes_the_agent_level_one() -> None:
    """Across the agent-to-session boundary the last one stated is selected, not combined.

    Which capability's `include_content` governs exported content is a privacy decision, so the
    session reads the settings the run would keep -- the last explicit `Instrumentation`, the same
    precedence the tool spans get. Taking the first would drive the session and chat spans from
    settings the effective configuration had already turned off.
    """
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.realtime import (
        RealtimeModel,
        RealtimeModelProfile,
        RealtimeModelSettings,
    )
    from pydantic_ai.realtime.codec import RealtimeConnection

    class _StubRealtimeModel(RealtimeModel):
        """A `RealtimeModel` in name only: session resolution must never open a connection."""

        @property
        def model_name(self) -> str:
            return 'stub_realtime'  # pragma: no cover

        @property
        def system(self) -> str:
            return 'stub-realtime'  # pragma: no cover

        @property
        def name(self) -> str:
            return 'stub_realtime'  # pragma: no cover

        @property
        def profile(self) -> RealtimeModelProfile:
            return RealtimeModelProfile()

        def connect(
            self,
            *,
            messages: Sequence[ModelMessage],
            model_settings: RealtimeModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> AbstractAsyncContextManager[RealtimeConnection]:  # pragma: no cover
            raise AssertionError('session resolution must not open a connection')

    agent = Agent(TestModel(), capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=True))])
    async with agent._resolve_realtime_session(  # pyright: ignore[reportPrivateUsage]
        _StubRealtimeModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=False))],
    ) as resolution:
        assert resolution.instrumentation_settings is not None
        assert resolution.instrumentation_settings.include_content is False, 'the session-level one wins'
        assert resolution.run_context.trace_include_content is False


async def test_two_capabilities_on_one_agent_merge_rather_than_override() -> None:
    """Within a layer they are one configuration stated twice, so both sides' domains survive.

    Two packaged capabilities each bringing a `WebSearch` is the shape this exists for: an agent
    composed of a coder and a researcher should reach the union of what each was allowed.
    """
    seen: list[Sequence[Any]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            WebSearch(allowed_domains=['stackoverflow.com', 'github.com']),
            WebSearch(allowed_domains=['wikipedia.org']),
        ],
    )

    await agent.run('hi')

    assert seen == snapshot([[WebSearchTool(allowed_domains=['stackoverflow.com', 'github.com', 'wikipedia.org'])]])


async def test_mutually_exclusive_fields_survive_a_run_level_override() -> None:
    """Overriding across layers never builds a combination no constructor would accept.

    `XSearch` refuses `allowed_x_handles` beside `excluded_x_handles`. Merging an agent-level
    instance carrying one into a run-level instance carrying the other produced exactly that, so a
    run that narrowed the handles raised instead of taking effect.
    """
    seen: list[Sequence[Any]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[XSearch(allowed_x_handles=['pydantic'])])

    await agent.run('hi', capabilities=[XSearch(excluded_x_handles=['spam'])])

    assert seen == snapshot([[XSearchTool(excluded_x_handles=['spam'])]])


async def test_an_id_two_capabilities_only_share_after_for_run_is_still_a_collision() -> None:
    """The run resolves duplicates before it validates ids, so this is caught there.

    `Agent(...)` sees only what it was handed, and these two agree on nothing at construction --
    the shared `id` appears when `for_run` hands back the capability the run actually uses. That is
    past the construction-time check, so the resolver applies the same rule itself.
    """

    @dataclass
    class Renaming(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return Renaming(id='chosen')

    agent = Agent(TestModel(call_tools=[]), capabilities=[Renaming(), Renaming()])

    with pytest.raises(UserError, match="Capability id 'chosen' is used by multiple capabilities"):
        await agent.run('hi')


async def test_two_capabilities_supplied_for_one_run_merge_like_two_on_the_agent() -> None:
    """A layer is a layer, whether the agent was constructed with it or a run supplied it.

    Native tools are keyed by the tool's own id, so reading them off the layer as supplied showed
    one id with two definitions and rejected a pair the run goes on to combine -- the agent's own
    layer is resolved in `__init__`, but a run's is only assembled at run setup (#6705).
    """
    seen: list[Sequence[Any]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    await agent.run('hi', capabilities=[WebSearch(search_context_size='low'), WebSearch(max_uses=3)])

    assert seen == snapshot([[WebSearchTool(search_context_size='low', max_uses=3)]])


def test_mcp_takes_the_same_derived_id_as_the_toolset_it_contributes() -> None:
    """A server's identity is its URL, so the capability is named by it too, not just its leaf.

    `MCP` already derived a stable id for the `MCPToolset` it builds, while the capability itself
    stayed anonymous and fell back to a positional `mcp` / `mcp_2` -- which reorders when the
    capability list does, so it is no use as a durable-operation name or an instruction key
    (following up #6334, which fixed the toolset half).
    """
    capability = MCP[Any](url='https://mcp.example.com/sse')

    assert capability.id == snapshot('mcp.example.com-sse')
    assert cast('AbstractToolset[Any]', capability.get_toolset()).id == capability.id

    # An explicit `id=` still wins, and a client that carries its own connection has nothing to
    # derive from, so it stays anonymous and the run tells duplicates apart itself.
    assert MCP[Any](url='https://mcp.example.com/sse', id='docs').id == 'docs'
    assert MCP[Any](local=lambda: FunctionToolset[Any]()).id is None


def test_a_deferred_mcp_capability_still_demands_an_id_of_its_own() -> None:
    """A deferred capability's id is shown to the model, so it may not be derived from the URL.

    The `load_capability` catalog lists every deferred capability by `id` as a dynamic instruction,
    and the derived id carries the URL's last path segment. Deriving one for a deferred `MCP` would
    put a signed path -- or a token-in-path server's token -- in the prompt, where a model can be
    talked into repeating it. Naming a durable operation is ours to do; naming something the model
    reads is the user's, so this keeps raising exactly as it did before ids were derived at all.
    """
    assert MCP[Any](url='https://mcp.example.com/s/sk-live-secret', defer_loading=True).id is None

    with pytest.raises(UserError, match='Deferred capabilities must use stable explicit `id` values'):
        Agent(
            TestModel(),
            capabilities=[MCP[Any](url='https://mcp.example.com/s/sk-live-secret', defer_loading=True)],
        )

    # An explicit `id=` is all it ever needed, and the URL is then nowhere near the prompt.
    deferred = MCP[Any](url='https://mcp.example.com/s/sk-live-secret', defer_loading=True, id='docs')
    assert deferred.id == 'docs'
