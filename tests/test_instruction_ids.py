"""Tests for [`InstructionPart.name`][pydantic_ai.messages.InstructionPart.name] and
[`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id].

Instruction parts carry a stable key so a consumer that receives
`ModelRequestParameters.instruction_parts` can address them — e.g. to override their text from a
remote configuration — without depending on their position or wording. An author declares a `name`;
the framework issues an `id`. That key is an `InstructionId`: its source (`AgentInstructionSource`,
`ToolsetInstructionSource`, `CapabilityInstructionSource`) addresses everything that source
contributes, and its optional `name` addresses one named part within it. Naming a part whose source
has no identity of its own leaves `id` as `None`, so the name says what the part is without making
it addressable. Keys render and serialize as `agent`, `toolset:x`, `capability:x:y` — the form
persisted configuration keys on.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, ModelRequestContext
from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CapabilityOrdering,
    CombinedCapability,
    Hooks,
    WrapperCapability,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentInstructionSource,
    CapabilityInstructionSource,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolsetInstructionSource,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.prepared import PreparedToolset
from pydantic_ai.toolsets.renamed import RenamedToolset

from .conftest import try_import

with try_import() as mcp_imports_successful:
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset

pytestmark = pytest.mark.anyio

instruction_part_ta = TypeAdapter(InstructionPart)


def agent_instruction_id(name: str | None = None) -> InstructionId:
    return InstructionId(AgentInstructionSource(), name=name)


def toolset_instruction_id(id: str, name: str | None = None) -> InstructionId:
    return InstructionId(ToolsetInstructionSource(id), name=name)


def capability_instruction_id(id: str, name: str | None = None) -> InstructionId:
    return InstructionId(CapabilityInstructionSource(id), name=name)


class InstructionsToolset(AbstractToolset[Any]):
    """A toolset with no tools that contributes whatever its `get_instructions` is given."""

    def __init__(
        self,
        instructions: str | InstructionPart | Sequence[str | InstructionPart] | None = None,
        id: str | None = None,
    ):
        self.instructions = instructions
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_instructions(
        self, ctx: RunContext[Any]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        return self.instructions

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


def capture_instruction_parts() -> tuple[FunctionModel, list[InstructionPart]]:
    """A model that records the instruction parts it was sent."""
    captured: list[InstructionPart] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(info.model_request_parameters.instruction_parts or [])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_fn), captured


def rendered_instructions(messages: list[ModelMessage]) -> str | None:
    """The single instructions string the blocks were rendered into."""
    request = messages[0]
    assert isinstance(request, ModelRequest)
    return request.instructions


async def run_and_capture(agent: Agent[Any, str], **kwargs: Any) -> list[InstructionPart]:
    model, captured = capture_instruction_parts()
    await agent.run('Hello', model=model, **kwargs)
    return captured


async def test_toolset_instructions_are_identified_by_toolset_id():
    agent = Agent(
        instructions='Agent instructions.',
        toolsets=[
            InstructionsToolset('Weather instructions.', id='weather'),
            InstructionsToolset('Calendar instructions.', id='calendar'),
            InstructionsToolset('Anonymous instructions.'),
        ],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='Weather instructions.', dynamic=True, id=toolset_instruction_id('weather')),
        InstructionPart(content='Calendar instructions.', dynamic=True, id=toolset_instruction_id('calendar')),
        InstructionPart(content='Anonymous instructions.', dynamic=True),
    ]


@pytest.mark.parametrize(
    'instructions',
    [
        'Weather instructions.',
        InstructionPart(content='Weather instructions.', dynamic=True),
        ['Weather instructions.'],
        [InstructionPart(content='Weather instructions.', dynamic=True)],
    ],
    ids=['str', 'part', 'str-sequence', 'part-sequence'],
)
async def test_every_toolset_instructions_shape_is_identified(
    instructions: str | InstructionPart | Sequence[str | InstructionPart],
):
    agent = Agent(toolsets=[InstructionsToolset(instructions, id='weather')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Weather instructions.', dynamic=True, id=toolset_instruction_id('weather'))
    ]


async def test_toolset_can_declare_ids_for_its_own_blocks():
    """A toolset has no `id=` parameter for its blocks, so it qualifies its own key on the part.

    An id the toolset set itself is closer to the source, so composition leaves it alone.
    """
    agent = Agent(
        toolsets=[
            InstructionsToolset(
                [
                    InstructionPart(
                        content='Tool usage.', name='limits', id=toolset_instruction_id('weather', 'limits')
                    ),
                    InstructionPart(content='General.'),
                ],
                id='weather',
            )
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Tool usage.', name='limits', id=toolset_instruction_id('weather', 'limits')),
        InstructionPart(content='General.', id=toolset_instruction_id('weather')),
    ]


@pytest.mark.parametrize('declared_id', ['toolset:ghost', 'agent'])
async def test_an_unidentified_toolset_cannot_mint_a_toolset_key(declared_id: str):
    """Names that flatten like framework keys are rejected at their authoring boundary."""
    agent = Agent(toolsets=[InstructionsToolset(InstructionPart(content='Minted.', name=declared_id))])

    with pytest.raises(UserError, match=r'cannot contain a colon|is reserved'):
        await run_and_capture(agent)


async def test_wrapped_toolset_instructions_keep_their_id():
    """Wrappers have no id of their own, so the wrapped toolset's identity must survive them."""
    agent = Agent(
        toolsets=[
            InstructionsToolset('Weather instructions.', id='weather')
            .prefixed('w')
            .filtered(lambda ctx, tool_def: True)
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Weather instructions.', dynamic=True, id=toolset_instruction_id('weather'))
    ]


class RelayingWrapper(WrapperToolset[Any]):
    """A wrapper that passes what it wraps back out, the usual reason to override `get_instructions`."""

    def __init__(self, wrapped: AbstractToolset[Any], id: str | None = None):
        super().__init__(wrapped)
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_instructions(
        self, ctx: RunContext[Any]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        return await super().get_instructions(ctx)


class RelayingWrapperReturning(WrapperToolset[Any]):
    """A wrapper returning instructions of its own choosing rather than what it wraps."""

    def __init__(
        self, wrapped: AbstractToolset[Any], instructions: Sequence[str | InstructionPart], id: str | None = None
    ):
        super().__init__(wrapped)
        self.instructions = instructions
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_instructions(
        self, ctx: RunContext[Any]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        return self.instructions


async def test_a_named_wrapper_relays_the_wrapped_key_rather_than_nesting_it():
    """A wrapper with an `id` must not re-resolve what it wraps beneath its own key.

    A key belongs to the toolset it names, so a wrapper handing one back out is relaying, not
    declaring. Reading the wrapped key as a declared segment would qualify it to
    `'toolset:wrapper:toolset:leaf'` — which the colon rule rejects outright, so the toolset stops
    working rather than merely being misfiled.
    """
    agent = Agent(toolsets=[RelayingWrapper(InstructionsToolset('Leaf instructions.', id='leaf'), id='wrapper')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Leaf instructions.', dynamic=True, id=toolset_instruction_id('leaf'))
    ]


async def test_a_wrapper_relays_a_key_from_under_a_wrapper_it_wraps():
    """Wrappers nest, so what a wrapper may relay is the whole subtree's keys, not its child's.

    Reading only the toolset directly wrapped — or only the leaves — would leave the outer wrapper
    treating `'toolset:leaf'` as a segment its own author had declared, which drops the key, or
    rejects its colon where the outer wrapper has an `id` of its own to qualify it against.
    """
    inner = RelayingWrapper(InstructionsToolset('Leaf.', id='leaf'), id='inner')
    agent = Agent(toolsets=[RelayingWrapper(inner, id='outer')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Leaf.', dynamic=True, id=toolset_instruction_id('leaf'))
    ]


async def test_a_wrapper_relays_keys_from_every_toolset_it_wraps_at_once():
    """A wrapper's subtree spreads sideways as well as down, so every branch's keys have to survive.

    A capability contributing more than one toolset is the shape this shows up in: the group arrives
    behind one wrapper, and relaying it must not favour whichever branch happens to come first.
    """
    grouped = CombinedToolset([InstructionsToolset('First.', id='first'), InstructionsToolset('Second.', id='second')])
    agent = Agent(toolsets=[RelayingWrapper(grouped, id='outer')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='First.', dynamic=True, id=toolset_instruction_id('first')),
        InstructionPart(content='Second.', dynamic=True, id=toolset_instruction_id('second')),
    ]


async def test_a_wrapper_relays_a_declared_segment_under_a_child_key():
    """Relaying is decided by which source a key belongs to, not by matching the key whole.

    A block the child declared an id for arrives as `'toolset:weather:limits'`, which is not itself a
    source key. Recognizing only exact keys would read it as something the wrapper declared and
    resolve it beneath the wrapper's own — rejecting its colons, or misfiling it.
    """
    leaf = InstructionsToolset([InstructionPart(content='Limits.', name='limits')], id='weather')
    agent = Agent(toolsets=[RelayingWrapper(leaf, id='outer')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Limits.', name='limits', id=toolset_instruction_id('weather', 'limits'))
    ]


async def test_a_wrapper_relays_a_key_a_wrapper_it_wraps_issued_itself():
    """A wrapper with an `id` that contributes a block of its own is a source like any other.

    Its key is owned by nothing it wraps, so a subtree's keys have to include the wrappers' own and
    not just the ones they pass along.
    """
    inner = RelayingWrapperReturning(InstructionsToolset(id='leaf'), [InstructionPart(content='Inner.')], id='inner')
    agent = Agent(toolsets=[RelayingWrapper(inner, id='outer')])

    assert await run_and_capture(agent) == [InstructionPart(content='Inner.', id=toolset_instruction_id('inner'))]


async def test_wrapping_a_toolset_whose_id_cannot_be_a_key_leaves_it_working():
    """An id is only rejected where it is turned into a key, and being wrapped isn't that.

    A toolset that says nothing to the model may carry a colon in its `id` — the key is never minted,
    so there is nothing to be ambiguous. Working out which keys a wrapper may relay reads the ids of
    everything below it, and reading one must not be what mints it.
    """
    agent = Agent(
        toolsets=[
            RelayingWrapper(InstructionsToolset(id='remote:weather')),
            InstructionsToolset('Calendar.', id='calendar'),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Calendar.', dynamic=True, id=toolset_instruction_id('calendar'))
    ]


async def test_a_wrapper_cannot_relay_a_key_for_a_toolset_it_does_not_wrap():
    """Only the keys below a wrapper survive it, so it cannot speak in another toolset's name.

    A wrapper is trusted to relay because the keys it returns come from the toolsets it wraps.
    Trusting the shape of the key instead would let any wrapper write `'toolset:<someone else>'` —
    or a segment beneath it, which the duplicate check compares whole ids and so never catches —
    and an application addressing that key would reach a block the named toolset never wrote.
    """
    agent = Agent(
        toolsets=[
            RelayingWrapperReturning(
                InstructionsToolset(id='wrapped'),
                [InstructionPart(content='Forged.', name='limits', id=toolset_instruction_id('victim', 'limits'))],
                id='attacker',
            ),
            InstructionsToolset('Genuine.', id='victim'),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Forged.', name='limits', id=toolset_instruction_id('attacker', 'limits')),
        InstructionPart(content='Genuine.', dynamic=True, id=toolset_instruction_id('victim')),
    ]


@pytest.mark.skipif(not mcp_imports_successful(), reason='mcp not installed')
async def test_mcp_server_instructions_are_identified():
    """The motivating case: an MCP server's own instructions, addressable by the toolset's id."""
    server: FastMCP[None] = FastMCP('test_server', instructions='You are an MCP test server.')
    agent = Agent(toolsets=[MCPToolset(server, id='test-server', include_instructions=True)])

    async with agent:
        assert await run_and_capture(agent) == [
            InstructionPart(content='You are an MCP test server.', id=toolset_instruction_id('test-server'))
        ]


async def test_capability_instructions_are_identified_by_capability_id():
    agent = Agent(
        instructions='Agent instructions.',
        capabilities=[
            Capability(instructions='Memory instructions.', id='memory'),
            Capability(instructions='Anonymous instructions.'),
        ],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='Memory instructions.', id=capability_instruction_id('memory')),
        InstructionPart(content='Anonymous instructions.'),
    ]


async def test_instructions_decorator_can_declare_an_id():
    """A function's own identity isn't stable, but its author can declare one within the source."""
    agent = Agent(instructions='Agent instructions.')

    @agent.instructions(name='local_time')
    def local_time() -> str:
        return 'The time is 10:00.'

    @agent.instructions
    def user_name(ctx: RunContext[Any]) -> str:
        return 'The user is Frank.'

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(
            content='The time is 10:00.', dynamic=True, name='local_time', id=agent_instruction_id('local_time')
        ),
        InstructionPart(content='The user is Frank.', dynamic=True),
    ]


async def test_capability_instructions_decorator_can_declare_an_id():
    """A declared block within a capability qualifies the capability's own key."""
    budget = Capability[Any](instructions='Stay within budget.', id='budget')

    @budget.instructions(name='remaining')
    def remaining(ctx: RunContext[Any]) -> str:
        return 'Remaining budget: $10.'

    @budget.instructions
    def undeclared(ctx: RunContext[Any]) -> str:
        return 'Report overruns.'

    agent = Agent(capabilities=[budget])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Stay within budget.', id=capability_instruction_id('budget')),
        InstructionPart(
            content='Remaining budget: $10.',
            dynamic=True,
            name='remaining',
            id=capability_instruction_id('budget', 'remaining'),
        ),
        InstructionPart(content='Report overruns.', dynamic=True, id=capability_instruction_id('budget')),
    ]


async def test_declared_id_without_a_source_key_stays_unresolved():
    """A bare author name survives when its capability has no source to resolve it against."""
    anonymous = Capability[Any](instructions='Stay within budget.')

    @anonymous.instructions(name='remaining')
    def remaining(ctx: RunContext[Any]) -> str:
        return 'Remaining budget: $10.'

    agent = Agent(capabilities=[anonymous])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Stay within budget.'),
        InstructionPart(content='Remaining budget: $10.', dynamic=True, name='remaining'),
    ]


async def test_a_capability_subclass_keeps_computing_its_own_instructions():
    """Overriding `get_instructions` still wins: the declared-id path only reads stored instructions."""

    class Computed(Capability[Any]):
        def get_instructions(self) -> str:
            return 'Computed by the subclass.'

    agent = Agent(capabilities=[Computed(instructions='Ignored.', id='computed')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Computed by the subclass.', id=capability_instruction_id('computed'))
    ]


async def test_combined_capability_subclass_get_instructions_override_is_authoritative():
    """A public override on a combined capability replaces its children's contributions."""

    class OverriddenCombined(CombinedCapability[Any]):
        id = 'group'

        def get_instructions(self) -> str:
            return 'Override.'

    agent = Agent(
        capabilities=[OverriddenCombined(capabilities=[Capability[Any](instructions='Child.', id='child')], id='group')]
    )

    assert await run_and_capture(agent) == [InstructionPart(content='Override.', id=capability_instruction_id('group'))]


async def test_two_toolsets_sharing_an_id_collide_even_under_different_declared_segments():
    """What a source owns is its key, so declaring different segments beneath it changes nothing.

    `toolset:same` means everything the toolset registered under `same` contributes. Comparing whole
    ids instead would let the pair through whenever each happened to declare a segment, leaving the
    key owned by two sources — exactly the ambiguity the rule exists to prevent.
    """
    agent = Agent(
        toolsets=[
            InstructionsToolset([InstructionPart(content='First.', name='first')], id='same'),
            InstructionsToolset([InstructionPart(content='Second.', name='second')], id='same'),
        ]
    )

    with pytest.raises(
        UserError,
        match=r"Two toolsets have the same `id` 'same' and both contribute instructions, so 'toolset:same' would address parts from each\.",
    ):
        await run_and_capture(agent)


async def test_a_combined_toolset_subclass_can_delegate_get_instructions_to_super():
    """Delegating to `super()` is how a subclass extends what it wraps, so it has to terminate.

    The override check routes an overriding subclass through the base's own collection, which asks
    `self.get_instructions()` — landing back in the override. Reaching the children without passing
    the check again is what stops that being a loop.
    """

    class Relaying(CombinedToolset[Any]):
        async def get_instructions(
            self, ctx: RunContext[Any]
        ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
            return await super().get_instructions(ctx)

    agent = Agent(toolsets=[Relaying([InstructionsToolset('Child.', id='child')])])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Child.', dynamic=True, id=toolset_instruction_id('child'))
    ]


async def test_a_combined_toolset_subclass_get_instructions_override_is_authoritative():
    """A public override on a combined toolset replaces its children's contributions."""

    class Overridden(CombinedToolset[Any]):
        async def get_instructions(
            self, ctx: RunContext[Any]
        ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
            return 'Override.'

    agent = Agent(toolsets=[Overridden([InstructionsToolset('Child.', id='child')])])

    assert await run_and_capture(agent) == [InstructionPart(content='Override.', dynamic=True)]


def _keep_every_tool(ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
    # Never called: the swept containers hold a toolset with no tools, because what is being swept is
    # instruction relay. `FilteredToolset` still requires a filter to construct.
    return True  # pragma: no cover


def _keep_every_tool_def(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return tool_defs


DELEGATING_TOOLSET_CONTAINERS: list[tuple[str, Callable[[AbstractToolset[Any]], AbstractToolset[Any]]]] = [
    ('wrapper', lambda leaf: _delegating_wrapper(leaf)),
    ('combined', lambda leaf: _delegating_combined([leaf])),
    ('prefixed', lambda leaf: _delegating(PrefixedToolset)(leaf, 'p')),
    ('filtered', lambda leaf: _delegating(FilteredToolset)(leaf, _keep_every_tool)),
    ('renamed', lambda leaf: _delegating(RenamedToolset)(leaf, {})),
    ('approval_required', lambda leaf: _delegating(ApprovalRequiredToolset)(leaf)),
    ('prepared', lambda leaf: _delegating(PreparedToolset)(leaf, _keep_every_tool_def)),
    ('combined_of_wrapper', lambda leaf: _delegating_combined([_delegating_wrapper(leaf)])),
    ('wrapper_of_combined', lambda leaf: _delegating_wrapper(_delegating_combined([leaf]))),
]


def _delegating(base: type[Any]) -> type[Any]:
    """A subclass of `base` whose `get_instructions` does nothing but call `super()`."""

    class Delegating(base):
        async def get_instructions(
            self, ctx: RunContext[Any]
        ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
            return await super().get_instructions(ctx)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    return Delegating


def _delegating_wrapper(wrapped: AbstractToolset[Any]) -> AbstractToolset[Any]:
    return _delegating(WrapperToolset)(wrapped)


def _delegating_combined(toolsets: list[AbstractToolset[Any]]) -> AbstractToolset[Any]:
    return _delegating(CombinedToolset)(toolsets)


@pytest.mark.parametrize(
    'container',
    [container for _, container in DELEGATING_TOOLSET_CONTAINERS],
    ids=[name for name, _ in DELEGATING_TOOLSET_CONTAINERS],
)
async def test_every_toolset_container_relays_through_a_delegating_subclass(
    container: Callable[[AbstractToolset[Any]], AbstractToolset[Any]],
):
    """Overriding `get_instructions` to call `super()` is the ordinary way to extend a container.

    Every container has to survive it identically, in either nesting order. Two failure modes have
    been reached this way already, and both are silent from the outside: the key is dropped, or the
    override is routed back into itself and never returns. Sweeping every container keeps a new one
    from being added with only one of the two halves right.
    """
    agent = Agent(toolsets=[container(InstructionsToolset('Leaf.', id='leaf'))])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Leaf.', dynamic=True, id=toolset_instruction_id('leaf'))
    ]


def test_get_instructions_returns_declared_blocks():
    """`get_instructions()` is public API, so it answers independently of the id-carrying collection path.

    The framework reads instructions through `_collect_instructions()` so blocks keep their ids; this
    pins the plain accessor a caller outside the framework reaches for, including that a capability
    with nothing to say answers `None` rather than an empty list.
    """
    capability = Capability[Any](instructions=['First block.', 'Second block.'])
    assert capability.get_instructions() == ['First block.', 'Second block.']
    assert Capability[Any]().get_instructions() is None


def test_wrapper_capability_get_instructions_delegates():
    """A wrapper that adds no instructions of its own reports the ones it wraps, unchanged."""
    inner = Capability[Any](instructions='Wrapped instructions.')
    assert WrapperCapability(wrapped=inner).get_instructions() == ['Wrapped instructions.']


def test_instruction_id_segments_reject_colons_at_registration():
    """The delimiter cannot occur inside a source or declared segment."""

    with pytest.raises(
        UserError,
        match=r"Capability id 'budget:remaining' cannot contain a colon because `:` is reserved as an instruction ID delimiter\.",
    ):
        Capability[Any](id='budget:remaining')

    class CustomCapability(AbstractCapability[Any]):
        def __init__(self) -> None:
            self.id = 'custom:capability'

    with pytest.raises(UserError, match="Capability id 'custom:capability' cannot contain a colon"):
        Agent(capabilities=[CustomCapability()])

    with pytest.raises(UserError, match="Capability id 'combined:capability' cannot contain a colon"):
        CombinedCapability[Any](capabilities=[], id='combined:capability')

    agent = Agent()
    with pytest.raises(UserError, match="Instruction name 'local:time' cannot contain a colon"):
        agent.instructions(name='local:time')

    capability = Capability[Any](id='budget')
    with pytest.raises(UserError, match="Instruction name 'remaining:usd' cannot contain a colon"):
        capability.instructions(name='remaining:usd')


async def test_a_toolset_id_with_a_colon_is_rejected_when_it_keys_a_block():
    """A toolset's id is only held to the delimiter rule where it has to become one.

    A colon breaks `'toolset:<id>'` apart, so it is rejected as the id is turned into a key. A
    toolset that contributes no instructions never gets one, and its id is its own business.
    """
    agent = Agent(toolsets=[InstructionsToolset('From a remote toolset.', id='weather:remote')])

    with pytest.raises(UserError, match="Toolset id 'weather:remote' cannot contain a colon"):
        await run_and_capture(agent)

    await run_and_capture(Agent(toolsets=[InstructionsToolset(id='weather:remote')]))


async def test_declared_instruction_ids_qualify_rather_than_collide():
    """A declared segment can't claim a source key, and the framework doesn't police duplicates.

    Two functions can declare the same id — like two blocks from one toolset sharing its key, and
    like `@agent.toolset(id=...)`, which validates nothing either.
    """
    agent = Agent(instructions='Agent instructions.')

    @agent.instructions(name='same')
    def first() -> str:
        return 'First.'

    @agent.instructions(name='same')
    def second() -> str:
        return 'Second.'

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='First.', dynamic=True, name='same', id=agent_instruction_id('same')),
        InstructionPart(content='Second.', dynamic=True, name='same', id=agent_instruction_id('same')),
    ]


async def test_a_source_gives_all_of_its_blocks_the_same_id():
    """Everything a capability tells the model is addressed by its id, computed blocks included.

    They share one key, so they can't be addressed individually — the known limitation of this
    first version, and the same way a toolset's several blocks share `toolset:<id>`.
    """

    def memory_instructions(ctx: RunContext[Any]) -> str:
        return 'Remembered.'

    agent = Agent(
        capabilities=[Capability[Any](instructions=['Memory instructions.', memory_instructions], id='memory')],
        toolsets=[
            InstructionsToolset(['Weather usage.', 'Weather limits.'], id='weather'),
        ],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Memory instructions.', id=capability_instruction_id('memory')),
        InstructionPart(content='Remembered.', dynamic=True, id=capability_instruction_id('memory')),
        InstructionPart(content='Weather usage.', dynamic=True, id=toolset_instruction_id('weather')),
        InstructionPart(content='Weather limits.', dynamic=True, id=toolset_instruction_id('weather')),
    ]


async def test_agent_instruction_functions_stay_unidentified():
    """`'agent'` names the literal base prompt only.

    An agent can register any number of `@agent.instructions` functions; sharing `'agent'` between
    them would make each one unaddressable forever, so they opt in individually with `id=`.
    """
    agent = Agent(instructions=['Agent instructions.', lambda: 'From a constructor callable.'])

    @agent.instructions
    def user_name(ctx: RunContext[Any]) -> str:
        return 'The user is Frank.'

    assert await run_and_capture(agent, instructions=lambda: 'From a run callable.') == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='From a constructor callable.', dynamic=True),
        InstructionPart(content='The user is Frank.', dynamic=True),
        InstructionPart(content='From a run callable.', dynamic=True),
    ]


async def test_capability_toolset_instructions_are_identified_by_toolset_id():
    """A toolset contributed by a capability is still addressed by its own id."""
    agent = Agent(
        capabilities=[
            Capability[Any](
                toolsets=[InstructionsToolset('Weather instructions.', id='weather')], id='weather-capability'
            )
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Weather instructions.', dynamic=True, id=toolset_instruction_id('weather'))
    ]


async def test_capability_without_id_gets_an_unidentified_part_of_its_own():
    """The reserved `agent` block holds only the agent's own instructions, so it isn't merged into."""
    agent = Agent(
        instructions='Agent instructions.', capabilities=[Capability[Any](instructions='Extra instructions.')]
    )

    model, captured = capture_instruction_parts()
    result = await agent.run('Hello', model=model)

    assert captured == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='Extra instructions.'),
    ]
    assert rendered_instructions(result.all_messages()) == 'Agent instructions.\n\nExtra instructions.'


async def test_wrapper_capability_passes_through_leaf_ids():
    class Transparent(WrapperCapability[Any]):
        pass

    class Opinionated(WrapperCapability[Any]):
        def get_instructions(self) -> str:
            return 'Wrapper instructions.'

    agent = Agent(
        capabilities=[
            Transparent(wrapped=Capability[Any](instructions='Memory instructions.', id='memory')),
            Opinionated(wrapped=Capability[Any](instructions='Ignored.', id='search')),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Memory instructions.', id=capability_instruction_id('memory')),
        InstructionPart(content='Wrapper instructions.', id=capability_instruction_id('search')),
    ]


async def test_an_overriding_wrapper_relays_a_combined_capabilitys_instruction_keys():
    """A container wrapper must preserve the authorship of every child it passes through."""

    class Relaying(WrapperCapability[Any]):
        def get_instructions(self):
            return super().get_instructions()

    combined = CombinedCapability[Any](
        [
            Capability[Any](instructions='First.', id='first'),
            Capability[Any](instructions='Second.', id='second'),
        ]
    )

    assert await run_and_capture(Agent(capabilities=[Relaying(combined)])) == [
        InstructionPart(content='First.', id=capability_instruction_id('first')),
        InstructionPart(content='Second.', id=capability_instruction_id('second')),
    ]


async def test_an_overriding_combined_capability_relays_its_childrens_instruction_keys():
    """Delegating to `super()` must not turn a combined capability into the author of its children."""

    class Relaying(CombinedCapability[Any]):
        def get_instructions(self):
            return super().get_instructions()

    combined = Relaying(
        [
            Capability[Any](instructions='First.', id='first'),
            Capability[Any](instructions='Second.', id='second'),
        ]
    )

    assert await run_and_capture(Agent(capabilities=[combined])) == [
        InstructionPart(content='First.', id=capability_instruction_id('first')),
        InstructionPart(content='Second.', id=capability_instruction_id('second')),
    ]


async def test_an_overriding_wrapper_relays_static_and_callable_instruction_keys():
    """Recipe identity must retain declared callable segments that bare public recipes cannot carry."""

    class Relaying(WrapperCapability[Any]):
        def get_instructions(self):
            return super().get_instructions()

    dynamic = Capability[Any](id='dyn')

    @dynamic.instructions(name='now')
    def now() -> str:
        return 'Time is 10:00.'

    combined = CombinedCapability[Any]([dynamic, Capability[Any](instructions='Lit.', id='lit')])

    assert await run_and_capture(Agent(capabilities=[Relaying(combined)])) == [
        InstructionPart(content='Lit.', id=capability_instruction_id('lit')),
        InstructionPart(content='Time is 10:00.', name='now', id=capability_instruction_id('dyn', 'now'), dynamic=True),
    ]


async def test_an_overriding_wrapper_attributes_only_the_instructions_it_adds():
    """Extending a container must preserve child keys without assigning a child to wrapper-owned text."""

    class Extending(WrapperCapability[Any]):
        def get_instructions(self) -> AgentInstructions[Any]:
            return [*normalize_instructions(super().get_instructions()), 'Wrapper extra.']

    combined = CombinedCapability[Any](
        [
            Capability[Any](instructions='First.', id='first'),
            Capability[Any](instructions='Second.', id='second'),
        ]
    )

    assert await run_and_capture(Agent(capabilities=[Extending(combined)])) == [
        InstructionPart(content='First.', id=capability_instruction_id('first')),
        InstructionPart(content='Second.', id=capability_instruction_id('second')),
        InstructionPart(content='Wrapper extra.'),
    ]


async def test_an_overriding_container_does_not_guess_between_the_same_recipe_object():
    """A shared recipe has multiple possible authors, so assigning either key would make it lie."""

    class Relaying(WrapperCapability[Any]):
        def get_instructions(self):
            return super().get_instructions()

    shared = 'Same text.'
    combined = CombinedCapability[Any](
        [Capability[Any](instructions=shared, id='first'), Capability[Any](instructions=shared, id='second')]
    )

    assert await run_and_capture(Agent(capabilities=[Relaying(combined)])) == [
        InstructionPart(content='Same text.'),
        InstructionPart(content='Same text.'),
    ]


def test_combined_capability_get_instructions_is_unattributed():
    """The public `get_instructions` view is unchanged: the same instructions, without their ids."""

    def dynamic_instructions(ctx: RunContext[Any]) -> str:  # pragma: no cover
        return 'Dynamic.'

    combined = CombinedCapability[Any](
        [
            Capability[Any](instructions=['Memory.', dynamic_instructions], id='memory'),
            Capability[Any](instructions='Hidden.', id='deferred', defer_loading=True),
        ]
    )

    assert combined.get_instructions() == ['Memory.', dynamic_instructions]
    assert CombinedCapability[Any]([]).get_instructions() is None


async def test_run_instructions_are_not_identified_but_override_replaces_the_agent_block():
    agent = Agent(
        instructions='Agent instructions.', capabilities=[Capability[Any](instructions='Memory.', id='memory')]
    )

    assert await run_and_capture(agent, instructions='Run instructions.') == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='Memory.', id=capability_instruction_id('memory')),
        InstructionPart(content='Run instructions.'),
    ]

    with agent.override(instructions=['One.', 'Two.']):
        assert await run_and_capture(agent) == [
            InstructionPart(
                content="""\
One.

Two.\
""",
                id=agent_instruction_id(),
            )
        ]

    with agent.override(instructions=InstructionPart(content='Named override.', name='override')):
        assert await run_and_capture(agent) == [
            InstructionPart(content='Named override.', name='override', id=agent_instruction_id('override'))
        ]


async def test_instruction_parts_are_separated_by_a_blank_line():
    """Every instruction part is joined the same way, and empty contributions leave no artifacts."""

    def empty_instructions(ctx: RunContext[Any]) -> str:
        return ''

    agent = Agent(
        instructions=['  ', 'First.', '\n', 'Second.', empty_instructions],
        capabilities=[Capability[Any](instructions='Third.', id='memory')],
        toolsets=[InstructionsToolset('   ', id='blank'), InstructionsToolset('Fourth.', id='weather')],
    )

    model, captured = capture_instruction_parts()
    result = await agent.run('Hello', model=model)

    assert captured == [
        InstructionPart(
            content="""\
First.

Second.\
""",
            id=agent_instruction_id(),
        ),
        InstructionPart(content='Third.', id=capability_instruction_id('memory')),
        InstructionPart(content='Fourth.', dynamic=True, id=toolset_instruction_id('weather')),
    ]
    assert rendered_instructions(result.all_messages()) == 'First.\n\nSecond.\n\nThird.\n\nFourth.'


async def test_before_model_request_can_rewrite_a_block_by_id():
    """Addressing a block by id is what the ids are for, so history has to show the rewrite."""

    def override_weather(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        parts = request_context.model_request_parameters.instruction_parts or []
        request_context.model_request_parameters.instruction_parts = [
            replace(part, content='Managed weather instructions.')
            if part.id == toolset_instruction_id('weather')
            else part
            for part in parts
        ]
        return request_context

    agent = Agent(
        instructions='Agent instructions.',
        toolsets=[InstructionsToolset('Weather instructions.', id='weather')],
        capabilities=[Hooks[Any](before_model_request=override_weather)],
    )

    model, captured = capture_instruction_parts()
    result = await agent.run('Hello', model=model)

    assert captured == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(content='Managed weather instructions.', dynamic=True, id=toolset_instruction_id('weather')),
    ]
    assert rendered_instructions(result.all_messages()) == 'Agent instructions.\n\nManaged weather instructions.'


async def test_before_model_request_instructions_edit_does_not_reach_the_model():
    """The parts are the source of truth: an edit to the message alone is overwritten, not propagated."""

    def edit_message(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request = request_context.messages[-1]
        assert isinstance(request, ModelRequest)
        request.instructions = 'Edited instructions.'
        return request_context

    agent = Agent(instructions='Agent instructions.', capabilities=[Hooks[Any](before_model_request=edit_message)])

    model, captured = capture_instruction_parts()
    result = await agent.run('Hello', model=model)

    assert captured == [InstructionPart(content='Agent instructions.', id=agent_instruction_id())]
    assert rendered_instructions(result.all_messages()) == 'Agent instructions.'


async def test_clearing_instruction_parts_leaves_the_recorded_instructions():
    """`None` parts means "unset", which is what makes a model fall back to the recorded request."""
    instructions_by_call: list[str | None] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.instruction_parts is None
        instructions_by_call.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    def clear_parts(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.model_request_parameters.instruction_parts = None
        return request_context

    agent = Agent(instructions='Agent instructions.', capabilities=[Hooks[Any](before_model_request=clear_parts)])

    result = await agent.run('Hello', model=FunctionModel(model_fn))

    assert instructions_by_call == ['Agent instructions.']
    assert rendered_instructions(result.all_messages()) == 'Agent instructions.'


async def test_before_model_request_can_rewrite_a_resumed_requests_instructions():
    """Resuming a paused turn echoes the recorded request back, so a rewrite has to land on it too."""
    parts_by_call: list[list[InstructionPart]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts_by_call.append(list(info.model_request_parameters.instruction_parts or []))
        return ModelResponse(parts=[TextPart('done')])

    def rewrite(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.model_request_parameters.instruction_parts = [InstructionPart(content='Managed instructions.')]
        return request_context

    agent = Agent(
        FunctionModel(model_fn),
        instructions='Agent instructions.',
        capabilities=[Hooks[Any](before_model_request=rewrite)],
    )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')], instructions='Agent instructions.'),
        ModelResponse(parts=[TextPart('paused')], state='suspended'),
    ]
    result = await agent.run(message_history=history)

    assert parts_by_call == [[InstructionPart(content='Managed instructions.')]]
    assert rendered_instructions(result.all_messages()) == 'Managed instructions.'


def test_id_does_not_affect_joining_or_sorting():
    parts = [
        InstructionPart(content='Dynamic.', dynamic=True, id=toolset_instruction_id('weather')),
        InstructionPart(content='Static.'),
        InstructionPart(content='Identified.', id=capability_instruction_id('memory')),
    ]

    assert InstructionPart.join(parts) == 'Dynamic.\n\nStatic.\n\nIdentified.'
    assert InstructionPart.sorted(parts) == [
        InstructionPart(content='Static.'),
        InstructionPart(content='Identified.', id=capability_instruction_id('memory')),
        InstructionPart(content='Dynamic.', dynamic=True, id=toolset_instruction_id('weather')),
    ]


@pytest.mark.parametrize(
    ('id', 'serialized'),
    [
        (agent_instruction_id(), 'agent'),
        (agent_instruction_id('persona'), 'agent:persona'),
        (toolset_instruction_id('weather'), 'toolset:weather'),
        (toolset_instruction_id('weather', 'limits'), 'toolset:weather:limits'),
        (capability_instruction_id('memory'), 'capability:memory'),
        (capability_instruction_id('memory', 'style'), 'capability:memory:style'),
    ],
)
def test_resolved_id_serialization_round_trip(id: InstructionId, serialized: str):
    part = InstructionPart(content='Instructions.', dynamic=True, name=id.name, id=id)
    payload = instruction_part_ta.dump_python(part, mode='json')

    assert payload == {
        'content': 'Instructions.',
        'dynamic': True,
        'name': id.name,
        'id': serialized,
        'part_kind': 'instruction',
    }
    assert instruction_part_ta.validate_python(payload) == part


def test_unresolved_and_missing_id_serialization_round_trip():
    # A name whose source has no identity of its own travels as a name, with no key beside it.
    unresolved = InstructionPart(content='Instructions.', name='limits')
    payload = instruction_part_ta.dump_python(unresolved, mode='json')
    assert payload['name'] == 'limits'
    assert payload['id'] is None
    assert instruction_part_ta.validate_python(payload) == unresolved
    assert instruction_part_ta.validate_python({**payload, 'id': toolset_instruction_id('weather')}).id == (
        toolset_instruction_id('weather')
    )

    # Payloads recorded before the fields existed still validate, and stay unidentified.
    assert instruction_part_ta.validate_python(
        {'content': 'Weather.', 'dynamic': True, 'part_kind': 'instruction'}
    ) == InstructionPart(content='Weather.', dynamic=True)

    # Instruction parts cross durable execution and UI boundaries, so a key written by a newer
    # version has to load here rather than raise. Its namespace names a source this version cannot
    # address anything under, so the part reads as unaddressable rather than being mangled into a
    # source that happens to be recognised.
    assert instruction_part_ta.validate_python({**payload, 'id': 'plugin:search:limits'}).id is None


def test_repr_omits_unset_id():
    assert repr(InstructionPart(content='Weather.')) == "InstructionPart(content='Weather.')"
    assert (
        repr(InstructionPart(content='Weather.', id=toolset_instruction_id('weather')))
        == "InstructionPart(content='Weather.', id=InstructionId(source=ToolsetInstructionSource(id='weather')))"
    )


async def test_deferred_capability_instructions_stay_hidden():
    """Deferred capabilities contribute no parts until loaded, identified or not."""

    agent = Agent(
        instructions='Agent instructions.',
        capabilities=[Capability[Any](instructions='Deferred instructions.', id='deferred', defer_loading=True)],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id=agent_instruction_id()),
        InstructionPart(
            content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- deferred\
""",
            dynamic=True,
        ),
    ]


async def test_combined_capability_override_survives_re_composition():
    """A run-level capability re-composes the tree; the container that owns the override is still there.

    `Agent.iter` builds a fresh `CombinedCapability` over the resolved layers whenever anything is
    contributed per run (also for auto-injected instrumentation), which re-derives the composition
    view from the agent's already-flattened root. Rebuilding it from the flattened children would
    splat the overriding container back out and send its children's blocks instead.
    """

    class OverriddenCombined(CombinedCapability[Any]):
        id = 'group'

        def get_instructions(self) -> str:
            return 'Override.'

    agent = Agent(
        capabilities=[OverriddenCombined(capabilities=[Capability[Any](instructions='Child.', id='child')], id='group')]
    )

    assert await run_and_capture(agent, capabilities=[Capability[Any](instructions='Per run.', id='per_run')]) == [
        InstructionPart(content='Override.', id=capability_instruction_id('group')),
        InstructionPart(content='Per run.', id=capability_instruction_id('per_run')),
    ]


async def test_instruction_parts_follow_the_capability_ordering():
    """Blocks come out in the order the ordering pass settled the capabilities into, not registration order.

    A capability that asks to be `outermost` wraps the others' hooks, and its instructions have
    always led the prompt to match — `DeferredCapabilityLoader` puts its catalog block first this
    way despite being appended last.
    """

    class Outermost(Capability[Any]):
        def get_ordering(self) -> CapabilityOrdering | None:
            return CapabilityOrdering(position='outermost')

    agent = Agent(
        capabilities=[
            Capability[Any](instructions='Registered first.', id='first'),
            Outermost(instructions='Registered last.', id='last'),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Registered last.', id=capability_instruction_id('last')),
        InstructionPart(content='Registered first.', id=capability_instruction_id('first')),
    ]


async def test_each_unidentified_source_gets_its_own_part():
    """Two sources that are both unaddressable are still two blocks, not one fused one.

    `None` is not a key. Fusing on it would run an anonymous capability's instructions together
    with the ones passed to this single run, which have nothing to do with each other and don't
    even share a lifetime.
    """

    agent = Agent(
        capabilities=[
            Capability[Any](instructions='From an anonymous capability.'),
            Capability[Any](instructions='From another anonymous capability.'),
        ]
    )

    assert await run_and_capture(agent, instructions='Passed to this run.') == [
        InstructionPart(content='From an anonymous capability.'),
        InstructionPart(content='From another anonymous capability.'),
        InstructionPart(content='Passed to this run.'),
    ]


async def test_an_unidentified_source_gets_a_part_per_block():
    """Blocks are fused by their id, so an unidentified source's blocks are never fused.

    Sharing an id is what makes several literals one part: a key has to name one block, so
    everything `Agent(instructions=[...])` was built with arrives as a single `'agent'`. `None`
    names nothing, and there is no boundary to preserve beyond the ones the author wrote — so an
    anonymous source contributes a part per block rather than one fused one, exactly like two
    anonymous sources beside each other. The rendered prompt is identical either way; only the
    part boundaries differ.
    """

    agent = Agent(capabilities=[Capability[Any](instructions=['Anonymous one.', 'Anonymous two.'])])

    assert await run_and_capture(agent, instructions=['Passed to this run.', 'And this.']) == [
        InstructionPart(content='Anonymous one.'),
        InstructionPart(content='Anonymous two.'),
        InstructionPart(content='Passed to this run.'),
        InstructionPart(content='And this.'),
    ]


async def test_a_callable_between_literals_does_not_split_the_agent_block():
    """A key names one block, so an intervening callable must not turn `'agent'` into two parts.

    `Agent(instructions=[...])` may mix literals and functions freely, and the literals are the
    agent's base prompt however they're interleaved. Flushing the group at each callable would make
    the identity of `'agent'` depend on where an unrelated function happens to sit in the list —
    reordering the argument would silently change how many parts an application addressing `'agent'`
    has to rewrite. The callable stays its own unidentified block, as any callable does.
    """

    def dynamic_instruction(ctx: RunContext[Any]) -> str:
        return 'From a function.'

    agent = Agent(instructions=['Literal one.', dynamic_instruction, 'Literal two.'])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Literal one.\n\nLiteral two.', id=agent_instruction_id()),
        InstructionPart(content='From a function.', dynamic=True),
    ]


async def test_an_instruction_part_between_literals_keeps_each_block_independent():
    """An authored `InstructionPart` remains a boundary between surrounding literals."""

    agent = Agent(instructions=['Literal one.', InstructionPart(content='Authored part.'), 'Literal two.'])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Literal one.', id=agent_instruction_id()),
        InstructionPart(content='Authored part.', id=agent_instruction_id()),
        InstructionPart(content='Literal two.', id=agent_instruction_id()),
    ]


def test_an_overriding_container_cannot_share_a_key_with_a_sibling():
    """A retained container is an instruction source, so its `id` competes for the same key.

    A `CombinedCapability` subclass that overrides `get_instructions` contributes as a source in its
    own right and is kept out of the flattened capability list so its override survives. That put it
    outside the uniqueness check that walks the flattened list: its leaves could be distinct while
    the container itself shared an id with a sibling, and `capability:<id>` then addressed both --
    rewriting that key would have replaced text from two unrelated owners.
    """

    class Group(CombinedCapability[Any]):
        def get_instructions(self) -> str:
            return 'Group override.'  # pragma: no cover -- construction raises before it is asked

    with pytest.raises(UserError, match="Capability id 'dup' is used by multiple capabilities that contribute"):
        Agent(
            capabilities=[
                Group(capabilities=[Capability[Any](instructions='Leaf.', id='leaf')], id='dup'),
                Capability[Any](instructions='Sibling.', id='dup'),
            ]
        )


async def test_a_run_capability_cannot_share_a_key_with_an_overriding_container():
    """The same collision is reachable from `run(capabilities=...)`, not just at construction."""

    class Group(CombinedCapability[Any]):
        def get_instructions(self) -> str:
            return 'Group override.'

    agent = Agent(capabilities=[Group(capabilities=[Capability[Any](instructions='Leaf.', id='leaf')], id='dup')])

    with pytest.raises(UserError, match="Capability id 'dup' is used by multiple capabilities that contribute"):
        await run_and_capture(agent, capabilities=[Capability[Any](instructions='Run-level.', id='dup')])


async def test_for_run_cannot_rebind_onto_a_container_key():
    """`for_run` can hand back an id construction never saw, so the resolved tree is checked too.

    Construction validates the capabilities it was given. A capability whose `for_run` returns a
    different instance can introduce a collision after that point -- here against a retained
    overriding container -- and no further layer is composed, so nothing else would notice.
    """

    class Group(CombinedCapability[Any]):
        def get_instructions(self) -> str:
            return 'Group override.'

    class Mutant(Capability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return Capability[Any](instructions='Mutated.', id='dup')

    agent = Agent(
        capabilities=[
            Group(capabilities=[Capability[Any](instructions='Leaf.', id='leaf')], id='dup'),
            Mutant(instructions='Original.', id='mutant'),
        ]
    )

    with pytest.raises(UserError, match="Capability id 'dup' is used by multiple capabilities that contribute"):
        await run_and_capture(agent)


async def test_resuming_does_not_stamp_instructions_onto_a_mock_request():
    """The rewrite lands on the message the echoed instructions came from, not on the trailing request.

    A turn suspended after a tool round-trip leaves a tool-return-only request last, which carries
    no instructions of its own — `_get_history_instructions` deliberately looks past it to the
    request before. Stamping that trailing one would record instructions on a message that was sent
    without any, and it is typically a message the caller handed in from a previous run.
    """

    def rewrite(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.model_request_parameters.instruction_parts = [InstructionPart(content='Managed instructions.')]
        return request_context

    agent = Agent(
        FunctionModel(lambda _m, _i: ModelResponse(parts=[TextPart('done')])),
        instructions='Agent instructions.',
        capabilities=[Hooks[Any](before_model_request=rewrite)],
    )

    tool_returns = ModelRequest(parts=[ToolReturnPart(tool_name='t', content='ok', tool_call_id='1')])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')], instructions='Agent instructions.'),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args={}, tool_call_id='1')]),
        tool_returns,
        ModelResponse(parts=[TextPart('paused')], state='suspended'),
    ]
    result = await agent.run(message_history=history)

    assert tool_returns.instructions is None
    assert rendered_instructions(result.all_messages()) == 'Managed instructions.'


async def test_a_declared_id_is_resolved_against_its_source_key():
    """One rule for declaring a block id, wherever the block comes from.

    An author names a block relative to what they own — `'persona'`, `'limits'` — and the framework
    qualifies it against the key of the source contributing it. Nobody repeats their own identity,
        and nobody can claim a top-level key.
    """

    class Coder(AbstractCapability[Any]):
        id = 'coder'

        def get_instructions(self) -> Any:
            return [InstructionPart(content='Style.', name='style'), InstructionPart(content='Scope.')]

    agent = Agent(
        instructions=[InstructionPart(content='Persona.', name='persona'), 'Unnamed.'],
        capabilities=[
            Coder(),
            Capability[Any](id='budget', instructions=[InstructionPart(content='Cap.', name='cap')]),
        ],
        toolsets=[InstructionsToolset([InstructionPart(content='Tool.', name='usage')], id='weather')],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Persona.', name='persona', id=agent_instruction_id('persona')),
        InstructionPart(content='Unnamed.', id=agent_instruction_id()),
        InstructionPart(content='Style.', name='style', id=capability_instruction_id('coder', 'style')),
        InstructionPart(content='Scope.', id=capability_instruction_id('coder')),
        InstructionPart(content='Cap.', name='cap', id=capability_instruction_id('budget', 'cap')),
        InstructionPart(content='Tool.', name='usage', id=toolset_instruction_id('weather', 'usage')),
    ]


async def test_a_declared_id_already_carrying_its_source_key_is_left_alone():
    """Writing the qualified form yields what the author meant, not a doubled key."""
    agent = Agent(
        toolsets=[
            InstructionsToolset(
                [
                    InstructionPart(content='Limits.', name='limits', id=toolset_instruction_id('weather', 'limits')),
                    InstructionPart(content='All.', id=toolset_instruction_id('weather')),
                ],
                id='weather',
            )
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Limits.', name='limits', id=toolset_instruction_id('weather', 'limits')),
        InstructionPart(content='All.', id=toolset_instruction_id('weather')),
    ]


async def declare_on_agent_constructor(name: str) -> None:
    Agent(instructions=InstructionPart(content='X.', name=name))


async def declare_on_agent_decorator(name: str) -> None:
    Agent().instructions(name=name)


async def declare_on_a_single_run(name: str) -> None:
    await run_and_capture(Agent(), instructions=InstructionPart(content='X.', name=name))


async def declare_on_capability_constructor(name: str) -> None:
    Capability[Any](id='budget', instructions=InstructionPart(content='X.', name=name))


async def declare_on_capability_decorator(name: str) -> None:
    Capability[Any](id='budget').instructions(name=name)


async def declare_on_a_toolset_block(name: str) -> None:
    await run_and_capture(Agent(toolsets=[InstructionsToolset(InstructionPart(content='X.', name=name), id='weather')]))


DECLARING_AUTHORS = [
    pytest.param(declare_on_agent_constructor, id='agent constructor'),
    pytest.param(declare_on_agent_decorator, id='@agent.instructions'),
    pytest.param(declare_on_a_single_run, id='run instructions'),
    pytest.param(declare_on_capability_constructor, id='capability constructor'),
    pytest.param(declare_on_capability_decorator, id='@capability.instructions'),
    pytest.param(declare_on_a_toolset_block, id='toolset block'),
]


@pytest.mark.parametrize('declare', DECLARING_AUTHORS)
@pytest.mark.parametrize(
    ('name', 'message'),
    [
        (
            'a:b',
            r"Instruction name 'a:b' cannot contain a colon because `:` is reserved as an instruction ID delimiter\.",
        ),
        (
            'agent',
            r"Instruction name 'agent' is reserved for the agent's own instructions; choose a different name\.",
        ),
    ],
)
async def test_no_author_can_declare_a_name_that_would_read_as_a_framework_key(
    declare: Callable[[str], Awaitable[None]], name: str, message: str
):
    """A name is rejected where it is written, so the rule doesn't depend on what it resolves against.

    A name that resolves against no source is written to the wire exactly as the author wrote it, so
    every name has to be one that could never be read back as a key the framework issued -- whatever
    the source it happens to be declared on today.
    """
    with pytest.raises(UserError, match=message):
        await declare(name)


async def test_an_instruction_part_keeps_its_own_dynamic_flag_and_block():
    """`dynamic` decides what falls inside the cacheable prefix, so a part never merges into a neighbour.

    Declaring it is the only way to say "computed once, then stable for the run" about literal text;
    a bare string from the same source would otherwise be fused into one block with one flag.
    """
    agent = Agent(
        capabilities=[
            Capability[Any](
                id='budget',
                instructions=[
                    InstructionPart(content='Stable.', name='stable'),
                    InstructionPart(content='Varies.', name='varies', dynamic=True),
                ],
            )
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Stable.', name='stable', id=capability_instruction_id('budget', 'stable')),
        InstructionPart(
            content='Varies.', dynamic=True, name='varies', id=capability_instruction_id('budget', 'varies')
        ),
    ]


async def test_a_run_level_part_keeps_its_unresolved_name():
    """Per-run instructions preserve a bare author name because they have no source to resolve it against."""
    agent = Agent(instructions='Agent block.')

    assert await run_and_capture(agent, instructions=[InstructionPart(content='Per run.', name='urgent')]) == [
        InstructionPart(content='Agent block.', id=agent_instruction_id()),
        InstructionPart(content='Per run.', name='urgent'),
    ]


async def test_a_deferred_capability_can_declare_its_blocks_as_parts():
    """A part reaches the model as tool-return text when the capability it belongs to is deferred.

    Loading delivers instructions as the `load_capability` result rather than as request parts, so
    the ids are flattened away here as they are for any deferred capability — but the part's content
    still has to arrive, which is a different code path from the literal-string one.
    """
    calls = 0

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'refunds'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            Capability[Any](
                id='refunds',
                description='Refund tools.',
                instructions=[
                    InstructionPart(content='Check the order first.', name='order'),
                    'Then issue the refund.',
                ],
                defer_loading=True,
            )
        ],
    )

    result = await agent.run('hi')
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if part.part_kind == 'tool-return'
    ]
    assert returns == [{'instructions': 'Check the order first.\n\nThen issue the refund.'}]


async def test_a_blank_instruction_part_contributes_nothing():
    """Whitespace-only content is dropped, the same as a blank literal, so it can't leave an empty block."""
    agent = Agent(
        instructions=[
            InstructionPart(content='Real.', name='real'),
            InstructionPart(content='   \n  ', name='blank'),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Real.', name='real', id=agent_instruction_id('real'))
    ]


async def test_a_source_without_a_key_cannot_claim_someone_elses():
    """A name needs a source key to hang off, and a toolset with no `id` has none.

    Left as written, an author's raw value would become a top-level key — `'agent'` here, taking over
    the agent's own block for anything keying configuration off these ids.
    """
    agent = Agent(
        instructions='The agent block.',
        toolsets=[
            InstructionsToolset([InstructionPart(content='From a nameless toolset.', id=agent_instruction_id())])
        ],
    )

    # Only the id it could not qualify is dropped; the part keeps everything else it declared.
    assert await run_and_capture(agent) == [
        InstructionPart(content='The agent block.', id=agent_instruction_id()),
        InstructionPart(content='From a nameless toolset.'),
    ]


async def test_a_retained_override_survives_a_rebind_that_replaces_its_children():
    """A container retained for its `get_instructions` override has to be rebound with its children.

    Flattening splats its children into `capabilities`, so the container itself is not in the list
    `for_agent`/`for_run` rebind and would otherwise keep answering from pre-bind children — and its
    leaves, absent from the ordering positions, would sort its block last despite `outermost`.
    """

    class Rebinding(Capability[Any]):
        """Returns a fresh instance from `for_agent`, which is what makes the container stale."""

        def for_agent(self, agent: Any) -> AbstractCapability[Any]:
            return Rebinding(instructions=self.get_instructions(), id=self.id)

    class OverriddenCombined(CombinedCapability[Any]):
        id = 'group'

        def get_instructions(self) -> Any:
            return ['Override.']

    agent = Agent(
        capabilities=[
            OverriddenCombined(capabilities=[Rebinding(instructions='Child.', id='child')], id='group'),
            Capability[Any](instructions='Plain.', id='plain'),
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Override.', id=capability_instruction_id('group')),
        InstructionPart(content='Plain.', id=capability_instruction_id('plain')),
    ]

    # The other side of the rebind: a sibling returns a fresh instance while the container's own
    # child does not, so the container is carried across untouched rather than rebuilt.
    unchanged = Agent(
        capabilities=[
            OverriddenCombined(capabilities=[Capability[Any](instructions='Child.', id='child')], id='group'),
            Rebinding(instructions='Sibling.', id='sibling'),
        ]
    )

    assert await run_and_capture(unchanged) == [
        InstructionPart(content='Override.', id=capability_instruction_id('group')),
        InstructionPart(content='Sibling.', id=capability_instruction_id('sibling')),
    ]


async def test_a_function_toolsets_instructions_can_declare_a_part():
    """A toolset's constructor is one of the places a part is accepted, so it keeps what the part says.

    Reducing it to text here would drop both the declared name and `dynamic`, which decides whether the
    block falls inside the cacheable prefix — silently, since neither has anywhere to surface.
    """
    toolset = FunctionToolset[Any](
        id='weather',
        instructions=[
            InstructionPart(content='Static and named.', name='limits'),
            InstructionPart(content='Recomputed elsewhere.', dynamic=True),
            'A plain string.',
        ],
    )

    # The declared block keeps its id, and the part that declared itself dynamic sorts after the
    # static ones -- which is the flag being honoured, since it decides the cache boundary.
    assert await run_and_capture(Agent(toolsets=[toolset])) == [
        InstructionPart(content='Static and named.', name='limits', id=toolset_instruction_id('weather', 'limits')),
        InstructionPart(content='A plain string.', id=toolset_instruction_id('weather')),
        InstructionPart(content='Recomputed elsewhere.', dynamic=True, id=toolset_instruction_id('weather')),
    ]
