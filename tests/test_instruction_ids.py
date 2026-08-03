"""Tests for [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id].

Instruction blocks carry a stable key so a consumer that receives
`ModelRequestParameters.instruction_parts` can address them — e.g. to override their text from a
remote configuration — without depending on their position or wording. One rule, in two halves: a
source key (`agent`, `toolset:x`, `capability:x`) addresses everything that source contributes, and
appending a segment (`agent:x`, `capability:x:y`) addresses one block declared within it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, Hooks, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from .conftest import try_import

with try_import() as mcp_imports_successful:
    from fastmcp.server import FastMCP

    from pydantic_ai.mcp import MCPToolset

pytestmark = pytest.mark.anyio

instruction_part_ta = TypeAdapter(InstructionPart)


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
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='Weather instructions.', dynamic=True, id='toolset:weather'),
        InstructionPart(content='Calendar instructions.', dynamic=True, id='toolset:calendar'),
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
        InstructionPart(content='Weather instructions.', dynamic=True, id='toolset:weather')
    ]


async def test_toolset_can_declare_ids_for_its_own_blocks():
    """A toolset has no `id=` parameter for its blocks, so it qualifies its own key on the part.

    An id the toolset set itself is closer to the source, so composition leaves it alone.
    """
    agent = Agent(
        toolsets=[
            InstructionsToolset(
                [
                    InstructionPart(content='Tool usage.', id='toolset:weather:limits'),
                    InstructionPart(content='General.'),
                ],
                id='weather',
            )
        ]
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Tool usage.', id='toolset:weather:limits'),
        InstructionPart(content='General.', id='toolset:weather'),
    ]


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
        InstructionPart(content='Weather instructions.', dynamic=True, id='toolset:weather')
    ]


@pytest.mark.skipif(not mcp_imports_successful(), reason='mcp not installed')
async def test_mcp_server_instructions_are_identified():
    """The motivating case: an MCP server's own instructions, addressable by the toolset's id."""
    server: FastMCP[None] = FastMCP('test_server', instructions='You are an MCP test server.')
    agent = Agent(toolsets=[MCPToolset(server, id='test-server', include_instructions=True)])

    async with agent:
        assert await run_and_capture(agent) == [
            InstructionPart(content='You are an MCP test server.', id='toolset:test-server')
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
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='Memory instructions.', id='capability:memory'),
        InstructionPart(content='Anonymous instructions.'),
    ]


async def test_instructions_decorator_can_declare_an_id():
    """A function's own identity isn't stable, but its author can declare one within the source."""
    agent = Agent(instructions='Agent instructions.')

    @agent.instructions(id='local_time')
    def local_time() -> str:
        return 'The time is 10:00.'

    @agent.instructions
    def user_name(ctx: RunContext[Any]) -> str:
        return 'The user is Frank.'

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='The time is 10:00.', dynamic=True, id='agent:local_time'),
        InstructionPart(content='The user is Frank.', dynamic=True),
    ]


async def test_capability_instructions_decorator_can_declare_an_id():
    """A declared block within a capability qualifies the capability's own key."""
    budget = Capability[Any](instructions='Stay within budget.', id='budget')

    @budget.instructions(id='remaining')
    def remaining(ctx: RunContext[Any]) -> str:
        return 'Remaining budget: $10.'

    @budget.instructions
    def undeclared(ctx: RunContext[Any]) -> str:
        return 'Report overruns.'

    agent = Agent(capabilities=[budget])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Stay within budget.', id='capability:budget'),
        InstructionPart(content='Remaining budget: $10.', dynamic=True, id='capability:budget:remaining'),
        InstructionPart(content='Report overruns.', dynamic=True, id='capability:budget'),
    ]


async def test_declared_id_without_a_source_key_stays_unidentified():
    """A capability with no `id` has no source key for a declared id to qualify."""
    anonymous = Capability[Any](instructions='Stay within budget.')

    @anonymous.instructions(id='remaining')
    def remaining(ctx: RunContext[Any]) -> str:
        return 'Remaining budget: $10.'

    agent = Agent(capabilities=[anonymous])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Stay within budget.'),
        InstructionPart(content='Remaining budget: $10.', dynamic=True),
    ]


async def test_a_capability_subclass_keeps_computing_its_own_instructions():
    """Overriding `get_instructions` still wins: the declared-id path only reads stored instructions."""

    class Computed(Capability[Any]):
        def get_instructions(self) -> str:
            return 'Computed by the subclass.'

    agent = Agent(capabilities=[Computed(instructions='Ignored.', id='computed')])

    assert await run_and_capture(agent) == [
        InstructionPart(content='Computed by the subclass.', id='capability:computed')
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

    assert await run_and_capture(agent) == [InstructionPart(content='Override.', id='capability:group')]


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
    with pytest.raises(UserError, match="Declared instruction id 'local:time' cannot contain a colon"):
        agent.instructions(id='local:time')

    capability = Capability[Any](id='budget')
    with pytest.raises(UserError, match="Declared instruction id 'remaining:usd' cannot contain a colon"):
        capability.instructions(id='remaining:usd')

    with pytest.raises(UserError, match="Toolset id 'weather:remote' cannot contain a colon"):
        Agent(toolsets=[InstructionsToolset(id='weather:remote')])


async def test_declared_instruction_ids_qualify_rather_than_collide():
    """A declared segment can't claim a source key, and the framework doesn't police duplicates.

    Two functions can declare the same id — like two blocks from one toolset sharing its key, and
    like `@agent.toolset(id=...)`, which validates nothing either.
    """
    agent = Agent(instructions='Agent instructions.')

    @agent.instructions(id='agent')
    def first() -> str:
        return 'First.'

    @agent.instructions(id='agent')
    def second() -> str:
        return 'Second.'

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='First.', dynamic=True, id='agent:agent'),
        InstructionPart(content='Second.', dynamic=True, id='agent:agent'),
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
        InstructionPart(content='Memory instructions.', id='capability:memory'),
        InstructionPart(content='Remembered.', dynamic=True, id='capability:memory'),
        InstructionPart(content='Weather usage.', dynamic=True, id='toolset:weather'),
        InstructionPart(content='Weather limits.', dynamic=True, id='toolset:weather'),
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
        InstructionPart(content='Agent instructions.', id='agent'),
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
        InstructionPart(content='Weather instructions.', dynamic=True, id='toolset:weather')
    ]


async def test_capability_without_id_gets_an_unidentified_part_of_its_own():
    """The reserved `agent` block holds only the agent's own instructions, so it isn't merged into."""
    agent = Agent(
        instructions='Agent instructions.', capabilities=[Capability[Any](instructions='Extra instructions.')]
    )

    model, captured = capture_instruction_parts()
    result = await agent.run('Hello', model=model)

    assert captured == [
        InstructionPart(content='Agent instructions.', id='agent'),
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
        InstructionPart(content='Memory instructions.', id='capability:memory'),
        InstructionPart(content='Wrapper instructions.', id='capability:search'),
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
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='Memory.', id='capability:memory'),
        InstructionPart(content='Run instructions.'),
    ]

    with agent.override(instructions=['One.', 'Two.']):
        assert await run_and_capture(agent) == [
            InstructionPart(
                content="""\
One.

Two.\
""",
                id='agent',
            )
        ]


async def test_instruction_blocks_are_separated_by_a_blank_line():
    """Every instruction block is joined the same way, and empty contributions leave no artifacts."""

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
            id='agent',
        ),
        InstructionPart(content='Third.', id='capability:memory'),
        InstructionPart(content='Fourth.', dynamic=True, id='toolset:weather'),
    ]
    assert rendered_instructions(result.all_messages()) == 'First.\n\nSecond.\n\nThird.\n\nFourth.'


async def test_before_model_request_can_rewrite_a_block_by_id():
    """Addressing a block by id is what the ids are for, so history has to show the rewrite."""

    def override_weather(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        parts = request_context.model_request_parameters.instruction_parts or []
        request_context.model_request_parameters.instruction_parts = [
            replace(part, content='Managed weather instructions.') if part.id == 'toolset:weather' else part
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
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(content='Managed weather instructions.', dynamic=True, id='toolset:weather'),
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

    assert captured == [InstructionPart(content='Agent instructions.', id='agent')]
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
        InstructionPart(content='Dynamic.', dynamic=True, id='toolset:weather'),
        InstructionPart(content='Static.'),
        InstructionPart(content='Identified.', id='capability:memory'),
    ]

    assert InstructionPart.join(parts) == 'Dynamic.\n\nStatic.\n\nIdentified.'
    assert InstructionPart.sorted(parts) == [
        InstructionPart(content='Static.'),
        InstructionPart(content='Identified.', id='capability:memory'),
        InstructionPart(content='Dynamic.', dynamic=True, id='toolset:weather'),
    ]


def test_serialization_round_trip():
    identified = InstructionPart(content='Weather.', dynamic=True, id='toolset:weather')
    assert instruction_part_ta.dump_python(identified, mode='json') == {
        'content': 'Weather.',
        'dynamic': True,
        'id': 'toolset:weather',
        'part_kind': 'instruction',
    }
    assert instruction_part_ta.validate_python(instruction_part_ta.dump_python(identified)) == identified

    # Payloads recorded before the field existed still validate, and stay unidentified.
    assert instruction_part_ta.validate_python(
        {'content': 'Weather.', 'dynamic': True, 'part_kind': 'instruction'}
    ) == InstructionPart(content='Weather.', dynamic=True)


def test_repr_omits_unset_id():
    assert repr(InstructionPart(content='Weather.')) == "InstructionPart(content='Weather.')"
    assert (
        repr(InstructionPart(content='Weather.', id='toolset:weather'))
        == "InstructionPart(content='Weather.', id='toolset:weather')"
    )


async def test_deferred_capability_instructions_stay_hidden():
    """Deferred capabilities contribute no parts until loaded, identified or not."""

    agent = Agent(
        instructions='Agent instructions.',
        capabilities=[Capability[Any](instructions='Deferred instructions.', id='deferred', defer_loading=True)],
    )

    assert await run_and_capture(agent) == [
        InstructionPart(content='Agent instructions.', id='agent'),
        InstructionPart(
            content="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool:
- deferred\
""",
            dynamic=True,
        ),
    ]
