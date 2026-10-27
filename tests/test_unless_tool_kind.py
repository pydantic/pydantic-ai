"""`ToolDefinition.unless_tool_kind`, `RunContext.tool_name_for_kind`, and the tool manager instructions see.

Not VCR tests: what is being pinned is which tools and instructions the agent hands the model, which is
decided before any provider is involved, so `FunctionModel` observes it directly.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pydantic_ai import Agent, RunContext, Tool, ToolKind
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolKindLike, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition, ToolPrepareFunc
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage
from tests.function_model_prefix import TOOLS_CHANGE_MID_RUN

FILE_READ = ToolKind('test_unless_tool_kind', 'file_read')
NOTES_READ = ToolKind('test_unless_tool_kind', 'notes_read')
INDEX_READ = ToolKind('test_unless_tool_kind', 'index_read')


def _with(
    *, tool_kind: ToolKindLike | None = None, unless_tool_kind: ToolKindLike | None = None
) -> ToolPrepareFunc[object]:
    def prepare(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind=tool_kind, unless_tool_kind=unless_tool_kind)

    return prepare


def read_file(path: str) -> str:
    # Only offered, never called.
    return f'contents of {path}'  # pragma: no cover


def read_notes() -> str:
    return 'notes'


def read_other_notes() -> str:
    # Only offered, never called.
    return 'other notes'  # pragma: no cover


def read_index() -> str:
    # Only offered, never called.
    return 'index'  # pragma: no cover


def _offered_tools(agent: Agent[object, str], prompt: str = 'hello') -> list[list[str]]:
    offered: list[list[str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append([t.name for t in info.function_tools])
        return ModelResponse(parts=[TextPart('done')])

    agent.run_sync(prompt, model=FunctionModel(model))
    return offered


def test_tool_kind_is_normalized_to_its_id():
    tool_def = ToolDefinition(name='read_notes', unless_tool_kind=FILE_READ)
    assert tool_def.unless_tool_kind == 'test_unless_tool_kind.file_read'
    assert replace(tool_def, unless_tool_kind=None).unless_tool_kind is None


def test_dropped_when_another_tool_has_the_kind():
    agent = Agent(
        tools=[
            Tool(read_notes, prepare=_with(unless_tool_kind=FILE_READ)),
            Tool(read_file, tool_kind=FILE_READ),
        ]
    )
    assert _offered_tools(agent) == [['read_file']]


def test_kept_when_no_tool_has_the_kind():
    agent = Agent(tools=[Tool(read_notes, prepare=_with(unless_tool_kind=FILE_READ))])
    assert _offered_tools(agent) == [['read_notes']]


def test_own_kind_does_not_supersede_itself():
    agent = Agent(tools=[Tool(read_notes, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=FILE_READ))])
    assert _offered_tools(agent) == [['read_notes']]


def test_dropped_tool_is_not_callable():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('read_notes', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        tools=[
            Tool(read_notes, prepare=_with(unless_tool_kind=FILE_READ)),
            Tool(read_file, tool_kind=FILE_READ),
        ],
    )
    result = agent.run_sync('hello')
    retry = result.all_messages()[2].parts[0]
    assert isinstance(retry, ToolReturnPart)
    assert retry.outcome == 'retried'
    assert retry.content == "Unknown tool name: 'read_notes'. Available tools: 'read_file'"


def test_deferred_tool_counts_as_present():
    agent = Agent(
        tools=[
            Tool(read_notes, prepare=_with(unless_tool_kind=FILE_READ)),
            Tool(read_file, tool_kind=FILE_READ, defer_loading=True),
        ]
    )
    offered = _offered_tools(agent)
    assert all('read_notes' not in names and 'read_file' not in names for names in offered)


def test_tools_naming_each_others_kind_keep_the_first():
    notes = Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=FILE_READ))
    file = Tool(read_file, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=NOTES_READ))
    assert _offered_tools(Agent(tools=[notes, file])) == [['read_notes']]
    assert _offered_tools(Agent(tools=[file, notes])) == [['read_file']]


def test_a_tool_leading_into_a_cycle_steps_aside_for_the_cycle_member_that_stays():
    # `read_file` only leads into the `read_notes`/`read_index` cycle, so the tie-break keeps `read_notes`,
    # the first tool on the cycle, and both tools that name its kind step aside.
    agent = Agent(
        tools=[
            Tool(read_file, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=NOTES_READ)),
            Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=INDEX_READ)),
            Tool(read_index, prepare=_with(tool_kind=INDEX_READ, unless_tool_kind=NOTES_READ)),
        ]
    )
    assert _offered_tools(agent) == [['read_notes']]


def test_a_pinned_tool_wins_against_every_tool_whose_kind_it_names():
    # `read_file` and `read_notes` form one cycle, `read_index` and `read_other_notes` another. Pinning
    # `read_file` also sets aside `read_other_notes`, which has the kind it names, so `read_index` stays.
    agent = Agent(
        tools=[
            Tool(read_file, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=NOTES_READ)),
            Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=FILE_READ)),
            Tool(read_other_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=INDEX_READ)),
            Tool(read_index, prepare=_with(tool_kind=INDEX_READ, unless_tool_kind=NOTES_READ)),
        ]
    )
    assert _offered_tools(agent) == [['read_file', 'read_index']]


def test_an_odd_cycle_never_leaves_a_tool_out_without_its_replacement():
    # A unless B, B unless C, C unless A can't satisfy every rule. `read_file` stays first on the cycle,
    # which sets `read_notes` aside and drops `read_index`; with `read_index` gone `read_notes` has
    # nothing to step aside for, so it comes back, beside `read_file` that names its kind.
    agent = Agent(
        tools=[
            Tool(read_file, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=NOTES_READ)),
            Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=INDEX_READ)),
            Tool(read_index, prepare=_with(tool_kind=INDEX_READ, unless_tool_kind=FILE_READ)),
        ]
    )
    assert _offered_tools(agent) == [['read_file', 'read_notes']]


def test_only_a_tool_that_stays_supersedes():
    # `read_index` drops `read_notes`, so `read_file`, which would only have stepped aside for `read_notes`, stays.
    agent = Agent(
        tools=[
            Tool(read_file, prepare=_with(tool_kind=FILE_READ, unless_tool_kind=NOTES_READ)),
            Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=INDEX_READ)),
            Tool(read_index, tool_kind=INDEX_READ),
        ]
    )
    assert _offered_tools(agent) == [['read_file', 'read_index']]


def test_an_undefined_kind_is_nothing_to_step_aside_for():
    # Only ids are compared, so a capability can name another package's kind without importing it.
    agent = Agent(tools=[Tool(read_notes, prepare=_with(unless_tool_kind='another_package.file_read'))])
    assert _offered_tools(agent) == [['read_notes']]


def test_tool_name_for_kind_names_the_prefixed_tool():
    seen: list[tuple[str | None, str | None]] = []
    agent = Agent(
        capabilities=[
            PrefixTools(wrapped=Toolset(FunctionToolset([Tool(read_file, tool_kind=FILE_READ)])), prefix='ws')
        ]
    )

    @agent.instructions
    def instructions(ctx: RunContext[object]) -> str:
        seen.append((ctx.tool_name_for_kind(FILE_READ), ctx.tool_name_for_kind(NOTES_READ.id)))
        return 'Read files.'

    assert _offered_tools(agent) == [['ws_read_file']]
    assert seen == [('ws_read_file', None)]


def test_tool_name_for_kind_returns_the_first_of_several():
    def read_other_file(path: str) -> str:
        return path  # pragma: no cover

    seen: list[str | None] = []
    agent = Agent(tools=[Tool(read_file, tool_kind=FILE_READ), Tool(read_other_file, tool_kind=FILE_READ)])

    @agent.instructions
    def instructions(ctx: RunContext[object]) -> str:
        seen.append(ctx.tool_name_for_kind(FILE_READ))
        return 'Read files.'

    _offered_tools(agent)
    assert seen == ['read_file']


def test_tool_name_for_kind_skips_a_superseded_tool():
    seen: list[str | None] = []
    agent = Agent(
        tools=[
            Tool(read_notes, prepare=_with(tool_kind=NOTES_READ, unless_tool_kind=FILE_READ)),
            Tool(read_file, tool_kind=FILE_READ),
        ]
    )

    @agent.instructions
    def instructions(ctx: RunContext[object]) -> str:
        seen.append(ctx.tool_name_for_kind(NOTES_READ))
        return 'Read notes.'

    _offered_tools(agent)
    assert seen == [None]


def test_tool_name_for_kind_without_a_tool_manager():
    ctx = RunContext(deps=None, model=FunctionModel(lambda m, i: ModelResponse(parts=[])), usage=RunUsage())
    assert ctx.tool_name_for_kind(FILE_READ) is None


@pytest.mark.moves_cache_prefix(reason=TOOLS_CHANGE_MID_RUN)
def test_instructions_see_the_current_steps_tools():
    """Instruction functions get the tool manager prepared for their own step, not the previous one's."""

    async def from_second_step(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        return tool_def if ctx.run_step > 1 else None

    instructions_by_step: list[str | None] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions_by_step.append(info.instructions)
        if len(instructions_by_step) == 1:
            return ModelResponse(parts=[ToolCallPart('read_notes', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        tools=[read_notes, Tool(read_file, tool_kind=FILE_READ, prepare=from_second_step)],
    )

    @agent.instructions
    def instructions(ctx: RunContext[object]) -> str:
        return f'File reader: {ctx.tool_name_for_kind(FILE_READ)}'

    agent.run_sync('hello')
    assert instructions_by_step == ['File reader: None', 'File reader: read_file']


def test_prepare_hooks_see_the_previous_steps_tools():
    """A `prepare` runs while the step's tools are collected, so it sees the previous step's, or none on the first."""
    seen_by_step: dict[int, str | None] = {}

    async def record(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        seen_by_step.setdefault(ctx.run_step, ctx.tool_name_for_kind(FILE_READ))
        return tool_def

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('read_notes', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), tools=[Tool(read_notes, prepare=record), Tool(read_file, tool_kind=FILE_READ)])
    agent.run_sync('hello')
    assert list(seen_by_step.values()) == [None, 'read_file']
