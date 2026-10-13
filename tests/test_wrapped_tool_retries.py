"""Retry counts a tool's `prepare` sees when its toolset is listed under a name-changing wrapper.

The run keys `ctx.retries` by the name the model calls a tool by, while a `FunctionToolset` holds
the tool under its own name, so `.prefixed()`, `.renamed()` and `PrefixTools` used to leave
`prepare` reading `ctx.retry == 0` after the tool failed. These tests run whole agent runs with a
`FunctionModel`, since the retry state lives in the agent loop; no provider is involved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, PrefixTools, Toolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset, PrefixedToolset, ToolsetTool


def calling(exposed: str) -> FunctionModel:
    """A model that calls `exposed` while it is listed, and answers once it isn't."""

    def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(tool.name == exposed for tool in info.function_tools):
            return ModelResponse(parts=[ToolCallPart(exposed, {})])
        return ModelResponse(parts=[TextPart('No tools left.')])

    return FunctionModel(call)


def scripted(calls: list[list[str]]) -> tuple[FunctionModel, list[list[str]]]:
    """A model that calls the listed ones of `calls[step]` at each step, then answers `'done'`.

    Also returns the tool names the model saw at each step.
    """
    visible: list[list[str]] = []

    def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = sorted(tool.name for tool in info.function_tools)
        visible.append(names)
        if len(visible) <= len(calls) and (
            parts := [ToolCallPart(n, {}) for n in calls[len(visible) - 1] if n in names]
        ):
            return ModelResponse(parts=parts)
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(call), visible


def failing_toolset(seen: list[int]) -> FunctionToolset[object]:
    """A toolset whose `fail` tool always fails, with a `prepare` that hides it once it has."""
    toolset = FunctionToolset[object]()

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        seen.append(ctx.retry)
        return None if ctx.retry else tool_def

    @toolset.tool(prepare=hide_after_failure)
    async def fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Choose another approach.')

    return toolset


@dataclass(frozen=True)
class Case:
    id: str
    wrap: Callable[[FunctionToolset[object]], AbstractToolset[object] | AbstractCapability[object]]
    exposed: str


CASES = [
    Case('unwrapped', lambda ts: ts, 'fail'),
    Case('prefixed', lambda ts: ts.prefixed('demo'), 'demo_fail'),
    Case('prefix_ending_in_underscore', lambda ts: ts.prefixed('demo_'), 'demo__fail'),
    Case('nested_prefixes', lambda ts: ts.prefixed('a').prefixed('b'), 'b_a_fail'),
    Case('renamed', lambda ts: ts.renamed({'x': 'fail'}), 'x'),
    Case('identity_rename', lambda ts: ts.renamed({'fail': 'fail'}), 'fail'),
    # The rename inverts the map, so the last entry names the tool.
    Case('two_names_for_one_tool', lambda ts: ts.renamed({'x': 'fail', 'y': 'fail'}), 'y'),
    Case('renamed_then_prefixed', lambda ts: ts.renamed({'x': 'fail'}).prefixed('demo'), 'demo_x'),
    Case('prefixed_then_renamed', lambda ts: ts.prefixed('demo').renamed({'x': 'demo_fail'}), 'x'),
    Case('filtered_inside_prefix', lambda ts: ts.filtered(lambda ctx, td: True).prefixed('demo'), 'demo_fail'),
    Case('prepared_inside_prefix', lambda ts: ts.prepared(lambda ctx, defs: defs).prefixed('demo'), 'demo_fail'),
    Case('dynamic_inside_prefix', lambda ts: DynamicToolset[object](lambda ctx: ts).prefixed('demo'), 'demo_fail'),
    Case('prefix_tools', lambda ts: PrefixTools(wrapped=Toolset(ts), prefix='cap'), 'cap_fail'),
    Case(
        'prefix_tools_toolset_func', lambda ts: PrefixTools(wrapped=Toolset(lambda ctx: ts), prefix='cap'), 'cap_fail'
    ),
    Case(
        'nested_prefix_tools',
        lambda ts: PrefixTools(wrapped=PrefixTools(wrapped=Toolset(ts), prefix='i'), prefix='o'),
        'o_i_fail',
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_prepare_sees_retry_under_wrapper(case: Case):
    """A `prepare` that hides its tool after a failure sees the failure under any name-changing wrapper.

    The `prefixed` and `nested_prefixes` cases are the reproduction in #9741.
    """
    seen: list[int] = []
    wrapped = case.wrap(failing_toolset(seen))
    if isinstance(wrapped, AbstractCapability):
        agent = Agent(calling(case.exposed), capabilities=[wrapped])
    else:
        agent = Agent(calling(case.exposed), toolsets=[wrapped])

    result = await agent.run('Try the tool.')

    assert (seen, result.output) == ([0, 1], 'No tools left.')


@pytest.mark.parametrize('wrapper', ['prefixed', 'renamed'])
async def test_wrapped_retry_count_is_per_tool_and_resets_on_success(wrapper: str):
    """Each tool's `prepare` sees only its own count, which resets once the tool succeeds, as unwrapped."""
    seen: dict[str, list[int]] = {'flaky': [], 'ok': []}
    toolset = FunctionToolset[object]()

    async def record(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        seen[tool_def.name].append(ctx.retry)
        return tool_def

    flaky_calls: list[object] = []

    @toolset.tool(prepare=record)
    async def flaky(ctx: RunContext[object]) -> str:
        flaky_calls.append(None)
        if len(flaky_calls) == 1:
            raise ModelRetry('Try again.')
        return 'ok'

    @toolset.tool(prepare=record)
    async def ok(ctx: RunContext[object]) -> str:
        return 'ok'

    if wrapper == 'prefixed':
        wrapped, exposed = toolset.prefixed('demo'), ['demo_flaky', 'demo_ok']
    else:
        wrapped, exposed = toolset.renamed({'p_flaky': 'flaky', 'p_ok': 'ok'}), ['p_flaky', 'p_ok']
    model, _ = scripted([exposed, exposed, exposed])

    await Agent(model, toolsets=[wrapped]).run('Call everything.')

    assert seen == {'flaky': [0, 1, 0, 0], 'ok': [0, 0, 0, 0]}


@pytest.mark.parametrize('wrapper', ['prefixed', 'renamed'])
async def test_wrapped_tool_does_not_see_same_named_outside_tool(wrapper: str):
    """A wrapped tool held as `fail` doesn't read the count of an unrelated tool the model calls `fail`."""
    seen: list[int] = []
    inner = FunctionToolset[object]()

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        seen.append(ctx.retry)
        return None if ctx.retry else tool_def

    @inner.tool(prepare=hide_after_failure)
    async def fail(ctx: RunContext[object]) -> str:
        return 'inner ok'  # pragma: no cover

    outside = FunctionToolset[object]()

    @outside.tool(name='fail', retries=3)
    async def outside_fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Outside failure.')

    if wrapper == 'prefixed':
        wrapped, inner_exposed = inner.prefixed('demo'), 'demo_fail'
    else:
        wrapped, inner_exposed = inner.renamed({'x': 'fail'}), 'x'
    model, visible = scripted([['fail'], ['fail']])

    await Agent(model, toolsets=[outside, wrapped]).run('Call fail.')

    assert seen == [0, 0, 0]
    assert all(inner_exposed in names for names in visible)


async def test_prepare_reading_exposed_name_keeps_working():
    """`ctx.retries` keys stay the names the model calls tools by, so code written against them keeps working."""

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        return None if ctx.retries.get('demo_fail', 0) else tool_def

    toolset = FunctionToolset[object]()

    @toolset.tool(prepare=hide_after_failure)
    async def fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Choose another approach.')

    filtered = failing_toolset([]).filtered(lambda ctx, td: not ctx.retries.get(f'demo_{td.name}', 0))

    for wrapped in (toolset.prefixed('demo'), filtered.prefixed('demo')):
        result = await Agent(calling('demo_fail'), toolsets=[wrapped]).run('Try the tool.')
        assert result.output == 'No tools left.'


async def test_filter_inside_prefix_sees_output_tool_count():
    """A filter inside a wrapper still sees the counts of tools outside it, like the output tool's."""

    class Answer(BaseModel):
        x: int

    toolset = FunctionToolset[object]()

    @toolset.tool_plain
    def search() -> str:
        return 'result'  # pragma: no cover

    visible: list[list[str]] = []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible.append(sorted(tool.name for tool in info.function_tools))
        return ModelResponse(parts=[ToolCallPart('final_result', {'x': len(visible) - 1})])

    agent = Agent(
        FunctionModel(answer),
        toolsets=[toolset.filtered(lambda ctx, td: not ctx.retries.get('final_result', 0)).prefixed('demo')],
        output_type=ToolOutput(Answer),
    )

    @agent.output_validator
    def positive(ctx: RunContext[object], output: Answer) -> Answer:
        if output.x == 0:
            raise ModelRetry('x must be positive')
        return output

    result = await agent.run('Answer.')

    assert (visible, result.output) == ([['demo_search'], []], Answer(x=1))


async def test_prepared_inside_rename_sees_outside_tool_named_like_renamed_away_original():
    """Inside `.renamed({'x': 'fail'})`, `ctx.retries['fail']` is still the count of the tool the model calls `fail`."""
    snapshots: list[dict[str, int]] = []
    inner = FunctionToolset[object]()

    @inner.tool_plain
    def fail() -> str:
        return 'inner ok'  # pragma: no cover

    outside = FunctionToolset[object]()

    @outside.tool(name='fail', retries=3)
    async def outside_fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Outside failure.')

    def spy(ctx: RunContext[object], defs: list[ToolDefinition]) -> list[ToolDefinition]:
        snapshots.append(dict(ctx.retries))
        return defs

    model, _ = scripted([['fail'], ['fail']])

    await Agent(model, toolsets=[outside, inner.prepared(spy).renamed({'x': 'fail'})]).run('Call fail.')

    assert snapshots == [{}, {'fail': 1}, {'fail': 2}]


async def test_prepared_inside_prefix_can_reset_counts_in_place():
    """`ctx.retries` inside a wrapped listing is still the run's own dict, so resetting it in place holds."""

    def reset(ctx: RunContext[object], defs: list[ToolDefinition]) -> list[ToolDefinition]:
        ctx.retries.pop('demo_fail', None)
        return defs

    toolset = FunctionToolset[object]()

    @toolset.tool
    async def fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Choose another approach.')

    model, _ = scripted([['demo_fail']] * 3)

    result = await Agent(model, toolsets=[toolset.prepared(reset).prefixed('demo')]).run('Try the tool.')

    assert result.output == 'done'


def helpers_toolset() -> FunctionToolset[object]:
    """An unwrapped toolset whose `h` always fails and is hidden once it has."""
    helpers = FunctionToolset[object]()

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        return None if ctx.retry else tool_def

    @helpers.tool(prepare=hide_after_failure, retries=3)
    async def h(ctx: RunContext[object]) -> str:
        raise ModelRetry('h is down')

    return helpers


async def test_tool_body_under_prefix_lists_another_toolset_with_its_own_counts():
    """A wrapped tool's context, handed to an unrelated toolset, doesn't rename that toolset's tools."""
    helpers = helpers_toolset()
    meta = FunctionToolset[object]()

    @meta.tool
    async def status(ctx: RunContext[object]) -> str:
        return f'helpers available: {sorted(await helpers.get_tools(ctx))}'

    script = [['h'], ['meta_status']]

    def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(m, ModelResponse) for m in messages)
        if step < len(script):
            return ModelResponse(parts=[ToolCallPart(name, {}) for name in script[step]])
        last = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        return ModelResponse(parts=[TextPart(str(last[0].content))])

    result = await Agent(FunctionModel(call), toolsets=[helpers, meta.prefixed('meta')]).run('Check helpers.')

    assert result.output == 'helpers available: []'


async def test_prepared_under_prefix_lists_another_toolset_with_its_own_counts():
    """A `.prepared()` callback inside a wrapper gets a context that lists an unrelated toolset correctly."""
    helpers = helpers_toolset()
    fallback = FunctionToolset[object]()

    @fallback.tool_plain
    def backup() -> str:
        return 'backup ok'

    async def only_when_h_is_gone(ctx: RunContext[object], defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [] if 'h' in await helpers.get_tools(ctx) else defs

    model, visible = scripted([['h'], ['fb_backup'], ['fb_backup']])

    await Agent(model, toolsets=[helpers, fallback.prepared(only_when_h_is_gone).prefixed('fb')]).run('Go.')

    assert visible == [['h'], ['fb_backup'], ['fb_backup'], ['fb_backup']]


async def test_self_renaming_wrapped_tool_reads_count_of_tool_named_like_its_original():
    """Documented limit: `prepare` reads the count under the name the wrappers would give the tool's key.

    Here the tool's own `prepare` renames `other` to `renamed_other`, so the model calls it
    `demo_renamed_other`, while the lookup uses `demo_other`: the name of an unrelated tool that
    keeps failing. That tool's count is read and the renamed tool is hidden. An unwrapped tool that
    renames itself already reads the count of an unrelated tool named like its original.
    """
    seen: list[int] = []
    inner = FunctionToolset[object]()

    async def rename_and_hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        seen.append(ctx.retry)
        return None if ctx.retry else replace(tool_def, name='renamed_other')

    @inner.tool(prepare=rename_and_hide_after_failure)
    async def other(ctx: RunContext[object]) -> str:
        return 'inner ok'  # pragma: no cover

    outside = FunctionToolset[object]()

    @outside.tool(name='demo_other', retries=5)
    async def outside_other(ctx: RunContext[object]) -> str:
        raise ModelRetry('Outside failure.')

    model, visible = scripted([['demo_other'], ['demo_other']])

    await Agent(model, toolsets=[outside, inner.prefixed('demo')]).run('Call demo_other.')

    assert (seen, visible) == snapshot(
        ([0, 1, 2], [['demo_other', 'demo_renamed_other'], ['demo_other'], ['demo_other']])
    )


async def test_renaming_wrapper_that_pushes_no_layer_reads_count_of_tool_named_like_inner_key():
    """Documented limit: only the built-in wrappers tell the leaf how they rename its tools.

    A `PrefixedToolset` subclass whose `get_tools` doesn't call `super()` exposes `fail` as `o_i_fail`,
    but the leaf only learns of the outer `o` prefix and reads the count of `o_fail`, an unrelated
    tool that keeps failing.
    """

    class OwnPrefixed(PrefixedToolset[object]):
        async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
            return {
                f'{self.prefix}_{name}': replace(
                    tool, toolset=self, tool_def=replace(tool.tool_def, name=f'{self.prefix}_{name}')
                )
                for name, tool in (await self.wrapped.get_tools(ctx)).items()
            }

    seen: list[int] = []
    inner = FunctionToolset[object]()

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        seen.append(ctx.retry)
        return None if ctx.retry else tool_def

    @inner.tool(prepare=hide_after_failure)
    async def fail(ctx: RunContext[object]) -> str:
        return 'inner ok'  # pragma: no cover

    outside = FunctionToolset[object]()

    @outside.tool(name='o_fail', retries=5)
    async def outside_fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Outside failure.')

    model, visible = scripted([['o_fail'], ['o_fail']])

    await Agent(model, toolsets=[outside, OwnPrefixed(inner, prefix='i').prefixed('o')]).run('Call o_fail.')

    assert (seen, visible) == snapshot(([0, 1, 2], [['o_fail', 'o_i_fail'], ['o_fail'], ['o_fail']]))
