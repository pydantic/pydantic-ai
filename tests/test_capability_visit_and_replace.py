"""Rewriting a capability tree with `AbstractCapability.visit_and_replace`."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities import (
    Capability,
    DynamicCapability,
    PrefixTools,
    Thinking,
    Toolset,
    WebSearch,
    WrapperCapability,
)
from pydantic_ai.capabilities._dynamic import ResolvedDynamicCapability
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

pytestmark = [
    pytest.mark.anyio,
]


def _visited(capability: AbstractCapability[Any]) -> list[tuple[str, str | None]]:
    """The type name and `id` of every capability `apply` visits, in order."""
    visited: list[AbstractCapability[Any]] = []
    capability.apply(visited.append)
    return [(type(cap).__name__, cap.id) for cap in visited]


def test_visit_and_replace_single_capability():
    """AbstractCapability.visit_and_replace() offers just the capability itself."""
    thinking = Thinking(effort='low')
    replacement = Thinking(effort='high')

    assert thinking.visit_and_replace(lambda _: replacement) is replacement
    assert thinking.visit_and_replace(lambda _: None) is None
    assert thinking.visit_and_replace(lambda cap: cap) is thinking


def test_visit_and_replace_combined_capability():
    """CombinedCapability.visit_and_replace() replaces children in place."""
    web_search = WebSearch(local='duckduckgo')
    combined = CombinedCapability[Any]([Thinking(effort='low', id='thinking'), web_search])

    rewritten = combined.visit_and_replace(
        lambda cap: Thinking(effort='high', id='thinking') if isinstance(cap, Thinking) else cap
    )

    assert isinstance(rewritten, CombinedCapability)
    assert _visited(rewritten) == snapshot([('Thinking', 'thinking'), ('WebSearch', 'web_search')])
    assert rewritten.capabilities[1] is web_search
    # The original is left alone: rewriting builds a new tree.
    assert _visited(combined)[0] == ('Thinking', 'thinking')
    assert combined.capabilities[0] is not rewritten.capabilities[0]


def test_visit_and_replace_combined_capability_unchanged():
    """A visitor that changes nothing hands back the very same tree."""
    combined = CombinedCapability[Any]([Thinking(), WebSearch(local='duckduckgo')])
    assert combined.visit_and_replace(lambda cap: cap) is combined


def test_visit_and_replace_empty_combined():
    """An already-empty container has nothing to remove, so it survives untouched."""
    combined = CombinedCapability[Any]([])
    assert combined.visit_and_replace(lambda _: None) is combined


def test_visit_and_replace_removes_combined_child():
    """Removing one child keeps the container and its remaining children."""
    web_search = WebSearch(local='duckduckgo')
    combined = CombinedCapability[Any]([Thinking(id='thinking'), web_search])

    rewritten = combined.visit_and_replace(lambda cap: None if isinstance(cap, Thinking) else cap)

    assert isinstance(rewritten, CombinedCapability)
    assert _visited(rewritten) == snapshot([('WebSearch', 'web_search')])
    assert rewritten.capabilities[0] is web_search


def test_visit_and_replace_removes_every_combined_child():
    """A container emptied by removals reports itself as removed."""
    combined = CombinedCapability[Any]([Thinking(), WebSearch(local='duckduckgo')])
    assert combined.visit_and_replace(lambda _: None) is None


def test_visit_and_replace_splats_a_combined_replacement():
    """A container handed back by the visitor is splatted into the parent, as on construction."""
    web_search = WebSearch(local='duckduckgo')
    combined = CombinedCapability[Any]([Thinking(id='thinking'), web_search])

    rewritten = combined.visit_and_replace(
        lambda cap: (
            CombinedCapability[Any]([Capability(tools=[], id='a'), Capability(tools=[], id='b')])
            if isinstance(cap, Thinking)
            else cap
        )
    )

    assert isinstance(rewritten, CombinedCapability)
    assert [type(cap).__name__ for cap in rewritten.capabilities] == snapshot(['Capability', 'Capability', 'WebSearch'])
    assert _visited(rewritten) == snapshot([('Capability', 'a'), ('Capability', 'b'), ('WebSearch', 'web_search')])


def test_visit_and_replace_wrapper_over_capability():
    """A wrapper over a leaf is offered instead of the leaf, and takes the leaf with it."""
    thinking = Thinking()
    wrapper = WrapperCapability(wrapped=thinking)

    offered: list[AbstractCapability[Any]] = []

    def visit(cap: AbstractCapability[Any]) -> AbstractCapability[Any]:
        offered.append(cap)
        return cap

    assert wrapper.visit_and_replace(visit) is wrapper
    assert offered == [wrapper]
    assert wrapper.visit_and_replace(lambda _: None) is None


def test_visit_and_replace_wrapper_over_combined_capability():
    """A capability nested in a wrapper is removed from the wrapper's delegate, not around it.

    Regression test for the flatten-and-rebuild alternative: because
    [`WrapperCapability.apply`][pydantic_ai.capabilities.WrapperCapability.apply] visits the wrapper
    *and* the leaves of the container it wraps, rebuilding a tree from that flat list keeps the
    wrapper delegating to the dropped capability and re-adds the container's other children next to
    it. Rewriting in place does neither.
    """
    stale = Thinking(effort='low', id='thinking')
    kept = Capability(tools=[], id='bundle')
    wrapper = WrapperCapability(wrapped=CombinedCapability[Any]([stale, kept]))

    assert _visited(wrapper) == snapshot(
        [('WrapperCapability', None), ('Thinking', 'thinking'), ('Capability', 'bundle')]
    )

    rewritten = wrapper.visit_and_replace(lambda cap: None if cap is stale else cap)

    assert isinstance(rewritten, WrapperCapability)
    assert _visited(rewritten) == snapshot([('WrapperCapability', None), ('Capability', 'bundle')])
    assert isinstance(rewritten.wrapped, CombinedCapability)
    assert rewritten.wrapped.capabilities == [kept]


def test_visit_and_replace_wrapper_over_combined_capability_unchanged():
    """A visitor that changes nothing inside a wrapped container hands back the same wrapper."""
    wrapper = WrapperCapability(wrapped=CombinedCapability[Any]([Thinking(), WebSearch(local='duckduckgo')]))
    assert wrapper.visit_and_replace(lambda cap: cap) is wrapper


def test_visit_and_replace_wrapper_over_emptied_combined_capability():
    """A wrapper whose whole subtree was removed is removed too: it has nothing left to modify."""
    wrapper = WrapperCapability(wrapped=CombinedCapability[Any]([Thinking(), WebSearch(local='duckduckgo')]))
    assert wrapper.visit_and_replace(lambda cap: cap if isinstance(cap, WrapperCapability) else None) is None


def test_visit_and_replace_replaces_wrapper_wholesale():
    """Replacing a wrapper takes its subtree with it, so its children are never offered."""
    inner = Thinking(id='thinking')
    wrapper = WrapperCapability(wrapped=CombinedCapability[Any]([inner]))
    replacement = WebSearch(local='duckduckgo')

    offered: list[AbstractCapability[Any]] = []

    def visit(cap: AbstractCapability[Any]) -> AbstractCapability[Any]:
        offered.append(cap)
        return replacement if isinstance(cap, WrapperCapability) else cap

    assert wrapper.visit_and_replace(visit) is replacement
    assert offered == [wrapper]


def test_visit_and_replace_keeps_wrapper_subclass_state():
    """Rebuilding a wrapper preserves subclass fields and re-adopts the new wrapped identity."""
    prefixed = PrefixTools(
        wrapped=CombinedCapability[Any]([Thinking(id='thinking'), Capability(tools=[], id='bundle')]),
        prefix='ns',
    )

    rewritten = prefixed.visit_and_replace(lambda cap: None if isinstance(cap, Thinking) else cap)

    assert isinstance(rewritten, PrefixTools)
    assert rewritten.prefix == 'ns'
    assert _visited(rewritten) == snapshot([('PrefixTools', None), ('Capability', 'bundle')])


async def test_visit_and_replace_resolved_dynamic_capability():
    """A `DynamicCapability` resolved for a run rewrites like any other wrapper."""

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return CombinedCapability([Thinking(id='thinking'), Capability(tools=[], id='bundle')])

    ctx = RunContext[Any](deps=None, model=TestModel(), usage=RunUsage(), run_step=0)
    ctx.agent = Agent(TestModel())
    resolved = await DynamicCapability[Any](factory).for_run(ctx)
    assert isinstance(resolved, ResolvedDynamicCapability)

    rewritten = resolved.visit_and_replace(lambda cap: None if isinstance(cap, Thinking) else cap)

    assert isinstance(rewritten, ResolvedDynamicCapability)
    assert rewritten.dynamic_toolset is resolved.dynamic_toolset
    assert _visited(rewritten) == snapshot([('ResolvedDynamicCapability', None), ('Capability', 'bundle')])


async def test_visit_and_replace_supersedes_nested_capability_in_a_run(allow_model_requests: None):
    """Last-wins supersession across layers keeps the surviving structure intact.

    A run-level capability reusing an agent-level `id` replaces it. Dropping the superseded
    occurrence in place leaves the wrapper prefixing what remains of its container, while rebuilding
    from the flat `apply()` list would keep the wrapper delegating to the dropped capability and
    duplicate the container's other child.
    """
    stale = FunctionToolset(id='stale')

    @stale.tool_plain
    def stale_tool() -> str:
        return 'stale'  # pragma: no cover

    kept = FunctionToolset(id='kept')

    @kept.tool_plain
    def kept_tool() -> str:
        return 'kept'  # pragma: no cover

    fresh = FunctionToolset(id='fresh')

    @fresh.tool_plain
    def fresh_tool() -> str:
        return 'fresh'  # pragma: no cover

    agent_layer = PrefixTools(
        wrapped=CombinedCapability[Any]([Toolset(stale, id='shared'), Toolset(kept, id='bundle')]),
        prefix='ns',
    )
    composed = CombinedCapability[Any]([agent_layer, Toolset(fresh, id='shared')])

    occurrences: dict[str, int] = {}
    for _, cap_id in _visited(composed):
        if cap_id is not None:
            occurrences[cap_id] = occurrences.get(cap_id, 0) + 1
    seen: dict[str, int] = {}

    def supersede(cap: AbstractCapability[Any]) -> AbstractCapability[Any] | None:
        if cap.id is None:
            return cap
        seen[cap.id] = seen.get(cap.id, 0) + 1
        return cap if seen[cap.id] == occurrences[cap.id] else None

    rewritten = composed.visit_and_replace(supersede)
    assert rewritten is not None
    assert _visited(rewritten) == snapshot([('PrefixTools', None), ('Toolset', 'bundle'), ('Toolset', 'shared')])

    seen_tools: list[tuple[str, str | None]] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.extend(sorted((tool.name, tool.capability_id) for tool in info.function_tools))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[rewritten])
    result = await agent.run('list tools')

    assert result.output == 'done'
    assert seen_tools == snapshot([('fresh_tool', 'shared'), ('ns_kept_tool', 'bundle')])
