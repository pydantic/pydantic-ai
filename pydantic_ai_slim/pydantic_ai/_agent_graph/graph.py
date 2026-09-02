from __future__ import annotations as _annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeGuard

from typing_extensions import TypeVar

from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from .. import result
from ..output import OutputSpec
from .state import GraphAgentDeps, GraphAgentState

if TYPE_CHECKING:
    from .model_request import ModelRequestNode as ModelRequestNode
    from .model_response import CallToolsNode as CallToolsNode
    from .user_prompt import UserPromptNode as UserPromptNode


DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
T = TypeVar('T')
S = TypeVar('S')


class AgentNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], result.FinalResult[NodeRunEndT]]):
    """The base class for all agent nodes.

    Using subclass of `BaseNode` for all nodes reduces the amount of boilerplate of generics everywhere
    """


def is_agent_node(
    node: BaseNode[GraphAgentState, GraphAgentDeps[T, Any], result.FinalResult[S]] | End[result.FinalResult[S]],
) -> TypeGuard[AgentNode[T, S]]:
    """Check if the provided node is an instance of `AgentNode`.

    Usage:

        if is_agent_node(node):
            # `node` is an AgentNode
            ...

    This method preserves the generic parameters on the narrowed type, unlike `isinstance(node, AgentNode)`.
    """
    return isinstance(node, AgentNode)


async def drain_node_event_stream(
    node: AgentNode[T, S],
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[T, S]],
) -> None:
    """Run the node's event stream to completion, so capabilities wrapping it see its events.

    `ModelRequestNode` and `CallToolsNode` are the nodes that emit events; the rest never do.
    Both record that their stream was opened, so a caller that streamed the node itself under
    [`agent.iter()`][pydantic_ai.agent.Agent.iter] doesn't get it streamed a second time when
    the run is then advanced.
    """
    from .model_request import ModelRequestNode
    from .model_response import CallToolsNode

    if isinstance(node, ModelRequestNode):
        if node._did_stream:  # pyright: ignore[reportPrivateUsage]
            return
        async with node.stream(ctx) as model_stream:
            async for _event in model_stream:
                pass
    elif isinstance(node, CallToolsNode):
        if node._wrapped_events_iterator is not None:  # pyright: ignore[reportPrivateUsage]
            return
        async with node.stream(ctx) as tool_stream:
            async for _event in tool_stream:
                pass


@dataclasses.dataclass
class SetFinalResult(AgentNode[DepsT, NodeRunEndT]):
    """A node that immediately ends the graph run after a streaming response produced a final result."""

    final_result: result.FinalResult[NodeRunEndT]

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[result.FinalResult[NodeRunEndT]]:
        return End(self.final_result)


def build_agent_graph(
    name: str | None,
    deps_type: type[DepsT],
    output_type: OutputSpec[OutputT],
) -> Graph[
    GraphAgentState,
    GraphAgentDeps[DepsT, OutputT],
    UserPromptNode[DepsT, OutputT],
    result.FinalResult[OutputT],
]:
    """Build the execution [Graph][pydantic_graph.Graph] for a given agent."""
    from .model_request import ModelRequestNode
    from .model_response import CallToolsNode
    from .user_prompt import UserPromptNode

    g = GraphBuilder(
        name=name or 'Agent',
        state_type=GraphAgentState,
        deps_type=GraphAgentDeps[DepsT, OutputT],
        input_type=UserPromptNode[DepsT, OutputT],
        output_type=result.FinalResult[OutputT],
        auto_instrument=False,
    )

    g.add(
        g.edge_from(g.start_node).to(UserPromptNode[DepsT, OutputT]),
        g.node(UserPromptNode[DepsT, OutputT]),
        g.node(ModelRequestNode[DepsT, OutputT]),
        g.node(CallToolsNode[DepsT, OutputT]),
        g.node(
            SetFinalResult[DepsT, OutputT],
        ),
    )
    return g.build(validate_graph_structure=False)
