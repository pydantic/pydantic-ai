import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel


class Answer(BaseModel):
    summary: str


@dataclass
class Context:
    customer_id: str
    locale: str
    service: 'Service'


class State(TypedDict):
    messages: list
    topics: list[str]
    evidence: Annotated[list[str], operator.add]
    approved: bool | None


@dynamic_prompt
def policy_prompt(_request) -> str:
    return 'Never disclose authenticated identity.'


@dynamic_prompt
def locale_prompt(request) -> str:
    return f'Reply using locale {request.runtime.context.locale}.'


@tool
async def lookup(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Look up an order for the authenticated customer."""
    return await runtime.context.service.lookup(runtime.context.customer_id, order_id)


@tool(return_direct=True)
async def export_report(report_id: str, runtime: ToolRuntime[Context]) -> str:
    """Export a completed report and end the agent run."""
    return await runtime.context.service.export(runtime.context.customer_id, report_id)


def build_agent(model: BaseChatModel):
    """Build the source agent with an injected deterministic model for offline probes."""
    return create_agent(
        model=model,
        tools=[lookup, export_report],
        context_schema=Context,
        response_format=Answer,
        middleware=[
            policy_prompt,
            locale_prompt,
            ToolRetryMiddleware(max_retries=1),
            ModelCallLimitMiddleware(run_limit=3, thread_limit=6),
            HumanInTheLoopMiddleware(interrupt_on={'export_report': True}),
        ],
        checkpointer=InMemorySaver(),
    )


def plan(_state: State) -> dict[str, list[str]]:
    return {'topics': ['facts', 'policy']}


def fan_out(state: State) -> list[Send]:
    return [Send('research', {'topic': topic}) for topic in state['topics']]


async def research(state: dict[str, str], runtime) -> dict[str, list[str]]:
    result = await runtime.context.service.research(state['topic'])
    return {'evidence': [result]}


def approve(_state: State) -> dict[str, bool]:
    decision = interrupt({'question': 'Export the report?'})
    return {'approved': bool(decision)}


builder = StateGraph(State, context_schema=Context)
builder.add_node('plan', plan)
builder.add_node('research', research)
builder.add_node('approve', approve)
builder.add_edge(START, 'plan')
builder.add_conditional_edges('plan', fan_out, ['research'])
builder.add_edge('research', 'approve')
builder.add_edge('approve', END)
graph = builder.compile(checkpointer=InMemorySaver())


async def stream(input_state: State, context: Context, thread_id: str):
    config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 8}
    async for part in graph.astream(
        input_state,
        config=config,
        context=context,
        stream_mode=['updates', 'messages', 'checkpoints'],
        version='v2',
    ):
        yield part
