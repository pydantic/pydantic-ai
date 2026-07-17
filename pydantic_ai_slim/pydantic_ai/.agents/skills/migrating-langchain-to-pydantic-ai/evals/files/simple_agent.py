from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel


class OrderAnswer(BaseModel):
    order_id: str
    status: str
    explanation: str


@dataclass
class RuntimeContext:
    customer_id: str
    locale: str
    orders: "OrderService"


@tool
async def lookup_order(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> str:
    """Look up an order belonging to the authenticated customer."""
    return await runtime.context.orders.lookup(runtime.context.customer_id, order_id)


@tool
async def refund_order(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> str:
    """Refund an order belonging to the authenticated customer."""
    return await runtime.context.orders.refund(runtime.context.customer_id, order_id)


@dynamic_prompt
def localized_prompt(request) -> str:
    return f"Reply using locale {request.runtime.context.locale}."


agent = create_agent(
    model="provider:model",
    tools=[lookup_order, refund_order],
    context_schema=RuntimeContext,
    response_format=OrderAnswer,
    middleware=[
        localized_prompt,
        HumanInTheLoopMiddleware(interrupt_on={"refund_order": True}),
    ],
)


async def handle(messages: list[dict[str, str]], context: RuntimeContext):
    return await agent.ainvoke({"messages": messages}, context=context)
