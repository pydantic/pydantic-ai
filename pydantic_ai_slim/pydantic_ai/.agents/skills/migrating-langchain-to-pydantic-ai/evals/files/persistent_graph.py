from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class CheckoutState(TypedDict):
    order_id: str
    amount_cents: int
    approval_id: str | None
    approved: bool | None
    charge_id: str | None
    status: Literal["new", "awaiting_approval", "charged", "denied"]


async def request_approval(state: CheckoutState) -> Command[Literal["charge", "deny"]]:
    decision = interrupt(
        {
            "approval_id": state["approval_id"],
            "order_id": state["order_id"],
            "amount_cents": state["amount_cents"],
        }
    )
    if decision["approved"]:
        return Command(goto="charge", update={"approved": True})
    return Command(goto="deny", update={"approved": False})


async def charge(state: CheckoutState) -> dict[str, object]:
    charge_id = await payments.charge(
        order_id=state["order_id"],
        amount_cents=state["amount_cents"],
        idempotency_key=f"checkout:{state['order_id']}",
    )
    return {"charge_id": charge_id, "status": "charged"}


async def deny(state: CheckoutState) -> dict[str, object]:
    return {"status": "denied"}


builder = StateGraph(CheckoutState)
builder.add_node("approval", request_approval)
builder.add_node("charge", charge)
builder.add_node("deny", deny)
builder.add_edge(START, "approval")
builder.add_edge("charge", END)
builder.add_edge("deny", END)


@asynccontextmanager
async def open_graph(database_path: str) -> AsyncIterator[object]:
    async with AsyncSqliteSaver.from_conn_string(database_path) as checkpointer:
        yield builder.compile(checkpointer=checkpointer)
