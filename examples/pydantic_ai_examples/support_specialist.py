"""Customer-support agent demonstrating capabilities as on-demand runbook skills.

The agent starts with only a compact catalog of runbooks. When a request needs
one, the model calls `load_capability(id)` and the matching capability supplies
its instructions, tools, model settings, hooks, and approval behavior.

Most support requests should not put every runbook, tool schema, and sensitive
procedure into the context window. This example keeps ordinary order tracking
small while still allowing account-security handling to load when needed.

Run with:

    uv run -m pydantic_ai_examples.support_specialist
"""

from dataclasses import dataclass, field

import logfire

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    FunctionToolset,
    ModelSettings,
    RunContext,
    ToolDefinition,
    ToolDenied,
    ToolReturnPart,
)
from pydantic_ai.capabilities import AbstractCapability, Capability

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


@dataclass
class Order:
    id: str
    item: str
    status: str


@dataclass
class LoginEvent:
    ip_address: str
    country: str
    timestamp: str
    trusted_device: bool


@dataclass
class Store:
    orders: dict[str, Order] = field(
        default_factory=lambda: {
            'A-1042': Order('A-1042', 'Pydantic mug', 'delivered'),
            'A-1099': Order('A-1099', 'Pydantic hoodie', 'in_transit'),
        }
    )
    recent_logins: list[LoginEvent] = field(
        default_factory=lambda: [
            LoginEvent('203.0.113.10', 'US', '2026-05-19T09:10:00Z', True),
            LoginEvent('198.51.100.7', 'DE', '2026-05-19T09:42:00Z', False),
        ]
    )


orders_tools = FunctionToolset[Store]()


@orders_tools.tool()
def order_status(ctx: RunContext[Store], order_id: str) -> str:
    """Look up the current status of an order."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    return f'Order {order.id} ({order.item}) is currently {order.status}.'


orders_capability = Capability[Store](
    id='orders',
    description='Look up order status by id.',
    instructions='Quote the item name in your reply.',
    toolset=orders_tools,
    defer_loading=True,
)


returns_tools = FunctionToolset[Store]()


@returns_tools.tool()
def start_return(ctx: RunContext[Store], order_id: str, reason: str) -> str:
    """Open a return request. Only delivered orders can be returned."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    if order.status != 'delivered':
        return f'Order {order.id} is {order.status}; only delivered orders can be returned.'
    return f'Return opened for {order.id}. Reason: {reason!r}.'


returns_capability = Capability[Store](
    id='returns',
    description='Open return requests and answer return-policy questions.',
    instructions='Returns are accepted within 30 days of delivery for unused items.',
    toolset=returns_tools,
    defer_loading=True,
)


security_tools = FunctionToolset[Store]()


@security_tools.tool()
def list_recent_logins(ctx: RunContext[Store]) -> list[LoginEvent]:
    """List recent login attempts for the current customer account."""
    return ctx.deps.recent_logins


@security_tools.tool(requires_approval=True)
def revoke_all_sessions(ctx: RunContext[Store], reason: str) -> str:
    """Revoke every active session for the current customer account."""
    return f'All active sessions revoked. Reason: {reason}'


@dataclass
class AccountSecurityRunbook(AbstractCapability[Store]):
    """Deferred runbook skill for sensitive account-security workflows."""

    id: str = 'account-security'
    description: str | None = (
        'Use for suspicious logins, stolen credentials, account takeover, session revocation, '
        'or urgent account lock requests.'
    )
    defer_loading: bool = True

    def get_toolset(self) -> FunctionToolset[Store]:
        return security_tools

    def get_instructions(self):
        def instructions(ctx: RunContext[Store]) -> str:
            return (
                'You are using the account-security runbook. First inspect recent logins, '
                'explain the risk plainly, and do not revoke sessions unless the user approves it. '
                f'The current account has {len(ctx.deps.recent_logins)} recent login events.'
            )

        return instructions

    def get_model_settings(self):
        def settings(ctx: RunContext[Store]) -> ModelSettings:
            if ctx.capability_loaded:
                return ModelSettings(temperature=0, thinking='high')
            return ModelSettings()

        return settings

    async def prepare_tools(
        self,
        ctx: RunContext[Store],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if not ctx.capability_loaded or self._recent_logins_checked(ctx):
            return tool_defs

        return [
            tool_def for tool_def in tool_defs if tool_def.name != 'revoke_all_sessions'
        ]

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[Store],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        if not ctx.capability_loaded or not requests.approvals:
            return None

        return DeferredToolResults(
            approvals={
                call.tool_call_id: ToolDenied(
                    'The example does not auto-approve session revocation.'
                )
                for call in requests.approvals
            }
        )

    @staticmethod
    def _recent_logins_checked(ctx: RunContext[Store]) -> bool:
        return any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'list_recent_logins'
            for message in ctx.messages
            for part in message.parts
        )


support_agent = Agent(
    model='openai-responses:gpt-5.2',
    deps_type=Store,
    instructions='You are a customer-support agent for an e-commerce store.',
    capabilities=[orders_capability, returns_capability, AccountSecurityRunbook()],
)


async def main() -> None:
    store = Store()
    for prompt in [
        'Where is order A-1042?',
        "I'd like to return A-1042 because it arrived damaged.",
        "I got a login alert from Germany that I don't recognize. Check whether my account is safe.",
    ]:
        print(f'\n> {prompt}')
        result = await support_agent.run(prompt, deps=store)
        print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
