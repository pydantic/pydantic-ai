"""Build a storefront support agent whose configuration can be rewritten from Logfire.

`AgentControl` makes an agent's instructions, model, model settings, and tool descriptions editable
from Logfire, one addressable piece at a time. So this agent is deliberately assembled from many
pieces: its prompt is six separately addressable blocks, written in five different ways, and its
four tools live in two toolsets.

| Where it is written                                    | Block id                  |
| ------------------------------------------------------ | ------------------------- |
| `Agent(instructions='You are the Northwind ...')`      | `agent`                   |
| `InstructionPart(name='escalation')` beside it         | `agent:escalation`        |
| `StorePolicy(id='store_policy')`, a capability         | `capability:store_policy` |
| `FunctionToolset(id='orders', instructions=...)`       | `toolset:orders`          |
| `FunctionToolset(id='catalog', instructions=...)`       | `toolset:catalog`         |
| `@agent.instructions(name='tenant')`, reading `deps`   | `agent:tenant` (dynamic)  |

Run it with no Logfire configured at all and it answers on the code-default path: nothing resolves,
nothing is different, and it says so. That is the behavior to check first -- an agent that only works
once its config store is reachable has moved a redeploy into an outage.

Point it at a Logfire to see a published config arrive: set `LOGFIRE_BASE_URL` and a
`LOGFIRE_API_KEY` with the `project:read_variables` scope (see `examples/README.md`). Every run
prints what resolved and the prompt block by block, so an edit saved in Logfire is visible on the
next run.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date

import logfire

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.logfire import AgentControl, resolution_reason

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

ORDERS = {
    'A-1001': {'customer': 'cus_42', 'status': 'delivered', 'total': '129.00', 'item': 'Oak desk lamp'},
    'A-1002': {'customer': 'cus_42', 'status': 'in transit', 'total': '38.50', 'item': 'Linen napkins'},
    'A-2001': {'customer': 'cus_99', 'status': 'delivered', 'total': '412.00', 'item': 'Standing desk'},
}

CATALOG = [
    {'sku': 'LMP-OAK-1', 'name': 'Oak desk lamp', 'price': '129.00'},
    {'sku': 'NAP-LIN-4', 'name': 'Linen napkins', 'price': '38.50'},
    {'sku': 'DSK-STD-2', 'name': 'Standing desk', 'price': '412.00'},
]


@dataclass
class SupportDeps:
    """Who this run is serving: the storefront, and the signed-in customer."""

    tenant: str
    customer_id: str


@dataclass
class StorePolicy(AbstractCapability[SupportDeps]):
    """Store policy as a capability, so the policy text is `capability:store_policy` in Logfire.

    Returning a plain `str` is what keeps it editable: a capability that recomputes its instructions
    per request contributes a dynamic block, which Logfire shows but never offers to change.
    """

    def get_instructions(self) -> str:
        """The policy the agent quotes, and the one block a policy change would edit."""
        return 'Store policy: a delivered order can be refunded, no questions asked.'


@dataclass
class RequestProbe(AbstractCapability[SupportDeps]):
    """Record what the model was actually shown, so this example can write out its configuration.

    Pinned `innermost` so it observes each request after every other capability -- `AgentControl`
    included -- has had its say, which is the only place the published config is visible as the
    prompt and tool definitions the model really got.
    """

    requests: list[ModelRequestContext] = field(default_factory=list[ModelRequestContext])
    resolutions: list[str] = field(default_factory=list[str])

    def get_ordering(self) -> CapabilityOrdering:
        """Observe the request last, after every capability that shapes it."""
        return CapabilityOrdering(position='innermost')

    async def before_model_request(
        self, ctx: RunContext[SupportDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Keep the assembled request, and what Logfire resolved while it was being assembled."""
        self.requests.append(request_context)
        # Read here rather than after the run: the resolution is scoped to the run, so outside it
        # `AgentControl.resolved` is `None` whether or not anything resolved.
        self.resolutions.append(describe_resolution())
        return request_context


# Both are module-level so `main()` can read them back: the resolution and the assembled prompt are
# what this example is written to show, and `build_agent()` returns the agent itself.
control = AgentControl[SupportDeps](label='production')
probe = RequestProbe()


def build_orders_toolset() -> FunctionToolset[SupportDeps]:
    """Order tools. The toolset's `id` is what makes its instructions `toolset:orders` in Logfire."""

    def lookup_order(ctx: RunContext[SupportDeps], order_id: str) -> str:
        """Look up one order belonging to the signed-in customer.

        Args:
            ctx: The run context, carrying the customer this run is serving.
            order_id: The order reference as the customer quotes it, e.g. 'A-1001'.
        """
        order = ORDERS.get(order_id.strip().upper())
        if order is None or order['customer'] != ctx.deps.customer_id:
            return f'No order {order_id} for this customer.'
        return f'Order {order_id}: {order["item"]}, {order["status"]}, total ${order["total"]}.'

    def refund_order(ctx: RunContext[SupportDeps], order_id: str, reason: str) -> str:
        """Start a refund for one of the signed-in customer's orders.

        Args:
            ctx: The run context, carrying the customer this run is serving.
            order_id: The order reference to refund.
            reason: Why the customer wants the refund, in their own words.
        """
        order = ORDERS.get(order_id.strip().upper())
        if order is None or order['customer'] != ctx.deps.customer_id:
            return f'No order {order_id} for this customer.'
        # The refund window in `StorePolicy` runs from delivery, so an order still on its way has
        # nothing to refund yet. A tool enforcing the half of the policy it owns keeps the prompt
        # and the code from disagreeing when Logfire reworks the policy text.
        if order['status'] != 'delivered':
            return f'Order {order_id} is {order["status"]}, so it is not refundable yet.'
        return f'Refund opened for {order_id} (${order["total"]}), reason: {reason}.'

    return FunctionToolset[SupportDeps](
        [lookup_order, refund_order],
        id='orders',
        instructions='Order tools are authoritative for status and refunds. Never guess an order status.',
    )


def build_catalog_toolset() -> FunctionToolset[SupportDeps]:
    """Catalog tools, in a second toolset so a published tool override can be narrowed by `toolset`."""

    def search_catalog(ctx: RunContext[SupportDeps], query: str, limit: int = 3) -> str:
        """Search the storefront catalog by product name.

        Args:
            ctx: The run context, unused here but part of the tool signature.
            query: Words from the product name, e.g. 'desk lamp'.
            limit: How many matches to return at most.
        """
        words = query.lower().split()
        matches = [item for item in CATALOG if all(word in item['name'].lower() for word in words)]
        if not matches:
            return f'No catalog matches for {query!r}.'
        # `max(limit, 0)`: a negative limit would otherwise slice from the end of the list and
        # return more matches the lower it went.
        shown = matches[: max(limit, 0)]
        return '; '.join(f'{item["sku"]} {item["name"]} ${item["price"]}' for item in shown)

    def lookup_stock(ctx: RunContext[SupportDeps], sku: str) -> str:
        """Report how many units of one SKU are on hand.

        Args:
            ctx: The run context, unused here but part of the tool signature.
            sku: The catalog SKU, e.g. 'LMP-OAK-1'.
        """
        if sku.strip().upper() not in {item['sku'] for item in CATALOG}:
            return f'Unknown SKU {sku}.'
        return f'{sku}: 7 units on hand.'

    return FunctionToolset[SupportDeps](
        [search_catalog, lookup_stock],
        id='catalog',
        instructions='Quote catalog prices exactly as returned; never round or convert them.',
    )


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[SupportDeps, str]:
    """Build the storefront support agent."""
    agent = Agent(
        model,
        # An explicit `name` is required: it is the `agent__storefront_support` Logfire variable this
        # agent's config lives in, and a config is not something a local rename should move.
        name='storefront_support',
        deps_type=SupportDeps,
        # A code-side setting, so a published `settings` section has something to patch. It patches
        # rather than replaces: a setting Logfire does not mention keeps the value set here.
        model_settings={'max_tokens': 400},
        instructions=[
            # -> block id 'agent'
            'You are the Northwind storefront support agent. Answer in one short sentence.',
            # -> block id 'agent:escalation'. Naming one part of a longer prompt is how you make that
            # part separately editable instead of publishing the whole prompt to change one line.
            InstructionPart(
                content='If the customer is angry, offer to escalate to a human within one reply.',
                name='escalation',
            ),
        ],
        toolsets=[build_orders_toolset(), build_catalog_toolset()],
        capabilities=[
            StorePolicy(id='store_policy'),  # -> block id 'capability:store_policy'
            control,  # resolves `agent__storefront_support` once per run
            probe,  # records what the model was shown; see `print_configuration`
        ],
    )

    @agent.instructions(name='tenant')
    def tenant_context(ctx: RunContext[SupportDeps]) -> str:  # pyright: ignore[reportUnusedFunction]
        """-> block id 'agent:tenant'.

        Recomputed per request because it reads `deps`, so Logfire shows the block and never offers
        to edit it: a published value here would freeze today's date and silence the tenant. Only the
        fact of the block is reported to Logfire, never what it rendered to.
        """
        return (
            f'Today is {date.today().isoformat()}. '
            f'You are serving the {ctx.deps.tenant} storefront for customer {ctx.deps.customer_id}.'
        )

    return agent


def describe_resolution() -> str:
    """One line for what Logfire resolved on the current run: the reason, label, version, sections."""
    resolved = control.resolved
    if resolved is None:
        return 'nothing (outside a run)'
    sections = sorted(resolved.value.model_dump(exclude_none=True))
    return (
        f'reason={resolution_reason(resolved)!r} label={resolved.label!r} '
        f'version={resolved.version} sections={sections or "none"}'
    )


def print_configuration(*, logfire_configured: bool) -> None:
    """Write out the configuration the last model request actually ran with."""
    request = probe.requests[-1]
    parameters = request.model_request_parameters

    print('\n--- what Logfire resolved')
    print(f'  {probe.resolutions[-1]}')
    if not logfire_configured:
        print('  Logfire is not configured, so the agent runs exactly as the code below says.')
        print('  Set LOGFIRE_BASE_URL and LOGFIRE_API_KEY to resolve a published config instead.')

    print('\n--- the prompt the model was shown, block by block')
    for part in parameters.instruction_parts or []:
        block_id = '<unaddressed>' if part.id is None else str(part.id)
        dynamic = ' (dynamic)' if part.dynamic else ''
        print(f'  {block_id}{dynamic}: {part.content[:100]}')

    print('\n--- the tools the model was offered')
    for tool in parameters.function_tools:
        print(f'  {tool.name}: {tool.description}')

    print('\n--- model and settings')
    print(f'  model={request.model.model_name!r} settings={request.model_settings}')


def main() -> None:
    """Answer one question and write out the configuration the run used."""
    # Only when the environment points at a Logfire: instrumentation is how the agent reports itself
    # (`agent_control_config_hint`) and how the version behind a run reaches a trace.
    logfire_configured = bool(os.environ.get('LOGFIRE_API_KEY'))
    if logfire_configured:
        logfire.configure(service_name='agent_control_example', environment='example')
        logfire.instrument_pydantic_ai()

    agent = build_agent()
    prompt = ' '.join(sys.argv[1:]) or "What's the status of order A-1001?"
    result = agent.run_sync(prompt, deps=SupportDeps(tenant='northwind', customer_id='cus_42'))

    print_configuration(logfire_configured=logfire_configured)
    print(f'\n--- answer\n  {result.output}')
    logfire.force_flush()


if __name__ == '__main__':
    main()
