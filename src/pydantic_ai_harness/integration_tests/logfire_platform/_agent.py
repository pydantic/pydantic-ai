"""The agent this suite runs, and the probe that records what the model was actually shown.

Agent Control's central claim is that an agent's prompt is a composition and every piece of it is
separately addressable, so this agent has one instruction source of every kind Pydantic AI can key,
each labelled with the `id` a published config addresses it by:

| Where it is written                            | Block `id`                |
| ---------------------------------------------- | ------------------------- |
| `Agent(instructions=...)`                      | `agent`                   |
| `InstructionPart(name='escalation')` beside it  | `agent:escalation`        |
| `StorePolicy(id='store_policy')`, a capability | `capability:store_policy` |
| `FunctionToolset(id='orders', instructions=)`  | `toolset:orders`          |
| `FunctionToolset(id='catalog', instructions=)` | `toolset:catalog`         |
| `@agent.instructions(name='tenant')`           | `agent:tenant` (dynamic)  |

The tools live in two toolsets so the `toolset` qualifier on a published tool override is exercised
against a real second toolset rather than a hypothetical one. They have distinct names because
Pydantic AI advertises every tool into one namespace and refuses two toolsets offering the same one,
which is a fact about core rather than about this feature (`tests/logfire_variables` is where that
belongs).

`examples/agent_control.py` is the readable version of the same agent. This one is separate on
purpose: it carries the knobs the scenarios need (`label`, `on_unmatched`, a code-side model) and an
agent name no demo or deployment resolves, and the example stays free of test scaffolding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from logfire.agent_control import AgentConfig, OnUnmatched

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.logfire import AgentControl, resolution_reason

from ._platform import AGENT_NAME

CODE_MODEL = 'anthropic:claude-haiku-4-5'
"""What the agent says in code: cheap, current, and it honors `temperature`, which the settings
scenario publishes and reads back off the request."""

MANAGED_MODEL = 'openai:gpt-5.4-nano'
"""A different provider entirely, so "the published model won" is read off the response rather than
inferred."""

PRODUCTION_LABEL = 'production'
CANARY_LABEL = 'canary'

CODE_SETTINGS: ModelSettings = {'temperature': 0.2, 'max_tokens': 400}
"""The agent's own settings, so a published `settings` section has something to patch over."""

CODE_BLOCKS: dict[str, str] = {
    'agent': 'You are the Northwind storefront support agent. Answer in one short sentence.',
    'agent:escalation': 'If the customer is angry, offer to escalate to a human within one reply.',
    'capability:store_policy': 'Store policy: a delivered order can be refunded, no questions asked.',
    'toolset:orders': 'Order tools are authoritative for status and refunds. Never guess an order status.',
    'toolset:catalog': 'Quote catalog prices exactly as returned; never round or convert them.',
}
"""Every addressable block's text, as written.

The scenarios compare what the model was shown against this rather than against string literals of
their own, so "the rest of the prompt is untouched" is checked against the code the agent runs.
`toolset:orders` contains the substring `auth`, which is what makes it the block a scrubber with a
default deny-list would rewrite on the way to the hint span.
"""

DYNAMIC_BLOCK_ID = 'agent:tenant'
"""The one block recomputed per request, and so the one whose text must never be published."""

ORDERS: dict[str, dict[str, str]] = {
    'A-1001': {'customer': 'cus_42', 'status': 'delivered', 'total': '129.00', 'item': 'Oak desk lamp'},
    'A-1002': {'customer': 'cus_42', 'status': 'in transit', 'total': '38.50', 'item': 'Linen napkins'},
    'A-2001': {'customer': 'cus_99', 'status': 'delivered', 'total': '412.00', 'item': 'Standing desk'},
}

CATALOG: list[dict[str, str]] = [
    {'sku': 'LMP-OAK-1', 'name': 'Oak desk lamp', 'price': '129.00'},
    {'sku': 'NAP-LIN-4', 'name': 'Linen napkins', 'price': '38.50'},
    {'sku': 'DSK-STD-2', 'name': 'Standing desk', 'price': '412.00'},
]


def todays_date() -> str:
    """Today, as the dynamic block writes it.

    Read once per render and kept in `LiveAgent.rendered_dynamic`, because a test that recomputes it
    to compare against a block rendered moments earlier fails whenever the two fall either side of
    midnight -- a real flake, on a suite someone runs by hand at whatever hour they are debugging.
    """
    return date.today().isoformat()


@dataclass
class SupportDeps:
    """Real dependencies: the storefront this run serves, and who is asking.

    Read by the dynamic instruction block and by the order tools, so "a dynamic block reads the run"
    is a fact about this agent rather than a claim about the feature.
    """

    tenant: str
    customer_id: str


DEFAULT_DEPS = SupportDeps(tenant='northwind', customer_id='cus_42')

DYNAMIC_DEPS_TOKENS = (DEFAULT_DEPS.tenant, DEFAULT_DEPS.customer_id)
"""What the dynamic block renders out of `deps`.

Searched for in two directions: it has to be in the prompt the model was shown, and it must not be
anywhere in the baseline reported to Logfire.
"""


@dataclass
class ToolCall:
    """One tool call, as the implementation saw it.

    The other half of the rename evidence: the trace shows the model calling the *managed* name, and
    this shows the code's own function running under its *code-side* `ctx.tool_name`.
    """

    function: str
    ctx_tool_name: str | None
    args: dict[str, object]


@dataclass
class Resolved:
    """What `AgentControl.resolved` said while one model request was being assembled."""

    reason: str | None
    label: str | None
    version: int | None
    config: AgentConfig | None

    @property
    def sections(self) -> list[str]:
        """The config sections the resolved value carries, or none at all."""
        return [] if self.config is None else sorted(self.config.model_dump(exclude_none=True))


@dataclass
class ObservedRequest:
    """One model request, as the model was given it."""

    blocks: list[tuple[str | None, bool, str]]
    """`(id, dynamic, text)` per instruction block, in the order the model sees them."""

    tools: list[tuple[str, str | None, dict[str, Any]]]
    """`(advertised name, description, parameters json schema)` per function tool."""

    model_name: str
    settings: ModelSettings
    resolved: Resolved | None

    @property
    def prompt(self) -> str:
        """The blocks joined the way the model receives them."""
        return '\n\n'.join(text for _, _, text in self.blocks)

    @property
    def block_ids(self) -> list[str | None]:
        """Every block's id, in prompt order. `None` is a block nothing addresses."""
        return [block_id for block_id, _, _ in self.blocks]

    @property
    def tool_names(self) -> list[str]:
        """The tool names the model was offered, in the order it was offered them."""
        return [name for name, _, _ in self.tools]

    def block(self, block_id: str) -> str | None:
        """The text of one block, or `None` when the prompt has no block with that id."""
        for candidate, _, text in self.blocks:
            if candidate == block_id:
                return text
        return None

    def tool(self, name: str) -> tuple[str, str | None, dict[str, Any]]:
        """One advertised tool, by the name the model was shown."""
        for entry in self.tools:
            if entry[0] == name:
                return entry
        raise AssertionError(f'the model was not offered a tool called {name!r}: {self.tool_names}')


@dataclass
class RunProbe(AbstractCapability[SupportDeps]):
    """Record each model request after every other capability has shaped it.

    A capability rather than a monkeypatch because that is the seam the framework offers, and pinned
    `innermost` so it observes the request after `AgentControl`'s overriding half has applied to it.
    """

    read_resolved: Callable[[], Resolved | None] = field(repr=False, default=lambda: None)
    requests: list[ObservedRequest] = field(default_factory=list[ObservedRequest])

    def get_ordering(self) -> CapabilityOrdering:
        """Observe the request last, after every capability that shapes it."""
        return CapabilityOrdering(position='innermost')

    async def before_model_request(
        self, ctx: RunContext[SupportDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Keep the assembled request, and what Logfire had resolved while it was assembled."""
        parameters = request_context.model_request_parameters
        self.requests.append(
            ObservedRequest(
                blocks=[
                    (None if part.id is None else str(part.id), part.dynamic, part.content)
                    for part in parameters.instruction_parts or []
                ],
                tools=[
                    (tool.name, tool.description, tool.parameters_json_schema) for tool in parameters.function_tools
                ],
                model_name=request_context.model.model_name,
                settings=request_context.model_settings or {},
                # Read here rather than after the run returns: the resolution is scoped to the run,
                # so outside it `AgentControl.resolved` is `None` whether or not anything resolved.
                resolved=self.read_resolved(),
            )
        )
        return request_context

    @property
    def last(self) -> ObservedRequest:
        """The most recent model request of the most recent run."""
        return self.requests[-1]


@dataclass
class StorePolicy(AbstractCapability[SupportDeps]):
    """A capability contributing instructions, so `capability:store_policy` has something to address.

    Returning a plain `str` is what keeps the block editable: a capability that recomputes its
    instructions per request contributes a dynamic block, which is shown but never overridable.
    """

    def get_instructions(self) -> str:
        """The store policy, as written in code."""
        return CODE_BLOCKS['capability:store_policy']


@dataclass
class LiveAgent:
    """The built agent, the `AgentControl` on it, and what its last run recorded."""

    agent: Agent[SupportDeps, str]
    control: AgentControl[SupportDeps]
    probe: RunProbe
    tool_calls: list[ToolCall]
    tool_hooks: list[Callable[[ToolCall], None]]
    """Called on every tool call. The once-per-run scenario publishes from inside one."""

    rendered_dynamic: list[str]
    """What the dynamic block returned, per render.

    A test asserts the block in the request against this rather than against a string it builds
    itself: that the block is what the code produced -- and not what someone published over it -- is
    the claim, and it is one no clock can turn over halfway through.
    """

    def run(self, prompt: str, *, model: str | None = None) -> AgentRunResult[str]:
        """Run the agent on this suite's deps, optionally overriding the model at the call site."""
        if model is None:
            return self.agent.run_sync(prompt, deps=DEFAULT_DEPS)
        return self.agent.run_sync(prompt, deps=DEFAULT_DEPS, model=model)

    @property
    def last(self) -> ObservedRequest:
        """The last model request the probe saw."""
        return self.probe.last

    @property
    def requests(self) -> list[ObservedRequest]:
        """Every model request the probe saw, across every run of this agent."""
        return self.probe.requests


def build_agent(
    *,
    label: str | None = PRODUCTION_LABEL,
    on_unmatched: OnUnmatched = 'warn',
    model: str = CODE_MODEL,
    agent_name: str = AGENT_NAME,
) -> LiveAgent:
    """Build the agent as written. Every scenario uses this one; only the knobs below differ.

    Args:
        label: The Logfire label to resolve. `None` lets the variable's rollout choose.
        on_unmatched: What a published entry that reaches nothing costs.
        model: The code-side model, which a published `model` section overrides.
        agent_name: The agent's name, and so the `agent__<name>` variable it resolves. A test that
            reads its hint span back out of the platform passes a name of its own, so the span it
            queries for is one no other test in the run could have emitted.
    """
    control = AgentControl[SupportDeps](label=label, on_unmatched=on_unmatched)
    tool_calls: list[ToolCall] = []
    tool_hooks: list[Callable[[ToolCall], None]] = []
    rendered_dynamic: list[str] = []

    def record(call: ToolCall) -> None:
        tool_calls.append(call)
        for hook in list(tool_hooks):
            hook(call)

    def read_resolved() -> Resolved | None:
        resolved = control.resolved
        if resolved is None:
            return None
        return Resolved(
            reason=resolution_reason(resolved),
            label=resolved.label,
            version=resolved.version,
            config=resolved.value,
        )

    probe = RunProbe(read_resolved=read_resolved)
    agent = Agent(
        model,
        # An explicit name is required, and it is what the `agent__<name>` variable is derived from.
        name=agent_name,
        deps_type=SupportDeps,
        model_settings=CODE_SETTINGS,
        instructions=[
            CODE_BLOCKS['agent'],
            InstructionPart(content=CODE_BLOCKS['agent:escalation'], name='escalation'),
        ],
        toolsets=[_orders_toolset(record), _catalog_toolset(record)],
        capabilities=[StorePolicy(id='store_policy'), control, probe],
    )

    @agent.instructions(name='tenant')
    def tenant_context(ctx: RunContext[SupportDeps]) -> str:
        """-> id `agent:tenant`: dynamic, because it reads the run."""
        rendered = (
            f'Today is {todays_date()}. '
            f'You are serving the {ctx.deps.tenant} storefront for customer {ctx.deps.customer_id}.'
        )
        rendered_dynamic.append(rendered)
        return rendered

    return LiveAgent(
        agent=agent,
        control=control,
        probe=probe,
        tool_calls=tool_calls,
        tool_hooks=tool_hooks,
        rendered_dynamic=rendered_dynamic,
    )


def _orders_toolset(record: Callable[[ToolCall], None]) -> FunctionToolset[SupportDeps]:
    """Order tools. The toolset's `id` makes its instructions addressable as `toolset:orders`."""

    def lookup_order(ctx: RunContext[SupportDeps], order_id: str) -> str:
        """Look up one order belonging to the signed-in customer.

        Args:
            ctx: The run context, carrying the customer this run serves.
            order_id: The order reference as the customer quotes it, e.g. 'A-1001'.
        """
        record(ToolCall('lookup_order', ctx.tool_name, {'order_id': order_id}))
        order = ORDERS.get(order_id.strip().upper())
        if order is None or order['customer'] != ctx.deps.customer_id:
            return f'No order {order_id} for this customer.'
        return f'Order {order_id}: {order["item"]}, {order["status"]}, total ${order["total"]}.'

    def refund_order(ctx: RunContext[SupportDeps], order_id: str, reason: str) -> str:
        """Start a refund for one of the signed-in customer's orders.

        Args:
            ctx: The run context, carrying the customer this run serves.
            order_id: The order reference to refund.
            reason: Why the customer wants the refund, in their own words.
        """
        record(ToolCall('refund_order', ctx.tool_name, {'order_id': order_id, 'reason': reason}))
        order = ORDERS.get(order_id.strip().upper())
        if order is None or order['customer'] != ctx.deps.customer_id:
            return f'No order {order_id} for this customer.'
        # `capability:store_policy` puts the refund window after delivery, so the tool refuses an
        # order still in transit rather than contradicting the block it is told to follow.
        if order['status'] != 'delivered':
            return f'Order {order_id} is {order["status"]}, so it is not refundable yet.'
        return f'Refund opened for {order_id} (${order["total"]}), reason: {reason}.'

    return FunctionToolset[SupportDeps](
        [lookup_order, refund_order], id='orders', instructions=CODE_BLOCKS['toolset:orders']
    )


def _catalog_toolset(record: Callable[[ToolCall], None]) -> FunctionToolset[SupportDeps]:
    """Catalog tools, in a second toolset so a published override can be narrowed by `toolset`."""

    def search_catalog(ctx: RunContext[SupportDeps], query: str, limit: int = 3) -> str:
        """Search the storefront catalog by product name.

        Args:
            ctx: The run context, unused here but part of the tool signature.
            query: Words from the product name, e.g. 'desk lamp'.
            limit: How many matches to return at most.
        """
        record(ToolCall('search_catalog', ctx.tool_name, {'query': query, 'limit': limit}))
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
        record(ToolCall('lookup_stock', ctx.tool_name, {'sku': sku}))
        if sku.strip().upper() not in {item['sku'] for item in CATALOG}:
            return f'Unknown SKU {sku}.'
        return f'{sku}: 7 units on hand.'

    return FunctionToolset[SupportDeps](
        [search_catalog, lookup_stock], id='catalog', instructions=CODE_BLOCKS['toolset:catalog']
    )
