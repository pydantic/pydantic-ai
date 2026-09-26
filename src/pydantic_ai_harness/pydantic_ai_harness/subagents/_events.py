"""Events emitted by the sub-agents capability.

One delegation is one `delegate_task` call, so its start and end events share
the `tool_call_id` core stamps on every event a capability tool emits; that is
how a subscriber pairs them when several delegations run at once.

Text in these events is bounded: `task` and `output` are cut at
`MAX_EVENT_TEXT_CHARS` with a `truncated` flag, so an event stream that is
persisted or forwarded to a UI cannot be flooded by one verbose delegation.

As with any capability event, a listener that raises aborts the parent run.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import CapabilityEvent
from pydantic_ai.usage import RunUsage

SUB_AGENTS_EVENTS = 'sub_agents'

MAX_EVENT_TEXT_CHARS = 4096
"""Characters kept of `DelegationStartEvent.task` and `DelegationEndEvent.output` before they are cut."""

DelegationOutcome = Literal['ok', 'timeout', 'budget', 'failed', 'contained']
"""How a delegation ended, mirroring the failure-handling contract of `SubAgent`.

- `ok`: the child finished and its output went back to the parent.
- `timeout`: the child exceeded `SubAgent.timeout_seconds`; the parent got a steering message.
- `budget`: the child reached its own `SubAgent.usage_limits`; the parent got a steering message.
- `failed`: the child raised a soft model error; the parent got `SubAgent.on_failure` or a `ModelRetry`.
- `contained`: the child crashed with `contain_errors` on; the parent got a `ModelRetry`.
"""


def bounded_text(text: str) -> tuple[str, bool]:
    """`text` cut at `MAX_EVENT_TEXT_CHARS`, and whether it was cut."""
    if len(text) <= MAX_EVENT_TEXT_CHARS:
        return text, False
    return text[:MAX_EVENT_TEXT_CHARS], True


@dataclass(kw_only=True)
class DelegationStartEvent(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='delegation_start'):
    """A sub-agent run is about to start for one delegation.

    Emitted once the delegation has passed every check that could refuse it (an
    unknown sub-agent, a model key off the menu, an exhausted `max_calls`
    budget), immediately before the child run starts.
    """

    agent_name: str
    task: str
    truncated: bool
    model: str | None
    """The menu key the delegation runs on, or `None` when no option was selected: there is
    no menu, or the delegate allows the whole menu and the parent named no key."""
    inherits_tools: bool
    """Whether the parent's own tools were passed to the child run (`SubAgents.inherit_tools`)."""


@dataclass(kw_only=True)
class DelegationEndEvent(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='delegation_end'):
    """A delegation settled into what the parent receives.

    `output` is what the delegate tool hands back to the parent: the child's
    output on `ok`, otherwise the steering message it returns or the
    `ModelRetry` it raises. An exception that propagates out of the delegate
    tool (a shared usage limit, an uncontained crash, a cancellation) ends
    without this event.
    """

    agent_name: str
    outcome: DelegationOutcome
    output: str
    truncated: bool
    usage: RunUsage | None
    """The child's own usage when it has separate accounting (`SubAgent.usage_limits`
    set, or `forward_usage` off); `None` when it accrues into the parent's usage."""
    duration_seconds: float
    """Wall-clock seconds from just after the start event was emitted until the delegation settled."""
