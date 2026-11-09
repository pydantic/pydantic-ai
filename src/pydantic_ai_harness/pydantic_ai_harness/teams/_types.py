"""The roster, message and snapshot types of a team."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import OwnedPlanItem
from pydantic_ai_harness.subagents import DelegationTask, PlanRequest

MemberState = Literal['working', 'queued', 'idle', 'stopped', 'failed']
"""Where a team member is.

- `'working'`: a turn of it is running.
- `'queued'`: it was started or woken, and waits for a turn under `DelegationTasks(max_active=...)`. Queued
  members start in the order they were queued.
- `'idle'`: its last turn succeeded; a message wakes it.
- `'stopped'`: the user stopped it, it was shut down, or its turn was cancelled; a message from the user wakes it.
- `'failed'`: its last turn failed, for example on its usage limit; a new message wakes it.
"""

ShutdownState = Literal['requested', 'done']
"""Where a team member's shutdown is: `'requested'` while it wraps up, `'done'` once it is shut down."""


def member_state(record: DelegationTask) -> MemberState:
    """The state of the team member `record`."""
    if record.status == 'running':
        return 'queued' if record.queued else 'working'
    if record.shut_down or record.user_stopped or record.outcome == 'cancelled':
        return 'stopped'
    return 'idle' if record.outcome == 'ok' else 'failed'


def shutdown_state(record: DelegationTask) -> ShutdownState | None:
    """Where the shutdown of the team member `record` is, or `None` when none was asked for."""
    if record.shut_down:
        return 'done'
    return 'requested' if record.shutdown_request is not None else None


@dataclass(frozen=True, kw_only=True)
class TeamMember:
    """A team member, as the roster lists it."""

    name: str
    """The member's name, unique in its conversation."""

    task_id: str
    """The ID of the member's `DelegationTask`."""

    agent_name: str
    """The member's agent type: `self` for the lead's own agent, or one of the owner's `agents`."""

    model: str | None
    """The model ID the member is pinned to, or `None` for the lead's model (or its agent type's own model)."""

    state: MemberState
    """See `MemberState`."""

    current_task: str | None = None
    """The ID of the board item the member is working on, if it holds one `in_progress`."""

    shutdown: ShutdownState | None = None
    """Where its shutdown is, which tells a member shut down (`'done'`) from one the user stopped (`None`)."""

    plan: PlanRequest | None = None
    """Its plan, while it waits for approval."""

    usage: RunUsage = field(default_factory=RunUsage, compare=False)
    """The usage of all its turns so far: requests, tokens, and cost when its model's prices are known."""


@dataclass(frozen=True, kw_only=True)
class TeamMessage:
    """A message between a team's lead, its members and the user, as the message feed shows it."""

    id: str
    """The message's ID, which its frame in the recipient's history carries."""

    sender: str
    """A member's name, `'lead'`, or `'user'`."""

    recipient: str
    """A member's name, or `'lead'`."""

    text: str
    """What it says, cut at `MAX_EVENT_TEXT_CHARS`."""

    truncated: bool = False
    """Whether `text` was cut."""


@dataclass(frozen=True, kw_only=True)
class TeamSnapshot:
    """A team's state at one moment, for a UI to start from before it reduces the `teams` events."""

    conversation_id: str
    """The lead's conversation, which the team belongs to."""

    members: list[TeamMember]
    """The roster, oldest member first."""

    board: list[OwnedPlanItem]
    """Every board item, in board order."""

    messages: list[TeamMessage]
    """The latest messages, oldest first, up to `Team(max_messages=...)`, since the process started."""

    lead_usage: RunUsage = field(default_factory=RunUsage, compare=False)
    """The usage of the lead's runs while it led the team."""

    usage: RunUsage = field(default_factory=RunUsage, compare=False)
    """The whole team's usage: the lead's and every member's."""
