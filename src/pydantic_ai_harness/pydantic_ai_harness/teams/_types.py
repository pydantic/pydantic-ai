"""The roster, message and snapshot types of a team."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai_harness.planning import OwnedPlanItem
from pydantic_ai_harness.subagents import DelegationTask

MemberState = Literal['working', 'idle', 'stopped', 'failed']
"""Where a team member is.

- `'working'`: a turn of it is running, or waiting for a turn under `DelegationTasks(max_active=...)`.
- `'idle'`: its last turn succeeded; a message wakes it.
- `'stopped'`: the user stopped it, or its turn was cancelled; a message from the user wakes it.
- `'failed'`: its last turn failed, for example on its usage limit; a new message wakes it.
"""


def member_state(record: DelegationTask) -> MemberState:
    """The state of the team member `record`."""
    if record.status == 'running':
        return 'working'
    if record.user_stopped or record.outcome == 'cancelled':
        return 'stopped'
    return 'idle' if record.outcome == 'ok' else 'failed'


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
