"""The team a session leads as browsers see it: its roster, its board, and its messages, as each changes."""

from collections.abc import Callable, Sequence

from pydantic_ai_harness.planning import OwnedPlanItem, TaskStatus
from pydantic_ai_harness.teams import TeamMember, TeamMessage, TeamMessageEvent, TeamSnapshot
from pydantic_clai2.builtin_plugins.web.protocol import (
    BoardItemChanged,
    BoardItemEntry,
    TeamChanged,
    TeamDelta,
    TeamMemberChanged,
    TeamMemberEntry,
    TeamMessageAdded,
    TeamMessageEntry,
    TeamState,
)
from pydantic_clai2.plugins import TeamProgress
from pydantic_clai2.runtime.team import display_name

_SETTLED = (TaskStatus.completed, TaskStatus.cancelled)


def member_entry(member: TeamMember) -> TeamMemberEntry:
    """A teammate's row, as browsers receive it."""
    return TeamMemberEntry(
        name=member.name,
        task_id=member.task_id,
        agent=display_name(member.agent_name),
        model=member.model,
        state=member.state,
        current_task=member.current_task,
    )


def board_entries(board: Sequence[OwnedPlanItem]) -> list[BoardItemEntry]:
    """The board's items, each with the items it still waits for."""
    waiting = {item.id for item in board if item.status not in _SETTLED}
    return [
        BoardItemEntry(
            id=item.id,
            content=item.content,
            status=item.status.value,
            owner=item.owner,
            depends_on=list(item.depends_on),
            blocked_by=[dependency for dependency in item.depends_on if dependency in waiting],
        )
        for item in board
    ]


def message_entry(message: TeamMessage) -> TeamMessageEntry:
    """A message of the team's feed, as browsers receive it."""
    return TeamMessageEntry(
        id=message.id,
        sender=message.sender,
        recipient=message.recipient,
        text=message.text,
        truncated=message.truncated,
    )


def team_state(snapshot: TeamSnapshot) -> TeamState:
    """The whole team, as browsers receive it."""
    return TeamState(
        members=[member_entry(member) for member in snapshot.members],
        board=board_entries(snapshot.board),
        messages=[message_entry(message) for message in snapshot.messages],
    )


class TeamFeed:
    """A session's team changes, from `on_team_progress`, as the messages that change a browser's Team pane.

    Each change comes with the team as it is after it, so this sends each row that differs from what it
    sent before. A change to one board item can unblock others, which are sent too.
    """

    def __init__(self, *, publish: Callable[[TeamDelta], None]) -> None:
        self._publish = publish
        self._members: dict[str, TeamMemberEntry] | None = None
        """What browsers were sent, by name; `None` before the first change, which sends the whole team."""
        self._board: dict[str, BoardItemEntry] = {}

    def observe(self, event: TeamProgress) -> None:
        state = team_state(event.team)
        members = {member.name: member for member in state.members}
        board = {item.id: item for item in state.board}
        sent = self._members
        self._members, previous = members, self._board
        self._board = board
        if sent is None or sent.keys() - members.keys() or previous.keys() - board.keys():
            # Nothing was sent since the session took this conversation, or a row left: the whole team.
            self._publish(TeamChanged(team=state))
            return
        for name, member in members.items():
            if sent.get(name) != member:
                self._publish(TeamMemberChanged(member=member))
        for item_id, item in board.items():
            if (before := previous.get(item_id)) != item:
                self._publish(BoardItemChanged(item=item, previous=before))
        if isinstance(event.event, TeamMessageEvent):
            self._publish(TeamMessageAdded(message=message_entry(event.event.message)))

    def reset(self) -> None:
        """Forget the team of a conversation the session no longer holds."""
        self._members = None
        self._board = {}
