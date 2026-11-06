"""Agent teams: a lead and named members working one shared board, checked by the same gates."""

from pydantic_ai_harness.teams._capability import USER, BoardResolver, Team, TeamEvent, TeamObserver
from pydantic_ai_harness.teams._events import (
    TEAMS_EVENTS,
    BoardItemChangedEvent,
    TeammateIdleEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMessageEvent,
    TeamTaskCompletingEvent,
    TeamTaskCreatingEvent,
)
from pydantic_ai_harness.teams._gates import check_task_completing, check_task_creating
from pydantic_ai_harness.teams._types import MemberState, TeamMember, TeamMessage, TeamSnapshot

__all__ = [
    'TEAMS_EVENTS',
    'USER',
    'BoardItemChangedEvent',
    'BoardResolver',
    'MemberState',
    'Team',
    'TeamEvent',
    'TeamMember',
    'TeamMessage',
    'TeamMessageEvent',
    'TeamObserver',
    'TeamSnapshot',
    'TeamTaskCompletingEvent',
    'TeamTaskCreatingEvent',
    'TeammateIdleEvent',
    'TeammateSpawnedEvent',
    'TeammateStateEvent',
    'check_task_completing',
    'check_task_creating',
]
