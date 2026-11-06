"""Agent teams: a lead and named members working one shared board, checked by the same gates."""

from pydantic_ai_harness.teams._capability import Team
from pydantic_ai_harness.teams._events import (
    TEAMS_EVENTS,
    BoardItemChangedEvent,
    TeammateIdleEvent,
    TeamTaskCompletingEvent,
    TeamTaskCreatingEvent,
)
from pydantic_ai_harness.teams._gates import check_task_completing, check_task_creating

__all__ = [
    'TEAMS_EVENTS',
    'BoardItemChangedEvent',
    'Team',
    'TeamTaskCompletingEvent',
    'TeamTaskCreatingEvent',
    'TeammateIdleEvent',
    'check_task_completing',
    'check_task_creating',
]
