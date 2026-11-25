"""Comparing a durable run's events with those of its replay."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from pydantic_ai.messages import ModelResponseEndEvent

EventT = TypeVar('EventT')


def without_run_identity(events: list[EventT]) -> list[EventT]:
    """`events`, without the run and conversation IDs on the responses that `ModelResponseEndEvent`s carry.

    A replay that re-enters a task is a new agent run with IDs of its own, which it stamps on each response it ends,
    while every other event it replays is identical.
    """
    return [
        replace(event, response=replace(event.response, run_id=None, conversation_id=None))
        if isinstance(event, ModelResponseEndEvent) and event.response is not None
        else event
        for event in events
    ]
