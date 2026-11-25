"""Comparing a durable run's events with those of its replay."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from pydantic_ai.messages import ModelResponseEndEvent

EventT = TypeVar('EventT')


def without_run_identity(events: list[EventT]) -> list[EventT]:
    """`events`, without the run and conversation IDs on the responses that `ModelResponseEndEvent`s carry.

    A workaround for https://github.com/pydantic/pydantic-ai/issues/14619: Absurd and pgtask mint a fresh run ID when
    a task is re-entered, where the replay should keep the crashed execution's. The replay then stamps that new ID on
    each response it ends, while every other event it replays is identical. Remove this once #14619 is fixed.
    """
    return [
        replace(event, response=replace(event.response, run_id=None, conversation_id=None))
        if isinstance(event, ModelResponseEndEvent) and event.response is not None
        else event
        for event in events
    ]
