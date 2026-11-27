"""Nudge the model to change its approach when it repeats the same tool calls or responses.

The built-in `detect_loops` plugin: harness `DetectLoops` with `action='nudge'`. It adds nothing to a
request until it finds a loop, and then appends a nudge to the conversation, so the turn goes on.
"""

from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.detect_loops import DetectLoops
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin


class DetectLoopsPlugin(Plugin[NoSettings, DepsT]):
    """Nudge the model when it repeats the same tool calls or responses without making progress."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        # Tool call arguments are left out of the in-response check: a file written on purpose with many identical
        # rows reads as a loop there, and rejecting the response would retry the write until the turn fails.
        return (DetectLoops[DepsT](action='nudge', stream_channels=('thinking', 'text')),)
