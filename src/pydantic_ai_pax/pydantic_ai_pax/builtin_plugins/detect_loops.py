"""Nudge the model to change its approach when it repeats the same tool calls or responses.

The built-in `detect_loops` plugin: harness `DetectLoops` with `action='nudge'`. It adds nothing to a
request until it finds a loop, and then appends a nudge to the conversation, so the turn goes on.
Within a response it checks only the thinking for a repeating passage, never the visible text or tool arguments.
"""

from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.detect_loops import DetectLoops
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin


class DetectLoopsPlugin(Plugin[NoSettings, DepsT]):
    """Nudge the model when it repeats the same tool calls or responses without making progress."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        # Only thinking is checked within a response. Visible text and tool call arguments can repeat on purpose (a line
        # the user asked for many times, a file of identical rows), and rejecting such a response retries it until the
        # turn fails on its output retries; a thinking loop is never what was asked for.
        return (DetectLoops[DepsT](action='nudge', stream_channels=('thinking',)),)
