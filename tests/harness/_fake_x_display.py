"""A scripted X display for testing a sandbox's `desktop=True` against its fake SDK."""

from __future__ import annotations

import base64

from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from .display_fakes import FAKE_SCREENSHOT


class FakeDisplay:
    """Answers the start script and `xdotool` like a sandbox whose X server is up or down."""

    def __init__(self, *, up: bool = False) -> None:
        self.up = up
        self.starts: list[str] = []
        self.xdotool: list[str] = []
        self.viewers: list[str] = []
        self.viewer_fails = False

    @staticmethod
    def answers(command: str) -> bool:
        """Whether `command` is the display's, for a fake that answers other commands itself."""
        return any(part in command for part in ('xdotool ', 'Xvfb', 'xwininfo -root', 'scrot'))

    def __call__(self, command: str, timeout: float | None) -> tuple[str, str, int]:
        if 'Xvfb' in command:
            self.starts.append(command)
            self.up = True
            return '', '', 0
        elif 'xwininfo -root' in command:
            self.xdotool.append('xwininfo -root')
            # The backend reads the size only from a display it started.
            assert self.up, 'xwininfo before the display started'
            return '  Width: 1280\n  Height: 800\n', '', 0
        elif 'scrot' in command:
            return base64.b64encode(FAKE_SCREENSHOT).decode(), '', 0
        elif command.startswith('xdotool ') or ' xdotool ' in command:  # input runs `xdotool` through a script
            self.xdotool.append(command)
        elif 'novnc_proxy' in command:
            self.viewers.append(command)
            if self.viewer_fails or not self.up:
                return '', 'the viewer did not start:\nno display', 1
            return 'Pass1234', '', 0
        return '', '', 0


def look_click_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that takes a screenshot, clicks at (3, 4) on what it showed, then answers."""
    steps = [[{'type': 'screenshot'}], [{'type': 'click', 'x': 3, 'y': 4}]]
    done = sum(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts)
    if done == len(steps):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': steps[done]})])


def computer_returns(messages: list[ModelMessage]) -> list[ComputerReturnPart]:
    """The `computer` results in `messages`, in order."""
    return [part for message in messages for part in message.parts if isinstance(part, ComputerReturnPart)]
