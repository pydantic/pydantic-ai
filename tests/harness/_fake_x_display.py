"""A scripted X display for testing a sandbox's `desktop=True` against its fake SDK."""

from __future__ import annotations

import base64

from .display_fakes import FAKE_SCREENSHOT

STARTED = 'pydantic-ai-display-started'
"""What the harness's start script prints when it started a new X server."""


class FakeDisplay:
    """Answers the start script and `xdotool` like a sandbox whose X server is up or down."""

    def __init__(self, *, up: bool = False) -> None:
        self.up = up
        self.starts: list[str] = []
        self.xdotool: list[str] = []
        self.viewers: list[str] = []

    @staticmethod
    def answers(command: str) -> bool:
        """Whether `command` is the display's, for a fake that answers other commands itself."""
        return command.startswith('xdotool ') or any(part in command for part in ('Xvfb', 'xwininfo -root'))

    def __call__(self, command: str, timeout: float | None) -> tuple[str, str, int]:
        if 'Xvfb' in command:
            self.starts.append(command)
            if self.up:
                return '', '', 0
            self.up = True
            return STARTED, '', 0
        elif 'xwininfo -root' in command:
            self.xdotool.append('xwininfo -root')
            # The backend reads the size only from a display it started.
            assert self.up, 'xwininfo before the display started'
            return '  Width: 1280\n  Height: 800\n', '', 0
        elif 'scrot' in command:
            return base64.b64encode(FAKE_SCREENSHOT).decode(), '', 0
        elif command.startswith('xdotool ') or ' xdotool ' in command:  # input runs `xdotool` through a script
            self.xdotool.append(command)
            if not self.up:
                return '', "Error: Can't open display: :1", 1
        elif 'novnc_proxy' in command:
            self.viewers.append(command)
            if not self.up:
                return '', 'the viewer did not start:\nno display', 1
            return 'Pass1234', '', 0
        return '', '', 0
