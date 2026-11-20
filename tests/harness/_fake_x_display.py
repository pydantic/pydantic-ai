"""A scripted X display for testing a sandbox's `desktop=True` against its fake SDK."""

from __future__ import annotations

import base64

from .display_fakes import FAKE_SCREENSHOT


class FakeDisplay:
    """Answers the start script and `xdotool` like a sandbox whose X server is up or down."""

    def __init__(self) -> None:
        self.up = False
        self.starts: list[str] = []
        self.xdotool: list[str] = []

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
            # The backend reads the size only from a display it started.
            assert self.up, 'xwininfo before the display started'
            return '  Width: 1280\n  Height: 800\n', '', 0
        elif 'scrot' in command:
            return base64.b64encode(FAKE_SCREENSHOT).decode(), '', 0
        elif command.startswith('xdotool ') or ' xdotool ' in command:  # input runs `xdotool` through a script
            self.xdotool.append(command)
        return '', '', 0
