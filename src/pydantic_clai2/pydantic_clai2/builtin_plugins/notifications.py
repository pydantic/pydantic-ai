"""Best-effort desktop notifications without conversation content."""

import os
import subprocess
import sys
from collections.abc import Sequence

import anyio

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, on_event
from pydantic_ai_harness.ask_user import AskUserRequestedEvent
from pydantic_clai2.plugins import NoSettings, Plugin, PluginHost, TurnEnd


async def notify(message: str) -> None:
    """Submit a local desktop notification; unavailable services are nonfatal."""
    if sys.platform == 'darwin':
        command = [
            '/usr/bin/osascript',
            '-e',
            'on run argv\n display notification (item 1 of argv) with title "CLAI2"\nend run',
            message,
        ]
    elif sys.platform.startswith('linux'):
        command = ['/usr/bin/notify-send', '--app-name=CLAI2', '--', 'CLAI2', message]
    else:
        return
    try:
        with anyio.fail_after(2):
            await anyio.run_process(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (OSError, TimeoutError):
        pass


class _QuestionNotice(AbstractCapability[None]):
    """Notify before the question picker waits for an answer."""

    @on_event(AskUserRequestedEvent)
    async def _asked(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        await notify('Your input is needed.')


class NotificationsPlugin(Plugin):
    """Notify on completed and failed turns, and before the question picker waits."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        # A remote process cannot address the local desktop notification service.
        self.local = host.console.is_terminal and not (os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (_QuestionNotice(),) if self.local else ()

    async def on_turn_end(self, event: TurnEnd) -> None:
        if not self.local:
            return
        if event.outcome == 'completed':
            await notify('Task finished.')
        elif event.outcome == 'failed':
            await notify('Task failed. Check the terminal for details.')
