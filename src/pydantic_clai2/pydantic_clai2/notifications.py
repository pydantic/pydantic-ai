"""Best-effort desktop notifications without conversation content."""

import os
import subprocess
import sys

import anyio
from pydantic_ai import RunContext
from pydantic_ai_harness.ask_user import AskUserRequestedEvent

from .plugins import PluginHost, TurnEnd


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


def activate(host: PluginHost[None]) -> None:
    """Notify on completed/failed turns and before the question picker waits."""
    # A remote process cannot address the local desktop notification service.
    if not host.console.is_terminal or os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'):
        return

    @host.on('turn_end')
    async def finished(event: TurnEnd) -> None:
        if event.outcome == 'completed':
            await notify('Task finished.')
        elif event.outcome == 'failed':
            await notify('Task failed. Check the terminal for details.')

    @host.on(AskUserRequestedEvent)
    async def question(ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        await notify('Your input is needed.')
