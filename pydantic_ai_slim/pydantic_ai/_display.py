"""First-run information for interactive Pydantic AI users."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from threading import Lock
from typing import get_args

from . import __version__

_banner_displayed = False
_banner_lock = Lock()
BANNER_ENABLED = True


def display_agent_banner(
    *,
    name: str | None,
    model: str,
    output_type: object,
    tools: int,
    instrumented: bool,
) -> None:
    """Display information about the first uninstrumented agent run in an interactive process."""
    if (
        not BANNER_ENABLED
        or instrumented
        or 'PYDANTIC_AI_NO_BANNER' in os.environ
        or 'CI' in os.environ
        or not sys.stderr.isatty()
    ):
        return

    global _banner_displayed
    with _banner_lock:
        if _banner_displayed:
            return
        _banner_displayed = True

    if isinstance(output_type, type) and not get_args(output_type):
        output_name = output_type.__name__
    else:
        output_name = str(output_type)

    if 'logfire' not in sys.modules and importlib.util.find_spec('logfire') is None:
        setup = (
            'install `logfire` and set `instrument=True`.\n'
            '  Then run `logfire.configure()` to see this agent live:\n'
            '  Every model call, tool call, and cost'
        )
    else:
        setup = (
            'set `instrument=True` and run `logfire.configure()` to see\n'
            '  this agent live: every model call, tool call, and cost'
        )

    print(
        f'pydantic-ai v{__version__} • Python {platform.python_version()}\n'
        f'agent: {name or "(unnamed)"} • model: {model} • output: {output_name} • tools: {tools}\n'
        f'observability: not configured — {setup}. Free with a Logfire account —\n'
        '  sign up: https://logfire.pydantic.dev (or use any OpenTelemetry backend)\n'
        '  docs: https://pydantic.dev/docs/ai/logfire/ • hide this banner: PYDANTIC_AI_NO_BANNER=1',
        file=sys.stderr,
    )
