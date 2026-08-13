"""First-run information for interactive Pydantic AI users."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from threading import Lock

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

    if isinstance(output_type, type):
        output_name = output_type.__name__
    else:
        output_name = str(output_type)

    if 'logfire' not in sys.modules and importlib.util.find_spec('logfire') is None:
        setup = 'install `logfire`, set `instrument=True`, then run `logfire.configure()`'
    else:
        setup = 'set `instrument=True` and run `logfire.configure()`'

    print(
        f'pydantic-ai v{__version__} • Python {platform.python_version()}\n'
        f'agent: {name or "(unnamed)"} • model: {model} • output: {output_name} • tools: {tools}\n'
        f'observability: not configured — {setup} to see every\n'
        '  model call and tool call (OpenTelemetry: works with Logfire or any OTel backend)\n'
        '  docs: https://pydantic.dev/docs/ai/logfire/ • hide this banner: PYDANTIC_AI_NO_BANNER=1',
        file=sys.stderr,
    )
