"""First-run information for interactive Pydantic AI users."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from itertools import zip_longest
from threading import Lock
from typing import get_args

from . import __version__

_banner_displayed = False
_banner_lock = Lock()
BANNER_ENABLED = True

_LOGO = r"""      /\
     /  \
    /    \
   /      \
  /________\
 /    ||    \
/     ||     \
\_____||_____/"""


def display_agent_banner(
    *,
    name: str | None,
    model: str,
    output_type: object,
    tools: int,
    capabilities: Sequence[str],
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

    info = f'agent: {name or "(unnamed)"} • model: {model}'
    if output_type is not str:
        if isinstance(output_type, type) and not get_args(output_type):
            output_name = output_type.__name__
        else:
            output_name = str(output_type)
        info += f' • output: {output_name}'
    info += f' • tools: {tools}'
    if capabilities:
        capability_names = ', '.join(capabilities[:4])
        remaining = len(capabilities) - 4
        if remaining > 0:
            capability_names += f' +{remaining} more'
        info += f' • capabilities: {capability_names}'

    harness_version = None
    if importlib.util.find_spec('pydantic_ai_harness') is not None:
        try:
            harness_version = metadata.version('pydantic-ai-harness')
        except metadata.PackageNotFoundError:
            pass

    version = f'pydantic-ai v{__version__}'
    if harness_version is not None:
        version += f' • pydantic-ai-harness v{harness_version}'
    version += f' • Python {platform.python_version()}'

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

    lines = (
        f'{version}\n'
        f'{info}\n'
        f'observability: not configured — {setup}. Free with a Logfire account —\n'
        '  sign up: `uvx logfire auth` or https://logfire.pydantic.dev\n'
        '  (or use any OpenTelemetry backend)\n'
        '  docs: https://pydantic.dev/docs/ai/logfire/ • hide this banner: PYDANTIC_AI_NO_BANNER=1'
    ).splitlines()
    logo_lines = _LOGO.splitlines()
    logo_width = max(map(len, logo_lines))
    banner = '\n'.join(
        f'{logo.ljust(logo_width)}  {line}' for logo, line in zip_longest(logo_lines, lines, fillvalue='')
    )
    print(banner, file=sys.stderr)
