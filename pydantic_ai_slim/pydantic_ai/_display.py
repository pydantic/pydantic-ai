"""First-run information for interactive Pydantic AI users."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from itertools import zip_longest
from textwrap import wrap
from threading import Lock
from typing import get_args

from . import __version__

_banner_displayed = False
_banner_lock = Lock()
BANNER_ENABLED = True

_LOGO = """\
         / \\
       /     \\
     /____.____\\
   /      |      \\
 /        |        \\
  ·.______|______.·"""

_LOGO_LINES = _LOGO.splitlines()
LOGO_WIDTH = max(map(len, _LOGO_LINES))
_GUTTER = 2
_TEXT_WIDTH = 100 - LOGO_WIDTH - _GUTTER
"""Wrap width for the text column, chosen so the whole banner fits in 100 columns."""


def banner_suppressed() -> bool:
    """Whether the user or the environment has asked not to be shown the banner."""
    return not BANNER_ENABLED or 'PYDANTIC_AI_NO_BANNER' in os.environ or 'CI' in os.environ


def claim_banner() -> bool:
    """Claim the once-per-process banner, returning whether the caller is the one that gets to show it."""
    global _banner_displayed
    with _banner_lock:
        if _banner_displayed:
            return False
        _banner_displayed = True
        return True


def render_banner(
    *,
    heading: str | None = None,
    name: str | None,
    model: str,
    output_type: object,
    tools: int,
    capabilities: int,
    observability: bool = True,
) -> str:
    """Render the banner: what's running, what the agent is, and how to see what it does.

    Args:
        heading: First line, naming the program the user launched. Defaults to the library version.
        name: Agent name, omitted from the banner when the agent doesn't have one.
        model: ID of the model the agent will use.
        output_type: The agent's output type, omitted from the banner when it's the default `str`.
        tools: Number of tools registered on the agent.
        capabilities: Number of capabilities registered on the agent.
        observability: Whether to include the pointer to setting up observability.
    """
    info = f'agent: {name} • ' if name else ''
    info += f'model: {model}'
    if output_type is not str:
        if isinstance(output_type, type) and not get_args(output_type):
            output_name = output_type.__name__
        else:
            output_name = str(output_type)
        info += f' • output: {output_name}'
    info += f' • tools: {tools}'
    if capabilities:
        info += f' • capabilities: {capabilities}'

    lines = [heading or _version_line(), info]
    if observability:
        lines += _observability_lines()
    return _beside_logo(lines)


def display_agent_banner(
    *,
    name: str | None,
    model: str,
    output_type: object,
    tools: int,
    capabilities: int,
    instrumented: bool,
) -> None:
    """Display information about the first uninstrumented agent run in an interactive process."""
    if banner_suppressed() or instrumented or not sys.stderr.isatty() or not claim_banner():
        return

    banner = render_banner(name=name, model=model, output_type=output_type, tools=tools, capabilities=capabilities)
    print(banner, file=sys.stderr)


def _version_line() -> str:
    """The versions in play: the library, the harness if it's installed, and Python."""
    harness_version = None
    if importlib.util.find_spec('pydantic_ai_harness') is not None:
        try:
            harness_version = metadata.version('pydantic-ai-harness')
        except metadata.PackageNotFoundError:
            pass

    version = f'pydantic-ai v{__version__}'
    if harness_version is not None:
        version += f' • pydantic-ai-harness v{harness_version}'
    return version + f' • Python {platform.python_version()}'


def _observability_lines() -> list[str]:
    """How to see what the agent actually did, for someone who hasn't set that up yet."""
    if 'logfire' not in sys.modules and importlib.util.find_spec('logfire') is None:
        setup = 'install `logfire`, set `instrument=True`, and run `logfire.configure()`'
    else:
        setup = 'set `instrument=True` and run `logfire.configure()`'

    return [
        *wrap(
            f'observability: not configured — {setup} to see this agent live: every model call, '
            'tool call, and cost. Free with a Logfire account — sign up: `uvx logfire auth` or '
            'https://logfire.pydantic.dev (or use any OpenTelemetry backend)',
            width=_TEXT_WIDTH,
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
        '  docs: https://pydantic.dev/docs/ai/logfire/',
        '  hide this banner: PYDANTIC_AI_NO_BANNER=1',
    ]


def _beside_logo(lines: Sequence[str]) -> str:
    """Lay `lines` out in a column to the right of the logo, vertically centered against it."""
    # Whichever column is shorter is padded on top, so neither is left dangling at the bottom.
    logo_padding = max(0, (len(lines) - len(_LOGO_LINES)) // 2)
    text_padding = max(0, (len(_LOGO_LINES) - len(lines)) // 2)
    return '\n'.join(
        f'{logo.ljust(LOGO_WIDTH)}{" " * _GUTTER}{line}'.rstrip()
        for logo, line in zip_longest(
            [*[''] * logo_padding, *_LOGO_LINES], [*[''] * text_padding, *lines], fillvalue=''
        )
    )
