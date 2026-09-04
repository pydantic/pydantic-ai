"""First-run information for interactive Pydantic AI users."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
from collections.abc import Sequence
from importlib import metadata
from itertools import zip_longest
from textwrap import wrap
from threading import Lock
from typing import Protocol, cast, get_args

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
_LOGO_WIDTH = max(map(len, _LOGO_LINES))
_GUTTER = 2
_TEXT_WIDTH = 100 - _LOGO_WIDTH - _GUTTER
"""Wrap width for the text column, chosen so the whole banner fits in 100 columns."""
_MAX_OUTPUT_TYPE_LENGTH = 40
"""How much of the output type's name the banner shows before cutting it off."""

_INFO_SEPARATOR = ' • '
_INFO_INDENT = '  '

# Written as ANSI rather than with `rich`, which isn't a dependency of the library the banner ships
# in. `clai` reads the codes back into its own console, so both paths colour the banner identically.
_LOGO_COLOR = '\x1b[35m'
_HIGHLIGHT_COLOR = '\x1b[32m'
_COLOR_RESET = '\x1b[0m'
_COLOR_PATTERN = re.compile(r'\x1b\[\d+m')


class BannerDisplay(Protocol):
    """Displays the banner for a run, once it knows what the agent alone couldn't say."""

    def __call__(self, *, model: str, tools: int) -> None: ...


def banner_available(*, is_terminal: bool) -> bool:
    """Whether this process still owes the user a banner, claiming it for the caller if so.

    Every would-be display asks this, so that the claim is spent last — only once a banner is
    actually about to be shown, and never by a caller that was going to be turned away anyway.

    Args:
        is_terminal: Whether the caller's own destination is a terminal. `clai` asks its console,
            which knows about `FORCE_COLOR` and `TERM`; anything writing to `stderr` asks that.
    """
    return not _banner_suppressed() and is_terminal and claim_banner()


def _banner_suppressed() -> bool:
    """Whether the user or the environment has asked not to be shown the banner."""
    return not BANNER_ENABLED or 'PYDANTIC_AI_NO_BANNER' in os.environ or 'CI' in os.environ


def claim_banner() -> bool:
    """Claim the once-per-process banner, returning whether the caller is the one that gets to show it.

    Called directly only to spend the claim without showing anything, by a caller that knows a
    banner would land in the wrong place later.
    """
    global _banner_displayed
    with _banner_lock:
        if _banner_displayed:
            return False
        _banner_displayed = True
        return True


def render_banner(
    *,
    name: str | None,
    model: str,
    output_type: object,
    tools: int,
    capabilities: int,
    observability: bool = True,
    color: bool = True,
) -> str:
    """Render the banner: what's running, what the agent is, and how to see what it does.

    Args:
        name: Agent name, omitted from the banner when the agent doesn't have one.
        model: ID of the model the agent will use.
        output_type: The agent's output type, omitted from the banner when it's the default `str`.
        tools: Number of tools the agent can call.
        capabilities: Number of capabilities registered on the agent.
        observability: Whether to include the pointer to setting up observability.
        color: Whether to emit the ANSI colour codes that highlight the logo and the agent's identity.
    """
    # What identifies the agent is highlighted; what it was given to work with is counted plainly.
    info = [('agent', name, True)] if name else []
    info.append(('model', model, True))
    if output_type is not str:
        info.append(('output', _output_type_name(output_type), False))
    info.append(('tools', str(tools), False))
    info.append(('capabilities', str(capabilities), False))

    lines = [_version_line(), *_info_lines(info)]
    if observability:
        lines += _observability_lines()

    banner = _beside_logo(lines)
    return banner if color else _COLOR_PATTERN.sub('', banner)


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
    # An instrumented run has what the banner would point it to, so it stays out of the way entirely
    # rather than spending the claim — unlike `clai`, whose banner is also its session header.
    if instrumented or not banner_available(is_terminal=sys.stderr.isatty()):
        return

    banner = render_banner(
        name=name,
        model=model,
        output_type=output_type,
        tools=tools,
        capabilities=capabilities,
        # Nothing renders this one for us, so the convention has to be honored here.
        color='NO_COLOR' not in os.environ,
    )
    print(banner, file=sys.stderr)


def _info_lines(info: Sequence[tuple[str, str, bool]]) -> list[str]:
    """Lay `(label, value, highlight)` details out over as many lines as they need.

    Packed here rather than by `textwrap`, which would count the colour codes as width and break
    lines in the middle of a value.
    """
    lines: list[str] = []
    width = 0
    for label, value, highlight in info:
        item = f'{label}: {value}'
        styled = f'{label}: {_colored(value, _HIGHLIGHT_COLOR)}' if highlight else item
        if lines and width + len(_INFO_SEPARATOR) + len(item) <= _TEXT_WIDTH:
            lines[-1] += _INFO_SEPARATOR + styled
            width += len(_INFO_SEPARATOR) + len(item)
        else:
            indent = _INFO_INDENT if lines else ''
            lines.append(indent + styled)
            width = len(indent) + len(item)
    return lines


def _colored(text: str, color: str) -> str:
    return f'{color}{text}{_COLOR_RESET}'


def _output_type_name(output_type: object) -> str:
    """Name the agent's output type the way the user wrote it, as far as it fits.

    A union or a list of output functions can `repr` into something many times wider than the
    banner, so what doesn't fit is cut off rather than left to wrap into a wall of text.
    """
    if isinstance(output_type, (list, tuple)):
        name = ' | '.join(_output_type_name(member) for member in cast(Sequence[object], output_type))
    elif get_args(output_type):
        # `list[str]`, `Foo | None`: only `str()` renders these the way they were written.
        name = str(output_type)
    else:
        # A class or an output function names itself; a marker like `ToolOutput(...)` falls back to its `repr`.
        name = getattr(output_type, '__name__', None) or str(output_type)

    if len(name) > _MAX_OUTPUT_TYPE_LENGTH:
        name = name[: _MAX_OUTPUT_TYPE_LENGTH - 1] + '…'
    return name


def _version_line() -> str:
    """The versions in play: the library, the harness if it's installed, and Python."""
    # Imported here rather than at module scope so that this module, which the agent graph reaches
    # for mid-run, stays importable from anywhere in the package without an import cycle.
    from . import __version__

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
        setup = 'install `logfire`, pass `instrument=True` to `Agent()`, and run `logfire.configure()`'
    else:
        setup = 'pass `instrument=True` to `Agent()` and run `logfire.configure()`'

    return [
        'observability: off — to see every model and tool call live with its cost,',
        *wrap(
            f'{setup}.',
            width=_TEXT_WIDTH,
            initial_indent='  ',
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
        *wrap(
            'sign up for Pydantic Logfire for free via `uvx logfire auth` or https://logfire.pydantic.dev, or use any OpenTelemetry backend.',
            width=_TEXT_WIDTH,
            initial_indent='  ',
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
        '  docs: https://pydantic.dev/docs/ai/logfire/',
        'hide this banner: PYDANTIC_AI_NO_BANNER=1',
    ]


def _beside_logo(lines: Sequence[str]) -> str:
    """Lay `lines` out in a column to the right of the logo, vertically centered against it."""
    # Whichever column is shorter is padded on top, so neither is left dangling at the bottom.
    logo_padding = max(0, (len(lines) - len(_LOGO_LINES)) // 2)
    text_padding = max(0, (len(_LOGO_LINES) - len(lines)) // 2)
    return '\n'.join(
        # Padded to width before it's coloured, so the codes never count towards the column.
        f'{_colored(logo, _LOGO_COLOR) if logo else ""}{" " * (_LOGO_WIDTH - len(logo) + _GUTTER)}{line}'.rstrip()
        for logo, line in zip_longest(
            [*[''] * logo_padding, *_LOGO_LINES], [*[''] * text_padding, *lines], fillvalue=''
        )
    )
