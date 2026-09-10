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
     /   \\
   /___.___\\
  /    |    \\
/      |      \\
`--.___|___.--'"""
"""The Pydantic kite: a Penrose kite deflated into two kites and a dart.

The `.` on the crossbar is the dart's notch, which juts up between the two upper panels. The closing
row falls from the shoulders to a centre vertex by glyph height rather than position: a backtick and
an apostrophe sit at the top of a character cell, `-` in the middle, `.` low, and `_` on the floor.
"""

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

_CODING_AGENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Each signal is a variable name, a `NAME*` prefix, or a `NAME=value` match; an agent matches on
    # any one of its own. Ordered, first match wins, most specific first. Lifted from the two
    # projects that maintain this table against the agents' own source: Dart's
    # `package:unified_analytics` and `am-i-vibing`, which Astro uses to answer this same question.
    ('claude-code', ('CLAUDECODE', 'CLAUDE_CODE', 'CLAUDE_CODE_IS_COWORK')),
    ('codex', ('CODEX_THREAD_ID', 'CODEX_CI', 'CODEX_SANDBOX')),
    ('gemini-cli', ('GEMINI_CLI', 'GEMINI_AGENT')),
    ('cursor', ('CURSOR_AGENT',)),
    (
        'opencode',
        ('OPENCODE', 'OPENCODE_BIN_PATH', 'OPENCODE_SERVER', 'OPENCODE_APP_INFO', 'OPENCODE_MODES', 'OPENCODE_CLIENT'),
    ),
    ('pi', ('PI_CODING_AGENT',)),
    ('amp', ('AMP_CURRENT_THREAD_ID',)),
    ('augment', ('AUGMENT_AGENT',)),
    ('antigravity', ('ANTIGRAVITY_AGENT', 'ANTIGRAVITY_PROJECT_ID')),
    ('crush', ('CRUSH',)),
    ('qwen-code', ('QWEN_CODE',)),
    ('windsurf', ('CODEIUM_EDITOR_APP_ROOT',)),
    ('warp', ('OZ_RUN_ID',)),
    ('copilot', ('COPILOT_*', 'GITHUB_COPILOT*')),
    ('aider', ('AIDER_*',)),
    # `REPL_ID` on its own is the Replit IDE, which is a person at a terminal.
    ('replit', ('REPLIT_MODE=assistant',)),
    ('swe-agent', ('SWE_AGENT',)),
)
"""Environment signals that mean a coding agent is running this process and reads what it writes.

Only variables an agent names itself in, so that this can't mistake a person for one: the heuristics
these were taken from also match on `TERM_PROGRAM`, `PAGER` and `SHELL`, which would count everyone
typing in a given terminal. Agents identified only by their parent process are left out for the same
reason a miss is cheap — see `detect_coding_agent`.
"""

_NAMED_AGENT_ENV_VARS = ('AI_AGENT', 'AGENT')
"""What an agent this table doesn't know can set to announce itself, the value naming it."""

_UNNAMED_AGENT_VALUES = frozenset({'1', 'true', 'yes'})
"""Values of those that say an agent is present without saying which."""


class BannerDisplay(Protocol):
    """Displays the banner for a run, once it knows what the agent alone couldn't say."""

    def __call__(self, *, model: str, tools: int) -> None: ...


def banner_pending() -> bool:
    """Whether a banner could still be shown in this process.

    Asked before gathering what a banner would say, on a path every agent run takes, so it stays
    down to reading a flag and the environment. Once a banner has been shown — or turned away for
    a reason a process doesn't change its mind about — the claim is spent and this stays False.
    """
    return not _banner_displayed and not _banner_suppressed()


def banner_available(*, is_terminal: bool) -> bool:
    """Whether this process still owes the user a banner, claiming it for the caller if so.

    Every would-be display asks this, so that the claim is spent last — only once a banner is
    actually about to be shown, and never by a caller that was going to be turned away anyway.

    Args:
        is_terminal: Whether the caller's own destination is a terminal. `clai` asks its console,
            which knows about `FORCE_COLOR` and `TERM`; anything writing to `stderr` asks that.
    """
    return not _banner_suppressed() and (is_terminal or detect_coding_agent() is not None) and claim_banner()


def detect_coding_agent() -> str | None:
    """Name the coding agent running this process, or `None` when nothing says one is.

    An agent's `stderr` is a pipe it reads back rather than a terminal, so without this the banner
    would reach nobody working through one — which is now how much of the writing of Pydantic AI
    code happens. It's shown to them exactly as it is to a human, and says the same thing, so what
    the user sees when they read along is what they'd have seen themselves.

    The table is best-effort and will always be behind, which is why it errs towards missing rather
    than guessing: an agent it doesn't know is left where every agent was before, with no banner,
    and `AI_AGENT` is there for anything that wants to say so itself. The name isn't used yet.
    """
    for name, signals in _CODING_AGENTS:
        if any(_agent_signal_matches(signal) for signal in signals):
            return name

    for var in _NAMED_AGENT_ENV_VARS:
        if value := os.environ.get(var):
            return 'agent' if value.lower() in _UNNAMED_AGENT_VALUES else value

    return None


def _agent_signal_matches(signal: str) -> bool:
    """Whether the environment carries `signal`: a variable name, a `NAME*` prefix, or `NAME=value`."""
    if signal.endswith('*'):
        prefix = signal[:-1]
        return any(name.startswith(prefix) for name in os.environ)
    name, _, value = signal.partition('=')
    return os.environ.get(name) == value if value else name in os.environ


def _banner_suppressed() -> bool:
    """Whether the user or the environment has asked not to be shown the banner."""
    return (
        not BANNER_ENABLED
        or 'PYDANTIC_AI_NO_BANNER' in os.environ
        or 'CI' in os.environ
        # A test run is nobody's first run, and its output is captured and read back only when
        # something fails. Without this, an agent running a suite would attach a banner to whichever
        # test happened to go first.
        or 'PYTEST_VERSION' in os.environ
    )


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

    # The versions carry no colour, so `textwrap` can be trusted to break them at a separator.
    lines = [*wrap(_version_line(), width=_TEXT_WIDTH, subsequent_indent=_INFO_INDENT), '', *_info_lines(info)]
    if observability:
        # Both halves of this are advice for someone who hasn't set observability up, so a session
        # that has stays out of it entirely rather than being told to do what it has already done.
        lines += ['', *_observability_lines()]
        # Only what the block above doesn't already say: it opens by telling them to set it up.
        lines += ['', 'hide: PYDANTIC_AI_NO_BANNER=1']

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
    # An instrumented run has what the banner would point it to, so it stays out of the way — and
    # leaves the claim alone, since another agent in this process may not be instrumented. `clai`
    # differs: its banner is also its session header, so it shows one either way.
    if instrumented:
        return

    stderr = sys.stderr
    is_terminal = _stderr_is_terminal()
    # Someone being there to read is not the same question as there being somewhere to write:
    # `pythonw` and some frozen interpreters have no `stderr` at all, and `print(file=None)` would
    # divert the banner to `stdout`, into whatever the program is actually writing there. Asking
    # whether `stderr` is a terminal used to rule that out on its own; an agent reading a pipe
    # doesn't have to be at one, so the destination is now checked in its own right.
    #
    # Whatever turns the banner away — a suppressing environment, no reader, nowhere to write — is
    # not something a process changes its mind about, so spend the claim. Without this, every run
    # in a non-interactive process would gather a banner's details all over again only to throw
    # them away here.
    if stderr is None or not banner_available(is_terminal=is_terminal):
        claim_banner()
        return

    try:
        banner = render_banner(
            name=name,
            model=model,
            output_type=output_type,
            tools=tools,
            capabilities=capabilities,
            # Nothing renders this one for us, so the conventions have to be honored here: colour
            # belongs to a terminal, and an agent reading `stderr` back would get the codes raw.
            color=is_terminal and 'NO_COLOR' not in os.environ,
        )
        # Written to the stream that was checked, rather than to whatever `sys.stderr` is by now.
        print(banner, file=stderr)
    except Exception:
        # A banner is a courtesy, and a courtesy that fails is not worth an agent run. A terminal
        # whose encoding can't take the logo (`LC_ALL=C`) raises here, as does a `stderr` that has
        # been closed or wrapped in something unusual.
        pass


def _stderr_is_terminal() -> bool:
    """Whether `stderr` is a terminal, asked defensively because it isn't always even there.

    `sys.stderr` is `None` under `pythonw` and in some frozen and embedded interpreters, where
    `print(file=None)` would quietly redirect the banner to `stdout`, and it can be closed or
    replaced by the time an agent runs.
    """
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except Exception:
        return False


def _info_lines(info: Sequence[tuple[str, str, bool]]) -> list[str]:
    """Lay `(label, value, highlight)` details out over as many lines as they need.

    Packed here rather than by `textwrap`, which would count the colour codes as width and break
    lines in the middle of a value.
    """
    lines: list[str] = []
    width = 0
    for label, value, highlight in info:
        value = _elided(value, _TEXT_WIDTH - len(_INFO_INDENT) - len(label) - len(': '))
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


def _elided(value: str, width: int) -> str:
    """`value` brought down to `width` by dropping its middle, which is the part that identifies least.

    A model ID too wide for the banner is usually an ARN or a deployment path, whose ends say the
    provider and the model and whose middle is an account and a region. Cutting the tail off instead
    would leave the banner naming a model without the model in it.
    """
    if len(value) <= width:
        return value
    head = (width - 1) // 2
    return value[:head] + '…' + value[len(value) - (width - 1 - head) :]


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
    try:
        if importlib.util.find_spec('pydantic_ai_harness') is not None:
            harness_version = metadata.version('pydantic-ai-harness')
    except Exception:
        # Best-effort enrichment: the harness is named only when it can be found and named without
        # trouble. `find_spec` raises for a module whose `__spec__` is None and for anything a
        # custom importer objects to, and the distribution can be missing or unreadable.
        pass

    version = f'pydantic-ai v{__version__}'
    if harness_version is not None:
        version += f' • pydantic-ai-harness v{harness_version}'
    return version + f' • Python {platform.python_version()}'


def _observability_lines() -> list[str]:
    """How to see what the agent actually did, for someone who hasn't set that up yet."""
    return [
        *wrap(
            'observability: off — see every model and tool call live, with cost',
            width=_TEXT_WIDTH,
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
        *wrap(
            'ask your agent to set it up: https://pydantic.dev/ai-setup.md',
            width=_TEXT_WIDTH,
            initial_indent='  ',
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
        *wrap(
            'free with Logfire and a GitHub login, or use any OpenTelemetry backend',
            width=_TEXT_WIDTH,
            initial_indent='  ',
            subsequent_indent='  ',
            break_on_hyphens=False,
        ),
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
