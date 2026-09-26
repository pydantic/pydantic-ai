"""The Speculative Execution switch, its session counters, and the pinned row they paint.

The switch is the `run.speculative_code_mode` setting, off by default. `Ctrl+X Ctrl+S` flips
it; the next turn honours the new value and a run already in flight keeps the tools it started
with. The wiring itself lives in `speculative_mode`, imported only while the switch is on so
Monty stays out of startup.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.tools import AgentDepsT
from rich.console import Console

from . import theme
from .command_context import CommandContext

SETTING = 'run.speculative_code_mode'
TOGGLE_KEYS = 'Ctrl+X Ctrl+S'


@dataclass(kw_only=True)
class SpeculationCounters:
    """Session totals behind the pinned row; toggling off hides the row but keeps them.

    - `hits`: sandbox calls that adopted a speculative launch.
    - `misses`: speculation-eligible calls that ran without a matching launch.
    - `wasted`: launches the harness discarded without a claim.
    - `speculative_ms`: summed durations of fully hidden claimed calls. Partial hits count as
      hits, but their timing is excluded because the harness does not report the wait.
    - `eager_ms`: summed sandbox call time that overlapped `run_code` argument streaming.

    Both timings are lower bounds on hidden call latency, not wall-clock speedup, since
    concurrent calls can overlap.
    """

    hits: int = 0
    misses: int = 0
    wasted: int = 0
    speculative_ms: float = 0.0
    eager_ms: float = 0.0

    def row(self) -> str:
        """One styled row; counts light up only when non-zero, the saved total is the headline.

        Only brand roles are used, so `/theme` recolours the row with the rest of the shell.
        """
        muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
        counts = f'{muted} \u00b7 {reset}'.join(
            _lit(bool(value), f'{value} {singular if value == 1 else plural}', role)
            for value, singular, plural, role in (
                (self.hits, 'hit', 'hits', theme.SUCCESS),
                (self.misses, 'miss', 'misses', theme.WARNING),
                (self.wasted, 'wasted', 'wasted', theme.ERROR),
            )
        )
        total = _seconds(self.speculative_ms + self.eager_ms)
        saved = _lit(total != '0.0', f'saved \u2265 {total}s', theme.SUCCESS)
        return f'{theme.sgr(theme.ACCENT)}Speculative Execution{reset}  {counts}    {saved}'


@dataclass(kw_only=True)
class Speculation:
    """The shell's handle on the switch: toggle it, paint its row, and bind it to a run."""

    context: CommandContext
    console: Console
    counters: SpeculationCounters = field(default_factory=SpeculationCounters)

    @property
    def enabled(self) -> bool:
        """Read the saved setting each time, so `/set` and the chord agree."""
        return self.context.settings.speculative_code_mode

    def toggle(self) -> str:
        """Flip and persist the switch; return the footer notice."""
        enabled = not self.enabled
        self.context.set_setting([SETTING, 'true' if enabled else 'false'])
        state = 'on' if enabled else 'off'
        return f'Speculative execution {state} from the next turn. {TOGGLE_KEYS} toggles it.'

    def row(self) -> str:
        """The pinned stats row while the switch is on; nothing while it is off."""
        return self.counters.row() if self.enabled else ''

    def capabilities(self, granted: Sequence[AgentCapability[AgentDepsT]]) -> 'list[AbstractCapability[AgentDepsT]]':
        """The speculative CodeMode bundle for one run, or nothing while the switch is off.

        `granted` is the run's other capabilities; the sandbox mounts only what their `FileSystem` allows.
        """
        if not self.enabled:
            return []
        try:
            from .speculative_mode import speculative_capabilities  # noqa: PLC0415 -- keep Monty out of startup.
        except ImportError as exc:
            self.console.print(
                f'Speculative execution is unavailable: {exc}', style=theme.color(theme.WARNING), markup=False
            )
            return []
        return speculative_capabilities(self.counters, granted)


def _lit(on: bool, text: str, role: str) -> str:
    return f'{theme.sgr(role, bold=True) if on else theme.sgr(theme.MUTED)}{text}\x1b[0m'


def _seconds(ms: float) -> str:
    """Round down so the displayed total stays a lower bound."""
    return f'{ms // 100 / 10:.1f}'
