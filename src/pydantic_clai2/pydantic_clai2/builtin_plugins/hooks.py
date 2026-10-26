"""Run commands from a `hooks` table before and after tool calls, and as a turn starts and ends.

The built-in `hooks` plugin runs your own table, saved as its settings. `project_hooks` runs the table a
project's `.clai/settings.json` declares, once `/plugins enable project_hooks` approves it.
"""

import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL
from typing import Literal

import anyio
from opentelemetry.trace import INVALID_SPAN, Span, StatusCode
from pydantic_core import to_json

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks, ValidatedToolArgs
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_clai2.config.hooks import HookCommand, HookEvent, HooksSettings
from pydantic_clai2.config.project_settings import find_project_file
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, TurnEnd, TurnStart
from pydantic_clai2.runtime import agent_instrumentation
from pydantic_clai2.ui.rendering import theme

BLOCK = 2
"""The exit code with which a `before_tool` command stops the call; the model gets its stderr as the result."""
RESULT_CHARS = 4000
"""How much of a tool's result an `after_tool` payload carries."""
_PLACEHOLDER = re.compile(r'\{(\w+)\}')

HookOutcome = Literal['ok', 'blocked', 'failed', 'timeout']


@dataclass(frozen=True, kw_only=True)
class HookRun:
    """What one command did: its outcome, and its stderr or the reason it did not run."""

    outcome: HookOutcome
    stderr: str = ''


def fill(part: str, args: Mapping[str, object]) -> str:
    """`part` with each `{name}` replaced by the tool's argument `name`; one that is absent or not a scalar stays."""

    def value(match: re.Match[str]) -> str:
        found = args.get(match[1])
        return str(found) if isinstance(found, str | int | float) else match[0]

    return _PLACEHOLDER.sub(value, part)


def summary(result: object) -> str:
    """A tool result as text, cut to `RESULT_CHARS`."""
    text = result if isinstance(result, str) else to_json(result, fallback=str).decode()
    return text if len(text) <= RESULT_CHARS else f'{text[:RESULT_CHARS]}...'


async def communicate(argv: str | list[str], data: bytes, *, cwd: Path) -> tuple[int, bytes]:
    """Run `argv` with `data` on stdin; its exit code and stderr. Stdout is discarded.

    A command may exit without reading its stdin, so a closed pipe is not an error. Cancelled, it is killed.
    """
    async with await anyio.open_process(argv, cwd=cwd, stdout=DEVNULL) as process:
        assert process.stdin is not None and process.stderr is not None
        stderr = process.stderr
        chunks: list[bytes] = []

        async def read() -> None:
            async for chunk in stderr:
                chunks.append(chunk)

        async with anyio.create_task_group() as group:
            group.start_soon(read)
            with suppress(anyio.BrokenResourceError):
                await process.stdin.send(data)
            await process.stdin.aclose()
        return await process.wait(), b''.join(chunks)


def _project_root(workspace: Path) -> Path | None:
    path = find_project_file(workspace)
    return None if path is None else path.parent.parent


class HooksPlugin(Plugin[HooksSettings, DepsT]):
    """Run commands you configure, or a project's `hooks` table, before and after tools and turns."""

    def __init__(self, host: PluginHost[DepsT], settings: HooksSettings) -> None:
        super().__init__(host, settings)
        self.root = None if settings.project is None else Path(settings.project)
        # The approval of a project's table is saved with your settings, which every repository shares.
        self.active = self.root is None or _project_root(host.workspace) == self.root

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        before, after = self.table('before_tool'), self.table('after_tool')
        if not (before or after):
            return ()
        hooks = Hooks[DepsT]()
        for command in before:
            hooks.on.before_tool_execute(tools=command.tools)(self._before(command))
        for command in after:
            hooks.on.after_tool_execute(tools=command.tools)(self._after(command))
        return (hooks,)

    def _before(self, command: HookCommand):
        async def before_tool(
            ctx: RunContext[DepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
        ) -> ValidatedToolArgs:
            payload = {'tool_name': call.tool_name, 'args': args}
            done = await self.run('before_tool', command, payload, args=args, tool=call.tool_name)
            if done.outcome == 'blocked':
                raise SkipToolExecution(done.stderr or f'A before_tool hook blocked {call.tool_name}.')
            return args

        return before_tool

    def _after(self, command: HookCommand):
        async def after_tool(
            ctx: RunContext[DepsT],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: ValidatedToolArgs,
            result: object,
        ) -> object:
            payload = {'tool_name': call.tool_name, 'args': args, 'result': summary(result)}
            await self.run('after_tool', command, payload, args=args, tool=call.tool_name)
            return result

        return after_tool

    def table(self, event: HookEvent) -> list[HookCommand]:
        """The commands for `event`, or none in a session outside the project whose table this is."""
        return self.settings.commands(event) if self.active else []

    async def on_turn_start(self, event: TurnStart) -> None:
        for command in self.table('turn_start'):
            await self.run('turn_start', command, {'prompt': event.text})

    async def on_turn_end(self, event: TurnEnd) -> None:
        for command in self.table('turn_end'):
            await self.run('turn_end', command, {'prompt': event.text, 'outcome': event.outcome})

    async def run(
        self,
        event: HookEvent,
        command: HookCommand,
        fields: Mapping[str, object],
        *,
        args: Mapping[str, object] | None = None,
        tool: str | None = None,
    ) -> HookRun:
        """Run `command` with the event's JSON payload on stdin; report anything but success, never raise."""
        workspace = self.host.workspace
        payload = {'event': event, 'session_id': self.host.session_id, 'cwd': str(workspace), **fields}
        argv = command.command if isinstance(command.command, str) else self._argv(command.command, args or {})
        shown = argv if isinstance(argv, str) else ' '.join(argv)
        tracer = agent_instrumentation.tracer()
        attributes = {'clai.hook.event': event, 'clai.hook.command': shown, 'clai.hook.tool': tool or ''}
        span_context = (
            tracer.start_as_current_span(f'clai hook {event}', attributes=attributes)
            if tracer is not None
            else nullcontext(INVALID_SPAN)
        )
        with span_context as span:
            done = await self._spawn(argv, payload, cwd=workspace, timeout=command.timeout, event=event)
            self._report(span, event, shown, done, tool=tool, timeout=command.timeout)
        return done

    def _argv(self, argv: list[str], args: Mapping[str, object]) -> list[str]:
        """Placeholders filled; a project's relative program path, such as `./scripts/x.sh`, taken from its root."""
        filled = [fill(part, args) for part in argv]
        program = Path(filled[0])
        if self.root is not None and len(program.parts) > 1 and not program.is_absolute():
            filled[0] = str(self.root / program)
        return filled

    @staticmethod
    async def _spawn(
        argv: str | list[str], payload: Mapping[str, object], *, cwd: Path, timeout: float, event: HookEvent
    ) -> HookRun:
        try:
            with anyio.fail_after(timeout):
                returncode, raw = await communicate(argv, to_json(payload, fallback=str), cwd=cwd)
        except TimeoutError:
            return HookRun(outcome='timeout')
        except OSError as exc:
            return HookRun(outcome='failed', stderr=str(exc))
        stderr = raw.decode(errors='replace').strip()
        if returncode == 0:
            return HookRun(outcome='ok', stderr=stderr)
        if returncode == BLOCK and event == 'before_tool':
            return HookRun(outcome='blocked', stderr=stderr)
        return HookRun(outcome='failed', stderr=stderr or f'exit code {returncode}')

    def _report(
        self, span: Span, event: HookEvent, shown: str, done: HookRun, *, tool: str | None, timeout: float
    ) -> None:
        """Record the outcome on the span, and print anything but success."""
        span.set_attribute('clai.hook.outcome', done.outcome)
        if done.outcome == 'ok':
            return
        if done.outcome == 'blocked':
            message = f'{event} hook blocked {tool}: {done.stderr or shown}'
        elif done.outcome == 'timeout':
            message = f'{event} hook timed out after {timeout:g}s: {shown}'
        else:
            message = f'{event} hook failed: {shown}: {done.stderr}'
        span.set_status(StatusCode.ERROR, message)
        self.host.console.print(message, style=theme.color(theme.WARNING), markup=False)
