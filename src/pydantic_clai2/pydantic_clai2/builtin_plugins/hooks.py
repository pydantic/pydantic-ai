"""Run commands from a `hooks` table before and after tool calls, and as a turn starts and ends.

The built-in `hooks` plugin runs your own table, saved as its settings. `project_hooks` runs the table a
project's `.clai/settings.json` declares, once `/plugins enable project_hooks` approves it.
"""

import asyncio
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL
from typing import IO, Literal

import anyio
from opentelemetry.trace import INVALID_SPAN, Span, StatusCode
from pydantic_core import to_json

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks, ValidatedToolArgs
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_clai2.config.hooks import PROJECT_HOOKS, HookCommand, HookEvent, HooksSettings
from pydantic_clai2.config.project_settings import load_project_settings
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, TurnEnd, TurnStart
from pydantic_clai2.runtime import agent_instrumentation
from pydantic_clai2.runtime._processes import kill_process_tree
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


async def communicate(command: str | list[str], data: bytes, *, cwd: Path) -> tuple[int, bytes]:
    """Run `command` with `data` on stdin in a process group of its own; its exit code and stderr.

    Stdout is discarded. Stdin and stderr are files, not pipes, so a child the command leaves running in
    the background cannot keep it waiting. Cancelled, the whole process group is killed.
    """
    stdin, stderr = await anyio.to_thread.run_sync(_files, data)
    try:
        return await _run(command, cwd=cwd, stdin=stdin, stderr=stderr)
    finally:
        stdin.close()
        stderr.close()


def _files(data: bytes) -> tuple[IO[bytes], IO[bytes]]:
    stdin = tempfile.TemporaryFile()
    stdin.write(data)
    stdin.seek(0)
    return stdin, tempfile.TemporaryFile()


def _read(file: IO[bytes]) -> bytes:
    file.seek(0)
    return file.read()


async def _run(command: str | list[str], *, cwd: Path, stdin: IO[bytes], stderr: IO[bytes]) -> tuple[int, bytes]:
    if isinstance(command, str):
        spawning = asyncio.create_subprocess_shell(
            command, cwd=cwd, start_new_session=True, stdin=stdin, stdout=DEVNULL, stderr=stderr
        )
    else:
        spawning = asyncio.create_subprocess_exec(
            *command, cwd=cwd, start_new_session=True, stdin=stdin, stdout=DEVNULL, stderr=stderr
        )
    spawn = asyncio.ensure_future(spawning)
    try:
        process = await asyncio.shield(spawn)
        returncode = await process.wait()
    except BaseException:
        with anyio.CancelScope(shield=True):
            process = await spawn
            await kill_process_tree(process)
            await process.wait()
        raise
    return returncode, await anyio.to_thread.run_sync(_read, stderr)


def _approved(workspace: Path, settings: HooksSettings) -> bool:
    """Whether the project `workspace` is in still declares exactly the approved table `settings`."""
    declared = load_project_settings(workspace).plugins
    return any(
        plugin.id == PROJECT_HOOKS and HooksSettings.model_validate(plugin.settings) == settings for plugin in declared
    )


class HooksPlugin(Plugin[HooksSettings, DepsT]):
    """Run commands you configure, or a project's `hooks` table, before and after tools and turns."""

    def __init__(self, host: PluginHost[DepsT], settings: HooksSettings) -> None:
        super().__init__(host, settings)
        self.root = None if settings.project is None else Path(settings.project)
        # The approval of a project's table is saved with your settings, which every repository shares:
        # it runs only while the session's project file declares that very table.
        self.active = self.root is None or _approved(host.workspace, settings)

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
        configured = command.command
        argv = configured if isinstance(configured, str) else self._argv(configured, args or {})
        shown = argv if isinstance(argv, str) else ' '.join(argv)
        instrumentation = agent_instrumentation.settings()
        # Filled placeholders and stderr can carry conversation content, which tracing may be told to leave out.
        content = instrumentation is not None and instrumentation.include_content
        traced = shown if content else configured if isinstance(configured, str) else ' '.join(configured)
        attributes = {'clai.hook.event': event, 'clai.hook.command': traced, 'clai.hook.tool': tool or ''}
        span_context = (
            instrumentation.tracer.start_as_current_span(f'clai hook {event}', attributes=attributes)
            if instrumentation is not None
            else nullcontext(INVALID_SPAN)
        )
        with span_context as span:
            done = await self._spawn(argv, payload, cwd=workspace, timeout=command.timeout, event=event)
            self._report(span, event, shown, done, tool=tool, timeout=command.timeout, content=content)
        return done

    def _argv(self, argv: list[str], args: Mapping[str, object]) -> list[str]:
        """The arguments after the program with placeholders filled, so tool arguments never choose the program.

        A project's relative program path, such as `./scripts/x.sh`, is taken from the project root.
        """
        program, *rest = argv
        if self.root is not None and ('/' in program or os.sep in program) and not Path(program).is_absolute():
            program = str(self.root / program)
        return [program, *(fill(part, args) for part in rest)]

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
        self,
        span: Span,
        event: HookEvent,
        shown: str,
        done: HookRun,
        *,
        tool: str | None,
        timeout: float,
        content: bool,
    ) -> None:
        """Record the outcome on the span, and print anything but success."""
        span.set_attribute('clai.hook.outcome', done.outcome)
        if done.outcome == 'ok':
            return
        if done.outcome == 'blocked':
            headline = f'{event} hook blocked {tool}'
            message = f'{headline}: {done.stderr or shown}'
        elif done.outcome == 'timeout':
            headline = f'{event} hook timed out after {timeout:g}s'
            message = f'{headline}: {shown}'
        else:
            headline = f'{event} hook failed'
            message = f'{headline}: {shown}: {done.stderr}'
        span.set_status(StatusCode.ERROR, message if content else headline)
        self.host.console.print(message, style=theme.color(theme.WARNING), markup=False)
