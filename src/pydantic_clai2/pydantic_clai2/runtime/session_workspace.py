"""`clai2 --web`'s session directories: each live session works in its own, moved to between turns.

The terminal shell stays in the directory it started in. Under `clai2 --web`, a session that moves
records its new directory as its workspace, then loads its plugins again there: the coding tools,
`repo_context`, skills, and MCP project files read it through `PluginHost.workspace`. The process's
working directory stays where CLAI started, as other live sessions work in other directories.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings, load_project_settings
from pydantic_clai2.plugins.loader import SessionPlugins
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.rendering.status import Status


def workspace_directory(path: str) -> Path:
    """`path` as a session's working directory, resolved; `ValueError` unless it is an existing absolute directory."""
    directory = Path(path)
    if not path or not directory.is_absolute():
        raise ValueError(f'Choose a directory by its absolute path, not {path!r}.')
    if not directory.is_dir():
        problem = 'is not a directory' if directory.exists() else 'does not exist'
        raise ValueError(f'{path} {problem}.')
    return directory.resolve()


@dataclass(kw_only=True)
class SessionWorkspace:
    """Moves `clai2 --web`'s live session into another directory, as if CLAI had started there.

    Never while a turn, a background fork, or a delegated task runs: their tools still use the plugins loaded now.
    """

    session: Session[None, str]
    context: CommandContext
    loader: SessionPlugins[None]
    status: Status
    settings: Callable[[ProjectSettings], Settings]
    """The settings CLAI starts with beside a project file: saved ones, then the project's, then launch flags."""
    busy: Callable[[], bool] = lambda: False
    """Whether a background fork or a delegated task runs."""

    def attach(self, *, sessions: Sessions[None, str], commands: Commands) -> None:
        """Make `/resume ID` move into the session's directory, and add `/workspace`."""
        sessions.relocate = self.move
        commands.register(
            Command(
                name='workspace',
                description='Start a new session in DIRECTORY, an absolute path; bare, show where this one works',
                handler=self.command,
                raw=True,
            )
        )

    async def move(self, path: str) -> None:
        """Make `path` the working directory of the next run; the conversation is left as it is.

        Raises `ValueError` for a path that is not a directory or a project file that does not load,
        and `RuntimeError` while something runs; either way nothing changes.
        """
        directory = workspace_directory(path)
        if str(directory) == self.session.workspace:
            return
        if self.session.running or self.busy():
            raise RuntimeError('Something is running. Switch directories once it ends.')
        project = load_project_settings(directory)
        settings = self.settings(project)
        self.session.workspace = self.status.workspace = str(directory)
        self.context.use_project(project, settings)
        await self.loader.restart(project=project.plugins)

    async def command(self, args: list[str]) -> str:
        """`/workspace DIRECTORY` starts a new session in DIRECTORY; bare, it names this session's."""
        if not args:
            return f'This session works in {self.session.workspace}.'
        await self.move(args[0])
        await self.session.clear()
        return f'New session in {self.session.workspace}.'
