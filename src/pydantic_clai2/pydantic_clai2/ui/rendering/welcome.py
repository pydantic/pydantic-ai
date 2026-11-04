"""The banner, version, and hints a fresh launch and `/clear` show."""

from rich.console import Console

from pydantic_clai2.cli.self_update import installed
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.runtime.worktrees import Worktree
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.branding import print_banner


def print_welcome(project: ProjectSettings, console: Console, *, worktree: Worktree | None = None) -> None:
    """The banner, version, and hints a fresh launch shows, which `/clear` returns to without the worktree notice."""
    console.print()
    print_banner(console)
    console.print(f'pydantic-clai2 {installed().label}', style=theme.color(theme.MUTED))
    if worktree is not None:
        # Soft wrap keeps the path copyable: hard wrapping would break it with newlines.
        console.print(worktree.notice, style=theme.color(theme.MUTED), markup=False, highlight=False, soft_wrap=True)
    console.print(
        '/new starts a session; /resume restores one; /exit quits. Esc or Ctrl-C interrupts a turn.',
        style=theme.color(theme.MUTED),
    )
    _report_project(project, console)


def _report_project(project: ProjectSettings, console: Console) -> None:
    if project.path is None:
        return
    console.print(f'Project settings: {project.path}', style=theme.color(theme.MUTED))
    if project.unknown:
        console.print(f'Ignoring unknown settings: {", ".join(project.unknown)}', style=theme.color(theme.WARNING))
