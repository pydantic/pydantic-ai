from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._display import DEFAULT_DESKTOP_SIZE
from pydantic_ai_harness.local_desktop._backend import LocalDesktopBackend, VirtualScreen

__all__ = ('LocalDesktop',)


@dataclass
class LocalDesktop(LocalWorkspace[AgentDepsT]):
    """A [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] with a screen, for [`ComputerUse`][pydantic_ai_harness.ComputerUse]. Linux only.

    By default the screen is a private, invisible one started for the agent, so it never touches your windows, mouse
    or keyboard. Commands and files are still on this machine, as with `LocalWorkspace`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ComputerUse, LocalDesktop

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalDesktop('.'), ComputerUse()])
    ```
    """

    _: KW_ONLY

    size: tuple[int, int] = DEFAULT_DESKTOP_SIZE
    """The private screen's width and height in pixels."""

    real_screen: bool = False
    """Use the X screen you're logged in to instead of a private one: the agent then moves your mouse and types into
    your windows. Not available in a Wayland session."""

    _screen: VirtualScreen | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        # One private screen for every run of this capability, so a later run finds the windows an earlier one left.
        self._screen = None if self.real_screen else VirtualScreen(self.size)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = LocalDesktopBackend(self.working_dir, env=self.env, real_screen=self.real_screen, screen=self._screen)
        if ref is not None and ref != backend.ref:
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
