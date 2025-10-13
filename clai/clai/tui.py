from __future__ import annotations

from pathlib import Path

from textual.app import App

from clai.main_screen import MainScreen
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import Agent
from pydantic_ai.output import OutputDataT

DEFAULT_THEME = 'nord'


class CLAIApp(App[None]):
    """The CLA TUI app."""

    BINDING_GROUP_TITLE = 'App'
    CSS_PATH = 'clai.tcss'

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        history_path: Path,
        prompt: str | None = None,
        title: str | None = None,
    ):
        super().__init__()
        self._agent = agent
        self.history_path = history_path
        self.title = title or 'PyDantic CLAI'
        self._prompt = prompt

    def on_load(self) -> None:
        """Called before application mode."""
        # Set the default theme here to avoid flash of different theme
        self.theme = DEFAULT_THEME

    def get_default_screen(self) -> MainScreen:
        return MainScreen(self._agent, self.history_path, self.title, prompt=self._prompt)
