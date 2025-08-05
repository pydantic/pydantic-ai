from __future__ import annotations

from textual.app import App

from clai.main_screen import MainScreen
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import Agent
from pydantic_ai.output import OutputDataT


class CLAIApp(App[None]):
    """The CLA TUI app."""

    CSS_PATH = 'clai.tcss'

    def __init__(self, agent: Agent[AgentDepsT, OutputDataT], prompt: str | None = None):
        self._agent = agent
        self._prompt = prompt
        super().__init__()

    def on_load(self) -> None:
        self.theme = 'nord'

    def get_default_screen(self) -> MainScreen:
        return MainScreen(self._agent, self._prompt)
