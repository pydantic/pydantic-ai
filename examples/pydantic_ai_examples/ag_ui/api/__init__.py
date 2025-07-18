"""Example API for a AG-UI compatible Pydantic AI Agent UI."""

from __future__ import annotations

from .agentic_chat import app as agentic_chat_app
from .agentic_generative_ui import app as agentic_generative_ui_app
from .human_in_the_loop import app as human_in_the_loop_app
from .predictive_state_updates import app as predictive_state_updates_app
from .shared_state import app as shared_state_app
from .tool_based_generative_ui import app as tool_based_generative_ui_app

__all__ = [
    'agentic_chat_app',
    'agentic_generative_ui_app',
    'human_in_the_loop_app',
    'predictive_state_updates_app',
    'shared_state_app',
    'tool_based_generative_ui_app',
]
