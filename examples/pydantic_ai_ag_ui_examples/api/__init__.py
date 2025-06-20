"""Example API for a AG-UI compatible Pydantic AI Agent UI."""

from __future__ import annotations

from .agentic_chat import router as agentic_chat_router
from .agentic_generative_ui import router as agentic_generative_ui_router
from .human_in_the_loop import router as human_in_the_loop_router
from .predictive_state_updates import router as predictive_state_updates_router
from .shared_state import router as shared_state_router
from .tool_based_generative_ui import router as tool_based_generative_ui_router

__all__: list[str] = [
    'agentic_chat_router',
    'agentic_generative_ui_router',
    'human_in_the_loop_router',
    'predictive_state_updates_router',
    'shared_state_router',
    'tool_based_generative_ui_router',
]
