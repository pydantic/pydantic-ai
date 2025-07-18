"""Example usage of the AG-UI adapter for Pydantic AI.

This provides a FastAPI application that demonstrates how to use the
Pydantic AI agent with the AG-UI protocol. It includes examples for
each of the AG-UI dojo features:
- Agentic Chat
- Human in the Loop
- Agentic Generative UI
- Tool Based Generative UI
- Shared State
- Predictive State Updates
"""

from __future__ import annotations

from fastapi import FastAPI

from .api import (
    agentic_chat_app,
    agentic_generative_ui_app,
    human_in_the_loop_app,
    predictive_state_updates_app,
    shared_state_app,
    tool_based_generative_ui_app,
)

app = FastAPI(title='Pydantic AI AG-UI server')
app.mount('/agentic_chat', agentic_chat_app, 'Agentic Chat')
app.mount('/agentic_generative_ui', agentic_generative_ui_app, 'Agentic Generative UI')
app.mount('/human_in_the_loop', human_in_the_loop_app, 'Human in the Loop')
app.mount(
    '/predictive_state_updates',
    predictive_state_updates_app,
    'Predictive State Updates',
)
app.mount('/shared_state', shared_state_app, 'Shared State')
app.mount(
    '/tool_based_generative_ui',
    tool_based_generative_ui_app,
    'Tool Based Generative UI',
)
