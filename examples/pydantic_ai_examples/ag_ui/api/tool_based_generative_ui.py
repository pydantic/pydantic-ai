"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('openai:gpt-4o-mini')
app = AGUIApp(agent)
