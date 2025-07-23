"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')
app = agent.to_ag_ui()
