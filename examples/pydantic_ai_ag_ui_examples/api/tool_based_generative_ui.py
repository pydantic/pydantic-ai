"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.ag_ui import AGUIApp

# Ensure environment variables are loaded.
load_dotenv()

agent: Agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
)

app: AGUIApp = agent.to_ag_ui()
