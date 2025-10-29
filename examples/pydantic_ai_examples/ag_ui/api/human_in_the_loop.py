"""Human in the Loop Feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent(
    'openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning tasks use tools only, without any other messages.
        IMPORTANT:
        - Use the `generate_task_steps` tool to display the suggested steps to the user
        - Never repeat the plan, or send a message detailing steps
        - If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
        - If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
        """
    ),
)

app = AGUIApp(agent)
