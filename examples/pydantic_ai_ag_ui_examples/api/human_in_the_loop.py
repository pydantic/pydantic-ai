"""Human in the Loop Feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai.ag_ui import FastAGUI

from .agent import agent

app: FastAGUI = agent(
    instructions="""When planning tasks use tools only, without any other messages.
IMPORTANT:
- Use the `generate_task_steps` tool to display the suggested steps to the user
- Never repeat the plan, or send a message detailing steps
- If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
- If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
"""
)
