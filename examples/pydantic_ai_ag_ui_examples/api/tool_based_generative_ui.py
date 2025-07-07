"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai.ag_ui import FastAGUI

from .agent import agent

app: FastAGUI = agent()
