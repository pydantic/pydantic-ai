"""AG-UI protocol integration for Pydantic AI agents."""

from typing import Any

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT

from .. import UIApp
from ._adapter import AGUIAdapter
from ._event_stream import AGUIEventStream

__all__ = [
    'AGUIAdapter',
    'AGUIEventStream',
    'AGUIApp',
]


class AGUIApp(UIApp[AgentDepsT, OutputDataT]):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(self, agent: AbstractAgent[AgentDepsT, OutputDataT], **kwargs: Any):
        super().__init__(AGUIAdapter[AgentDepsT, OutputDataT], agent, **kwargs)
