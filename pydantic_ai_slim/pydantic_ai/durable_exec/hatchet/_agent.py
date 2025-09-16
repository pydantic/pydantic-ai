from __future__ import annotations

from pydantic_ai.agent import AbstractAgent, WrapperAgent
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import (
    AgentDepsT,
)


class HatchetAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    ):
        """Wrap an agent to enable it with DBOS durable workflows, by automatically offloading model requests, tool calls, and MCP server communication to DBOS steps.

        After wrapping, the original agent can still be used as normal outside of the DBOS workflow.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use as the DBOS configured instance name. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            mcp_step_config: The base DBOS step config to use for MCP server steps. If no config is provided, use the default settings of DBOS.
            model_step_config: The DBOS step config to use for model request steps. If no config is provided, use the default settings of DBOS.
        """
        super().__init__(wrapped)

        pass
