import logging
from functools import lru_cache

from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """PydanticAI Agent registry."""

    def __init__(self) -> None:
        self.chat_completions_agents: dict[str, Agent] = {}
        self.responses_agents: dict[str, Agent] = {}

    def _get_name(self, agent_name: str | None, override_name: str | None) -> str:
        name = override_name or agent_name
        if name is None:
            raise ValueError(
                'Agent has no model name set and override_name has not been provided. Define a model in the agent or explicitly pass the name.',
            )

        return name

    def register_responses_agent(
        self,
        agent: Agent,
        override_name: str | None = None,
    ) -> 'AgentRegistry':
        """Register an agent that will be exposed as /v1/responses API model."""
        name = self._get_name(agent.name, override_name)

        if name in self.responses_agents:
            logger.warning('Overriding responses agent that has already been set in registry.')

        self.responses_agents[name] = agent
        return self

    def register_completions_agent(
        self,
        agent: Agent,
        override_name: str | None = None,
    ) -> 'AgentRegistry':
        """Register an agent that will be exposed as /v1/chat/completions API model."""
        name = self._get_name(agent.name, override_name)

        if name in self.chat_completions_agents:
            logger.warning(
                'Overriding chat completions agent that has already been set in registry.',
            )

        self.chat_completions_agents[name] = agent
        return self

    def get_responses_agent(self, name: str) -> Agent:
        """Get responses API agent."""
        agent = self.responses_agents.get(name)

        if agent is None:
            raise KeyError('Responses agent with %s has not been registered.', name)
        return agent

    def get_completions_agent(self, name: str) -> Agent:
        """Get chat completions API agent."""
        agent = self.chat_completions_agents.get(name)

        if agent is None:
            raise KeyError('Completions agent with %s has not been registered.', name)
        return agent

    @property
    @lru_cache
    def all_agents(self) -> list[str]:
        """Retrieves all registered agents in the registry."""
        unique_agent_set = {*self.chat_completions_agents.keys(), *self.responses_agents.keys()}
        return [*unique_agent_set]
