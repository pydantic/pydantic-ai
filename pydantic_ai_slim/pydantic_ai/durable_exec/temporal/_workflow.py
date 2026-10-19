from collections.abc import Sequence
from typing import Any

from pydantic_ai.agent.abstract import AbstractAgent


class PydanticAIWorkflow:
    """Temporal Workflow base class that provides `__pydantic_ai_agents__` for direct agent registration.

    Accepts any `AbstractAgent` — either a regular `Agent` carrying a
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
    capability, or the deprecated
    [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] wrapper.
    [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin]
    walks the sequence and registers each agent's activities with the worker.
    """

    __pydantic_ai_agents__: Sequence[AbstractAgent[Any, Any]]
