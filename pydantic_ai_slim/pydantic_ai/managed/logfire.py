"""Logfire managed variables integration for Pydantic AI agents."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import Any, cast

try:
    from logfire.variables import ResolvedVariable, Variable
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'The `pydantic_ai.managed.logfire` module requires `logfire>=4.24.0`. '
        'Install it with `pip install "pydantic-ai-slim[logfire]"`.'
    ) from e

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.agent.abstract import AgentModelSettings
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

__all__ = ['Managed']


@dataclass
class Managed(AbstractCapability[AgentDepsT]):
    """Capability that injects Logfire managed variables into an agent run.

    See the [Logfire managed variables guide](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    for more background.

    Wire up agent fields to values controlled from the Logfire UI: change a
    prompt or a model setting without redeploying. Each variable is resolved
    once per run via `variable.get()`; its current `.value` feeds the matching
    capability hook, and the resolved variable is entered as a context manager
    around the run so downstream Logfire spans and logs are tagged with the
    active label.

    Stack multiple `Managed` capabilities (or list a single one with several
    fields set) to layer variables. Capability ordering rules apply: the first
    in the list is outermost, and later contributions merge on top.

    Phase 1 supports `instructions` and `model_settings`. Additional fields
    (model, metadata, whole-spec variables) are planned follow-ups.

    Example:
    ```python {test="skip"}
    import logfire

    from pydantic_ai import Agent
    from pydantic_ai.managed.logfire import Managed

    logfire.configure()

    prompt = logfire.var('main_agent_prompt', default='You are a helpful assistant.')
    settings = logfire.var(
        'main_agent_settings',
        type=dict,
        default={'temperature': 0.0},
    )

    agent = Agent(
        'openai:gpt-4o',
        capabilities=[Managed(instructions=prompt, model_settings=settings)],
    )
    ```
    """

    instructions: Variable[str] | None = None
    """Managed variable contributing agent instructions.

    Its current value is appended to the agent's static instructions on each run.
    """

    model_settings: Variable[dict[str, Any]] | None = None
    """Managed variable contributing model settings.

    Its current value is merged on top of the agent's static model settings on each run.
    """

    _resolved_instructions: ResolvedVariable[str] | None = field(default=None, init=False, repr=False, compare=False)
    _resolved_model_settings: ResolvedVariable[dict[str, Any]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Managed[AgentDepsT]:
        resolved = replace(self)
        resolved._resolved_instructions = self.instructions.get() if self.instructions is not None else None
        resolved._resolved_model_settings = self.model_settings.get() if self.model_settings is not None else None
        return resolved

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        if self._resolved_instructions is None:
            return None
        return self._resolved_instructions.value

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        if self._resolved_model_settings is None:
            return None
        return cast(ModelSettings, self._resolved_model_settings.value)

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        async with AsyncExitStack() as stack:
            if self._resolved_instructions is not None:
                stack.enter_context(self._resolved_instructions)
            if self._resolved_model_settings is not None:
                stack.enter_context(self._resolved_model_settings)
            return await handler()
