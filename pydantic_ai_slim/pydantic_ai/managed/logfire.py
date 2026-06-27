"""Logfire managed variables integration for Pydantic AI agents."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import Any, cast

from pydantic_ai.agent.spec import AgentSpec

try:
    from logfire.variables import ResolvedVariable, Variable
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'The `pydantic_ai.managed.logfire` module requires `logfire>=4.24.0`. '
        'Install it with `pip install "pydantic-ai-slim[logfire]"`, '
        'or upgrade an existing install with `pip install -U "logfire>=4.24.0"`.'
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

    Currently, `instructions` and `model_settings` are supported. Support for
    additional fields (including the model itself, per-run metadata, and
    whole-spec variables) is planned.

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
        'openai:gpt-5.2',
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
        # TODO: Aditya: Concern here is if the keys coming in from ResolvedVariable are even valid keys in ModelSettings.
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


@dataclass
class ManagedAgentSpec(AbstractCapability[AgentDepsT]):
    agent_spec: Variable[AgentSpec] | None = None
    """ Managed spec contributing the entire spec."""

    def contribute_run_spec(self) -> AgentSpec | None:
        return self.agent_spec.get().value if self.agent_spec is not None else None

    # TODO(baggage): like `Managed`, this should resolve `agent_spec` once and enter the
    # `ResolvedVariable` as a context manager around the run (a `wrap_run` override) so child
    # Logfire spans are tagged with the spec's label/version. Not yet implemented: `contribute_run_spec`
    # resolves in the pre-model window, but `wrap_run` runs later, so the resolved variable needs to be
    # stashed where `wrap_run` can re-enter it. See `Managed.wrap_run` for the pattern to mirror.
