"""Back an agent's instructions with a Logfire-managed prompt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from logfire.variables import Variable

from pydantic_ai import TemplateStr
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

# Logfire exposes a managed prompt with slug `<slug>` as a variable named `prompt__<slug>`,
# with hyphens replaced by underscores (see the Logfire prompt-management docs). `prompt__`
# is reserved for these system-managed prompts.
_PROMPT_VARIABLE_PREFIX = 'prompt__'


@dataclass
class ManagedPrompt(ManagedVariableCapability[AgentDepsT, str]):
    """Back an agent's instructions with a Logfire-managed prompt.

    > **Legacy.** `ManagedPrompt` predates
    [`AgentControl`][pydantic_ai_harness.logfire.AgentControl]; a prompt-only `AgentControl` (an
    `AgentConfig` with just `instructions`) now covers the same use case and is the recommended
    path. `ManagedPrompt` keeps working for existing setups.

    **Prompt-cache trade-off:** the resolved value lands in the system instructions block, so any
    Logfire-side change to the prompt (new version rollout, label flip, A/B targeting) invalidates
    the provider's prompt cache for the affected runs. Pin a `label` (e.g. `'production'`) for the
    cache-stable path; treat percentage rollouts and per-user targeting as opt-in cache cost. See
    the README's "Prompt-cache trade-off" section for the full picture.

    Pass the managed prompt name and a default value and the capability declares the backing
    [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    for you -- a name of `support_agent` resolves the variable `prompt__support_agent`, matching
    the naming Logfire's [Prompt management](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/)
    uses. You can iterate on the prompt from the Logfire UI -- versioned, labelled, and rolled
    out -- without redeploying, while the code default keeps the agent working when no remote
    value is available.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedPrompt

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            ManagedPrompt(
                'support_agent',
                default='You are a helpful customer support agent. Be friendly and concise.',
                label='production',
            )
        ],
    )
    result = agent.run_sync('My order never arrived.')
    ```

    The prompt value is resolved **once per run**, inside the run's
    [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] hook, using the
    [`ResolvedVariable`][logfire.variables.ResolvedVariable] as a context manager that stays open for the
    whole run -- so the selected label and version are attached as baggage to every child span
    of the agent run.

    Declaring the same name more than once is fine -- each `ManagedPrompt` constructs its own
    backing variable, so sharing a prompt across several agents just works. Pass an existing
    [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead of a prompt name
    when you want to use a variable you defined yourself (for example a `template_var`, or one
    registered for [`variables_push`][logfire.Logfire.variables_push]).
    """

    name: str | Variable[str] | None = None
    """The managed prompt name (declared as the variable `prompt__<name>`), or a pre-built
    `logfire.Variable`. When omitted, the variable is derived from the agent's own `name` at run time
    (`prompt__<agent name>`); the agent must then have a `name`."""

    default: str | None = None
    """Code-default prompt text. Required when `name` is a prompt name (or omitted); ignored when
    `name` is a `Variable`."""

    render_template: bool = field(default=False, kw_only=True)
    """When `True`, render the resolved prompt as a Handlebars template against the agent's
    `deps` (the same mechanism as [`TemplateStr`][pydantic_ai.TemplateStr]); `{{field}}` is
    filled from `deps`. Requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`).
    Defaults to `False`, so the resolved prompt is used verbatim.

    Keyword-only: this capability's other options moved to keyword-only when it was refactored onto
    [`ManagedVariableCapability`][pydantic_ai_harness.logfire.ManagedVariableCapability], and a third
    positional argument used to be `label`. Binding one here instead would silently turn a label into
    template rendering, so a call that passes one raises `TypeError` and says so."""

    def __post_init__(self) -> None:
        # A prompt name (given or derived from the agent) needs a code default; only a pre-built
        # `Variable` carries its own.
        if not isinstance(self.name, Variable) and self.default is None:
            raise TypeError('`default` is required when `name` is a prompt name rather than a `Variable`.')

        self._setup_variable(self.name, prefix=_PROMPT_VARIABLE_PREFIX, value_type=str, default=self.default or '')

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Provide the resolved prompt to the agent's system prompt."""

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            resolved = self.resolved
            if resolved is None:
                # No active run -- contribute no instructions.
                return None
            if self.render_template:
                return TemplateStr[AgentDepsT](resolved.value).render(ctx.deps)
            return resolved.value

        return instructions
