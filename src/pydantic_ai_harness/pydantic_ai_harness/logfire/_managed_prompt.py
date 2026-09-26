"""Back an agent's instructions with a Logfire-managed prompt."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import logfire
from logfire.variables import Variable
from pydantic_ai import TemplateStr
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from logfire import Logfire
    from logfire.variables import ResolvedVariable
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult


# Logfire exposes a managed prompt with slug `<slug>` as a variable named `prompt__<slug>`,
# with hyphens replaced by underscores (see the Logfire prompt-management docs). `prompt__`
# is reserved for these system-managed prompts.
_PROMPT_VARIABLE_PREFIX = 'prompt__'


def _new_resolved_var() -> ContextVar[ResolvedVariable[str] | None]:
    # `None` means nothing has been resolved for the active run.
    return ContextVar('managed_prompt_resolved', default=None)


@dataclass
class ManagedPrompt(AbstractCapability[AgentDepsT]):
    """Back an agent's instructions with a Logfire-managed prompt.

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

    name: str | Variable[str]
    """The managed prompt name (declared as the variable `prompt__<name>`), or a pre-built `logfire.Variable`."""

    default: str | None = None
    """Code-default prompt text. Required when `name` is a prompt name; ignored when `name` is a `Variable`."""

    label: str | None = None
    """Explicit targeting label on the Logfire managed prompt to resolve (e.g. `'production'`).
    When `None`, the targeting rules on the managed variable select the label."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = None
    """Stable key that seeds Logfire's deterministic rollout assignment -- the same key always
    lands in the same percentage bucket, so a given user keeps the same label across runs.
    Accepts a static value or a callable that derives it from the
    [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire falls back to its own
    targeting context and then the active trace id."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = None
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    render_template: bool = False
    """When `True`, render the resolved prompt as a Handlebars template against the agent's
    `deps` (the same mechanism as [`TemplateStr`][pydantic_ai.TemplateStr]); `{{field}}` is
    filled from `deps`. Requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`).
    Defaults to `False`, so the resolved prompt is used verbatim."""

    logfire_instance: Logfire | None = None
    """Logfire instance to resolve the variable on. When `None`, the global default instance
    (the one backing the module-level [`logfire.var`][logfire.var]) is used. Ignored when
    `name` is a `Variable`."""

    _variable: Variable[str] = field(init=False, repr=False, compare=False)
    """The managed variable backing the prompt (declared from the slug, or the one passed in)."""

    _resolved: ContextVar[ResolvedVariable[str] | None] = field(
        default_factory=_new_resolved_var, init=False, repr=False, compare=False
    )
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            if self.logfire_instance is not None:
                warnings.warn(
                    '`logfire_instance` is ignored when `name` is a `Variable`; '
                    'the variable already carries its own Logfire instance.',
                    stacklevel=2,
                )
            self._variable = self.name
            return

        if self.default is None:
            raise TypeError('`default` is required when `name` is a prompt name rather than a `Variable`.')

        # Strip the prefix if the user accidentally passed it so we can still apply
        # hyphen-to-underscore normalization, then re-add the prefix below.
        name = self.name
        if name.startswith(_PROMPT_VARIABLE_PREFIX):
            warnings.warn(
                f'The {_PROMPT_VARIABLE_PREFIX!r} prefix is added automatically; '
                f'pass the bare prompt name rather than {name!r}.',
                stacklevel=2,
            )
            name = name[len(_PROMPT_VARIABLE_PREFIX) :]

        variable_name = f'{_PROMPT_VARIABLE_PREFIX}{name.replace("-", "_")}'
        if not variable_name.isidentifier():
            raise ValueError(
                f'Prompt name {self.name!r} produces an invalid variable name {variable_name!r}; '
                'names may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so redeclaring the
        # same name is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one prompt across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        self._variable = Variable(variable_name, type=str, default=self.default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[str] | None:
        """The prompt resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.variables.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which prompt version is in play.
        """
        return self._resolved.get()

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the prompt's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

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

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Resolve the prompt once and keep its baggage active for the duration of the run."""
        if callable(self.targeting_key):
            targeting_key = self.targeting_key(ctx)
        else:
            targeting_key = self.targeting_key

        if callable(self.attributes):
            attributes = self.attributes(ctx)
        else:
            attributes = self.attributes

        resolved = self._variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)
