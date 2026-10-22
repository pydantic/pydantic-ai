"""Back an agent's instructions with a Logfire-managed prompt."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import logfire
from logfire.variables import Variable
from pydantic_handlebars import HandlebarsError

from pydantic_ai import TemplateStr
from pydantic_ai._instructions import SourcedInstruction, append_on_change_by_default
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

_ID_PREFIX = 'managed-prompt'

# ASCII only, matching the variable-name rule `logfire.var` enforces.
_PROMPT_NAME = re.compile(r'[A-Za-z0-9_-]+')


@dataclass
class _ActivePrompt:
    """The prompt resolved for the active run."""

    resolved: ResolvedVariable[str]
    template: TemplateStr[object] | None
    """The compiled template when `render_template=True`, compiled once per run."""


def _is_template_variable(variable: Variable[str]) -> bool:
    try:
        from logfire.variables import TemplateVariable
    except ImportError:  # pragma: no cover -- `TemplateVariable` was added in logfire 4.36
        return False
    return isinstance(variable, TemplateVariable)


def _new_active_var() -> ContextVar[_ActivePrompt | None]:
    # `None` means nothing has been resolved for the active run.
    return ContextVar('managed_prompt_active', default=None)


@dataclass
class ManagedPrompt(AbstractCapability[AgentDepsT]):
    """Back an agent's instructions with a Logfire-managed prompt.

    Pass the managed prompt name and a default value and the capability declares the backing
    [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    for you: a name of `support_agent` resolves the variable `prompt__support_agent`, the naming
    Logfire's [Prompt management](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/)
    uses. Edit, version, label, and roll out the prompt in the Logfire UI; the next run picks it
    up. The code default is used until a remote value is published.

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

    The prompt is resolved once per run, in the run's
    [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] hook. The
    [`ResolvedVariable`][logfire.variables.ResolvedVariable] stays open as a context manager for
    the whole run, so the selected label (and the version, on Logfire 4.34 and later) is attached as
    baggage to every span of the run.

    The resolved value lands in the system instructions, so a new value starts new conversations on
    their own cached prompt prefix; a conversation already under way gets the change appended
    instead. See the docs' "Prompt-cache trade-off" section.

    Each `ManagedPrompt` constructs its own unregistered variable, so several agents can use the
    same prompt name. Within one agent, the capability's `id` is derived from the prompt name
    (`managed-prompt-support_agent` above), so two prompts with different names each get their own
    instruction block, and two with the same name raise `UserError` when the agent is built. Pass
    `id=` to override it. Pass a [`logfire.variables.Variable`][logfire.variables.Variable] as
    `name` to use one you declared yourself, for example one registered for
    [`variables_push`][logfire.Logfire.variables_push]. `TemplateVariable` is not supported.
    """

    name: str | Variable[str]
    """The managed prompt name (declared as the variable `prompt__<name>`), or a pre-built `logfire.Variable`."""

    default: str | None = None
    """Code-default prompt text. Required when `name` is a prompt name; ignored, with a warning, when
    `name` is a `Variable`."""

    label: str | None = None
    """Label on the Logfire managed prompt to resolve (e.g. `'production'`).
    When `None`, the targeting rules on the managed variable select the label. When the label does
    not exist on the variable, Logfire falls back to the rollout and targeting rules."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = None
    """Stable key that seeds Logfire's deterministic rollout assignment: the same key always
    lands in the same percentage bucket, so a given user keeps the same label across runs.
    Accepts a static value or a callable that derives it from the
    [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire uses its
    `targeting_context`, then the trace id of the span enclosing the run."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = None
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    render_template: bool = False
    """When `True`, render the resolved prompt as a Handlebars template against the agent's
    `deps` (the same mechanism as [`TemplateStr`][pydantic_ai.template.TemplateStr]); `{{field}}` is
    filled from `deps`. Anyone who can edit the prompt in Logfire can then place any `deps` value
    in the system prompt. A remote prompt that fails to compile falls back to the code default
    with a `RuntimeWarning` and a Logfire warning log, and the run is then reported as the code
    default. Defaults to `False`, so the resolved prompt is used verbatim."""

    logfire_instance: Logfire | None = None
    """Logfire instance to resolve the variable on. When `None`, the global default instance
    (the one backing the module-level [`logfire.var`][logfire.var]) is used. Ignored, with a
    warning, when `name` is a `Variable`."""

    _variable: Variable[str] = field(init=False, repr=False, compare=False)
    """The managed variable backing the prompt (declared from the slug, or the one passed in)."""

    _active: ContextVar[_ActivePrompt | None] = field(
        default_factory=_new_active_var, init=False, repr=False, compare=False
    )
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    def __post_init__(self) -> None:
        self._variable = self._build_variable()
        if self.id is None:
            self.id = f'{_ID_PREFIX}-{self._variable.name.removeprefix(_PROMPT_VARIABLE_PREFIX)}'

    def _build_variable(self) -> Variable[str]:
        if not isinstance(self.name, str):
            if _is_template_variable(self.name):
                raise TypeError(
                    '`ManagedPrompt` does not support `TemplateVariable`; pass a plain `Variable` '
                    'and set `render_template=True` to render the prompt against `deps`.'
                )
            if self.default is not None:
                warnings.warn(
                    '`default` is ignored when `name` is a `Variable`; the variable already carries its own default.',
                    stacklevel=4,
                )
            if self.logfire_instance is not None:
                warnings.warn(
                    '`logfire_instance` is ignored when `name` is a `Variable`; '
                    'the variable already carries its own Logfire instance.',
                    stacklevel=4,
                )
            return self.name

        if self.default is None:
            raise TypeError('`default` is required when `name` is a prompt name rather than a `Variable`.')

        # Strip the prefix if the user accidentally passed it so we can still apply
        # hyphen-to-underscore normalization, then re-add the prefix below.
        name = self.name
        if name.startswith(_PROMPT_VARIABLE_PREFIX):
            warnings.warn(
                f'The {_PROMPT_VARIABLE_PREFIX!r} prefix is added automatically; '
                f'pass the bare prompt name rather than {name!r}.',
                stacklevel=4,
            )
            name = name[len(_PROMPT_VARIABLE_PREFIX) :]

        if not _PROMPT_NAME.fullmatch(name):
            raise ValueError(
                f'Prompt name {self.name!r} is not a valid managed prompt name; '
                'names may only contain ASCII letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so redeclaring the
        # same name is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one prompt across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        variable_name = f'{_PROMPT_VARIABLE_PREFIX}{name.replace("-", "_")}'
        return Variable(variable_name, type=str, default=self.default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[str] | None:
        """The prompt resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.variables.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which prompt version is in play.
        """
        active = self._active.get()
        return active.resolved if active is not None else None

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the prompt's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Provide the resolved prompt to the agent's system prompt."""

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            active = self._active.get()
            if active is None:
                # No active run -- contribute no instructions.
                return None
            if active.template is not None:
                return active.template.render(ctx.deps)
            return active.resolved.value

        return instructions

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        # The prompt is resolved once per run, so a new version published between runs of a conversation is
        # appended as a change (a diff, when that's shorter) rather than rewriting the cached prompt. The text
        # is the operator's own prompt, from Logfire or the code default (rendered against `deps` when
        # `render_template=True`).
        return append_on_change_by_default(super()._collect_instructions(), key=_ID_PREFIX)

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
        template = None
        if self.render_template:
            resolved, template = self._compile_template(resolved, targeting_key, attributes)
        with resolved:
            token = self._active.set(_ActivePrompt(resolved, template))
            try:
                return await handler()
            finally:
                self._active.reset(token)

    def _compile_template(
        self, resolved: ResolvedVariable[str], targeting_key: str | None, attributes: Mapping[str, Any] | None
    ) -> tuple[ResolvedVariable[str], TemplateStr[object]]:
        """Compile the resolved prompt, falling back to the code default when it is not a valid template.

        On fallback, the returned resolution describes the code default that actually runs, so the
        run's baggage and `resolved` don't credit the remote version that failed to compile.
        """
        try:
            return resolved, TemplateStr[object](resolved.value)
        except HandlebarsError as e:
            # Resolve the code default through Logfire, so a callable default sees the same merged
            # targeting context (resource attributes, baggage, targeting key fallback) it would in `get()`.
            with self._variable.override(self._variable.default):
                fallback = self._variable.get(targeting_key=targeting_key, attributes=attributes)
            if fallback.value == resolved.value:
                raise
            template = TemplateStr[object](fallback.value)  # an invalid code default raises here
            message = (
                f'Managed prompt {self._variable.name!r} (label {resolved.label!r}, version {resolved.version!r}) '
                f'is not a valid template, so the code default is used: {e}'
            )
            warnings.warn(message, RuntimeWarning)
            # Also log it where whoever edited the prompt in Logfire will see it.
            self._variable.logfire_instance.warning(
                'Managed prompt {variable} (label {label}, version {version}) is not a valid template; '
                'using the code default',
                variable=self._variable.name,
                label=resolved.label,
                version=resolved.version,
                _exc_info=e,
            )
            fallback.exception = e
            return fallback, template
