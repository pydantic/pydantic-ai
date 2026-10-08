"""Contain a plugin capability that rejects its own configuration while a run is set up.

A setting the reader cannot honour, saved by another CLAI build, should cost that capability,
not every turn. `PluginGuard` turns a `UserError` raised while the run is being built into
`CapabilitySetupError`, which names the plugin. That turn still fails closed: nothing is retried,
so no other capability is set up twice. The loader then leaves the capability out of later turns.

Only setup is guarded: `for_run`, and `wrap_run` until it hands over to the rest of the run.
Anything raised once the run is under way, by the model, a tool, or a hook, propagates as is.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapperCapability
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT


class CapabilitySetupError(Exception):
    """A plugin capability raised `UserError` before its run started, so the run can go on without it."""

    def __init__(self, *, plugin: str, capability: object, error: UserError) -> None:
        """Name the plugin so the user knows which one to fix."""
        super().__init__(f'Plugin {plugin!r}: {type(error).__name__}: {error}')
        self.plugin = plugin
        self.capability = capability
        self.error = error


@dataclass
class PluginGuard(WrapperCapability[AgentDepsT]):
    """Report a plugin capability's setup `UserError` as `CapabilitySetupError`, delegating everything else."""

    plugin: str = field(default='', kw_only=True)
    origin: object = field(init=False)
    """The capability the plugin registered; a per-run copy keeps pointing at it."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Per-run copies are shallow and skip `__post_init__`, so this stays the registered capability.
        self.origin = self.wrapped

    def _setup_error(self, error: UserError) -> CapabilitySetupError:
        return CapabilitySetupError(plugin=self.plugin, capability=self.origin, error=error)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        try:
            return await super().for_run(ctx)
        except UserError as exc:
            raise self._setup_error(exc) from exc

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        started = False

        async def run() -> AgentRunResult[object]:
            nonlocal started
            started = True
            return await handler()

        try:
            return await super().wrap_run(ctx, handler=run)
        except UserError as exc:
            if started:
                raise
            raise self._setup_error(exc) from exc


def setup_errors(error: BaseException) -> list[CapabilitySetupError] | None:
    """The setup errors `error` carries, also when several guards failed at once; `None` if anything else failed.

    Core sets capabilities up concurrently, so two failing at once arrive as an exception group.
    """
    if isinstance(error, CapabilitySetupError):
        return [error]
    if not isinstance(error, BaseExceptionGroup):
        return None
    found: list[CapabilitySetupError] = []
    group = cast('BaseExceptionGroup[BaseException]', error)
    for inner in group.exceptions:
        errors = setup_errors(inner)
        if errors is None:
            return None
        found.extend(errors)
    return found


def raised_here(capabilities: Sequence[AgentCapability[AgentDepsT]], error: CapabilitySetupError) -> bool:
    """Whether one of `capabilities` is the guard `error` came from, not a guard bound elsewhere."""
    return any(
        isinstance(capability, PluginGuard) and capability.origin is error.capability for capability in capabilities
    )
