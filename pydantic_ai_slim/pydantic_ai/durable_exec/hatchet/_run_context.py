from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypeVar

from pydantic_ai.durable_exec._run_context import DurableRunContext, restore_tracer
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext

from ._spec import HATCHET_ENGINE_SPEC

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)

_SYNTHETIC_CAPABILITY_ID = re.compile(r'<([a-z0-9_]+):[0-9a-f]{6}>')
"""The run-local key core gives a capability with no `id`, with a random suffix that differs every run."""


def stable_capability_ids(ctx: RunContext[Any]) -> dict[str, str]:
    """Replacements for the run's synthetic capability keys that a replay of the durable task reproduces.

    Hatchet checks a replayed child task's input against the recorded one, so it can't carry a key that's random
    every run. Numbering the keys in attachment order, `<hatchet_durability:0>`, keeps them distinct and makes them
    the same on a replay. They name nothing in the task either way: its agent keys its own capabilities.
    """
    try:
        registry = ctx.capabilities
    except UserError:
        # A context rebuilt inside a task has no registry to read, nor synthetic keys of its own to replace.
        return {}
    replacements: dict[str, str] = {}
    counts: dict[str, int] = {}
    for capability_id in registry:
        if (match := _SYNTHETIC_CAPABILITY_ID.fullmatch(capability_id)) is not None:
            base = match.group(1)
            replacements[capability_id] = f'<{base}:{counts.get(base, 0)}>'
            counts[base] = counts.get(base, 0) + 1
    return replacements


class HatchetRunContext(DurableRunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] used inside the Hatchet tasks a durable agent run spawns.

    A task may run on another worker, so only the fields
    [`DurableRunContext`][pydantic_ai.durable_exec.DurableRunContext] serializes cross into it, and reading
    any other field raises a `UserError` explaining how to make it available. `agent` is re-attached from the
    worker's agent instance, and `tracer` is resolved from it.

    To make another attribute available, create a `HatchetRunContext` subclass with a custom
    `serialize_run_context` class method that returns a dictionary that includes the attribute, and pass it
    as the `run_context_type` argument to [`HatchetDurability`][pydantic_ai.durable_exec.hatchet.HatchetDurability].
    """

    def unavailable_field_message(self, name: str) -> str:
        return (
            f'{name!r} is not available on {self.__class__.__name__!r} inside a Hatchet task. '
            'To make the attribute available, create a `HatchetRunContext` subclass with a custom '
            '`serialize_run_context` class method that returns a dictionary that includes the attribute and pass it '
            'as the `run_context_type` argument to `HatchetDurability`.'
        )

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context so a replay of the durable task serializes it identically.

        Sets cross as sorted lists, since a set's order differs between processes, and synthetic capability keys
        are replaced by `stable_capability_ids`.
        """
        serialized = super().serialize_run_context(ctx)
        replacements = stable_capability_ids(ctx)
        for key, value in serialized.items():
            if isinstance(value, set | frozenset):
                items = cast('set[str]', value)
                if key.endswith('capability_ids'):
                    items = {replacements.get(item, item) for item in items}
                serialized[key] = sorted(items)
        return serialized

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> HatchetRunContext[Any]:
        return cls(**ctx, deps=deps)


def deserialize_run_context(
    run_context_type: type[HatchetRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
    """Rebuild a task's run context, resolving `tracer` from the worker's agent unless the subclass set it."""
    ctx = run_context_type.restore(serialized, deps=deps, agent=agent, engine_spec=HATCHET_ENGINE_SPEC)
    restore_tracer(ctx, agent)
    return ctx
