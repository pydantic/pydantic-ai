from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import Field, JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import DurabilityEngineSpec, DurableRunContext
from pydantic_ai.tools import RunContext

DepsT = TypeVar('DepsT')
_JSON = TypeAdapter[JsonValue](JsonValue)


class _ActivityRunContext(DurableRunContext[DepsT]):
    def unavailable_field_message(self, name: str) -> str:
        return f'{name!r} is not available inside an Azure Durable Task activity.'


@dataclass(kw_only=True)
class _ContextSnapshot:
    context: dict[str, Any]
    deps_input: JsonValue = Field(default=None, exclude=True)

    @classmethod
    def capture(cls, ctx: RunContext[object]) -> _ContextSnapshot:
        return cls(
            context=_ActivityRunContext.serialize_run_context(ctx),
            deps_input=_JSON.validate_python(to_jsonable_python(ctx.deps, round_trip=True)),
        )


def restore_context(
    snapshot: _ContextSnapshot,
    *,
    deps: DepsT,
    agent: AbstractAgent[DepsT, Any] | None,
    engine_spec: DurabilityEngineSpec,
) -> RunContext[DepsT]:
    return _ActivityRunContext.restore(snapshot.context, deps=deps, agent=agent, engine_spec=engine_spec)
