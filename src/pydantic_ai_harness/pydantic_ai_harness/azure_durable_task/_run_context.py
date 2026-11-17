from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import Field, JsonValue
from pydantic_core import to_jsonable_python

from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.tools import RunContext

DepsT = TypeVar('DepsT')


class ActivityRunContext(DurableRunContext[DepsT]):
    def unavailable_field_message(self, name: str) -> str:
        return f'{name!r} is not available inside an Azure Durable Task activity.'


@dataclass(kw_only=True)
class ContextSnapshot:
    context: dict[str, Any]
    deps_input: JsonValue = Field(default=None, exclude=True)

    @classmethod
    def capture(cls, ctx: RunContext[object]) -> ContextSnapshot:
        return cls(
            context=ActivityRunContext.serialize_run_context(ctx),
            deps_input=to_jsonable_python(ctx.deps, round_trip=True),
        )
