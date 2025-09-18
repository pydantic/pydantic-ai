from __future__ import annotations

from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext


class HatchetRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Hatchet task.

    By default, only the `deps`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry` and `run_step` attributes will be available.
    To make another attribute available, create a `HatchetRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`HatchetAgent`][pydantic_ai.durable_exec.hatchet.HatchetAgent].
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:  # pragma: no cover
            if name in RunContext.__dataclass_fields__:
                raise UserError(
                    f'{self.__class__.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, create a `HatchetRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to `HatchetAgent`.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        return {
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: AgentDepsT) -> HatchetRunContext[AgentDepsT]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)
