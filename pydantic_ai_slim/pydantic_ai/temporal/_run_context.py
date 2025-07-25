from __future__ import annotations

from typing import Any

from pydantic_ai._run_context import AgentDepsT, RunContext


class TemporalRunContext(RunContext[AgentDepsT]):
    def __init__(self, **kwargs: Any):
        self.__dict__ = kwargs
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in kwargs},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if name in RunContext.__dataclass_fields__:
                raise AttributeError(
                    f'Temporalized {RunContext.__name__!r} object has no attribute {name!r}. To make the attribute available, pass a `TemporalSettings` object to `temporalize_agent` that has a custom `serialize_run_context` function that returns a dictionary that includes the attribute.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[AgentDepsT]) -> dict[str, Any]:
        return {
            'deps': ctx.deps,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any]) -> RunContext[AgentDepsT]:
        return cls(**ctx)
