from __future__ import annotations

from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError


class TemporalRunContext(RunContext[Any]):
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
                    f'{self.__class__.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to `TemporalAgent`.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        return {
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any]) -> RunContext[Any]:
        return cls(**ctx)


class TemporalRunContextWithDeps(TemporalRunContext):
    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        if not isinstance(ctx.deps, dict):
            raise UserError(
                'The `deps` object must be a JSON-serializable dictionary in order to be used with Temporal. '
                'To use a different type, pass a `TemporalRunContext` subclass to `TemporalAgent` with custom `serialize_run_context` and `deserialize_run_context` class methods.'
            )
        return {**super().serialize_run_context(ctx), 'deps': ctx.deps}  # pyright: ignore[reportUnknownMemberType]
