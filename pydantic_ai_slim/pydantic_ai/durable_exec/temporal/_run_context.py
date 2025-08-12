from __future__ import annotations

from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError


class TemporalRunContext(RunContext[Any]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `retries`, `tool_call_id`, `tool_name`, `retry` and `run_step` attributes will be available.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent].

    If `deps` is a JSON-serializable dictionary, like a `TypedDict`, you can use [`TemporalRunContextWithDeps`][pydantic_ai.durable_exec.temporal.TemporalRunContextWithDeps] to make the `deps` attribute available as well.
    If `deps` is of a different type, create a `TemporalRunContext` subclass with custom `serialize_run_context` and `deserialize_run_context` class methods.
    """

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
        except AttributeError as e:  # pragma: no cover
            if name in RunContext.__dataclass_fields__:
                raise AttributeError(
                    f'{self.__class__.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to `TemporalAgent`.'
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
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any]) -> TemporalRunContext:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx)


class TemporalRunContextWithDeps(TemporalRunContext):
    """[`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass that includes JSON-serializable dictionary `deps`, like a `TypedDict`."""

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        if not isinstance(ctx.deps, dict):
            raise UserError(
                '`TemporalRunContextWithDeps` requires the `deps` object to be a JSON-serializable dictionary, like a `TypedDict`. '
                'To support `deps` of a different type, pass a `TemporalRunContext` subclass to `TemporalAgent` with custom `serialize_run_context` and `deserialize_run_context` class methods.'
            )
        return {**super().serialize_run_context(ctx), 'deps': ctx.deps}  # pyright: ignore[reportUnknownMemberType]
