from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, SkipValidation, model_serializer

if TYPE_CHECKING:
    from pydantic_ai.tools import AgentDepsT, RunContext
else:
    from pydantic_ai.tools import AgentDepsT
    RunContext = Any


class SerializableRunContext(BaseModel):
    """A wrapper around RunContext that provides serialization for Prefect cache keys.

    This wrapper excludes non-serializable fields (model, usage, messages, tracer, retries)
    from serialization while maintaining full access to the wrapped RunContext.

    The wrapper is transparent - all attribute access is delegated to the wrapped RunContext,
    so it can be used as a drop-in replacement wherever RunContext is expected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: Annotated[Any, SkipValidation]

    @classmethod
    def wrap(cls, ctx: RunContext[AgentDepsT]) -> SerializableRunContext:
        """Wrap a RunContext to make it serializable for Prefect cache keys.

        Args:
            ctx: The RunContext to wrap.

        Returns:
            A SerializableRunContext that wraps the given context.
        """
        return cls(wrapped=ctx)

    def unwrap(self) -> Any:  # type: ignore[return]
        """Unwrap to get the original RunContext.

        Returns:
            The wrapped RunContext.
        """
        return self.wrapped

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the wrapped RunContext."""
        return getattr(self.wrapped, name)

    @model_serializer
    def ser_model(self) -> dict[str, Any]:
        """Serialize only the cacheable fields of RunContext for Prefect.

        Excludes non-serializable fields:
        - model: Contains HTTP clients and other non-serializable objects
        - usage: Contains internal state that shouldn't affect caching
        - messages: May contain binary data or non-serializable content
        - tracer: OpenTelemetry tracer object
        - retries: Internal state dict
        - trace_include_content: Internal flag

        Includes cacheable fields:
        - deps: User-provided dependencies (if serializable)
        - prompt: User prompt string
        - tool_call_id: Tool call identifier
        - tool_name: Tool name
        - retry: Current retry count
        - max_retries: Maximum retry count
        - run_step: Current step in the run
        - tool_call_approved: Approval status
        """
        ctx = self.wrapped
        result: dict[str, Any] = {
            'prompt': ctx.prompt,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'tool_call_approved': ctx.tool_call_approved,
        }

        # Try to include deps if it's serializable
        # If deps is not serializable, Prefect will handle the error appropriately
        try:
            result['deps'] = ctx.deps
        except Exception:
            # If deps can't be serialized, omit it from the cache key
            pass

        return result
