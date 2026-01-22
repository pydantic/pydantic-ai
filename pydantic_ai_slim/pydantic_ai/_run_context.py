from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic

from opentelemetry.trace import NoOpTracer, Tracer
from typing_extensions import TypeVar

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION

from . import _utils, messages as _messages

if TYPE_CHECKING:
    from .models import Model
    from .result import RunUsage

# TODO (v2): Change the default for all typevars like this from `None` to `object`
AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
"""Type variable for agent dependencies."""

RunContextAgentDepsT = TypeVar('RunContextAgentDepsT', default=None, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    """Information about the current call."""

    deps: RunContextAgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: RunUsage
    """LLM usage associated with the run."""
    prompt: str | Sequence[_messages.UserContent] | None = None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    validation_context: Any = None
    """Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) for tool args and run outputs."""
    tracer: Tracer = field(default_factory=NoOpTracer)
    """The tracer to use for tracing the run."""
    trace_include_content: bool = False
    """Whether to include the content of the messages in the trace."""
    instrumentation_version: int = DEFAULT_INSTRUMENTATION_VERSION
    """Instrumentation settings version, if instrumentation is enabled."""
    retries: dict[str, int] = field(default_factory=dict)
    """Number of retries for each tool so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries of this tool so far."""
    max_retries: int = 0
    """The maximum number of retries of this tool."""
    run_step: int = 0
    """The current step in the run."""
    tool_call_approved: bool = False
    """Whether a tool call that required approval has now been approved."""
    tool_call_metadata: Any = None
    """Metadata from `DeferredToolResults.metadata[tool_call_id]`, available when `tool_call_approved=True`."""
    partial_output: bool = False
    """Whether the output passed to an output validator is partial."""
    run_id: str | None = None
    """"Unique identifier for the agent run."""
    metadata: dict[str, Any] | None = None
    """Metadata associated with this agent run, if configured."""

    @property
    def last_attempt(self) -> bool:
        """Whether this is the last attempt at running this tool before an error is raised."""
        return self.retry == self.max_retries

    __repr__ = _utils.dataclasses_no_defaults_repr


_CURRENT_RUN_CONTEXT: ContextVar[RunContext[Any] | None] = ContextVar(
    'pydantic_ai.current_run_context',
    default=None,
)
"""Context variable storing the current [`RunContext`][pydantic_ai.tools.RunContext]."""


def get_current_run_context() -> RunContext[Any] | None:
    """Get the current run context, if one is set.

    Returns:
        The current [`RunContext`][pydantic_ai.tools.RunContext], or `None` if not in an agent run.
    """
    return _CURRENT_RUN_CONTEXT.get()


@contextmanager
def set_current_run_context(run_context: RunContext[Any]) -> Iterator[None]:
    """Context manager to set the current run context.

    Args:
        run_context: The run context to set as current.

    Yields:
        None
    """
    token = _CURRENT_RUN_CONTEXT.set(run_context)
    try:
        yield
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)
