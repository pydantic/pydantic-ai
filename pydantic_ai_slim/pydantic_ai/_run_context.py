from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Generic

from typing_extensions import TypeVar

from . import _utils, messages as _messages

if TYPE_CHECKING:
    from .models import Model
    from .result import Usage

AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
"""Type variable for agent dependencies."""


@dataclasses.dataclass(repr=False)
class RunContext(Generic[AgentDepsT]):
    """Information about the current call."""

    deps: AgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: Usage
    """LLM usage associated with the run."""
    prompt: str | Sequence[_messages.UserContent] | None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries so far."""
    run_step: int = 0
    """The current step in the run."""

    def replace_with(
        self,
        retry: int | None = None,
        tool_name: str | None | _utils.Unset = _utils.UNSET,
    ) -> RunContext[AgentDepsT]:
        # Create a new `RunContext` a new `retry` value and `tool_name`.
        kwargs = {}
        if retry is not None:
            kwargs['retry'] = retry
        if tool_name is not _utils.UNSET:  # pragma: no branch
            kwargs['tool_name'] = tool_name
        return dataclasses.replace(self, **kwargs)

    __repr__ = _utils.dataclasses_no_defaults_repr
