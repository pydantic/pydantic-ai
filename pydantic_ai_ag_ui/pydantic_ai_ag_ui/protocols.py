"""Protocols for the AG-UI to PydanticAI agent adapter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ag_ui.core import State


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs."""

    def set_state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            ValidationError: If `state` does not match the expected model.
        """
        ...
