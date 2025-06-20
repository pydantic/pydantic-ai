from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ag_ui.core import State
from pydantic import BaseModel, ValidationError

from ._exceptions import InvalidStateError

StateT = TypeVar('StateT', bound=BaseModel, contravariant=True)


@dataclass(kw_only=True)
class StateDeps(Generic[StateT]):
    """Provides AG-UI state management."""

    state_type: type[StateT]
    state: StateT = field(init=False)

    def set_state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Implements the `StateHandler` protocol.

        Args:
            state: The run state, which should match the expected model type or be `None`.

        Raises:
            InvalidStateError: If `state` does not match the expected model and is not `None`.
        """
        if state is None:
            return

        try:
            self.state = self.state_type.model_validate(state)
        except ValidationError as e:  # pragma: no cover
            raise InvalidStateError from e
