from abc import ABC

from typing_extensions import TypeVar

__all__ = 'AbstractState', 'StateT'


class AbstractState(ABC):
    """Abstract class for a state object."""

    def __init__(self):
        pass

    # TODO serializing and deserialize state


StateT = TypeVar('StateT', None, AbstractState, default=None)
