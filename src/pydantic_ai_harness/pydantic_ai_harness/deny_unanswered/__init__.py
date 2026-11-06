"""The `DenyUnanswered` capability: deny calls nobody approved or answered."""

from ._capability import DEFAULT_MESSAGE, DenyUnanswered, find_deny_unanswered

__all__ = ['DEFAULT_MESSAGE', 'DenyUnanswered', 'find_deny_unanswered']
