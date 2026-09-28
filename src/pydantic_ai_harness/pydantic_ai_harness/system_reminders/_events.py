"""Events emitted by system reminders."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

SYSTEM_REMINDERS_EVENTS = 'system_reminders'


@dataclass(kw_only=True)
class ReminderFiredEvent(CapabilityEvent, namespace=SYSTEM_REMINDERS_EVENTS, name='fired'):
    """A rendered reminder was appended to a model request."""

    text: str
