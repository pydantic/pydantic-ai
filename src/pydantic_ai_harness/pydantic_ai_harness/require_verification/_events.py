"""Events emitted by `RequireVerification`."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

REQUIRE_VERIFICATION_EVENTS = 'require_verification'


@dataclass(kw_only=True)
class CompletionUnverifiedEvent(CapabilityEvent, namespace=REQUIRE_VERIFICATION_EVENTS, dispatch='immediate'):
    """The run ended after code edits without passing verification, with every redirect used.

    Dispatched immediately: it is emitted as the run ends, after its event stream has closed.
    """

    attempts: int
    """How many times the run was sent back to verify before it ended."""

    failed_commands: list[str]
    """The verification commands whose latest run since the last edit failed."""
