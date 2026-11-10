"""Events emitted around a question to the user.

Both dispatch immediately: listeners run before the answerer is awaited and again before the tool
result reaches the model, so an observer that shows a "waiting for you" state sees the wait start
and end in step with the run rather than at the next stream drain.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent
from pydantic_ai_harness.ask_user._types import AskUserRequest, AskUserResponse

ASK_USER_EVENTS = 'ask_user'


@dataclass(kw_only=True)
class AskUserRequestedEvent(CapabilityEvent, namespace=ASK_USER_EVENTS, name='requested', dispatch='immediate'):
    """The model asked the user something; the answerer is about to be called, or the call is deferred."""

    request: AskUserRequest


@dataclass(kw_only=True)
class AskUserAnsweredEvent(CapabilityEvent, namespace=ASK_USER_EVENTS, name='answered', dispatch='immediate'):
    """The answerer returned, timed out, or raised; if it returned, `response.cancelled` says whether the user declined.

    Emitted before the response is checked against the request, so a listener waiting since the
    request event is released even when the answerer misbehaved and the run is about to fail.
    Not emitted for a deferred call, whose answer arrives in a later run from the host, or when
    the run itself is cancelled while the answerer waits: clear waiting state when the run ends.
    """

    request_id: str
    response: AskUserResponse
    timed_out: bool = False
    """The answerer ran past `AskUser.timeout` and was cancelled; `response` is then a cancelled one."""
    failed: bool = False
    """The answerer raised; `response` is then a cancelled one, and the error propagates from the tool call."""
