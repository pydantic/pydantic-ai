"""The `AskUser` capability: one tool, one answerer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.ask_user._toolset import AskUserToolset
from pydantic_ai_harness.ask_user._types import TOOL_NAME, Answerer, AnyAnswerer, ContextAnswerer

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    f'When the task is ambiguous and the answer is not in the workspace, call `{TOOL_NAME}` with '
    'concrete options rather than guessing or burying the question in prose. Batch related questions '
    'into one call. If the user declines, make a reasonable choice and say which one you made.'
)


@dataclass(kw_only=True)
class AskUser(AbstractCapability[AgentDepsT]):
    """Let the model ask the user multiple-choice questions mid-run.

    The capability owns the question schema and validation; the `answerer` owns the person.
    Whatever it is, a terminal menu, a web form, or a scripted function in a test, the run waits
    for it inside the tool call and the model gets the picked labels back. Pass `answerer=None` to
    defer each call instead, so the run pauses with `DeferredToolRequests` until the host answers.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import AskUser
    from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse


    async def pick_first(request: AskUserRequest) -> AskUserResponse:
        answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
        return AskUserResponse(answers=tuple(answers))


    agent = Agent('anthropic:claude-fable-5-1', capabilities=[AskUser(answerer=pick_first)])
    ```
    """

    answerer: AnyAnswerer[AgentDepsT] | None
    """Presents each `AskUserRequest` to the user and returns their `AskUserResponse`.

    An answerer with two required positional parameters receives the asking run's context before
    the request. If it also accepts just one, annotate the first as `RunContext[DepsT]` to select
    the context form. Request-only answerers receive just the request.

    `None` defers every call: the run ends with `DeferredToolRequests` output, and the host answers
    in a later run with `ask_user_result` in `DeferredToolResults.calls`.
    """

    timeout: float | None = None
    """Seconds to wait for the answerer before cancelling it and telling the model the user did not answer.

    Must be positive; `None` waits indefinitely. Needs an answerer: a deferred call has nothing to wait on.
    """

    id: str | None = 'ask_user'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    An agent has one `ask_user_question` tool, so two `AskUser` sharing this `id` are one answerer stated
    twice when they agree, and an error when they differ. To keep two, give each its own `id` and wrap them
    in `PrefixTools`, since their tool names are the same.
    """

    if TYPE_CHECKING:
        # Preserve request-only lambda inference while retaining the generated dataclass initializer.
        @overload
        def __init__(
            self,
            *,
            answerer: Answerer | None,
            timeout: float | None = None,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            answerer: ContextAnswerer[AgentDepsT],
            timeout: float | None = None,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        def __init__(
            self,
            *,
            answerer: AnyAnswerer[AgentDepsT] | None,
            timeout: float | None = None,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise UserError('`AskUser.timeout` must be positive, or `None` to wait indefinitely.')
        if self.answerer is None and self.timeout is not None:
            raise UserError('`AskUser.timeout` bounds the answerer, so it cannot be combined with `answerer=None`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one answerer stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on when to ask."""
        return _INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The `ask_user_question` tool bound to this capability's answerer."""
        return AskUserToolset[AgentDepsT](answerer=self.answerer, timeout=self.timeout)
