"""The lifecycle events a realtime connection yields, for implementing a realtime provider.

The [codec vocabulary][pydantic_ai.realtime.codec] says *what* the model said, but not which response it
belongs to, when that response began or ended, or which of the session's inputs it answers. Only the
connection sees the provider frames that settle those questions, so a
[`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] states them outright, interleaved with
its codec events in the [`RealtimeFrame`][pydantic_ai.realtime.lifecycle.RealtimeFrame]s it yields from
[`frames()`][pydantic_ai.realtime.codec.RealtimeConnection.frames]:

- a response is bracketed by exactly one [`ResponseStarted`][pydantic_ai.realtime.lifecycle.ResponseStarted]
  and one [`ResponseEnded`][pydantic_ai.realtime.lifecycle.ResponseEnded], and its content arrives only in
  between;
- [`ResponseStarted.answers`][pydantic_ai.realtime.lifecycle.ResponseStarted.answers] names the inputs whose
  reply the response is, which is what [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply]
  waits for, and [`AnswersRevised`][pydantic_ai.realtime.lifecycle.AnswersRevised] corrects them as the connection
  learns better: `claimed` while the response is under way, `released` until its reply (tool rounds included) is
  over;
- a spoken user turn starts with [`UserTurnStarted`][pydantic_ai.realtime.lifecycle.UserTurnStarted], joins the
  conversation with [`UserTurnEnded`][pydantic_ai.realtime.lifecycle.UserTurnEnded], and gets no more audio or
  transcript after [`UserTurnClosed`][pydantic_ai.realtime.lifecycle.UserTurnClosed] (which drops one that hadn't
  joined);
- [`InputAdded`][pydantic_ai.realtime.lifecycle.InputAdded] marks where an input joined the provider's
  conversation, so history can follow the provider's order;
- [`InputLost`][pydantic_ai.realtime.lifecycle.InputLost] and
  [`ResponseRequestRefused`][pydantic_ai.realtime.lifecycle.ResponseRequestRefused] settle the inputs no
  response will ever answer, and [`ToolCallsAbandoned`][pydantic_ai.realtime.lifecycle.ToolCallsAbandoned] the
  tool calls whose results none will;
- [`OutputItemDetails`][pydantic_ai.realtime.lifecycle.OutputItemDetails] keeps what the provider says about an
  output item on the part it makes, and [`OutputItemCut`][pydantic_ai.realtime.lifecycle.OutputItemCut] records
  that the listener heard only part of it, even after its response ended.

An input is identified by its position among every [`send()`][pydantic_ai.realtime.codec.RealtimeConnection.send]
call made on the connection, counting from `0`, as
[`InputRejected.input_index`][pydantic_ai.realtime.codec.InputRejected.input_index] identifies it.

A protocol that identifies its responses, turns, and inputs (as the OpenAI Realtime API does) is best
translated directly. One that identifies none of them, like Gemini Live and OpenAI GPT-Live, can have
[`InferredLifecycle`][pydantic_ai.realtime.lifecycle.InferredLifecycle] work them out from the order of what
the connection sends and receives.
"""

from __future__ import annotations as _annotations

from ._inferred_lifecycle import InferredLifecycle
from ._lifecycle import (
    AnswersBasis,
    AnswersRevised,
    InputAdded,
    InputId,
    InputLost,
    LifecycleEvent,
    OutputItemCut,
    OutputItemDetails,
    RealtimeFrame,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    ResponseStatus,
    StaleEvent,
    ToolCallsAbandoned,
    UserTurnClosed,
    UserTurnEnded,
    UserTurnStarted,
)

__all__ = (
    'AnswersBasis',
    'AnswersRevised',
    'InferredLifecycle',
    'InputAdded',
    'InputId',
    'InputLost',
    'LifecycleEvent',
    'OutputItemCut',
    'OutputItemDetails',
    'RealtimeFrame',
    'ResponseEnded',
    'ResponseRequestRefused',
    'ResponseStarted',
    'ResponseStatus',
    'StaleEvent',
    'ToolCallsAbandoned',
    'UserTurnClosed',
    'UserTurnEnded',
    'UserTurnStarted',
)
