"""Hand a realtime voice conversation off to a text agent for a typed, structured result.

Realtime speech-to-speech models are great conversationalists, but they don't produce structured
output. The robust pattern is to let the realtime model run the live conversation, then hand its
message history to a normal `Agent.run(..., output_type=...)` to extract a typed result.

This works because a realtime session records the *same* `ModelMessage` history a text agent
produces: handing the conversation off is just passing `session.all_messages()` along. Realtime and
non-realtime runs are peers that interoperate through message history.

The example models a short support call: a caller describes a problem to the realtime voice agent,
then the accumulated conversation is handed to a text agent that distills it into a typed
`SupportTicket`. The caller's side is driven with text turns so the example runs without a
microphone — a real app would stream microphone audio with `session.send_audio(...)` instead (see
the `realtime_voice` example). Either way, the model replies with speech, and its transcripts land
in history for the handoff.

It needs an OpenAI API key set via `OPENAI_API_KEY`.

Run with:

    uv run -m pydantic_ai_examples.realtime_handoff
"""

from __future__ import annotations

import asyncio
from typing import Literal

import logfire
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.realtime import PartEndEvent, SpeechPart, TurnCompleteEvent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class SupportTicket(BaseModel):
    """The structured ticket distilled from the spoken support call."""

    summary: str
    category: Literal['hardware', 'software', 'billing', 'other']
    priority: Literal['low', 'medium', 'high']
    follow_up_questions: list[str]


# The realtime model runs the live conversation.
voice_agent = Agent(
    instructions='You are a friendly, concise phone support agent. Ask one question at a time.'
)

# A normal text agent turns the finished conversation into a typed result — something a realtime
# model can't do itself.
triage_agent = Agent(
    'openai:gpt-5.2',
    output_type=SupportTicket,
    instructions='Summarize the support call as a structured ticket.',
)

# What the caller "says" — each line is one spoken turn, driven as text so the example runs without
# a microphone.
CALLER_TURNS = [
    "Hi, my laptop won't charge anymore — the light doesn't come on when I plug it in.",
    'I already tried a different outlet and it still does nothing. I need it for a presentation tomorrow.',
]


async def main() -> None:
    async with voice_agent.realtime_session(
        model=OpenAIRealtimeModel('gpt-realtime')
    ) as session:
        # A session is consumed with a single event loop. We drive the caller's turns from inside it:
        # send the first line, then send the next one each time the model finishes a turn.
        remaining_turns = iter(CALLER_TURNS)
        first_turn = next(remaining_turns)
        print(f'caller: {first_turn}')
        # Sending text into an OpenAI realtime session asks the model to respond right away.
        await session.send(first_turn)

        async for event in session:
            match event:
                case PartEndEvent(
                    part=SpeechPart(speaker='assistant', transcript=transcript)
                ) if transcript:
                    print(f'agent: {transcript}')
                case TurnCompleteEvent():
                    next_turn = next(remaining_turns, None)
                    if next_turn is None:
                        break  # The caller has said everything; end the call.
                    print(f'caller: {next_turn}')
                    await session.send(next_turn)
                case _:
                    pass

        # The realtime session recorded ordinary `ModelMessage` history; hand it off to the text
        # agent, which can do the structured extraction the realtime model can't.
        handoff_history = session.all_messages()

    ticket = await triage_agent.run(
        'Create the support ticket for this call.', message_history=handoff_history
    )
    print(f'\nStructured ticket:\n{ticket.output.model_dump_json(indent=2)}')


if __name__ == '__main__':
    asyncio.run(main())
