Realtime speech-to-speech models are great conversationalists, but they don't produce structured
output. This example shows the robust pattern: let the realtime model run the live conversation,
then hand its [message history](../message-history.md) to a normal
[`Agent.run()`][pydantic_ai.agent.AbstractAgent.run] with `output_type` to extract a typed result.

Because a [realtime session](../realtime/overview.md) records the *same*
[`ModelMessage`][pydantic_ai.messages.ModelMessage] history a text agent produces, the handoff is
just passing [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] along —
realtime and non-realtime runs are peers that interoperate through message history.

Demonstrates:

- [realtime sessions](../realtime/overview.md)
- [structured output](../output.md) via a text-agent handoff
- [message history](../message-history.md) shared across realtime and non-realtime runs

The example models a short support call: a caller describes a problem to the realtime voice agent,
then the accumulated conversation is handed to a text agent that distills it into a typed
`SupportTicket`. The caller's side is driven with text turns so the example runs without a
microphone — a real app would stream microphone audio with
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] instead (see the
[voice assistant example](./realtime-voice.md)).

The handoff only runs after every scripted caller turn receives a
[`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent]. If the realtime connection ends
early, the example raises an error rather than creating a ticket from a partial call.

## Running the Example

Both the realtime `gpt-realtime` model and the text triage agent run on OpenAI, so you'll need an
OpenAI API key set via `OPENAI_API_KEY`.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.realtime_handoff
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/realtime_handoff.py"}```
