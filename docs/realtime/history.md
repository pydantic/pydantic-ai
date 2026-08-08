# History and handoff

A realtime session builds the same [`ModelMessage`][pydantic_ai.messages.ModelMessage] history as a
standard agent run — see [Messages and chat history](../message-history.md). Voice conversations
can start from earlier text or voice history, continue in a new realtime session, or hand off to a
text model for summarization, extraction, and follow-up.

Spoken turns use [`SpeechPart`][pydantic_ai.messages.SpeechPart]; text, images, and tools retain the
ordinary [`ModelRequest`][pydantic_ai.messages.ModelRequest] and
[`ModelResponse`][pydantic_ai.messages.ModelResponse] shape.

## Reading session history

The session exposes copy-on-read snapshots:

| Method | Returns |
| --- | --- |
| [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] | Seeded history plus messages recorded during this session. |
| [`new_messages()`][pydantic_ai.realtime.RealtimeSession.new_messages] | Only messages recorded during this session. |

## Seeding a session

Pass `message_history=` to seed a new session. Replayable text, speech transcripts, thinking text,
tool rounds, and supported images are projected into provider conversation items.
[History processors](../message-history.md#processing-message-history) do not run at seeding; see
[Capabilities and hooks](capabilities.md#seeded-history-is-not-processed).

```python
from pydantic_ai import Agent

voice = Agent(instructions='You are a helpful voice assistant.')


async def main(prior_history=()):
    async with voice.realtime(
        'openai:gpt-realtime',
        message_history=prior_history,
    ).session() as session:
        await session.send('Continue where we left off.')
```

Providers replay native function calls where their protocol permits. Gemini represents seeded tool
calls and results as readable text because Live cannot put function parts in seeded turns. Thinking
signatures and provider-native execution metadata are omitted because they belong to the session
that produced them.

Content-less speech parts are skipped because they carry no replayable content. Unsupported content
raises [`UserError`][pydantic_ai.exceptions.UserError] instead of being silently dropped. Video,
documents, uploaded-file references, and model-generated files cannot be seeded.

Speech transcripts are preferred over retained audio. OpenAI and Azure OpenAI can replay retained
user audio when no transcript exists; Gemini and xAI cannot. Assistant speech always needs a
transcript for seeding. Check `supports_session_seeding`, `supports_seeding_images`, and
`supports_seeding_audio` on the
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] (see
[Provider support](overview.md#provider-support) for how profiles resolve) before constructing
portable flows.

## Handing off to a text agent

Pass the session snapshot directly to [`Agent.run()`][pydantic_ai.agent.AbstractAgent.run]:

```python
from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

voice = Agent(instructions='You are a helpful voice assistant.')
notetaker = Agent('openai:gpt-5', instructions='Summarize the conversation as bullet points.')


async def main():
    async with voice.realtime('openai:gpt-realtime').session() as session:
        await session.send('Please remind me to book a train tomorrow.')
        async for event in session:
            if isinstance(event, RealtimeTurnCompleteEvent):
                break

    result = await notetaker.run(
        'Summarize the conversation.', message_history=session.all_messages()
    )
    print(result.output)
    #> - Book a train tomorrow.
```

Retained user audio is forwarded to standard models whose profile supports audio input; other models
receive the transcript. Assistant speech is always handed off as transcript text. Interrupted
assistant turns receive a readable interruption note when Pydantic AI prepares the text-model
request.

For structured work that must finish while the call remains open, expose a delegated text agent as
a [realtime function tool](tools.md#delegating-work-during-a-call).

## Retaining audio

By default, only transcripts are retained and `SpeechPart.audio` is `None`. Pass `audio_retention=`
to [`session()`][pydantic_ai.agent.AgentRealtime.session] to retain finalized WAV audio in history:

| [`AudioRetention`][pydantic_ai.realtime.AudioRetention] value | Retains |
| --- | --- |
| `'transcript_only'` (default) | Transcripts only |
| `'input_audio'` | User audio |
| `'output_audio'` | Model audio |
| `'all'` | Both sides' audio |

Retention affects history only. Live input and output remain raw PCM16; finalized retained audio is
wrapped in a WAV container.

Input retention follows provider-reported boundaries rather than locally trimming speech. OpenAI,
Azure OpenAI, and xAI normally retain microphone input between reported speech-end boundaries.
Gemini does not report those boundaries, so it retains input between response completions. Either
form can include silence or other microphone input and should not be treated as a precisely trimmed
utterance.

## Retaining images

Pass `retain_images_every_n=` and `retain_images_max=` to
[`session()`][pydantic_ai.agent.AgentRealtime.session] to bound how many images stay in local
history:

```python
from pydantic_ai import Agent

agent = Agent()


async def main():
    async with agent.realtime('openai:gpt-realtime').session(
        retain_images_every_n=10, retain_images_max=25
    ):
        ...
```

Images sent through [`send()`][pydantic_ai.realtime.RealtimeSession.send] are recorded by default.
`retain_images_every_n=N` keeps the first image and then one of every `N`; the provider still receives
every frame. `retain_images_max` defaults to `100` and evicts the oldest retained image when the cap
is reached. Set it to `0` to retain none or `None` to remove the bound.

Sampling controls history growth rate; the maximum provides the actual memory bound. Streaming
images continuously approximates live video — the [camera example](../examples/realtime-camera.md)
sends one frame per second — so for camera and screen streams, use both deliberately.

## Transcription and history edge cases

Input transcription defaults to `'auto'`; see [Input transcription](audio.md#input-transcription)
and each provider page for configuration. With transcription disabled:

- retained input audio creates an audio-only user `SpeechPart`;
- without input retention, the session records a content-less user `SpeechPart`;
- content-less parts preserve the local turn boundary but contribute no words to a text handoff and
  are skipped when seeding another realtime session;
- transcript-less assistant audio cannot be handed off or seeded on any provider.

If a future session must be portable across providers or models, retain transcripts. Filter or
transform unsupported parts before passing `message_history`; history-processing capabilities do
not run during realtime seeding.
