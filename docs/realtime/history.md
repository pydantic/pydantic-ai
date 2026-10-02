---
description: "Start Pydantic AI voice sessions from earlier text or voice history, continue them later, or hand the conversation to a text model for summaries and extraction."
---

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

Seeded tool calls and results are replayed as native function calls where the provider's protocol
permits, and as readable text where it doesn't. Thinking signatures and provider-native execution
metadata are omitted because they belong to the session that produced them.

Content-less speech parts are skipped because they carry no replayable content. Unsupported content
raises [`UserError`][pydantic_ai.exceptions.UserError] instead of being silently dropped. Video,
documents, uploaded-file references, and model-generated files cannot be seeded.

Speech transcripts are preferred over retained audio. Where no transcript exists, retained user
audio is replayed on models whose profile sets `supports_seeding_audio`, as long as it was recorded
at the model's input sample rate. Assistant speech always needs a transcript for seeding. Check
`supports_session_seeding`, `supports_seeding_images`, and `supports_seeding_audio` on the
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

## Context window

[`RealtimeSession.context_window_used`][pydantic_ai.realtime.RealtimeSession.context_window_used]
reports the fraction of the model's context window in use, and
[`RunContext.context_window_used`][pydantic_ai.tools.RunContext.context_window_used] returns the same
value inside a session's tools and hooks. Where the provider reports the fraction itself, the session
keeps the latest value; otherwise it is computed as in a standard run, from the latest response's
token usage over the model's
[`context_window`][pydantic_ai.realtime.RealtimeModelProfile.context_window]:

| Provider | Source |
| --- | --- |
| OpenAI | GPT-Live: reported by Live. gpt-realtime: latest response's `total_tokens` over the context window, when the window is known |
| Azure OpenAI, Gemini Live | Latest response's `total_tokens` over the context window, when the window is known |
| xAI Grok Voice | `None`: a response's usage counts only the input it added, not the whole conversation |

The value is `None` until it can be calculated, and it can go down during a session: providers
compact or truncate the conversation server-side as it grows, and none of them report when that
happens. To control how the provider manages a long conversation, use
[`openai_truncation`](openai.md#gpt-realtime-settings) on OpenAI gpt-realtime and Azure OpenAI or
[`google_context_compression`](gemini.md#settings) on Gemini. To carry a long conversation on
elsewhere, [hand it off to a text agent](#handing-off-to-a-text-agent) or seed a new session with a
summary.

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

These settings bound local history, not provider context: the provider still receives every frame.
Images sent through [`send()`][pydantic_ai.realtime.RealtimeSession.send] are recorded by default;
`retain_images_every_n=N` keeps the first image and then one of every `N`. `retain_images_max`
defaults to `100` and evicts the oldest retained image when the cap is reached. Set it to `0` to
retain none or `None` to remove the bound.

Sampling controls history growth rate; the maximum provides the actual memory bound. Streaming
images continuously approximates live video — the [camera example](../examples/realtime-camera.md)
sends one frame per second — so for camera and screen streams, use both deliberately.

## Transcription and history edge cases

Input transcription defaults to `'auto'`; see [Input transcription](audio.md#input-transcription)
and each provider page for configuration. Transcripts are recorded with the user turn they describe,
even when they arrive after that turn's response or overlap the following turn.

On OpenAI, Azure OpenAI, and xAI, history follows the order of the provider's own conversation:

- A spoken turn sits where the provider added it, and text, images, and tool results where they reached
  it. A turn the user started while the model was still answering, but which the provider only committed
  after that answer ended, is recorded after the answer.
- A message appears in `all_messages()` once everything before it is final. A reply waits for the
  transcript of the spoken turn before it, which can arrive after the reply itself is done, for up to five
  seconds after that; past that, the turn is recorded with the transcript it has so far.
- Nothing is inserted ahead of messages already returned, so each snapshot starts with the one before it.
  The one exception is a tool's return, which always directly follows the response that called it.
- Assistant messages from `gpt-realtime-2` models carry their `phase` (`'commentary'` on the way to a tool
  call, or `'final_answer'`) as `'phase'` in the part's `provider_details`, the way a standard OpenAI run
  records [text phases](../models/openai.md#text-phases).

On Gemini Live and GPT-Live, a turn the user starts while the model is still answering, whether they
[barge in](turns.md#barge-in) or push to talk over it, is recorded after that answer. Such a turn joins
history once the provider ends the answer it cut off, or after a few seconds if the provider never does.
In that fallback the turn is recorded where history stands, so it lands before the answer it interrupted,
and ahead of anything sent with [`send()`][pydantic_ai.realtime.RealtimeSession.send] while that answer was
still in flight.

If a reported speech segment never receives a transcript, the session still records its retained audio or
a content-less `SpeechPart`, at the latest when the session closes.

With transcription disabled:

- retained input audio creates an audio-only user `SpeechPart`;
- without input retention, the session records a content-less user `SpeechPart`;
- on providers that report speech boundaries (OpenAI, Azure, and xAI with server VAD), each turn is
  the speech between them: the silence an always-on microphone streams between utterances is no turn.
  Gemini Live reports none, so a turn there runs to the response that answers it, and audio sent after
  the last response is recorded as one more turn when the session closes;
- with [push-to-talk](turns.md#push-to-talk), each `commit_audio()` after sending audio is a turn once the
  provider has it (on xAI, commits made while a reply is still under way go out together after it, as one turn,
  and one still held back when the session closes records nothing); a commit with no audio since the last one
  records nothing;
- content-less parts preserve the local turn boundary but contribute no words to a text handoff and
  are skipped when seeding another realtime session;
- transcript-less assistant audio cannot be handed off or seeded on any provider.

If a future session must be portable across providers or models, retain transcripts. Filter or
transform unsupported parts before passing `message_history`; history-processing capabilities do
not run during realtime seeding.
