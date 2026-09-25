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

## Tool calls in history

A tool round is recorded as in a standard run: a
[`ModelResponse`][pydantic_ai.messages.ModelResponse] with the
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart], then a
[`ModelRequest`][pydantic_ai.messages.ModelRequest] with its
[`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart]. Realtime tools
[run in the background](tools.md#concurrent-tool-execution), so the conversation can move on before
a result arrives. History still records each result directly after its call, because request-response
APIs such as OpenAI Chat Completions and Anthropic reject a tool call whose result isn't in the next
message:

- During a Gemini [asynchronous tool call](gemini.md#asynchronous-tool-calls), the model keeps
  talking in the turn that called the tool. The calling `ModelResponse` stays open, so that speech is
  recorded in it after the `ToolCallPart` and before the result, where it happened. It still hands
  off to every provider. The response is recorded, and never changed afterwards, as soon as the
  result must follow it or the model's turn ends. Speech still in progress when the result goes out
  is split: what came after is recorded as the next response. A user turn that starts while the model
  is still talking follows the response, as it does any response in progress. With several asynchronous calls in one response, their
  results are recorded together after it, so speech between two of them follows both.
- A user turn during the tool run can't keep its real position. Neither can the reply to it or any
  later response. The result is recorded directly after its call, ahead of all of them.
  [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] streams in the real
  completion order, and the `ToolReturnPart`'s `timestamp` records when the tool finished.

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
| OpenAI GPT-Live | Reported by Live |
| OpenAI and Azure OpenAI Realtime, Gemini Live | Latest response's `total_tokens` over the context window, when the window is known |
| xAI Grok Voice | `None`: a response's usage counts only the input it added, not the whole conversation |

The value is `None` until it can be calculated, and it can go down during a session: providers
compact or truncate the conversation server-side as it grows, and none of them report when that
happens. To control how the provider manages a long conversation, use
[`openai_truncation`](openai.md#settings) on OpenAI Realtime and Azure OpenAI or
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
even when they arrive after that turn's response or overlap the following turn. A turn the user
starts while the model is still answering, whether they [barge in](turns.md#barge-in) or push to talk
over it, is recorded after that answer. Such a turn joins history once the provider ends the answer it
cut off, or after a few seconds if the provider never does. In that fallback the turn is recorded where
history stands, so it lands before the answer it interrupted, and ahead of anything sent with
[`send()`][pydantic_ai.realtime.RealtimeSession.send] while that answer was still in flight. If a reported speech
segment never receives a transcript, the session still records its retained audio or a content-less
`SpeechPart` when the session closes.

With transcription disabled:

- retained input audio creates an audio-only user `SpeechPart`;
- without input retention, the session records a content-less user `SpeechPart`;
- on providers that report speech boundaries (OpenAI, Azure, and xAI with server VAD), each turn is
  the speech between them: the silence an always-on microphone streams between utterances is no turn.
  Gemini Live reports none, so a turn there runs to the response that answers it, and audio sent after
  the last response is recorded as one more turn when the session closes;
- with [push-to-talk](turns.md#push-to-talk), each `commit_audio()` after sending audio is a turn; a
  commit with no audio since the last one records nothing;
- content-less parts preserve the local turn boundary but contribute no words to a text handoff and
  are skipped when seeding another realtime session;
- transcript-less assistant audio cannot be handed off or seeded on any provider.

If a future session must be portable across providers or models, retain transcripts. Filter or
transform unsupported parts before passing `message_history`; history-processing capabilities do
not run during realtime seeding.
