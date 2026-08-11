# Azure OpenAI Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure OpenAI's GA
realtime protocol with the server-side Pydantic AI agent loop. Start with the
[realtime quickstart](overview.md#quickstart) or [text-to-audio example](../examples/realtime-text-to-audio.md).

## Setup

Azure OpenAI realtime uses the OpenAI realtime stack, so install `pydantic-ai-slim` with the
`openai-realtime` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai-realtime]"
```

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` as for the
[Azure AI Foundry provider](../models/openai.md#azure-ai-foundry). Use the `azure:` prefix followed
by your Azure deployment name:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime('azure:my-realtime-deployment').session() as session:
        await session.send('Say hello.')

        async for part in session.stream_transcripts():
            print(f'{part.speaker}: {part.transcript}')
            #> assistant: Hello from the realtime assistant.
            if part.speaker == 'assistant':
                break  # keep listening in a real call; we stop after one reply
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

For explicit configuration, use
[`AzureProvider.for_realtime()`][pydantic_ai.providers.azure.AzureProvider.for_realtime]. It accepts
a bare resource endpoint or its `/openai/v1` form. The GA realtime protocol uses `/openai/v1/realtime`
and does not take an `api_version`. Requests authenticate with the resource API key by default, or
with a Microsoft Entra ID token when a `credential` is passed (see
[Browser WebRTC and Microsoft Entra ID](#browser-webrtc-and-microsoft-entra-id)).

## Model names

Pass the Azure **deployment name**, which is chosen when the model is deployed and need not match
the underlying model ID. Available realtime models and regions are documented in the
[Azure OpenAI realtime documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-quickstart).

## Settings

Azure uses
[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] — the
realtime counterpart of [model run settings](../agent.md#model-run-settings) — including the
[shared settings](overview.md#shared-settings) plus:

- `openai_voice` for the provider voice;
- `openai_input_noise_reduction` and `openai_output_speed`;
- `openai_turn_detection` for server or semantic VAD (see [turn detection](turns.md#automatic-turn-detection));
- `openai_truncation` for session context management.

See [OpenAI settings](openai.md#settings) for the common settings shape. Azure realtime does not
expose `temperature` through Pydantic AI.

### Input transcription deployment

Azure resolves the [`input_transcription_model`](audio.md#input-transcription) setting against
deployments in your resource. The default `'auto'` selects `gpt-realtime-whisper`; a resource
without a matching deployment emits a `DeploymentNotFound` transcription error on every turn.

Deploy a realtime-capable transcription model such as `gpt-realtime-whisper` or
`gpt-4o-transcribe`, then set `input_transcription_model` to that deployment name. A classic
`whisper` deployment is not accepted. Set the field to `None` to disable transcription and use
[`audio_retention='input_audio'`](history.md#retaining-audio) if the spoken turn must remain
available as audio.

## Browser WebRTC and Microsoft Entra ID

Azure OpenAI supports the same browser WebRTC flow as OpenAI — the audio flows browser ↔ Azure directly
while your backend runs a control-plane **sideband**. See [Connecting a frontend](deployment.md#browser-webrtc-server-sideband)
for the topology, and use
[`AgentRealtime.answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] /
[`AgentRealtime.create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret] exactly as on OpenAI.
Azure relays the offer with `webrtcfilter=on`, which limits the events forwarded to the browser to a
safe subset so the session instructions stay on the server's control connection.

!!! note "Capturing sideband transcripts needs a deployed transcription model"
    The server side of a WebRTC call never receives the user's audio (it flows browser ↔ Azure
    directly), so the only way to capture the *words* the user speaks is a transcription model — the
    `audio_retention='input_audio'` fallback can't apply (there's no audio to retain). Without one, the
    user's turns are still represented in history, but as content-less
    [`SpeechPart`][pydantic_ai.messages.SpeechPart]s. To capture what users say, deploy a transcription
    model on your Azure resource (the default `gpt-realtime-whisper` fails with `DeploymentNotFound`
    until you deploy it, or point `input_transcription_model` at a transcription deployment you have).

!!! note "The browser's filtered event stream differs from the raw protocol"
    `webrtcfilter=on` means the events Azure forwards over the browser's data channel are a privacy-safe
    subset: the browser sees `output_audio_buffer.started` / `output_audio_buffer.stopped` for
    speaking-state, not the raw `response.created` / `response.done`. A frontend that keys "assistant is
    speaking" or latency telemetry off `response.*` needs to map the `output_audio_buffer.*` events
    instead. This affects only client code reading the data channel directly; the server-side session's
    [event stream](events.md) is unaffected — verified live: the session receives the
    `output_audio_buffer.*` frames in full and reports them as
    [`RealtimeOutputSpeechStartEvent`][pydantic_ai.realtime.RealtimeOutputSpeechStartEvent] /
    [`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent], so a
    listening/speaking indicator can be driven from the server rather than reconstructed in the browser
    (see [Connecting a frontend](deployment.md#browser-webrtc-server-sideband)).

Azure requests authenticate with the resource's API key by default. To use **Microsoft Entra ID**
instead — so no API key is involved, e.g. when the resource is locked to managed identity — pass a
`credential` (any [`azure.identity`](https://learn.microsoft.com/python/api/overview/azure/identity-readme)
credential, e.g. `DefaultAzureCredential`). It authenticates **every** request to the resource — the
realtime WebSocket session and the WebRTC signaling — with a bearer token for the Azure OpenAI data
plane (scope `https://ai.azure.com/.default`), which requires the **Cognitive Services User** role on
the resource:

```python {test="skip"}
from azure.identity import DefaultAzureCredential

from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

model = AzureRealtimeModel(
    'gpt-realtime',
    # `entra_authenticated=True` so no resource key is required — a resource locked to managed
    # identity has none. Omit `provider=` entirely to take the endpoint from `AZURE_OPENAI_ENDPOINT`.
    provider=AzureProvider.for_realtime(
        azure_endpoint='https://my-resource.openai.azure.com', entra_authenticated=True
    ),
    credential=DefaultAzureCredential(),
)
# The realtime session, `answer_webrtc_offer`, and `create_client_secret` now authenticate with an Entra
# bearer token; the browser only ever receives the short-lived ephemeral secret, never it or the API key.
```

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16, 24 kHz input and output |
| Text output | Full feature support | Select with `output_modality='text'` |
| Image input | Full feature support | [Images](audio.md#images) provide context for the next turn |
| Manual turns | Full feature support | `turn_detection=False` plus [commit/create verbs](turns.md#push-to-talk) |
| Interruption/truncation | Full feature support | [`interrupt(played_ms=...)`](turns.md#barge-in) records the heard cutoff |
| Input transcription | Limited parameter support | Requires a [compatible transcription deployment](#input-transcription-deployment) in the Azure resource |
| Native tools | Unsupported | Configure [local fallbacks](tools.md#native-tools) for web capabilities |
| Usage | Full feature support | Token, audio, and cache breakdowns |
| Reconnection | Full feature support | Pydantic AI [replays completed local history](lifecycle.md#state-restoration); in-flight media is lost |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

## Provider-specific quirks

- A failed input transcription leaves the user turn represented as
  [retained audio](history.md#retaining-audio) when available, or as a content-less `SpeechPart`
  otherwise.
- This page covers Azure OpenAI Realtime only; Azure AI Voice Live support is coming in
  [#6642](https://github.com/pydantic/pydantic-ai/pull/6642).
