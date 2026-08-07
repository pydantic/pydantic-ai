# Azure OpenAI Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure OpenAI's GA
realtime protocol with the server-side Pydantic AI agent loop. Start with the
[realtime quickstart](index.md#quickstart) or [text-to-audio example](../examples/realtime-text-to-audio.md).

## Setup

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. Use the `azure:` prefix followed by your
Azure deployment name:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime('azure:my-realtime-deployment').session() as session:
        await session.send('Say hello.')
```

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
[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings], including
the shared settings plus:

- `openai_voice` for the provider voice;
- `openai_input_noise_reduction` and `openai_output_speed`;
- `openai_turn_detection` for server or semantic VAD;
- `openai_truncation` for session context management.

See [OpenAI settings](openai.md#settings) for the common settings shape. Azure realtime does not
expose `temperature` through Pydantic AI.

### Input transcription deployment

Azure resolves `input_transcription_model` against deployments in your resource. The default
`'auto'` selects `gpt-realtime-whisper`; a resource without a matching deployment emits a
`DeploymentNotFound` transcription error on every turn.

Deploy a realtime-capable transcription model such as `gpt-realtime-whisper` or
`gpt-4o-transcribe`, then set `input_transcription_model` to that deployment name. A classic
`whisper` deployment is not accepted. Set the field to `None` to disable transcription and use
`audio_retention='input_audio'` if the spoken turn must remain available as audio.

## Browser WebRTC and Microsoft Entra ID

Azure OpenAI supports the same browser WebRTC flow as OpenAI — the audio flows browser ↔ Azure directly
while your backend runs a control-plane **sideband**. See [Browser / WebRTC](lifecycle.md#browser-webrtc)
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
    [event stream](audio.md#event-reference) is unaffected — verified live: the session receives the
    `output_audio_buffer.*` frames in full and reports them as
    [`RealtimeOutputSpeechStartEvent`][pydantic_ai.realtime.RealtimeOutputSpeechStartEvent] /
    [`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent], so a
    listening/speaking indicator can be driven from the server rather than reconstructed in the browser
    (see [Knowing when the model is speaking](lifecycle.md#knowing-when-the-model-is-speaking)).

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
    provider=AzureProvider(azure_endpoint='https://my-resource.openai.azure.com'),
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
| Image input | Full feature support | Images provide context for the next turn |
| Manual turns | Full feature support | `turn_detection=False` plus commit/create verbs |
| Interruption/truncation | Full feature support | `interrupt(played_ms=...)` records the heard cutoff |
| Input transcription | Limited parameter support | Requires a compatible transcription deployment in the Azure resource |
| Native tools | Unsupported | Configure local fallbacks for web capabilities |
| Usage | Full feature support | Token, audio, and cache breakdowns |
| Reconnection | Full feature support | Pydantic AI replays completed local history; in-flight media is lost |

## Gateway

Azure OpenAI realtime gateway routing is not currently exposed. Connect through the `azure:` model
prefix or an explicitly configured `AzureProvider`.

## Provider-specific quirks

- Azure uses the OpenAI event and settings protocol but deployment names and resource-scoped
  transcription models make setup different.
- A failed input transcription leaves the user turn represented as retained audio when available,
  or as a content-less `SpeechPart` otherwise.
- Azure AI Voice Live is not supported; this page covers Azure OpenAI Realtime only.
