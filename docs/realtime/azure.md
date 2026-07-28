# Azure OpenAI Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure OpenAI's GA
realtime protocol with the server-side Pydantic AI agent loop. See the [realtime overview](index.md).

## Installation

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

## Configuration

Azure exposes the GA protocol at `/openai/v1/realtime`. Set `AZURE_OPENAI_ENDPOINT` and
`AZURE_OPENAI_API_KEY`, then use the `azure:` prefix:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime('azure:gpt-realtime').session() as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

You can also configure the resource explicitly. Use the endpoint's `/openai/v1` form — the realtime
protocol lives under the [v1 GA API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle),
so no `api_version` is involved (with a bare resource endpoint, `AzureProvider` would require the
`api_version` its general-purpose SDK client needs):

```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

provider = AzureProvider(
    azure_endpoint='https://my-resource.openai.azure.com/openai/v1',
    api_key='...',
)
model = AzureRealtimeModel('gpt-realtime', provider=provider)
```

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] reuses
[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] for endpoint and API key, and uses the
same settings/event protocol as
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]. Both the WebSocket transport
and browser [WebRTC signaling](#browser-webrtc-and-microsoft-entra-id) authenticate with the API key by
default, or with a Microsoft Entra ID token when you pass a `credential`. Noise reduction, output
speed, server/semantic VAD, and truncation use
[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]. Azure
realtime does not expose `temperature`. Input transcription defaults to `'auto'`; see
[Transcribing user input](index.md#transcribing-user-input).

!!! note "Capturing input transcripts needs a deployed transcription model"
    Azure resolves the input-transcription model against your resource's own **deployments**, not
    OpenAI's hosted models, so on a resource without a matching deployment input transcription fails
    with `DeploymentNotFound` on every turn — including for the default, `gpt-realtime-whisper`. The
    failed turn remains represented in history as retained audio when available, or as a content-less
    user [`SpeechPart`][pydantic_ai.messages.SpeechPart] otherwise, but its words are not captured.

    To capture transcripts, deploy a **realtime-capable** transcription model — `gpt-realtime-whisper`
    (which makes the default work as-is) or `gpt-4o-transcribe` — and point
    `input_transcription_model` on
    [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] at the
    deployment name. A classic `whisper` deployment is *not* accepted here and is rejected with the same
    `DeploymentNotFound`. If you don't need transcripts, disable transcription with
    `input_transcription_model=None`; pass `audio_retention='input_audio'` if the spoken turn should be
    kept as audio rather than a content-less part.

## Browser WebRTC and Microsoft Entra ID

Azure OpenAI supports the same browser WebRTC flow as OpenAI — the audio flows browser ↔ Azure directly
while your backend runs a control-plane **sideband**. See [Browser / WebRTC](index.md#browser-webrtc)
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
    [event stream](index.md#event-reference) is unaffected.

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

## Azure AI Voice Live support is coming soon

Azure AI Voice Live support is coming soon.
