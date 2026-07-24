# Azure Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure's realtime
speech-to-speech with the server-side Pydantic AI agent loop — either the **Azure OpenAI GA** protocol
(the default) or **Azure AI Voice Live** (opt-in). See the [realtime overview](index.md).

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

!!! note "Input transcription needs a deployed transcription model"
    Azure resolves the input-transcription model against your resource's own **deployments**, not
    OpenAI's hosted models. The default (`gpt-realtime-whisper`) is not a deployment, so on a resource
    without a transcription deployment, input transcription fails with `DeploymentNotFound`. Since that's a
    misconfiguration that fails every turn (silently dropping spoken user turns), the session **raises**
    rather than surfacing a recoverable event. Deploy a transcription model (e.g. `whisper`) and set it via
    `input_transcription_model` on
    [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]. If you don't
    need transcripts, disable transcription with `input_transcription_model=None` and pass
    `audio_retention='input_audio'` so the spoken turn is still kept as audio.

## Browser WebRTC and Microsoft Entra ID

Azure OpenAI supports the same browser WebRTC flow as OpenAI — the audio flows browser ↔ Azure directly
while your backend runs a control-plane **sideband**. See [Browser / WebRTC](index.md#browser-webrtc)
for the topology, and use
[`AgentRealtime.answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] /
[`AgentRealtime.create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret] exactly as on OpenAI.
Azure relays the offer with `webrtcfilter=on`, which limits the events forwarded to the browser to a
safe subset so the session instructions stay on the server's control connection.

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

## Azure AI Voice Live

[Azure AI Voice Live](https://learn.microsoft.com/azure/ai-services/speech-service/voice-live) is
Microsoft's managed speech-to-speech service — a superset of the GA realtime API with extra session
options. It's the **same [`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel]**: opt in
with [`azure_voice_live=True`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live]
and the model targets the Voice Live endpoint and beta session protocol; GA stays the default.

Voice Live is a distinct Azure resource with its own credentials, so set `AZURE_VOICELIVE_ENDPOINT`,
`AZURE_VOICELIVE_API_KEY`, and `AZURE_VOICELIVE_API_VERSION` — [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider]
reads these as a fallback to the `AZURE_OPENAI_*` variables — or pass them to `AzureProvider` explicitly.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime.azure import AzureRealtimeModelSettings

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime(
        'azure:gpt-realtime',
        model_settings=AzureRealtimeModelSettings(azure_voice_live=True),
    ).session() as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

Voice-Live-only knobs use the `azure_voice_live_*` prefix (e.g.
[`azure_voice_live_turn_detection`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live_turn_detection]).

!!! note "Browser WebRTC is WebSocket-only for Voice Live"
    The [browser WebRTC](#browser-webrtc-and-microsoft-entra-id) flow above is for the GA Azure OpenAI
    realtime path. Voice Live negotiates WebRTC over its own WebSocket control channel instead, which
    isn't implemented yet, so `answer_webrtc_offer` / `create_client_secret` raise with
    `azure_voice_live=True`. Use a WebSocket session with Voice Live for now
    ([issue #6702](https://github.com/pydantic/pydantic-ai/issues/6702)).
