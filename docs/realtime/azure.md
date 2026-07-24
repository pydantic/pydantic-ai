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

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime('azure:gpt-realtime').session() as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

You can also configure the resource explicitly:

```python {test="skip" lint="skip"}
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

provider = AzureProvider(
    azure_endpoint='https://my-resource.openai.azure.com',
    api_key='...',
)
model = AzureRealtimeModel('gpt-realtime', provider=provider)
```

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] reuses
[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] for endpoint and API key, and uses the
same settings/event protocol as
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]. The WebSocket transport
authenticates with the API key; browser [WebRTC signaling](#browser-webrtc-and-microsoft-entra-id) can
additionally use a Microsoft Entra ID token. Noise reduction, output
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
[`answer_webrtc_offer`][pydantic_ai.realtime.RealtimeModel.answer_webrtc_offer] /
[`create_client_secret`][pydantic_ai.realtime.RealtimeModel.create_client_secret] exactly as on OpenAI.
Azure relays the offer with `webrtcfilter=on`, which limits the events forwarded to the browser to a
safe subset so the session instructions stay on the server's control connection.

The signaling calls authenticate with the resource's API key by default. To use **Microsoft Entra ID**
instead — so no API key is involved in minting the browser's short-lived credential — pass a
`credential` (any [`azure.identity`](https://learn.microsoft.com/python/api/overview/azure/identity-readme)
credential, e.g. `DefaultAzureCredential`). The credential mints a bearer token for the Azure OpenAI
data plane (scope `https://ai.azure.com/.default`), which requires the **Cognitive Services User** role
on the resource:

```python {test="skip" lint="skip"}
from azure.identity import DefaultAzureCredential

from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

model = AzureRealtimeModel(
    'gpt-realtime',
    provider=AzureProvider(azure_endpoint='https://my-resource.openai.azure.com'),
    credential=DefaultAzureCredential(),
)
# `answer_webrtc_offer` / `create_client_secret` now authenticate with an Entra bearer token; the
# browser only ever receives the short-lived ephemeral secret, never the token or the API key.
```

## Azure AI Voice Live support is coming soon

Azure AI Voice Live support is coming soon.
