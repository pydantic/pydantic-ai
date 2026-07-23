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

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime_session(model='azure:gpt-realtime') as session:
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
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]. API-key authentication is
supported; Microsoft Entra ID is not supported for realtime connections. Noise reduction, output
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
    async with agent.realtime_session(
        model='azure:gpt-realtime',
        model_settings=AzureRealtimeModelSettings(azure_voice_live=True),
    ) as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

Voice-Live-only knobs use the `azure_voice_live_*` prefix (e.g.
[`azure_voice_live_turn_detection`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live_turn_detection]).
