# Azure OpenAI Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure OpenAI's GA
realtime protocol with the server-side Pydantic AI agent loop. See the [realtime overview](index.md).

## Installation

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

## Configuration

Azure exposes the GA protocol at `/openai/v1/realtime`. Set `AZURE_OPENAI_ENDPOINT` and
`AZURE_OPENAI_API_KEY`, then use the `azure:` prefix followed by the name of your Azure realtime
deployment. Deployment names are chosen when deploying the model and do not need to match the
underlying model name:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime('azure:my-realtime-deployment').session() as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

You can also configure the resource explicitly with
[`AzureProvider.for_realtime()`][pydantic_ai.providers.azure.AzureProvider.for_realtime]. It accepts
either a bare resource endpoint or its `/openai/v1` form. The realtime protocol lives under the
[v1 GA API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle), so no
`api_version` is involved:

```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

provider = AzureProvider.for_realtime(
    azure_endpoint='https://my-resource.openai.azure.com',
    api_key='...',
)
model = AzureRealtimeModel('my-realtime-deployment', provider=provider)
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

## Azure AI Voice Live support is coming soon

Azure AI Voice Live support is coming soon.
