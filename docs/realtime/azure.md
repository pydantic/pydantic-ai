# Azure OpenAI Realtime

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] connects to Azure OpenAI's GA
realtime protocol with the server-side Pydantic AI agent loop. Start with the
[realtime quickstart](overview.md#quickstart) or [text-to-audio example](../examples/realtime-text-to-audio.md).

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
and does not take an `api_version`. API-key authentication is supported; Microsoft Entra ID is not
supported for realtime connections.

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

- A failed input transcription leaves the user turn represented as retained audio when available,
  or as a content-less `SpeechPart` otherwise.
- Azure AI Voice Live is not supported; this page covers Azure OpenAI Realtime only.
