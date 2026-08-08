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
and does not take an `api_version`. API-key authentication is supported; Microsoft Entra ID is not
supported for realtime connections.

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
