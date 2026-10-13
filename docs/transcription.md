---
title: Transcription
description: "Transcribe recorded speech to text with Pydantic AI's Transcriber API, using OpenAI's Whisper and GPT-4o transcription models or any OpenAI-compatible provider."
---

# Transcription

Pydantic AI provides a provider-agnostic API for batch speech-to-text: one request per audio clip, audio in and text out.
Use [`Transcriber`][pydantic_ai.transcription.Transcriber] when your application needs the words in a recording, for
example to answer a voice message with an agent whose model doesn't take audio input.

For live, bidirectional voice conversations with an agent, use [realtime](realtime/overview.md) instead.

## Quick Start

Install the optional group for the provider you want to use, for example OpenAI:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

Set its API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

Then pass a provider-prefixed model name to [`Transcriber`][pydantic_ai.transcription.Transcriber], and call
[`transcribe()`][pydantic_ai.transcription.Transcriber.transcribe] with the audio:

```python {title="transcription_quickstart.py"}
from pydantic_ai import AudioUrl, Transcriber

transcriber = Transcriber('openai:gpt-4o-transcribe')


async def main():
    result = await transcriber.transcribe(AudioUrl('https://example.com/voice-message.ogg'))
    print(result.text)
    #> Hi, I'd like to book a table for two tonight at seven.
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

[`transcribe_sync()`][pydantic_ai.transcription.Transcriber.transcribe_sync] provides the same interface for synchronous code.

## Audio Input

The audio can be an [`AudioUrl`][pydantic_ai.messages.AudioUrl], which Pydantic AI downloads before sending it on, or
[`BinaryContent`][pydantic_ai.messages.BinaryContent] holding the audio bytes, such as a voice message received by your
application or a file read with [`BinaryContent.from_path()`][pydantic_ai.messages.BinaryContent.from_path]. The media
type tells the provider how the audio is encoded, so it must be an audio type, or a video type for containers like MP4
and WebM whose audio track is transcribed.

OpenAI accepts FLAC, MP3, MP4, MPEG, M4A, Ogg, WAV, and WebM, up to 25 MB per file. Any other media type raises
[`UserError`][pydantic_ai.exceptions.UserError] before a request is made. Audio URLs downloaded by Pydantic AI are
limited to 50 MiB.

## Using Transcription with an Agent

A transcript is plain text, so it can be passed to any agent, including one whose model doesn't take audio input:

```python {title="voice_message_agent.py"}
from pydantic_ai import Agent, BinaryContent, Transcriber

transcriber = Transcriber('openai:gpt-4o-mini-transcribe', settings={'language': 'en'})
agent = Agent('openai:gpt-6-sol', instructions='You take restaurant bookings.')


async def handle_voice_message(audio: bytes) -> str:
    transcript = await transcriber.transcribe(BinaryContent(data=audio, media_type='audio/ogg'))
    result = await agent.run(transcript.text)
    return result.output
```

## Providers

A `'provider:model-name'` string configures the provider from its usual environment variables.

### OpenAI

[`OpenAITranscriptionModel`][pydantic_ai.transcription.openai.OpenAITranscriptionModel] works with OpenAI's
[speech-to-text models](https://developers.openai.com/api/docs/guides/speech-to-text), such as `gpt-4o-transcribe`,
`gpt-4o-mini-transcribe`, `gpt-transcribe`, and `whisper-1`.
[`KnownTranscriptionModelName`][pydantic_ai.transcription.KnownTranscriptionModelName] carries the names Pydantic AI
recognizes for autocompletion; any other name is passed through unchanged.

#### Install

To use OpenAI transcription models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the
`openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

#### Configuration

Set `OPENAI_API_KEY` and use the `openai:` prefix, or build the model yourself to configure the provider:

```python {title="openai_transcription_model.py"}
from pydantic_ai import Transcriber
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.transcription.openai import OpenAITranscriptionModel

model = OpenAITranscriptionModel(
    'gpt-4o-transcribe',
    provider=OpenAIProvider(api_key='your-api-key'),
)
transcriber = Transcriber(model)
```

#### OpenAI-Compatible Providers {#openai-compatible}

Since [`OpenAITranscriptionModel`][pydantic_ai.transcription.openai.OpenAITranscriptionModel] uses the same provider
system as [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], you can use it with any
[OpenAI-compatible provider](models/overview.md#openai-compatible-providers) that serves the transcriptions endpoint,
like Groq's Whisper models:

```python {title="openai_compatible_transcription.py"}
from pydantic_ai import Transcriber
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.transcription.openai import OpenAITranscriptionModel

model = OpenAITranscriptionModel(
    'whisper-large-v3-turbo',
    provider=OpenAIProvider(
        base_url='https://api.groq.com/openai/v1',
        api_key='your-groq-api-key',
    ),
)
transcriber = Transcriber(model)
```

Providers with a dedicated provider class built on the OpenAI client, like
[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider], also take the shorthand syntax, such as
`Transcriber('azure:whisper')` with the name of your Azure deployment. Check that your provider serves the
transcription model you want to use: a provider that doesn't raises
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError].

## Settings

[`TranscriptionSettings`][pydantic_ai.transcription.TranscriptionSettings] holds the settings shared across providers.
Pass them to the [`Transcriber`][pydantic_ai.transcription.Transcriber] as defaults for every call, or to a single
[`transcribe()`][pydantic_ai.transcription.Transcriber.transcribe] call, which takes precedence:

```python {title="transcription_settings.py"}
from pydantic_ai import BinaryContent, Transcriber

# Default settings for all calls
transcriber = Transcriber('openai:gpt-4o-transcribe', settings={'language': 'de'})


async def transcribe_english(audio: BinaryContent) -> str:
    # Override the language for this call
    result = await transcriber.transcribe(audio, settings={'language': 'en'})
    return result.text
```

`language` is the most important setting to get right: transcription models detect the spoken language when it isn't
given, but can guess wrong on short or accented clips, and transcribe or translate into the wrong language as a result.
Pass the [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) code whenever you know it.

### OpenAI-Specific Settings

[`OpenAITranscriptionSettings`][pydantic_ai.transcription.openai.OpenAITranscriptionSettings] adds `openai_prompt`, to
guide the spelling of names and domain terms, and `openai_response_format`. Set the latter to `'verbose_json'` to get
timestamped [`segments`][pydantic_ai.transcription.TranscriptionResult.segments] from `whisper-1`, the only OpenAI
model that supports it:

```python {title="openai_transcription_segments.py"}
from pydantic_ai import BinaryContent, Transcriber
from pydantic_ai.transcription.openai import OpenAITranscriptionSettings

transcriber = Transcriber('openai:whisper-1')


async def print_subtitles(audio: BinaryContent) -> None:
    result = await transcriber.transcribe(
        audio,
        settings=OpenAITranscriptionSettings(
            openai_prompt='Pydantic, Logfire',
            openai_response_format='verbose_json',
        ),
    )
    for segment in result.segments:
        print(f'[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}')
```

## Results and Usage

[`TranscriptionResult`][pydantic_ai.transcription.TranscriptionResult] carries the transcript `text`, any timestamped
`segments`, request usage, model and provider identity, and any provider-specific response details, such as the language
`whisper-1` detected.

Transcription models are billed either by token, like `gpt-4o-transcribe`, which reports `input_tokens`,
`output_tokens` and `input_audio_tokens` in [`usage`][pydantic_ai.transcription.TranscriptionResult.usage], or by audio
duration, like `whisper-1`, which reports `audio_seconds`.
[`cost()`][pydantic_ai.transcription.TranscriptionResult.cost] uses
[`genai-prices`](https://github.com/pydantic/genai-prices) to price either:

```python {title="transcription_result.py"}
from pydantic_ai import AudioUrl, Transcriber

transcriber = Transcriber('openai:gpt-4o-transcribe')


async def main():
    result = await transcriber.transcribe(AudioUrl('https://example.com/voice-message.ogg'))
    print(result.model_name)
    #> gpt-4o-transcribe
    print(result.usage.output_tokens)
    #> 13
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

## Error Handling

Transcription raises the same exceptions as the rest of Pydantic AI:

- [`UserError`][pydantic_ai.exceptions.UserError] when the request cannot be built, such as for audio in a format the
  provider doesn't accept.
- [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] for 4xx and 5xx provider responses, and
  [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] when the provider cannot be reached.

## Instrumentation

Enable OpenTelemetry instrumentation for one transcriber or for all transcribers:

```python {title="instrumented_transcription.py"}
import logfire

from pydantic_ai import Transcriber

logfire.configure()

transcriber = Transcriber('openai:gpt-4o-transcribe', instrument=True)

# Or instrument all transcribers globally
Transcriber.instrument_all()
```

Each call opens a span named `transcription {model}` carrying `gen_ai.operation.name='transcription'` and
`gen_ai.output.type='text'`, with the model identity, usage, cost, and the number of segments. `transcription` is a custom
operation name: the OpenTelemetry GenAI conventions enumerate no value for speech-to-text. The transcript and the
`openai_prompt` setting are recorded unless
[`include_content`][pydantic_ai.models.instrumented.InstrumentationSettings] is disabled. The audio itself, audio
URLs, and the `extra_headers` and `extra_body` request escape hatches are never recorded.

See the [Debugging and Monitoring guide](logfire.md) for more details on using Logfire with Pydantic AI.

## Testing

Use [`TestTranscriptionModel`][pydantic_ai.transcription.TestTranscriptionModel] for deterministic tests without API
calls:

```python {title="test_transcription.py"}
from pydantic_ai import BinaryContent, Transcriber
from pydantic_ai.transcription import TestTranscriptionModel


async def test_voice_message_workflow():
    transcriber = Transcriber('openai:gpt-4o-transcribe')
    test_model = TestTranscriptionModel(text='Book a table for two.')

    with transcriber.override(model=test_model):
        audio = BinaryContent(data=b'...', media_type='audio/ogg')
        result = await transcriber.transcribe(audio, settings={'language': 'en'})

        assert result.text == 'Book a table for two.'

        # Check what audio and settings were used
        assert test_model.last_audio == audio
        assert test_model.last_settings == {'language': 'en'}
```

Setting [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] to `False` also blocks transcription requests,
so a transcriber you forgot to override raises instead of quietly calling the provider.

## Building Custom Transcription Models

To integrate a speech-to-text provider Pydantic AI does not ship, subclass
[`TranscriptionModel`][pydantic_ai.transcription.TranscriptionModel]:

```python {title="custom_transcription_model.py"}
from pydantic_ai.transcription import (
    TranscriptionInput,
    TranscriptionModel,
    TranscriptionResult,
    TranscriptionSettings,
)


class MyCustomTranscriptionModel(TranscriptionModel):
    @property
    def model_name(self) -> str:
        return 'my-custom-model'

    @property
    def system(self) -> str:
        return 'my-provider'

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        settings = self.prepare_transcribe(audio, settings)

        # Call your transcription API here
        text = '...'  # Placeholder

        return TranscriptionResult(text, model_name=self.model_name, provider_name=self.system)
```

`prepare_transcribe()` validates the audio input and merges the model's own default settings under the ones passed in,
so a subclass gets the same portable behavior as the built-in adapters.

Use [`WrapperTranscriptionModel`][pydantic_ai.transcription.WrapperTranscriptionModel] if you want to wrap an existing
model to add custom behavior like caching or logging.
