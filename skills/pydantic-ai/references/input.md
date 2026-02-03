# Input Reference (Images, Audio, Video, Documents)

Source: `pydantic_ai_slim/pydantic_ai/_parts.py`

## URL-Based Input

Pass multimedia content to agents using URL types:

```python {title="multimodal_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ImageUrl, AudioUrl, VideoUrl, DocumentUrl

agent = Agent('openai:gpt-5')

# Image input
result = agent.run_sync([
    'What company is this logo from?',
    ImageUrl(url='https://iili.io/3Hs4FMg.png'),
])

# Document input (PDF)
result = agent.run_sync([
    'Summarize this document.',
    DocumentUrl(url='https://example.com/doc.pdf'),
])

# Audio input
result = agent.run_sync([
    'Transcribe this audio.',
    AudioUrl(url='https://example.com/audio.mp3'),
])

# Video input
result = agent.run_sync([
    'Describe what happens in this video.',
    VideoUrl(url='https://example.com/video.mp4'),
])
```

## Binary Content (Local Files)

Use `BinaryContent` for local files or downloaded data:

```python {title="binary_content.py" test="skip" lint="skip"}
from pathlib import Path

from pydantic_ai import Agent, BinaryContent

agent = Agent('anthropic:claude-sonnet-4-5')

# Local PDF
pdf_data = Path('document.pdf').read_bytes()
result = agent.run_sync([
    'What is this document about?',
    BinaryContent(data=pdf_data, media_type='application/pdf'),
])

# Local image
image_data = Path('photo.png').read_bytes()
result = agent.run_sync([
    'Describe this image.',
    BinaryContent(data=image_data, media_type='image/png'),
])
```

## Force Download

When a provider cannot access a URL directly, force local download:

```python
from pydantic_ai import ImageUrl, DocumentUrl

# Force PydanticAI to download and send bytes instead of URL
ImageUrl(url='https://example.com/image.png', force_download=True)
DocumentUrl(url='https://example.com/doc.pdf', force_download=True)
```

## Provider Compatibility Matrix

| Model | Send URL Directly | Download & Send Bytes | Unsupported |
|-------|-------------------|----------------------|-------------|
| OpenAIChatModel | `ImageUrl` | `AudioUrl`, `DocumentUrl` | `VideoUrl` |
| OpenAIResponsesModel | `ImageUrl`, `AudioUrl`, `DocumentUrl` | — | `VideoUrl` |
| AnthropicModel | `ImageUrl`, `DocumentUrl` (PDF) | `DocumentUrl` (text) | `AudioUrl`, `VideoUrl` |
| GoogleModel (Vertex) | All URL types | — | — |
| GoogleModel (GLA) | YouTube, Files API | All other URLs | — |
| MistralModel | `ImageUrl`, `DocumentUrl` (PDF) | — | `AudioUrl`, `VideoUrl` |
| BedrockConverseModel | S3 URLs (`s3://`) | `ImageUrl`, `DocumentUrl`, `VideoUrl` | `AudioUrl` |

## Uploaded Files

Some providers support files uploaded to their platforms:

- **Google**: Use Files API for large files
- **Bedrock**: Use S3 URIs (`s3://bucket/key`)

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ImageUrl` | `pydantic_ai.ImageUrl` | URL to an image |
| `AudioUrl` | `pydantic_ai.AudioUrl` | URL to audio content |
| `VideoUrl` | `pydantic_ai.VideoUrl` | URL to video content |
| `DocumentUrl` | `pydantic_ai.DocumentUrl` | URL to a document (PDF, etc.) |
| `BinaryContent` | `pydantic_ai.BinaryContent` | Raw binary data with media type |

## See Also

- [tools-advanced.md](tools-advanced.md) — Tools can return multimodal content via `ToolReturn`
- [messages.md](messages.md) — Message parts include multimedia types
- [observability.md](observability.md) — Logfire debugging
