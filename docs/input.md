# Image, Audio, Video & Document Input

Some LLMs are now capable of understanding audio, video, image and document content.

## Image Input

!!! info
    Some models do not support image input. Please check the model's documentation to confirm whether it supports image input.

If you have a direct URL for the image, you can use [`ImageUrl`][pydantic_ai.ImageUrl]:

```py {title="image_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-5')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://iili.io/3Hs4FMg.png'),
    ]
)
print(result.output)
#> This is the logo for Pydantic, a data validation and settings management library in Python.
```

If you have the image locally, you can also use [`BinaryContent`][pydantic_ai.BinaryContent]:

```py {title="local_image_input.py" test="skip" lint="skip"}
import httpx

from pydantic_ai import Agent, BinaryContent

image_response = httpx.get('https://iili.io/3Hs4FMg.png')  # Pydantic logo

agent = Agent(model='openai:gpt-5')
result = agent.run_sync(
    [
        'What company is this logo from?',
        BinaryContent(data=image_response.content, media_type='image/png'),  # (1)!
    ]
)
print(result.output)
#> This is the logo for Pydantic, a data validation and settings management library in Python.
```

1. To ensure the example is runnable we download this image from the web, but you can also use `Path().read_bytes()` to read a local file's contents.

## Audio Input

!!! info
    Some models do not support audio input. Please check the model's documentation to confirm whether it supports audio input.

You can provide audio input using either [`AudioUrl`][pydantic_ai.AudioUrl] or [`BinaryContent`][pydantic_ai.BinaryContent]. The process is analogous to the examples above.

## Video Input

!!! info
    Some models do not support video input. Please check the model's documentation to confirm whether it supports video input.

You can provide video input using either [`VideoUrl`][pydantic_ai.VideoUrl] or [`BinaryContent`][pydantic_ai.BinaryContent]. The process is analogous to the examples above.

## Document Input

!!! info
    Some models do not support document input. Please check the model's documentation to confirm whether it supports document input.

You can provide document input using either [`DocumentUrl`][pydantic_ai.DocumentUrl] or [`BinaryContent`][pydantic_ai.BinaryContent]. The process is similar to the examples above.

If you have a direct URL for the document, you can use [`DocumentUrl`][pydantic_ai.DocumentUrl]:

```py {title="document_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, DocumentUrl

agent = Agent(model='anthropic:claude-sonnet-4-5')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
    ]
)
print(result.output)
#> This document is the technical report introducing Gemini 1.5, Google's latest large language model...
```

The supported document formats vary by model.

You can also use [`BinaryContent`][pydantic_ai.BinaryContent] to pass document data directly:

```py {title="binary_content_input.py" test="skip" lint="skip"}
from pathlib import Path
from pydantic_ai import Agent, BinaryContent

pdf_path = Path('document.pdf')
agent = Agent(model='anthropic:claude-sonnet-4-5')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf'),
    ]
)
print(result.output)
#> The document discusses...
```

## User-side download vs. direct file URL

When using one of `ImageUrl`, `AudioUrl`, `VideoUrl` or `DocumentUrl`, Pydantic AI will default to sending the URL to the model, so the file is downloaded on their side.

Support for file URLs varies depending on type and provider. Pydantic AI handles this as follows:

| Model | Supported URL types | Sends URL directly |
|-------|---------------------|-------------------|
| [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] | `ImageUrl`, `AudioUrl`, `DocumentUrl` | `ImageUrl` only |
| [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] | `ImageUrl`, `AudioUrl`, `DocumentUrl` | Yes |
| [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel] | `ImageUrl`, `DocumentUrl` | Yes, except `DocumentUrl` (`text/plain`) |
| [`GoogleModel`][pydantic_ai.models.google.GoogleModel] (Vertex) | All URL types | Yes |
| [`GoogleModel`][pydantic_ai.models.google.GoogleModel] (GLA) | All URL types | [YouTube](models/google.md#document-image-audio-and-video-input) and [Files API](https://ai.google.dev/gemini-api/docs/files) URLs only |
| [`MistralModel`][pydantic_ai.models.mistral.MistralModel] | `ImageUrl`, `DocumentUrl` (PDF) | Yes |
| [`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] | `ImageUrl`, `DocumentUrl`, `VideoUrl` | No, defaults to `force_download` |

### How to use private URLs or disallow local download

The `force_download` parameter controls how file URLs are handled:

| Value | Behavior |
|-------|----------|
| `None` (default) | Provider decides - downloads if needed, sends URL if supported |
| `True` | Always download content locally and send as bytes |
| `False` | Never download - always pass URL directly to the API |

Use `force_download=True` when a model API is unable to download a file (e.g., because the URL isn't publicly accessible, or may get blocked by crawling or access restrictions):

```py {title="force_download_true.py" test="skip" lint="skip"}
from pydantic_ai import ImageUrl, DocumentUrl

# Force local download when the API can't fetch the URL
ImageUrl(url='https://example.com/image.png', force_download=True)
DocumentUrl(url='https://example.com/doc.pdf', force_download=True)
```

Use `force_download=False` to pass URLs directly without any download attempt. This is useful when you have a proxy or middleware that intercepts requests:

```py {title="force_download_false.py" test="skip" lint="skip"}
from pydantic_ai import DocumentUrl

# Pass URL as-is (useful for proxies handling custom URL schemes)
DocumentUrl(url='file://local/path/to/doc.pdf', force_download=False)
```

!!! note
    Setting `force_download=False` for providers or content types that don't support URL passthrough will result in an API error. For example, Bedrock doesn't support URLs at all, and Anthropic doesn't support URL passthrough for `text/plain` documents. (As of Dec 2025)

## Uploaded Files

Some model providers like Google's Gemini API support [uploading files](https://ai.google.dev/gemini-api/docs/files). You can upload a file using the provider's client and passing the resulting URL as input:

```py {title="file_upload.py" test="skip"}
from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider()
file = provider.client.files.upload(file='pydantic-ai-logo.png')
assert file.uri is not None

agent = Agent(GoogleModel('gemini-2.5-flash', provider=provider))
result = agent.run_sync(
    [
        'What company is this logo from?',
        DocumentUrl(url=file.uri, media_type=file.mime_type),
    ]
)
print(result.output)
```
