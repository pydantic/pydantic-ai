# Image, Audio, Video & Document Input

Some LLMs are now capable of understanding audio, video, image and document content.

## Image Input

!!! info
    Some models do not support image input. Please check the model's documentation to confirm whether it supports image input.

If you have a direct URL for the image, you can use [`ImageUrl`][pydantic_ai.ImageUrl]:

```py {title="image_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://iili.io/3Hs4FMg.png'),
    ]
)
print(result.output)
# > This is the logo for Pydantic, a data validation and settings management library in Python.
```

If you have the image locally, you can also use [`BinaryContent`][pydantic_ai.BinaryContent]:

```py {title="local_image_input.py" test="skip" lint="skip"}
import httpx

from pydantic_ai import Agent, BinaryContent

image_response = httpx.get('https://iili.io/3Hs4FMg.png')  # Pydantic logo

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
    [
        'What company is this logo from?',
        BinaryContent(data=image_response.content, media_type='image/png'),  # (1)!
    ]
)
print(result.output)
# > This is the logo for Pydantic, a data validation and settings management library in Python.
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

agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
    ]
)
print(result.output)
# > This document is the technical report introducing Gemini 1.5, Google's latest large language model...
```

The supported document formats vary by model.

You can also use [`BinaryContent`][pydantic_ai.BinaryContent] to pass document data directly:

```py {title="binary_content_input.py" test="skip" lint="skip"}
from pathlib import Path
from pydantic_ai import Agent, BinaryContent

pdf_path = Path('document.pdf')
agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf'),
    ]
)
print(result.output)
# > The document discusses...
```

## User-side download vs. direct file URL

As a general rule, when you provide a URL using any of `ImageUrl`, `AudioUrl`, `VideoUrl` or `DocumentUrl`, Pydantic AI downloads the file content and then sends it as part of the API request.

The situation is different for certain models:

- [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel]: if you provide a PDF document via `DocumentUrl`, the URL is sent directly in the API request, so no download happens on the user side.

- [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] and [`GoogleModel`][pydantic_ai.models.google.GoogleModel] on Vertex AI: any URL provided using `ImageUrl`, `AudioUrl`, `VideoUrl`, or `DocumentUrl` is sent as-is in the API request and no data is downloaded beforehand.

  See the [Gemini API docs for Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#filedata) to learn more about supported URLs, formats and limitations:

  - Cloud Storage bucket URIs (with protocol `gs://`)
  - Public HTTP(S) URLs
  - Public YouTube video URL (maximum one URL per request)

  However, because of crawling restrictions, it may happen that Gemini can't access certain URLs. In that case, you can instruct Pydantic AI to download the file content and send that instead of the URL by setting the boolean flag `force_download` to `True`. This attribute is available on all objects that inherit from [`FileUrl`][pydantic_ai.messages.FileUrl].

- [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] and [`GoogleModel`][pydantic_ai.models.google.GoogleModel] on GLA: YouTube video URLs are sent directly in the request to the model.
