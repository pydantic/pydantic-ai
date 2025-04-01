# Image, Audio, Document & File URL Input

Some LLMs are now capable of understanding both audio, image and document content.

## Image Input

!!! info
    Some models do not support image input. Please check the model's documentation to confirm whether it supports image input.

If you have a direct URL for the image, you can use [`ImageUrl`][pydantic_ai.ImageUrl]:

```py {title="main.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://iili.io/3Hs4FMg.png'),
    ]
)
print(result.data)
#> This is the logo for Pydantic, a data validation and settings management library in Python.
```

If you have the image locally, you can also use [`BinaryContent`][pydantic_ai.BinaryContent]:

```py {title="main.py" test="skip" lint="skip"}
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
print(result.data)
#> This is the logo for Pydantic, a data validation and settings management library in Python.
```

1. To ensure the example is runnable we download this image from the web, but you can also use `Path().read_bytes()` to read a local file's contents.

## Audio Input

!!! info
    Some models do not support audio input. Please check the model's documentation to confirm whether it supports audio input.

You can provide audio input using either [`AudioUrl`][pydantic_ai.AudioUrl] or [`BinaryContent`][pydantic_ai.BinaryContent]. The process is analogous to the examples above.

## Document Input

!!! info
    Some models do not support document input. Please check the model's documentation to confirm whether it supports document input.

!!! warning
    When using Gemini models, the document content will always be sent as binary data, regardless of whether you use `DocumentUrl` or `BinaryContent`. This is due to differences in how Vertex AI and Google AI handle document inputs.

    On Vertex AI, you can instead use [`FileUrl`][pydantic_ai.FileUrl] to instruct Gemini models to fetch document content themselves as explained in the [section below](#file-url-input).

You can provide document input using either [`DocumentUrl`][pydantic_ai.DocumentUrl] or [`BinaryContent`][pydantic_ai.BinaryContent]. The process is similar to the examples above.

If you have a direct URL for the document, you can use [`DocumentUrl`][pydantic_ai.DocumentUrl]:

```py {title="main.py" test="skip" lint="skip"}
from pydantic_ai import Agent, DocumentUrl

agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
    ]
)
print(result.data)
#> This document is the technical report introducing Gemini 1.5, Google's latest large language model...
```

The supported document formats vary by model.

You can also use [`BinaryContent`][pydantic_ai.BinaryContent] to pass document data directly:

```py {title="main.py" test="skip" lint="skip"}
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
print(result.data)
#> The document discusses...
```

## File URL input

!!! info
    Only Gemini models support direct file URL as input.

You can provide a file URL directly to a Gemini model by using [`FileUrl`][pydantic_ai.FileUrl]. The process is similar to the examples above, but no download is performed by PydanticAI and various formats and URL types are supported.

The following URLs are supported on Vertex AI:

- Google Cloud Storage bucket URI (whose protocol is `gs://`)
- Public HTTP URL
- Public YouTube video URL

See the [Vertex AI Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#filedata) and the [GLA Gemini API docs](https://ai.google.dev/api/caching#FileData) for use cases and limitations.

Here's an example of using PDF URLs with the VertexAI provider (`media_type` is required):

```py {title="main.py" test="skip" lint="skip"}
from pydantic_ai import Agent, FileUrl

agent = Agent(model='google-vertex:gemini-2.0-flash')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        FileUrl(
            url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
            media_type='application/pdf',
        ),
    ]
)
print(result.data)
#> The document is...
```

And similarly, for Youtube URLs and the GLA provider (`media_type` is optional):

```py {title="main.py" test="skip" lint="skip"}
from pydantic_ai import Agent, FileUrl

agent = Agent(model='google-gla:gemini-2.0-flash')
result = agent.run_sync(
    [
        'What is the main content of this video?',
        FileUrl(
            url='https://www.youtube.com/watch?v=bG2NQIwyEUU'
        ),
    ]
)
print(result.data)
#> The video shows a comparison...
```
