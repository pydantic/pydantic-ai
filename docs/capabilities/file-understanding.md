# File Understanding

The [`FileUnderstanding`][pydantic_ai.capabilities.FileUnderstanding] [capability](overview.md) lets an agent take images, documents and video its model cannot read. Before each request, every such file in a user prompt is replaced by a text description of it, written by a model that can read it. The agent's model sees the description in the file's place, and files it does accept are sent as they are.

```python {title="file_understanding_capability.py"}
from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.capabilities import FileUnderstanding

agent = Agent(
    'typesafe:jev-latest',
    output_type=bool,
    instructions='Is this document about animals?',
    capabilities=[FileUnderstanding(fallback_model='openai:gpt-5.6-sol')],
)
result = agent.run_sync([DocumentUrl('https://example.com/field-guide.pdf')])
print(result.output)
#> True
```

Which files need describing comes from the model's [profile](../models/overview.md#inspecting-a-models-profile): [`supports_image_input`][pydantic_ai.profiles.ModelProfile.supports_image_input], [`supports_document_input`][pydantic_ai.profiles.ModelProfile.supports_document_input] and [`supports_video_input`][pydantic_ai.profiles.ModelProfile.supports_video_input]. Most models accept images and documents, so with them the capability only has video described. A text-only model like [Jev](../models/typesafe.md) has everything described.

`fallback_model` is the model that writes the descriptions, as a `'provider:model'` name or a [`Model`][pydantic_ai.models.Model] instance, and it must accept the files itself. `instructions` replaces the default request for a detailed description:

```python {title="file_understanding_instructions.py"}
from pydantic_ai.capabilities import FileUnderstanding

FileUnderstanding(
    fallback_model='anthropic:claude-sonnet-5',
    instructions='Transcribe the text in this file. Keep headings and tables.',
)
```

The description reaches the model between `-----BEGIN FILE-----` and `-----END FILE-----` lines that name the file, the same way text files are inlined for models that take no attachments. A description is prose, so its block is marked `text/plain` whatever the file was; a text-like document (plain text, CSV, JSON, XML, YAML) needs no describing, is inlined as it is, and keeps its own media type. A provider file reference ([`UploadedFile`][pydantic_ai.messages.UploadedFile]) is left alone, since only its own provider can read it.

A file is described once and the description reused for the steps that follow, so a long conversation does not describe the same file again; the last 256 descriptions are kept. Binary content is reused for as long as the capability lives, because the bytes are their own identity. A file given by URL is reused only within the run that fetched it: the same URL can serve different bytes to different callers, and stop resolving once a signature expires, so one run's description of it is never handed to the next.

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] has no single profile, so the capability leaves its files alone. Use the capability with a single model when files need to be described.
