# File Understanding

The [`FileUnderstanding`][pydantic_ai.capabilities.FileUnderstanding] [capability](overview.md) lets an agent take images, documents and video its model cannot read. Before each request, every such file in a user prompt is replaced by a text description of it, written by a model that can read it. The agent's model sees the description in the file's place, and files it does accept are sent as they are.

```python {title="file_understanding_capability.py"}
from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.capabilities import FileUnderstanding

agent = Agent(
    'typesafe:jev-latest',
    output_type=bool,
    instructions='Is this document about animals?',
    capabilities=[FileUnderstanding(fallback_model='{{ openai:medium }}')],
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
    fallback_model='{{ anthropic:medium }}',
    instructions='Transcribe the text in this file. Keep headings and tables.',
)
```

The description reaches the model between `-----BEGIN FILE-----` and `-----END FILE-----` lines that name the file, the same way text files are inlined for models that take no attachments. A description is prose, so its block is marked `text/plain` whatever the file was; a text-like document (plain text, CSV, JSON, XML, YAML) needs no describing, is inlined as it is, and keeps its own media type. A provider file reference ([`UploadedFile`][pydantic_ai.messages.UploadedFile]) is left alone, since only its own provider can read it.

A file is described once and the description reused for the steps that follow, so a long conversation does not describe the same file again; the last 256 descriptions are kept. Binary content is reused for as long as the capability lives, because the bytes are their own identity. A file given by URL is reused only within the run that fetched it: the same URL can serve different bytes to different callers, and stop resolving once a signature expires, so one run's description of it is never handed to the next. A run's id is what scopes it, and the caller can pass that id to [`run()`][pydantic_ai.agent.AbstractAgent.run] as `run_id`: two runs given the same `run_id` share their URL descriptions, so the second gets the first run's description without the URL being fetched again.

The files a request needs described are described concurrently, each once however often it appears. Describing a file is a model run of its own, but its requests, tokens and cost are added to the agent run's [usage][pydantic_ai.usage.RunUsage], and the run's [usage limits][pydantic_ai.usage.UsageLimits] are checked before any describing request is sent, as if those were the run's next requests: a `request_limit` bounds them together with the agent's own.

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] has no single profile, so the capability leaves its files alone. Use the capability with a single model when files need to be described.
