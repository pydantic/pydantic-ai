# File Understanding

The [`FileUnderstanding`][pydantic_ai.capabilities.FileUnderstanding] [capability](overview.md) lets an agent take images, documents, audio and video its model cannot read. Before each request, every such file in a user prompt or a tool result is replaced by a text description of it, written by a model that can read it. The agent's model sees the description in the file's place, and files it does accept are sent as they are. Without the capability, a file a tool returned that the model cannot read is [replaced by a note](../tools-advanced.md#tool-return-file-provenance) saying so, and one in a user prompt fails the request.

```python {title="file_understanding_capability.py"}
from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.capabilities import FileUnderstanding

agent = Agent(
    'typesafe:jev-latest',
    output_type=bool,
    instructions='Is this document about animals?',
    capabilities=[FileUnderstanding(fallback_subagent_model='{{ openai:medium }}')],
)
result = agent.run_sync([DocumentUrl('https://example.com/field-guide.pdf')])
print(result.output)
#> True
```

Which files need describing comes from the model's [profile](../models/overview.md#inspecting-a-models-profile): [`supports_image_input`][pydantic_ai.profiles.ModelProfile.supports_image_input], [`supports_document_input`][pydantic_ai.profiles.ModelProfile.supports_document_input], [`supported_document_media_types`][pydantic_ai.profiles.ModelProfile.supported_document_media_types], [`supports_audio_file_input`][pydantic_ai.profiles.ModelProfile.supports_audio_file_input] and [`supports_video_input`][pydantic_ai.profiles.ModelProfile.supports_video_input]. Most models accept images and documents, so with them the capability mostly has audio and video described; on [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel], Word and Excel documents too. A text-only model like [Jev](../models/typesafe.md) has everything described.

A flag is `True` unless an adapter or a model family is known not to take that kind of file, so it is not a check that the model reads it. A text-only model served through an adapter that sends images, such as a text-only Llama behind vLLM, needs its profile to say so, or its images are sent and the provider rejects them:

```python {title="file_understanding_text_only_profile.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import FileUnderstanding
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.vllm import VLLMProvider

model = OpenAIChatModel(
    'meta-llama/Llama-3.3-70B-Instruct',
    provider=VLLMProvider(base_url='http://localhost:8000/v1'),
    profile=ModelProfile(supports_image_input=False, supports_document_input=False),
)
agent = Agent(model, capabilities=[FileUnderstanding(fallback_subagent_model='{{ openai:medium }}')])
```

`fallback_subagent_model` is the model that writes the descriptions, as a `'provider:model'` name or a [`Model`][pydantic_ai.models.Model] instance, and it must accept the files itself. `instructions` replaces the default request for a detailed description:

```python {title="file_understanding_instructions.py"}
from pydantic_ai.capabilities import FileUnderstanding

FileUnderstanding(
    fallback_subagent_model='{{ anthropic:medium }}',
    instructions='Transcribe the text in this file. Keep headings and tables.',
)
```

The description reaches the model between `-----BEGIN FILE-----` and `-----END FILE-----` lines that name the file, the same way text files are inlined for models that take no attachments. A description is prose, so its block is marked `text/plain` whatever the file was; a text-like document (plain text, CSV, JSON, XML, YAML) needs no describing, is inlined as it is, and keeps its own media type. A provider file reference ([`UploadedFile`][pydantic_ai.messages.UploadedFile]) is left alone, since only its own provider can read it.

A file is described once per conversation. The description is recorded in the message history, under the framework's reserved `__pydantic_ai__` namespace in the [`metadata`][pydantic_ai.messages.ModelRequest.metadata] of the request that holds the file (which is not sent to the model), and every later request uses that text instead of describing the file again. A turn that continues from `message_history`, another worker serving the next turn, and a [durable](../durable_execution/overview.md) run replaying its history all read the same description. Earlier requests therefore keep the same content from one request to the next, which keeps the prompt cache valid, and the describer is paid for once. A file given by URL is identified by its URL, and binary content by its bytes and media type. The downloaded text of a text document given by URL is recorded the same way, so it is not downloaded again. A separate conversation describes the file again. The [UI adapters](../ui/overview.md) leave that namespace out of the history they send to a client, so in a conversation whose history lives on the client the file is described again each turn, and a new description can differ, which changes earlier requests and costs the cached prefix.

The files a request needs described are described concurrently, each once however often it appears. Describing a file is a model run of its own, but its requests, tokens and cost are added to the agent run's [usage][pydantic_ai.usage.RunUsage], and the run's [usage limits][pydantic_ai.usage.UsageLimits] are checked before any describing request is sent, as if those were the run's next requests: a `request_limit` bounds them together with the agent's own.

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] has no single profile, so the capability leaves its files alone, and each model it tries is sent a note in place of a tool's file that model can't read. Use the capability with a single model when files need to be described.
