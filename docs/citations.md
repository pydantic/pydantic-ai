# Citations

Pydantic AI puts the web and document citations a provider returns on
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations], without changing the model's text. Some providers only
return citations when asked. Set `include_citations=True` to ask for them:

```python {test="skip"}
from pydantic_ai import Agent, BinaryContent

agent = Agent('anthropic:claude-sonnet-4-5', model_settings={'include_citations': True})
result = agent.run_sync(
    [
        'How long do customers have to return an item?',
        BinaryContent(data=b'Items may be returned within 30 days.', media_type='text/plain'),
    ]
)
```

Providers that don't need an opt-in ignore this setting, and a model may still return no citations.

To render citations, read each text part's citations and their sources:

```python {test="skip"}
from pydantic_ai import (
    Agent,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool

agent = Agent('openai-responses:gpt-5.2', capabilities=[NativeTool(WebSearchTool())])
result = agent.run_sync('What is the tallest mountain in Alberta?')

for message in result.all_messages():
    for part in message.parts:
        if isinstance(part, TextPart):
            for citation in part.citations or []:
                anchor = citation.anchor
                if isinstance(anchor, ContentCitationAnchor):
                    location = f'supported text: {part.content[anchor.start : anchor.end]!r}'
                elif isinstance(anchor, MarkerCitationAnchor):
                    location = f'citation marker: {part.content[anchor.start : anchor.end]!r}'
                else:
                    location = 'somewhere in the text part'

                for source in citation.sources:
                    if isinstance(source, WebCitationSource):
                        label = source.title or source.url
                    else:
                        label = source.title or source.document_id or 'Document source'
                    print(location, label, source.excerpts)
```

A citation has one or more sources and an optional [`anchor`][pydantic_ai.messages.CitationAnchor], which holds
Python character offsets into the text: `part.content[anchor.start:anchor.end]`. A
[`ContentCitationAnchor`][pydantic_ai.messages.ContentCitationAnchor] selects the supported text, a
[`MarkerCitationAnchor`][pydantic_ai.messages.MarkerCitationAnchor] selects a citation marker the model wrote, and a
citation without an anchor belongs to the text part, but its position in the text is unknown. Handle all three:

```python
from pydantic_ai import (
    Citation,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)

# Google: both sources support the selected text.
TextPart(
    'Pydantic validates data.',
    citations=[
        Citation(
            sources=[WebCitationSource('https://a.example'), WebCitationSource('https://b.example')],
            anchor=ContentCitationAnchor(start=0, end=24),
        )
    ],
)

# OpenAI: the selected text is the citation marker.
TextPart(
    'Pydantic validates data. [1]',
    citations=[
        Citation(
            sources=[WebCitationSource('https://example.com')],
            anchor=MarkerCitationAnchor(start=25, end=28),
        )
    ],
)

# Anthropic: no text range.
TextPart(
    'Pydantic validates data.',
    citations=[
        Citation(
            sources=[
                WebCitationSource(
                    'https://example.com',
                    excerpts=['Pydantic provides data validation using Python type hints.'],
                )
            ]
        )
    ],
)
```

A [`DocumentCitationSource`][pydantic_ai.messages.DocumentCitationSource] is any non-web source. Its `document_id` is
the provider's identifier, not a local path, and inline documents may have neither an ID nor a title. A source's
`excerpts` are the passages the provider returned as evidence: depending on the provider, an exact quote or a wider
chunk of the source.

Treat citation URLs, titles, and excerpts as untrusted data. Excerpts can contain private retrieved content, so choose
deliberately whether to log, render, or send them to a client.

## Citations in message history

[Stored message history](message-history.md#storing-and-loading-messages-to-json) keeps
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations]. When you send that history to the provider that
produced the citations, these citations are sent back with the text:

- **Anthropic**: web search citations, and citations of plain-text documents with `include_citations=True`.
- **Amazon Bedrock**: citations of plain-text documents, with `include_citations=True`.
- **OpenAI Responses**: URL and file citations, when item IDs are sent (see
  [`openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids]).

An Anthropic or Bedrock document citation is only sent back while the cited document is still in the message history
and the cited text still matches it. Everything else, including all citations from a different provider, is sent as
plain text. Citations always stay on the stored messages, and Pydantic AI never adds a list of sources to the text.

The [Vercel AI adapter](ui/vercel-ai.md#citations) keeps citations when the frontend holds the message history.

!!! warning "The model may not see the source"
    A follow-up such as "Tell me more about source [1]" may reach a model that sees the `[1]` marker but not its URL
    or excerpt. If the model needs the source, include it in the new prompt or let the model retrieve it again.

## Provider support

| Provider/API | Citations returned | How to enable | Provider support notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web search and document citations | `include_citations=True` enables citations for documents and requests them for Web Fetch; Web Search returns citations without it | Anthropic [rejects](https://platform.claude.com/docs/en/build-with-claude/citations#feature-compatibility) document citations combined with [`NativeOutput`][pydantic_ai.output.NativeOutput]. Citations of client-provided search results are not included |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html) | Document citations | `include_citations=True` enables citations for text and PDF documents | The anchor covers the whole cited block of text; the location in the document is in the source's `provider_details` |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Search, file search and Web Fetch grounding | Enable the grounding tool | |
| [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#GroundingMetadata) | Search and Vertex retrieval grounding | Enable the grounding tool | A retrieved document's resource name is its `document_id` |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | URL citations, and Responses file citations | Enable Web Search for URL citations, or File Search for file citations | Other annotation types, such as `container_file_citation` and `file_path`, are only available as raw annotations |
| [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/web-search) | Web search URL citations | Enable web search; Perplexity Sonar models always search | Citations have no anchor when the model gives no position in the text, as with Perplexity Sonar |
| [xAI](https://docs.x.ai/developers/tools/citations) | Web, X, and collection citations | `include_citations=True` requests inline citations | Inline citations have marker anchors; collection citations are document sources |
