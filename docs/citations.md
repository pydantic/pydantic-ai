# Citations

Pydantic AI normalizes provider-returned web and document citations onto
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations] without changing the model's text. Some providers need
citations to be explicitly requested. Set `include_citations=True` to request them where supported:

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

This setting is a best-effort request: providers that do not require an explicit opt-in ignore it, and a model may
still return no citations. It is separate from provider-specific settings that retain raw annotation payloads.

```python {test="skip"}
from pydantic_ai import (
    Agent,
    ContentCitationAnchor,
    DocumentCitationSource,
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
                    location = 'the text part as a whole'

                for source in citation.sources:
                    if isinstance(source, WebCitationSource):
                        label = source.title or source.url
                    elif isinstance(source, DocumentCitationSource):
                        label = source.title or source.document_id or 'Document source'
                    print(location, label, source.excerpts)
```

A citation can reference one or more web or document sources. Its optional
[`anchor`][pydantic_ai.messages.CitationAnchor] uses Python character offsets into the containing text:
`part.content[anchor.start:anchor.end]`. A
[`ContentCitationAnchor`][pydantic_ai.messages.ContentCitationAnchor] identifies supported text, while a
[`MarkerCitationAnchor`][pydantic_ai.messages.MarkerCitationAnchor] identifies a citation marker already present in the
model output. An absent anchor means the provider did not supply a text range that Pydantic AI could safely normalize.

Google grounding can produce content anchors, including citations where several sources jointly support one span.
OpenAI web-search citations can produce marker anchors, while OpenAI file citations and Anthropic web-search citations
may be unanchored. Consumers should therefore handle all three cases rather than assuming every source identifies a
specific assertion.

A [`DocumentCitationSource`][pydantic_ai.messages.DocumentCitationSource] means non-web evidence, not necessarily a
file uploaded through Pydantic AI. Its optional `document_id` is an opaque identifier in the provider's storage system;
it is not a local path and does not imply that the application can download the document. Inline documents may have no
stable identifier or title, in which case a renderer should use a generic fallback label.

Both source types contain an `excerpts` list of provider-selected supporting passages. An item can be an exact cited
passage or a broader retrieval chunk, depending on the provider. The list preserves separate passages when a provider
returns more than one for the same source. It lets applications show evidence previews without separately retrieving
the source, but should not be interpreted as a provider-independent exact quotation or as the text selected by the
citation's anchor.

## Provider support

| Provider/API | Normalized response | Request behavior | Provider support notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web-search and document citations | `include_citations=True` enables citations for inline documents and Anthropic Web Fetch; Web Search supplies citations with grounded output | Client-provided `search_result` citations are outside this initial normalized surface |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Web grounding and file-search grounding | No citation-specific setting; enable the corresponding native grounding tool | Grounding byte offsets are normalized to Python character offsets |
| [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#GroundingMetadata) | Google Search and Vertex retrieval grounding | No citation-specific setting; enable the corresponding native grounding tool | Retrieved document resource names map to `document_id` |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | URL annotations and Responses file citations | No citation-specific setting; enable the corresponding native search tool | `container_file_citation` annotations are not file-search evidence and remain available only through raw provider annotations |

For example, the normalized provider results can have these shapes:

```python
from pydantic_ai import (
    Citation,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)

# Google: both sources support the selected assertion.
TextPart(
    'Pydantic validates data.',
    citations=[
        Citation(
            sources=[WebCitationSource('https://a.example'), WebCitationSource('https://b.example')],
            anchor=ContentCitationAnchor(start=0, end=24),
        )
    ],
)

# OpenAI: the selected text is the rendered citation marker, not the supported assertion.
TextPart(
    'Pydantic validates data. [1]',
    citations=[
        Citation(
            sources=[WebCitationSource('https://example.com')],
            anchor=MarkerCitationAnchor(start=25, end=28),
        )
    ],
)

# Anthropic: the source qualifies this text part, but no character range was supplied.
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

Treat citation URLs, titles, and excerpts as untrusted data. Excerpts can contain private retrieved content, so
applications should choose deliberately whether to log, render, or send them to a client.
