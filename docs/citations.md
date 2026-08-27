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

## Citations in message history

Citations are primarily application metadata for rendering source links, evidence previews, and highlighted text.
Serializing and reloading [message history](message-history.md#storing-and-loading-messages-to-json) preserves
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations], but that does not guarantee that the next model receives
the structured citation data.

| Destination API | What the model receives from citation-bearing history | Notes |
| --- | --- | --- |
| Anthropic Messages, same provider | Native web and document citations | Replayed only when every citation on the text part contains the required Anthropic data |
| Amazon Bedrock Converse, same provider | A native `citationsContent` block | Replayed only when the complete block can be reconstructed |
| OpenAI Responses, same provider | Native URL and file annotations | Requires the native output item ID and `openai_send_reasoning_ids=True` |
| Google Gemini / Vertex AI, OpenAI Chat, OpenRouter, or xAI | Text only | These APIs have no confirmed assistant-history input matching Pydantic AI's normalized citations |
| Any different provider | Text only | Provider identifiers, locations, grouping, and offset meanings are not safely interchangeable |

"Text only" describes what is sent to the model; the citations remain on the stored Pydantic AI messages for the
application to use. Same-provider replay is all-or-nothing for each text part. If any citation is incomplete, malformed,
or belongs to another provider, the adapter sends that whole part as ordinary text rather than silently changing which
claims appear sourced.

!!! warning "Do not assume the model remembers a citation"
    A follow-up such as "Tell me more about source [1]" may reach a model that sees the rendered `[1]` marker but not
    its URL, excerpt, or document location. If the model needs the source, include the relevant source content in the
    new user prompt or let it retrieve the source again.

Pydantic AI does not append a synthetic source list or an explanatory citation message to history. Doing so could
change the meaning of the conversation and teach the model to imitate an application-specific citation format.

### Prompt caching and structured output

Citation handling does not add synthetic history messages, so ordinary provider prompt-cache rules continue to apply.
Anthropic supports citations with prompt caching, but generated citation blocks cannot themselves be cached; cache the
source document instead. See [Anthropic's citation and prompt-caching guidance](https://platform.claude.com/docs/en/build-with-claude/citations#using-prompt-caching-with-citations).

Anthropic rejects citations combined with its native JSON-schema structured output (`output_config.format`). In
Pydantic AI, do not combine an Anthropic citation request with
[`NativeOutput`][pydantic_ai.output.NativeOutput]; use a text response or a separate structured-output call. See
[Anthropic's feature compatibility notes](https://platform.claude.com/docs/en/build-with-claude/citations#feature-compatibility).

## Provider support

| Provider/API | Normalized response | Request behavior | Provider support notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web-search and document citations | `include_citations=True` enables citations for inline documents and Anthropic Web Fetch; Web Search supplies citations with grounded output | Client-provided `search_result` citations are outside this initial normalized surface |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html) | Converse document citations | `include_citations=True` enables citations for TXT and PDF document inputs | Citation locations within the source document remain in `provider_details`; the generated text block is the normalized content anchor |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Web, Maps, image-search, and file-search grounding | No citation-specific setting; enable the corresponding native grounding tool | Grounding byte offsets are normalized to Python character offsets; image citations link to their attribution page rather than the image asset |
| [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#GroundingMetadata) | Google Search and Vertex retrieval grounding | No citation-specific setting; enable the corresponding native grounding tool | Retrieved document resource names map to `document_id` |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | URL annotations and Responses file citations | No citation-specific setting; enable the corresponding native search tool | `container_file_citation` annotations are not file-search evidence and remain available only through raw provider annotations |
| [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/web-search) | Web-search URL annotations | No citation-specific setting; enable OpenRouter web search | Annotations without usable output offsets are normalized without an anchor |
| [xAI](https://docs.x.ai/developers/tools/citations) | Web, X, and collections inline citations | `include_citations=True` requests inline citations | Web and X links use marker anchors; collections citations map to document sources |

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
