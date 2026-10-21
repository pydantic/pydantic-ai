# Citations

A citation links part of a model's answer to the source that supports it: a web page the model searched or fetched, a
document you sent with the prompt, or a file in a provider's file store. Providers return citations in two situations:

- When the provider runs a search or retrieval tool for the model, such as [web search](native-tools.md#web-search-tool),
  [web fetch](native-tools.md#web-fetch-tool) or [file search](native-tools.md#file-search-tool). Most providers cite
  these sources without being asked.
- When you send documents with the prompt. Providers that support this only cite documents when asked.

Pydantic AI puts the citations a provider returns on [`TextPart.citations`][pydantic_ai.messages.TextPart.citations], in
the same shape for every provider, without changing the model's text. Set `citations=True` to ask the
providers that only cite when asked:

```python {title="document_citations.py"}
from pydantic_ai import Agent, BinaryContent, TextPart

agent = Agent('{{ anthropic:medium }}', model_settings={'citations': True})
result = agent.run_sync(
    [
        'How long do customers have to return an item?',
        BinaryContent(data=b'Items may be returned within 30 days.', media_type='text/plain'),
    ]
)
for part in result.response.parts:
    if isinstance(part, TextPart):
        for citation in part.citations or []:
            print(citation.sources[0].excerpts)
            #> ['Items may be returned within 30 days.']
```

Providers that don't need to be asked ignore this setting, and a model may still answer without citations.

## Reading citations

A [`Citation`][pydantic_ai.messages.Citation] has one or more `sources` and an optional
[`anchor`][pydantic_ai.messages.CitationAnchor], which holds Python character offsets into the text part:
`part.content[anchor.start:anchor.end]`. Providers point at different things, so there are three cases:

- A [`ContentCitationAnchor`][pydantic_ai.messages.ContentCitationAnchor] selects the text the sources support.
- A [`MarkerCitationAnchor`][pydantic_ai.messages.MarkerCitationAnchor] selects a citation marker, such as `[1]`, that
  the model wrote into its answer.
- A citation without an anchor belongs to the text part, but its position in the text is unknown.

```python {title="render_citations.py"}
from pydantic_ai import (
    Agent,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool

agent = Agent('{{ openai:medium }}', capabilities=[NativeTool(WebSearchTool())])
result = agent.run_sync('What is the tallest mountain in Alberta?')

for part in result.response.parts:
    if isinstance(part, TextPart):
        for citation in part.citations or []:
            anchor = citation.anchor
            if isinstance(anchor, ContentCitationAnchor):
                location = f'supports {part.content[anchor.start : anchor.end]!r}'
            elif isinstance(anchor, MarkerCitationAnchor):
                location = f'marker {part.content[anchor.start : anchor.end]!r}'
            else:
                location = 'somewhere in the text'

            for source in citation.sources:
                if isinstance(source, WebCitationSource):
                    label = source.title or source.url
                else:
                    label = source.title or source.document_id or 'document'
                print(location, label)
                #> marker '[1]' Mount Columbia - Wikipedia
```

The same answer looks like this from each kind of provider:

```python
from pydantic_ai import (
    Citation,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)

# Google and Anthropic select the supported text. Google lists both sources for the selected text.
TextPart(
    'Pydantic validates data.',
    citations=[
        Citation(
            sources=[WebCitationSource('https://a.example'), WebCitationSource('https://b.example')],
            anchor=ContentCitationAnchor(start=0, end=24),
        )
    ],
)

# OpenAI selects the citation marker the model wrote.
TextPart(
    'Pydantic validates data. [1]',
    citations=[
        Citation(
            sources=[WebCitationSource('https://example.com')],
            anchor=MarkerCitationAnchor(start=25, end=28),
        )
    ],
)

# Some providers give no position in the text.
TextPart(
    'Pydantic validates data.',
    citations=[Citation(sources=[WebCitationSource('https://example.com')])],
)
```

A [`WebCitationSource`][pydantic_ai.messages.WebCitationSource] has the page's `url`. A
[`DocumentCitationSource`][pydantic_ai.messages.DocumentCitationSource] is any other source. Its `document_id` is the
provider's identifier, not a local path, and documents sent inline may have neither an ID nor a title. A source's
`excerpts` are the passages the provider returned as evidence: depending on the provider, an exact quote or a wider
chunk of the source.

Treat citation URLs, titles, and excerpts as untrusted data. Excerpts can contain private retrieved content, so choose
deliberately whether to log, render, or send them to a client.

## Provider data

The citation types only hold what every provider can fill in. Everything else a provider returned for a part's
citations, such as where in a document the cited passage is, Google's confidence scores, or the position of an OpenAI
file citation, is kept in the provider's own format in the part's `provider_details['citations']`. Its shape is the
provider's and can change with the provider's API.

## Citations in message history

[Stored message history](message-history.md#storing-and-loading-messages-to-json) keeps
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations] and the provider data in `provider_details['citations']`.
When you send that history to the provider that produced the citations, its own citation data is sent back with the
text:

- **Anthropic**: web search citations, and citations of plain-text documents with `citations=True`.
- **Amazon Bedrock**: citations of plain-text documents, with `citations=True` on Anthropic models.
- **OpenAI Responses**: URL and file citations, when item IDs are sent (see
  [`openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids]).

An Anthropic or Bedrock document citation is only sent back while the cited document is still in the message history
and the cited text still matches it. Everything else, including all citations from a different provider, is sent as
plain text. Citations always stay on the stored messages, and Pydantic AI never adds a list of sources to the text.

!!! warning "The model may not see the source"
    A follow-up such as "Tell me more about source [1]" may reach a model that sees the `[1]` marker but not its URL
    or excerpt. If the model needs the source, include it in the new prompt or let the model retrieve it again.

## Provider support

| Provider/API | Citations returned | How to enable | Provider support notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web Search, Web Fetch and document citations | Web Search always returns citations. `citations=True` enables them for Web Fetch and for all documents, including those in tool results. [`WebFetchTool(enable_citations=False)`][pydantic_ai.native_tools.WebFetchTool.enable_citations] turns them off for Web Fetch alone | Anthropic splits its answer into text parts and cites whole parts, so the anchor covers the part. A fetched PDF is cited as a document source. Anthropic [rejects](https://platform.claude.com/docs/en/build-with-claude/citations#feature-compatibility) document citations combined with [`NativeOutput`][pydantic_ai.output.NativeOutput]. Citations of client-provided search results are not included |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html) | Document citations from Anthropic models | `citations=True` enables citations for text and PDF documents, including those in tool results | Other models on Bedrock reject citations, so the setting is ignored for them. Bedrock cites whole blocks of text, so the anchor covers the part |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Search, file search and Web Fetch grounding | Enable the grounding tool | |
| [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#GroundingMetadata) | Search and Vertex retrieval grounding | Enable the grounding tool | A retrieved document's resource name is its `document_id` |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | URL citations, and Responses file citations | Enable Web Search for URL citations, or File Search for file citations | Other annotation types, such as `container_file_citation` and `file_path`, are only available as raw annotations |
| [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/web-search) | Web search URL citations | Enable web search; Perplexity Sonar models always search | Citations have no anchor when the model gives no position in the text, as with Perplexity Sonar |
| [xAI](https://docs.x.ai/developers/tools/citations) | Web, X, and collection citations | `citations=True` requests inline citations | With inline citations, the model writes Markdown citation markers such as `[[1]](https://x.com/...)` into its answer, and the anchor selects the marker. Collection citations are document sources |
