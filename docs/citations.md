# Citations

A citation tells you which source backs up part of a model's answer: a web page it searched or fetched, a document
you sent, or a file in a provider's file store.

Pydantic AI puts citations on [`TextPart.citations`][pydantic_ai.messages.TextPart.citations], in the same shape for
every provider. The model's text is never changed.

## Turning citations on

Models cite the results of search tools, such as [web search](native-tools.md#web-search-tool),
[web fetch](native-tools.md#web-fetch-tool) and [file search](native-tools.md#file-search-tool), without being asked.
Some providers only cite the documents you send when you ask. Add the [`Citations`][pydantic_ai.capabilities.Citations]
capability to ask them:

```python {title="document_citations.py"}
from pydantic_ai import Agent, BinaryContent, TextPart
from pydantic_ai.capabilities import Citations

agent = Agent('{{ anthropic:medium }}', capabilities=[Citations()])
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

The model can still answer without citations.

## Reading citations

A [`Citation`][pydantic_ai.messages.Citation] has:

- `sources`: where the information came from. A [`WebCitationSource`][pydantic_ai.messages.WebCitationSource] has the
  page's `url`. A [`DocumentCitationSource`][pydantic_ai.messages.DocumentCitationSource] is any other source. Its
  `document_id` is the provider's ID, not a file path, and a document you sent in the prompt may have no ID or title.
  Each source's `excerpts` are the passages the provider quoted as evidence.
- `anchor`: the part of the text the citation is about, as character positions: `part.content[anchor.start:anchor.end]`.
    - A [`ContentCitationAnchor`][pydantic_ai.messages.ContentCitationAnchor] covers the text the sources support.
      Some providers, such as Anthropic, cite whole text parts, so the anchor covers the whole part.
    - A [`MarkerCitationAnchor`][pydantic_ai.messages.MarkerCitationAnchor] covers a marker the model wrote, such as
      `[1]` or a link.
    - `None` means the provider didn't say where in the text the citation belongs.

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
                #> marker '([example.com](https://example.com))' Mount Columbia
```

The same answer looks like this from different providers:

```python
from pydantic_ai import (
    Citation,
    ContentCitationAnchor,
    MarkerCitationAnchor,
    TextPart,
    WebCitationSource,
)

# Google selects the supported text, and here lists both sources for it. Anthropic's anchor covers the whole part.
TextPart(
    'Pydantic validates data.',
    citations=[
        Citation(
            sources=[WebCitationSource('https://a.example'), WebCitationSource('https://b.example')],
            anchor=ContentCitationAnchor(start=0, end=24),
        )
    ],
)

# OpenAI selects the link the model wrote.
TextPart(
    'Pydantic validates data. ([example.com](https://example.com))',
    citations=[
        Citation(
            sources=[WebCitationSource('https://example.com')],
            anchor=MarkerCitationAnchor(start=25, end=61),
        )
    ],
)

# Some providers give no position in the text.
TextPart(
    'Pydantic validates data.',
    citations=[Citation(sources=[WebCitationSource('https://example.com')])],
)
```

## Good to know

- Citations are only on text parts. If your output comes from a [tool call](output.md#output-modes), as structured
  output does by default, the output itself has no citations.
- When streaming, Google and OpenAI Chat send citations at the end of the response, sometimes after a text part has
  finished. Read citations from the final response.
- Citation URLs, titles and excerpts come from outside sources, so treat them as untrusted. Excerpts can also contain
  private content from your documents, so choose whether to log them or show them to users.
- Anything else the provider returned about a citation, such as Google's confidence scores, is kept as the provider
  sent it in the part's `provider_details['citations']`. Its shape is the provider's and can change.

## Citations in message history

[Stored message history](message-history.md#storing-and-loading-messages-to-json) keeps
[`TextPart.citations`][pydantic_ai.messages.TextPart.citations] and the provider data in `provider_details['citations']`.
When you send that history to the provider that produced the citations, Pydantic AI sends some of the provider's own
citation data back with the text, so the model still knows which of its earlier statements came from which source:

| Provider | Citations sent back |
| --- | --- |
| Anthropic | Web search citations |
| OpenAI Responses | URL and file citations, when item IDs are sent; see [`openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids] |

These citations aren't sent back, and the text they belong to is sent as plain text:

- citations of documents and fetched pages, on Anthropic and Amazon Bedrock
- citations from other providers, whose APIs can't receive them
- citations from a different provider than the one the history is sent to
- an OpenAI URL citation whose text around the link was edited

Citations always stay on the stored messages, and Pydantic AI never adds a list of sources to the text.

Sending citations back doesn't affect prompt caching: the same history is sent the same way on every turn.

!!! warning "The model may not see the source"
    A follow-up such as "Tell me more about source [1]" may reach a model that sees the `[1]` marker but not its URL
    or excerpt. If the model needs the source, include it in the new prompt or let the model retrieve it again.

## Provider support

| Provider | Citations | How to turn them on | Notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web search, web fetch and documents | Web search: always. Web fetch and documents, including documents returned by tools: `Citations()`. [`WebFetchTool(enable_citations=True)`][pydantic_ai.native_tools.WebFetchTool.enable_citations] turns them on for web fetch only | Anchors cover whole text parts. A fetched PDF is a document source. Anthropic doesn't allow document citations with [`NativeOutput`][pydantic_ai.output.NativeOutput]. Citations of search results you provide yourself aren't included |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html) | Documents, on Anthropic models | `Citations()`, for text and PDF documents, including documents returned by tools | Other models on Bedrock don't support citations, so they ignore the setting. Anchors cover whole text parts. A document's title is the name Pydantic AI gives it, such as `Document 1` |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Google Search, file search and web fetch | Turn on the tool | |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) | Google Search | Turn on the tool | |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | Web pages, and files on Responses | Turn on web search, or file search for files | Other annotation types, such as `container_file_citation`, are only available with [`openai_include_raw_annotations`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_include_raw_annotations] |
| [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/web-search) | Web pages | Turn on web search. Perplexity Sonar models always search | OpenRouter doesn't say where in the text a citation belongs, so citations have no anchor |
| [xAI](https://docs.x.ai/developers/tools/citations) | Web, X and collection citations | `Citations()` | Turning citations on changes the answer: the model writes Markdown markers such as `[[1]](https://x.com/...)` into it, and the anchor covers the marker. Collection citations are document sources |
