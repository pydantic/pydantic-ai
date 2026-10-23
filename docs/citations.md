# Citations

A citation tells you which source backs up part of a model's answer: a web page it searched or fetched, a document
you sent, or a file in a provider's file store.

Pydantic AI puts citations on [`TextPart.citations`][pydantic_ai.messages.TextPart.citations], in the same shape for
every provider. The model's text is never changed.

## Turning citations on

Most models cite the results of search tools, such as [web search](native-tools.md#web-search-tool),
[web fetch](native-tools.md#web-fetch-tool) and [file search](native-tools.md#file-search-tool), without being asked.
Some providers only cite the documents you send, or on xAI any source, when you ask. Add the [`Citations`][pydantic_ai.capabilities.Citations]
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
    - `None` means the provider didn't say which text the citation is about. OpenAI file citations, for example, only
      give a position, which is kept in the part's `provider_details['citations']`.

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

## Citations, tools and structured output

Citations are attached to text, so they only come back when your agent's output is text. With
[structured output](output.md), including an output type that combines `str` with a structured type, such as
`[str, Answer]`, the answer comes back as tool call arguments, which can't carry citations.

On Anthropic, and on Bedrock with Anthropic models, the model writes its citation markup into anything that isn't
text, such as `summary='<cite index="1-1">Items may be returned within 30 days.</cite>'`. So with
`Citations()`:

- **Structured output:** the model isn't asked for citations, so your fields stay clean and you get no document
  citations. (Anthropic rejects citations with `NativeOutput`.)
- **A tool call forced with [`tool_choice`](tools-advanced.md#dynamic-tool-choice-via-capabilities):** raises a
  `UserError`, because the markup reaches the tool's arguments every time.
- **Tools the model chooses to call:** citations stay on, but some models still write the markup into the tool's
  arguments now and then. Check arguments that end up in front of people, such as an email body.

On xAI, the model answers in text instead of calling the output tool when citations are on, so with structured output
`Citations()` doesn't ask for them either. Tools work as usual.

If you need citations, use text output.

Anthropic always cites [web search](native-tools.md#web-search-tool) results, so with web search the markup can show
up in structured output even without `Citations()`. The same goes for [web fetch](native-tools.md#web-fetch-tool)
when you set `enable_citations=True` on the tool.

On other providers, `Citations()` has no effect on structured output. With `NativeOutput` or `PromptedOutput` the
output is JSON text, so web search citations can come back on it, with positions in the JSON text rather than in your
fields.

## Good to know

- When streaming, citations can arrive after their text: at the end of each text part on Anthropic, Bedrock and
  OpenAI Responses, and at the end of the response on Google and OpenAI Chat. Read citations from the final response.
  On xAI they go on the last text part at the end of the response, so a streamed answer followed by a tool call gets
  none.
- The [AG-UI adapter](ui/ag-ui.md) doesn't send citations to the frontend.
- Citation URLs, titles and excerpts come from outside sources, so treat them as untrusted. Excerpts can also contain
  private content from your documents, so choose whether to log them or show them to users.
- Anything else the provider returned about a citation, such as Google's confidence scores, is kept as the provider
  sent it in the part's `provider_details['citations']`. Its shape is the provider's and can change.

## Citations in message history

[Saved message history](message-history.md#storing-and-loading-messages-to-json) keeps every citation. When you send
it back to the same provider, these citations go with the text, so the model knows where its earlier statements came
from:

| Provider | Citations sent back |
| --- | --- |
| Anthropic | Web search citations |
| OpenAI Responses | Web and file citations, when message IDs are sent, which is the default for reasoning models. See [`openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids] |

Other citations stay on the saved messages, but their text is sent as plain text. That covers document and
fetched-page citations, citations from other providers and APIs (such as OpenAI Chat), and history sent to a different
provider. A citation is also left out when its text was changed or joined with other text.

If you edit the text of a saved answer, also clear that part's `citations` and its `provider_details['citations']`.
Pydantic AI leaves out citations whose text clearly changed, but an edit that keeps the same length can go unnoticed.

History that's sent back unchanged goes out the same way every turn, so prompt caching keeps working.

!!! warning "The model may not see the source"
    If a user later asks about "source [1]", the model may see the `[1]` but not what it pointed to. Put the source in
    the new prompt, or let the model look it up again.

## Provider support

| Provider | Citations | How to turn them on | Notes |
| --- | --- | --- | --- |
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/citations) | Web search, web fetch and documents | Web search: always. Web fetch and documents, including documents returned by tools: `Citations()`. [`WebFetchTool(enable_citations=True)`][pydantic_ai.native_tools.WebFetchTool.enable_citations] turns them on for web fetch only | Anchors cover whole text parts. A fetched PDF is a document source. Citations of search results you provide yourself aren't included |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html) | Documents, on Anthropic models | `Citations()`, for text and PDF documents, including documents returned by tools | Other models on Bedrock don't support citations, so they ignore the setting. Anchors cover whole text parts. A document's title is the name Pydantic AI gives it, such as `Document 1`. Documents read from S3 aren't cited, as Bedrock doesn't allow it |
| [Google Gemini API](https://ai.google.dev/gemini-api/docs/google-search) | Google Search, file search and web fetch | Turn on the tool | Google Search URLs are Google redirect links, and titles are often just the site's domain |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) | Google Search | Turn on the tool | Same as the Gemini API |
| [OpenAI Chat and Responses](https://platform.openai.com/docs/guides/tools-web-search) | Web pages, and files on Responses | Turn on web search, or file search for files | Other annotation types, such as `container_file_citation`, are only available with [`openai_include_raw_annotations`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_include_raw_annotations] |
| [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/web-search) | Web pages | Turn on web search. Perplexity Sonar models always search, and OpenRouter rejects them with the web search tool | OpenRouter doesn't say where in the text a citation belongs, so citations have no anchor |
| [xAI](https://docs.x.ai/developers/tools/citations) | Web, X and collection citations | `Citations()` | Turning citations on changes the answer: the model writes Markdown markers such as `[[1]](https://x.com/...)` into it, and the anchor covers the marker. Collection citations are document sources |
