"""Requesting Anthropic citations, mapping the citations a response carries to `Citation`s, and replaying them from history."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Any, cast

from anthropic.types.beta import (
    BetaCitationCharLocation,
    BetaCitationContentBlockLocation,
    BetaCitationPageLocation,
    BetaCitationsConfigParamParam,
    BetaCitationsWebSearchResultLocation,
    BetaCitationWebSearchResultLocationParam,
    BetaMessageParam,
    BetaPlainTextSource,
    BetaRequestDocumentBlockParam,
    BetaTextCitation,
    BetaWebFetchBlock,
    BetaWebFetchToolResultBlock,
)
from pydantic import ValidationError

from ... import _utils
from ...exceptions import UserError
from ...messages import (
    Citation,
    CitationSource,
    ContentCitationAnchor,
    DocumentCitationSource,
    ModelMessage,
    ModelResponse,
    NativeToolReturnPart,
    TextPart,
    WebCitationSource,
)
from ...native_tools import WebFetchTool
from ...settings import ModelSettings
from .. import ModelRequestParameters
from .._tool_choice import resolve_tool_choice


def fetched_page(block: BetaWebFetchToolResultBlock) -> tuple[str, str] | None:
    """Return the URL and text of a page Web Fetch fetched, or `None` for a failed fetch or a PDF."""
    if isinstance(block.content, BetaWebFetchBlock) and isinstance(block.content.content.source, BetaPlainTextSource):
        return block.content.url, block.content.content.source.data
    return None


_CITATIONS_WITH_FORCED_TOOL_ERROR = (
    "`Citations()` can't be used with a `tool_choice` that forces a tool call on Anthropic models: "
    "the model writes its citation markup into the tool's arguments. "
    'Remove `Citations()`, or let the model choose whether to call the tool.'
)


def citations_included(model_settings: ModelSettings, model_request_parameters: ModelRequestParameters) -> bool:
    """Whether the `citations` setting turns on document and Web Fetch citations for this request.

    Anthropic rejects citations with native structured output, and writes its `<cite>` markup into tool and prompted
    output, so they're only turned on for text output.
    """
    return model_settings.get('citations', False) and model_request_parameters.output_mode == 'text'


def check_citations_tool_choice(
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
    messages: list[BetaMessageParam],
    include_citations: bool,
) -> None:
    """Raise if the `citations` setting turned on citations for a request that forces a function tool call.

    The model writes its `<cite>` markup into a forced tool call's arguments. A request with nothing to cite, no
    document and no Web Fetch tool, is unchanged by the setting, so it's allowed.
    """
    if not include_citations:
        return
    tool_choice = resolve_tool_choice(model_settings, model_request_parameters)
    if not (tool_choice == 'required' or (isinstance(tool_choice, tuple) and tool_choice[0] == 'required')):
        return
    if any(isinstance(tool, WebFetchTool) for tool in model_request_parameters.native_tools) or has_cited_document(
        messages
    ):
        raise UserError(_CITATIONS_WITH_FORCED_TOOL_ERROR)


def has_cited_document(value: object) -> bool:
    """Whether a request's messages hold a document block with citations turned on, including in tool results."""
    if isinstance(value, list):
        return any(has_cited_document(item) for item in cast(list[object], value))
    if _utils.is_str_dict(value):
        if value.get('type') == 'document' and value.get('citations'):
            return True
        return any(has_cited_document(item) for item in value.values())
    return False


def fetched_pages_from_history(messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
    """Return the URL and text of each page Web Fetch fetched earlier in the conversation, which a new answer can cite."""
    pages: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, NativeToolReturnPart) and part.tool_name == WebFetchTool.kind:
                    try:
                        page = BetaWebFetchBlock.model_validate(part.content)
                    except ValidationError:
                        continue
                    if isinstance(page.content.source, BetaPlainTextSource):
                        pages.append((page.url, page.content.source.data))
    return pages


def fetched_page_url(citation: BetaCitationCharLocation, fetched_pages: Sequence[tuple[str, str]]) -> str | None:
    """Return the URL of the fetched page a character citation quotes, if exactly one page matches.

    Anthropic cites a fetched page like a document, by index and character range, without its URL. When pages with
    different URLs hold the same text at that range, the citation stays a document source rather than guess.
    """
    if not citation.cited_text:
        return None
    start, end = citation.start_char_index, citation.end_char_index
    urls = {url for url, page in fetched_pages if page[start:end] == citation.cited_text}
    return next(iter(urls)) if len(urls) == 1 else None


_CITATION_TYPES = frozenset({'web_search_result_location', 'char_location', 'page_location', 'content_block_location'})


def map_citations(
    citations: Sequence[BetaTextCitation], text: str, fetched_pages: Sequence[tuple[str, str]] = ()
) -> list[Citation] | None:
    sources: list[CitationSource] = []
    for citation in citations:
        # The SDK doesn't validate responses, so it builds a citation of a type it doesn't know into one of its classes.
        if citation.type not in _CITATION_TYPES:
            continue
        excerpts = [citation.cited_text] if citation.cited_text else []
        if isinstance(citation, BetaCitationsWebSearchResultLocation):
            sources.append(WebCitationSource(url=citation.url, title=citation.title, excerpts=excerpts))
        elif isinstance(citation, BetaCitationCharLocation) and (url := fetched_page_url(citation, fetched_pages)):
            sources.append(WebCitationSource(url=url, title=citation.document_title, excerpts=excerpts))
        elif isinstance(
            citation,
            (BetaCitationCharLocation, BetaCitationPageLocation, BetaCitationContentBlockLocation),
        ) and (citation.file_id or citation.document_title or excerpts):
            # These locations select evidence inside the input document, so they stay in the raw citation data.
            sources.append(
                DocumentCitationSource(document_id=citation.file_id, title=citation.document_title, excerpts=excerpts)
            )
    if not sources:
        return None
    # Anthropic splits its answer into text blocks and cites whole blocks, so the citation covers the whole part.
    return [Citation(sources=sources, anchor=ContentCitationAnchor(start=0, end=len(text)) if text else None)]


def citations_provider_details(citations: Sequence[BetaTextCitation]) -> dict[str, Any]:
    return {'citations': [citation.model_dump(mode='json') for citation in citations]}


def citations_fit_text(part: TextPart) -> bool:
    """Whether a text part is still the block Anthropic cited, so its citations can be sent back.

    Anthropic cites whole blocks, so each citation covers the whole text. One that doesn't, because the text was
    edited or joined with other text, such as by a UI adapter, would vouch for text it never cited.
    """
    return bool(part.citations) and all(
        citation.anchor == ContentCitationAnchor(start=0, end=len(part.content)) for citation in part.citations or []
    )


def map_citation_for_replay(citation: dict[str, Any]) -> BetaCitationWebSearchResultLocationParam | None:
    """Rebuild a web search citation Anthropic returned from its raw data on the text part, or return `None`.

    Document citations aren't sent back: they name their document by its position in the request, which changes when
    history is edited, and the citation can't tell which document it was.
    """
    if citation.get('type') != 'web_search_result_location':
        return None
    cited_text, url, title = citation.get('cited_text'), citation.get('url'), citation.get('title')
    encrypted_index = citation.get('encrypted_index')
    if (
        not isinstance(cited_text, str)
        or not isinstance(url, str)
        or not isinstance(title, str | None)
        or not isinstance(encrypted_index, str)
    ):
        return None
    return BetaCitationWebSearchResultLocationParam(
        type='web_search_result_location',
        url=url,
        title=title,
        cited_text=cited_text,
        encrypted_index=encrypted_index,
    )


def with_document_citations(
    block: BetaRequestDocumentBlockParam, include_citations: bool
) -> BetaRequestDocumentBlockParam:
    if include_citations:
        block['citations'] = BetaCitationsConfigParamParam(enabled=True)
    return block
