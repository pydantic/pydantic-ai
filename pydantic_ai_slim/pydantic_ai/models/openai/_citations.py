"""URL and file citations from both APIs, and their replay."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from openai.types import chat, responses
from pydantic import BaseModel, TypeAdapter, ValidationError

from ... import _utils
from ..._thinking_part import split_content_into_text_and_thinking
from ...messages import (
    Citation,
    DocumentCitationSource,
    MarkerCitationAnchor,
    TextPart,
    ThinkingPart,
    WebCitationSource,
)
from ...profiles import DEFAULT_THINKING_TAGS
from ...profiles.openai import OpenAIModelProfile

responses_output_text_annotations_ta = TypeAdapter(list[responses.response_output_text.Annotation])


responses_output_text_annotation_ta: TypeAdapter[responses.response_output_text.Annotation] = TypeAdapter(
    responses.response_output_text.Annotation
)


chat_annotations_ta = TypeAdapter(list[chat.chat_completion_message.Annotation])


chat_annotation_ta = TypeAdapter(chat.chat_completion_message.Annotation)


def jsonable_annotation(annotation: object) -> object:
    """A streamed annotation as JSON data: a model the SDK parsed it into is dumped, anything else is kept as is."""
    return annotation.model_dump(mode='json') if isinstance(annotation, BaseModel) else annotation


def marker_citation_anchor(start: int, end: int, text: str) -> MarkerCitationAnchor | None:
    # OpenAI URL annotation offsets select the rendered link already present in the output, not the supported claim.
    return MarkerCitationAnchor(start=start, end=end) if 0 <= start < end <= len(text) else None


def responses_citation_annotations(
    annotations: Sequence[responses.response_output_text.Annotation],
) -> list[responses.response_output_text.AnnotationURLCitation | responses.response_output_text.AnnotationFileCitation]:
    """Return the valid URL and file citations.

    The SDK doesn't validate responses, so it builds an annotation of a type it doesn't know, or with a missing field,
    into one of its classes.
    """
    citations: list[
        responses.response_output_text.AnnotationURLCitation | responses.response_output_text.AnnotationFileCitation
    ] = []
    for annotation in annotations:
        try:
            annotation = responses_output_text_annotation_ta.validate_python(annotation.model_dump(warnings=False))
        except ValidationError:
            continue
        if isinstance(
            annotation,
            responses.response_output_text.AnnotationURLCitation
            | responses.response_output_text.AnnotationFileCitation,
        ):
            citations.append(annotation)
    return citations


def map_responses_citations(
    annotations: Sequence[responses.response_output_text.Annotation],
    text: str,
) -> list[Citation] | None:
    citations: list[Citation] = []
    for annotation in responses_citation_annotations(annotations):
        if isinstance(annotation, responses.response_output_text.AnnotationURLCitation):
            start, end = annotation.start_index, annotation.end_index
            citations.append(
                Citation(
                    sources=[WebCitationSource(url=annotation.url, title=annotation.title)],
                    anchor=marker_citation_anchor(start, end, text),
                )
            )
        elif annotation.file_id or annotation.filename:  # a file citation naming no file has no source to show
            citations.append(
                Citation(
                    sources=[DocumentCitationSource(document_id=annotation.file_id, title=annotation.filename)],
                )
            )
    return citations or None


def map_responses_citations_for_replay(
    citations: object, text: str
) -> list[responses.response_output_text_param.Annotation]:
    """Rebuild the URL and file citation annotations OpenAI returned from their raw data on the text part.

    Annotations that can't be rebuilt are left out.
    """
    annotations: list[responses.response_output_text_param.Annotation] = []
    for citation in cast(list[Any], citations) if isinstance(citations, list) else []:
        if not _utils.is_str_dict(citation):
            continue
        if citation.get('type') == 'url_citation':
            url, title = citation.get('url'), citation.get('title')
            start, end = citation.get('start_index'), citation.get('end_index')
            # The range covers the link the model wrote into its answer, so it must still contain the URL. OpenAI
            # accepts a range that has moved or runs past the text, so a range edited history moved is left out here.
            if (
                isinstance(url, str)
                and url  # OpenAI rejects an empty URL or file ID
                and isinstance(title, str)
                and isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(text)
                and url in text[start:end]
            ):
                annotations.append(
                    responses.response_output_text_param.AnnotationURLCitation(
                        type='url_citation', url=url, title=title, start_index=start, end_index=end
                    )
                )
        elif citation.get('type') == 'file_citation':
            file_id, filename, index = citation.get('file_id'), citation.get('filename'), citation.get('index')
            if isinstance(file_id, str) and file_id and isinstance(filename, str) and isinstance(index, int):
                annotations.append(
                    responses.response_output_text_param.AnnotationFileCitation(
                        type='file_citation', file_id=file_id, filename=filename, index=index
                    )
                )
    return annotations


def responses_citations_provider_details(
    annotations: Sequence[responses.response_output_text.Annotation],
) -> dict[str, Any]:
    return {
        'citations': [
            annotation.model_dump(mode='json', warnings=False)
            for annotation in responses_citation_annotations(annotations)
        ]
    }


def chat_citations_provider_details(annotations: Sequence[BaseModel]) -> dict[str, Any]:
    citations = [
        annotation for annotation in annotations if isinstance(annotation, chat.chat_completion_message.Annotation)
    ]
    return {'citations': chat_annotations_ta.dump_python(citations, warnings=False)}


def map_chat_citations(annotations: Sequence[BaseModel], text: str) -> list[Citation] | None:
    citations: list[Citation] = []
    for annotation in annotations:
        # OpenRouter's annotation list can also hold its own file annotations, which aren't citations.
        if not isinstance(annotation, chat.chat_completion_message.Annotation):
            continue
        url_citation = annotation.url_citation
        start, end = url_citation.start_index, url_citation.end_index
        content = (url_citation.model_extra or {}).get('content')
        citations.append(
            Citation(
                sources=[
                    WebCitationSource(
                        url=url_citation.url,
                        title=url_citation.title,
                        excerpts=[content] if isinstance(content, str) and content else [],
                    )
                ],
                anchor=marker_citation_anchor(start, end, text),
            )
        )
    return citations or None


def map_chat_content(
    message: chat.ChatCompletionMessage,
    profile: OpenAIModelProfile,
    provider_name: str,
    *,
    include_raw_annotations: bool = False,
) -> list[TextPart | ThinkingPart]:
    content_parts = split_content_into_text_and_thinking(
        message.content or '', profile.get('thinking_tags', DEFAULT_THINKING_TAGS)
    )
    if message.annotations:
        attach_chat_annotations(
            content_parts, message.annotations, provider_name, include_raw_annotations=include_raw_annotations
        )
    return [
        replace(part, id='content', provider_name=provider_name) if isinstance(part, ThinkingPart) else part
        for part in content_parts
    ]


def attach_chat_annotations(
    content_parts: Sequence[TextPart | ThinkingPart],
    annotations: Sequence[BaseModel],
    provider_name: str,
    *,
    include_raw_annotations: bool,
) -> None:
    """Attach a Chat Completions message's annotations to the text parts its content was split into."""
    text_parts = [part for part in content_parts if isinstance(part, TextPart)]
    if len(content_parts) == 1 and text_parts:
        if citations := map_chat_citations(annotations, text_parts[0].content):
            text_parts[0].citations = citations
            text_parts[0].provider_name = provider_name
            text_parts[0].provider_details = chat_citations_provider_details(annotations)
    # Annotation positions count into OpenAI's original text, so they can't be used once thinking tags are removed
    # from it. `openai_include_raw_annotations` still keeps them.
    if include_raw_annotations and text_parts:
        text_parts[-1].provider_name = provider_name
        text_parts[-1].provider_details = {
            **(text_parts[-1].provider_details or {}),
            # Dumped one by one, so OpenRouter's own annotation types keep their fields.
            'annotations': [annotation.model_dump(mode='json') for annotation in annotations],
        }
