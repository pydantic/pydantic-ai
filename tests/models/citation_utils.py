from typing import Any, cast

from pydantic_ai import Citation, ModelMessage, ModelResponse, TextPart


class IsCitationList(list[Citation]):
    """Match a non-empty list containing only citations in snapshots."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, list):  # pragma: no cover
            return False  # pragma: no cover
        citations = cast(list[object], other)
        return bool(citations) and all(isinstance(item, Citation) for item in citations)


class IsUnsupportedCitationDetails(dict[str, Any]):
    """Match provider details containing unsupported citation annotations."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):  # pragma: no cover
            return False  # pragma: no cover
        details = cast(dict[object, object], other)
        return bool(details.get('unsupported_annotations'))


def citations_from_messages(messages: list[ModelMessage]) -> list[Citation]:
    return [
        citation
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart)
        for citation in part.citations or []
    ]
