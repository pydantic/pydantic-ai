"""Helper functions for working with citations."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .messages import Citation, TextPart, URLCitation


def merge_citations(*citation_lists: list[Citation] | None) -> list[Citation]:
    """Combine multiple citation lists into one.

    Takes any number of citation lists (or None) and merges them all together.
    Skips None values and empty lists.

    Args:
        *citation_lists: One or more lists of citations to merge. Can be None.

    Returns:
        A single list with all citations from all the input lists.
    """
    from .messages import Citation  # noqa: F401, RUF100  # Import here to avoid circular dependencies

    result: list[Citation] = []
    for citation_list in citation_lists:
        if citation_list is not None:
            result.extend(citation_list)
    return result


def validate_citation_indices(citation: URLCitation, content_length: int) -> bool:
    """Check if citation indices are valid for the given content length.

    Makes sure the start/end indices are non-negative, start <= end, and
    end doesn't exceed the content length.

    Args:
        citation: The citation to check.
        content_length: How long the content is.

    Returns:
        True if valid, False otherwise.
    """
    if citation.start_index < 0 or citation.end_index < 0:
        return False
    if citation.start_index > citation.end_index:
        return False
    if citation.end_index > content_length:
        return False
    return True


def map_citation_to_text_part(
    citation: URLCitation,
    text_parts: list[TextPart],
    content_offsets: list[int],
) -> int | None:
    """Figure out which TextPart a citation belongs to.

    Looks at where the citation starts and matches it to the right TextPart
    based on the offsets. The offsets tell us where each TextPart starts
    in the original content.

    Args:
        citation: The citation to map.
        text_parts: List of TextParts to check.
        content_offsets: Where each TextPart starts in the original content.
            First should be 0, then cumulative lengths.

    Returns:
        The index of the matching TextPart, or None if it doesn't match any.
    """
    if len(text_parts) != len(content_offsets):
        raise ValueError('text_parts and content_offsets must have the same length')

    if not text_parts:
        return None

    # Find which part contains the citation's start position
    for i, offset in enumerate(content_offsets):
        part_length = len(text_parts[i].content)
        part_start = offset
        part_end = offset + part_length

        # Citation starts somewhere in this part
        if part_start <= citation.start_index < part_end:
            return i

        # Edge case: citation is exactly at the end of the last part
        if i == len(text_parts) - 1 and citation.start_index == part_end:
            return i

    # Didn't find a match
    return None


def normalize_citation(citation: Citation) -> Citation:
    """Normalize a citation.

    Currently just returns the citation as-is. Can be extended later to
    normalize URLs, fix indices, merge duplicates, etc.

    Args:
        citation: The citation to normalize.

    Returns:
        The citation unchanged for now.
    """
    # TODO: Add normalization - URL cleanup, index validation, etc.
    return citation
