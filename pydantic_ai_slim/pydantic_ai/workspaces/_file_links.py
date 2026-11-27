"""Files the model links to in its text."""

from __future__ import annotations as _annotations

import ntpath
import posixpath
import re
from urllib.parse import unquote

from pydantic_ai.messages import Citation, FileCitationSource, MarkerCitationAnchor, WorkspaceLocation

_CODE_OR_LINK = re.compile(
    r'(?m)'
    r'(?P<code>^[ \t]*+(?P<fence>`{3,}|~{3,})[\s\S]*?(?:\n[ \t]*+(?P=fence)(?:(?<=`)`*+|(?<=~)~*+)[ \t]*+$|\Z)|`[^`\n]++`)'
    r'|!?\[(?P<title>[^\[\]\n]*+)\]\(\s*+'
    r'(?:<sandbox:(?P<bracketed>[^\[<>\n]++)>|sandbox:(?P<bare>(?:[^\[\s()<>]|\([^\[\s()<>]*+\))++))'
    r'(?:[ \t]++"[^\["\n]*+")?[ \t]*+\)'
)
"""A fenced code block or inline code span, which is skipped, or a Markdown link or image to a `sandbox:` path:
`[name](sandbox:/report(1).csv)`, `![chart](<sandbox:/a b.png>)`. A fence opens only at the start of a line, and
may be indented, as in a list. It closes on a line holding only a run of its character at least as long.

The text is model output, which fetched content can steer, so matching must stay linear. Every part of a link
stops at the next `[`, so an attempt that fails scans no further than where the next one starts. A fence either
closes or runs to the end, and an inline span stops at the next backtick. Paths can't contain `[`.
"""


def file_links(text: str) -> list[Citation]:
    """Citations for the files `text` links to as `[name](sandbox:/absolute/path)`.

    Models write these links when their instructions ask for them, such as
    "To give the user a file, link it as [name](sandbox:/absolute/path)."
    Each path is in the workspace the reply ran in,
    [`ModelResponse.workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref].

    Each citation's anchor is the link's position in `text`, so you can replace the link with your own
    download link. A link inside code, or to a relative path or one with a NUL byte, is left as text.
    Nothing is read, so the file may not exist. The path comes from the model: check it before reading
    it with [`Workspace.in_working_dir`][pydantic_ai.workspaces.Workspace.in_working_dir].
    """
    citations: list[Citation] = []
    for match in _CODE_OR_LINK.finditer(text):
        if match['code'] is not None:
            continue
        path = unquote(match['bracketed'] or match['bare'])
        if '\0' in path or not (posixpath.isabs(path) or ntpath.isabs(path)):
            continue
        source = FileCitationSource(WorkspaceLocation(path=path), title=match['title'] or None)
        citations.append(Citation(sources=[source], anchor=MarkerCitationAnchor(start=match.start(), end=match.end())))
    return citations
