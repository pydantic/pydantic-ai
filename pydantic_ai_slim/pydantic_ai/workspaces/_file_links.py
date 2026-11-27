"""Files the model links to in its text."""

from __future__ import annotations as _annotations

import ntpath
import posixpath
import re
from urllib.parse import unquote

from pydantic_ai.messages import Citation, FileCitationSource, MarkerCitationAnchor, WorkspaceLocation

_CODE_OR_LINK = re.compile(
    r'(?P<code>(?P<fence>`{3,}|~{3,})[\s\S]*?(?:\n(?P=fence)|\Z)|`[^`\n]+`)'
    r'|!?\[(?P<title>[^\[\]\n]*+)\]\(\s*'
    r'(?:<sandbox:(?P<bracketed>[^>\n]++)>|sandbox:(?P<bare>(?:[^\s()<>]|\([^\s()<>]*+\))++))'
    r'(?:\s+"[^"\n]*+")?\s*\)'
)
"""A fenced code block or inline code span, which is skipped, or a Markdown link or image to a `sandbox:` path:
`[name](sandbox:/report(1).csv)`, `![chart](<sandbox:/a b.png>)`.

Matching both in one pass skips links inside code in linear time. The title can't hold `[`, and the possessive
repeats never give back what they matched, so a reply full of unclosed brackets can't make a match restart its scan.
"""


def file_links(text: str) -> list[Citation]:
    """Citations for the files `text` links to as `[name](sandbox:/absolute/path)`.

    Models write these links when their instructions ask for them, such as
    "To give the user a file, link it as [name](sandbox:/absolute/path)."
    Each path is in the workspace the reply ran in,
    [`ModelResponse.workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref].

    Each citation's anchor is the link's position in `text`, so you can replace the link with your own
    download link. A link inside code, or to a relative path, is left as text.
    Nothing is read, so the file may not exist. The path comes from the model: check it before reading
    it with [`Workspace.in_working_dir`][pydantic_ai.workspaces.Workspace.in_working_dir].
    """
    citations: list[Citation] = []
    for match in _CODE_OR_LINK.finditer(text):
        if match['code'] is not None:
            continue
        path = unquote(match['bracketed'] or match['bare'])
        if not (posixpath.isabs(path) or ntpath.isabs(path)):
            continue
        source = FileCitationSource(WorkspaceLocation(path=path), title=match['title'] or None)
        citations.append(Citation(sources=[source], anchor=MarkerCitationAnchor(start=match.start(), end=match.end())))
    return citations
