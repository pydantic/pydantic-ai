"""Files the model links to in its text."""

from __future__ import annotations as _annotations

import ntpath
import posixpath
import re
from urllib.parse import unquote

from pydantic_ai.messages import Citation, FileCitationSource, MarkerCitationAnchor, WorkspaceLocation, WorkspaceRef

_SANDBOX_LINK = re.compile(
    r'!?\[(?P<title>[^\]\n]*)\]\(\s*'
    r'(?:<sandbox:(?P<bracketed>[^>\n]+)>|sandbox:(?P<bare>(?:[^\s()<>]|\([^\s()<>]*\))+))'
    r'(?:\s+"[^"\n]*")?\s*\)'
)
"""A Markdown link or image to a `sandbox:` path: `[name](sandbox:/report(1).csv)`, `![chart](<sandbox:/a b.png>)`."""

_CODE = re.compile(r'(?P<fence>`{3,}|~{3,})[\s\S]*?(?:\n(?P=fence)|\Z)|`[^`\n]+`')
"""A fenced code block (to its closing fence, or the end of the text) or an inline code span."""


def file_links(text: str, workspace_ref: WorkspaceRef | None) -> list[Citation]:
    """Citations for the files `text` links to as `[name](sandbox:/absolute/path)`.

    Models write these links when their instructions ask for them, such as
    "To give the user a file, link it as [name](sandbox:/absolute/path)."
    Pass the reply's text and the workspace it ran in, `result.response.workspace_ref`.

    Each citation's anchor is the link's position in `text`, so you can replace the link with your own
    download link. A link inside code, or to a relative path, is left as text, and so is every link when
    `workspace_ref` is `None`.
    Nothing is read, so the file may not exist. The path comes from the model: check it before reading
    it with [`Workspace.in_working_dir`][pydantic_ai.workspaces.Workspace.in_working_dir].
    """
    if workspace_ref is None:
        return []
    code = [m.span() for m in _CODE.finditer(text)]
    citations: list[Citation] = []
    for match in _SANDBOX_LINK.finditer(text):
        if any(start <= match.start() < end for start, end in code):
            continue
        path = unquote(match['bracketed'] or match['bare'])
        if not (posixpath.isabs(path) or ntpath.isabs(path)):
            continue
        source = FileCitationSource(WorkspaceLocation(workspace=workspace_ref, path=path), title=match['title'] or None)
        citations.append(Citation(sources=[source], anchor=MarkerCitationAnchor(start=match.start(), end=match.end())))
    return citations
