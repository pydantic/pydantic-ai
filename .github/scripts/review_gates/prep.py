"""The material every gate reads: the pull request's text, its changed files and their functions.

Head and base text come from the contents API and are parsed with `ast`, which reads Python without
running it. TypeScript and Markdown are read line by line.
"""

from __future__ import annotations

import ast
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .github import COMPARE_FILE_LIMIT, GitHub, Issue, PullRequest

_SOURCE_SUFFIXES = frozenset({'.py', '.ts', '.tsx', '.md'})
_CODE_SUFFIXES = ('.py', '.ts', '.tsx')
# Agent-instruction Markdown, whose shell steps locator d reads.
AGENT_INSTRUCTIONS = re.compile(
    r'(^|/)(AGENTS|CLAUDE|SKILL)\.md$|^\.(agents|claude)/|^agent_docs/|^\.github/workflows/.+\.md$'
)
_GENERATED = re.compile(r'(^|/)cassettes/|(^|/)static/assets/|\.lock(\.yml)?$|(^|/)uv\.lock$')
_HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
_CLOSING_REF = re.compile(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b', re.IGNORECASE)
_LINKED_ISSUE_LIMIT = 3
_FETCH_CONCURRENCY = 8


@dataclass(frozen=True)
class DiffLine:
    """One line of a patch: `kind` is `+`, `-` or ` `; `new` is its line in the head, or where it was removed."""

    kind: str
    new: int
    text: str


@dataclass(frozen=True)
class Function:
    """A top-level function or a method, by its line span in one version of a file."""

    qualname: str
    owner: str | None
    start: int
    end: int
    node: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class SourceFile:
    """A changed file: its head and base text and the patch between them."""

    path: str
    head: str | None
    base: str | None
    diff: list[DiffLine]

    @property
    def added(self) -> set[int]:
        return {line.new for line in self.diff if line.kind == '+'}

    @property
    def touched(self) -> set[int]:
        """Head lines the patch adds, or next to which it removes lines."""
        return {line.new for line in self.diff if line.kind != ' '}

    def lines(self, start: int, end: int) -> str:
        """Head lines `start` to `end`, inclusive and 1-based."""
        assert self.head is not None
        return '\n'.join(self.head.splitlines()[start - 1 : end])

    def hunk(self, start: int, end: int) -> str:
        """The patch lines that fall within head lines `start` to `end`."""
        return '\n'.join(f'{line.kind}{line.text}' for line in self.diff if start <= line.new <= end)

    def head_functions(self) -> list[Function]:
        return python_functions(self.head) if self.path.endswith('.py') and self.head else []

    def changed_functions(self) -> list[Function]:
        touched = self.touched
        return [f for f in self.head_functions() if any(f.start <= n <= f.end for n in touched)]


@dataclass
class Prepared:
    """Everything the gates read about one pull request at one head."""

    pull_request: PullRequest
    head_sha: str
    base_sha: str
    issues: list[Issue]
    files: list[SourceFile]
    # What was left out, for the check summary to say so.
    partial: list[str] = field(default_factory=list[str])

    @property
    def labels(self) -> list[str]:
        return [label.name for label in self.pull_request.labels]


def parse_patch(patch: str) -> list[DiffLine]:
    """The lines of a unified diff, each placed at its line in the head."""
    lines: list[DiffLine] = []
    new = 0
    for raw in patch.splitlines():
        if header := _HUNK_HEADER.match(raw):
            new = int(header.group(2))
        elif raw.startswith('+'):
            lines.append(DiffLine('+', new, raw[1:]))
            new += 1
        elif raw.startswith('-'):
            lines.append(DiffLine('-', new, raw[1:]))
        elif raw.startswith(' '):
            lines.append(DiffLine(' ', new, raw[1:]))
            new += 1
    return lines


def python_functions(text: str) -> list[Function]:
    """Module-level functions and methods of module-level classes; `[]` when the text is not valid Python."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    found: list[Function] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(_function(node, None))
        elif isinstance(node, ast.ClassDef):
            found.extend(
                _function(child, node.name)
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return found


def _function(node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str | None) -> Function:
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    assert node.end_lineno is not None
    qualname = f'{owner}.{node.name}' if owner else node.name
    return Function(qualname, owner, start, node.end_lineno, node)


def is_source(path: str) -> bool:
    """Whether a gate reads this file: Python, TypeScript or Markdown, and not generated."""
    return PurePosixPath(path).suffix in _SOURCE_SUFFIXES and not _GENERATED.search(path)


def linked_issue_numbers(body: str) -> list[int]:
    """The issues the body says it closes, fixes or resolves, at most three."""
    return list(dict.fromkeys(int(n) for n in _CLOSING_REF.findall(body)))[:_LINKED_ISSUE_LIMIT]


async def prepare(github: GitHub, pull_request: PullRequest, head_sha: str) -> Prepared:
    """Read the PR's diff against its merge base, and the head and base text of every file a gate reads."""
    comparison = await github.comparison(pull_request.base.ref, head_sha)
    base_sha = comparison.merge_base_commit.sha
    partial: list[str] = []
    if len(comparison.files) >= COMPARE_FILE_LIMIT:
        partial.append(f'only the first {COMPARE_FILE_LIMIT} changed files were read')
    semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)
    changed = [f for f in comparison.files if is_source(f.filename)]
    # Only what a gate reads: Python base text for docstrings, and head text of code and agent instructions.
    # Docs claims come from the patch alone. A task group, so one failed read cancels the rest.
    async with asyncio.TaskGroup() as group:
        heads = [
            group.create_task(
                _text(
                    github,
                    semaphore,
                    f.filename,
                    head_sha
                    if f.status != 'removed'
                    and (f.filename.endswith(_CODE_SUFFIXES) or AGENT_INSTRUCTIONS.search(f.filename))
                    else None,
                )
            )
            for f in changed
        ]
        bases = [
            group.create_task(
                _text(
                    github,
                    semaphore,
                    f.previous_filename or f.filename,
                    base_sha if f.status != 'added' and f.filename.endswith('.py') else None,
                )
            )
            for f in changed
        ]
        issues = [group.create_task(github.issue(n)) for n in linked_issue_numbers(pull_request.body or '')]
    files = [
        SourceFile(f.filename, head.result(), base.result(), parse_patch(f.patch) if f.patch else [])
        for f, head, base in zip(changed, heads, bases, strict=True)
    ]
    linked = [issue for task in issues if (issue := task.result()) is not None]
    if unpatched := [f.filename for f in changed if f.patch is None and f.status != 'removed']:
        partial.append(f'{len(unpatched)} file(s) had no patch from the API: {", ".join(unpatched[:5])}')
    return Prepared(pull_request, head_sha, base_sha, linked, files, partial)


async def _text(github: GitHub, semaphore: asyncio.Semaphore, path: str, ref: str | None) -> str | None:
    if ref is None:
        return None
    async with semaphore:
        return await github.file_text(path, ref)
