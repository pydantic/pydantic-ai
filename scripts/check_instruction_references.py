#!/usr/bin/env python
"""Fail when an agent instruction names a repository path or a skill entry that does not exist.

An instruction file names a path in a code span (`tests/AGENTS.md`). A path that starts at a top-level entry of the
repository must name a tracked file or directory, from the repository root or from inside it: a package guide names
its own `docs/`, and `agent_docs/complexity.md` names `pydantic_graph/graph_builder.py` inside its package. Paths git
ignores, such as branch-context state, are runtime files and pass. A code span that is the text of a link to a URL
names that URL's file and is skipped.

Each shared skill under `.agents/skills/` needs its `.claude/skills/` entry, or Claude Code never loads it. The
retired `pydantic/pydantic-ai-harness` repository may appear only in issue and pull request links, which still
resolve.

Pre-commit runs it on every commit, because moving a file can break a path an instruction names.

Usage:
    python scripts/check_instruction_references.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath

# The file set `scripts/check_glossary_spellings.py` reads: every instruction file an agent loads.
INSTRUCTION_FILE = re.compile(r'(^|/)(AGENTS|CLAUDE|SKILL)\.md$|(^|/)(agent_docs|\.agents|\.claude)/.*\.md$')
# Skills the packages ship to their users describe the user's repository, not this one.
SHIPPED_SKILLS = ('pydantic_ai_slim/pydantic_ai/.agents/', 'src/pydantic_ai_harness/pydantic_ai_harness/.agents/')
RETIRED_SLUG = re.compile(r'pydantic/pydantic-ai-harness(?!/(?:issues|pull)/\d)(?![\w-])')

_FENCED_BLOCK = re.compile(r'^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$', re.MULTILINE | re.DOTALL)
_URL_LINK_TEXT = re.compile(r'\[([^\]\n]*)\]\([a-z][a-z0-9+.-]*:[^)\n]*\)')
# A code span may wrap onto the next line, but never across a blank line.
_CODE_SPAN = re.compile(r'(`+)((?:(?!\n[ \t]*\n)[\s\S])+?)\1')
# A path-shaped span: no spaces, placeholders, globs, test node ids or URL schemes, and it ends in a file extension or
# a directory slash.
_PATH = re.compile(r'[\w.-]+(?:/[\w.-]+)*(?:\.\w{1,5}|/)')

Read = Callable[[str], str]
Ignored = Callable[[Iterable[str]], set[str]]


def _blank(match: re.Match[str]) -> str:
    """Blank out a match but keep its newlines, so line numbers still match the file."""
    return re.sub(r'[^\n]', ' ', match[0])


def named_paths(text: str) -> list[tuple[int, str]]:
    """The path-shaped code spans in `text`, with their line numbers."""
    text = _FENCED_BLOCK.sub(_blank, text)
    text = _URL_LINK_TEXT.sub(lambda m: '[' + re.sub(r'[^\n]', ' ', m[1]) + m[0][len(m[1]) + 1 :], text)
    return [
        (text.count('\n', 0, match.start()) + 1, span)
        for match in _CODE_SPAN.finditer(text)
        if '/' in (span := match[2].strip()) and _PATH.fullmatch(span)
    ]


def find_problems(tracked: Sequence[str], instructions: Sequence[str], read: Read, ignored: Ignored) -> list[str]:
    """Every unresolved reference in `instructions`, resolved against the `tracked` paths."""
    entries = {*tracked, *(str(parent) for path in tracked for parent in PurePosixPath(path).parents)}
    # Each entry and every trailing run of its components, so a path resolves from inside the repository too.
    known = {'/'.join(parts[i:]) for entry in entries if (parts := entry.split('/')) for i in range(len(parts))}
    top = {path.split('/')[0] for path in tracked}

    def exists(path: str) -> bool:
        return path.rstrip('/') in known

    problems: list[str] = []
    missing: list[tuple[str, int, str]] = []
    for name in instructions:
        text = read(name)
        for match in RETIRED_SLUG.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            problems.append(f'{name}:{line}: link to `pydantic/pydantic-ai`, not the retired `{match[0]}`')
        if name.startswith(SHIPPED_SKILLS):
            continue
        for line, path in named_paths(text):
            if path.split('/')[0] in top and not exists(path):
                missing.append((name, line, path))
    runtime = ignored({path for _, _, path in missing})
    problems += [f'{name}:{line}: `{path}` does not exist' for name, line, path in missing if path not in runtime]

    skills = {path.split('/')[2] for path in tracked if path.startswith('.agents/skills/') and path.count('/') > 2}
    problems += [
        f'.agents/skills/{skill}: add the `.claude/skills/{skill}` symlink so Claude Code loads the skill'
        for skill in sorted(skills)
        if not exists(f'.claude/skills/{skill}')
    ]
    return problems


def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(('git', *args), input=stdin, capture_output=True, text=True, check=False).stdout


def main() -> int:
    """Print each unresolved reference in the working tree's instruction files; return 1 when there is one."""
    tracked = [path for path in _git('ls-files', '-z').split('\0') if path]
    # A symlinked `CLAUDE.md` or skill entry is checked through the file it points to.
    instructions = [path for path in tracked if INSTRUCTION_FILE.search(path) and not Path(path).is_symlink()]
    problems = find_problems(
        tracked,
        instructions,
        lambda path: Path(path).read_text(encoding='utf-8', errors='replace'),
        lambda paths: set(_git('check-ignore', '--no-index', '--stdin', stdin='\n'.join(paths)).splitlines()),
    )
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
