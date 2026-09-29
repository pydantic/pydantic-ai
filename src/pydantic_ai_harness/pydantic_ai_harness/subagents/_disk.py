"""Load sub-agent definitions from markdown files on disk.

A definition is a markdown file with optional YAML-style frontmatter:

```markdown
---
name: researcher
description: Researches a topic and reports findings
tools: Read, Grep
---
You research topics. Report findings with sources.
```

The frontmatter is parsed by a small, dependency-free reader limited to the keys
coding assistants write (`name`, `description`, `model`, `color`, and `tools` or
`allowed-tools`); `pyyaml` is not a runtime dependency of harness. The body after
the frontmatter is the agent's instructions. `model` and `color` are ignored: the
model is inherited from the parent (overridable via `SubAgents.agent_overrides`),
and `color` has no pyai equivalent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ThinkingLevel


@dataclass(frozen=True)
class AgentOverride:
    """Per-agent override for a disk-loaded sub-agent, keyed by the agent's name.

    Both fields are optional. An unset `model` inherits the parent run's model; an
    unset `effort` runs at the capability's minimum effort floor (see
    `clamp_effort`).
    """

    model: Model | KnownModelName | str | None = None
    """Model to run this disk agent with, in place of inheriting the parent's."""

    effort: ThinkingLevel | None = None
    """Thinking/reasoning level for this disk agent. Raised to at least the floor."""


@dataclass(frozen=True)
class ParsedAgent:
    """One parsed agent definition: frontmatter fields plus the markdown body."""

    name: str | None
    description: str | None
    tools: tuple[str, ...]
    body: str


def _strip_quotes(value: str) -> str:
    """Drop a single layer of matching single or double quotes from a scalar."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _parse_frontmatter(lines: Sequence[str]) -> dict[str, str | list[str]]:
    """Parse `key: value` and block-list (`- item`) frontmatter lines.

    Only the shape coding assistants emit is supported: scalar values and, for
    list keys, either a `key: a, b` inline form (handled by callers) or a block
    list of `- item` lines under a key with an empty value.
    """
    result: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    for raw in lines:
        if not raw.strip():
            continue
        stripped = raw.lstrip()
        if current_list_key is not None and stripped.startswith('- '):
            item = stripped[2:].strip()
            existing = result[current_list_key]
            if isinstance(existing, list) and item:
                existing.append(item)
            continue
        if ':' not in raw:
            current_list_key = None
            continue
        key, _, value = raw.partition(':')
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = _strip_quotes(value)
            current_list_key = None
        else:
            result[key] = []
            current_list_key = key
    return result


def _parse_tools(fields: dict[str, str | list[str]]) -> tuple[str, ...]:
    """Read the `tools` or `allowed-tools` key as a tuple of tool-name strings."""
    raw = fields.get('tools')
    if raw is None:
        raw = fields.get('allowed-tools')
    if raw is None:
        return ()
    if isinstance(raw, list):
        return tuple(item for item in raw if item)
    return tuple(name.strip() for name in raw.split(',') if name.strip())


def parse_agent_markdown(text: str) -> ParsedAgent:
    """Parse a markdown agent definition into frontmatter fields and a body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        return ParsedAgent(None, None, (), text.strip())
    closing: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == '---':
            closing = index
            break
    if closing is None:
        return ParsedAgent(None, None, (), text.strip())
    fields = _parse_frontmatter(lines[1:closing])
    body = '\n'.join(lines[closing + 1 :]).strip()
    name = fields.get('name')
    description = fields.get('description')
    return ParsedAgent(
        name=name if isinstance(name, str) else None,
        description=description if isinstance(description, str) else None,
        tools=_parse_tools(fields),
        body=body,
    )


def _convention_folder(root: Path, leaf: str) -> Path:
    """`<root>/.agents/<leaf>` when `.agents/` exists, else the `.claude/` equivalent."""
    if (root / '.agents').is_dir():
        return root / '.agents' / leaf
    return root / '.claude' / leaf


def resolve_folders(agent_folders: str | Sequence[Path], cwd: Path, home: Path) -> list[Path]:
    """Resolve the configured disk source into a precedence-ordered list of folders.

    - a `str`: the convention `<root>/.agents/<str>/` (or `.claude/<str>/`) for the
      project root (`cwd`) then the home root, project first.
    - a sequence of paths: those folders verbatim, in order.

    Folders resolving to the same absolute path are deduped (keeping the first), so
    a project root equal to the home root does not scan and warn about every agent
    twice.
    """
    if isinstance(agent_folders, str):
        folders = [_convention_folder(cwd, agent_folders), _convention_folder(home, agent_folders)]
    else:
        folders = list(agent_folders)
    seen: dict[Path, Path] = {}
    for folder in folders:
        seen.setdefault(folder.resolve(), folder)
    return list(seen.values())
