"""Load sub-agent definitions from Markdown or standalone TOML through a workspace.

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

Codex standalone `.toml` definitions require nonempty string `name`, `description`,
and `developer_instructions` fields. Optional `tools` or `allowed-tools` are lists
of nonempty strings or comma-separated strings. Supplying both is rejected.
`model`, `effort`, `model_reasoning_effort`, and `color` are ignored with a warning;
model and effort overrides belong in `agent_overrides`. All other fields cause
that file to be skipped, including permission/sandbox settings and the old
`[agents.name] config_file` format. No configuration is executed or followed.
TOML requires Python 3.11+ for stdlib `tomllib`; Markdown also works on Python 3.10.
"""

from __future__ import annotations

import posixpath
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.workspaces import Workspace


@dataclass(frozen=True)
class AgentOverride:
    """Per-agent override for a disk-loaded sub-agent, keyed by the agent's name.

    Both fields are optional. An unset `model` inherits the parent run's model; an
    unset `effort` leaves the inherited model's thinking setting unchanged.
    """

    model: Model | KnownModelName | str | None = None
    """Model to run this disk agent with, in place of inheriting the parent's."""

    effort: ThinkingLevel | None = None
    """Thinking/reasoning level for this disk agent, passed through unchanged."""


@dataclass(frozen=True)
class ParsedAgent:
    """One parsed agent definition: identity, tool names, and instructions."""

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


@dataclass(frozen=True)
class DiskDefinition:
    """One loaded definition file: the delegate name it resolves to, plus its parsed contents.

    Hashable, so equal definitions share one built `SubAgent` across runs.
    """

    name: str
    parsed: ParsedAgent


def _is_object_list(*, value: object) -> TypeGuard[list[object]]:
    """Narrow TOML arrays without introducing unknown element types."""
    return isinstance(value, list)


def _toml_tools(*, fields: dict[str, object]) -> tuple[str, ...]:
    if 'tools' in fields and 'allowed-tools' in fields:
        raise ValueError('use only one of `tools` and `allowed-tools`')
    if 'tools' not in fields and 'allowed-tools' not in fields:
        return ()
    raw = fields.get('tools', fields.get('allowed-tools'))
    if isinstance(raw, str):
        items: Sequence[object] = raw.split(',')
    elif _is_object_list(value=raw):
        items = raw
    else:
        raise ValueError('`tools` / `allowed-tools` must be a string or list of strings')
    tools: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('tool names must be nonempty strings')
        tools.append(item.strip())
    return tuple(tools)


def _parse_agent_toml(*, text: str, path: str) -> ParsedAgent:
    if sys.version_info < (3, 11):
        raise ValueError('TOML disk agents require Python 3.11+ (`tomllib`)')
    import tomllib

    fields: dict[str, object] = tomllib.loads(text)
    supported = {'name', 'description', 'developer_instructions', 'tools', 'allowed-tools'}
    ignored = {'model', 'effort', 'model_reasoning_effort', 'color'}
    unsupported = fields.keys() - supported - ignored
    if unsupported:
        raise ValueError(
            f'unsupported TOML settings {sorted(unsupported)!r}; permission/sandbox configuration '
            'and `[agents.name] config_file` are not supported'
        )
    required: list[str] = []
    for key in ('name', 'description', 'developer_instructions'):
        value = fields.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'`{key}` must be a nonempty string')
        required.append(value.strip())
    tools = _toml_tools(fields=fields)
    ignored_fields = fields.keys() & ignored
    if ignored_fields:
        warnings.warn(
            f'Ignoring TOML disk sub-agent settings {sorted(ignored_fields)!r} in {path!r}; '
            'use `agent_overrides` for model and effort',
            stacklevel=3,
        )
    return ParsedAgent(name=required[0], description=required[1], tools=tools, body=required[2])


def _warn_unreadable(path: str, exc: Exception) -> None:
    warnings.warn(f'Skipping unreadable disk sub-agent file {path!r}: {exc}', stacklevel=3)


async def _is_dir(workspace: Workspace, path: str) -> bool:
    try:
        return (await workspace.stat(path)).is_dir
    except (FileNotFoundError, NotADirectoryError):
        return False


async def _load_folder(workspace: Workspace, folder: str) -> list[DiskDefinition]:
    """Every `.md` or `.toml` definition directly in `folder`, sorted by name; a missing folder has none.

    Sorted order keeps the roster, and so the prompt listing, the same for every run over the same files.
    """
    if not await _is_dir(workspace, folder):
        return []
    result: list[DiskDefinition] = []
    entries = sorted(await workspace.list_dir(folder), key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_dir or not entry.name.endswith(('.md', '.toml')):
            continue
        try:
            text = await workspace.read_text(entry.path)
        except (OSError, UnicodeDecodeError) as exc:
            _warn_unreadable(entry.path, exc)
            continue
        try:
            parsed = (
                _parse_agent_toml(text=text, path=entry.path)
                if entry.name.endswith('.toml')
                else parse_agent_markdown(text)
            )
        except ValueError as exc:
            warnings.warn(f'Skipping invalid disk sub-agent file {entry.path!r}: {exc}', stacklevel=3)
            continue
        result.append(DiskDefinition(parsed.name or posixpath.splitext(entry.name)[0], parsed))
    return result


async def load_definitions(workspace: Workspace, agent_folders: str | Sequence[str]) -> list[DiskDefinition]:
    """Load definitions through `workspace`, in precedence order.

    - a `str`: `.agents/<str>/`, `.claude/<str>/`, then `.codex/<str>/` under the working directory.
      All are scanned; on a name collision the earlier folder wins.
    - a sequence of workspace paths: those folders in order, each read once (by real path).
    """
    if isinstance(agent_folders, str):
        folders = [posixpath.join(root, agent_folders) for root in ('.agents', '.claude', '.codex')]
    else:
        folders = agent_folders
    result: list[DiskDefinition] = []
    seen: set[str] = set()
    for folder in folders:
        resolved = await workspace.resolve(folder)
        real_path = await workspace.realpath(resolved)
        if real_path in seen:
            continue
        seen.add(real_path)
        result.extend(await _load_folder(workspace, resolved))
    return result
