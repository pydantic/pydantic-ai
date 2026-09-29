"""Discover and parse Agent Skill packages from libraries in the run's workspace."""

from __future__ import annotations

import posixpath
import unicodedata
import warnings
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from pydantic_ai.workspaces import FileEntry, Workspace
from pydantic_ai_harness._workspace import workspace_path

# Imported with the module rather than on first parse: a durable engine such as Temporal parses skills in
# workflow code, where importing a module for the first time fails the workflow task.
try:
    import yaml
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'PyYAML is required to load Agent Skills. Install it with: pip install "pydantic-ai-harness[skills]"'
    ) from _import_error

# These fields affect invocation, permissions, model selection, execution, or
# prompt rendering in clients that implement them. Skills accepts their files
# for compatibility but reports that the behavior is not active.
_BEHAVIORAL_FRONTMATTER_FIELDS = frozenset(
    {
        'agent',
        'allowed-tools',
        'argument-hint',
        'arguments',
        'context',
        'dependencies',
        'disable-model-invocation',
        'disallowed-tools',
        'effort',
        'hooks',
        'model',
        'paths',
        'shell',
        'tools',
        'user-invocable',
        'when_to_use',
    }
)


class _SkillFrontmatter(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str | None = None
    description: str

    @field_validator('description', mode='after')
    @classmethod
    def _strip_description(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError('must not be empty')
        return stripped


@dataclass(frozen=True)
class SkillDefinition:
    """Validated skill metadata and body from a single `SKILL.md`."""

    name: str
    description: str
    body: str
    ignored_behavioral_fields: tuple[str, ...]
    path: str
    """Absolute workspace path of the `SKILL.md` it was read from."""


def _extract_frontmatter(text: str, source: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'{source} must start with YAML frontmatter delimited by `---`.')

    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == '---'), None)
    if closing is None:
        raise ValueError(f'{source} has unclosed YAML frontmatter.')

    frontmatter = '\n'.join(lines[1:closing])
    body_lines = lines[closing + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = '\n'.join(body_lines)
    return frontmatter, body


def _parse_frontmatter(frontmatter: str, source: str) -> _SkillFrontmatter:
    # Agent Skills frontmatter fields are strings. BaseLoader preserves valid
    # scalar names such as `123` and `on` instead of applying YAML implicit types.
    # PyYAML otherwise also accepts duplicate mapping keys and keeps the last value.
    class UniqueKeyLoader(yaml.BaseLoader):
        def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, object]:
            keys: set[str] = set()
            for key_node, _ in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        'found a non-scalar key',
                        key_node.start_mark,
                    )
                key: str = key_node.value
                if key in keys:
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        f'found duplicate key {key!r}',
                        key_node.start_mark,
                    )
                keys.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        parsed: object = yaml.load(frontmatter, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML frontmatter in {source}: {exc}') from exc

    if not isinstance(parsed, dict):
        raise ValueError(f'YAML frontmatter in {source} must be a mapping.')
    try:
        return _SkillFrontmatter.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(f'Invalid Agent Skill frontmatter in {source}: {exc}') from exc


def _normalize_name(name: str) -> str:
    return unicodedata.normalize('NFKC', name)


async def _stat(workspace: Workspace, path: str) -> FileEntry | None:
    try:
        return await workspace.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


async def _is_file(workspace: Workspace, path: str) -> bool:
    entry = await _stat(workspace, path)
    return entry is not None and not entry.is_dir


async def _discover_skills(workspace: Workspace, libraries: Sequence[str]) -> list[tuple[str, str]]:
    discovered: list[tuple[str, str]] = []
    for library in libraries:
        # Sorted by name so the catalog, and so the instructions, are the same for every run over the same files.
        for child in sorted(await workspace.list_dir(library), key=lambda entry: entry.name):
            if not child.is_dir:
                continue
            skill_file = posixpath.join(library, child.name, 'SKILL.md')
            if await _is_file(workspace, skill_file):
                discovered.append((_normalize_name(child.name), skill_file))
    return discovered


def _validate_name(name: str, source: str) -> str:
    normalized = _normalize_name(name)
    if (
        not normalized
        or len(normalized) > 64
        or normalized != normalized.lower()
        or normalized.startswith('-')
        or normalized.endswith('-')
        or '--' in normalized
        or not all(character.isalnum() or character == '-' for character in normalized)
    ):
        raise ValueError(
            f'Invalid skill name {name!r} in {source}; expected at most 64 lowercase Unicode letters or numbers '
            'and single hyphens, without a leading or trailing hyphen.'
        )
    return normalized


def parse_skill(text: str, skill_file: str) -> SkillDefinition:
    """Parse one `SKILL.md`, read from `skill_file`, into a validated definition."""
    frontmatter_text, body = _extract_frontmatter(text, skill_file)
    frontmatter = _parse_frontmatter(frontmatter_text, skill_file)

    directory_name = posixpath.basename(posixpath.dirname(skill_file))
    name = frontmatter.name if frontmatter.name is not None else directory_name
    normalized_name = _validate_name(name, skill_file)
    if frontmatter.name is not None and normalized_name != _normalize_name(directory_name):
        raise ValueError(f'Skill name {name!r} in {skill_file} must match its parent directory {directory_name!r}.')

    ignored_fields = tuple(sorted(_BEHAVIORAL_FRONTMATTER_FIELDS.intersection(frontmatter.model_extra or {})))
    return SkillDefinition(
        name=normalized_name,
        description=frontmatter.description,
        body=body,
        ignored_behavioral_fields=ignored_fields,
        path=skill_file,
    )


async def load_skill_libraries(
    workspace: Workspace,
    directories: Sequence[str | Path],
    *,
    include: Collection[str] | None,
    exclude: Collection[str],
) -> tuple[SkillDefinition, ...]:
    """Discover immediate child skill packages under configured directories in `workspace`.

    Relative directories resolve against the workspace's working directory.
    """
    libraries: list[str] = []
    for configured in directories:
        library = await workspace.resolve(workspace_path(Path(configured)))
        if library in libraries:
            continue
        entry = await _stat(workspace, library)
        if entry is None:
            raise ValueError(f'Skill library directory does not exist in the workspace: {configured}')
        if not entry.is_dir:
            raise ValueError(f'Skill library path is not a directory: {configured}')
        if await _is_file(workspace, posixpath.join(library, 'SKILL.md')):
            raise ValueError(
                f'Skill library path points to a skill package: {configured}. Pass its parent directory instead.'
            )
        libraries.append(library)

    discovered = await _discover_skills(workspace, libraries)
    available_names = frozenset(name for name, _ in discovered)
    normalized_include = None if include is None else frozenset(_normalize_name(name) for name in include)
    normalized_exclude = frozenset(_normalize_name(name) for name in exclude)
    _validate_selection('include', normalized_include, available_names)
    _validate_selection('exclude', normalized_exclude, available_names)
    selected_names = (
        normalized_include if normalized_include is not None else available_names.difference(normalized_exclude)
    )

    selected_files: list[str] = []
    paths_by_name: dict[str, str] = {}
    for name, skill_file in discovered:
        if name not in selected_names:
            continue
        if previous := paths_by_name.get(name):
            raise ValueError(f'Duplicate skill name {name!r}: {previous} and {skill_file}.')
        paths_by_name[name] = skill_file
        selected_files.append(skill_file)

    parsed: list[SkillDefinition] = []
    for skill_file in selected_files:
        try:
            try:
                text = await workspace.read_text(skill_file)
            except UnicodeDecodeError as error:
                raise ValueError(f'{skill_file} is not valid UTF-8: {error}') from error
            parsed.append(parse_skill(text, skill_file))
        except ValueError as error:
            # A model-editable skill should not make unrelated valid skills unusable.
            warnings.warn(f'Skipping {skill_file}: {error}', UserWarning, stacklevel=2)
    return tuple(parsed)


def _validate_selection(
    name: str,
    selected: Collection[str] | None,
    available: Collection[str],
) -> None:
    if selected is None:
        return
    unknown = sorted(set(selected).difference(available))
    if not unknown:
        return
    noun = 'skill' if len(unknown) == 1 else 'skills'
    available_text = ', '.join(sorted(available)) or '(none)'
    raise ValueError(f'Unknown {noun} in {name}: {", ".join(unknown)}. Available skills: {available_text}.')
