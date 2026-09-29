"""Discover and parse Agent Skill packages from local libraries."""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

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


def _extract_frontmatter(text: str, source: Path) -> tuple[str, str]:
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


def _parse_frontmatter(frontmatter: str, source: Path) -> _SkillFrontmatter:
    try:
        import yaml
    except ImportError:  # pragma: no cover - exercised in an environment without the skills extra
        raise ImportError(
            'PyYAML is required to load Agent Skills. Install it with: pip install "pydantic-ai-harness[skills]"'
        ) from None

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


def _discover_skills(libraries: Sequence[Path]) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for library in libraries:
        with os.scandir(library) as children:
            for child in sorted(children, key=lambda entry: entry.name):
                if not child.is_dir():
                    continue
                skill_file = Path(child.path) / 'SKILL.md'
                if skill_file.is_file():
                    discovered.append((_normalize_name(child.name), skill_file))
    return discovered


def _validate_name(name: str, source: Path) -> str:
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


def load_skill(skill_file: Path) -> SkillDefinition:
    """Load one `SKILL.md` into a validated construction-time definition."""
    frontmatter_text, body = _extract_frontmatter(skill_file.read_text(encoding='utf-8'), skill_file)
    frontmatter = _parse_frontmatter(frontmatter_text, skill_file)

    directory_name = skill_file.parent.name
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
    )


def load_skill_libraries(
    directories: Sequence[str | Path],
    *,
    include: Collection[str] | None,
    exclude: Collection[str],
) -> tuple[SkillDefinition, ...]:
    """Discover immediate child skill packages under configured directories."""
    libraries: list[Path] = []
    seen_libraries: set[Path] = set()
    for configured in directories:
        library = Path(configured)
        if not library.exists():
            raise ValueError(f'Skill library directory does not exist: {library}')
        if not library.is_dir():
            raise ValueError(f'Skill library path is not a directory: {library}')
        if (library / 'SKILL.md').is_file():
            raise ValueError(
                f'Skill library path points to a skill package: {library}. Pass its parent directory instead.'
            )
        resolved = library.resolve()
        if resolved in seen_libraries:
            continue
        seen_libraries.add(resolved)
        libraries.append(library)

    discovered = _discover_skills(libraries)
    available_names = frozenset(name for name, _ in discovered)
    normalized_include = None if include is None else frozenset(_normalize_name(name) for name in include)
    normalized_exclude = frozenset(_normalize_name(name) for name in exclude)
    _validate_selection('include', normalized_include, available_names)
    _validate_selection('exclude', normalized_exclude, available_names)
    selected_names = (
        normalized_include if normalized_include is not None else available_names.difference(normalized_exclude)
    )

    selected_files: list[Path] = []
    paths_by_name: dict[str, Path] = {}
    for name, skill_file in discovered:
        if name not in selected_names:
            continue
        if previous := paths_by_name.get(name):
            raise ValueError(f'Duplicate skill name {name!r}: {previous} and {skill_file}.')
        paths_by_name[name] = skill_file
        selected_files.append(skill_file)

    return tuple(load_skill(skill_file) for skill_file in selected_files)


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
