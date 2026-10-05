"""Discover and parse Agent Skill packages from libraries in the run's workspace."""

from __future__ import annotations

import posixpath
import unicodedata
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

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


MissingDirectories: TypeAlias = Literal['error', 'skip']
"""What `Skills` does with a library directory that does not exist: fail the run, or skip it."""

DuplicateNames: TypeAlias = Literal['error', 'keep_first']
"""What `Skills` does when two valid skills with different `SKILL.md` files share a name: fail the run, or keep the first with a warning."""

_ARGUMENTS_PLACEHOLDER = '$ARGUMENTS'


@dataclass(frozen=True, kw_only=True)
class SkillDefinition:
    """Validated skill metadata and body from a single `SKILL.md`."""

    name: str
    """The skill's name: its `name` frontmatter field, or else its directory's name."""

    description: str
    """The `description` frontmatter field, listed in the model's catalog."""

    body: str
    """The Markdown after the frontmatter, without leading and trailing blank lines."""

    ignored_behavioral_fields: tuple[str, ...]
    """Frontmatter fields such as `allowed-tools` whose behavior `Skills` does not implement."""

    path: str
    """Absolute workspace path of the `SKILL.md` it was read from."""

    in_run_workspace: bool = True
    """Whether it was read from the run's workspace, where the model's file and shell tools can reach its directory.

    `False` for a library read from `Skills(workspace=...)`; `render` then leaves the directory out.
    """

    @property
    def directory(self) -> str:
        """Absolute workspace path of the skill's directory, which holds any `scripts/` or `references/`."""
        return posixpath.dirname(self.path)

    def render(self, arguments: str | None = None) -> str:
        """The skill's instructions as the model receives them: a heading, the skill's directory, and the body.

        Pass `arguments` when a user invokes the skill with a command such as `/code-review src/app.py`:
        every `$ARGUMENTS` in the body becomes `arguments`, and a body without `$ARGUMENTS` gets
        `ARGUMENTS: <arguments>` appended. Other placeholders, such as Claude Code's indexed `$0`, are
        left unchanged. Without `arguments`, the result is what `load_capability` returns.

        The directory line is left out unless `in_run_workspace`.
        """
        body = self.body
        if arguments is not None:
            if _ARGUMENTS_PLACEHOLDER in body:
                body = body.replace(_ARGUMENTS_PLACEHOLDER, arguments)
            elif arguments:
                body = '\n\n'.join(part for part in (body, f'ARGUMENTS: {arguments}') if part)
        heading = f'# Skill: {self.name}'
        if self.in_run_workspace:
            heading += f'\n\nSkill directory: `{self.directory}`. Relative paths in this skill resolve against it.'
        return f'{heading}\n\n{body}' if body else heading


@dataclass(frozen=True, kw_only=True)
class SkillCatalog:
    """The skills `Skills.load` read, and what it skipped or ignored on the way."""

    skills: tuple[SkillDefinition, ...]
    """The selected, valid skills, in catalog order."""

    skipped: tuple[str, ...]
    """One message per `SKILL.md` left out: malformed, or named like a valid skill found earlier.

    A run emits each as a `UserWarning`; a host showing skills to a person can show them instead.
    """


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


async def _libraries(
    workspace: Workspace, directories: Sequence[str | Path], missing_directories: MissingDirectories
) -> list[str]:
    """The configured libraries that exist, in order, each listed once."""
    libraries: list[str] = []
    for configured in directories:
        library = await workspace.resolve(workspace_path(Path(configured)))
        if library in libraries:
            continue
        entry = await _stat(workspace, library)
        if entry is None:
            if missing_directories == 'skip':
                continue
            raise ValueError(f'Skill library directory does not exist in the workspace: {configured}')
        if not entry.is_dir:
            raise ValueError(f'Skill library path is not a directory: {configured}')
        if await _is_file(workspace, posixpath.join(library, 'SKILL.md')):
            raise ValueError(
                f'Skill library path points to a skill package: {configured}. Pass its parent directory instead.'
            )
        libraries.append(library)
    return libraries


async def same_skill(first: tuple[Workspace, str], second: tuple[Workspace, str]) -> bool:
    """Whether two `SKILL.md` files, each in its workspace, hold one skill: the same file, or a byte-identical copy.

    The same file is recognized through symlinks within one workspace. Only `SKILL.md` is compared,
    not bundled files. Asked only when two skills share a name, so a catalog without clashes costs
    no extra round trips.
    """
    (first_workspace, first_path), (second_workspace, second_path) = first, second
    if first_workspace is second_workspace and await first_workspace.realpath(
        first_path
    ) == await second_workspace.realpath(second_path):
        return True
    return await first_workspace.read_bytes(first_path) == await second_workspace.read_bytes(second_path)


def skip_duplicate_name(name: str, kept: str, skipped: str, duplicate_names: DuplicateNames) -> str:
    """The warning for `skipped`, whose name an earlier `SKILL.md`, `kept`, has; raises when duplicates are errors."""
    if duplicate_names == 'error':
        raise ValueError(f'Duplicate skill name {name!r}: {kept} and {skipped}.')
    return f'Skipping {skipped}: skill name {name!r} is already taken by {kept}.'


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
    missing_directories: MissingDirectories,
    duplicate_names: DuplicateNames,
) -> tuple[list[SkillDefinition], list[str]]:
    """Discover immediate child skill packages under configured directories in `workspace`.

    Relative directories resolve against the workspace's working directory. Returns the parsed
    skills and a message for each skipped `SKILL.md`. A skill found twice, through a symlinked
    library or skill directory or as a byte-identical `SKILL.md`, counts once. Names are compared
    after parsing, so an invalid `SKILL.md` does not hide a valid one with the same name.
    """
    libraries = await _libraries(workspace, directories, missing_directories)
    discovered = await _discover_skills(workspace, libraries)
    available_names = frozenset(name for name, _ in discovered)
    normalized_include = None if include is None else frozenset(_normalize_name(name) for name in include)
    normalized_exclude = frozenset(_normalize_name(name) for name in exclude)
    _validate_selection('include', normalized_include, available_names)
    _validate_selection('exclude', normalized_exclude, available_names)
    selected_names = (
        normalized_include if normalized_include is not None else available_names.difference(normalized_exclude)
    )

    skipped: list[str] = []
    by_name: dict[str, SkillDefinition] = {}
    for name, skill_file in discovered:
        if name not in selected_names:
            continue
        try:
            try:
                text = await workspace.read_text(skill_file)
            except UnicodeDecodeError as error:
                raise ValueError(f'{skill_file} is not valid UTF-8: {error}') from error
            skill = parse_skill(text, skill_file)
        except ValueError as error:
            # A model-editable skill should not make unrelated valid skills unusable.
            skipped.append(f'Skipping {skill_file}: {error}')
            continue
        if (previous := by_name.get(skill.name)) is None:
            by_name[skill.name] = skill
        elif not await same_skill((workspace, previous.path), (workspace, skill_file)):
            skipped.append(skip_duplicate_name(skill.name, previous.path, skill_file, duplicate_names))
    return list(by_name.values()), skipped


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
