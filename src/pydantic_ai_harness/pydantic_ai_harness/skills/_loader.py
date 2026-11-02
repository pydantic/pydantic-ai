"""Discover and parse Agent Skill packages from libraries in the run's workspace."""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pydantic_ai.workspaces import FileEntry, Workspace
from pydantic_ai_harness._workspace import workspace_path
from pydantic_ai_harness.artifact_scanning import ArtifactScanResult

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
        'disallowed-tools',
        'effort',
        'hooks',
        'model',
        'paths',
        'shell',
        'tools',
        'when_to_use',
    }
)


class _SkillFrontmatter(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str | None = None
    description: str
    disable_model_invocation: bool = Field(default=False, alias='disable-model-invocation')
    user_invocable: bool = Field(default=True, alias='user-invocable')

    @field_validator('disable_model_invocation', 'user_invocable', mode='before')
    @classmethod
    def _yaml_boolean(cls, value: object) -> object:
        # Frontmatter is loaded without implicit types, so a YAML boolean arrives as text. Only YAML 1.2's
        # spellings count: `yes` or `1` failing the skill is safer than guessing for a field that gates invocation.
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        raise ValueError(f'must be `true` or `false`, not {value!r}')

    @field_validator('description', mode='after')
    @classmethod
    def _strip_description(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError('must not be empty')
        return stripped


MissingDirectories: TypeAlias = Literal['error', 'skip']
"""What `Skills` does with a library path that is not a library: fail the run, or skip it.

A path is not a library when it does not exist, is not a directory, or is itself a skill package.
"""

DuplicateNames: TypeAlias = Literal['error', 'keep_first']
"""What `Skills` does when two valid skills with different `SKILL.md` files share a name.

Fail the run, or keep the first with a warning.
"""

UnknownNames: TypeAlias = Literal['error', 'skip']
"""What `Skills` does with an `include` or `exclude` name that no library has: fail the run, or skip the name.

Skipping suits a selection shared by several projects, such as a person's list of skills to hide.
"""

_ARGUMENTS_PLACEHOLDER = re.compile(r'\$ARGUMENTS(?![\[\w])')
"""`$ARGUMENTS` itself, not Claude Code's indexed `$ARGUMENTS[0]` or a longer name such as `$ARGUMENTS_LIST`."""


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

    model_invocable: bool = True
    """Whether the model may load it; `False` when its frontmatter has `disable-model-invocation: true`.

    A run leaves such a skill out of the model's catalog. `Skills.load` still returns it, for a host
    that lets a person invoke it.
    """

    user_invocable: bool = True
    """Whether a host should let a person invoke it; `False` when its frontmatter has `user-invocable: false`.

    `Skills` itself only offers skills to the model, so this is for a host, such as one with a `/skill-name` command.
    """

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
        `ARGUMENTS: <arguments>` appended. Other placeholders, such as Claude Code's indexed `$0` and
        `$ARGUMENTS[0]`, are left unchanged. Without `arguments`, the result is the instructions the
        model gets when it loads the skill.

        The directory line is left out unless `in_run_workspace`.
        """
        body = self.body
        if arguments is not None:
            if _ARGUMENTS_PLACEHOLDER.search(body):
                # A function, so backslashes in the arguments are not read as group references.
                body = _ARGUMENTS_PLACEHOLDER.sub(lambda _: arguments, body)
            elif arguments:
                body = '\n\n'.join(part for part in (body, f'ARGUMENTS: {arguments}') if part)
        heading = f'# Skill: {self.name}'
        if self.in_run_workspace:
            heading += f'\n\nSkill directory: `{self.directory}`. Relative paths in this skill resolve against it.'
        return f'{heading}\n\n{body}' if body else heading


SkipReason: TypeAlias = Literal['invalid', 'duplicate_name', 'not_a_directory', 'skill_package', 'flagged']
"""Why `Skills` left a path out of the catalog.

- `'invalid'`: a `SKILL.md` with malformed frontmatter, invalid UTF-8, or an invalid or mismatched name.
- `'duplicate_name'`: a `SKILL.md` named like a valid skill found earlier, with `duplicate_names='keep_first'`.
- `'not_a_directory'`: a library path that is a file, with `missing_directories='skip'`.
- `'skill_package'`: a library path that holds a `SKILL.md` itself, with `missing_directories='skip'`.
- `'flagged'`: a skill whose description or body the `Skills(scanner=...)` artifact scan flagged.
"""


@dataclass(frozen=True, kw_only=True)
class SkippedSkill:
    """A `SKILL.md` or skill library path that `Skills` left out, and why."""

    path: str
    """Absolute workspace path of the skipped `SKILL.md`, or of the skipped library."""

    reason: SkipReason
    """Why it was left out."""

    message: str
    """A sentence naming the path and the problem, as a run's `UserWarning` says it."""

    scan: ArtifactScanResult | None = None
    """The artifact scan's findings when `reason` is `'flagged'`, otherwise `None`."""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, kw_only=True)
class SkillCatalog:
    """The skills `Skills.load` read, and each `SKILL.md` or library it left out.

    Run-time advice, such as frontmatter fields whose behavior is not implemented, is not included:
    a run emits it as a `UserWarning`, and each skill's `ignored_behavioral_fields` lists its own.
    """

    skills: tuple[SkillDefinition, ...]
    """The selected, valid skills, in catalog order, including those the model may not load."""

    skipped: tuple[SkippedSkill, ...]
    """Each `SKILL.md` or library path left out, in the order it was found.

    A run emits each `message` as a `UserWarning`; a host showing skills to a person can show them instead.
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
) -> tuple[list[str], list[SkippedSkill]]:
    """The configured libraries that exist, in order, each listed once, and the paths skipped as not libraries."""
    libraries: list[str] = []
    skipped: list[SkippedSkill] = []
    seen: set[str] = set()
    for configured in directories:
        library = await workspace.resolve(workspace_path(Path(configured)))
        if library in seen:
            continue
        seen.add(library)
        entry = await _stat(workspace, library)
        if entry is None:
            if missing_directories == 'skip':
                continue
            raise ValueError(f'Skill library directory does not exist in the workspace: {configured}')
        if entry.is_dir and not await _is_file(workspace, posixpath.join(library, 'SKILL.md')):
            libraries.append(library)
            continue
        # The model's tools can create either mid-session, so with 'skip' they must not fail every later run.
        if missing_directories == 'skip':
            reason: SkipReason = 'skill_package' if entry.is_dir else 'not_a_directory'
            problem = 'it is a skill package, not a library' if entry.is_dir else 'it is not a directory'
            skipped.append(SkippedSkill(path=library, reason=reason, message=f'Skipping {library}: {problem}.'))
        elif entry.is_dir:
            raise ValueError(
                f'Skill library path points to a skill package: {configured}. Pass its parent directory instead.'
            )
        else:
            raise ValueError(f'Skill library path is not a directory: {configured}')
    return libraries, skipped


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


def skip_duplicate_name(name: str, kept: str, skipped: str, duplicate_names: DuplicateNames) -> SkippedSkill:
    """The entry for `skipped`, whose name an earlier `SKILL.md`, `kept`, has; raises when duplicates are errors."""
    if duplicate_names == 'error':
        raise ValueError(f'Duplicate skill name {name!r}: {kept} and {skipped}.')
    return SkippedSkill(
        path=skipped,
        reason='duplicate_name',
        message=f'Skipping {skipped}: skill name {name!r} is already taken by {kept}.',
    )


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
        model_invocable=not frontmatter.disable_model_invocation,
        user_invocable=frontmatter.user_invocable,
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
    unknown_names: UnknownNames,
) -> tuple[list[SkillDefinition], list[SkippedSkill]]:
    """Discover immediate child skill packages under configured directories in `workspace`.

    Relative directories resolve against the workspace's working directory. Returns the parsed
    skills and an entry for each skipped `SKILL.md` or library path. A skill found twice, through a symlinked
    library or skill directory or as a byte-identical `SKILL.md`, counts once. Names are compared
    after parsing, so an invalid `SKILL.md` does not hide a valid one with the same name.
    """
    libraries, skipped = await _libraries(workspace, directories, missing_directories)
    discovered = await _discover_skills(workspace, libraries)
    available_names = frozenset(name for name, _ in discovered)
    normalized_include = None if include is None else frozenset(_normalize_name(name) for name in include)
    normalized_exclude = frozenset(_normalize_name(name) for name in exclude)
    if unknown_names == 'error':
        _validate_selection('include', normalized_include, available_names)
        _validate_selection('exclude', normalized_exclude, available_names)
    selected_names = (
        normalized_include if normalized_include is not None else available_names.difference(normalized_exclude)
    )

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
            skipped.append(SkippedSkill(path=skill_file, reason='invalid', message=f'Skipping {skill_file}: {error}'))
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
