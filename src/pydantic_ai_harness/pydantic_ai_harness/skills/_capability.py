"""Load Agent Skill instructions from the run's workspace as deferred capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, get_args, overload
from weakref import WeakValueDictionary

from pydantic import Field

from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Toolset
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset, CombinedToolset, FunctionToolset
from pydantic_ai.workspaces import (
    AuthorityWorkspace,
    SupportsCommands,
    SupportsRealpath,
    Workspace,
    WorkspaceBackend,
)
from pydantic_ai_harness._workspace import innermost_backend, require_workspace, secondary_workspace
from pydantic_ai_harness.artifact_scanning import ArtifactScanner
from pydantic_ai_harness.artifact_scanning._scanner import record_flagged
from pydantic_ai_harness.skills._loader import (
    DuplicateNames,
    MissingDirectories,
    SkillCatalog,
    SkillDefinition,
    SkippedSkill,
    UnknownNames,
    catalog_description,
    load_skill_libraries,
    same_skill,
    skip_duplicate_name,
)
from pydantic_ai_harness.skills._toolset import UntrustedSkills, UntrustedSkillsToolset

_MAX_DESCRIPTION_LENGTH = 1024
_MAX_READ_CHARS = 50_000
"""The most of a skill file `read_skill_file` returns, and scans, at once, as `FileSystem` caps a read."""
_SCAN_OVERLAP = 1_000
"""How far past each end of a window the scan reaches, so a phrase split between two windows is still seen."""
_MAX_FILE_BYTES = 10 * 1024 * 1024
"""The largest skill file `read_skill_file` reads, since a workspace read loads the whole file."""
_UNTRUSTED_SKILLS_NOTE = (
    'Skills, instructions for particular tasks, are listed by `list_skills`. At the start of a task, '
    'call `list_skills` to check whether one applies, and `load_skill` to read the one that does.'
)


@dataclass(frozen=True)
class _SkillSource:
    """One `Skills(...)` configuration: libraries, the selection applied to them, and where they live."""

    directories: tuple[str | Path, ...]
    include: frozenset[str] | None
    exclude: frozenset[str]
    missing_directories: MissingDirectories
    duplicate_names: DuplicateNames
    unknown_names: UnknownNames
    workspace: Workspace | None
    """The `workspace=` the libraries are read from, or `None` for the run's workspace."""
    scanner: ArtifactScanner | None
    read_files: bool


@dataclass(frozen=True)
class _Entry:
    """A skill in the catalog, with the workspace it was read from and the configuration that selected it."""

    skill: SkillDefinition
    workspace: Workspace
    source: _SkillSource


class _Skill(AbstractCapability[AgentDepsT]):
    """One skill: its instructions, loaded by the model with `load_capability`.

    Instructions only, with no toolset, so a durable engine accepts it although it is built per run.
    """

    def __init__(self, skill: SkillDefinition) -> None:
        self.id = skill.name
        # Continuation lines are indented so a multiline description doesn't read as separate catalog entries.
        self.description = catalog_description(skill).replace('\n', '\n  ')
        self.defer_loading = True
        self.instructions = skill.render()

    def get_instructions(self) -> str:
        return self.instructions


@dataclass(init=False, repr=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Offer Agent Skill instructions from the run's workspace as deferred capabilities.

    Skill libraries are directories in the run's workspace (`ctx.workspace`), read at the
    start of every run; relative paths resolve against its working directory. Attach
    `LocalWorkspace` to read directories on this machine, or pass `workspace=` to read them
    from a workspace of their own. A run with neither fails at its start.

    Each selected immediate child containing `SKILL.md` becomes a deferred capability named after
    the skill: the model sees its name, description and `when_to_use`, and loads its Markdown body, with the
    skill's directory in the run's workspace, with `load_capability`. Bundled files are not executed,
    and are read only with `read_files=True`. Descriptions longer than the Agent Skills limit are
    preserved and emit a warning. A skill found twice, through a symlink or as a byte-identical
    `SKILL.md`, counts once. A skill whose frontmatter has `disable-model-invocation: true` is left
    out of the model's catalog.

    A skill whose `SKILL.md` is below `'operator'` [authority][pydantic_ai.workspaces.Workspace.authority_of] in the
    workspace its library is read from, such as one in a repository the operator doesn't control, is not in the
    catalog, since the model reads the catalog as the operator's instructions: the model calls `list_skills` for
    such skills' names and descriptions, and `load_skill` for one's instructions, which it receives as tool output.

    `load` reads the same skills outside a run, for a host that lets a person invoke a skill
    itself; render one with `SkillDefinition.render`.

    With `read_files=True`, the model also gets a `read_skill_file` tool that reads the text files in
    a loaded skill's directory, such as `references/guide.md`, and nothing outside it. A skill below `'operator'`
    can be read as soon as `list_skills` lists it, and its files reach the model as tool output, as its instructions
    from `load_skill` do.

    Two `Skills` on one agent combine, so every library either names stays reachable.
    """

    directories: tuple[str | Path, ...]
    """Skill-library paths in the workspace, read at the start of each run."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the catalog."""

    missing_directories: MissingDirectories
    """`'error'` fails the run on a library path that is not a library; `'skip'` leaves it out, with a warning
    unless it does not exist."""

    duplicate_names: DuplicateNames
    """`'error'` fails the run when two different `SKILL.md` files share a name; `'keep_first'` keeps the first."""

    unknown_names: UnknownNames
    """`'error'` fails the run on an `include` or `exclude` name no library has; `'skip'` ignores that name."""

    workspace: WorkspaceBackend | None
    """Where the libraries live, when not in the run's workspace; see `__init__`."""

    scanner: ArtifactScanner | None
    """Scans each skill's description, `when_to_use` and body before it is offered, skipping flagged skills; see `__init__`."""

    read_files: bool
    """Whether the model gets a `read_skill_file` tool for a loaded skill's files; see `__init__`."""

    id: str | None = 'skills'
    """One per agent: two `Skills` combine into one catalog."""

    _sources: tuple[_SkillSource, ...] = field(default=(), init=False, repr=False, compare=False)
    """Every configuration this instance serves: its own, plus those of any `Skills` combined into it."""

    _toolset: FunctionToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The `read_skill_file` tool, when any configuration this instance serves has `read_files=True`."""

    _untrusted_toolset: UntrustedSkillsToolset[AgentDepsT] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """`list_skills` and `load_skill`, built once and shared with each run's copy, so durable execution sees the
    toolset it registered."""

    _offers_untrusted: bool | None = field(default=None, init=False, repr=False, compare=False)
    """Whether this run's copy offers skills below `'operator'` authority; `None` outside a run."""

    _lowered_names: frozenset[str] | None = field(default=None, init=False, repr=False, compare=False)
    """The skills below `'operator'` authority this run's copy offers, as `for_run` found them; `None` outside a run."""

    _runs: WeakValueDictionary[str, Skills[AgentDepsT]] = field(init=False, repr=False, compare=False)
    """Each live run's copy by run ID, shared with the copies, so the shared toolset finds the run's lowered names."""

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str],
        exclude: None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        unknown_names: UnknownNames = 'error',
        workspace: WorkspaceBackend | None = None,
        scanner: ArtifactScanner | None = None,
        read_files: bool = False,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        unknown_names: UnknownNames = 'error',
        workspace: WorkspaceBackend | None = None,
        scanner: ArtifactScanner | None = None,
        read_files: bool = False,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        unknown_names: UnknownNames = 'error',
        workspace: WorkspaceBackend | None = None,
        scanner: ArtifactScanner | None = None,
        read_files: bool = False,
    ) -> None:
        """Configure the skill libraries to read at the start of each run.

        Args:
            directories: One skill-library path or a sequence of paths in the workspace, in precedence
                order. `~` is not expanded: the workspace may be a sandbox with a home of its own.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
            missing_directories: `'error'` (the default) fails the run when a library path does not
                exist, is not a directory, is itself a skill package, or the workspace raises
                `PermissionError` for it. `'skip'` leaves such a path out, for conventional locations
                such as `.agents/skills` that a project may not have: one that does not exist quietly,
                any other with a warning.
            duplicate_names: `'error'` (the default) fails the run when two different `SKILL.md` files
                share a name. `'keep_first'` keeps the one from the earlier directory and skips the
                other with a warning, the way coding agents layer project skills over personal ones.
            unknown_names: `'error'` (the default) fails the run when an `include` or `exclude` name
                matches no skill. `'skip'` ignores that name, for a selection shared by projects that
                do not all have every skill it names.
            workspace: A workspace backend to read the libraries from instead of the run's, such as
                `LocalWorkspaceBackend('/app')` for skills shipped with the code while the agent
                works in a sandbox. It is read in-process: a durable engine does not route it
                through its workflow machinery.
            scanner: An `ArtifactScanner` that checks each skill's description, `when_to_use` and body for prompt injection,
                credential exfiltration, and hidden content. A flagged skill is left out of the catalog, with
                a warning, as a `SkippedSkill` whose `reason` is `'flagged'`. Omit to load skills unscanned.
                With `read_files`, it also scans the text each file read returns, and withholds it if flagged.
            read_files: Give the model a `read_skill_file` tool that reads a UTF-8 text file in the
                directory of a skill it has loaded, from the workspace the skill was read from. It
                works without file tools and for a `workspace=` library, and does not reach files
                outside the skill's directory, through `..` or a symlink. A workspace whose backend
                implements neither `SupportsRealpath` nor `SupportsCommands` cannot resolve symlinks,
                so the tool reads nothing from it.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')
        # Typed as literals, but agent specs and untyped callers reach here with any string.
        if missing_directories not in get_args(MissingDirectories):
            raise ValueError(f"missing_directories must be 'error' or 'skip', not {missing_directories!r}.")
        if duplicate_names not in get_args(DuplicateNames):
            raise ValueError(f"duplicate_names must be 'error' or 'keep_first', not {duplicate_names!r}.")
        if unknown_names not in get_args(UnknownNames):
            raise ValueError(f"unknown_names must be 'error' or 'skip', not {unknown_names!r}.")

        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()
        self.missing_directories = missing_directories
        self.duplicate_names = duplicate_names
        self.unknown_names = unknown_names
        self.workspace = workspace
        self.scanner = scanner
        self.read_files = read_files
        own = secondary_workspace(workspace, 'Skills')
        self._sources = (
            _SkillSource(
                self.directories,
                self.include,
                self.exclude,
                missing_directories,
                duplicate_names,
                unknown_names,
                own,
                scanner,
                read_files,
            ),
        )
        self._toolset = self._build_toolset()
        self._runs = WeakValueDictionary()

    @classmethod
    def from_spec(
        cls,
        directories: str | Sequence[str],
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        missing_directories: MissingDirectories = 'error',
        duplicate_names: DuplicateNames = 'error',
        unknown_names: UnknownNames = 'error',
        read_files: bool = False,
        **kwargs: Any,
    ) -> Skills[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. Inherited, it would read
        `__init__`, whose `Collection[str]` selectors and `workspace` backend have no JSON schema
        and would drop the long form, leaving only `Skills: <directories>`. `**kwargs` reaches
        `__init__` unchanged, so a spec that set anything else still loads.
        """
        options: dict[str, Any] = {
            'include': include,
            'exclude': exclude,
            'missing_directories': missing_directories,
            'duplicate_names': duplicate_names,
            'unknown_names': unknown_names,
            'read_files': read_files,
            **kwargs,
        }
        # One dict, because `__init__`'s overloads reject an `include` and `exclude` that are both optional.
        return cls(directories, **options)

    def __repr__(self) -> str:
        """Show only the `Skills` configuration that callers control."""
        return (
            f'{type(self).__name__}('
            f'directories={self.directories!r}, include={self.include!r}, exclude={self.exclude!r}, '
            f'missing_directories={self.missing_directories!r}, duplicate_names={self.duplicate_names!r}, '
            f'unknown_names={self.unknown_names!r}'
            + ('' if self.scanner is None else f', scanner={self.scanner!r}')
            + (', read_files=True' if self.read_files else '')
            + ')'
        )

    @staticmethod
    def _normalize_directories(
        directories: str | Path | Sequence[str | Path],
    ) -> tuple[str | Path, ...]:
        if isinstance(directories, (str, Path)):
            return (directories,)
        normalized = tuple(directories)
        if not normalized:
            raise ValueError('Skills requires at least one skill-library directory.')
        return normalized

    @staticmethod
    def _normalize_selection(name: str, values: Collection[object]) -> frozenset[str]:
        if isinstance(values, str):
            raise TypeError(f'{name} must be a collection of skill names, not a string.')
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f'{name} must contain only skill names as strings.')
            normalized.add(value)
        return frozenset(normalized)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Serve every combined configuration's libraries through one catalog.

        The field-by-field default would keep only the last configuration's directories, dropping the
        other libraries. A skill name selected by two configurations must name the same `SKILL.md`, or
        a byte-identical copy, unless the later configuration has `duplicate_names='keep_first'`.
        """
        first = capabilities[0]
        assert isinstance(first, cls)
        sources: list[_SkillSource] = []
        for capability in capabilities:
            assert isinstance(capability, cls)
            sources.extend(source for source in capability._sources if source not in sources)
        merged = replace_no_init(first)
        merged._sources = tuple(sources)
        merged._toolset = merged._build_toolset()
        merged._untrusted_toolset = None
        return merged

    def _build_toolset(self) -> FunctionToolset[AgentDepsT] | None:
        if not any(source.read_files for source in self._sources):
            return None
        toolset = FunctionToolset[AgentDepsT](id='skills')
        toolset.add_function(self._read_skill_file, name='read_skill_file')
        return toolset

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """`list_skills` and `load_skill`, and the `read_skill_file` tool when `read_files` is set.

        Outside a run, both, so a durable engine registers them with the agent. A run's copy, which offers skills
        below `'operator'` authority, gives `list_skills` and `load_skill` only: `for_run` adds `read_skill_file`.
        """
        if self._untrusted_toolset is None:
            # Its own id, so a durable engine names its activities apart from `read_skill_file`'s.
            self._untrusted_toolset = UntrustedSkillsToolset[AgentDepsT](
                self._untrusted_skills, id=None if self.id is None else f'{self.id}_lowered'
            )
        if self._offers_untrusted or self._toolset is None:
            return self._untrusted_toolset
        return CombinedToolset([self._untrusted_toolset, self._toolset])

    async def _read_skill_file(
        self, ctx: RunContext[AgentDepsT], skill: str, path: str, offset: Annotated[int, Field(ge=0)] = 0
    ) -> str:
        """Read a text file bundled with a skill you loaded or `list_skills` lists, such as `references/guide.md`.

        Args:
            ctx: The run context.
            skill: The skill's name.
            path: The file's path relative to the skill's directory, such as `references/guide.md`, without
                the directory itself in front.
            offset: The character to start at, to continue a file longer than one read returns.
        """
        try:
            entries, _ = await self._read(ctx.workspace)
        except ValueError as error:
            raise ToolFailed(f'The skill libraries cannot be read: {error}') from None
        entry = entries.get(skill)
        if entry is None or not entry.source.read_files or not entry.skill.model_invocable:
            raise ModelRetry(f'The files of skill {skill!r} cannot be read with this tool.')
        if skill not in ctx.loaded_capability_ids:
            # A skill `list_skills` lists is never loaded with `load_capability`: its files are tool output, like its
            # instructions from `load_skill`.
            listed = self._run_lowered_names(ctx)
            if (listed is not None and skill not in listed) or skill not in await _lowered({skill: entry}):
                raise ModelRetry(f'Load the skill {skill!r} with `load_capability` before reading its files.')
        text = await _read_text(entry, path)
        if offset and offset >= len(text):
            raise ModelRetry(f'offset={offset} is past the end of {path!r}, which has {len(text):,} characters.')
        end = offset + _MAX_READ_CHARS
        window = text[offset:end]
        scanner = entry.source.scanner
        scanned = text[max(0, offset - _SCAN_OVERLAP) : end + _SCAN_OVERLAP]
        if scanner is not None and (scan := scanner.scan(scanned, source=f'skill:{skill}/{path}')).flagged:
            record_flagged(
                ctx.tracer, scan, artifact='skill', action='withheld', include_content=ctx.trace_include_content
            )
            raise ModelRetry(f'{path!r} in skill {skill!r} was withheld: an artifact scan flagged it.')
        if end >= len(text):
            return window
        return f'{window}\n\n[Characters {offset:,} to {end:,} of {len(text):,}; read on from offset={end}.]'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Read the selected skills, from each configuration's `workspace=` or else the run's workspace.

        Returns one deferred capability per `'operator'` skill the model may load, with the `read_skill_file` tool
        when one of the skills the model may load has `read_files` set, plus, when skills below `'operator'`
        authority are among them, a copy of this capability that offers those through `list_skills` and
        `load_skill`. Without any skills, it adds no instructions or tools. Emits a `UserWarning` for each skipped
        `SKILL.md` or library path, an overlong description, or unsupported frontmatter. Raises `UserError` when a
        configuration without `workspace=` meets a run without a workspace.
        """
        if any(source.workspace is None for source in self._sources):
            require_workspace(ctx.workspace, 'Skills', ctx.messages)
        entries, skipped = await self._read(ctx.workspace)
        self._report_skipped(ctx, entries, skipped)
        return self._offer(entries, await _lowered(entries), ctx.run_id)

    def _offer(
        self, entries: dict[str, _Entry], lowered: frozenset[str], run_id: str | None
    ) -> AbstractCapability[AgentDepsT]:
        """One deferred capability per `'operator'` skill, plus this capability's run copy for the `lowered` ones."""
        invocable = [entry for entry in entries.values() if entry.skill.model_invocable]
        trusted = [entry for entry in invocable if entry.skill.name not in lowered]
        offered: list[AbstractCapability[AgentDepsT]] = [_Skill[AgentDepsT](entry.skill) for entry in trusted]
        if self._toolset is not None and any(entry.source.read_files for entry in invocable):
            # The toolset built at construction, so a durable engine finds the one it registered.
            offered.append(Toolset(self._toolset))
        if len(trusted) == len(invocable):
            return CombinedCapability(offered)
        run = replace_no_init(self)
        run._offers_untrusted = True
        run._lowered_names = frozenset(entry.skill.name for entry in invocable if entry.skill.name in lowered)
        if run_id is not None:
            self._runs[run_id] = run
        # Its own id, the one its toolset has: a wrapper such as `PrefixTools` carries this capability's id, and
        # would claim it alongside the copy it holds, as an `'operator'` skill of that name would. No skill can
        # have this one, since skill names have no underscores. A run's copy still replaces the agent's.
        run.id = None if self.id is None else f'{self.id}_lowered'
        return CombinedCapability([*offered, run])

    async def load(self, workspace: WorkspaceBackend | None = None) -> SkillCatalog:
        """Read the selected skills now, as a run does at its start, without emitting warnings.

        For a host that offers skills to a person as well as to the model, such as a `/code-review`
        command that sends `skill.render(arguments)` as a prompt. The catalog includes skills the model
        may not load (`model_invocable`). Its `skipped` entries say which `SKILL.md` files and library
        paths were left out, and its `advice` holds what a run would also warn about, for the host to show.

        Args:
            workspace: Where libraries without their own `workspace=` live, as the run's workspace
                would be, such as `LocalWorkspaceBackend('.')`.

        Raises:
            UserError: A configuration without `workspace=` needs `workspace`, and none was passed.
            ValueError: A configuration problem that would also fail a run, such as, with
                `unknown_names='error'`, an unknown `include` name or, with `duplicate_names='error'`,
                two skills with one name.
        """
        run_workspace = secondary_workspace(workspace, 'Skills.load')
        if run_workspace is None and any(source.workspace is None for source in self._sources):
            raise UserError(
                '`Skills.load()` needs the workspace the libraries are in, such as `LocalWorkspaceBackend(".")`.'
            )
        return await self._load(run_workspace)

    def get_instructions(self) -> str | None:
        """Point at `list_skills` in a run that has skills below `'operator'` authority."""
        return _UNTRUSTED_SKILLS_NOTE if self._offers_untrusted else None

    async def _untrusted_skills(self, ctx: RunContext[AgentDepsT]) -> UntrustedSkills:
        """The lowered skills, read again for a `list_skills` or `load_skill` call, and what that read skipped.

        Only the skills `for_run` found below `'operator'` are read again. `for_run` already warned about skips, so
        this reports them to the model instead: a warning per tool call would repeat, and turn into an error where
        warnings are errors. Where the run's copy isn't reachable, such as in a durable engine's activity, every
        library is read, and only the skills still below `'operator'` are offered.
        """
        entries, skipped = await self._read(ctx.workspace, only=self._run_lowered_names(ctx))
        lowered = await _lowered(entries)
        return UntrustedSkills(
            skills=[entry.skill for name, entry in entries.items() if name in lowered and entry.skill.model_invocable],
            skipped=[entry.message for entry in skipped],
        )

    def _run_lowered_names(self, ctx: RunContext[AgentDepsT]) -> frozenset[str] | None:
        """The lowered skill names this run's copy of this capability found at the start of the run, if it's here."""
        if self._lowered_names is not None:
            return self._lowered_names
        # The shared toolset calls the instance that built it, which keeps each live run's copy.
        run = self._runs.get(ctx.run_id) if ctx.run_id is not None else None
        return None if run is None else run._lowered_names

    def _report_skipped(
        self, ctx: RunContext[AgentDepsT], entries: dict[str, _Entry], skipped: tuple[SkippedSkill, ...]
    ) -> None:
        """Record each flagged skill on the run's trace and warn about each skipped `SKILL.md` or library path."""
        for entry in skipped:
            if entry.scan is not None:
                record_flagged(
                    ctx.tracer,
                    entry.scan,
                    artifact='skill',
                    action='skipped',
                    include_content=ctx.trace_include_content,
                )
        advice = self._advice(tuple(entry.skill for entry in entries.values()))
        for message in (*(entry.message for entry in skipped), *advice):
            warnings.warn(message, UserWarning, stacklevel=3)

    async def _load(self, run_workspace: Workspace | None) -> SkillCatalog:
        entries, skipped = await self._read(run_workspace)
        skills = tuple(entry.skill for entry in entries.values())
        return SkillCatalog(skills=skills, skipped=skipped, advice=self._advice(skills))

    async def _read(
        self, run_workspace: Workspace | None, *, only: frozenset[str] | None = None
    ) -> tuple[dict[str, _Entry], tuple[SkippedSkill, ...]]:
        """The selected skills by name, and what was skipped; with `only`, just those of the selected skills."""
        # Where each name's first `SKILL.md` is, whether it was kept or flagged.
        claimed: dict[str, tuple[Workspace, str]] = {}
        by_name: dict[str, _Entry] = {}
        messages: list[SkippedSkill] = []
        for source in self._sources:
            workspace = source.workspace or run_workspace
            assert workspace is not None, 'a configuration without `workspace=` is only loaded with a run workspace'
            skills, skipped = await load_skill_libraries(
                workspace,
                source.directories,
                include=_selection(source.include, source.exclude, only),
                exclude=source.exclude,
                missing_directories=source.missing_directories,
                duplicate_names=source.duplicate_names,
                # A name read again need not be in every library.
                unknown_names=source.unknown_names if only is None else 'skip',
            )
            messages.extend(skipped)
            for skill in skills:
                previous = claimed.get(skill.name)
                if previous is not None:
                    if not await same_skill(previous, (workspace, skill.path)):
                        messages.append(
                            skip_duplicate_name(skill.name, previous[1], skill.path, source.duplicate_names)
                        )
                    continue
                # A flagged skill keeps its name, so a copy in a library that is not scanned cannot take its place.
                claimed[skill.name] = (workspace, skill.path)
                if (flagged := _flagged(source.scanner, skill)) is not None:
                    messages.append(flagged)
                    continue
                # A library read from `workspace=` is not where the model's file tools work.
                by_name[skill.name] = _Entry(
                    replace(skill, in_run_workspace=source.workspace is None), workspace, source
                )
        return by_name, tuple(messages)

    @staticmethod
    def _advice(definitions: tuple[SkillDefinition, ...]) -> tuple[str, ...]:
        advice: list[str] = []
        overlong_descriptions = [
            f'{skill.name} ({len(skill.description):,} characters)'
            for skill in definitions
            if len(skill.description) > _MAX_DESCRIPTION_LENGTH
        ]
        if overlong_descriptions:
            advice.append(
                f'Agent Skill descriptions exceed the {_MAX_DESCRIPTION_LENGTH:,}-character limit: '
                + '; '.join(overlong_descriptions)
            )
        ignored = [
            f'{skill.name}: {", ".join(skill.ignored_behavioral_fields)}'
            for skill in definitions
            if skill.ignored_behavioral_fields
        ]
        if ignored:
            advice.append('Ignoring unsupported Agent Skill behavioral frontmatter fields: ' + '; '.join(ignored))
        return tuple(advice)


async def _read_text(entry: _Entry, path: str) -> str:
    """The text of `path` in `entry`'s skill directory, refusing a path that leads outside it."""
    skill, workspace, directory = entry.skill.name, entry.workspace, entry.skill.directory
    paths = workspace.paths
    # Rejected before anything resolves it: an absolute or drive path would replace the directory when joined.
    if paths.isabs(path) or paths.splitdrive(path)[0]:
        raise ModelRetry(
            f"Give a path relative to the directory of skill {skill!r}, such as 'references/guide.md', not {path!r}."
        )
    if not isinstance(innermost_backend(workspace), (SupportsRealpath, SupportsCommands)):
        raise ToolFailed(f'The workspace of skill {skill!r} cannot resolve symlinks, so its files are not read.')
    try:
        target = await _contained(workspace, directory, path)
        if target is None:
            raise ModelRetry(f'{path!r} is outside the directory of skill {skill!r}.')
        if (size := (await workspace.stat(target)).size) is not None and size > _MAX_FILE_BYTES:
            raise ToolFailed(f'{path!r} in skill {skill!r} is {size:,} bytes, over the {_MAX_FILE_BYTES:,}-byte limit.')
        return await workspace.read_text(target)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise ModelRetry(f'Skill {skill!r} has no file {path!r}.') from None
    except PermissionError:
        raise ToolFailed(f'{path!r} in skill {skill!r} cannot be read: permission denied.') from None
    except UnicodeDecodeError:
        raise ModelRetry(f'{path!r} in skill {skill!r} is not a UTF-8 text file.') from None


async def _contained(workspace: Workspace, directory: str, path: str) -> str | None:
    """Where `path`, relative to `directory`, leads with every symlink followed, or `None` when that is outside it."""
    root = await workspace.realpath(directory)
    # Joined, not normalized: `link/..` climbs from the link's target, as it would in a shell.
    target = await workspace.realpath(workspace.paths.join(directory, path))
    # A resolution that stopped at its symlink limit still holds a link, so resolving it again moves it.
    if not target.startswith(root + workspace.paths.sep) or await workspace.realpath(target) != target:
        return None
    return target


def _flagged(scanner: ArtifactScanner | None, skill: SkillDefinition) -> SkippedSkill | None:
    """The entry for `skill` when `scanner` flags its catalog text or body, otherwise `None`."""
    if scanner is None:
        return None
    # Scan exactly what reaches the model: the catalog entry, including `when_to_use`, and the body.
    scan = scanner.scan(f'{catalog_description(skill)}\n\n{skill.body}', source=f'skill:{skill.name}')
    if not scan.flagged:
        return None
    return SkippedSkill(
        path=skill.path,
        reason='flagged',
        message=f'Skipping {skill.path}: an artifact scan flagged it ({scan.summary()}).',
        scan=scan,
    )


def _selection(
    include: frozenset[str] | None, exclude: frozenset[str], only: frozenset[str] | None
) -> frozenset[str] | None:
    """The names a configuration includes, narrowed to `only` when that's given.

    The loader ignores `exclude` once there's an `include` set, so narrowing to `only` takes the exclusions out here:
    a name read again in one library mustn't bring back a skill that library excluded.
    """
    if only is None:
        return include
    return only - exclude if include is None else include & only


async def _lowered(entries: dict[str, _Entry]) -> frozenset[str]:
    """The names of the skills whose `SKILL.md` is below `'operator'` in the workspace each was read from."""
    return frozenset(
        [
            name
            for name, entry in entries.items()
            if await _skill_authority(entry.workspace, entry.skill.path) != 'operator'
        ]
    )


async def _skill_authority(workspace: Workspace, skill_file: str) -> Authority:
    """The level of the skill whose `SKILL.md` is `skill_file`, with its directory counted as part of `workspace`.

    `authority_of` puts a path outside the working directory at `'none'` unless a `path_authority` entry covers it,
    but a library is configured by the operator, and a skill directory in it may link elsewhere, such as to skills
    shared between projects. Naming the skill's directory at `'operator'` in a wrapper gives it the level of the entry
    covering it, if any, and otherwise the workspace's own; its link target gets no more than the link's location.
    The wrapper never raises a level, so a plain workspace offers its skills as before and a lowered one lowers them.
    """
    directory = workspace.paths.dirname(await workspace.resolve(skill_file))
    view = AuthorityWorkspace(workspace, authority='operator', path_authority={directory: 'operator'})
    return await view.authority_of(skill_file)
