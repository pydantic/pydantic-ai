"""Validated settings, independent of persistence and terminal code."""

import re
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from pydantic_ai_pax.config.input_bindings import (
    BusyDelivery,
    InputBindings,
    KeyboardLabels,
    QuestionPlacement,
    QuestionSubmit,
)
from pydantic_ai_pax.config.theme_names import names

_WORKTREE_COMMAND_PLACEHOLDER = re.compile(r'{[A-Za-z_]\w*}')
_STORED_FOOTER_FIELDS: TypeAdapter[list[str]] = TypeAdapter(list[str])

UpdateChannel = Literal['stable', 'main']
UPDATE_CHANNELS: tuple[UpdateChannel, ...] = get_args(UpdateChannel)
STORED_MAIN_CHANNEL = 'bleeding'
"""`main`'s former name, still what is saved for it: older builds accept only this name."""
ToolCallDisplay = Literal['detailed', 'grouped']
TOOL_CALL_DISPLAYS: tuple[ToolCallDisplay, ...] = get_args(ToolCallDisplay)
TrustLevel = Literal['trusted', 'untrusted']
FooterField = Literal[
    'model',
    'workspace',
    'permissions',
    'context',
    'output',
    'cost',
    'activity',
    'plugins',
    'updates',
    'live',
    'queued',
    'assistant_age',
]
FOOTER_FIELDS: tuple[FooterField, ...] = get_args(FooterField)
DEFAULT_FOOTER_FIELDS: tuple[FooterField, ...] = tuple(name for name in FOOTER_FIELDS if name != 'assistant_age')
PermissionMode = Literal['full', 'auto', 'read-only']
"""harness `PermissionMode`, spelled out so reading settings imports no harness code."""
PERMISSION_MODES: tuple[PermissionMode, ...] = get_args(PermissionMode)
PERMISSIONS_SETTING = 'run.permissions'
"""The setting `/permissions` changes, defined once for every reader."""


def _validate_user_path(value: str) -> str:
    path = Path(value)
    if '\0' in value or not value.strip() or (not path.is_absolute() and path.parts[:1] != ('~',)):
        raise ValueError('Use an absolute path or a path starting with `~/`.')
    return value


_UserPath = Annotated[str, AfterValidator(_validate_user_path)]


def _validate_worktree_remove_command(argv: list[str]) -> None:
    placeholders = {'{path}', '{branch}', '{repo}'}
    if not argv or not argv[0]:
        raise ValueError('Worktree removal commands must have a non-empty executable and argv')
    if any('\0' in token for token in argv):
        raise ValueError('Worktree removal command arguments cannot contain NUL')
    if any(token not in placeholders and any(name in token for name in placeholders) for token in argv):
        raise ValueError('Worktree command placeholders must fill a whole argv token')
    unknown = {token for token in argv if _WORKTREE_COMMAND_PLACEHOLDER.fullmatch(token) and token not in placeholders}
    if unknown:
        raise ValueError(f'Unknown worktree command placeholders: {", ".join(sorted(unknown))}')


class Settings(BaseModel):
    """An immutable snapshot; storage contains only explicit overrides."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    model: str | None = Field(
        default='openai-codex:gpt-6-astra',
        min_length=1,
        description='Provider-qualified model for every turn. Overrides the model baked into the agent.',
    )
    input_bindings: InputBindings = Field(
        default=InputBindings(),
        description='Decoded send, steer and newline keys. /input test checks your terminal.',
    )
    keyboard_labels: KeyboardLabels = Field(
        default='neutral',
        description='Keyboard labels only: neutral, mac or pc. Does not detect SSH keys.',
    )
    busy_delivery: BusyDelivery = Field(
        default='after-turn',
        description=(
            'Busy composer prompts: after-turn queues a separate turn; next-checkpoint uses native steering '
            'after current tool work, before the next feasible model request. Mixed queues and saved edits ask '
            'whether to steer or keep queue position. Commands retain their own routing.'
        ),
    )
    question_submit: QuestionSubmit = Field(
        default='ask', description='When a question waits: ask, answer it, or queue a follow-up. Never approves tools.'
    )
    question_placement: QuestionPlacement = Field(
        default='picker',
        description=(
            'Where questions and tool approvals wait: picker takes the keys as each one arrives; composer keeps '
            'your draft and keys, and F2 reviews the choices.'
        ),
    )
    request_limit: int = Field(
        default=10000, gt=0, description='Most model requests one prompt may make before the turn stops.'
    )
    tool_retries: int = Field(
        default=3, ge=0, description='Default retries per tool call. Explicit tool retry limits take precedence.'
    )
    pool_accounts: bool = Field(
        default=True,
        description=(
            'A model without @PROFILE runs on every signed-in account of its provider, in /accounts order, '
            'as @* does. PROVIDER@default:MODEL runs the default account alone.'
        ),
    )
    code_mode: bool = Field(
        default=True,
        description=(
            'Fold tools into a sandboxed run_code that runs each statement while the model writes. '
            'Off, every tool is called natively.'
        ),
    )
    speculative_code_mode: bool = Field(
        default=False,
        description=(
            'Also start read-only calls in run_code before the model reaches them. Turns on code mode '
            'for its runs. Ctrl+X Ctrl+S toggles it.'
        ),
    )
    permissions: PermissionMode = Field(
        default='full',
        description=(
            'What runs without asking: full runs everything; auto asks before commands that delete, push, '
            'install, or reach the network; read-only asks before every file write and command. /permissions sets it.'
        ),
    )
    instructions: str = Field(
        default='',
        description=(
            "Your own instructions, sent after Pax's built-in, AGENTS.md, and plugin instructions. "
            '/system_prompt edits them.'
        ),
    )
    repo_context_walk_up_roots: list[_UserPath] = Field(
        default=['~'],
        description=(
            'User-only JSON list of inclusive stops for repo_context walk_up; the nearest one containing the '
            'workspace wins, and none scans only the workspace. Requires walk_up enabled.'
        ),
    )
    repo_context_autoload_symlink_roots: list[_UserPath] = Field(
        default=[],
        description='User-only JSON list of trusted target directories for external instruction symlinks. Autoload only.',
    )
    session_namer: bool = Field(
        default=True,
        description=(
            'Name saved sessions in the background using a model, at extra cost. By default only sessions held with '
            'the current model are named; once saved, even as true, every session from any workspace is.'
        ),
    )
    session_namer_model: str | None = Field(
        default=None, description='Naming model override; null uses the current model.'
    )
    theme: str = Field(
        default='default',
        description='Keep Pax colours, readable on light and dark terminals, or select a bundled Termflow palette with /theme.',
    )
    spinner: str = Field(
        default='working',
        min_length=1,
        description='Working animation; /spinner previews the choices. An unknown name shows working.',
    )
    footer_fields: list[FooterField] = Field(
        default=list(DEFAULT_FOOTER_FIELDS),
        description=(
            'Visible main-footer fields in display order, as a JSON list. An empty list hides the footer figures. '
            'Use /set to configure the layout or reset it. assistant_age is off by default.'
        ),
    )
    thinking: bool = Field(default=True, description="Show the model's thinking as it streams.")
    task_models: bool = Field(
        default=True,
        description=(
            'Name the model each subagent runs on in its start row, which waits for the child to report it. '
            'Off, the row prints at once without it.'
        ),
    )
    splash: bool = Field(default=True, description='Animate the startup splash. Takes effect next start.')
    full_screen: bool = Field(
        default=False,
        description=(
            'Take the whole terminal height from the start, instead of starting compact below the shell '
            'and growing as the session needs room. Takes effect next start.'
        ),
    )
    tool_calls: ToolCallDisplay = Field(
        default='detailed',
        description=(
            'How tool calls print: detailed gives each call its own line with arguments; '
            'grouped counts consecutive calls by tool on one live line. Grouped ignores display.tool_output '
            'and display.tool_arg_chars; file diffs still print.'
        ),
    )
    tool_output: bool = Field(
        default=False, description='Show shell and grep output below tool summaries. File diffs are always shown.'
    )
    shell_lines: int = Field(
        default=20, ge=0, le=1000, description='Shell preview lines when display.tool_output is enabled.'
    )
    grep_lines: int = Field(
        default=20, ge=0, le=1000, description='Grep preview lines when display.tool_output is enabled.'
    )
    tool_arg_chars: int = Field(
        default=40, ge=0, le=1000, description='Characters shown per tool argument value; 0 hides arguments.'
    )
    smooth_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5,
        allow_inf_nan=False,
        description='Catch-up window for smoothed response streaming, 0.1 to 5 seconds.',
    )
    update_channel: UpdateChannel = Field(
        default='stable',
        description='Where /update looks: stable PyPI releases, or main for the newest Pax commit on main.',
    )
    worktree_remove_commands: dict[str, list[str]] = Field(
        default={},
        description='Confirmed removal on exit: absolute Git common directory -> argv.',
    )

    @field_validator('footer_fields')
    @classmethod
    def validate_footer_fields(cls, value: list[FooterField]) -> list[FooterField]:
        """One occurrence per visible field; omitted fields are hidden."""
        if len(value) != len(set(value)):
            raise ValueError('Footer fields must be unique')
        return value

    @field_validator('worktree_remove_commands')
    @classmethod
    def validate_worktree_remove_commands(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        """Validate executable argv entries and canonical absolute repository keys."""
        normalized: dict[str, list[str]] = {}
        original_keys: dict[str, str] = {}
        for directory, argv in value.items():
            if '\0' in directory:
                raise ValueError('Worktree command keys cannot contain NUL')
            if not Path(directory).is_absolute():
                raise ValueError(f'Worktree command key must be an absolute path: {directory!r}')
            _validate_worktree_remove_command(argv)
            canonical = str(Path(directory).resolve())
            if canonical in original_keys:
                raise ValueError(
                    f'Worktree command keys {original_keys[canonical]!r} and {directory!r} resolve to the same directory'
                )
            original_keys[canonical] = directory
            normalized[canonical] = argv
        return normalized

    @field_validator('update_channel', mode='before')
    @classmethod
    def read_former_channel_name(cls, value: object) -> object:
        """Read `bleeding`, saved by older builds and by this one, as `main`."""
        return 'main' if value == STORED_MAIN_CHANNEL else value

    @field_validator('theme')
    @classmethod
    def validate_theme(cls, value: str) -> str:
        """Keep picker, project files, and saved preferences on the same registry."""
        if value not in names():
            raise ValueError(f'Unknown theme: {value}. Choose from: {", ".join(names())}')
        return value


SETTING_FIELDS = {
    'model': 'model',
    'run.request_limit': 'request_limit',
    'display.thinking': 'thinking',
    'display.task_models': 'task_models',
    'display.theme': 'theme',
    'display.spinner': 'spinner',
    'display.footer_fields': 'footer_fields',
    'display.splash': 'splash',
    'display.full_screen': 'full_screen',
    'display.tool_calls': 'tool_calls',
    'display.tool_output': 'tool_output',
    'display.shell_lines': 'shell_lines',
    'display.grep_lines': 'grep_lines',
    'display.tool_arg_chars': 'tool_arg_chars',
    'display.smooth_seconds': 'smooth_seconds',
    'run.tool_retries': 'tool_retries',
    'run.code_mode': 'code_mode',
    'run.speculative_code_mode': 'speculative_code_mode',
    'run.permissions': 'permissions',
    'run.instructions': 'instructions',
    'accounts.pool': 'pool_accounts',
    'sessions.naming': 'session_namer',
    'sessions.naming_model': 'session_namer_model',
    'updates.channel': 'update_channel',
    'repo_context.walk_up_roots': 'repo_context_walk_up_roots',
    'repo_context.autoload_symlink_roots': 'repo_context_autoload_symlink_roots',
    'worktrees.remove_commands': 'worktree_remove_commands',
    'input.bindings': 'input_bindings',
    'input.keyboard_labels': 'keyboard_labels',
    'input.question_submit': 'question_submit',
    'input.questions': 'question_placement',
    'input.busy_delivery': 'busy_delivery',
}

SETTING_KEYS = {name: key for key, name in SETTING_FIELDS.items()}
"""`Settings` field name to `/set` key, so a project file can use the model's own names."""

USER_ONLY_SETTINGS = frozenset(
    {
        'input_bindings',
        'keyboard_labels',
        'question_submit',
        'question_placement',
        'busy_delivery',
        'repo_context_walk_up_roots',
        'repo_context_autoload_symlink_roots',
        'worktree_remove_commands',
        'update_channel',
    }
)
"""`Settings` fields a project file cannot set, even with approval; a file that sets one stops startup."""

STRING_SETTINGS = frozenset(
    {
        'model',
        'run.instructions',
        'input.keyboard_labels',
        'input.question_submit',
        'input.questions',
        'input.busy_delivery',
        'run.permissions',
        'display.theme',
        'display.spinner',
        'display.tool_calls',
        'updates.channel',
    }
)
"""Keys whose typed value is taken as text rather than parsed as JSON."""


def resolve_settings(overrides: dict[str, JsonValue], *, stored: bool = False) -> Settings:
    """Validate values; stored layouts may name fields supported only by another build."""
    unknown = overrides.keys() - SETTING_FIELDS.keys()
    if unknown:
        raise ValueError(f'Unknown settings: {", ".join(sorted(unknown))}')
    values: dict[str, JsonValue] = {SETTING_FIELDS[key]: value for key, value in overrides.items()}
    if stored and 'footer_fields' in values:
        # Project and store overrides remain raw; only the effective snapshot skips future names.
        fields = _STORED_FOOTER_FIELDS.validate_python(values['footer_fields'], strict=True)
        values['footer_fields'] = [field for name in fields for field in FOOTER_FIELDS if field == name]
    return Settings.model_validate(values)


_CODER_FACTORY = 'pydantic_ai_harness.coder:Coder'
_FORMER_PACKAGE = re.compile(r'^pydantic_clai2(?=[.:]|$)')


class PluginSettings(BaseModel):
    """Declaration for a trusted plugin: a module defining one `Plugin` subclass, `module:PluginClass`, or `module:Capability`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    id: str = Field(min_length=1)
    factory: str = Field(pattern=r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*)?$')
    path: str | None = None
    enabled: bool = True
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator('factory')
    @classmethod
    def _renamed_package(cls, factory: str) -> str:
        """Name a module of `pydantic_clai2`, this package's name before `pydantic_ai_pax`, by its current path.

        `pydantic_clai2` still imports, so without this one module would pass checks that compare factories.
        """
        return _FORMER_PACKAGE.sub('pydantic_ai_pax', factory)

    @model_validator(mode='after')
    def _coder_without_delegation(self) -> 'PluginSettings':
        """Preserve the delegation opt-out of declarations saved before `sub_agents` existed.

        The stock declaration explicitly opts in. Supplied agents still receive plugins per run,
        where self-delegation requires capabilities bound to the agent instead.
        """
        if self.factory != _CODER_FACTORY or 'sub_agents' in self.settings:
            return self
        return self.model_copy(update={'settings': {**self.settings, 'sub_agents': False}})
