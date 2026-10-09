"""The declarative plugin API: subclass `Plugin` and override what it contributes, as with `AbstractCapability`."""

import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import ClassVar, Generic, Literal, Never, Protocol, Self, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from rich.console import Console, RenderableType
from typing_extensions import TypeVar as DefaultTypeVar, get_original_bases

from pydantic_ai import AgentRunResult, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.providers import infer_provider_class
from pydantic_ai_harness.step_persistence import StepStore
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.plugin_requirements import Requirements, declared_requirements
from pydantic_clai2.models import CLAI_PROVIDERS, LOGIN_ALIASES, LOGINS
from pydantic_clai2.ui.rendering.spinners import Spinner
from pydantic_clai2.ui.rendering.status import Status, StatusSegment

DepsT = DefaultTypeVar('DepsT', default=None)
ModelT = TypeVar('ModelT', bound=BaseModel)
_SAVED_SETTINGS = TypeAdapter(dict[str, JsonValue])

SessionEndReason = Literal['exit', 'eof', 'error']
TurnOutcome = Literal['completed', 'failed', 'cancelled']


class Conversation(Protocol):
    """The retained history as a plugin sees it. The shell's `Session` is one; `Transcript` is the plain one."""

    step_store: StepStore | None

    async def commit_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Persist and publish a between-turn history replacement."""
        ...

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages."""
        ...

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history, as `/compact` does after summarising it."""
        ...

    async def resolved_model(self) -> Model | str | None:
        """The model CLAI or the user selected for the next run; `None` when neither has chosen one.

        A capability that selects a model can replace CLAI's default per request, so a run may use another.
        """
        ...


class Transcript:
    """An in-memory `Conversation` for hosts built outside the shell, such as in a plugin's tests."""

    def __init__(self, *, messages: Sequence[ModelMessage] = (), model: Model | str | None = None) -> None:
        """Start with `messages` retained and `model` as what `resolved_model` reports."""
        self._messages = list(messages)
        self.step_store: StepStore | None = None
        self.model = model

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages, like `Session.messages`; edit through `replace_messages`."""
        return list(self._messages)

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history."""
        self._messages = list(messages)

    async def commit_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Publish an in-memory history replacement."""
        self.replace_messages(messages)

    async def resolved_model(self) -> Model | str | None:
        """The `model` given at construction."""
        return self.model


@dataclass(kw_only=True)
class SessionStart:
    """CLAI is ready for prompts. Each plugin receives this once, when it loads."""

    agent: AbstractAgent[Never, object]
    settings: Settings


@dataclass(kw_only=True)
class PluginLoadFailed:
    """A plugin failed to load; delivered to loaded plugins after startup loading finishes."""

    plugin: str
    error: BaseException


@dataclass(kw_only=True)
class SessionEnd:
    """CLAI is quitting, or this plugin is being unloaded."""

    reason: SessionEndReason


@dataclass(kw_only=True)
class TurnStart:
    """A prompt was submitted. Edit `text` or call `cancel()` before the agent sees it."""

    text: str
    cancelled: bool = False
    cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Stop this turn before it starts; the reason is shown to the user."""
        self.cancelled = True
        self.cancel_reason = reason


@dataclass(kw_only=True)
class TurnEnd:
    """The turn finished. `result` is set when it completed, `error` when it failed."""

    text: str
    outcome: TurnOutcome
    result: AgentRunResult[object] | None = None
    error: BaseException | None = None


SettingsProvider = Literal['anthropic', 'google', 'openai', 'openai-chat']
"""Providers whose `/model settings` controls a plugin's models can take."""


@dataclass(frozen=True, kw_only=True)
class ModelProvider:
    """Models a plugin runs under its own `PREFIX:`; return it from `Plugin.get_model_providers`.

    `resolve` gets NAME without the prefix. CLAI calls it in a worker thread before every run with
    one of these models, so it may read the keyring; raise `UserError` saying how to set up when it
    cannot build the model. A prefix Pydantic AI or CLAI already runs is rejected; when two plugins
    offer one prefix, the later one wins. Unloading the plugin removes the prefix, and a run with
    a model under it then fails as an unknown provider until the plugin is enabled again.

    When `resolve` returns that provider's model class, such as an `AnthropicModel` subclass, set
    `settings_from` so `/model settings` offers its controls (thinking, effort) for these models.
    """

    prefix: str
    resolve: Callable[[str], Model]
    """Build the model for a name given without its prefix."""
    models: tuple[str, ...] = ()
    """Names without the prefix, offered by `/model add` and `/set model`."""
    settings_from: SettingsProvider | None = None
    """The provider whose `/model settings` controls these models take; `None` offers the generic ones."""
    resolve_profile: Callable[[str, str], Model] | None = None
    """Build NAME for another account: called as `resolve_profile(NAME, PROFILE)` for `PREFIX@PROFILE:NAME`.

    Called in a worker thread like `resolve`. `None` means the prefix has one account, and a model
    naming a profile fails with a message saying so.
    """

    def __post_init__(self) -> None:
        """Reject a malformed prefix, one CLAI already runs, or an unknown `settings_from`."""
        _require_name('Model prefix', self.prefix)
        if _runs_already(self.prefix):
            raise ValueError(f'Model prefix {self.prefix!r} is a provider CLAI already runs; choose your own.')
        if self.settings_from is not None and self.settings_from not in get_args(SettingsProvider):
            raise ValueError(
                f'settings_from must be one of {", ".join(get_args(SettingsProvider))}; got {self.settings_from!r}.'
            )

    @property
    def names(self) -> tuple[str, ...]:
        """`models` as CLAI shows and saves them, with the prefix."""
        return tuple(f'{self.prefix}:{name}' for name in self.models)


@dataclass(frozen=True, kw_only=True)
class UsageWindow:
    """One limit on an account, such as a five-hour or weekly window, as `/accounts` shows it."""

    label: str
    """A short name for the window: `5h`, `7d`, or `premium`."""
    used_percent: float
    """How much of the limit is used, from 0 to 100."""
    resets_at: datetime | None = None
    """When the window resets, timezone-aware; `None` when the service does not say."""


@dataclass(frozen=True, kw_only=True)
class AccountUsage:
    """An account's current usage: its limits, most pressing first, and its plan when known."""

    windows: tuple[UsageWindow, ...]
    plan: str | None = None


@dataclass(frozen=True, kw_only=True)
class PluginLogin:
    """A sign-in a plugin adds as `/login NAME`; return it from `Plugin.get_logins`.

    For sign-ins that store credentials, such as the subscription behind a `ModelProvider`. Keep
    secrets in the keyring, never in plugin settings, and raise `UserError` when signing in fails.
    Name it after the model prefix it unlocks, as CLAI's own `openai-codex` and `github-copilot` are.
    NAME cannot be a sign-in CLAI ships (including the earlier `codex` and `copilot`); when two
    plugins add one name, the later one wins. Unloading the plugin removes it. Once the sign-in
    succeeds, `models` (as `PREFIX:NAME`, such as a `ModelProvider`'s `names`) are added to the saved
    model list, so `/model` and `/model settings` offer them without `/model add`.
    """

    name: str
    handler: Callable[[], Awaitable[str]]
    """Sign in and return the message to show."""
    models: tuple[str, ...] = ()
    """Models, as `PREFIX:NAME`, added to the saved model list once the sign-in succeeds."""
    profile_handler: Callable[[str], Awaitable[str]] | None = None
    """Sign in to another account: called as `profile_handler(PROFILE)` for `/login NAME@PROFILE`.

    On success `models` are saved with the profile, as `PREFIX@PROFILE:NAME`. `None` means the
    sign-in has one account, and `/login NAME@PROFILE` says so.
    """
    usage: Callable[[str | None], Awaitable[AccountUsage]] | None = None
    """The account's current usage, for `/accounts`: called as `usage(PROFILE)`, `None` for the default.

    `/accounts` calls it in the background for each listed account while the menu is open, with a
    short timeout. Raise `UserError` with a short reason when usage is unavailable.
    """

    def __post_init__(self) -> None:
        """Reject a malformed name, or one CLAI already signs in to, before any plugin can offer it."""
        _require_name('Login name', self.name)
        if self.name in LOGINS or self.name in LOGIN_ALIASES:
            raise ValueError(f'Login name {self.name!r} is a sign-in CLAI already has; choose your own.')


_PROVIDER_PREFIX = re.compile(r'[a-z][a-z0-9-]*')


def _require_name(kind: str, name: str) -> None:
    """Model prefixes and login names share one format: what users type after `/login` or before `:`."""
    if not _PROVIDER_PREFIX.fullmatch(name):
        raise ValueError(
            f'{kind} {name!r} must start with a lowercase letter, followed by lowercase letters, digits, and hyphens.'
        )


def _runs_already(prefix: str) -> bool:
    """Whether CLAI or Pydantic AI already runs `prefix:` models, aliases such as `openai-chat` included.

    `infer_model` accepts exactly the prefixes `infer_provider_class` knows, so that is the source of truth.
    An unknown prefix raises `ValueError` without importing anything; a known one whose SDK is missing
    raises `ImportError`, and is still taken.
    """
    if prefix in CLAI_PROVIDERS:
        return True
    try:
        infer_provider_class(prefix)
    except ValueError:
        return False
    except ImportError:
        return True
    return True


HostEvent = SessionStart | SessionEnd | TurnStart | TurnEnd | PluginLoadFailed
Renderer = Callable[[AgentStreamEvent], RenderableType | None]
"""Draws an event, or returns `None` to fall back to the default display; see `Plugin.render`."""
FullScreen = Callable[[], AbstractAsyncContextManager[None]]
"""Enter it to own the whole terminal for a widget while the agent runs; see `PluginHost.full_screen`."""


@asynccontextmanager
async def bare_screen() -> AsyncGenerator[None]:
    """The `FullScreen` of a host with no shell around it: nothing is streaming, so nothing to pause."""
    yield


class PluginHost(Generic[DepsT]):
    """What CLAI gives a plugin to talk to: the terminal, the conversation, and its saved settings.

    A plugin declares what it contributes on its `Plugin` subclass; the host only carries context.
    """

    def __init__(
        self,
        *,
        name: str,
        console: Console,
        settings: dict[str, JsonValue],
        full_screen: FullScreen = bare_screen,
        conversation: Conversation | None = None,
        session_id: Callable[[], str | None] = lambda: None,
        status: Status | None = None,
        save_settings: Callable[[dict[str, JsonValue]], None] = lambda _settings: None,
        requirements: Requirements | None = None,
    ) -> None:
        """`settings` is the raw JSON from `plugins add`; validate it with `settings(Model)`.

        `save_settings` writes changed settings back to the plugin's declaration; without it they
        last until the plugin unloads.

        The shell passes its own `conversation` and `status`; a host built elsewhere gets a
        `Transcript` and a detached status row, so a plugin needs no special case for either.
        """
        self.name = name
        self.console = console
        self.full_screen = full_screen
        """Own the whole terminal for a widget mid-run.

        `async with host.full_screen():` flushes streamed output and pauses the status row until
        the block exits, so a full-screen menu opened from inside a tool call draws on a settled
        screen. Between turns it is a no-op.
        """
        self.conversation: Conversation = conversation if conversation is not None else Transcript()
        self._session_id = session_id
        self.status = status if status is not None else Status()
        self._settings = settings
        self._persist = save_settings
        self._requirements: Requirements = dict(requirements or {})

    @property
    def session_id(self) -> str | None:
        """The current saved conversation's ID, or `None` for a host without session persistence."""
        return self._session_id()

    @property
    def requirements(self) -> Requirements:
        """Feature names each setting key needs, as declared with `settings(Model, requires=...)`."""
        return dict(self._requirements)

    def settings(self, model: type[ModelT], /, *, requires: Mapping[str, Iterable[str]] | None = None) -> ModelT:
        """Validate the JSON given to `plugins add` against the plugin's own model.

        `requires` names the features a setting's saved value depends on, such as
        `{'sub_agents': ['stock-bound-delegation']}`. CLAI stores the tags beside the declaration
        whenever the settings are saved, and a build missing a feature ignores that setting and uses
        its default. Keys are the saved names (aliases included); see `PLUGINS.md`.
        """
        if requires is not None:
            declared = declared_requirements(requires)
            known = {name for key, info in model.model_fields.items() for name in (key, info.alias) if name}
            unknown = declared.keys() - known
            if unknown:
                raise ValueError(
                    f'{model.__name__} has no setting {", ".join(sorted(unknown))} to require features for.'
                )
            self._requirements = declared
        return model.model_validate(self._settings)

    def save_settings(self, settings: BaseModel, /) -> None:
        """Remember new settings for this plugin, as `plugins add` would; they are stored in plaintext.

        Never save a secret: keep it in `/keys` and save a `KeyReference` naming it. A host built
        outside the loader keeps the change for this load only. Raises `ValueError`, saving nothing,
        when the dump is not a JSON object that validates back into the model, as `host.settings`
        will need on the next load.
        """
        try:
            saved = _SAVED_SETTINGS.validate_python(settings.model_dump(mode='json', by_alias=True))
            type(settings).model_validate(saved)
        except ValidationError as exc:
            raise ValueError(
                f'{type(settings).__name__} cannot be saved as plugin settings: they must dump to a JSON'
                ' object that validates back into the model.'
            ) from exc
        self._persist(saved)
        self._settings = saved


class NoSettings(BaseModel):
    """The settings of a plugin that takes none: any key in its declaration is an error."""

    model_config = ConfigDict(extra='forbid', frozen=True)


SettingsT = DefaultTypeVar('SettingsT', bound=BaseModel, default=NoSettings, covariant=True)


def _settings_argument(cls: type) -> object:
    """`Plugin`'s settings argument as `cls` binds it, in terms of `cls`'s own type parameters."""
    bases = ((base, get_origin(base) or base) for base in get_original_bases(cls))
    base, origin = next(
        (base, origin) for base, origin in bases if isinstance(origin, type) and Plugin in origin.__mro__
    )
    settings = SettingsT if origin is Plugin else _settings_argument(origin)
    parameters: tuple[object, ...] = getattr(origin, '__parameters__', ())
    return _bind(settings, dict(zip(parameters, get_args(base))))


def _bind(argument: object, bound: dict[object, object]) -> object:
    """Substitute `bound` type parameters into `argument`, including a generic settings model's own."""
    model = _as_model(argument)
    if model is None:
        return bound.get(argument, argument)
    parameters = model.__pydantic_generic_metadata__['parameters']
    arguments = tuple(_bind(parameter, bound) for parameter in parameters)
    if arguments == parameters:
        return model
    specialized: object = model[arguments]
    return specialized


def _as_model(argument: object) -> type[BaseModel] | None:
    return argument if isinstance(argument, type) and issubclass(argument, BaseModel) else None


class Plugin(Generic[SettingsT, DepsT]):
    """A CLAI plugin, declared by overriding what it contributes, as an `AbstractCapability` is.

    Parametrize with the settings model, `class Linear(Plugin[LinearSettings])`: the loader validates
    the declaration's JSON against it and passes the result as `settings`. When the plugin loads,
    CLAI calls every `get_*` method once and keeps what they return until it unloads, so build
    tools, commands, and other contributions there or in `__init__`, not per call. Every method has
    a default that contributes nothing, so a plugin overrides only what it needs.

    Hooks into the agent run itself (`before_model_request`, typed events, and the rest) belong on
    a capability returned from `get_capabilities`: `Hooks`, or an `AbstractCapability` subclass.
    """

    settings_type: ClassVar[type[BaseModel]] = NoSettings
    """The settings model, read from the class's `Plugin[...]` parameter; there is no need to set it."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Record the settings model named in `Plugin[SettingsModel, ...]`."""
        super().__init_subclass__(**kwargs)
        settings = _as_model(_settings_argument(cls))
        if settings is not None:
            cls.settings_type = settings

    def __init__(self, host: PluginHost[DepsT], settings: SettingsT) -> None:
        """Keep the host and validated settings; override to set up state the `get_*` methods share."""
        self.host = host
        self._settings = settings

    @classmethod
    def from_host(cls, host: PluginHost[DepsT]) -> Self:
        """Build the plugin from the settings saved in its declaration, as the loader does."""
        return cls(host, cast(SettingsT, host.settings(cls.settings_type)))

    @property
    def settings(self) -> SettingsT:
        """The settings the plugin loaded with. A settings menu reads fresh ones with `host.settings`."""
        return self._settings

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        """Tools, instructions, hooks, or a capability chosen per run, bound on every run."""
        return ()

    def get_commands(self) -> Sequence[Command]:
        """Slash commands, registered while the plugin is loaded."""
        return ()

    def get_status_segments(self) -> Sequence[StatusSegment]:
        """Short fragments for the status row, such as the working directory.

        The shell repaints the row about ten times a second, so keep each fragment cheap and
        synchronous: it is called for every frame, not once per turn. Fragments are appended in
        load order, painted `MUTED`, and truncated from the right on a narrow terminal.
        """
        return ()

    def get_spinners(self) -> Sequence[Spinner]:
        """Working animations for `/spinner`; build each with `make_spinner`.

        A plugin spinner replaces a builtin of the same name, and the user's `spinners.json` replaces
        both. Unloading the plugin removes it; a selected spinner that is gone shows `working`.
        """
        return ()

    def get_model_providers(self) -> Sequence[ModelProvider]:
        """Model prefixes this plugin runs, offered in `/model add` and `/set model`."""
        return ()

    def get_logins(self) -> Sequence[PluginLogin]:
        """Sign-ins this plugin adds to `/login`, such as for the subscription behind a model provider."""
        return ()

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        """Draw an event yourself; return `None` to fall back to the default display."""
        return None

    async def configure(self) -> str:
        """A settings menu, opened by `/plugins configure NAME`, `c` in `/plugins`, and on enable or add.

        Build it on `FieldMenu` and `run_flow` so it ends with the shared Save & close row. Save each
        change with `host.save_settings` as the user makes it and return a line to show. When the
        settings changed, the loader loads the plugin again afterwards, so it builds from them.
        """
        raise NotImplementedError(f'{type(self).__name__} has no settings menu.')

    async def on_session_start(self, event: SessionStart) -> None:
        """CLAI is ready for prompts; called once, when the plugin loads."""

    async def on_session_end(self, event: SessionEnd) -> None:
        """CLAI is quitting, the plugin is unloading, or it failed to load after it was built."""

    async def on_plugin_load_failed(self, event: PluginLoadFailed) -> None:
        """A startup plugin failed to load; called after all enabled plugins have been tried."""

    async def on_turn_start(self, event: TurnStart) -> None:
        """A prompt was submitted; edit `event.text` or call `event.cancel()`. A failure cancels the turn."""

    async def on_turn_end(self, event: TurnEnd) -> None:
        """The turn finished."""

    @property
    def has_configure(self) -> bool:
        """Whether this plugin overrides `configure`, offering a settings menu."""
        return type(self).configure is not Plugin.configure

    @property
    def has_render(self) -> bool:
        """Whether this plugin overrides `render`."""
        return type(self).render is not Plugin.render

    @property
    def hook_count(self) -> int:
        """How many host events this plugin overrides a handler for."""
        return sum(getattr(type(self), name) is not getattr(Plugin, name) for name in _HANDLERS.values())


_HANDLERS: dict[type[HostEvent], str] = {
    SessionStart: 'on_session_start',
    SessionEnd: 'on_session_end',
    TurnStart: 'on_turn_start',
    TurnEnd: 'on_turn_end',
    PluginLoadFailed: 'on_plugin_load_failed',
}


@dataclass(frozen=True, kw_only=True)
class LoadedPlugin(Generic[DepsT]):
    """What a loaded plugin contributes, collected once from its `get_*` methods."""

    plugin: Plugin[BaseModel, DepsT]
    capabilities: tuple[AgentCapability[DepsT], ...]
    commands: Commands
    status_segments: tuple[StatusSegment, ...]
    spinners: tuple[Spinner, ...]
    model_providers: tuple[ModelProvider, ...]
    logins: tuple[PluginLogin, ...]

    @property
    def host(self) -> PluginHost[DepsT]:
        """The host the plugin was built with."""
        return self.plugin.host

    async def dispatch(self, event: HostEvent) -> None:
        """Hand a host event to the plugin's matching `on_*` method."""
        await getattr(self.plugin, _HANDLERS[type(event)])(event)

    def summary(self) -> str:
        """One line for the `/plugins` menu, naming only what the plugin contributes."""
        counts = (
            (len(list(self.commands)), 'command', 'commands'),
            (self.plugin.hook_count, 'hook', 'hooks'),
            (len(self.capabilities), 'capability', 'capabilities'),
            (int(self.plugin.has_render), 'renderer', 'renderers'),
            (len(self.status_segments), 'status segment', 'status segments'),
        )
        parts = [f'{count} {one if count == 1 else many}' for count, one, many in counts if count]
        return ', '.join(parts) or 'nothing yet'


def collect(plugin: Plugin[BaseModel, DepsT]) -> LoadedPlugin[DepsT]:
    """Call each of `plugin`'s `get_*` methods once; a command name collision raises `ValueError`."""
    commands = Commands()
    commands.register_many(plugin.get_commands())
    return LoadedPlugin(
        plugin=plugin,
        capabilities=tuple(plugin.get_capabilities()),
        commands=commands,
        status_segments=tuple(plugin.get_status_segments()),
        spinners=tuple(replace(spinner, source='plugin') for spinner in plugin.get_spinners()),
        model_providers=tuple(plugin.get_model_providers()),
        logins=tuple(plugin.get_logins()),
    )


def load_plugin(plugin_type: type[Plugin[BaseModel, DepsT]], host: PluginHost[DepsT]) -> LoadedPlugin[DepsT]:
    """Build `plugin_type` from `host`'s settings and collect what it contributes, as the loader does.

    This is also how a plugin's tests load it without a shell; fire host events with `dispatch`.
    """
    return collect(plugin_type.from_host(host))
