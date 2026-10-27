"""The `browse_web` toolset and the factory contract for building browser-use agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Literal, Protocol
from urllib.parse import quote_plus, urlsplit

import anyio
from pydantic import BaseModel, JsonValue, ValidationError

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.browser_use._model import RunChatModel, resolve_chat_model
from pydantic_ai_harness.browser_use._settings import BrowserAgentSettings

try:
    from browser_use import Agent as _BrowserUseAgent, Tools
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.browser.profile import DOMAIN_OPTIMIZATION_THRESHOLD
    from browser_use.llm.base import BaseChatModel
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

logger = logging.getLogger(__name__)

_TOOL_NAME = 'browse_web'
# browser-use matches a prohibited URL pattern case-sensitively against the raw URL, while
# Chromium reads the scheme case-insensitively and drops leading spaces or control characters and
# any tab or newline. `file://*` alone would let `FILE://localhost/...`, ` file://localhost/...`,
# or `fi<TAB>le://localhost/...` through, so the patterns also refuse a URL that does not start
# with a letter or that contains a tab or newline (each ends in `/*` so browser-use matches it
# against the whole URL).
_LOCAL_FILE_PATTERNS = ['[fF][iI][lL][eE]://*', '[!a-zA-Z]*/*', '*[\t\n\r]*/*']
_LOCALHOST_PATTERNS = ['localhost', 'localhost.*', '*.localhost']
_LOCALHOST_HOSTS = ('localhost', 'localhost.example', 'example.localhost')

_HTTP_SCHEME_GLOB = 'http*'
_AUTHORITY_END = re.compile(r'[/?#]')

_FILE_SCHEME_ALLOWLIST_ERROR = (
    'An `allowed_domains` entry that permits only `file://` URLs is not supported; '
    'local file navigation is always prohibited.'
)

# Teardown runs shielded from cancellation, so an unresponsive browser could otherwise hang the
# caller forever on exit. Bound it instead: a browser that will not close within this window is
# retained for a later cleanup attempt, which is strictly better than wedging the run.
_TEARDOWN_TIMEOUT = 30


def _strip_trailing_dot(hostname: str) -> str:
    """Drop a terminal DNS dot, which names the same host: `localhost.` is `localhost`.

    Dropping it makes an entry marginally stricter than the caller wrote, because
    browser-use matches hostnames literally and so will not match a `https://host./`
    URL against the canonical pattern. That is the safe direction: left in place, the
    dot lets an entry name a host without resembling it.
    """
    return hostname[:-1] if hostname.endswith('.') else hostname


def _restrict_to_http_schemes(domain: str) -> list[str]:
    """Scheme-qualify a host-only allowlist entry so it cannot admit a `file://` URL."""
    domain = _strip_trailing_dot(domain)
    restricted = [f'{_HTTP_SCHEME_GLOB}://{domain}']
    if '*' not in domain and domain.count('.') == 1:
        restricted.append(f'{_HTTP_SCHEME_GLOB}://www.{domain}')
    return restricted


def _normalize_allowed_domain(domain: str) -> list[str]:
    """Make an allowlist entry safe for browser-use's URL matching.

    browser-use consults `allowed_domains` or `prohibited_domains`, never both:
    a non-empty allowlist short-circuits the prohibition list entirely. A host-only
    entry matches on hostname regardless of scheme, so it would admit
    `file://<host>/...` and override the local-file prohibition. Qualifying such an
    entry to `http`/`https` keeps the caller's intent (including a local dev server
    on `http://localhost`) while leaving the file scheme unreachable.

    Entries already scoped to a scheme are left alone unless that scheme admits `file`:
    a scheme glob is narrowed to whichever of `http`/`https` it already matched, and an
    entry naming `file` alone is dropped. Narrowing emits the matched schemes explicitly
    rather than a single `http*`, because a glob can admit `file` and only one of the two
    (`????` matches `http` but not `https`), and widening to both would newly permit a
    scheme the caller never allowed. browser-use restricts bare `*.example.com` patterns
    to `http`/`https` itself, so those need no qualification.

    The scheme and host are lowercased. browser-use compares a host-only entry
    case-insensitively, but matches a scheme-qualified one with case-sensitive `fnmatch`
    against the lowercased URL, so `Example.com` would otherwise block its own site.
    """
    if '://' in domain:
        scheme, rest = domain.split('://', maxsplit=1)
        authority, slash, path = rest.partition('/')
        scheme = scheme.lower()
        rest = f'{authority.lower()}{slash}{path}'
        domain = f'{scheme}://{rest}'
        if fnmatch('file', scheme):
            # Each narrowed entry is normalized again, so it gets the path boundary below.
            return [
                entry
                for matched in ('http', 'https')
                if fnmatch(matched, scheme)
                for entry in _normalize_allowed_domain(f'{matched}://{rest}')
            ]
        parsed = urlsplit(domain)
        if (
            parsed.scheme in ('http', 'https')
            and parsed.netloc
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and '*' not in domain
        ):
            return [f'{domain}/*']
        return [domain]
    domain = domain.lower()
    if domain.startswith('*.'):
        return [domain]
    return _restrict_to_http_schemes(domain)


def _normalize_allowed_domains(allowed_domains: list[str] | set[str]) -> list[str]:
    """Normalize an allowlist into the pattern list browser-use matches entry by entry.

    browser-use matches a set, and any list of `DOMAIN_OPTIMIZATION_THRESHOLD` or more
    entries (which it converts to a set), by exact hostname only, so a normalized,
    scheme-qualified entry would match nothing and block every URL. A set therefore
    becomes a sorted list, and a list that would cross the threshold is rejected.
    """
    domains = sorted(allowed_domains) if isinstance(allowed_domains, set) else allowed_domains
    normalized = list(dict.fromkeys(entry for domain in domains for entry in _normalize_allowed_domain(domain)))
    if domains and not normalized:
        raise ValueError(_FILE_SCHEME_ALLOWLIST_ERROR)
    if len(normalized) >= DOMAIN_OPTIMIZATION_THRESHOLD:
        raise ValueError(
            f'`allowed_domains` normalizes to {len(normalized)} entries, but browser-use matches an allowlist of '
            f'{DOMAIN_OPTIMIZATION_THRESHOLD} or more entries by exact hostname only, which would block every URL. '
            'Use fewer entries, for example `*.`-prefixed domain globs.'
        )
    return normalized


def _glob_hostname(pattern: str) -> str:
    """The hostname of an allowlist pattern, read as an fnmatch glob rather than a URL.

    `urlsplit` cannot be used here: it rejects a leading character class such as
    `[ab]*.example.com` as a malformed IPv6 literal, and it will not parse a glob
    scheme like `http*://`. The authority is split by hand instead. A trailing
    `:port` is dropped only when the colon falls outside a bracketed IPv6 literal,
    so `[::1]` keeps its address while `localhost:*` loses its port glob.
    """
    authority = pattern.split('://', maxsplit=1)[-1].split('/', maxsplit=1)[0]
    authority = authority.rpartition('@')[2]
    host, separator, port = authority.rpartition(':')
    if separator and ']' not in port:
        authority = host
    return _strip_trailing_dot(authority.lower())


def _pattern_allows_localhost(pattern: str) -> bool:
    """Whether a browser-use allowlist pattern would permit a localhost URL.

    Every allowlist reaching here is normalized first, so an entry is either
    scheme-qualified or a `*.`-prefixed domain glob.

    Matching representative hosts against the entry catches globs, but only for hosts
    the samples name. A concrete `<label>.localhost` is loopback under RFC 6761 without
    resembling any sample, so the suffix is tested directly as well.
    """
    if pattern.startswith('*.'):
        domain = _strip_trailing_dot(pattern[2:].lower())
        return any(host == domain or host.endswith(f'.{domain}') for host in _LOCALHOST_HOSTS)
    hostname = _glob_hostname(pattern)
    if hostname.endswith('.localhost') or hostname.startswith('localhost.'):
        return True
    return any(fnmatch(host, hostname) for host in _LOCALHOST_HOSTS)


def _exclude_localhost_allowlist_entries(allowed_domains: list[str]) -> list[str]:
    """Keep an allowlist from overriding localhost prohibitions."""
    filtered = [domain for domain in allowed_domains if not _pattern_allows_localhost(domain)]
    if allowed_domains and not filtered:
        raise ValueError('An `allowed_domains` entry that permits only localhost requires `block_ip_addresses=False`.')
    return filtered


def _session_prohibited_domains(
    prohibited_domains: list[str] | set[str] | None,
    *,
    block_ip_addresses: bool,
    allowed_domains: list[str] | set[str] | None,
) -> list[str]:
    """The profile's prohibited patterns plus the harness's `file://` and, when blocking IPs, localhost rules.

    browser-use matches a prohibited list of `DOMAIN_OPTIMIZATION_THRESHOLD` or more entries
    (which it converts to a set) by exact hostname only, which would disable those rules, so
    such a list is rejected. An allowlist replaces the prohibited list in browser-use's
    navigation guard, so the size only matters without one.
    """
    merged = list(prohibited_domains or ())
    rules = [*_LOCAL_FILE_PATTERNS, *(_LOCALHOST_PATTERNS if block_ip_addresses else ())]
    merged.extend(rule for rule in rules if rule not in merged)
    if not allowed_domains and len(merged) >= DOMAIN_OPTIMIZATION_THRESHOLD:
        raise ValueError(
            f"`prohibited_domains` grows to {len(merged)} entries with the harness's own rules, but browser-use "
            f'matches a list of {DOMAIN_OPTIMIZATION_THRESHOLD} or more entries by exact hostname only, which would '
            'disable the `file://` and localhost rules. Use fewer entries, or set `allowed_domains`.'
        )
    return merged


def _secret_scope(pattern: str) -> str:
    """The browser-use `sensitive_data` domain key matching the same sites as a normalized allowlist entry.

    browser-use's secret matcher compares scheme and host only and rejects path globs, so
    the path, query, fragment, and userinfo are dropped. A host-only `*.`-glob gets the `http`/`https` scheme glob the
    navigation guard implies for it.
    """
    scheme, separator, rest = pattern.partition('://')
    if not separator:
        return f'{_HTTP_SCHEME_GLOB}://{pattern}'
    return f'{scheme}://{_AUTHORITY_END.split(rest, maxsplit=1)[0].rpartition("@")[2]}'


def _scope_flat_secrets(
    sensitive_data: dict[str, str | dict[str, str]] | None,
    allowed_domains: list[str] | set[str] | None,
) -> dict[str, str | dict[str, str]] | None:
    """Scope flat secrets to the allowlisted sites instead of every page.

    browser-use substitutes a flat secret on any page, including `data:` URLs, which its
    navigation guard admits whatever the allowlist says. Keying each flat secret by the
    allowlist entries restricts substitution to the hosts the allowlist names. A
    domain-scoped value the caller wrote for the same placeholder wins over the flat one.
    """
    if sensitive_data is None or not allowed_domains:
        return sensitive_data
    flat = {key: value for key, value in sensitive_data.items() if isinstance(value, str)}
    if not flat:
        return sensitive_data
    scoped: dict[str, str | dict[str, str]] = {_secret_scope(pattern): dict(flat) for pattern in allowed_domains}
    # Added after the allowlist scopes: browser-use merges every key matching a page in dict
    # order, so the later, caller-written value wins over the flat one.
    for key, value in sensitive_data.items():
        if isinstance(value, dict):
            existing = scoped.pop(key, None)
            scoped[key] = {**existing, **value} if isinstance(existing, dict) else value
    return scoped


def _secret_placeholders(sensitive_data: dict[str, str | dict[str, str]] | None) -> dict[str, str]:
    """Each secret value, and its JSON-escaped and `quote_plus`-encoded forms, mapped to its placeholder.

    The mapping redacts tool output.

    browser-use substitutes a placeholder in the parameters of every action, the final
    `done` text included, so a value can come back in the result. Schema output escapes
    characters such as `"`, hence the JSON-escaped form. A blocked `search` reports the
    URL it built with `quote_plus` (`p%40ss+word`), its only URL encoding of a value.
    """
    if sensitive_data is None:
        return {}
    secrets: list[tuple[str, str]] = []
    for key, value in sensitive_data.items():
        secrets.extend(value.items() if isinstance(value, dict) else [(key, value)])
    named = [(value, f'<secret>{key}</secret>') for key, value in secrets if value]
    placeholders: dict[str, str] = {}
    # Placeholders and raw values first, so a text that is one secret's value and another's
    # encoded form keeps the placeholder of the secret whose value it is.
    for value, placeholder in named:
        # Mapped to itself so a placeholder already in the text is consumed whole,
        # before a value it happens to contain can match inside it.
        placeholders.setdefault(placeholder, placeholder)
        placeholders.setdefault(value, placeholder)
    for value, placeholder in named:
        placeholders.setdefault(json.dumps(value, ensure_ascii=False)[1:-1], placeholder)
        # `surrogatepass` keeps a lone surrogate (as `os.environ` decodes invalid UTF-8) from raising.
        placeholders.setdefault(quote_plus(value, errors='surrogatepass'), placeholder)
    return placeholders


async def _kill(session: BrowserSession) -> bool:
    """Close a browser session, even while the caller is being cancelled.

    `BrowserSession.kill` is not a single round-trip: it saves storage state,
    dispatches a stop event, and drains the event bus, so it suspends several
    times over CDP. Unshielded, the first of those awaits inside a cancelled
    scope raises and leaves a live Chromium behind -- holding a lock on
    `user_data_dir` if the profile has one. The same shape as `ModalSandbox`'s
    teardown, for the same reason.

    Its failures are swallowed for that same reason. `kill()` can raise -- it awaits a
    storage-state save, a forced stop and an event-bus drain, none of which are wrapped
    upstream -- and it runs in a `finally`, where a raise replaces whatever was unwinding
    through it: a completed browse becomes a `TimeoutError` to the caller, a real error
    becomes a teardown error, and a cancellation stops propagating. The browser is left to
    the caller retains the session for another attempt, so nothing is gained by
    letting it through.
    """
    succeeded = True
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(_TEARDOWN_TIMEOUT) as timeout_scope:
            try:
                await session.kill()
            except Exception:
                succeeded = False
                logger.warning('browser-use session teardown failed; retaining the session for retry', exc_info=True)
        if timeout_scope.cancel_called:
            succeeded = False
            logger.warning(
                'browser-use session teardown timed out after %s seconds; retaining the session for retry',
                _TEARDOWN_TIMEOUT,
            )
    return succeeded


class BrowserAgentHistory(Protocol):
    """The subset of browser-use's `AgentHistoryList` that the `browse_web` tool reads.

    `final_result` is the text of the agent's final `done` action (or `None` when
    it never finished), `errors` collects per-step error messages, `is_successful`
    is the agent's own verdict on the finished task (`None` while not done), and
    `structured_output` is the final result parsed against the configured output
    schema (`None` when no schema was configured; raises a pydantic
    `ValidationError` when the result does not parse). A real `AgentHistoryList`
    satisfies this protocol as-is.
    """

    def final_result(self) -> None | str:
        """The text of the final result, or `None` when the agent never finished."""
        ...  # pragma: no cover

    def errors(self) -> list[str | None]:
        """One entry per step: the step's error message, or `None` for clean steps."""
        ...  # pragma: no cover

    def is_successful(self) -> bool | None:
        """The agent's own success verdict for a finished task; `None` while not done."""
        ...  # pragma: no cover

    @property
    def structured_output(self) -> BaseModel | None:
        """The final result parsed against the configured output schema, if any."""
        ...  # pragma: no cover


class BrowserAgent(Protocol):
    """A ready-to-run browser agent for one task, as built by a `BrowserAgentFactory`."""

    def run(self, max_steps: int = 500) -> Awaitable[BrowserAgentHistory]:
        """Run the agent's own loop until the task finishes or `max_steps` is reached.

        Declared as returning `Awaitable` (not `async def`) so that
        `browser_use.Agent.run`, whose tracing decorator types it as returning
        a plain `Coroutine`, satisfies the protocol; an `async def`
        implementation satisfies it too.
        """
        ...  # pragma: no cover


@dataclass
class BrowserTask:
    """Everything the `browse_web` tool passes to a `BrowserAgentFactory` for one call.

    A dataclass rather than keyword arguments so that new fields can be added
    without breaking existing factories: unpack what you forward, ignore the
    rest.
    """

    task: str
    """The natural-language goal for the browser agent."""

    llm: BaseChatModel
    """The resolved chat model: the capability's `llm`, or the host run's model wrapped in `PydanticAIChatModel`."""

    browser_session: BrowserSession
    """The session to browse in. Owned by the tool: killed after the call in
    `'call'` scope, kept alive and reused in `'agent'` scope."""

    use_vision: bool | Literal['auto']
    """Whether to send page screenshots to the model (`'auto'` follows the model's capabilities)."""

    output_schema: type[BaseModel] | None
    """Schema the agent's final result must conform to, forwarded as browser-use's `output_model_schema`."""

    sensitive_data: dict[str, str | dict[str, str]] | None = field(repr=False)
    """Secret placeholders for browser-use to substitute without showing the values to the model.

    Flat values arrive keyed by each `allowed_domains` entry, so browser-use substitutes them
    only on allowlisted pages. Kept out of `repr()`: a `BrowserTask` is what a factory receives,
    so it is the object most likely to end up in a log line or a traceback."""

    extend_system_message: str | None
    """Extra instructions appended to the browser agent's own system prompt."""

    settings: BrowserAgentSettings
    """The remaining browser-use `Agent` options, always a concrete instance.

    Its `*_llm` fields arrive resolved to browser-use chat models, so factories
    can forward them verbatim.
    """


class _DisabledTelemetry:
    """A no-op stand-in for browser-use's process-wide `ProductTelemetry` client on one agent.

    browser-use's end-of-run event carries the task, the visited URLs, and the final
    result. Replacing the client on the agent instance keeps that event in the process
    without touching `ANONYMIZED_TELEMETRY` or the singleton other browser-use users share.
    """

    def capture(self, event: object) -> None:
        """Drop the event."""

    def flush(self) -> None:
        """Nothing is queued, so there is nothing to flush."""


class BrowserAgentFactory(Protocol):
    """Builds the browser agent that `browse_web` runs for one task.

    The default factory constructs a real `browser_use.Agent` from the
    `BrowserTask`, forwarding `BrowserTask.settings` in full. Pass a custom one
    via `BrowserUse.browser_agent` to intercept construction, or to substitute
    a fake in tests. Two rules: the factory must not start or stop the session
    itself (`browse_web` owns the session lifecycle), and it should keep
    browser-use's signal handling off (`enable_signal_handler=False`) -- the
    sub-agent must not install its own SIGINT handling inside a host
    application.

    With `sensitive_data` set, `request.settings.tools` arrives without
    browser-use's `evaluate` action. A factory that builds its own tools or
    agent must exclude it too; the capability cannot enforce that.
    """

    def __call__(self, request: BrowserTask) -> BrowserAgent:
        """Build a runnable browser agent for one `browse_web` call."""
        ...  # pragma: no cover


def _safe_tools(settings: BrowserAgentSettings) -> Tools[None]:
    """Return tools without unapproved file reads or uploads.

    browser-use 0.13.7's `read_file` action calls `FileSystem.read_file_structured`,
    which imports pypdf 6.10.2 for PDFs. Re-evaluate this restriction when
    browser-use publishes a release with pypdf 6.14.2 or later:
    https://github.com/browser-use/browser-use/commit/5405febce2d8834737bc7cd9afee9ad4604ec447

    browser-use substitutes secret placeholders into every action's parameters, and its
    allowlist checks navigations, not requests a page script makes. With secrets configured,
    the `evaluate` action could therefore send a substituted value to any origin, so it is
    removed.
    """
    excluded = ['read_file', 'upload_file']
    if settings.tools is None:
        return Tools(exclude_actions=excluded, display_files_in_done_text=settings.display_files_in_done_text)
    for action in excluded:
        settings.tools.exclude_action(action)
    return settings.tools


def _without_evaluate(settings: BrowserAgentSettings) -> BrowserAgentSettings:
    """`settings` with `evaluate` removed from its tools, for any factory to forward beside secrets.

    Only `evaluate`: unlike the file actions, which a custom factory may re-enable with its own
    controls, nothing a factory adds makes a page script safe beside substituted secrets.
    """
    if settings.tools is None:
        tools = Tools[None](
            exclude_actions=['evaluate'], display_files_in_done_text=settings.display_files_in_done_text
        )
        return replace(settings, tools=tools)
    settings.tools.exclude_action('evaluate')
    return settings


def default_browser_agent(request: BrowserTask) -> BrowserAgent:
    """Build a real `browser_use.Agent` (the default `BrowserAgentFactory`).

    The `resolve_chat_model` calls on the settings' `*_llm` fields narrow their
    static type; the toolset already resolved the values, so at runtime they
    pass through unchanged.
    """
    settings = request.settings
    if request.sensitive_data is not None:
        # A no-op for a request `browse_web` built; it covers a `BrowserTask` built by hand.
        settings = _without_evaluate(settings)
    # Explicit type arguments: `Agent`'s context and structured-output type
    # variables are unconstrained by this call, and neither is used here.
    # Signal handling stays off: the sub-agent must not install its own SIGINT
    # pause/resume handling inside a host application.
    return _BrowserUseAgent[None, BaseModel](
        task=request.task,
        llm=request.llm,
        browser_session=request.browser_session,
        use_vision=request.use_vision,
        output_model_schema=request.output_schema,
        sensitive_data=request.sensitive_data,
        extend_system_message=request.extend_system_message,
        enable_signal_handler=False,
        tools=_safe_tools(settings),
        override_system_message=settings.override_system_message,
        max_failures=settings.max_failures,
        max_actions_per_step=settings.max_actions_per_step,
        use_thinking=settings.use_thinking,
        flash_mode=settings.flash_mode,
        max_history_items=settings.max_history_items,
        page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
        fallback_llm=resolve_chat_model(settings.fallback_llm),
        use_judge=settings.use_judge,
        judge_llm=resolve_chat_model(settings.judge_llm),
        ground_truth=settings.ground_truth,
        calculate_cost=settings.calculate_cost,
        vision_detail_level=settings.vision_detail_level,
        llm_screenshot_size=settings.llm_screenshot_size,
        llm_timeout=settings.llm_timeout,
        step_timeout=settings.step_timeout,
        directly_open_url=settings.directly_open_url,
        include_recent_events=settings.include_recent_events,
        final_response_after_failure=settings.final_response_after_failure,
        enable_planning=settings.enable_planning,
        planning_replan_on_stall=settings.planning_replan_on_stall,
        planning_exploration_limit=settings.planning_exploration_limit,
        loop_detection_enabled=settings.loop_detection_enabled,
        loop_detection_window=settings.loop_detection_window,
        message_compaction=settings.message_compaction,
        max_clickable_elements_length=settings.max_clickable_elements_length,
        include_tool_call_examples=settings.include_tool_call_examples,
        initial_actions=settings.initial_actions,
        file_system_path=settings.file_system_path,
        display_files_in_done_text=settings.display_files_in_done_text,
        save_conversation_path=settings.save_conversation_path,
        save_conversation_path_encoding=settings.save_conversation_path_encoding,
        include_attributes=settings.include_attributes,
        extraction_schema=settings.extraction_schema,
        sample_images=settings.sample_images,
        skills=settings.skills,
        skill_ids=settings.skill_ids,
        pricing_url=settings.pricing_url,
        generate_gif=settings.generate_gif,
        demo_mode=settings.demo_mode,
    )


def _disable_browser_use_telemetry(agent: BrowserAgent) -> None:
    """Replace a browser-use agent's telemetry client with a no-op; other agents are left alone."""
    if isinstance(agent, _BrowserUseAgent):
        # browser-use annotates the attribute with its `ProductTelemetry` class, but the
        # `@singleton` decorator turns that name into a factory function, so no subclass
        # of it can exist; the no-op matches the two methods the agent calls.
        agent.telemetry = _DisabledTelemetry()  # pyright: ignore[reportAttributeAccessIssue]


class BrowserUseToolset(FunctionToolset[AgentDepsT]):
    """Provides the `browse_web` tool: run an autonomous browser-use agent per task."""

    def __init__(
        self,
        *,
        browser_agent: BrowserAgentFactory,
        llm: BaseChatModel | None,
        browser_profile: BrowserProfile | None,
        allowed_domains: list[str] | None,
        block_ip_addresses: bool,
        headless: bool | None,
        max_steps: int,
        use_vision: bool | Literal['auto'],
        output_schema: type[BaseModel] | None,
        sensitive_data: dict[str, str | dict[str, str]] | None,
        extend_system_message: str | None,
        settings: BrowserAgentSettings,
        session_scope: Literal['call', 'agent'],
        cdp_url: str | None,
        browser_use_telemetry: bool = False,
    ) -> None:
        super().__init__()
        self._browser_agent = browser_agent
        self._llm = llm
        self._browser_use_telemetry = browser_use_telemetry
        self._browser_profile = browser_profile
        self._allowed_domains = allowed_domains
        self._block_ip_addresses = block_ip_addresses
        self._headless = headless
        self._max_steps = max_steps
        self._use_vision: bool | Literal['auto'] = use_vision
        self._output_schema = output_schema
        self._sensitive_data = sensitive_data
        self._secret_placeholders = _secret_placeholders(sensitive_data)
        self._secret_pattern = (
            re.compile('|'.join(re.escape(value) for value in sorted(self._secret_placeholders, key=len, reverse=True)))
            if self._secret_placeholders
            else None
        )
        self._extend_system_message = extend_system_message
        # Resolve the settings' chat models once, so every factory (custom
        # ones included) receives a `BrowserTask` with ready-to-use models.
        self._settings = replace(
            settings,
            page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
            fallback_llm=resolve_chat_model(settings.fallback_llm),
            judge_llm=resolve_chat_model(settings.judge_llm),
        )
        self._session_scope: Literal['call', 'agent'] = session_scope
        self._cdp_url = cdp_url
        self._shared_session: BrowserSession | None = None
        self._pending_cleanup: list[BrowserSession] = []
        self._session_closed = False
        self._active_call_sessions = 0
        self._call_cleanup_in_progress = False
        self._call_condition = asyncio.Condition()
        self._cleanup_lock = anyio.Lock(fast_acquire=True)
        self._session_lock = anyio.Lock(fast_acquire=True)
        self.add_function(self.browse_web, name=_TOOL_NAME)

    def _build_session(self) -> BrowserSession:
        """A fresh session, merging the profile with the capability's overrides.

        `BrowserSession` itself merges a provided `browser_profile` with directly
        passed fields, letting the non-`None` direct fields win, so the
        capability's `headless`, `allowed_domains`, and `cdp_url` override the
        profile exactly like they would on a hand-built session. `headless`
        defaults to on only when no profile is given; a profile keeps its own
        setting.

        In `'agent'` scope the session is created with `keep_alive=True`:
        without it, `browser_use.Agent` kills the session at the end of each
        run, which would break reuse across calls. The toolset's own
        `kill()` (in `aclose` and on a failed run) is a force stop and closes
        the browser regardless.
        """
        headless = self._headless
        if headless is None and self._browser_profile is None:
            headless = True
        browser_profile = self._browser_profile if self._browser_profile is not None else BrowserProfile()
        # `BrowserSession` lets a non-`None` `allowed_domains` replace the profile's own
        # list, so the profile allowlist only reaches the navigation guard when the
        # direct one is absent. Normalization also runs in `BrowserUse.__post_init__`,
        # which reports a bad allowlist at construction; repeating it here covers a
        # `BrowserUseToolset` built directly. The transform is idempotent.
        allowed_domains = (
            self._allowed_domains if self._allowed_domains is not None else browser_profile.allowed_domains
        )
        if allowed_domains is not None:
            allowed_domains = _normalize_allowed_domains(allowed_domains)
            if self._block_ip_addresses:
                allowed_domains = _exclude_localhost_allowlist_entries(allowed_domains)
        update: dict[str, object] = {
            'block_ip_addresses': self._block_ip_addresses,
            'prohibited_domains': _session_prohibited_domains(
                browser_profile.prohibited_domains,
                block_ip_addresses=self._block_ip_addresses,
                allowed_domains=allowed_domains,
            ),
        }
        if self._sensitive_data is not None:
            update['cross_origin_iframes'] = False
        browser_profile = browser_profile.model_copy(update=update)
        return BrowserSession(
            cdp_url=self._cdp_url,
            browser_profile=browser_profile,
            headless=headless,
            allowed_domains=allowed_domains,
            keep_alive=True if self._session_scope == 'agent' else None,
        )

    async def _run_agent(self, task: str, llm: BaseChatModel, session: BrowserSession) -> BrowserAgentHistory:
        """Build the sub-agent for `task` against `session` and run its loop."""
        agent = self._browser_agent(
            BrowserTask(
                task=task,
                llm=llm,
                browser_session=session,
                use_vision=self._use_vision,
                output_schema=self._output_schema,
                sensitive_data=_scope_flat_secrets(self._sensitive_data, session.browser_profile.allowed_domains),
                extend_system_message=self._extend_system_message,
                # Here rather than only in `default_browser_agent`, so a custom factory that
                # forwards `settings.tools` drops `evaluate` too.
                settings=self._settings if self._sensitive_data is None else _without_evaluate(self._settings),
            )
        )
        if not self._browser_use_telemetry:
            # Applied here rather than in `default_browser_agent`, so a custom factory
            # that builds a real browser-use agent gets the same policy.
            _disable_browser_use_telemetry(agent)
        return await agent.run(max_steps=self._max_steps)

    def _render_result(self, history: BrowserAgentHistory) -> str:
        """The tool result for a finished run: text, schema JSON, or a failure report."""
        result = history.final_result()
        if result is None:
            step_errors = [error for error in history.errors() if error]
            detail = self._redact('; '.join(step_errors)) if step_errors else 'no further details'
            return f'The browser agent stopped without producing a result ({detail}).'
        answer = self._render_answer(result, history)
        # The verdict is applied to whatever the answer turned out to be, schema JSON included:
        # `structured_output` parses the final result whether or not the sub-agent called `done`
        # with `success=False`, so reading it alone would present a run it gave up on as a clean
        # answer.
        if history.is_successful() is False:
            return f'The browser agent could not fully complete the task. Its final result: {answer}'
        return answer

    def _render_answer(self, result: str, history: BrowserAgentHistory) -> str:
        """The answer itself, redacted: schema JSON when one is configured, otherwise the agent's own text."""
        if self._output_schema is None:
            return self._redact(result)
        try:
            structured = history.structured_output
        except ValidationError as error:
            # The error's text quotes the rejected input as a truncated `repr()`, which redaction
            # cannot match, so with secrets configured only each failing location and message is kept.
            detail = (
                str(error)
                if self._secret_pattern is None
                else '; '.join(
                    f'{".".join(str(part) for part in item["loc"])}: {item["msg"]}'
                    for item in error.errors(include_input=False, include_url=False)
                )
            )
            raise ModelRetry(
                self._redact(
                    f'The browser agent finished, but its result did not match the configured output schema: {detail}'
                )
            ) from error
        # A `None` here is unreachable with browser-use's own history, which parses whenever there
        # is a final result and a schema -- both already true. Only a custom factory's history can
        # land here, and its prose is a better answer than an invented failure.
        if structured is None:
            return self._redact(result)
        if self._secret_pattern is None:
            return structured.model_dump_json()
        # Redacted value by value rather than as text, so the result stays valid JSON.
        return json.dumps(
            self._redact_json(structured.model_dump(mode='json')), ensure_ascii=False, separators=(',', ':')
        )

    async def browse_web(self, ctx: RunContext[AgentDepsT], task: str) -> str:
        """Have an autonomous browser agent carry out a web task and return its result.

        Args:
            ctx: The run context; its model drives the browser agent when no `llm` is configured.
            task: One self-contained web goal in natural language, e.g.
                "find the price of the Pro plan on example.com and return it".

        Returns:
            The browser agent's final text result, or JSON conforming to the
            configured output schema when one is set.
        """
        # Resolved per call rather than cached: the run's model can differ between runs
        # (`agent.run(model=...)`, `agent.override(model=...)`) while the toolset is shared.
        llm = self._llm if self._llm is not None else RunChatModel(ctx)
        if self._session_scope == 'call':
            history = await self._run_in_fresh_session(task, llm)
        else:
            await self._retry_pending_cleanup()
            history = await self._run_in_shared_session(task, llm)
        return self._render_result(history)

    def _redact_json(self, value: JsonValue) -> JsonValue:
        """`value` with every secret in its strings replaced; any other leaf holding one becomes a redacted string.

        Raises `ModelRetry` when two keys of one object redact to the same text, which a
        placeholder shared by domain-scoped secrets allows, rather than dropping one of them.
        """
        if isinstance(value, dict):
            redacted_object: dict[str, JsonValue] = {}
            for key, item in value.items():
                redacted_key = self._redact(key)
                if redacted_key in redacted_object:
                    raise ModelRetry(
                        'The browser agent finished, but more than one key in its result redacts to '
                        f'`{redacted_key}`, so redacting it would drop an entry. '
                        'Ask for a result that is not keyed by secret values.'
                    )
                redacted_object[redacted_key] = self._redact_json(item)
            return redacted_object
        if isinstance(value, list):
            return [self._redact_json(item) for item in value]
        if isinstance(value, str):
            return self._redact(value)
        text = json.dumps(value)
        redacted = self._redact(text)
        return value if redacted == text else redacted

    def _redact(self, text: str) -> str:
        """Replace each configured secret value in tool output with its placeholder.

        One pass, longest match first, so a value that contains another is replaced whole,
        and a placeholder already in the text is left as it is.
        """
        if self._secret_pattern is None:
            return text
        return self._secret_pattern.sub(lambda match: self._secret_placeholders[match.group()], text)

    async def _close_session(self, session: BrowserSession) -> None:
        """Close `session`, retaining its identity when cleanup needs another attempt."""
        with anyio.CancelScope(shield=True):
            if not await _kill(session):
                async with self._cleanup_lock:
                    self._pending_cleanup.append(session)

    async def _retry_pending_cleanup(self) -> None:
        """Retry sessions whose previous teardown failed or timed out."""
        async with self._cleanup_lock:
            pending, self._pending_cleanup = self._pending_cleanup, []
            for session in pending:
                if not await _kill(session):
                    self._pending_cleanup.append(session)

    async def _run_in_fresh_session(self, task: str, llm: BaseChatModel) -> BrowserAgentHistory:
        """One disposable session for one call, killed when the call ends, on success or failure."""
        async with self._call_condition:
            await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
            await self._retry_pending_cleanup()
            self._active_call_sessions += 1
        session: BrowserSession | None = None
        try:
            session = self._build_session()
            return await self._run_agent(task, llm, session)
        finally:
            with anyio.CancelScope(shield=True):
                if session is not None:
                    await self._close_session(session)
                async with self._call_condition:
                    self._active_call_sessions -= 1
                    self._call_condition.notify_all()

    async def _run_in_shared_session(self, task: str, llm: BaseChatModel) -> BrowserAgentHistory:
        """The `'agent'`-scoped shared session; the lock serializes calls -- one browser, one driver at a time."""
        async with self._session_lock:
            if self._session_closed:
                # A call that was queued behind `aclose()` reaches the lock after the browser
                # is gone. Without this it would lazily start a fresh `keep_alive` session that
                # nothing is left to close, so the process would exit with a live Chromium.
                raise RuntimeError(
                    'The shared browser session is closed: `aclose()` was called, so `browse_web` '
                    'cannot open another one. Build a new capability to browse again.'
                )
            if self._shared_session is None:
                self._shared_session = self._build_session()
            try:
                return await self._run_agent(task, llm, self._shared_session)
            except BaseException:
                # A failed or cancelled run can leave the shared browser in an
                # unknown state; kill it so the next call starts fresh. Dropping
                session, self._shared_session = self._shared_session, None
                await self._close_session(session)
                raise

    async def aclose(self) -> None:
        """Kill the shared browser session and refuse to open another.

        In `'agent'` session scope it closes for good, so a later `browse_web`
        raises rather than starting a browser nothing would close. In either
        scope it retries sessions retained after an earlier teardown failure or
        timeout. Safe to call multiple times.

        It coordinates with `browse_web`, so it waits for in-flight calls to
        finish before the final cleanup attempt -- and a call can run for
        `max_steps` steps of up to `BrowserAgentSettings.step_timeout` each.
        Cancel the run first if you need to close sooner.
        """
        if self._session_scope == 'call':
            async with self._call_condition:
                await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
                self._call_cleanup_in_progress = True
                try:
                    await self._call_condition.wait_for(lambda: self._active_call_sessions == 0)
                    await self._retry_pending_cleanup()
                finally:
                    self._call_cleanup_in_progress = False
                    self._call_condition.notify_all()
            return

        async with self._session_lock:
            self._session_closed = True
            if self._shared_session is not None:
                session, self._shared_session = self._shared_session, None
                await self._close_session(session)
        await self._retry_pending_cleanup()
