"""UI interaction telemetry: shared UI helpers say what the user did, and a subscribed Logfire instance exports it.

Nothing is recorded until a sink subscribes. The built-in `observability` plugin subscribes its own instance when
its `ui_events` setting is on, and unsubscribes before it shuts that instance down. Every span and log uses
the `clai2` instrumentation scope, like everything else CLAI emits itself.

Instrument the shared chokepoints (`run_worker`, `Commands.execute_async`, `FieldMenu`, the plugin loader,
`/keys`, the prompt editor) rather than individual menus, so a new menu is covered without extra code.
Attributes name what was chosen (a command, a menu, a setting, a plugin, a key's name), not what was typed:
secrets and free-text values stay out. The one exception is a submitted prompt's text, added through
`prompt_text` only when the subscriber records message content, as agent spans do.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import logfire
from opentelemetry.trace import Span, SpanKind, get_current_span, use_span

Attribute = str | int | float | bool
SCOPE = 'clai2'
"""The instrumentation scope for everything CLAI emits itself: session roots, UI records, and plugin errors."""

NAMES = frozenset({'command', 'menu', 'field', 'choice', 'setting', 'plugin', 'label', 'key_name', 'new_key_name'})
"""Attributes that only ever hold names and listed choices, which `keep_names` exempts from scrubbing."""
PROMPT = 'prompt'
"""The submitted prompt, which `keep_names` also keeps: agent spans carry the same text unscrubbed."""
MAX_CONTENT_CHARS = 64_000
"""Typed text beyond this is cut, so a huge paste cannot make the exporter drop the whole record."""


@dataclass(kw_only=True)
class _Sink:
    instance: logfire.Logfire
    root: Callable[[], Span | None]
    include_content: bool


_sinks: list[_Sink] = []
"""Subscribed instances, newest last; only the newest receives UI telemetry."""
_emitting: ContextVar[bool] = ContextVar('_emitting', default=False)
"""Set while a UI record is handed to its sink, which is when Logfire scrubs it: `keep_names` checks it."""


def subscribe(
    sink: logfire.Logfire, *, root: Callable[[], Span | None] = lambda: None, include_content: bool = False
) -> Callable[[], None]:
    """Send UI telemetry to `sink` until the returned function is called; calling it again does nothing.

    The caller supplies an instance in `SCOPE`. Telemetry goes to the most recently subscribed instance,
    so each destination gets whole, correctly nested traces; when it unsubscribes, the previous one takes over.
    `include_content` lets `prompt_text` add what the user typed, like `InstrumentationSettings.include_content`.
    """
    subscribed = _Sink(instance=sink, root=root, include_content=include_content)
    _sinks.append(subscribed)

    def unsubscribe() -> None:
        if subscribed in _sinks:
            _sinks.remove(subscribed)

    return unsubscribe


@contextmanager
def parent_span(root: Span | None) -> Generator[None]:
    """Use the session root unless the current span already belongs to its trace."""
    if root is None or get_current_span().get_span_context().trace_id == root.get_span_context().trace_id:
        yield
    else:
        with use_span(root, record_exception=False, set_status_on_exception=False):
            yield


@contextmanager
def _exempt() -> Generator[None]:
    token = _emitting.set(True)
    try:
        yield
    finally:
        _emitting.reset(token)


def prompt_text(text: str) -> dict[str, Attribute]:
    """A submitted prompt as the `PROMPT` attribute, cut to `MAX_CONTENT_CHARS`, if the subscriber records content."""
    if not _sinks or not _sinks[-1].include_content:
        return {}
    return {PROMPT: text[:MAX_CONTENT_CHARS]}


def record(msg_template: str, /, **attributes: Attribute) -> None:
    """Log one UI interaction, such as a setting change or a key saved."""
    if _sinks:
        sink = _sinks[-1]
        with parent_span(sink.root()), _exempt():
            sink.instance.log('info', msg_template, attributes=dict(attributes))


class UiSpan:
    """The open span, if anything is subscribed; `set` adds what is only known at the end, such as a cancel."""

    def __init__(self, span: logfire.LogfireSpan | None) -> None:
        """Wrap the span; `None` when nothing is subscribed."""
        self._span = span

    def set(self, key: str, value: Attribute) -> None:
        """Set `key` on the span."""
        if self._span is not None:
            self._span.set_attribute(key, value)


@contextmanager
def span(msg_template: str, /, **attributes: Attribute) -> Generator[UiSpan]:
    """Time a UI interaction that contains others, such as a command that opens a menu.

    Spans nest: a menu opened by a command, and the setting it changes, land under that command's span.
    An exception propagates, but the span records only its type as `error`: messages can quote what was typed,
    such as a token a plugin's settings rejected.
    """
    if not _sinks:
        yield UiSpan(None)
        return
    sink = _sinks[-1]
    with parent_span(sink.root()):
        with _exempt():
            opened = _open(sink.instance, msg_template, attributes).__enter__()
        ui = UiSpan(opened)
        try:
            yield ui
        except BaseException as error:
            ui.set('error', type(error).__name__)
            raise
        finally:
            with _exempt():
                opened.__exit__(None, None, None)


def _open(sink: logfire.Logfire, msg_template: str, attributes: dict[str, Attribute]) -> logfire.LogfireSpan:
    # Every underscored option is spelled out, so no attribute can be mistaken for one.
    return sink.span(
        msg_template,
        _tags=(),
        _span_name=None,
        _level=None,
        _links=(),
        _span_kind=SpanKind.INTERNAL,
        **attributes,
    )


def operation_name(operation: object) -> str:
    """A stable, readable id for the callable behind a menu, such as `ui.menus.model_picker:model_command`.

    Menus are opened through lambdas and partials, so the id is where that callable was written:
    `<locals>` and `<lambda>` are dropped, and so is the `pydantic_clai2.` prefix.
    """
    while isinstance(operation, partial):
        operation = operation.func
    module: object = getattr(operation, '__module__', None)
    qualname: object = getattr(operation, '__qualname__', None)
    if not isinstance(qualname, str):
        qualname = type(operation).__qualname__
    path = '.'.join(part for part in qualname.split('.') if part not in ('<locals>', '<lambda>'))
    prefix = module.removeprefix('pydantic_clai2.') if isinstance(module, str) else ''
    return f'{prefix}:{path}' if prefix else path


def keep_names(match: logfire.ScrubMatch) -> object:
    """A Logfire scrubbing callback that keeps UI telemetry's names, which can look like secrets but are not.

    A setting such as `sessions.naming`, a key's name such as `OPENAI_API_KEY`, or a field such as `auth` trips
    Logfire's default patterns. Only the top-level `NAMES` attributes of UI records, and the message placeholders
    filled from them, are kept, along with `PROMPT`, whose words would otherwise trip the same patterns ("the
    session bug"). Everything else, including every agent span, is scrubbed as usual.
    """
    if (
        _emitting.get()
        and len(match.path) == 2
        and match.path[0] in ('attributes', 'message')
        and (match.path[1] in NAMES or match.path[1] == PROMPT)
    ):
        return match.value
    return None
