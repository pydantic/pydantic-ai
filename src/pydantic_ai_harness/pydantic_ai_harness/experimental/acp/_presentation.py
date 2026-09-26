"""Map Pydantic AI tool calls onto ACP's rich tool-call presentation fields.

ACP lets an agent annotate a tool call with a `kind` (read/edit/execute/...), the file
`locations` it touches, and `content` such as an inline diff. A TUI renders those as
click-to-file links and diff views instead of opaque JSON. This module turns a recognized
`FileSystem`/`Shell` tool call into that presentation; unrecognized calls (or ones whose
arguments do not match the expected shape) return `None` so the adapter falls back to its
generic rendering.

A presenter sees a tool call's (workspace-relative) arguments; ACP requires absolute paths,
so the adapter resolves locations and diffs against the session's working directory via
`absolutize` before sending.
"""

from __future__ import annotations

import os.path
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import acp
from acp import schema
from pydantic_ai.messages import ToolCallPart

# The tool-call content variants ACP accepts on a `session/update` (inline content, a file
# diff, or a reference to a live terminal).
ToolCallContent = schema.ContentToolCallContent | schema.FileEditToolCallContent | schema.TerminalToolCallContent


@dataclass(frozen=True, kw_only=True)
class ToolCallPresentation:
    """How a tool call should be shown in an ACP client.

    Mirrors the rich fields of ACP's tool-call updates. Every field is optional: a presenter
    fills in only what it knows, and the adapter defaults the title to the tool name. Build
    `content` with ACP's helpers, e.g. `acp.tool_diff_content(path, new_text, old_text)` for a
    file edit. Paths may be workspace-relative; the adapter makes them absolute.
    """

    kind: schema.ToolKind | None = None
    title: str | None = None
    locations: tuple[schema.ToolCallLocation, ...] = ()
    content: tuple[ToolCallContent, ...] = ()


# A function deciding how to present a tool call, or `None` to leave it to generic rendering.
# `ToolCallPresentation` is defined above, so the reference is eager (not a string forward ref):
# the alias stays fully resolved when it appears in another module's annotations under
# `typing.get_type_hints`.
ToolCallPresenter = Callable[[ToolCallPart], ToolCallPresentation | None]


def _nonempty_str(args: Mapping[str, object], key: str) -> str | None:
    """Return `args[key]` when it is a non-empty string, else `None`."""
    value = args.get(key)
    return value if isinstance(value, str) and value else None


def _any_str(args: Mapping[str, object], key: str) -> str | None:
    """Return `args[key]` when it is a string -- empty is legitimate for content text -- else `None`."""
    value = args.get(key)
    return value if isinstance(value, str) else None


def _location(path: str) -> schema.ToolCallLocation:
    return schema.ToolCallLocation(path=path)


def _present_read(args: Mapping[str, object]) -> ToolCallPresentation | None:
    path = _nonempty_str(args, 'path')
    if path is None:
        return None
    return ToolCallPresentation(kind='read', locations=(_location(path),))


def _present_edit(args: Mapping[str, object]) -> ToolCallPresentation | None:
    path = _nonempty_str(args, 'path')
    old_text = _nonempty_str(args, 'old_text')
    new_text = _any_str(args, 'new_text')
    if path is None or old_text is None or new_text is None:
        return None
    diff = acp.tool_diff_content(path=path, new_text=new_text, old_text=old_text)
    return ToolCallPresentation(kind='edit', locations=(_location(path),), content=(diff,))


def _present_write(args: Mapping[str, object]) -> ToolCallPresentation | None:
    path = _nonempty_str(args, 'path')
    content = _any_str(args, 'content')
    if path is None or content is None:
        return None
    # `write_file` is usually a create, where omitting `old_text` (the ACP convention for a new
    # file) is correct. On an overwrite the prior contents are unknown from the args alone, so
    # the diff understates what is replaced -- an accepted limitation until reads route through
    # the client.
    diff = acp.tool_diff_content(path=path, new_text=content)
    return ToolCallPresentation(kind='edit', locations=(_location(path),), content=(diff,))


def _present_create_directory(args: Mapping[str, object]) -> ToolCallPresentation | None:
    path = _nonempty_str(args, 'path')
    if path is None:
        return None
    return ToolCallPresentation(kind='other', locations=(_location(path),))


def _present_list(args: Mapping[str, object]) -> ToolCallPresentation | None:
    # `list_directory` takes only an optional `path`, so it has no required argument to validate
    # against -- it is matched by name alone (a same-named custom tool would also render here).
    path = _nonempty_str(args, 'path')
    locations = (_location(path),) if path is not None else ()
    return ToolCallPresentation(kind='search', locations=locations)


def _present_grep(args: Mapping[str, object]) -> ToolCallPresentation | None:
    if _nonempty_str(args, 'pattern') is None:
        return None
    path = _nonempty_str(args, 'path')
    locations = (_location(path),) if path is not None else ()
    return ToolCallPresentation(kind='search', locations=locations)


def _present_run(args: Mapping[str, object]) -> ToolCallPresentation | None:
    if _nonempty_str(args, 'command') is None:
        return None
    return ToolCallPresentation(kind='execute')


def _present_command_ref(args: Mapping[str, object]) -> ToolCallPresentation | None:
    if _nonempty_str(args, 'command_id') is None:
        return None
    return ToolCallPresentation(kind='execute')


# Recognized `FileSystem`/`Shell` tool names mapped to their presenters. Coupling is by tool
# name (a rename in those capabilities silently falls back to generic rendering).
_HANDLERS: dict[str, Callable[[Mapping[str, object]], ToolCallPresentation | None]] = {
    'read_file': _present_read,
    'file_info': _present_read,
    'edit_file': _present_edit,
    'write_file': _present_write,
    'create_directory': _present_create_directory,
    'list_directory': _present_list,
    'search_files': _present_grep,
    'find_files': _present_grep,
    'run_command': _present_run,
    'start_command': _present_run,
    'check_command': _present_command_ref,
    'stop_command': _present_command_ref,
}


def default_coding_presenter(call: ToolCallPart) -> ToolCallPresentation | None:
    """Present a recognized `FileSystem`/`Shell` tool call, else `None`.

    Matches the tool by name and validates its argument shape; a mismatch returns `None` so the
    call falls back to generic rendering. The exception is `list_directory`, whose only argument
    is optional, so it is matched by name alone.
    """
    handler = _HANDLERS.get(call.tool_name)
    if handler is None:
        return None
    # Malformed arguments surface from `args_as_dict` as a sentinel dict no handler matches.
    return handler(call.args_as_dict())


def chain_presenters(*presenters: ToolCallPresenter) -> ToolCallPresenter:
    """Combine presenters, using the first one that returns a presentation.

    Each presenter returns `None` to mean "I do not handle this call"; the chain tries them in
    order and returns the first non-`None` result (or `None` if none match). Use it to add
    rendering for your own tools while keeping the built-in one:
    `chain_presenters(my_presenter, default_coding_presenter)`.
    """

    def presenter(call: ToolCallPart) -> ToolCallPresentation | None:
        for candidate in presenters:
            result = candidate(call)
            if result is not None:
                return result
        return None

    return presenter


def _resolve_within(path: str, cwd: str) -> str | None:
    """Resolve a relative `path` against `cwd`, or `None` if it escapes it; absolute paths pass through."""
    if os.path.isabs(path):
        # May name an advertised additional directory, which the presenter cannot see to validate.
        return path
    resolved = os.path.normpath(os.path.join(cwd, path))
    relative = os.path.relpath(resolved, os.path.normpath(cwd))
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return None
    return resolved


def _within_content(item: ToolCallContent, cwd: str) -> ToolCallContent | None:
    if isinstance(item, schema.FileEditToolCallContent):
        resolved = _resolve_within(item.path, cwd)
        return None if resolved is None else item.model_copy(update={'path': resolved})
    return item


def absolutize(presentation: ToolCallPresentation, cwd: str) -> ToolCallPresentation:
    """Resolve a presentation's workspace-relative paths against the session `cwd`.

    ACP requires tool-call locations and file-edit diffs to carry absolute paths. Absolute paths
    are left unchanged; a relative path that escapes `cwd` via `..` is dropped rather than
    absolutized, so the editor is never pointed outside the workspace. Assumes the agent's
    filesystem tools are rooted at `cwd`.
    """
    locations = tuple(
        loc.model_copy(update={'path': resolved})
        for loc in presentation.locations
        if (resolved := _resolve_within(loc.path, cwd)) is not None
    )
    content = tuple(c for c in (_within_content(item, cwd) for item in presentation.content) if c is not None)
    return replace(presentation, locations=locations, content=content)
