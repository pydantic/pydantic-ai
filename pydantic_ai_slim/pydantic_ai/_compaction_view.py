"""How a request's view of the message history relates to the persistent history, for compaction.

`before_model_request`, `wrap_model_request` and `prepare_model_request` hooks can change what one request sends
without changing the history, by assigning a new list to `request_context.messages`. By the documented contract
that list holds the history's own message objects, except for the ones a hook replaced, added or left out, and a
replaced message holds the original's own parts, except for the ones a hook replaced, added or left out. Aligning
the two by object identity tells those edits apart:

- a *substitution* is a hook's copy of something the history holds, standing in for it in this request, like a
  file described for a model that can't read it, or a redacted tool result;
- an *insertion* is something only this request adds, like a reminder appended to a copy of the last request;
- a *removal* is something the history holds that this request leaves out, like content filtered out of it.

A compaction is made from the history as this request shows it, without what it only adds: substitutions applied, and
insertions and removals left out, so a compaction never sees what this request doesn't send. What's committed is the
compacted history with the persistent originals of everything it kept, removals among them, so nothing request-only
becomes persistent, unless the summary itself absorbed it. What's sent is the compacted history with this request's
edits applied to what it kept.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from .messages import ModelMessage, ModelRequest, ModelResponse

T = TypeVar('T')

__all__ = ('CompactionView',)


@dataclass(frozen=True)
class _Aligned:
    """What one item of a request's view is, relative to the originals it was made from."""

    kind: Literal['same', 'copy', 'substitute', 'insert']
    start: int
    """The first original it stands for, or, for an insertion, the original it's inserted before."""
    end: int
    """One past the last original it stands for; for an insertion, `start`."""


def _align(
    originals: Sequence[T],
    view: Sequence[T],
    parts_of: Callable[[T], Sequence[object]] | None = None,
) -> list[_Aligned]:
    """Align `view` with `originals` by identity: what each view item is, and which originals it stands for.

    A view item is the same as an original if it's that object, and, with `parts_of`, a copy of it if they hold one
    part object in common. View items that are neither, in a run between two that are, stand in for the originals
    missing between those two: they're a substitution if any are missing, and an insertion if none are.
    """
    index = {id(item): position for position, item in enumerate(originals)}
    part_index = (
        {id(part): position for position, item in enumerate(originals) for part in parts_of(item)}
        if parts_of is not None
        else {}
    )
    matched: list[tuple[int, Literal['same', 'copy']] | None] = []
    last = -1
    for item in view:
        position = index.get(id(item))
        kind: Literal['same', 'copy'] = 'same'
        if position is None and parts_of is not None:
            kind = 'copy'
            position = next(
                (
                    shared
                    for part in parts_of(item)
                    if (shared := part_index.get(id(part))) is not None and shared > last
                ),
                None,
            )
        if position is not None and position > last:
            matched.append((position, kind))
            last = position
        else:
            matched.append(None)

    aligned: list[_Aligned] = []
    previous = -1
    position = 0
    while position < len(view):
        if (match := matched[position]) is not None:
            original, kind = match
            aligned.append(_Aligned(kind, original, original + 1))
            previous = original
            position += 1
            continue
        run_end = position
        while run_end < len(view) and matched[run_end] is None:
            run_end += 1
        following_match = matched[run_end] if run_end < len(view) else None
        following = len(originals) if following_match is None else following_match[0]
        gap = _Aligned('substitute' if following > previous + 1 else 'insert', previous + 1, following)
        aligned.extend(gap for _ in range(position, run_end))
        position = run_end
    return aligned


def _without_insertions(original: ModelMessage, copy: ModelMessage) -> ModelMessage:
    """`copy` without the parts it only adds to `original`: the original's parts as this request shows them.

    Returns `original` itself when `copy` only added parts.
    """
    if isinstance(original, ModelRequest) and isinstance(copy, ModelRequest):
        request_parts = _shown_parts(original.parts, copy.parts)
        return original if request_parts is None else replace(copy, parts=request_parts)
    if isinstance(original, ModelResponse) and isinstance(copy, ModelResponse):
        response_parts = _shown_parts(original.parts, copy.parts)
        return original if response_parts is None else replace(copy, parts=response_parts)
    return copy


def _shown_parts(original: Sequence[T], copy: Sequence[T]) -> list[T] | None:
    """The parts of `copy` that aren't only added to `original`, or `None` when they're the original's own parts."""
    parts = [part for part, aligned in zip(copy, _align(original, copy)) if aligned.kind != 'insert']
    if len(parts) == len(original) and all(a is b for a, b in zip(parts, original)):
        return None
    return parts


@dataclass
class CompactionView:
    """A request's view of the history, split into what a compaction is made from and what only this request has."""

    history: list[ModelMessage]
    """The persistent history."""
    view: list[ModelMessage]
    """What this request sends, as the model request hooks left it."""
    messages: list[ModelMessage]
    """What the compaction is made from: the history with this request's substitutions, without its insertions."""
    _origins: list[tuple[int, int]]
    """For each of `messages`, the range of `history` it stands for."""
    _aligned: list[_Aligned]
    """For each of `view`, what it is relative to `history`."""
    _persistent: dict[int, ModelMessage]
    """The persistent original of a message in `messages` that stands for one history message, by `id`."""
    _sent: dict[int, ModelMessage]
    """This request's version of a message in `messages` that stands for one history message, by `id`."""

    @classmethod
    def of(cls, history: Sequence[ModelMessage], view: Sequence[ModelMessage]) -> CompactionView:
        history = list(history)
        view = list(view)
        aligned = _align(history, view, lambda message: message.parts)
        messages: list[ModelMessage] = []
        origins: list[tuple[int, int]] = []
        persistent: dict[int, ModelMessage] = {}
        sent: dict[int, ModelMessage] = {}
        # What this request leaves out isn't given to the compaction either, as this request doesn't send it. It's
        # still history: committed if the compaction keeps the messages around it.
        index = 0
        while index < len(view):
            item = aligned[index]
            if item.kind == 'insert':
                index += 1
                continue
            run_end = index + 1
            while run_end < len(view) and aligned[run_end] == item and item.kind == 'substitute':
                run_end += 1
            if item.kind == 'same':
                messages.append(history[item.start])
                origins.append((item.start, item.end))
            elif item.kind == 'copy' or (run_end - index == 1 and item.end - item.start == 1):
                original = history[item.start]
                version = _without_insertions(original, view[index])
                messages.append(version)
                origins.append((item.start, item.end))
                persistent[id(version)] = original
                sent[id(version)] = view[index]
            else:
                # Several messages standing in for several others can't be told apart, so they're one substitution.
                for substitute in view[index:run_end]:
                    messages.append(substitute)
                    origins.append((item.start, item.end))
            index = run_end
        return cls(history, view, messages, origins, aligned, persistent, sent)

    def _kept_start(self, kept: int) -> int:
        """Where in `history` the originals of the last `kept` of `messages` start.

        What this request left out right before them is kept with them: only the messages it showed were compacted.
        """
        first_kept = len(self.messages) - kept
        start = self._origins[first_kept - 1][1] if first_kept > 0 else 0
        if kept:
            # A substitution standing in for several messages is kept whole if any of it is.
            start = min(start, self._origins[first_kept][0])
        return start

    def committed(self, compacted: Sequence[ModelMessage], kept_from: int) -> list[ModelMessage]:
        """The history to commit: `compacted`, with the persistent originals of everything it kept.

        `compacted[kept_from:]` are the last messages of `messages`; their originals start at the same index of the
        result. A message of `messages` kept elsewhere is replaced by its original too.
        """
        start = self._kept_start(len(compacted) - kept_from)
        prefix = [self._persistent.get(id(message), message) for message in compacted[:kept_from]]
        return [*prefix, *self.history[start:]]

    def sent(self, compacted: Sequence[ModelMessage], kept_from: int) -> list[ModelMessage]:
        """The messages to send: `compacted`, with this request's edits to everything it kept.

        A substitution is kept if anything it stands for was kept, and an insertion if it's inserted after the last
        message that wasn't.
        """
        start = self._kept_start(len(compacted) - kept_from)
        prefix = [self._sent.get(id(message), message) for message in compacted[:kept_from]]
        tail = [
            message
            for message, item in zip(self.view, self._aligned)
            if (item.end >= start if item.kind == 'insert' else item.end > start)
        ]
        return [*prefix, *tail]
