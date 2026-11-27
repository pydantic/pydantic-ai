"""The `search_conversation_history` toolset with dependency-free BM25 ranking.

The rendering core is ported from VStorm's `pydantic-deepagents`
(`pydantic_deep/features/history_archive/toolset.py`); ranking is `_history_bm25`, with BM25
parameters taken from the owning capability. The corpus comes from a `HistorySource` (one document per
persisted message, across the runs selected by the configured scope); ranking is global
within that scope, while context windows around a match stay within the match's run.
"""

from __future__ import annotations

import inspect
import json
import math
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from pydantic_core import ErrorDetails

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
    CapabilityPart,
    CompactionPart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelSettingsDeltaPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    error_details_ta,
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._history_bm25 import Bm25Params, bm25_rank, clip, tokenize
from pydantic_ai_harness._message_text import capability_part_text, tool_return_text, user_prompt_texts
from pydantic_ai_harness._warn import warn_default_changed
from pydantic_ai_harness.conversation_search._source import SUMMARY_PREFIX, HistorySource, message_hash

SEARCH_HISTORY_DESCRIPTION = """\
Search persisted conversation history: earlier turns of this conversation \
(including messages that context compaction dropped from the live context) and \
past runs reachable under the configured scope.

When the conversation is compacted, older messages are replaced by a summary in \
the active context, but the persisted step history keeps the originals. Use this \
tool to find specific details the live context no longer carries.

Results are ranked by relevance using BM25 -- rare terms and exact matches \
score higher. Multi-word queries search each word independently. Each result \
names the run it came from; pass `run_id` to search only one run, e.g. a run a \
compaction receipt references as its full transcript.

When to use:
- You need to recall exact details from earlier in the conversation
- The conversation summary or a compaction receipt lacks a detail you need
- You need specific code, file paths, or decisions from a past run

When NOT to use:
- The information is still in the current conversation context
- You need external or real-time information (use web tools instead)"""

_CONVERSATION_SCOPE_NOTE = (
    '\n\nThis search covers only the current conversation; runs belonging to other '
    'conversations in the same store are not reachable.'
)

SearchScope = Literal['all', 'conversation']
"""How much of the store one search may reach.

`conversation` restricts the corpus to runs whose `conversation_id` matches the calling
run's. `all` searches every run the source enumerates and must only be used when the store
is already isolated to one principal; otherwise, a run can retrieve verbatim excerpts
from other conversations in the store.
"""

SCOPE_DEFAULT_CHANGE_IMPACT = (
    "Searches are now limited to the calling run's conversation instead of every run in the store, "
    'so a caller relying on the old default stops seeing other conversations without an error. '
    'This transitional warning will be removed in the next breaking release; the conversation-scoped default will remain.'
)
"""Why the `scope` default change matters, for the deprecation warning both public entry points emit."""

_EXCERPT_CHARS = 500
"""Display cap for conversational text: prompts, replies, tool returns, retries, transcripts."""

_SHORT_EXCERPT_CHARS = 200
"""Display cap for system text and tool-call arguments."""

_MAX_QUERY_CHARS = 1_000
_MAX_QUERY_TERMS = 32
"""Query bounds, the same as `Memory`'s `search_memory`: ranking costs unique terms times scanned messages."""

_ECHOED_QUERY_CHARS = 200
"""How much of the query the result header repeats back."""

DEFAULT_MAX_SCANNED_MESSAGES = 2_000
"""How many of the newest indexed messages one search scores by default.

Scoring runs synchronously on the event loop, so this bounds how long one search holds it:
a 32-term query over 2,000 short messages takes tens of milliseconds.
"""


def _check_query(query: str) -> None:
    """Reject a query whose ranking would hold the event loop, so the model can retry with a shorter one."""
    if len(query) > _MAX_QUERY_CHARS:
        raise ModelRetry(f'Search queries must be at most {_MAX_QUERY_CHARS} characters.')
    if len(set(tokenize(query))) > _MAX_QUERY_TERMS:
        raise ModelRetry(f'Search queries must contain at most {_MAX_QUERY_TERMS} unique terms.')


def _echo_query(query: str) -> str:
    """The query as the result header repeats it, cut to `_ECHOED_QUERY_CHARS`."""
    return query if len(query) <= _ECHOED_QUERY_CHARS else query[:_ECHOED_QUERY_CHARS] + '...'


def _retry_text(content: list[ErrorDetails] | str) -> str:
    """Retry feedback as text: the plain message, or its validation errors as compact JSON.

    Serialized with core's `error_details_ta`, without `ctx`, like the errors of a retried
    `ToolReturnPart` (which core also strips of an `input` repeated across root-level errors).
    `RetryFeedbackPart.model_response()` would also add the fence and the line that says the
    application is speaking, which do not help a search. The excerpt is clipped like the rest.
    """
    if isinstance(content, str):
        return content
    return error_details_ta.dump_json(content, exclude={'__all__': {'ctx'}}).decode()


def _format_request_part(part: ModelRequestPart, *, terms: frozenset[str] | None) -> str | None:
    """Render one request part to a searchable line, or `None` for non-content parts."""
    match part:
        case UserPromptPart():
            return f'User: {clip(" ".join(user_prompt_texts(part)), _EXCERPT_CHARS, terms)}'
        case SystemPromptPart():
            # Defensive for sources that do not filter compaction artifacts;
            # `SnapshotHistorySource` already excludes them from the corpus.
            if part.content.startswith(SUMMARY_PREFIX):
                return '[Compaction summary]'
            return f'System: {clip(part.content, _SHORT_EXCERPT_CHARS, terms)}'
        case InstructionDeltaPart():
            return f'System: {clip(part.render(), _SHORT_EXCERPT_CHARS, terms)}'
        case ToolReturnPart():
            if part.outcome == 'retried':
                # A retried call reads as a retry rather than a return: why a run changed course is what
                # makes it worth recalling. `model_response_str` renders a validation failure's serialized
                # errors the way the model was shown them, unwrapped because the `{"error": ...}` envelope
                # is a wire detail; `str(part.content)` would put a Python repr in the index.
                retry = part.model_response_str(wrap_if_error=False)
                return f'Retry [{part.tool_name}]: {clip(retry, _EXCERPT_CHARS, terms)}'
            # `tool_return_text` names each file instead of spelling out its bytes.
            return f'Tool [{part.tool_name}]: {clip(tool_return_text(part), _EXCERPT_CHARS, terms)}'
        case RetryFeedbackPart():
            # The same feedback for a response that answered no particular call. No tool name: it is
            # never bound to a call.
            return f'Retry: {clip(_retry_text(part.content), _EXCERPT_CHARS, terms)}'
        # TODO(v3): remove RetryPromptPart
        case RetryPromptPart():  # pyright: ignore[reportDeprecated]
            # The part the two above replaced. A stored history loads as one of them, but code that
            # builds one by hand and passes it as `message_history` still puts an instance here, and
            # this reads the history before the model translates it.
            return f'Retry [{part.tool_name}]: {clip(_retry_text(part.content), _EXCERPT_CHARS, terms)}'
        case CompactionPart():
            # An application's own summary is conversation content; `SummarizingCompaction`'s are dropped upstream.
            return f'Summary: {clip(part.content, _EXCERPT_CHARS, terms)}' if part.content else None
        case SpeechPart():
            # A realtime user turn: index the spoken transcript so speech is searchable, not the
            # raw audio. `transcript` is optional, so an audio-only part contributes no text.
            if not part.transcript:
                return None
            return f'Speech [{part.speaker}]: {clip(part.transcript, _EXCERPT_CHARS, terms)}'
        case CapabilityPart():
            return _format_capability_part(part, terms=terms)
        # Tool-list, capability, instruction and effort bookkeeping rather than conversation. Listed explicitly so the
        # fallback below stays a real runtime branch.
        case (
            ToolAvailabilityDeltaPart()
            | CapabilityLoadDeltaPart()
            | InstructionBaselinePart()
            | ModelSettingsDeltaPart()
        ):
            return None
        # Reachable only on a pydantic-ai release newer than this one: a part type newer than this code contributes
        # nothing. `assert_never` under `TYPE_CHECKING` keeps the match exhaustive for pyright without crashing at
        # runtime on a newer pydantic-ai.
        case _:  # pragma: no cover
            if TYPE_CHECKING:
                assert_never(part)
            return None


def _format_capability_part(part: CapabilityPart, *, terms: frozenset[str] | None) -> str | None:
    """What a capability told the model, such as a sub-agent's report or a background tool's result, or `None`.

    Labeled with its kind, never as the user's, the way `SummarizingCompaction` shows it. A request-scoped part applied
    to its own request only, so it isn't conversation.
    """
    if part.scope == 'request':
        return None
    return f'[{part.kind}]: {clip(capability_part_text(part), _EXCERPT_CHARS, terms)}'


def _format_message(message: ModelMessage, *, terms: frozenset[str] | None) -> str:
    """Render one message to text.

    `terms=None` keeps content full-length so the BM25 index matches terms past the display
    cutoff; otherwise each part is clipped for display around the first of `terms` it contains.
    """
    lines: list[str] = []

    if isinstance(message, ModelRequest):
        for part in message.parts:
            line = _format_request_part(part, terms=terms)
            if line is not None:
                lines.append(line)
    else:
        for part in message.parts:
            if isinstance(part, TextPart):
                lines.append(f'Assistant: {clip(part.content, _EXCERPT_CHARS, terms)}')
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                lines.append(f'Tool Call [{part.tool_name}]: {clip(args, _SHORT_EXCERPT_CHARS, terms)}')
            elif isinstance(part, SpeechPart) and part.transcript:
                # `SpeechPart` is in the response union too: a realtime assistant turn. Index the
                # transcript so spoken replies are searchable. Any other response part (thinking,
                # file, native-tool bookkeeping) is deliberately skipped rather than misrendered,
                # so a new upstream part degrades to "not indexed" instead of crashing the run.
                lines.append(f'Speech [{part.speaker}]: {clip(part.transcript, _EXCERPT_CHARS, terms)}')

    return '\n'.join(lines)


@dataclass
class _RunSection:
    """One run's slice of the corpus.

    `messages` and `index_lines` are index-aligned. `first_index` is the position of
    `messages[0]` in the run's full record, so excerpt numbering matches a `run_id` search
    even when history inherited from earlier runs was left out of this section.
    """

    label: str
    messages: list[ModelMessage]
    index_lines: list[str]
    first_index: int

    def display_line(self, index: int, terms: frozenset[str]) -> str:
        """The numbered excerpt line for `messages[index]`.

        The `[index]` prefix lives only here, not in the indexed rendering: BM25 would
        otherwise treat each ordinal as a rare, high-IDF token, letting a numeric query match
        a message by position instead of content. A message with no renderable text (e.g. a
        thinking-only response) gets a placeholder so excerpts do not show bare numbered blanks.
        """
        line = _format_message(self.messages[index], terms=terms)
        return f'[{self.first_index + index}] {line or "[no text content]"}'


def _adds_instruction_change(message: ModelMessage, earlier: ModelMessage) -> bool:
    """Whether `message` carries an instruction change that `earlier`, its hash-equal copy, lacks."""
    return isinstance(message, ModelRequest) and any(
        isinstance(part, InstructionDeltaPart) and part not in earlier.parts for part in message.parts
    )


def _inherited_prefix(messages: list[ModelMessage], hashes: list[str], seen: dict[str, ModelMessage]) -> int:
    """How many leading `messages` are in `seen` without an instruction change the copy there lacks."""
    inherited = 0
    while (
        inherited < len(hashes)
        and hashes[inherited] in seen
        and not _adds_instruction_change(messages[inherited], seen[hashes[inherited]])
    ):
        inherited += 1
    return inherited


def _newest_sections(loaded: list[tuple[str, list[ModelMessage], int]], limit: int) -> tuple[list[_RunSection], int]:
    """Sections holding the newest `limit` of the `(label, messages, first_index)` runs, and the total message count.

    Runs arrive oldest first, so this walks them backwards and cuts the oldest run that crosses
    `limit`, keeping its numbering. Only kept messages are rendered for the index.
    """
    total = sum(len(messages) for _, messages, _ in loaded)
    sections: list[_RunSection] = []
    for label, messages, first_index in reversed(loaded):
        if limit == 0:
            break
        skipped = max(0, len(messages) - limit)
        kept = messages[skipped:]
        limit -= len(kept)
        sections.append(
            _RunSection(
                label=label,
                messages=kept,
                index_lines=[_format_message(message, terms=None) for message in kept],
                first_index=first_index + skipped,
            )
        )
    sections.reverse()
    return sections, total


def _construction_stacklevel() -> int:
    """The `warn_default_changed` stacklevel that reaches the code constructing the toolset.

    Called from `ConversationSearchToolset.__init__`. Constructing through a subscripted
    generic (`ConversationSearchToolset[Deps](...)`) adds a `typing` frame between `__init__`
    and that code, which a fixed level would blame instead.
    """
    # 1 is `warn_default_changed`, 2 is `__init__`, 3 is whatever called `__init__`.
    level = 3
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    while caller is not None and caller.f_code.co_filename == typing.__file__:
        level += 1
        caller = caller.f_back
    del frame, caller
    return level


class ConversationSearchToolset(FunctionToolset[AgentDepsT]):
    """A single `search_conversation_history` tool over a `HistorySource`."""

    def __init__(
        self,
        source: HistorySource,
        *,
        tool_id: str,
        max_matches: int,
        context_lines: int,
        bm25_k1: float,
        bm25_b: float,
        scope: SearchScope | None = None,
        max_scanned_messages: int = DEFAULT_MAX_SCANNED_MESSAGES,
    ) -> None:
        # `None` means the caller did not choose. It resolves to `conversation` and warns once
        # here, for the same reason `ConversationSearch` does: this default was `all`, and the
        # change is invisible to a store-wide caller who keeps working with a narrower corpus.
        # `ConversationSearch` always passes a resolved scope, so it never double-warns.
        if scope is None:
            warn_default_changed(
                owner='ConversationSearchToolset',
                option='scope',
                old='all',
                new='conversation',
                impact=SCOPE_DEFAULT_CHANGE_IMPACT,
                stacklevel=_construction_stacklevel(),
            )
            scope = 'conversation'
        super().__init__(id=tool_id)
        if max_matches < 0:
            raise ValueError(f'max_matches must be non-negative, got {max_matches!r}.')
        if context_lines < 0:
            raise ValueError(f'context_lines must be non-negative, got {context_lines!r}.')
        # A negative k1 can zero `bm25_score`'s denominator (k1=-1, b=0, tf=1); `nan`
        # and `inf` slip past an ordering check (`nan < 0` is false) and turn every
        # score into `nan`, which the `score > 0` filter then drops as a non-match.
        # The `bm25_b` range check already excludes both.
        if not math.isfinite(bm25_k1) or bm25_k1 < 0:
            raise ValueError(f'bm25_k1 must be a non-negative finite number, got {bm25_k1!r}.')
        if not 0.0 <= bm25_b <= 1.0:
            raise ValueError(f'bm25_b must be between 0.0 and 1.0, got {bm25_b!r}.')
        if max_scanned_messages < 1:
            raise ValueError(f'max_scanned_messages must be positive, got {max_scanned_messages!r}.')
        self._source = source
        self._max_scanned_messages = max_scanned_messages
        self._max_matches = max_matches
        self._context_lines = context_lines
        self._params = Bm25Params(k1=bm25_k1, b=bm25_b)
        self._scope: SearchScope = scope
        description = SEARCH_HISTORY_DESCRIPTION
        if scope == 'conversation':
            description += _CONVERSATION_SCOPE_NOTE
        self.add_function(
            self.search_conversation_history,
            name='search_conversation_history',
            description=description,
        )

    async def search_conversation_history(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
        run_id: str | None = None,
    ) -> str:
        """Search persisted conversation history using BM25 ranking.

        Args:
            ctx: The active run context.
            query: Text to search for. Multi-word queries search each word
                independently -- rare terms score higher than common ones.
                At most 1,000 characters and 32 unique terms.
            run_id: Search only this run's history. Use it to resolve a run
                another tool or a compaction receipt referenced.
        """
        _check_query(query)
        return await self._search(ctx, query, run_id)

    async def _search(self, ctx: RunContext[AgentDepsT], query: str, run_id: str | None) -> str:
        conversation_id: str | None = None
        if self._scope == 'conversation':
            # Fail closed. Filtering on `conversation_id is None` would match every
            # unlabelled run in the store, which is the exposure the scope exists to
            # prevent, so an unlabelled run searches nothing rather than everything.
            if ctx.conversation_id is None:
                return (
                    'Conversation-scoped search needs a conversation id, and this run has none. '
                    'Pass `conversation_id=` to `Agent.run(...)` so the search can tell this '
                    "conversation's history from other conversations in the same store."
                )
            conversation_id = ctx.conversation_id

        sections, total_messages = await self._load_sections(run_id, conversation_id)
        if run_id is not None and not sections:
            # Same answer whether the run is absent or out of scope: a distinct
            # message would tell the model which run ids exist in other conversations.
            return f"No persisted history for run '{run_id}'."
        if total_messages == 0:
            if conversation_id is not None:
                # Naming the scope keeps the "pair with `StepPersistence`" advice below from
                # misdirecting a caller who already did: under conversation scope an empty
                # corpus usually means the earlier runs carry a different conversation id.
                # It reveals nothing about whether other conversations exist in the store.
                return (
                    'No persisted history in this conversation yet. Search covers only runs '
                    "sharing this run's conversation id, which comes from an explicit "
                    '`conversation_id=` argument or is inherited from `message_history`; runs '
                    'started with neither belong to other conversations.'
                )
            return (
                'No persisted conversation history to search yet. History accumulates '
                'as runs persist their steps (pair this capability with `StepPersistence` '
                'on a shared store).'
            )

        return self._render_matches(query, sections, total_messages)

    def _render_matches(self, query: str, sections: list[_RunSection], total_messages: int) -> str:
        # Rank on the untruncated rendering so terms past the display cutoff stay
        # findable, then show excerpts clipped around the query terms. Documents are
        # index-aligned across the flattened sections; context windows stay within one section.
        flat_index_lines: list[str] = []
        locations: list[tuple[int, int]] = []
        for section_index, section in enumerate(sections):
            for line_index, line in enumerate(section.index_lines):
                flat_index_lines.append(line)
                locations.append((section_index, line_index))

        ranked = bm25_rank(query, flat_index_lines, self._params)
        echoed = _echo_query(query)
        searched = f'{total_messages} persisted messages'
        if len(flat_index_lines) < total_messages:
            # Say the corpus was cut, so the model never reads "no matches" as "never said".
            searched = (
                f'the newest {len(flat_index_lines)} of {total_messages} persisted messages; '
                'older messages were not searched'
            )
        if not ranked:
            return f"No matches for '{echoed}' in {searched}."

        terms = frozenset(tokenize(query))
        results: list[str] = []
        shown: set[tuple[int, int]] = set()
        for doc_index, score in ranked:
            if len(results) == self._max_matches:
                break
            section_index, line_index = locations[doc_index]
            if (section_index, line_index) in shown:
                continue
            section = sections[section_index]
            start = max(0, line_index - self._context_lines)
            end = min(len(section.messages), line_index + self._context_lines + 1)
            # Record the whole emitted window so a neighbouring match whose context
            # overlaps this one is skipped instead of repeating the same lines.
            shown.update((section_index, index) for index in range(start, end))
            excerpt = '\n'.join(section.display_line(index, terms) for index in range(start, end))
            results.append(f'[score: {score:.1f} | {section.label}]\n{excerpt}')

        header = f"Found {len(results)} match(es) for '{echoed}' in {searched}:\n\n"
        return header + '\n\n---\n\n'.join(results)

    async def _load_sections(self, run_id: str | None, conversation_id: str | None) -> tuple[list[_RunSection], int]:
        """The sections to score, newest `max_scanned_messages` only, and how many messages are in scope in total."""
        runs = await self._source.list_runs()
        if conversation_id is not None:
            runs = [run for run in runs if run.conversation_id == conversation_id]
        if run_id is not None:
            runs = [run for run in runs if run.run_id == run_id]
        loaded: list[tuple[str, list[ModelMessage], int]] = []
        # A run started with `message_history` persists that history again as the head of its
        # own record, so without this each inherited message would be indexed once per later
        # run, crowding `max_matches` with copies. Runs arrive in `started_at` order, so only
        # the leading messages an earlier run already holds are left out; everything from the
        # first new message on stays, repeats included. The hash ignores instruction changes, so
        # an inherited request that core gave a new one in this run counts as new, and is indexed
        # whole: its other parts repeat once. `seen` keeps each hash's latest copy, so a run that
        # branches from before a change, followed by one continuing past it, re-indexes from the
        # changed request onward once more.
        seen: dict[str, ModelMessage] = {}
        for run in runs:
            messages = await self._source.run_history(run_id=run.run_id)
            hashes = [message_hash(message) for message in messages]
            inherited = _inherited_prefix(messages, hashes, seen)
            seen.update(zip(hashes, messages))
            messages = messages[inherited:]
            if not messages:
                continue
            label = f'run: {run.run_id}'
            if run.conversation_id is not None:
                label += f' | conversation: {run.conversation_id}'
            loaded.append((label, messages, inherited))
        return _newest_sections(loaded, self._max_scanned_messages)
