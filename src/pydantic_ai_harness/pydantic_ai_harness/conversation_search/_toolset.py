"""The `search_conversation_history` toolset with dependency-free BM25 ranking.

The ranking and rendering core is ported from VStorm's `pydantic-deepagents`
(`pydantic_deep/features/history_archive/toolset.py`). BM25 tuning is dependency-free
(`math` + `re`) and takes its parameters from the owning capability instead of module
constants, so a caller can tune ranking without editing the port. The corpus comes
from a `HistorySource` (one document per persisted message, across the runs selected
by the configured scope); ranking is global within that scope, while context windows
around a match stay within the match's run.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import typing
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from pydantic_core import ErrorDetails

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
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
from pydantic_ai_harness._message_text import tool_return_text, user_prompt_texts
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

_TOKENIZE_RE = re.compile(r'\w+')
"""Tokenizer regex: word runs (unicode letters, digits, underscore).

Scripts written without spaces (Chinese, Japanese, Thai) form one token per unbroken run,
so a query matches only the whole run, not a word inside it.
"""

_EXCERPT_CHARS = 500
"""Display cap for conversational text: prompts, replies, tool returns, retries, transcripts."""

_SHORT_EXCERPT_CHARS = 200
"""Display cap for system text and tool-call arguments."""


@dataclass(frozen=True)
class _Bm25Params:
    """BM25 tuning supplied by the owning capability."""

    k1: float
    b: float


def _tokenize(text: str) -> list[str]:
    return [match.group().lower() for match in _TOKENIZE_RE.finditer(text)]


def _compute_idf(term: str, term_counts: list[Counter[str]]) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
    n = len(term_counts)
    df = sum(1 for counts in term_counts if term in counts)
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_score(
    query_terms: list[str],
    counts: Counter[str],
    dl: int,
    idf_scores: dict[str, float],
    avgdl: float,
    params: _Bm25Params,
) -> float:
    if dl == 0 or avgdl == 0:
        return 0.0
    score = 0.0
    for term in query_terms:
        tf = counts[term]
        if tf == 0:
            continue
        numerator = tf * (params.k1 + 1.0)
        denominator = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
        score += idf_scores[term] * numerator / denominator
    return score


def _bm25_rank(query: str, documents: list[str], params: _Bm25Params) -> list[tuple[int, float]]:
    """Rank documents by BM25 relevance, descending; only scores greater than zero."""
    query_terms = list(dict.fromkeys(_tokenize(query)))
    if not query_terms or not documents:
        return []

    # Tokenize each document once; df, tf, and length all read the same counts.
    term_counts = [Counter(_tokenize(doc)) for doc in documents]
    lengths = [counts.total() for counts in term_counts]
    avgdl = sum(lengths) / len(lengths)
    idf_scores = {term: _compute_idf(term, term_counts) for term in query_terms}

    results: list[tuple[int, float]] = []
    for index, (counts, dl) in enumerate(zip(term_counts, lengths)):
        score = _bm25_score(query_terms, counts, dl, idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _clip(text: str, limit: int, terms: frozenset[str] | None) -> str:
    """Cut `text` to about `limit` characters for display; `terms=None` keeps it whole for the index.

    A head cut alone would report a match whose term the excerpt never shows, so when the
    first query term ends past `limit` the window is centred on that term instead. It shifts
    to keep a long term whole when it fits in `limit`, and ends with the text when the term
    is near its end, so the excerpt keeps the full `limit`.
    """
    if terms is None or len(text) <= limit:
        return text
    focus = next((match for match in _TOKENIZE_RE.finditer(text) if match.group().lower() in terms), None)
    if focus is None or focus.end() <= limit:
        return text[:limit] + '...'
    # Keep the whole term when it fits (a term longer than `limit // 2` would lose its tail to
    # plain centring), and pull the window back near the end so a late term keeps the full `limit`.
    start = min(max(focus.start() - limit // 2, focus.end() - limit), len(text) - limit)
    end = start + limit
    return ('...' if start else '') + text[start:end] + ('...' if end < len(text) else '')


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
    if isinstance(part, UserPromptPart):
        return f'User: {_clip(" ".join(user_prompt_texts(part)), _EXCERPT_CHARS, terms)}'
    if isinstance(part, SystemPromptPart):
        # Defensive for sources that do not filter compaction artifacts;
        # `SnapshotHistorySource` already excludes them from the corpus.
        if part.content.startswith(SUMMARY_PREFIX):
            return '[Compaction summary]'
        return f'System: {_clip(part.content, _SHORT_EXCERPT_CHARS, terms)}'
    if isinstance(part, InstructionDeltaPart):
        return f'System: {_clip(part.render(), _SHORT_EXCERPT_CHARS, terms)}'
    if isinstance(part, ToolReturnPart):
        if part.outcome == 'retried':
            # A retried call reads as a retry rather than a return: why a run changed course is what
            # makes it worth recalling. `model_response_str` renders a validation failure's serialized
            # errors the way the model was shown them, unwrapped because the `{"error": ...}` envelope
            # is a wire detail; `str(part.content)` would put a Python repr in the index.
            retry = part.model_response_str(wrap_if_error=False)
            return f'Retry [{part.tool_name}]: {_clip(retry, _EXCERPT_CHARS, terms)}'
        # `tool_return_text` names each file instead of spelling out its bytes.
        return f'Tool [{part.tool_name}]: {_clip(tool_return_text(part), _EXCERPT_CHARS, terms)}'
    if isinstance(part, RetryFeedbackPart):
        # The same feedback for a response that answered no particular call. No tool name: it is
        # never bound to a call.
        return f'Retry: {_clip(_retry_text(part.content), _EXCERPT_CHARS, terms)}'
    # TODO(v3): remove RetryPromptPart
    if isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
        # The part the two above replaced. A stored history loads as one of them, but code that
        # builds one by hand and passes it as `message_history` still puts an instance here, and
        # this reads the history before the model translates it.
        return f'Retry [{part.tool_name}]: {_clip(_retry_text(part.content), _EXCERPT_CHARS, terms)}'
    if isinstance(part, CompactionPart):
        # An application's own summary is conversation content; `SummarizingCompaction`'s are dropped upstream.
        if not part.content:
            return None
        return f'Summary: {_clip(part.content, _EXCERPT_CHARS, terms)}'
    if isinstance(part, SpeechPart):
        # A realtime user turn: index the spoken transcript so speech is searchable, not the
        # raw audio. `transcript` is optional, so an audio-only part contributes no text.
        if not part.transcript:
            return None
        return f'Speech [{part.speaker}]: {_clip(part.transcript, _EXCERPT_CHARS, terms)}'
    # Tool-list, capability, instruction and effort bookkeeping rather than conversation, so there is no line to
    # contribute. Redundant against the union as it stands today, but kept explicit so the fallthrough below stays a
    # real branch at runtime rather than dead code.
    if isinstance(
        part,
        (ToolAvailabilityDeltaPart, CapabilityLoadDeltaPart, InstructionBaselinePart, ModelSettingsDeltaPart),
    ):
        return None
    # A part pydantic-ai added after this was written. Indexing it would mean guessing which
    # of its fields read as conversation, and a search index is not worth failing a run over,
    # so it contributes nothing. `assert_never` under `TYPE_CHECKING` keeps the chain
    # exhaustive at type-check time -- a new upstream part fails `make typecheck` -- without
    # crashing a user who upgrades pydantic-ai ahead of us.
    if TYPE_CHECKING:
        assert_never(part)
    return None  # pragma: no cover - reachable only on a pydantic-ai release newer than this one


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
                lines.append(f'Assistant: {_clip(part.content, _EXCERPT_CHARS, terms)}')
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                lines.append(f'Tool Call [{part.tool_name}]: {_clip(args, _SHORT_EXCERPT_CHARS, terms)}')
            elif isinstance(part, SpeechPart) and part.transcript:
                # `SpeechPart` is in the response union too: a realtime assistant turn. Index the
                # transcript so spoken replies are searchable. Any other response part (thinking,
                # file, native-tool bookkeeping) is deliberately skipped rather than misrendered,
                # so a new upstream part degrades to "not indexed" instead of crashing the run.
                lines.append(f'Speech [{part.speaker}]: {_clip(part.transcript, _EXCERPT_CHARS, terms)}')

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
        # A negative k1 can zero `_bm25_score`'s denominator (k1=-1, b=0, tf=1); `nan`
        # and `inf` slip past an ordering check (`nan < 0` is false) and turn every
        # score into `nan`, which the `score > 0` filter then drops as a non-match.
        # The `bm25_b` range check already excludes both.
        if not math.isfinite(bm25_k1) or bm25_k1 < 0:
            raise ValueError(f'bm25_k1 must be a non-negative finite number, got {bm25_k1!r}.')
        if not 0.0 <= bm25_b <= 1.0:
            raise ValueError(f'bm25_b must be between 0.0 and 1.0, got {bm25_b!r}.')
        self._source = source
        self._max_matches = max_matches
        self._context_lines = context_lines
        self._params = _Bm25Params(k1=bm25_k1, b=bm25_b)
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
            run_id: Search only this run's history. Use it to resolve a run
                another tool or a compaction receipt referenced.
        """
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

        sections = await self._load_sections(run_id, conversation_id)
        if run_id is not None and not sections:
            # Same answer whether the run is absent or out of scope: a distinct
            # message would tell the model which run ids exist in other conversations.
            return f"No persisted history for run '{run_id}'."
        total_messages = sum(len(section.messages) for section in sections)
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

        # Rank on the untruncated rendering so terms past the display cutoff stay
        # findable, then show excerpts clipped around the query terms. Documents are
        # index-aligned across the flattened sections; context windows stay within one section.
        flat_index_lines: list[str] = []
        locations: list[tuple[int, int]] = []
        for section_idx, section in enumerate(sections):
            for line_idx, line in enumerate(section.index_lines):
                flat_index_lines.append(line)
                locations.append((section_idx, line_idx))

        ranked = _bm25_rank(query, flat_index_lines, self._params)
        if not ranked:
            return f"No matches for '{query}' in {total_messages} persisted messages."

        terms = frozenset(_tokenize(query))
        results: list[str] = []
        shown: set[tuple[int, int]] = set()
        for doc_idx, score in ranked:
            if len(results) == self._max_matches:
                break
            section_idx, line_idx = locations[doc_idx]
            if (section_idx, line_idx) in shown:
                continue
            section = sections[section_idx]
            start = max(0, line_idx - self._context_lines)
            end = min(len(section.messages), line_idx + self._context_lines + 1)
            # Record the whole emitted window so a neighbouring match whose context
            # overlaps this one is skipped instead of repeating the same lines.
            shown.update((section_idx, idx) for idx in range(start, end))
            excerpt = '\n'.join(section.display_line(idx, terms) for idx in range(start, end))
            results.append(f'[score: {score:.1f} | {section.label}]\n{excerpt}')

        header = f"Found {len(results)} match(es) for '{query}' in {total_messages} persisted messages:\n\n"
        return header + '\n\n---\n\n'.join(results)

    async def _load_sections(self, run_id: str | None, conversation_id: str | None) -> list[_RunSection]:
        runs = await self._source.list_runs()
        if conversation_id is not None:
            runs = [run for run in runs if run.conversation_id == conversation_id]
        if run_id is not None:
            runs = [run for run in runs if run.run_id == run_id]
        sections: list[_RunSection] = []
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
            inherited = 0
            while (
                inherited < len(hashes)
                and hashes[inherited] in seen
                and not _adds_instruction_change(messages[inherited], seen[hashes[inherited]])
            ):
                inherited += 1
            seen.update(zip(hashes, messages))
            messages = messages[inherited:]
            if not messages:
                continue
            label = f'run: {run.run_id}'
            if run.conversation_id is not None:
                label += f' | conversation: {run.conversation_id}'
            sections.append(
                _RunSection(
                    label=label,
                    messages=messages,
                    index_lines=[_format_message(message, terms=None) for message in messages],
                    first_index=inherited,
                )
            )
        return sections
