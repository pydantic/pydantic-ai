"""The `search_conversation_history` toolset with dependency-free BM25 ranking.

The ranking and rendering core is ported from VStorm's `pydantic-deepagents`
(`pydantic_deep/features/history_archive/toolset.py`). BM25 tuning is dependency-free
(`math` + `re`) and takes its parameters from the owning capability instead of module
constants, so a caller can tune ranking without editing the port. The corpus comes
from a `HistorySource` (one document per persisted message, across the runs selected
by the configured scope); ranking is global within that scope, while context windows
around a match stay within the match's run. Over a `SqliteStepStore` the corpus is read
from the FTS5 index in `_fts`, which finds the candidate messages; the scoring is the same.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import re
import sqlite3
import typing
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from pydantic import TypeAdapter
from pydantic_core import ErrorDetails

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
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
from pydantic_ai_harness.conversation_search._fts import Fts5Unavailable, IndexedMessage, IndexReader, SqliteSearchIndex
from pydantic_ai_harness.conversation_search._source import SUMMARY_PREFIX, HistorySource, message_hash
from pydantic_ai_harness.step_persistence import RunRecord

_logger = logging.getLogger(__name__)

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

_LAYOUT_ADAPTER: TypeAdapter[list[tuple[int, int, int]]] = TypeAdapter(list[tuple[int, int, int]])


@dataclass(frozen=True)
class _Bm25Params:
    """BM25 tuning supplied by the owning capability."""

    k1: float
    b: float


def _tokenize(text: str) -> list[str]:
    return [match.group().lower() for match in _TOKENIZE_RE.finditer(text)]


def _compute_idf(df: int, n: int) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
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


def _bm25_scores(
    query_terms: list[str], term_counts: list[Counter[str]], n: int, avgdl: float, params: _Bm25Params
) -> list[tuple[int, float]]:
    """Score documents against a corpus of `n` documents averaging `avgdl` tokens.

    `term_counts` must hold every document of the corpus that contains a query term, since
    document frequency is counted over it; documents holding none may be left out. Returns
    the indices into `term_counts` scoring above zero, best first, ties in input order.
    """
    idf_scores = {term: _compute_idf(sum(1 for counts in term_counts if term in counts), n) for term in query_terms}
    results: list[tuple[int, float]] = []
    for index, counts in enumerate(term_counts):
        score = _bm25_score(query_terms, counts, counts.total(), idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _bm25_rank(query_terms: list[str], documents: list[str], params: _Bm25Params) -> list[tuple[int, float]]:
    """Rank a whole, non-empty corpus by BM25 relevance, descending; only scores greater than zero."""
    # Tokenize each document once; df, tf, and length all read the same counts.
    term_counts = [Counter(_tokenize(doc)) for doc in documents]
    avgdl = sum(counts.total() for counts in term_counts) / len(term_counts)
    return _bm25_scores(query_terms, term_counts, len(term_counts), avgdl, params)


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


_Line = tuple[str, str, int]
"""One rendered line: its label prefix, its body, and the body's display limit."""


def _request_part_line(part: ModelRequestPart) -> _Line | None:
    """Render one request part to a searchable line, or `None` for non-content parts."""
    if isinstance(part, UserPromptPart):
        return 'User: ', ' '.join(user_prompt_texts(part)), _EXCERPT_CHARS
    if isinstance(part, SystemPromptPart):
        # Defensive for sources that do not filter compaction artifacts;
        # `SnapshotHistorySource` already excludes them from the corpus.
        if part.content.startswith(SUMMARY_PREFIX):
            return '[Compaction summary]', '', 0
        return 'System: ', part.content, _SHORT_EXCERPT_CHARS
    if isinstance(part, InstructionDeltaPart):
        return 'System: ', part.render(), _SHORT_EXCERPT_CHARS
    if isinstance(part, ToolReturnPart):
        if part.outcome == 'retried':
            # A retried call reads as a retry rather than a return: why a run changed course is what
            # makes it worth recalling. `model_response_str` renders a validation failure's serialized
            # errors the way the model was shown them, unwrapped because the `{"error": ...}` envelope
            # is a wire detail; `str(part.content)` would put a Python repr in the index.
            return f'Retry [{part.tool_name}]: ', part.model_response_str(wrap_if_error=False), _EXCERPT_CHARS
        # `tool_return_text` names each file instead of spelling out its bytes.
        return f'Tool [{part.tool_name}]: ', tool_return_text(part), _EXCERPT_CHARS
    if isinstance(part, RetryFeedbackPart):
        # The same feedback for a response that answered no particular call. No tool name: it is
        # never bound to a call.
        return 'Retry: ', _retry_text(part.content), _EXCERPT_CHARS
    # TODO(v3): remove RetryPromptPart
    if isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
        # The part the two above replaced. A stored history loads as one of them, but code that
        # builds one by hand and passes it as `message_history` still puts an instance here, and
        # this reads the history before the model translates it.
        return f'Retry [{part.tool_name}]: ', _retry_text(part.content), _EXCERPT_CHARS
    if isinstance(part, SpeechPart):
        # A realtime user turn: index the spoken transcript so speech is searchable, not the
        # raw audio. `transcript` is optional, so an audio-only part contributes no text.
        if not part.transcript:
            return None
        return f'Speech [{part.speaker}]: ', part.transcript, _EXCERPT_CHARS
    # Tool-list and instruction bookkeeping rather than conversation, so there is no line to contribute.
    # Redundant against the union as it stands today, but kept explicit so the fallthrough
    # below stays a real branch at runtime rather than dead code.
    if isinstance(part, (ToolAvailabilityDeltaPart, InstructionBaselinePart)):
        return None
    # A part pydantic-ai added after this was written. Indexing it would mean guessing which
    # of its fields read as conversation, and a search index is not worth failing a run over,
    # so it contributes nothing. `assert_never` under `TYPE_CHECKING` keeps the chain
    # exhaustive at type-check time -- a new upstream part fails `make typecheck` -- without
    # crashing a user who upgrades pydantic-ai ahead of us.
    if TYPE_CHECKING:
        assert_never(part)
    return None  # pragma: no cover - reachable only on a pydantic-ai release newer than this one


def _message_lines(message: ModelMessage) -> list[_Line]:
    """Render one message to its searchable lines, full length."""
    lines: list[_Line] = []
    if isinstance(message, ModelRequest):
        for part in message.parts:
            line = _request_part_line(part)
            if line is not None:
                lines.append(line)
    else:
        for part in message.parts:
            if isinstance(part, TextPart):
                lines.append(('Assistant: ', part.content, _EXCERPT_CHARS))
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                lines.append((f'Tool Call [{part.tool_name}]: ', args, _SHORT_EXCERPT_CHARS))
            elif isinstance(part, SpeechPart) and part.transcript:
                # `SpeechPart` is in the response union too: a realtime assistant turn. Index the
                # transcript so spoken replies are searchable. Any other response part (thinking,
                # file, native-tool bookkeeping) is deliberately skipped rather than misrendered,
                # so a new upstream part degrades to "not indexed" instead of crashing the run.
                lines.append((f'Speech [{part.speaker}]: ', part.transcript, _EXCERPT_CHARS))
    return lines


def _render(lines: list[_Line], terms: frozenset[str] | None) -> str:
    """Join rendered lines into text.

    `terms=None` keeps content full-length so the BM25 index matches terms past the display
    cutoff; otherwise each line is clipped for display around the first of `terms` it contains.
    """
    return '\n'.join(prefix + _clip(body, limit, terms) for prefix, body, limit in lines)


def _layout(lines: list[_Line]) -> str:
    """Where each line's prefix and body sit in the full-length text, as the index stores it."""
    return json.dumps([[len(prefix), len(body), limit] for prefix, body, limit in lines])


def _from_layout(text: str, layout: str) -> list[_Line]:
    """Split full-length text back into its lines, by the `_layout` stored beside it."""
    lines: list[_Line] = []
    start = 0
    for prefix_length, body_length, limit in _LAYOUT_ADAPTER.validate_json(layout):
        body_start = start + prefix_length
        lines.append((text[start:body_start], text[body_start : body_start + body_length], limit))
        start = body_start + body_length + 1
    return lines


def _display_line(index: int, text: str) -> str:
    """A numbered excerpt line.

    The `[index]` prefix lives only here, not in the indexed rendering: BM25 would otherwise
    treat each ordinal as a rare, high-IDF token, letting a numeric query match a message by
    position instead of content. A message with no renderable text (e.g. a thinking-only
    response) gets a placeholder so excerpts do not show bare numbered blanks.
    """
    return f'[{index}] {text or "[no text content]"}'


def _delta_keys(message: ModelMessage) -> str:
    """Keys of the instruction changes `message` carries, which `message_hash` leaves out.

    Built from the fields that define a change, not its `repr`: the index stores these keys, so
    they must match the ones a later pydantic-ai release computes for the same change.
    """
    if not isinstance(message, ModelRequest):
        return ''
    return ' '.join(
        hashlib.sha256(json.dumps([str(part.id), part.content, part.base_hash]).encode()).hexdigest()
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    )


def _inherited_counts(runs: list[list[tuple[str, str]]]) -> list[int]:
    """How many leading messages of each run an earlier run already holds, given `(hash, delta keys)` per message.

    A run started with `message_history` persists that history again as the head of its own
    record, so without this each inherited message would be indexed once per later run,
    crowding `max_matches` with copies. Runs arrive in `started_at` order, so only the leading
    messages an earlier run already holds are left out; everything from the first new message
    on stays, repeats included. The hash ignores instruction changes, so an inherited request
    that core gave a new one in this run counts as new, and is indexed whole: its other parts
    repeat once. `seen` keeps each hash's latest copy, so a run that branches from before a
    change, followed by one continuing past it, re-indexes from the changed request onward
    once more.
    """
    seen: dict[str, frozenset[str]] = {}
    counts: list[int] = []
    for keys in runs:
        inherited = 0
        while (
            inherited < len(keys)
            and keys[inherited][0] in seen
            and frozenset(keys[inherited][1].split()) <= seen[keys[inherited][0]]
        ):
            inherited += 1
        seen.update((message_hash, frozenset(deltas.split())) for message_hash, deltas in keys)
        counts.append(inherited)
    return counts


@dataclass
class _Section:
    """One run's slice of the corpus.

    `first_index` is the position of the section's first message in the run's full record, so
    excerpt numbering matches a `run_id` search even when history inherited from earlier runs
    was left out of this section. Context windows stay inside the section, so they never reach
    into inherited history either.
    """

    run_id: str
    label: str
    first_index: int
    length: int


def _section(run: RunRecord, *, inherited: int, length: int) -> _Section:
    label = f'run: {run.run_id}'
    if run.conversation_id is not None:
        label += f' | conversation: {run.conversation_id}'
    return _Section(run_id=run.run_id, label=label, first_index=inherited, length=length - inherited)


class _Corpus(ABC):
    """The in-scope messages one search ranks, inherited history left out."""

    def __init__(self, sections: list[_Section]) -> None:
        self.sections = sections

    @abstractmethod
    def rank(self, terms: list[str]) -> list[tuple[int, int, float]]:
        """`(section, line, score)` of each message scoring above zero, best first, ties in corpus order."""

    @abstractmethod
    def display(self, section: int, start: int, end: int, terms: frozenset[str]) -> list[str]:
        """Numbered excerpt lines for a section's lines `start` up to `end`, clipped around `terms`."""


class _ScanCorpus(_Corpus):
    """Every in-scope message, loaded from the source and rendered in memory."""

    def __init__(self, sections: list[_Section], lines: list[list[list[_Line]]], params: _Bm25Params) -> None:
        super().__init__(sections)
        self._lines = lines
        self._params = params

    def rank(self, terms: list[str]) -> list[tuple[int, int, float]]:
        # Rank on the untruncated rendering so terms past the display cutoff stay findable.
        documents: list[str] = []
        locations: list[tuple[int, int]] = []
        for section_idx, section_lines in enumerate(self._lines):
            for line_idx, lines in enumerate(section_lines):
                documents.append(_render(lines, None))
                locations.append((section_idx, line_idx))
        return [(*locations[doc_idx], score) for doc_idx, score in _bm25_rank(terms, documents, self._params)]

    def display(self, section: int, start: int, end: int, terms: frozenset[str]) -> list[str]:
        first = self.sections[section].first_index
        return [_display_line(first + idx, _render(self._lines[section][idx], terms)) for idx in range(start, end)]


class _IndexCorpus(_Corpus):
    """The in-scope messages as the FTS5 index holds them, all read in one transaction."""

    def __init__(self, reader: IndexReader, sections: list[_Section], params: _Bm25Params) -> None:
        super().__init__(sections)
        self._reader = reader
        self._params = params

    @classmethod
    def load(cls, reader: IndexReader, runs: list[RunRecord], params: _Bm25Params) -> _IndexCorpus:
        keys = reader.message_keys([run.run_id for run in runs])
        run_keys = [keys[run.run_id] for run in runs]
        sections = [
            _section(run, inherited=inherited, length=len(message_keys))
            for run, message_keys, inherited in zip(runs, run_keys, _inherited_counts(run_keys))
            if inherited < len(message_keys)
        ]
        return cls(reader, sections, params)

    def rank(self, terms: list[str]) -> list[tuple[int, int, float]]:
        # FTS5 only finds the candidates. Scoring them here with the statistics of these
        # sections alone gives the scan's scores, and keeps other conversations out of them.
        spans = {section.run_id: section.first_index for section in self.sections}
        order = {section.run_id: idx for idx, section in enumerate(self.sections)}
        candidates = sorted(
            (order[run_id], position - spans[run_id], text)
            for run_id, position, text in self._reader.candidates(terms, spans)
        )
        n = sum(section.length for section in self.sections)
        avgdl = self._reader.total_length(spans) / n
        term_counts = [Counter(words.split(' ')) for _, _, words in candidates]
        return [
            (candidates[idx][0], candidates[idx][1], score)
            for idx, score in _bm25_scores(terms, term_counts, n, avgdl, self._params)
        ]

    def display(self, section: int, start: int, end: int, terms: frozenset[str]) -> list[str]:
        first = self.sections[section].first_index
        rows = self._reader.window(self.sections[section].run_id, first + start, first + end)
        return [
            _display_line(first + start + idx, _render(_from_layout(text, layout), terms))
            for idx, (text, layout) in enumerate(rows)
        ]


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
        self._index = SqliteSearchIndex.for_source(source)
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

        runs = await self._scoped_runs(run_id, conversation_id)
        if self._index is not None:
            try:
                return await self._search_index(self._index, runs, query, run_id, conversation_id)
            except (Fts5Unavailable, sqlite3.DatabaseError) as exc:
                if isinstance(exc, Fts5Unavailable):
                    self._index = None
                _logger.warning('Conversation search index unavailable, scanning persisted history instead: %s', exc)
                ctx.tracer.start_span(
                    'conversation_search index fallback',
                    attributes={'conversation_search.fallback_reason': type(exc).__name__},
                ).end()
        return self._answer(await self._scan(runs), query, run_id, conversation_id)

    async def _scoped_runs(self, run_id: str | None, conversation_id: str | None) -> list[RunRecord]:
        runs = await self._source.list_runs()
        if conversation_id is not None:
            runs = [run for run in runs if run.conversation_id == conversation_id]
        if run_id is not None:
            runs = [run for run in runs if run.run_id == run_id]
        return runs

    async def _scan(self, runs: list[RunRecord]) -> _ScanCorpus:
        histories = [await self._source.run_history(run_id=run.run_id) for run in runs]
        keys = [[(message_hash(message), _delta_keys(message)) for message in history] for history in histories]
        sections: list[_Section] = []
        lines: list[list[list[_Line]]] = []
        for run, history, inherited in zip(runs, histories, _inherited_counts(keys)):
            if inherited < len(history):
                sections.append(_section(run, inherited=inherited, length=len(history)))
                lines.append([_message_lines(message) for message in history[inherited:]])
        return _ScanCorpus(sections, lines, self._params)

    async def _search_index(
        self,
        index: SqliteSearchIndex,
        runs: list[RunRecord],
        query: str,
        run_id: str | None,
        conversation_id: str | None,
    ) -> str:
        await index.refresh([run.run_id for run in runs], self._indexed_messages)
        return await index.read(
            lambda reader: self._answer(_IndexCorpus.load(reader, runs, self._params), query, run_id, conversation_id)
        )

    async def _indexed_messages(self, run_id: str) -> list[IndexedMessage]:
        indexed: list[IndexedMessage] = []
        for message in await self._source.run_history(run_id=run_id):
            lines = _message_lines(message)
            text = _render(lines, None)
            words = _tokenize(text)
            indexed.append(
                IndexedMessage(
                    hash=message_hash(message),
                    deltas=_delta_keys(message),
                    words=' '.join(words),
                    length=len(words),
                    text=text,
                    layout=_layout(lines),
                )
            )
        return indexed

    def _answer(self, corpus: _Corpus, query: str, run_id: str | None, conversation_id: str | None) -> str:
        sections = corpus.sections
        if run_id is not None and not sections:
            # Same answer whether the run is absent or out of scope: a distinct
            # message would tell the model which run ids exist in other conversations.
            return f"No persisted history for run '{run_id}'."
        total_messages = sum(section.length for section in sections)
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

        query_terms = list(dict.fromkeys(_tokenize(query)))
        ranked = corpus.rank(query_terms) if query_terms else []
        if not ranked:
            return f"No matches for '{query}' in {total_messages} persisted messages."

        # Show excerpts clipped around the query terms; context windows stay within one section.
        terms = frozenset(query_terms)
        results: list[str] = []
        shown: set[tuple[int, int]] = set()
        for section_idx, line_idx, score in ranked:
            if len(results) == self._max_matches:
                break
            if (section_idx, line_idx) in shown:
                continue
            section = sections[section_idx]
            start = max(0, line_idx - self._context_lines)
            end = min(section.length, line_idx + self._context_lines + 1)
            # Record the whole emitted window so a neighbouring match whose context
            # overlaps this one is skipped instead of repeating the same lines.
            shown.update((section_idx, idx) for idx in range(start, end))
            excerpt = '\n'.join(corpus.display(section_idx, start, end, terms))
            results.append(f'[score: {score:.1f} | {section.label}]\n{excerpt}')

        header = f"Found {len(results)} match(es) for '{query}' in {total_messages} persisted messages:\n\n"
        return header + '\n\n---\n\n'.join(results)
