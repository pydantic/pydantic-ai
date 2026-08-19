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

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import assert_never

from pydantic_ai_harness._warn import warn_default_changed
from pydantic_ai_harness.conversation_search._source import SUMMARY_PREFIX, HistorySource

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
    'so a caller relying on the old default silently stops seeing other conversations rather than erroring.'
)
"""Why the `scope` default change matters, for the deprecation warning both public entry points emit."""

_TOKENIZE_RE = re.compile(r'\w+')
"""Tokenizer regex: word runs (unicode letters, digits, underscore)."""


@dataclass(frozen=True)
class _Bm25Params:
    """BM25 tuning supplied by the owning capability."""

    k1: float
    b: float


def _tokenize(text: str) -> list[str]:
    return [match.group().lower() for match in _TOKENIZE_RE.finditer(text)]


def _compute_idf(term: str, doc_tokens: list[list[str]]) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
    n = len(doc_tokens)
    df = sum(1 for tokens in doc_tokens if term in set(tokens))
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf_scores: dict[str, float],
    avgdl: float,
    params: _Bm25Params,
) -> float:
    dl = len(doc_tokens)
    if dl == 0 or avgdl == 0:
        return 0.0

    tf_map: dict[str, int] = {}
    for token in doc_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = idf_scores.get(qt, 0.0)
        numerator = tf * (params.k1 + 1.0)
        denominator = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
        score += idf * numerator / denominator

    return score


def _bm25_rank(query: str, documents: list[str], params: _Bm25Params) -> list[tuple[int, float]]:
    """Rank documents by BM25 relevance, descending; only scores greater than zero."""
    query_tokens = _tokenize(query)
    if not query_tokens or not documents:
        return []

    doc_tokens = [_tokenize(doc) for doc in documents]
    avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)

    unique_query_tokens = list(dict.fromkeys(query_tokens))
    idf_scores = {qt: _compute_idf(qt, doc_tokens) for qt in unique_query_tokens}

    results: list[tuple[int, float]] = []
    for index, tokens in enumerate(doc_tokens):
        score = _bm25_score(unique_query_tokens, tokens, idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _user_prompt_text(part: UserPromptPart) -> str:
    """Render only the textual content of a user prompt.

    `UserPromptPart.content` can interleave text with binary and URL parts
    (`BinaryContent`, `ImageUrl`, ...). Only `str` and `TextContent` carry
    searchable text; str()-ing the whole content would fold a `BinaryContent`
    part's byte representation into the BM25 corpus (a 70 KiB image renders as
    hundreds of thousands of escaped-byte characters). Mirrors the text
    extraction in `pydantic_ai_harness.compaction._shared`.
    """
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts)


def _format_request_part(part: ModelRequestPart, *, truncate: bool) -> str | None:
    """Render one request part to a searchable line, or `None` for non-content parts."""
    if isinstance(part, UserPromptPart):
        content = _user_prompt_text(part)
        if truncate and len(content) > 500:
            content = content[:500] + '...'
        return f'User: {content}'
    if isinstance(part, SystemPromptPart):
        content = part.content
        # Defensive for sources that do not filter compaction artifacts;
        # `SnapshotHistorySource` already excludes them from the corpus.
        if content.startswith(SUMMARY_PREFIX):
            return '[Compaction summary]'
        if truncate:
            content = content[:200]
        return f'System: {content}'
    if isinstance(part, ToolReturnPart):
        content = str(part.content)
        if truncate and len(content) > 500:
            content = content[:500] + '...'
        return f'Tool [{part.tool_name}]: {content}'
    if isinstance(part, RetryPromptPart):
        # A retry or validation-error prompt is worth recalling, so index it in full.
        return f'Retry [{part.tool_name}]: {part.content}'
    if isinstance(part, SpeechPart):
        # A realtime user turn: index the spoken transcript so speech is searchable, not the
        # raw audio. `transcript` is optional, so an audio-only part contributes no text.
        if not part.transcript:
            return None
        transcript = part.transcript
        if truncate and len(transcript) > 500:  # cap the display excerpt like `UserPromptPart`
            transcript = transcript[:500] + '...'
        return f'Speech [{part.speaker}]: {transcript}'
    # Tool-list bookkeeping rather than conversation, so there is no line to contribute.
    # Redundant against the union as it stands today, but kept explicit so the fallthrough
    # below stays a real branch at runtime rather than dead code.
    if isinstance(part, ToolAvailabilityDeltaPart):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    # A part pydantic-ai added after this was written. Indexing it would mean guessing which
    # of its fields read as conversation, and a search index is not worth failing a run over,
    # so it contributes nothing. `assert_never` under `TYPE_CHECKING` keeps the chain
    # exhaustive at type-check time -- a new upstream part fails `make typecheck` -- without
    # crashing a user who upgrades pydantic-ai ahead of us.
    if TYPE_CHECKING:
        assert_never(part)
    return None  # pragma: no cover - reachable only on a pydantic-ai release newer than this one


def _format_message(message: ModelMessage, *, truncate: bool) -> str:
    """Render one message to text.

    `truncate=False` keeps content full-length so the BM25 index matches terms past the
    display cutoff; `truncate=True` produces the shorter excerpt shown to the model.
    """
    lines: list[str] = []

    if isinstance(message, ModelRequest):
        for part in message.parts:
            line = _format_request_part(part, truncate=truncate)
            if line is not None:
                lines.append(line)
    else:
        for part in message.parts:
            if isinstance(part, TextPart):
                content = part.content
                if truncate and len(content) > 500:
                    content = content[:500] + '...'
                lines.append(f'Assistant: {content}')
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                if truncate and len(args) > 200:
                    args = args[:200] + '...'
                lines.append(f'Tool Call [{part.tool_name}]: {args}')
            elif isinstance(part, SpeechPart) and part.transcript:
                # `SpeechPart` is in the response union too: a realtime assistant turn. Index the
                # transcript so spoken replies are searchable. Any other response part (thinking,
                # file, native-tool bookkeeping) is deliberately skipped rather than misrendered,
                # so a new upstream part degrades to "not indexed" instead of crashing the run.
                transcript = part.transcript
                if truncate and len(transcript) > 500:  # cap like the `TextPart` branch above
                    transcript = transcript[:500] + '...'
                lines.append(f'Speech [{part.speaker}]: {transcript}')

    return '\n'.join(lines)


def _format_messages(messages: list[ModelMessage], *, truncate: bool) -> list[str]:
    """Render messages to text lines, index-aligned with `messages`."""
    return [_format_message(message, truncate=truncate) for message in messages]


def _display_lines(messages: list[ModelMessage]) -> list[str]:
    """Render the numbered excerpt lines shown to the model.

    The `[index]` prefix lives only here, not in the indexed rendering: BM25
    would otherwise treat each ordinal as a rare, high-IDF token, letting a
    numeric query match a message by position instead of content. A message
    with no renderable text (e.g. a thinking-only response) gets a placeholder
    so excerpts do not show bare numbered blanks.
    """
    lines = _format_messages(messages, truncate=True)
    return [f'[{index}] {line or "[no text content]"}' for index, line in enumerate(lines)]


@dataclass
class _RunSection:
    """One run's slice of the corpus, index-aligned per run."""

    label: str
    index_lines: list[str]
    display_lines: list[str]


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
                stacklevel=3,
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
        total_messages = sum(len(section.display_lines) for section in sections)
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
        # findable, then show the truncated rendering. Documents are index-aligned
        # across the flattened sections; context windows stay within one section.
        flat_index_lines: list[str] = []
        locations: list[tuple[int, int]] = []
        for section_idx, section in enumerate(sections):
            for line_idx, line in enumerate(section.index_lines):
                flat_index_lines.append(line)
                locations.append((section_idx, line_idx))

        ranked = _bm25_rank(query, flat_index_lines, self._params)
        if not ranked:
            return f"No matches for '{query}' in {total_messages} persisted messages."

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
            end = min(len(section.display_lines), line_idx + self._context_lines + 1)
            # Record the whole emitted window so a neighbouring match whose context
            # overlaps this one is skipped instead of repeating the same lines.
            shown.update((section_idx, idx) for idx in range(start, end))
            excerpt = '\n'.join(section.display_lines[start:end])
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
        for run in runs:
            messages = await self._source.run_history(run_id=run.run_id)
            if not messages:
                continue
            label = f'run: {run.run_id}'
            if run.conversation_id is not None:
                label += f' | conversation: {run.conversation_id}'
            sections.append(
                _RunSection(
                    label=label,
                    index_lines=_format_messages(messages, truncate=False),
                    display_lines=_display_lines(messages),
                )
            )
        return sections
