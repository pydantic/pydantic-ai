"""Conversation search capability: BM25 recall over persisted step history."""

from pydantic_ai_harness.conversation_search._capability import ConversationSearch
from pydantic_ai_harness.conversation_search._source import (
    HistorySource,
    SnapshotHistorySource,
    SnapshotStore,
)
from pydantic_ai_harness.conversation_search._toolset import ConversationSearchToolset, SearchScope

__all__ = [
    'ConversationSearch',
    'ConversationSearchToolset',
    'HistorySource',
    'SearchScope',
    'SnapshotHistorySource',
    'SnapshotStore',
]
