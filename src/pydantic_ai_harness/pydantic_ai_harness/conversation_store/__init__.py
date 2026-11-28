"""SQLite, PostgreSQL and MongoDB backends of [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore].

`MongoConversationStore` needs the `mongodb` extra (`pip install pydantic-ai-harness[mongodb]`); it is
imported lazily so the rest of this module stays usable without `pymongo` installed.
`PostgresConversationStore` imports no driver (it takes a caller-owned asyncpg-compatible pool), so it is
exported eagerly.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.conversation_store._postgres import PostgresConversationStore
from pydantic_ai_harness.conversation_store._sqlite import SqliteConversationStore

if TYPE_CHECKING:
    from pydantic_ai_harness.conversation_store._mongo import MongoConversationStore

__all__ = [
    'MongoConversationStore',
    'PostgresConversationStore',
    'SqliteConversationStore',
]


def __getattr__(name: str) -> object:
    if name == 'MongoConversationStore':
        from pydantic_ai_harness.conversation_store._mongo import MongoConversationStore

        return MongoConversationStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
