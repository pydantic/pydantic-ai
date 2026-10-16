"""Step-event persistence: append-only event log, continuable snapshots, tool-effect ledger.

Also the Harness backends of [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore]:
`SqliteConversationStore`, `PostgresConversationStore` and `MongoConversationStore`.

`MongoStepStore` and `MongoConversationStore` need the `mongodb` extra (`pip install
pydantic-ai-harness[mongodb]`); it is imported lazily so the rest of this
module stays usable without `pymongo` installed. `PostgresStepStore` and
`PostgresConversationStore` import no driver (it takes a caller-owned asyncpg-compatible pool), so it is exported
eagerly.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.media import PostgresConnection, PostgresPool
from pydantic_ai_harness.step_persistence._capability import StepPersistence
from pydantic_ai_harness.step_persistence._conversation_postgres import PostgresConversationStore
from pydantic_ai_harness.step_persistence._conversation_sqlite import SqliteConversationStore
from pydantic_ai_harness.step_persistence._events import SnapshotSaved
from pydantic_ai_harness.step_persistence._helpers import (
    annotate_tool_effect,
    continue_run,
    fork_run,
    is_provider_valid,
)
from pydantic_ai_harness.step_persistence._postgres import PostgresStepStore
from pydantic_ai_harness.step_persistence._store import (
    FileStepStore,
    InMemoryStepStore,
    SqliteStepStore,
    StepStore,
)
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
    ToolEffectStatus,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.step_persistence._conversation_mongo import MongoConversationStore
    from pydantic_ai_harness.step_persistence._mongo import MongoStepStore

__all__ = [
    'ContinuableSnapshot',
    'EventKind',
    'FileStepStore',
    'InMemoryStepStore',
    'MongoConversationStore',
    'MongoStepStore',
    'PostgresConnection',
    'PostgresConversationStore',
    'PostgresPool',
    'PostgresStepStore',
    'RunRecord',
    'SnapshotSaved',
    'SnapshotState',
    'SqliteConversationStore',
    'SqliteStepStore',
    'StepEvent',
    'StepPersistence',
    'StepStore',
    'ToolEffectRecord',
    'ToolEffectStatus',
    'annotate_tool_effect',
    'continue_run',
    'fork_run',
    'is_provider_valid',
]


def __getattr__(name: str) -> object:
    if name == 'MongoConversationStore':
        from pydantic_ai_harness.step_persistence._conversation_mongo import MongoConversationStore

        return MongoConversationStore
    if name == 'MongoStepStore':
        from pydantic_ai_harness.step_persistence._mongo import MongoStepStore

        return MongoStepStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
