"""Step-event persistence: append-only event log, continuable snapshots, tool-effect ledger.

`MongoStepStore` needs the `mongodb` extra (`pip install
pydantic-ai-harness[mongodb]`); it is imported lazily so the rest of this
module stays usable without `pymongo` installed.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.step_persistence._capability import StepPersistence
from pydantic_ai_harness.step_persistence._helpers import (
    annotate_tool_effect,
    continue_run,
    fork_run,
    is_provider_valid,
)
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
    from pydantic_ai_harness.step_persistence._mongo import MongoStepStore

__all__ = [
    'ContinuableSnapshot',
    'EventKind',
    'FileStepStore',
    'InMemoryStepStore',
    'MongoStepStore',
    'RunRecord',
    'SnapshotState',
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
    if name == 'MongoStepStore':
        from pydantic_ai_harness.step_persistence._mongo import MongoStepStore

        return MongoStepStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
