---
title: Conversation Stores
description: "Save Pydantic AI conversations between runs in SQLite, PostgreSQL or MongoDB, with each message stored once and conflicting saves caught."
---

# Conversation Stores

SQLite, PostgreSQL and MongoDB backends of Pydantic AI's
[`ConversationStore`][pydantic_ai.conversation_store.ConversationStore], for saving a chat and loading it again
later. [Storing conversations](../persistence.md#using-a-conversation-store) covers how a conversation store works;
this page covers what each backend stores and how. This is supporting infrastructure, not a capability: nothing on
this page mounts on `Agent(capabilities=[...])`.

[Step Persistence](step-persistence.md) is a different thing: it records what each run did so a run can be
recovered, and isn't meant for storing sessions.

!!! note "Import path"
    Import the stores from their submodule -- there is no top-level `pydantic_ai_harness` re-export:

    ```python
    from pydantic_ai_harness.conversation_store import (
        MongoConversationStore,
        PostgresConversationStore,
        SqliteConversationStore,
    )
    ```

## Storing conversations

`SqliteConversationStore`, `PostgresConversationStore`, and `MongoConversationStore` store a whole
[`Conversation`][pydantic_ai.conversation.Conversation] between runs. They implement Pydantic AI's
[`ConversationStore`][pydantic_ai.conversation_store.ConversationStore], which keeps a conversation as a log of
messages and stores each one once: continuing a conversation appends only its new messages, and a
[`fork`][pydantic_ai.conversation_store.ConversationStore.fork] shares the history it keeps rather than copying it.

```python
import asyncio
import sqlite3

from pydantic_ai import Agent, Conversation
from pydantic_ai_harness.conversation_store import SqliteConversationStore

store = SqliteConversationStore(connection=sqlite3.connect(':memory:', check_same_thread=False))
agent = Agent('{{ openai:medium }}')


async def chat(conversation_id: str, prompt: str) -> str:
    conversation = await store.load(conversation_id) or Conversation(conversation_id=conversation_id)
    result = await agent.run(prompt, conversation=conversation)
    await store.save(result.conversation)
    return result.output


async def main():
    await chat('support-42', 'What is the capital of France?')
    await chat('support-42', 'And of Germany?')


asyncio.run(main())
```

A save is checked against the conversation's [`revision`][pydantic_ai.conversation.Conversation.revision], which
`load` sets and a run carries through to its result. When someone else saved the conversation in between, or a
conversation with that ID already exists, `save` raises
[`ConversationConflict`][pydantic_ai.conversation_store.ConversationConflict] instead of overwriting it. `save` advances
the revision of the conversation you pass it, so keep that conversation in a variable to save again or continue from.

A conversation's `scope` is kept on its record only. Entries, messages and branches have none, and a message that
conversations in two scopes hold is stored once. So `scope` filters what the store returns, and isolating tenants
stays with your application: row-level security on the entries or messages tables would make a second tenant's
`load` fail.

Each message's JSON is stored once, keyed by its SHA-256, however many conversations hold it. Binary and text parts
at or above `media_threshold_bytes` (64 KiB by default) are externalized like step snapshots' (see
[Persisting media](step-persistence.md#persisting-media)). The constructors take the same connection arguments as the matching
[step store](step-persistence.md#backends):

- `SqliteConversationStore(database= or connection=)` -- tables `conversation_records`, `conversation_entries`,
  `conversation_messages`, and `conversation_branches`. A save writes its new entries and the record in one
  transaction. Deleting a conversation also deletes the entries and messages no other conversation's history or
  branches use, including the branches it left.
- `PostgresConversationStore(pool, table='step_persistence')` -- tables `{table}_conv_records`,
  `{table}_conv_entries`, `{table}_conv_messages`, and `{table}_conv_branches`, which don't clash with `PostgresStepStore`'s, so the two
  can share a prefix. Saves and deletes work as for SQLite. A save that continues from an entry a
  concurrent delete is removing raises `ConversationConflict`; a message the delete is removing is written again.
- `MongoConversationStore(client= or db_url=, database=...)` -- collections `conversation_records`,
  `conversation_entries`, and `conversation_messages`. The store uses no transactions, so deleting a conversation
  deletes only its record: without one, it can't tell an entry no conversation uses from one a concurrent save or
  fork is about to point at. Its entries and messages stay.

All three support [`update`][pydantic_ai.conversation_store.ConversationStore.update]. A conversation's title and
metadata are stored beside its record, so a save leaves what `update` set even when the save read the conversation
first.

Externalized media is never deleted, as for step snapshots.

## Related

- [Storing conversations](../persistence.md#using-a-conversation-store) -- how a conversation store works, in core.
- [Step Persistence](step-persistence.md) -- recording and recovering runs, and
  [moving sessions](step-persistence.md#moving-sessions-to-a-conversationstore) from it to a conversation store.
- [Media Externalization](media.md) -- where large binary and text parts go.
