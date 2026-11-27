# Peers

Let an agent discover and message other agent sessions without controlling their runs. Use `Peers` when separately owned sessions need to exchange context, rather than delegate work to a child agent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/peers/)

## In-process example

`InMemoryPeerNetwork` connects sessions in one application. Pass it to each agent's `Peers` capability: the agent joins the network under its `name` when it is built, and receives in a new `LocalPeerInbox`.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Peers
from pydantic_ai_harness.peers import InMemoryPeerNetwork

network = InMemoryPeerNetwork()
alice = Agent('anthropic:claude-opus-5-5', name='alice', capabilities=[Peers(network)])
bob = Agent('anthropic:claude-opus-5-5', name='bob', capabilities=[Peers(network)])

alice.run_sync('Use list_peers to find bob, then send_message with your findings about the migration.')
result = bob.run_sync('Review the migration findings sent by alice.')
print(result.output)
```

Handles are unique within a network, and an agent with no `name` raises `UserError` when it is built. Binding `Peers(network)` to the same agent again, as a run capability or from a capability function, keeps its handle and inbox; passing another inbox raises `UserError`. Pass `Peers(network, handle='alice')` to join under another handle, or `Peers(network, inbox=LocalPeerInbox(path=...))` to keep incoming messages in a file.

The model gets `list_peers(query='')` and `send_message(to, text, reply_to=None)`. The list shows other reachable sessions, including their handle, ID, status, `waiting_for`, `status_since`, `accepts_messages`, pending-message count, title, workspace, and branch. The optional query filters by handle, title, workspace, or branch; the list is capped at `max_listed` (default `30`). Send to a listed handle or ID. The send result says whether the message was delivered to a running peer, queued for a peer with no active run, or refused, and includes the message ID for accepted messages. `reply_to` can reference an earlier message ID; it does not wait for a reply.

`Peers` reads new messages before and after each model request. Messages waiting when a run starts are included in its first model request. A message that arrives during a run can cause an additional model request before that same run ends. `InMemoryPeerNetwork` never starts a run: a message to a peer with no active run waits until the application starts the peer's next turn. Whether a message wakes an idle recipient is the host network's policy. In sequential mode, `InputGuardrail` checks the user prompt before queued peer messages are added, so a blocked run sends no request and the messages stay in the inbox. With `parallel=True`, the request runs alongside the guard, so peer messages can enter history and leave the inbox even if the guard later blocks. Messages that have not entered conversation history remain in the inbox across runs; an agent may complete its run before receiving a reply.

## Transport and persistence

To set the `workspace` or `title` that `list_peers` shows, or to join a peer that is not one agent, call `join()` yourself. It returns this peer's `PeerConnection`; pass it to `Peers` together with the same inbox. Joining without an inbox is send-only: incoming messages are refused.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Peers
from pydantic_ai_harness.peers import InMemoryPeerNetwork, LocalPeerInbox

network = InMemoryPeerNetwork()
inbox = LocalPeerInbox()
connection = network.join(handle='alice', inbox=inbox, workspace='/work/api', title='Fix auth')
alice = Agent('anthropic:claude-opus-5-5', capabilities=[Peers(connection, inbox=inbox)])
```

`InMemoryPeerNetwork` is an in-process example, not a cross-process transport. Its members stay joined, and their handles taken, for the network's lifetime; to retire a session, set its inbox's `accepting` to `False`. For sessions across processes or machines, implement `PeerConnection.peers()` and `PeerConnection.send()` in the host. The host owns peer identity, routing, authentication, and any user-visible notifications; the model supplies the recipient, text, and optional `reply_to` ID. `Peers` also passes a `key` built from the sending tool call (`run_id:run_step:tool_call_id`); a retried send with the same key must not deliver a second message. `InMemoryPeerNetwork` derives the message ID from it, so `LocalPeerInbox` drops a repeat, both while the first copy is pending and after it was read: the inbox remembers the IDs of its last `max_pending` acknowledged messages. `PeerConnection.peers()` returns other peers as `PeerInfo` objects. `PeerConnection.send()` builds a `PeerMessage` whose `sender` is a `PeerSender` supplied by the host from its own authenticated identity, never from the model's tool arguments. It delivers the message and returns a `PeerDelivery` with status `delivered`, `queued`, or `refused`; it raises `LookupError` for a peer that disappeared.

`LocalPeerInbox(path=...)` saves pending messages to an atomically replaced JSON file at a host-chosen per-recipient path. Use a private directory for that path. An inbox without `path` keeps messages only in memory. Use `LocalPeerInbox(accepting=False)` to refuse new messages, or change its `accepting` flag later; pending messages are retained. A file-backed inbox does not provide a network or coordinate simultaneous writers across processes. Pass the same inbox to the recipient's transport endpoint and `Peers(connection, inbox=...)`; without an inbox bound to `Peers`, the capability provides sending tools but does not read a recipient inbox.

To hold messages in host storage instead, implement `PeerInbox`. `Peers` calls `take(run_id)` before and after each model request: the first call claims the inbox for that run, and each call returns the messages the run has not taken yet. It calls `acknowledge(message_id)` once a message has entered conversation history, and `release(run_id)` when the run ends; messages taken but not acknowledged stay held for the next run. While one run holds the inbox, `take` returns nothing to another. `LocalPeerInbox` is the in-process implementation.

### Wire format

`TypeAdapter(PeerMessage)` is the harness-neutral message format, so an adapter for an agent outside Pydantic AI can exchange messages with a host network:

```json
{
  "id": "msg-1",
  "sender": {"id": "peer-alice", "handle": "alice", "workspace": "/work/api"},
  "recipient": "peer-bob",
  "text": "The migration is done.",
  "sent_at": "2026-11-13T12:00:00Z",
  "reply_to": null
}
```

`text` is limited to `MAX_MESSAGE_BYTES` UTF-8 bytes. The inbox file of `LocalPeerInbox` is not a cross-process contract: other processes reach a recipient through the host's transport, which admits each message into the recipient's inbox.

Incoming text is delivered as a `PeerMessagePart`, a [capability part](https://pydantic.dev/docs/ai/capabilities/custom/#capability-parts) with no authority. The model sees it on the user channel as `<peer_message sender="alice" id="msg-1">...</peer_message>`, with the sender handle the host supplied and a note that the peer wrote it, not the user or the application, so it never reads as the user speaking. The text is sent as written: text in it that imitates a frame gets a note naming the peer instead of being escaped. It is not an instruction from the receiving user and cannot grant approval. A host that delivers peer messages itself must send each one as its own message part: merged into a user prompt, peer text loses its provenance and reads as the user's own words. A peer message that reaches a running agent can still steer it into any tool that does not require approval, so if peers can reach the agent, require approval for tools with side effects. Find peer messages in history by type, or with `read_peer_message(part)`, which reads the `PeerMessage` a part carries, not its displayed text. It also reads a message an earlier version delivered as a `UserPromptPart` marked by its metadata.

## Limits and scope

`MAX_MESSAGE_BYTES` (16384 UTF-8 bytes) and `MAX_PENDING` (50) are public transport contract defaults. A host transport can use them to reject oversized wire messages and bound incoming queues before messages reach `LocalPeerInbox.receive()`. `Peers` also defaults to `max_messages_per_run=10` and `max_listed=30`. Only delivered or queued messages count toward `max_messages_per_run`. Sending the same text to the same peer twice in one run, or exceeding the limit, tells the model without sending the message. All limits must be positive. See the [peer API reference](https://pydantic.dev/docs/ai/harness/peers/#api-reference) for the host-facing transport types and helpers.

Unlike [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/), `Peers` does not create a child run, forward dependencies or usage, or wait for another agent's result. Its tools use core's tool-call spans. Delivered peer content is part of the conversation and follows core's content-capture setting. Delivery is observable through `EnqueuedMessagesEvent`; the host can inspect `LocalPeerInbox.pending()` for outstanding messages. The capability adds no per-message or persistence spans, leaving transport admission and inbox telemetry to the host instead of duplicating message records.

`Peers` does not support durable execution yet: the engine records neither inbox reads nor sends, so a replay could deliver different messages. Combining it with any [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) capability raises `UserError` when the agent is constructed.
