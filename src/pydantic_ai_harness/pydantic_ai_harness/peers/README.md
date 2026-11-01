# Peers

Let an agent discover and message other agent sessions without controlling their runs. Use `Peers` when separately owned sessions need to exchange context, rather than delegate work to a child agent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/peers/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## In-process example

`InMemoryPeerNetwork` connects sessions in one application. Give each recipient a `LocalPeerInbox`, and pass that same inbox both to `join()` and to its `Peers` capability. Joining without an inbox is send-only: incoming messages are refused.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Peers
from pydantic_ai_harness.peers import InMemoryPeerNetwork, LocalPeerInbox

network = InMemoryPeerNetwork()
alice_inbox = LocalPeerInbox()
bob_inbox = LocalPeerInbox()
alice_view = network.join(handle='alice', inbox=alice_inbox)
bob_view = network.join(handle='bob', inbox=bob_inbox)

alice = Agent('anthropic:claude-opus-5-5', capabilities=[Peers(network=alice_view, inbox=alice_inbox)])
bob = Agent('anthropic:claude-opus-5-5', capabilities=[Peers(network=bob_view, inbox=bob_inbox)])

alice.run_sync('Use list_peers to find bob, then send_message with your findings about the migration.')
result = bob.run_sync('Review the migration findings sent by alice.')
print(result.output)
```

The model gets `list_peers(query='')` and `send_message(to, text, reply_to=None)`. The list shows other reachable sessions, including their handle, ID, status, `waiting_for`, `status_since`, `accepts_messages`, pending-message count, title, workspace, and branch. The optional query filters by handle, title, workspace, or branch; the list is capped at `max_listed` (default `30`). Send to a listed handle or ID. The send result says whether the message was delivered to a running peer, queued for a peer with no active run, or refused, and includes the message ID for accepted messages. `reply_to` can reference an earlier message ID; it does not wait for a reply.

`Peers` reads new messages before and after each model request. Messages waiting when a run starts are included in its first model request. A message that arrives during a run can cause an additional model request before that same run ends. `InMemoryPeerNetwork` never starts a run: a message to a peer with no active run waits until the application starts the peer's next turn. Whether a message wakes an idle recipient is the host network's policy. In sequential mode, `InputGuardrail` checks the user prompt before queued peer messages are added, so a blocked run sends no request and the messages stay in the inbox. With `parallel=True`, the request runs alongside the guard, so peer messages can enter history and leave the inbox even if the guard later blocks. Messages that have not entered conversation history remain in the inbox across runs; an agent may complete its run before receiving a reply.

## Transport and persistence

`InMemoryPeerNetwork` is an in-process example, not a cross-process transport. For sessions across processes or machines, implement `PeerNetwork.peers()` and `PeerNetwork.send()` in the host. The host owns peer identity, routing, authentication, and any user-visible notifications; the model supplies the recipient, text, and optional `reply_to` ID. `Peers` also passes a `key` built from the sending tool call (`run_id:run_step:tool_call_id`); a retried send with the same key must not deliver a second message. `InMemoryPeerNetwork` derives the message ID from it, so `LocalPeerInbox` drops a repeat, both while the first copy is pending and after it was read: the inbox remembers the IDs of its last `max_pending` acknowledged messages. `PeerNetwork.peers()` returns other peers as `PeerInfo` objects. `PeerNetwork.send()` builds a `PeerMessage` whose `sender` is a `PeerSender` supplied by the host from its own authenticated identity, never from the model's tool arguments. It delivers the message and returns a `PeerDelivery` with status `delivered`, `queued`, or `refused`; it raises `LookupError` for a peer that disappeared.

`LocalPeerInbox(path=...)` saves pending messages to an atomically replaced JSON file at a host-chosen per-recipient path. Use a private directory for that path. An inbox without `path` keeps messages only in memory. Use `LocalPeerInbox(accepting=False)` to refuse new messages, or change its `accepting` flag later; pending messages are retained. A file-backed inbox does not provide a network or coordinate simultaneous writers across processes. Pass the same inbox to the recipient's transport endpoint and `Peers(inbox=...)`; without an inbox bound to `Peers`, the capability provides sending tools but does not read a recipient inbox.

To hold messages in host storage instead, implement `PeerInbox`. `Peers` calls `take(run_id)` before and after each model request: the first call claims the inbox for that run, and each call returns the messages the run has not taken yet. It calls `acknowledge(message_id)` once a message has entered conversation history, and `release(run_id)` when the run ends; messages taken but not acknowledged stay held for the next run. While one run holds the inbox, `take` returns nothing to another. `LocalPeerInbox` is the in-process implementation.

Incoming text is framed as untrusted peer data on the user channel, with typed provenance metadata. It is not an instruction from the receiving user and cannot grant approval. A host that delivers peer messages itself must send each one as its own message part: merged into a user prompt, peer text loses its provenance and reads as the user's own words. A peer message that reaches a running agent can still steer it into any tool that does not require approval, so if peers can reach the agent, require approval for tools with side effects. `read_peer_message(part)` reads a `PeerMessage` from a tagged message part by its metadata, not by matching the displayed text. Native message serialization preserves that metadata; adapters that discard `TextContent.metadata` lose the provenance marker on round-trip.

## Limits and scope

`MAX_MESSAGE_BYTES` (16384 UTF-8 bytes) and `MAX_PENDING` (50) are public transport contract defaults. A host transport can use them to reject oversized wire messages and bound incoming queues before messages reach `LocalPeerInbox.receive()`. `Peers` also defaults to `max_messages_per_run=10` and `max_listed=30`. Only delivered or queued messages count toward `max_messages_per_run`. Sending the same text to the same peer twice in one run, or exceeding the limit, tells the model without sending the message. All limits must be positive. See the [peer API reference](https://pydantic.dev/docs/ai/harness/peers/#api-reference) for the host-facing transport types and helpers.

Unlike [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/), `Peers` does not create a child run, forward dependencies or usage, or wait for another agent's result. Its tools use core's tool-call spans. Delivered peer content is part of the conversation and follows core's content-capture setting. Delivery is observable through `EnqueuedMessagesEvent`; the host can inspect `LocalPeerInbox.pending()` for outstanding messages. The capability adds no per-message or persistence spans, leaving transport admission and inbox telemetry to the host instead of duplicating message records.

`Peers` does not support durable execution yet: the engine records neither inbox reads nor sends, so a replay could deliver different messages. Combining it with any [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) capability raises `UserError` when the agent is constructed.
