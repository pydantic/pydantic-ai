---
title: Peers
description: "Let an agent discover and message other agent sessions without starting or controlling their runs."
---

# Peers

Let an agent discover and message other agent sessions without starting or controlling their runs. Use [Peers][pydantic_ai_harness.peers.Peers] when separately owned sessions need to exchange context, rather than delegate work to a child agent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/peers/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## In-process example

[InMemoryPeerNetwork][pydantic_ai_harness.peers.InMemoryPeerNetwork] connects sessions in one application. Give each recipient a [PeerInbox][pydantic_ai_harness.peers.PeerInbox], and pass that same inbox both to `join()` and to its [Peers][pydantic_ai_harness.Peers] capability. Joining without an inbox is send-only: incoming messages are refused.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Peers
from pydantic_ai_harness.peers import InMemoryPeerNetwork, PeerInbox

network = InMemoryPeerNetwork()
alice_inbox = PeerInbox()
bob_inbox = PeerInbox()
alice_view = network.join(handle='alice', inbox=alice_inbox)
bob_view = network.join(handle='bob', inbox=bob_inbox)

alice = Agent('{{ anthropic:large }}', capabilities=[Peers(network=alice_view, inbox=alice_inbox)])
bob = Agent('{{ anthropic:large }}', capabilities=[Peers(network=bob_view, inbox=bob_inbox)])

alice.run_sync('Use list_peers to find bob, then send_message with your findings about the migration.')
result = bob.run_sync('Review the migration findings sent by alice.')
print(result.output)
```

The model gets `list_peers(query='')` and `send_message(to, text, reply_to=None)`. The list shows other reachable sessions, including their handle, ID, status, `waiting_for`, `status_since`, `accepts_messages`, pending-message count, title, workspace, and branch. The optional query filters by handle, title, workspace, or branch; the list is capped at `max_listed` (default `30`). Send to a listed handle or ID. The send result says whether the message was delivered to a running peer, queued for an idle peer, or refused, and includes the message ID for accepted messages. `reply_to` can reference an earlier message ID; it does not wait for a reply.

Sending to an idle peer queues a message but does not start that peer's run. The application starts the next turn. Messages waiting when a run starts are included in its first model request. A message delivered during a run can cause an additional model request before that same run ends, but does not start a new run. In sequential mode, `InputGuardrail` checks the user prompt first, so blocked input is not restarted with queued peer messages. With `parallel=True`, the model request runs concurrently with the guard; peer messages may enter history even if the guard later blocks, following the usual acknowledgement behavior. Messages that have not entered conversation history remain in the inbox across runs; an agent may complete its run before receiving a reply.

## Transport and persistence

[InMemoryPeerNetwork][pydantic_ai_harness.peers.InMemoryPeerNetwork] is an in-process example, not a cross-process transport. For sessions across processes or machines, implement [PeerNetwork.peers()][pydantic_ai_harness.peers.PeerNetwork.peers] and [PeerNetwork.send()][pydantic_ai_harness.peers.PeerNetwork.send] in the host. The host owns peer identity, routing, authentication, and any user-visible notifications; the model supplies the recipient, text, and optional `reply_to` ID. [PeerNetwork.peers()][pydantic_ai_harness.peers.PeerNetwork.peers] returns other peers as [PeerInfo][pydantic_ai_harness.peers.PeerInfo] objects. [PeerNetwork.send()][pydantic_ai_harness.peers.PeerNetwork.send] builds a [PeerMessage][pydantic_ai_harness.peers.PeerMessage] whose `sender` is a [PeerSender][pydantic_ai_harness.peers.PeerSender] supplied by the host from its own authenticated identity, never from the model's tool arguments. It delivers the message and returns a [PeerDelivery][pydantic_ai_harness.peers.PeerDelivery] with status `delivered`, `queued`, or `refused`; it raises `LookupError` for a peer that disappeared.

`PeerInbox(path=...)` saves pending messages to an atomically replaced JSON file at a host-chosen per-recipient path. Use a private directory for that path. An inbox without `path` keeps messages only in memory. A file-backed inbox does not provide a network or coordinate simultaneous writers across processes. Pass the same inbox to the recipient's transport endpoint and `Peers(inbox=...)`; without an inbox bound to [Peers][pydantic_ai_harness.peers.Peers], the capability provides sending tools but does not attach a recipient inbox to the run.

Incoming text is framed as untrusted peer data on the user channel, with typed provenance metadata. It is not an instruction from the receiving user and cannot grant approval. [read_peer_message(part)][pydantic_ai_harness.peers.read_peer_message] reads a [PeerMessage][pydantic_ai_harness.peers.PeerMessage] from a tagged message part by its metadata, not by matching the displayed text. Native message serialization preserves that metadata; adapters that discard `TextContent.metadata` lose the provenance marker on round-trip.

## Limits and scope

[MAX_MESSAGE_BYTES][pydantic_ai_harness.peers.MAX_MESSAGE_BYTES] (16384 UTF-8 bytes) and [MAX_PENDING][pydantic_ai_harness.peers.MAX_PENDING] (50) are public transport contract defaults. A host transport can use them to reject oversized wire messages and bound incoming queues before messages reach [PeerInbox.receive()][pydantic_ai_harness.peers.PeerInbox.receive]. [Peers][pydantic_ai_harness.peers.Peers] also defaults to `max_messages_per_run=10` and `max_listed=30`. Only delivered or queued messages count toward `max_messages_per_run`. Sending the same text to the same peer twice in one run, or exceeding the limit, tells the model without sending the message. All limits must be positive. See the [peer API reference](#api-reference) for the host-facing transport types and helpers.

Unlike [Subagents](subagents.md), [Peers][pydantic_ai_harness.peers.Peers] does not create a child run, forward dependencies or usage, or wait for another agent's result. Its tools run as ordinary function tools, so core's tool-call spans cover them; the capability adds no dedicated telemetry spans.

## API reference

::: pydantic_ai_harness.Peers

::: pydantic_ai_harness.peers.InMemoryPeerNetwork

::: pydantic_ai_harness.peers.PeerNetwork

::: pydantic_ai_harness.peers.PeerInfo

::: pydantic_ai_harness.peers.PeerDelivery

::: pydantic_ai_harness.peers.PeerInbox

::: pydantic_ai_harness.peers.PeerSender

::: pydantic_ai_harness.peers.PeerMessage

::: pydantic_ai_harness.peers.read_peer_message

::: pydantic_ai_harness.peers.DeliveryStatus

::: pydantic_ai_harness.peers.PeerStatus

::: pydantic_ai_harness.peers.MAX_MESSAGE_BYTES

::: pydantic_ai_harness.peers.MAX_PENDING
