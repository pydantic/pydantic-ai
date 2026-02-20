/**
 * AI SDK client <-> Pydantic AI server integration test for tool approval.
 *
 * Exercises the full approval lifecycle: request, approve, deny, deny with reason.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  AbstractChat,
  DefaultChatTransport,
  isToolUIPart,
  lastAssistantMessageIsCompleteWithApprovalResponses,
  type ChatState,
  type ChatStatus,
  type UIMessage,
} from 'ai';

const SERVER_URL = process.env.SERVER_URL;
if (!SERVER_URL) {
  console.error('Set SERVER_URL environment variable');
  process.exit(2);
}

class SimpleChatState implements ChatState<UIMessage> {
  status: ChatStatus = 'ready';
  error: Error | undefined = undefined;
  messages: UIMessage[];

  constructor(messages: UIMessage[] = []) {
    this.messages = messages;
  }

  pushMessage(message: UIMessage) {
    this.messages = [...this.messages, message];
  }

  popMessage() {
    this.messages = this.messages.slice(0, -1);
  }

  replaceMessage(index: number, message: UIMessage) {
    this.messages = [
      ...this.messages.slice(0, index),
      message,
      ...this.messages.slice(index + 1),
    ];
  }

  snapshot<T>(thing: T): T {
    return structuredClone(thing);
  }
}

class TestChat extends AbstractChat<UIMessage> {
  constructor(url: string) {
    super({
      transport: new DefaultChatTransport({ api: `${url}/api/chat` }),
      state: new SimpleChatState(),
      sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    });
  }
}

function awaitRoundTrip(chat: TestChat) {
  let resolve: () => void;
  const promise = new Promise<void>((r) => { resolve = r; });
  let captured: Error | null = null;

  chat.onError = (err) => { captured = err; resolve(); };
  chat.onFinish = () => resolve();

  return {
    done: promise.then(() => waitForStatus(chat, ['ready', 'error'])),
    error() { return captured; },
  };
}

function waitForStatus(
  chat: TestChat,
  statuses: ChatStatus[],
  timeoutMs = 10_000,
): Promise<ChatStatus> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    (function poll() {
      if (statuses.includes(chat.status)) return resolve(chat.status);
      if (Date.now() - start > timeoutMs) return reject(new Error(`Timed out (status=${chat.status})`));
      setTimeout(poll, 50);
    })();
  });
}

async function sendAndGetApprovalId(chat: TestChat): Promise<string> {
  const trip = awaitRoundTrip(chat);
  chat.sendMessage({ text: 'Delete test.txt' });
  await trip.done;
  assert.equal(trip.error(), null, 'initial request should succeed');

  const assistantMsg = chat.messages.find((m) => m.role === 'assistant');
  assert.ok(assistantMsg, 'should have an assistant message');

  for (const part of assistantMsg.parts) {
    if (isToolUIPart(part) && part.state === 'approval-requested') {
      return part.approval.id;
    }
  }
  return assert.fail('no tool part with state=approval-requested');
}

describe('tool approval', () => {
  it('returns approval-requested state on initial request', async () => {
    const chat = new TestChat(SERVER_URL);
    await sendAndGetApprovalId(chat);
  });

  it('completes round-trip after approval', async () => {
    const chat = new TestChat(SERVER_URL);
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: approvalId, approved: true });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after approval should succeed');
  });

  it('completes round-trip after denial', async () => {
    const chat = new TestChat(SERVER_URL);
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: approvalId, approved: false });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after denial should succeed');
  });

  it('completes round-trip after denial with reason', async () => {
    const chat = new TestChat(SERVER_URL);
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({
      id: approvalId,
      approved: false,
      reason: 'Not allowed by policy',
    });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after denial with reason should succeed');
  });
});
