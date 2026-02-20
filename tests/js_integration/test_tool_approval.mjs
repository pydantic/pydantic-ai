/**
 * AI SDK client ↔ Pydantic AI server integration test for tool approval.
 *
 * Uses node:test. Run with: node --test test_tool_approval.mjs <server-url>
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  AbstractChat,
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
} from 'ai';

const SERVER_URL = process.env.SERVER_URL;
if (!SERVER_URL) {
  console.error('Set SERVER_URL environment variable');
  process.exit(2);
}

class SimpleChatState {
  constructor(messages = []) {
    this.status = 'ready';
    this.error = undefined;
    this.messages = messages;
  }
  pushMessage(message) {
    this.messages = [...this.messages, message];
  }
  popMessage() {
    this.messages = this.messages.slice(0, -1);
  }
  replaceMessage(index, message) {
    this.messages = [...this.messages.slice(0, index), message, ...this.messages.slice(index + 1)];
  }
  snapshot(thing) {
    return structuredClone(thing);
  }
}

class TestChat extends AbstractChat {
  constructor(url) {
    super({
      transport: new DefaultChatTransport({ api: `${url}/api/chat` }),
      state: new SimpleChatState(),
      sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    });
  }
}

function awaitRoundTrip(chat) {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  let captured = null;

  chat.onError = (err) => { captured = err; resolve(); };
  chat.onFinish = () => resolve();

  return {
    done: promise.then(() => waitForStatus(chat, ['ready', 'error'])),
    error() { return captured; },
  };
}

function waitForStatus(chat, statuses, timeoutMs = 10_000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    (function poll() {
      if (statuses.includes(chat.status)) return resolve(chat.status);
      if (Date.now() - start > timeoutMs) return reject(new Error(`Timed out (status=${chat.status})`));
      setTimeout(poll, 50);
    })();
  });
}

describe('tool approval', () => {
  it('round-trips approval-requested and approval-responded states', async () => {
    const chat = new TestChat(SERVER_URL);

    const initial = awaitRoundTrip(chat);
    chat.sendMessage({ text: 'Delete test.txt' });
    await initial.done;
    assert.equal(initial.error(), null, 'initial request should succeed');

    const assistantMsg = chat.messages.find((m) => m.role === 'assistant');
    assert.ok(assistantMsg, 'should have an assistant message');

    const toolPart = assistantMsg.parts.find((p) => p.state === 'approval-requested');
    assert.ok(toolPart, 'should have a tool part with state=approval-requested');
    assert.ok(toolPart.approval?.id, 'tool part should have approval.id');

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: toolPart.approval.id, approved: true });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit with approval-responded should succeed');
  });
});
