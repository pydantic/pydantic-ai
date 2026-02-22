import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isToolUIPart, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { TestChat, awaitRoundTrip } from './helpers.ts';

async function sendAndGetApprovalId(chat: TestChat): Promise<string> {
  const trip = awaitRoundTrip(chat);
  chat.sendMessage({ text: 'Delete test.txt' });
  await trip.done;
  assert.equal(trip.error(), null, 'initial request should succeed');

  const assistant = chat.messages.find((m) => m.role === 'assistant');
  assert.ok(assistant, 'should have an assistant message');

  const toolParts = assistant.parts.filter(isToolUIPart);
  const approvalPart = toolParts.find((p) => p.state === 'approval-requested');
  assert.ok(approvalPart, 'no tool part with state=approval-requested');
  return approvalPart.approval.id;
}

function createApprovalChat() {
  return new TestChat(lastAssistantMessageIsCompleteWithApprovalResponses);
}

describe('tool approval', () => {
  it('returns approval-requested state on initial request', async () => {
    const chat = createApprovalChat();
    await sendAndGetApprovalId(chat);
  });

  it('completes round-trip after approval', async () => {
    const chat = createApprovalChat();
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: approvalId, approved: true });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after approval should succeed');
  });

  it('completes round-trip after denial', async () => {
    const chat = createApprovalChat();
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: approvalId, approved: false });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after denial should succeed');
  });

  it('completes round-trip after denial with reason', async () => {
    const chat = createApprovalChat();
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
