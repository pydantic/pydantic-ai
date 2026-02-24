import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isToolUIPart, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { TestChat, awaitRoundTrip } from './helpers.ts';

function getLatestApprovalId(chat: TestChat): string {
  const assistant = [...chat.messages].reverse().find((m) => m.role === 'assistant');
  assert.ok(assistant, 'should have an assistant message');

  const toolParts = assistant.parts.filter(isToolUIPart);
  const approvalPart = [...toolParts].reverse().find((p) => p.state === 'approval-requested');
  assert.ok(approvalPart, 'no tool part with state=approval-requested');
  return approvalPart.approval.id;
}

async function sendAndGetApprovalId(chat: TestChat): Promise<string> {
  const trip = awaitRoundTrip(chat);
  chat.sendMessage({ text: 'Delete test.txt' });
  await trip.done;
  assert.equal(trip.error(), null, 'initial request should succeed');
  return getLatestApprovalId(chat);
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

  it('retries with new tool call after denial', async () => {
    const chat = createApprovalChat();
    const approvalId = await sendAndGetApprovalId(chat);

    const resubmit = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: approvalId, approved: false });
    await resubmit.done;
    assert.equal(resubmit.error(), null, 'resubmit after denial should succeed');

    // Server retries after first denial — should get a new approval-requested
    const retryApprovalId = getLatestApprovalId(chat);
    assert.notEqual(retryApprovalId, approvalId, 'retry should produce a new approval id');
  });

  it('completes after deny then approve', async () => {
    const chat = createApprovalChat();
    const firstId = await sendAndGetApprovalId(chat);

    // Deny the first tool call — server retries
    const retry = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: firstId, approved: false });
    await retry.done;
    assert.equal(retry.error(), null, 'retry after denial should succeed');

    // Approve the retried tool call
    const secondId = getLatestApprovalId(chat);
    const approve = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: secondId, approved: true });
    await approve.done;
    assert.equal(approve.error(), null, 'approve after retry should succeed');
  });

  it('completes after deny then deny', async () => {
    const chat = createApprovalChat();
    const firstId = await sendAndGetApprovalId(chat);

    // Deny the first tool call — server retries
    const retry = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: firstId, approved: false });
    await retry.done;
    assert.equal(retry.error(), null, 'retry after first denial should succeed');

    // Deny the retried tool call — server gives up and returns text
    const secondId = getLatestApprovalId(chat);
    const final = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: secondId, approved: false });
    await final.done;
    assert.equal(final.error(), null, 'second denial should complete with text');
  });

  it('completes after denial with reason', async () => {
    const chat = createApprovalChat();
    const approvalId = await sendAndGetApprovalId(chat);

    // Deny with reason — server retries
    const retry = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({
      id: approvalId,
      approved: false,
      reason: 'Not allowed by policy',
    });
    await retry.done;
    assert.equal(retry.error(), null, 'resubmit after denial with reason should succeed');

    // Deny again — server gives up
    const secondId = getLatestApprovalId(chat);
    const final = awaitRoundTrip(chat);
    await chat.addToolApprovalResponse({ id: secondId, approved: false });
    await final.done;
    assert.equal(final.error(), null, 'second denial should complete with text');
  });
});
