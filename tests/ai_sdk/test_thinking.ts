import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isReasoningUIPart, isTextUIPart } from 'ai';
import { SERVER_URL, TestChat, awaitRoundTrip } from './helpers.ts';

describe('thinking', () => {
  it('receives reasoning and text parts', async () => {
    const chat = new TestChat(`${SERVER_URL}/api/chat/thinking`);
    const trip = awaitRoundTrip(chat);
    chat.sendMessage({ text: 'What is the answer?' });
    await trip.done;
    assert.equal(trip.error(), null, 'request should succeed');

    const assistant = chat.messages.find((m) => m.role === 'assistant');
    assert.ok(assistant, 'should have an assistant message');

    const reasoningParts = assistant.parts.filter(isReasoningUIPart);
    assert.ok(reasoningParts.length > 0, 'should have at least one reasoning part');

    const reasoningText = reasoningParts.map((p) => p.text).join('');
    assert.ok(reasoningText.includes('think'), `expected thinking content, got: ${reasoningText}`);

    const textParts = assistant.parts.filter(isTextUIPart);
    assert.ok(textParts.length > 0, 'should have at least one text part');

    const fullText = textParts.map((p) => p.text).join('');
    assert.ok(fullText.includes('42'), `expected answer text, got: ${fullText}`);
  });
});
