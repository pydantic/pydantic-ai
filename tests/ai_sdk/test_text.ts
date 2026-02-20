import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isTextUIPart } from 'ai';
import { TestChat, awaitRoundTrip } from './helpers.ts';

describe('text streaming', () => {
  it('receives streamed text parts', async () => {
    const chat = new TestChat();
    const trip = awaitRoundTrip(chat);
    chat.sendMessage({ text: 'Say hello' });
    await trip.done;
    assert.equal(trip.error(), null, 'request should succeed');

    const assistant = chat.messages.find((m) => m.role === 'assistant');
    assert.ok(assistant, 'should have an assistant message');

    const textParts = assistant.parts.filter(isTextUIPart);
    assert.ok(textParts.length > 0, 'should have at least one text part');

    const fullText = textParts.map((p) => p.text).join('');
    assert.ok(fullText.includes('Hello, world!'), `expected greeting, got: ${fullText}`);
  });
});
