import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isTextUIPart, isToolUIPart } from 'ai';
import { TestChat, awaitRoundTrip } from './helpers.ts';

describe('multiple tool calls', () => {
  it('executes multiple tools and returns text', async () => {
    const chat = new TestChat();
    const trip = awaitRoundTrip(chat);
    chat.sendMessage({ text: 'Weather and time please' });
    await trip.done;
    assert.equal(trip.error(), null, 'request should succeed');

    const assistant = chat.messages.find((m) => m.role === 'assistant');
    assert.ok(assistant, 'should have an assistant message');

    const toolParts = assistant.parts.filter(isToolUIPart);
    assert.equal(toolParts.length, 2, 'should have two tool parts');

    assert.ok(
      toolParts.every((p) => p.state === 'output-available'),
      'all tools should have output-available state',
    );

    const textParts = assistant.parts.filter(isTextUIPart);
    assert.ok(textParts.length > 0, 'should have at least one text part after tool execution');
  });
});
