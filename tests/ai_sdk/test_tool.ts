import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isTextUIPart, isToolUIPart } from 'ai';
import { SERVER_URL, TestChat, awaitRoundTrip } from './helpers.ts';

describe('tool call without approval', () => {
  it('executes tool and returns text result', async () => {
    const chat = new TestChat(`${SERVER_URL}/api/chat/tool`);
    const trip = awaitRoundTrip(chat);
    chat.sendMessage({ text: 'What is the weather?' });
    await trip.done;
    assert.equal(trip.error(), null, 'request should succeed');

    const assistant = chat.messages.find((m) => m.role === 'assistant');
    assert.ok(assistant, 'should have an assistant message');

    const toolParts = assistant.parts.filter(isToolUIPart);
    assert.ok(toolParts.length > 0, 'should have at least one tool part');
    assert.ok(
      toolParts.some((p) => p.state === 'output-available'),
      'should have a tool with output-available state',
    );

    const textParts = assistant.parts.filter(isTextUIPart);
    assert.ok(textParts.length > 0, 'should have at least one text part after tool execution');
  });
});
