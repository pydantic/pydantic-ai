// Run with: node --experimental-vm-modules tests/ui_roundtrip/chat_app_stream.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as nodeModule from 'node:module';
import { setImmediate } from 'node:timers/promises';
import { test } from 'node:test';
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm';

if (!nodeModule.stripTypeScriptTypes) {
  console.error('The chat example regression needs Node >=22.13 for TypeScript stripping.');
  process.exit(77);
}

// Compile the shipped client, with only a test export appended; no parser is reimplemented here.
const source = nodeModule.stripTypeScriptTypes(
  readFileSync(new URL('../../examples/pydantic_ai_examples/chat_app.ts', import.meta.url), 'utf8') +
    '\nexport { onFetchResponse };',
);

async function client() {
  const messages = new Map();
  const input = { disabled: true, focus() {} };
  const elements = {
    conversation: { appendChild(element) { messages.set(element.id, element); } },
    'prompt-input': input,
    spinner: { classList: { remove() {} } },
  };
  const context = createContext({
    TextDecoder,
    document: {
      getElementById: (id) => elements[id] ?? messages.get(id),
      createElement: () => ({ classList: { add() {} }, innerHTML: '' }),
      querySelector: () => ({ addEventListener() {} }),
      body: { scrollHeight: 0 },
    },
    window: { scrollTo() {} },
    // Keep the client's automatic history fetch pending; each test supplies its own response.
    fetch: () => new Promise(() => {}),
  });
  const module = new SourceTextModule(source, { context });
  await module.link((specifier) => {
    assert.equal(specifier, 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js');
    return new SyntheticModule(['marked'], function () {
      this.setExport('marked', { parse: (content) => content });
    }, { context });
  });
  await module.evaluate();
  return { consume: module.namespace.onFetchResponse, messages, input };
}

const records = [
  { role: 'user', timestamp: '2026-01-01T00:00:00Z', content: 'Hello' },
  { role: 'model', timestamp: '2026-01-01T00:00:01Z', content: 'Hello! ' },
  { role: 'model', timestamp: '2026-01-01T00:00:01Z', content: 'Hello! How can I help?' },
];
const lines = records.map((record) => JSON.stringify(record));
const wire = lines.join('\n') + '\n';
const encoder = new TextEncoder();

for (const [name, chunks] of Object.entries({
  'one read with several records': [wire],
  'a record split between reads': [wire.slice(0, 37), wire.slice(37)],
  'one byte per read': [...encoder.encode(wire)].map((byte) => Uint8Array.of(byte)),
  'newlines in their own reads': lines.flatMap((line) => [line, '\n']),
  'complete records followed by a partial record': [wire.slice(0, -9), wire.slice(-9)],
  'final record without a newline': [wire.slice(0, -1)],
})) {
  test(name, async () => {
    const app = await client();
    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(typeof chunk === 'string' ? encoder.encode(chunk) : chunk);
        }
        controller.close();
      },
    });
    await app.consume(new Response(stream));
    assert.deepEqual([...app.messages.values()].map((message) => message.innerHTML), [
      records[0].content, records[2].content,
    ]);
    assert.equal(app.input.disabled, false);
  });
}

test('complete records render while the next record is still arriving', async () => {
  const app = await client();
  let controller;
  const stream = new ReadableStream({ start(c) { controller = c; } });
  const done = app.consume(new Response(stream));
  controller.enqueue(encoder.encode(lines[0] + '\n' + lines[1].slice(0, 20)));
  await setImmediate();
  assert.deepEqual([...app.messages.values()].map((message) => message.innerHTML), [records[0].content]);
  assert.equal(app.input.disabled, true);
  controller.enqueue(encoder.encode(lines[1].slice(20) + '\n'));
  controller.close();
  await done;
  assert.equal(app.messages.get(`msg-${records[1].timestamp}`).innerHTML, records[1].content);
});

test('empty history completes and malformed JSON still fails', async () => {
  const app = await client();
  await app.consume(new Response(''));
  assert.equal(app.messages.size, 0);
  await assert.rejects(app.consume(new Response('{"role":')), { name: 'SyntaxError' });
  await assert.rejects(app.consume(new Response('{"role":\n')), { name: 'SyntaxError' });
});
