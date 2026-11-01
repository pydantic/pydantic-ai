'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../workflows/shared/engine-zai.md'), 'utf8');
const body = source.split('      script: |\n')[1].split('\nsafe-outputs:')[0]
  .split('\n').map(line => line.replace(/^ {8}/, '')).join('\n');
const execute = new (Object.getPrototypeOf(async function () {}).constructor)('require', 'process', 'fetch', 'core', body);

async function forward(content, regular = true) {
  const sent = [];
  const endpoints = [{ url: 'https://collector.invalid', headers: 'host-only' }];
  let closed = false;
  let removed = false;
  const files = {
    constants: fs.constants,
    existsSync: () => content !== undefined,
    openSync: (filename, flags) => {
      assert.equal(filename, '/tmp/gh-aw/pydantic-ai-private-otel.jsonl');
      assert.equal(flags, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
      return 3;
    },
    fstatSync: () => ({ isFile: () => regular }),
    readFileSync: descriptor => { assert.equal(descriptor, 3); return content; },
    closeSync: descriptor => { assert.equal(descriptor, 3); closed = true; },
    unlinkSync: () => { removed = true; },
  };
  let error;
  try {
    await execute(name => {
      if (name === 'fs') return files;
      if (name === 'path') return path;
      assert.equal(name, '/runner/gh-aw/actions/send_otlp_span.cjs');
      return {
        parseOTLPEndpoints: () => endpoints,
        parseOTLPHeaders: value => { assert.equal(value, 'host-only'); return { Authorization: value }; },
      };
    }, { env: { RUNNER_TEMP: '/runner' } }, async (url, options) => {
      assert.equal(url, 'https://collector.invalid/v1/traces');
      assert.equal(options.method, 'POST');
      assert.equal(options.headers.Authorization, 'host-only');
      sent.push(JSON.parse(options.body));
      return { ok: true, status: 200 };
    }, { warning: () => assert.fail('unexpected export failure') });
  } catch (caught) {
    error = caught;
  }
  return { sent, closed, removed, error };
}

test('private workflow spans use host destinations without an artifact mirror', async () => {
  const messages = JSON.stringify({ content: 'private probe '.repeat(400) });
  const payload = { resourceSpans: [{ scopeSpans: [{ spans: [{ name: 'invoke_agent', traceId: '1'.repeat(32),
    attributes: [{ key: 'gen_ai.input.messages', value: { stringValue: messages } }] }] }] }] };
  const result = await forward(JSON.stringify(payload) + '\n');
  assert.deepEqual(result, { sent: [payload], closed: true, removed: true, error: undefined });
});

test('missing spans after a noop are a no-op', async () => {
  assert.deepEqual(await forward(undefined), { sent: [], closed: false, removed: false, error: undefined });
});

test('invalid diagnostic input is removed without exporting it', async () => {
  const result = await forward('{"notResourceSpans":[]}');
  assert.equal(result.sent.length, 0);
  assert.equal(result.closed, true);
  assert.equal(result.removed, true);
  assert.match(result.error.message, /Invalid workflow span payload/);
});

test('malformed private JSON never appears in the reported error', async () => {
  const result = await forward('private-sentinel');
  assert.equal(result.sent.length, 0);
  assert.equal(result.removed, true);
  assert.equal(result.error.message.includes('private-sentinel'), false);
});

test('the host refuses non-regular diagnostic files', async () => {
  const result = await forward('{}', false);
  assert.equal(result.sent.length, 0);
  assert.equal(result.closed, true);
  assert.match(result.error.message, /regular file/);
});
