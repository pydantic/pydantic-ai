'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

// Execute the exact github-script body in-process, with only I/O replaced.
const source = fs.readFileSync(path.join(__dirname, '../workflows/shared/engine-zai.md'), 'utf8');
const body = source.split('          script: |\n')[1].split('    # Detection uses')[0]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const execute = new (Object.getPrototypeOf(async function () {}).constructor)('require', 'process', body);
const directory = '/tmp/gh-aw/threat-detection';
const usagePath = '/tmp/gh-aw/sandbox/firewall/logs/api-proxy-logs/token-usage.jsonl';
const cleanVerdict = { prompt_injection: false, secret_leak: false, malicious_patch: false };
const request = {
  event: 'token_usage', timestamp: '2026-10-15T20:47:27.019Z', request_id: 'response-3',
  model: 'glm-5.3-flash', status: 200, input_tokens: 780, output_tokens: 204,
  cache_read_tokens: 33792, cache_write_tokens: 0, duration_ms: 7840,
  ai_credits_this_response: 0.123276,
};

async function report(files, outcome = 'success', cancelled = false) {
  const spans = [];
  await execute(name => {
    if (name === 'fs') return {
      existsSync: file => Object.hasOwn(files, file),
      readFileSync: file => files[file],
    };
    if (name === 'path') return path;
    assert.equal(name, '/runner/gh-aw/actions/otlp.cjs');
    return { logSpan: async (name, attributes, options) => spans.push({ name, attributes, options }) };
  }, {
    env: {
      RUNNER_TEMP: '/runner', DETECTOR_OUTCOME: outcome,
      DETECTOR_JOB_STATUS: cancelled ? 'cancelled' : 'success', GITHUB_AW_OTEL_JOB_START_MS: '1000',
    },
  });
  return spans;
}

test('proxy usage retains cached tokens, duration and real credits without prompt content', async () => {
  const spans = await report({
    [usagePath]: JSON.stringify(request),
    [directory + '/detection_result.json']: JSON.stringify(cleanVerdict),
  });
  assert.equal(spans.length, 2);
  const span = spans[0];
  assert.equal(span.name, 'gh-aw.detector.request');
  assert.deepEqual(span.attributes, {
    'gh-aw.telemetry.source': 'awf-proxy',
    'gen_ai.request.model': 'glm-5.3-flash',
    'gh-aw.detector.usage.input_tokens': 780,
    'gh-aw.detector.usage.output_tokens': 204,
    'gh-aw.detector.usage.cache_read_tokens': 33792,
    'gh-aw.detector.usage.cache_write_tokens': 0,
    'http.response.status_code': 200,
    'gh-aw.request.id': 'response-3',
    'gh-aw.request.duration_ms': 7840,
    'gh-aw.ai_credits': 0.123276,
  });
  assert.deepEqual(span.options, {
    startMs: Date.parse(request.timestamp) - 7840, endMs: Date.parse(request.timestamp), isError: false,
  });
  assert.equal(spans[1].attributes['gh-aw.detector.verdict'], 'clean');
  assert.equal(spans[1].options.isError, false);
});

test('forced timeout exposes SDK retries and keeps completed response usage', async () => {
  const retry = { type: 'system', subtype: 'api_retry', attempt: 1, max_retries: 10, retry_delay_ms: 509, error_status: 502 };
  const spans = await report({
    [usagePath]: JSON.stringify(request),
    [directory + '/detection.log']: [
      '[engine] ' + JSON.stringify(retry),
      '[engine] ' + JSON.stringify({ ...retry, attempt: 2 }),
      '[engine] ' + JSON.stringify({ type: 'assistant', message: { content: 'private text' } }),
      '[engine] {"type":"assistant","message":{"content":"truncated at cancellation',
      'THREAT_DETECTION_STATUS: reason=engine_timeout exit=2',
    ].join('\n'),
  }, 'failure');
  const result = spans.at(-1);
  assert.equal(spans[0].attributes['gh-aw.detector.usage.output_tokens'], 204);
  assert.equal(result.attributes['gh-aw.detector.outcome'], 'failure');
  assert.equal(result.attributes['gh-aw.detector.reason'], 'engine_timeout');
  assert.equal(result.attributes['gh-aw.detector.verdict'], 'missing');
  assert.equal(result.attributes['gh-aw.detector.retries'], 2);
  assert.equal(result.attributes['gh-aw.detector.retry_delay_ms'], 1018);
  assert.equal(result.attributes['gh-aw.detector.retries.http_502'], 2);
  assert.equal(result.options.isError, true);
  assert.equal(result.options.errorMessage, 'engine_timeout');
  assert.equal(result.options.startMs, 1000);
  assert.equal(JSON.stringify(spans).includes('private text'), false);
});

test('a threat verdict is a completed detector result', async () => {
  const spans = await report({
    [directory + '/detection_result.json']: JSON.stringify({ ...cleanVerdict, prompt_injection: true }),
  });
  assert.equal(spans[0].attributes['gh-aw.detector.verdict'], 'blocked');
  assert.equal(spans[0].options.isError, false);
});

test('missing evidence never reports a successful verdict or known zero usage', async () => {
  const spans = await report({});
  assert.equal(spans[0].attributes['gh-aw.detector.verdict'], 'missing');
  assert.equal(spans[0].attributes['gh-aw.detector.usage_available'], false);
  assert.equal(spans[0].attributes['gh-aw.detector.log_available'], false);
  assert.equal(spans[0].options.isError, true);
});

test('cancellation wins over a prior success step outcome', async () => {
  const spans = await report({}, 'success', true);
  assert.equal(spans[0].attributes['gh-aw.detector.outcome'], 'cancelled');
  assert.equal(spans[0].options.isError, true);
});

test('malformed usage is visible as a reporting-step error', async () => {
  await assert.rejects(report({ [usagePath]: '{' }), SyntaxError);
});
