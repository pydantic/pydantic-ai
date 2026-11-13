'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

// Execute the exact github-script body in-process, with only I/O replaced.
const source = fs.readFileSync(path.join(__dirname, '../workflows/shared/engine-zai.md'), 'utf8');
const body = source.split('          script: |\n')[1].split('\n---')[0]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const execute = new (Object.getPrototypeOf(async function () {}).constructor)('require', 'process', body);
const directory = '/tmp/gh-aw/threat-detection';
const streamPath = '/runner/gh-aw-opencode-detector-1.ndjson';
const cleanVerdict = { prompt_injection: false, secret_leak: false, malicious_patch: false, reasons: [] };
const stepFinish = (input, output, cacheRead, cost) => JSON.stringify({
  type: 'step_finish', timestamp: 1794165925626,
  part: { type: 'step-finish', reason: 'stop', cost, tokens: { input, output, reasoning: 6, cache: { read: cacheRead, write: 0 } } },
});

// The step outputs the detector writes for the plan of its last OpenCode run.
const servedPlan = { DETECTOR_PROVIDER: 'zai-coding-plan', DETECTOR_PLAN: 'the backup Coding Plan' };

async function report(files, outcome = 'success', cancelled = false, plan = servedPlan) {
  const spans = [];
  await execute(name => {
    if (name === 'fs') return {
      existsSync: file => Object.hasOwn(files, file),
      readFileSync: file => files[file],
      readdirSync: directory => Object.keys(files).filter(file => path.dirname(file) === directory).map(file => path.basename(file)),
    };
    if (name === 'path') return path;
    assert.equal(name, '/runner/gh-aw/actions/otlp.cjs');
    return { logSpan: async (name, attributes, options) => spans.push({ name, attributes, options }) };
  }, {
    env: {
      RUNNER_TEMP: '/runner', DETECTOR_OUTCOME: outcome,
      DETECTOR_JOB_STATUS: cancelled ? 'cancelled' : 'success', GITHUB_AW_OTEL_JOB_START_MS: '1000', ...plan,
    },
  });
  return spans;
}

test('OpenCode step usage is summed without model-authored text', async () => {
  const spans = await report({
    [streamPath]: [
      JSON.stringify({ type: 'text', part: { text: 'private text' } }),
      stepFinish(1644, 306, 0, 0.01),
      stepFinish(571, 228, 1536, 0.002),
    ].join('\n') + '\n',
    [directory + '/detection_result.json']: JSON.stringify(cleanVerdict),
    [directory + '/detection.log']: 'THREAT_DETECTION_STATUS: reason=result_recorded exit=0\n',
  });
  assert.equal(spans.length, 1);
  assert.equal(spans[0].name, 'gh-aw.detector.result');
  assert.deepEqual(spans[0].attributes, {
    'gh-aw.telemetry.source': 'detector-evidence',
    'gh-aw.detector.outcome': 'success',
    'gh-aw.detector.reason': 'result_recorded',
    'gh-aw.detector.verdict': 'clean',
    'gh-aw.detector.provider': 'zai-coding-plan',
    'gh-aw.detector.plan': 'the backup Coding Plan',
    'gh-aw.detector.attempts': 1,
    'gh-aw.detector.requests': 2,
    'gh-aw.detector.usage_available': true,
    'gh-aw.detector.usage.input_tokens': 2215,
    'gh-aw.detector.usage.output_tokens': 534,
    'gh-aw.detector.usage.reasoning_tokens': 12,
    'gh-aw.detector.usage.cache_read_tokens': 1536,
    'gh-aw.detector.usage.cache_write_tokens': 0,
    'gh-aw.detector.cost': 0.012,
    'gh-aw.detector.log_available': true,
  });
  assert.deepEqual(spans[0].options, { startMs: 1000, isError: false, errorMessage: undefined });
  assert.equal(JSON.stringify(spans).includes('private text'), false);
});

test('step usage does not depend on event key order, and counts only step_finish events', async () => {
  const { type, ...rest } = JSON.parse(stepFinish(780, 204, 0, 0));
  const spans = await report({
    [streamPath]: [
      JSON.stringify({ ...rest, type }),
      JSON.stringify({ type: 'text', part: { tool: 'step_finish' } }),
    ].join('\n') + '\n',
  });
  assert.equal(spans[0].attributes['gh-aw.detector.requests'], 1);
  assert.equal(spans[0].attributes['gh-aw.detector.usage.output_tokens'], 204);
});

test('retried attempts all count toward usage', async () => {
  const spans = await report({
    [streamPath]: stepFinish(1644, 306, 0, 0.01) + '\n',
    '/runner/gh-aw-opencode-detector-2.ndjson': stepFinish(571, 228, 1536, 0.002) + '\n',
    '/runner/unrelated.ndjson': stepFinish(1, 1, 1, 1) + '\n',
  });
  assert.equal(spans[0].attributes['gh-aw.detector.attempts'], 2);
  assert.equal(spans[0].attributes['gh-aw.detector.requests'], 2);
  assert.equal(spans[0].attributes['gh-aw.detector.usage.input_tokens'], 2215);
});

test('an invalid report keeps its usage and status reason', async () => {
  const spans = await report({
    [streamPath]: stepFinish(780, 204, 0, 0) + '\n',
    [directory + '/detection.log']: 'THREAT_DETECTION_STATUS: reason=invalid_report_exhausted exit=1\n',
  }, 'failure');
  const result = spans[0];
  assert.equal(result.attributes['gh-aw.detector.usage.output_tokens'], 204);
  assert.equal(result.attributes['gh-aw.detector.outcome'], 'failure');
  assert.equal(result.attributes['gh-aw.detector.reason'], 'invalid_report_exhausted');
  assert.equal(result.attributes['gh-aw.detector.verdict'], 'missing');
  assert.equal(result.options.isError, true);
  assert.equal(result.options.errorMessage, 'invalid_report_exhausted');
});

test('a timeout keeps its reason and earlier usage when the killed run left a partial line', async () => {
  const spans = await report({
    [streamPath]: stepFinish(780, 204, 0, 0) + '\n',
    '/runner/gh-aw-opencode-detector-2.ndjson': stepFinish(571, 228, 0, 0) + '\n' + stepFinish(1, 1, 1, 1).slice(0, 40),
    [directory + '/detection.log']: 'THREAT_DETECTION_STATUS: reason=engine_timeout exit=2\n',
  }, 'failure');
  assert.equal(spans[0].attributes['gh-aw.detector.reason'], 'engine_timeout');
  assert.equal(spans[0].attributes['gh-aw.detector.requests'], 2);
  assert.equal(spans[0].attributes['gh-aw.detector.usage.output_tokens'], 432);
});

test('a step_finish without token counters still reports the result', async () => {
  const spans = await report({ [streamPath]: JSON.stringify({ type: 'step_finish', part: { reason: 'stop' } }) + '\n' });
  assert.equal(spans[0].attributes['gh-aw.detector.requests'], 1);
  assert.equal(spans[0].attributes['gh-aw.detector.usage.input_tokens'], 0);
  assert.equal(spans[0].attributes['gh-aw.detector.cost'], 0);
});

test('a threat verdict is a completed detector result', async () => {
  const spans = await report({
    [directory + '/detection_result.json']: JSON.stringify({ ...cleanVerdict, prompt_injection: true }),
  });
  assert.equal(spans[0].attributes['gh-aw.detector.verdict'], 'blocked');
  assert.equal(spans[0].options.isError, false);
});

test('a detector that never ran OpenCode names no plan', async () => {
  // A step that wrote no outputs leaves GitHub's expressions empty.
  const spans = await report({}, 'failure', false, { DETECTOR_PROVIDER: '', DETECTOR_PLAN: '' });
  assert.equal(spans[0].attributes['gh-aw.detector.provider'], 'unknown');
  assert.equal(spans[0].attributes['gh-aw.detector.plan'], 'unknown');
});

test('missing evidence never reports a successful verdict or known zero usage', async () => {
  const spans = await report({});
  assert.equal(spans[0].attributes['gh-aw.detector.verdict'], 'missing');
  assert.equal(spans[0].attributes['gh-aw.detector.reason'], 'missing_verdict');
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
  await assert.rejects(report({ [streamPath]: '{"type":"step_finish",\n' }), SyntaxError);
});
