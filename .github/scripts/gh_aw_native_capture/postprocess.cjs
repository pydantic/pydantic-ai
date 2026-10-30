// Only GitHub Actions presentation is adapted here. Capture validation, parser,
// redaction, and final artifact generation execute the unchanged released code.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const [helpers, scenario, evidence, expectedExitText, expectedPythonText, ownerMarker] = process.argv.slice(2);
const root = '/tmp/gh-aw';
const secret = 'acceptance-secret-unique-123';
const { installActionsAdapter, prepareConclusionEnvironment } = require('./actions_adapter.cjs');
const { failures, logs, summaries } = installActionsAdapter();
const released = name => require(path.join(helpers, name));
const { readProtectedCommandOutput } = released('protected_command_output.cjs');
const parser = released('pydantic-ai_log_parser.cjs');
const { redactManifestValue, collectArtifactSecretValues } = released('safe_output_manifest.cjs');
const save = (name, value) => fs.writeFileSync(path.join(evidence, name), JSON.stringify(value, null, 2) + '\n');
const eventLines = file => fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
(async () => {
  const captured = readProtectedCommandOutput(process.env.GH_AW_AGENT_OUTPUT);
  assert.equal(captured.exitCode, Number(expectedExitText));
  assert(!captured.content.includes('forged-native-fixture'), 'ordinary-stream forgery entered protected stderr');
  const parsed = parser.parseLog(captured.content);
  save('authenticated-parser-result.json', redactManifestValue(parsed, collectArtifactSecretValues()));
  const events = scenario === 'incomplete' ? parsed.diagnosticLogEntries : parsed.logEntries;
  assert(events.length >= 4, 'real model-free agent did not record the tool execution');
  const toolResult = events.find(event => event.type === 'tool.execution_complete');
  assert(toolResult, 'missing true hostile-probe tool result');
  const report = JSON.parse(toolResult.data.output);
  assert.deepEqual(report.violations, []);
  assert.equal(report.same_uid, true);
  assert.equal(report.direct_parent, true);
  assert.equal(report.parent_dumpable, 0);
  assert.equal(report.parent_capture_inheritable, false);
  assert.equal(report.python_executable, '/usr/bin/python3');
  assert.deepEqual(report.python_version, JSON.parse(expectedPythonText));
  assert.deepEqual(report.inherited_capture_fds, []);
  assert.equal(report.parent_process_vm_readv.bytes, -1);
  assert([1, 13].includes(report.parent_process_vm_readv.errno));
  save('probe-report.json', report);
  if (scenario === 'incomplete') {
    assert.deepEqual(parsed.logEntries, []);
    assert(!events.some(event => event.type === 'session.result'));
    assert(parsed.markdown.includes('Incomplete agent recording'));
  } else {
    const result = events.at(-1);
    assert.equal(result.type, 'session.result');
    assert.equal(result.data.status, scenario === 'success' ? 'success' : 'failure');
    assert.equal(result.data.usage.input_tokens, 100);
    assert.equal(result.data.usage.output_tokens, 2);
    const message = events.find(event => event.type === 'assistant.message');
    assert.equal(message.data.content, secret);
    assert.equal(Boolean(message.data.partial), scenario === 'failure');
  }
  // Actual compiler-packaged entry point: default root /tmp/gh-aw, no options.
  await parser.main();
  const parserFailed = failures.length > 0;
  save('parser-step.json', { failed: parserFailed, failures: [...failures], publicationGateAllowsAgentArtifacts: !parserFailed });
  assert.equal(parserFailed, scenario === 'incomplete');
  const canonical = path.join(root, 'agent-session.jsonl');
  if (scenario === 'incomplete') assert(!fs.existsSync(canonical), 'real parser failure retained an unverified canonical session');
  else {
    const text = fs.readFileSync(canonical, 'utf8');
    assert(!text.includes(secret), 'configured secret survived published canonical session');
    assert(text.includes('***'), 'redaction marker absent from canonical session');
    fs.copyFileSync(canonical, path.join(evidence, 'agent-session.jsonl'));
  }
  const previousFailures = failures.length;
  await released('redact_secrets.cjs').main();
  assert.equal(failures.length, previousFailures, 'released artifact redaction failed');
  // Reproduce the actual cross-job artifact boundary before invoking conclusion:
  // only compiler-declared upload paths cross, and both uploads are parser-gated.
  const transfers = JSON.parse(fs.readFileSync(path.join(helpers, 'artifact-transfer.json'), 'utf8'));
  const staging = path.join(evidence, 'conclusion-input');
  fs.mkdirSync(staging, { recursive: true });
  const selected = new Set();
  for (const upload of transfers) {
    assert(upload.condition.includes("steps.parse_agent_logs.outcome == 'success'"));
    assert(upload.condition.includes("steps.redact_secrets.outcome == 'success'"));
    if (!parserFailed) {
      for (const pattern of upload.paths) {
        for (const file of fs.globSync(pattern, { cwd: root })) {
          const destination = path.join(staging, file);
          fs.mkdirSync(path.dirname(destination), { recursive: true });
          fs.cpSync(path.join(root, file), destination, { recursive: true });
          selected.add(file);
        }
      }
    }
  }
  save('artifact-transfer.json', { uploads: transfers, parserFailed, selectedPaths: [...selected],
                                  method: 'local staging of exact emitted upload allowlist; no GitHub artifact service call' });
  const ownerFile = path.join(root, '.native-capture-owner');
  assert.equal(fs.readFileSync(ownerFile, 'utf8'), ownerMarker, 'lost task ownership before conclusion staging');
  for (const name of fs.readdirSync(root)) {
    if (name !== '.native-capture-owner') fs.rmSync(path.join(root, name), { recursive: true, force: true });
  }
  for (const name of fs.readdirSync(staging)) fs.cpSync(path.join(staging, name), path.join(root, name), { recursive: true });
  // Conclusion has its own step environment in the released lock. Its collector
  // does not receive the parser's protected flag or configured-secret bindings.
  prepareConclusionEnvironment();
  // Exact generated conclusion entry point and default options. Never add
  // protectedSessionCapture:true to make an acceptance assertion pass.
  await released('generate_usage_artifacts.cjs').main();
  const finalPath = path.join(root, 'usage/aw_session.jsonl');
  assert(fs.existsSync(finalPath), 'released final path did not produce usage/aw_session.jsonl');
  fs.mkdirSync(path.join(evidence, 'usage'), { recursive: true });
  fs.copyFileSync(finalPath, path.join(evidence, 'usage/aw_session.jsonl'));
  const unified = eventLines(finalPath);
  const finalText = fs.readFileSync(finalPath, 'utf8');
  assert(!finalText.includes(secret), 'configured secret survived the final published session');
  assert(!finalText.includes('forged-native-fixture'), 'released conclusion published hostile fallback artifacts');
  const conversation = unified.filter(event => event.provenance.component === 'agent');
  if (scenario === 'incomplete') assert(!conversation.some(event => ['assistant.message', 'tool.execution_start', 'session.result'].includes(event.type)), 'released conclusion promoted an incomplete or hostile conversation');
  else {
    assert(conversation.some(event => event.type === 'tool.execution_complete'));
    assert(conversation.some(event => event.type === 'assistant.message'));
    assert(conversation.some(event => event.type === 'session.result' && event.data.status === (scenario === 'success' ? 'success' : 'failure')));
    assert(!unified.some(event => event.type === 'session.collection_warning' && event.data.code === 'malformed_jsonl'));
  }
  save('conclusion-step.json', { entryPoint: 'generate_usage_artifacts.cjs.main()', options: {}, finalPath, events: unified.length, status: 'passed' });
})().then(() => {
  save('actions-log.json', { logs, failures, summaries });
  console.log(`${scenario}: actual released parse, redaction, and final collection assertions passed`);
}).catch(error => {
  save('actions-log.json', { logs, failures, summaries });
  save('assertion-failure.json', { name: error.name, message: error.message, stack: error.stack });
  console.error(error.stack);
  process.exitCode = 1;
});
