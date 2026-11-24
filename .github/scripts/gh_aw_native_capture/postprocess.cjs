// Only GitHub Actions presentation is adapted here. Capture validation, parser,
// redaction, and final artifact generation execute the unchanged released code.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const [helpers, scenario, evidence, expectedExitText, expectedPythonText, ownerMarker] = process.argv.slice(2);
const root = '/tmp/gh-aw';
const secret = 'acceptance-secret-unique-123';
const runtimeOnlySecret = 'native-runtime-only-mask-456';
const forgedErrorMarker = 'forged-agent-errors-native-fixture';
const { parseEventLines, agentConversation, installActionsAdapter, prepareConclusionEnvironment, assertConclusionResults } = require('./actions_adapter.cjs');
const { failures, logs, summaries } = installActionsAdapter();
const released = name => require(path.join(helpers, name));
const { readProtectedCommandOutput } = released('protected_command_output.cjs');
const parser = released('pydantic-ai_log_parser.cjs');
const { redactManifestValue, collectArtifactSecretValues } = released('safe_output_manifest.cjs');
const save = (name, value) => fs.writeFileSync(path.join(evidence, name), JSON.stringify(value, null, 2) + '\n');
(async () => {
  const captured = readProtectedCommandOutput(process.env.GH_AW_AGENT_OUTPUT);
  // Run the actual host forwarding step on spans produced inside the native
  // sandbox. Only the network destination is replaced; private content is not saved.
  const shared = fs.readFileSync(process.env.PYDANTIC_AI_WORKFLOW_ENGINE_SOURCE, 'utf8');
  const forwardBody = shared.split('      script: |\n')[1].split('\nsafe-outputs:')[0]
    .split('\n').map(line => line.replace(/^ {8}/, '')).join('\n');
  const forward = new (Object.getPrototypeOf(async function () {}).constructor)('require', 'process', 'fetch', 'core', forwardBody);
  const forwarded = [];
  await forward(name => {
    if (name === 'fs') return fs;
    if (name === 'path') return path;
    assert.equal(name, path.join(process.env.RUNNER_TEMP, 'gh-aw', 'actions', 'send_otlp_span.cjs'));
    return {
      parseOTLPEndpoints: () => [{ url: 'https://collector.invalid', headers: 'host-only' }],
      parseOTLPHeaders: value => { assert.equal(value, 'host-only'); return { Authorization: value }; },
    };
  }, process, async (url, options) => {
    assert.equal(url, 'https://collector.invalid/v1/traces');
    assert.equal(options.headers.Authorization, 'host-only');
    forwarded.push(JSON.parse(options.body));
    return { ok: true, status: 200 };
  }, { warning: () => assert.fail('unexpected export failure') });
  assert(forwarded.length > 0, 'native sandbox did not export Python spans');
  assert(!fs.existsSync('/tmp/gh-aw/pydantic-ai-private-otel.jsonl'), 'host forwarding retained private content');
  const spanNames = forwarded.flatMap(payload => payload.resourceSpans.flatMap(resource => resource.scopeSpans.flatMap(scope => scope.spans.map(span => span.name))));
  save('private-telemetry-forwarding.json', { status: 'passed', payloads: forwarded.length, spanNames });
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
    assert.equal(result.data.metadata.provider_health.task_key, 'native-capture');
    assert.equal(result.data.metadata.provider_health.failure?.kind, scenario === 'failure' ? 'rate_limit' : undefined);
    assert.equal(result.data.usage.input_tokens, 100);
    assert.equal(result.data.usage.output_tokens, 3);
    const message = events.find(event => event.type === 'assistant.message');
    assert.equal(message.data.content, `${secret} ${runtimeOnlySecret}`);
    assert.equal(Boolean(message.data.partial), scenario === 'failure');
  }
  assert.equal(report.untrusted_exit_sidecar_written, true);
  assert.equal(report.untrusted_agent_errors_written, true);
  const exitSidecar = path.join(root, 'agent_execution_exit_code.txt');
  const errorSidecar = path.join(root, 'agent-errors.jsonl');
  assert.equal(fs.readFileSync(exitSidecar, 'utf8').trim(), '0', 'hostile false-success sidecar was not written');
  const forgedError = parseEventLines(fs.readFileSync(errorSidecar, 'utf8'));
  assert.equal(forgedError.length, 1);
  assert.equal(forgedError[0].type, 'agent.execution');
  released('agent_execution.cjs').validateAgentExecution(forgedError[0].data);
  assert(forgedError[0].data.errorCodes.includes(forgedErrorMarker), 'valid hostile execution diagnostic was not written');
  // Actual compiler-packaged entry point: default root /tmp/gh-aw, no options.
  await parser.main();
  const parserFailed = failures.length > 0;
  save('parser-step.json', { failed: parserFailed, failures: [...failures], publicationGateAllowsAgentArtifacts: !parserFailed });
  assert.equal(parserFailed, scenario === 'incomplete');
  // A rejected incomplete recording still retains the native verified exit
  // sidecar on the agent host; its failed parser gate prevents its publication.
  assert.equal(Number(fs.readFileSync(exitSidecar, 'utf8').trim()), captured.exitCode, 'shared false success overrode the native exit');
  assert(!fs.existsSync(errorSidecar), 'protected parsing retained unverified agent-errors');
  save('untrusted-sidecars.json', { attackerExitCode: 0, verifiedExitCode: captured.exitCode,
                                   validForgedErrorPresentBeforeParse: true, forgedErrorRemoved: true,
                                   parserRejected: parserFailed });
  const canonical = path.join(root, 'agent-session.jsonl');
  if (scenario === 'incomplete') assert(!fs.existsSync(canonical), 'real parser failure retained an unverified canonical session');
  else {
    const text = fs.readFileSync(canonical, 'utf8');
    assert(!text.includes(secret), 'configured secret survived published canonical session');
    assert(!text.includes(runtimeOnlySecret), 'runtime-only mask did not redact published canonical content');
    assert(!text.includes(forgedErrorMarker), 'unverified agent-errors entered the canonical session');
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
  assert(!fs.existsSync(path.join(staging, 'agent-errors.jsonl')), 'unverified agent-errors crossed the publication boundary');
  if (parserFailed) assert(!fs.existsSync(path.join(staging, 'agent_execution_exit_code.txt')), 'rejected parse published an exit sidecar');
  else assert.equal(Number(fs.readFileSync(path.join(staging, 'agent_execution_exit_code.txt'), 'utf8').trim()), captured.exitCode);
  const ownerFile = path.join(root, '.native-capture-owner');
  assert.equal(fs.readFileSync(ownerFile, 'utf8'), ownerMarker, 'lost task ownership before conclusion staging');
  for (const name of fs.readdirSync(root)) {
    if (name !== '.native-capture-owner') fs.rmSync(path.join(root, name), { recursive: true, force: true });
  }
  for (const name of fs.readdirSync(staging)) fs.cpSync(path.join(staging, name), path.join(root, name), { recursive: true });
  // Conclusion has its own step environment in the released lock. Its collector
  // does not receive the parser's protected flag or configured-secret bindings.
  const expectedJobs = prepareConclusionEnvironment(scenario);
  // Exact generated conclusion entry point and default options. Never add
  // protectedSessionCapture:true to make an acceptance assertion pass.
  await released('generate_usage_artifacts.cjs').main();
  const finalPath = path.join(root, 'usage/aw_session.jsonl');
  assert(fs.existsSync(finalPath), 'released final path did not produce usage/aw_session.jsonl');
  fs.mkdirSync(path.join(evidence, 'usage'), { recursive: true });
  fs.copyFileSync(finalPath, path.join(evidence, 'usage/aw_session.jsonl'));
  const unified = parseEventLines(fs.readFileSync(finalPath, 'utf8'));
  const callerResults = assertConclusionResults(unified, scenario, expectedJobs);
  const finalText = fs.readFileSync(finalPath, 'utf8');
  assert(!finalText.includes(secret), 'configured secret survived the final published session');
  assert(!finalText.includes(runtimeOnlySecret), 'runtime-only mask survived the final published session');
  assert(!finalText.includes(forgedErrorMarker), 'unverified agent-errors entered the final published session');
  assert(!finalText.includes('forged-native-fixture'), 'released conclusion published hostile fallback artifacts');
  const conversation = agentConversation(unified);
  if (scenario === 'incomplete') assert(!conversation.some(event => ['assistant.message', 'tool.execution_start', 'session.result'].includes(event.type)), 'released conclusion promoted an incomplete or hostile conversation');
  else {
    assert(unified.some(event => event.type === 'agent.execution' && event.data.exitCode === captured.exitCode), 'final native exit evidence missing or overridden');
    assert(conversation.some(event => event.type === 'tool.execution_complete'));
    assert(conversation.some(event => event.type === 'assistant.message'));
    assert(conversation.some(event => event.type === 'session.result' && event.data.status === (scenario === 'success' ? 'success' : 'failure')));
    assert(!unified.some(event => event.type === 'session.collection_warning' && event.data.code === 'malformed_jsonl'));
  }
  save('conclusion-step.json', { entryPoint: 'generate_usage_artifacts.cjs.main()', options: {}, finalPath, events: unified.length, callerResults, status: 'passed' });
})().then(() => {
  save('actions-log.json', { logs, failures, summaries });
  console.log(`${scenario}: actual released parse, redaction, and final collection assertions passed`);
}).catch(error => {
  save('actions-log.json', { logs, failures, summaries });
  save('assertion-failure.json', { name: error.name, message: error.message, stack: error.stack });
  console.error(error.stack);
  process.exitCode = 1;
});
