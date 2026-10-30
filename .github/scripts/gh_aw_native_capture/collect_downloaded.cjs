// Consume the producer evidence after the workflow's real upload/download steps.
// No capture metadata, signatures, or native-success attestations are generated.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { installActionsAdapter, prepareConclusionEnvironment, assertConclusionResults } = require('./actions_adapter.cjs');

function copyVerifiedHelpers(input, output, manifest) {
  for (const [name, digest] of Object.entries(manifest)) {
    assert(!path.isAbsolute(name) && !name.split('/').includes('..'), 'unsafe helper manifest path');
    const source = path.join(input, name);
    assert(fs.lstatSync(source).isFile(), `released helper is not a regular file: ${name}`);
    const content = fs.readFileSync(source);
    assert.equal(crypto.createHash('sha256').update(content).digest('hex'), digest, `released helper digest mismatch: ${name}`);
    const destination = path.join(output, name);
    fs.mkdirSync(path.dirname(destination), { recursive: true, mode: 0o700 });
    fs.writeFileSync(destination, content, { mode: 0o600, flag: 'wx' });
  }
}

function requireRegularTree(directory) {
  assert(fs.lstatSync(directory).isDirectory(), 'conclusion input is not a real directory');
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    const stat = fs.lstatSync(file);
    assert(!stat.isSymbolicLink(), `symlink in downloaded conclusion input: ${entry.name}`);
    if (stat.isDirectory()) requireRegularTree(file);
    else assert(stat.isFile(), `unsupported downloaded conclusion input: ${entry.name}`);
  }
}

async function main() {
  const [inputArgument, outputArgument] = process.argv.slice(2);
  assert(inputArgument && outputArgument, 'usage: collect_downloaded.cjs DOWNLOADED_EVIDENCE OUTPUT_DIRECTORY');
  const input = path.resolve(inputArgument);
  const output = path.resolve(outputArgument);
  assert(!fs.existsSync(output), 'round-trip output directory already exists');
  const root = '/tmp/gh-aw';
  assert(!fs.existsSync(root), '/tmp/gh-aw already exists; refusing to replace another task');
  const producer = JSON.parse(fs.readFileSync(path.join(input, 'result.json'), 'utf8'));
  assert.equal(producer.status, 'passed', 'native producer did not finish acceptance');
  assert.equal(producer.gh_aw, '1.0.1');
  assert.equal(producer.awf, '0.29.0');
  assert.equal(producer.source_commit, '0627fdbe4beb5f3b71aa569f8d97150b87f2111a');
  const cases = [['success', 0], ['failure', 1], ['incomplete', 137]];
  assert.equal(producer.cases.length, cases.length);
  for (const [scenario, exitCode] of cases) {
    const result = producer.cases.find(item => item.case === scenario);
    assert(result && result.status === 'passed');
    assert.equal(result.awf_exit_code, exitCode);
    assert.equal(result.native_exit_code, exitCode);
  }
  fs.mkdirSync(output, { recursive: true, mode: 0o700 });
  const verified = path.join(output, 'verified-released-helpers');
  fs.mkdirSync(verified, { mode: 0o700 });
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, 'released_helpers.json'), 'utf8'));
  copyVerifiedHelpers(path.join(input, 'released-helpers'), verified, manifest);
  // Require only the verified copy. Extra downloaded JS or node_modules cannot
  // participate in released helper resolution.
  const adapter = installActionsAdapter();
  const generate = require(path.join(verified, 'generate_usage_artifacts.cjs')).main;
  const marker = `gh-aw-downloaded-conclusion-${process.pid}-${crypto.randomBytes(12).toString('hex')}`;
  fs.mkdirSync(root, { mode: 0o700 });
  const owner = path.join(root, '.native-capture-owner');
  fs.writeFileSync(owner, marker, { mode: 0o600, flag: 'wx' });
  const results = [];
  try {
    for (const [scenario] of cases) {
      const staging = path.join(input, scenario, 'conclusion-input');
      const transfer = JSON.parse(fs.readFileSync(path.join(input, scenario, 'artifact-transfer.json'), 'utf8'));
      if (scenario === 'incomplete') {
        assert.equal(transfer.parserFailed, true);
        assert.deepEqual(transfer.selectedPaths, [], 'rejected parser selected agent artifacts');
        // Artifact services need not retain empty directories. Absence represents
        // the producer's verified rejection, never a substitute successful session.
        if (fs.existsSync(staging)) {
          requireRegularTree(staging);
          assert.deepEqual(fs.readdirSync(staging), [], 'rejected parser uploaded agent artifacts');
        }
      } else {
        assert.equal(transfer.parserFailed, false);
        requireRegularTree(staging);
        assert(fs.existsSync(path.join(staging, 'agent-session.jsonl')), 'verified canonical session missing after download');
      }
      const sidecarReceipt = JSON.parse(fs.readFileSync(path.join(input, scenario, 'untrusted-sidecars.json'), 'utf8'));
      const expectedNativeExit = cases.find(([name]) => name === scenario)[1];
      assert.equal(sidecarReceipt.attackerExitCode, 0);
      assert.equal(sidecarReceipt.verifiedExitCode, expectedNativeExit);
      assert.equal(sidecarReceipt.validForgedErrorPresentBeforeParse, true);
      assert.equal(sidecarReceipt.forgedErrorRemoved, true);
      assert.equal(sidecarReceipt.parserRejected, scenario === 'incomplete');
      assert(!fs.existsSync(path.join(staging, 'agent-errors.jsonl')), 'unverified agent-errors survived artifact download');
      if (scenario === 'incomplete') assert(!fs.existsSync(path.join(staging, 'agent_execution_exit_code.txt')), 'rejected parser transferred an exit sidecar');
      else assert.equal(Number(fs.readFileSync(path.join(staging, 'agent_execution_exit_code.txt'), 'utf8').trim()), expectedNativeExit);
      assert(!fs.existsSync(path.join(staging, '.native-capture-owner')), 'downloaded input carries an ownership marker');
      assert(!fs.existsSync(path.join(staging, 'sandbox/agent/logs/copilot-session-state')), 'hostile native fallback crossed the producer publication boundary');
      assert.equal(fs.readFileSync(owner, 'utf8'), marker, 'lost task ownership before copying downloaded inputs');
      if (fs.existsSync(staging)) fs.cpSync(staging, root, { recursive: true });
      const expectedJobs = prepareConclusionEnvironment(scenario);
      process.env.RUNNER_TEMP = path.join(output, 'runner');
      // Same released entry point and zero options as the emitted conclusion.
      await generate();
      assert.deepEqual(adapter.failures, [], 'released conclusion reported a failed action');
      const finalPath = path.join(root, 'usage/aw_session.jsonl');
      const text = fs.readFileSync(finalPath, 'utf8');
      assert(!text.includes('acceptance-secret-unique-123'), 'configured secret survived the artifact round trip');
      assert(!text.includes('forged-native-fixture'), 'hostile fallback became authoritative after download');
      assert(!text.includes('forged-agent-errors-native-fixture'), 'unverified agent-errors became authoritative after download');
      const events = text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
      const callerResults = assertConclusionResults(events, scenario, expectedJobs);
      const conversation = events.filter(event => event.provenance.component === 'agent');
      if (scenario === 'incomplete') assert.equal(conversation.length, 0, 'download promoted an incomplete conversation');
      else {
        assert(events.some(event => event.type === 'agent.execution' && event.data.exitCode === expectedNativeExit), 'native exit changed after artifact download');
        const result = conversation.findLast(event => event.type === 'session.result');
        assert(result, 'terminal session result missing after download');
        assert.equal(result.data.status, scenario === 'success' ? 'success' : 'failure');
        assert.equal(result.data.usage.inputTokens, 100);
        assert.equal(result.data.usage.outputTokens, 2);
        const tool = conversation.find(event => event.type === 'tool.execution_complete' && event.data.toolName === 'hostile_probe');
        assert(tool, 'true hostile-probe tool result missing');
        const toolReport = JSON.parse(tool.data.output);
        assert.deepEqual(toolReport, JSON.parse(fs.readFileSync(path.join(input, scenario, 'probe-report.json'), 'utf8')), 'true hostile-probe report changed during the round trip');
        const message = conversation.find(event => event.type === 'assistant.message');
        assert(message && message.data.content.includes('***'), 'configured-secret redaction disappeared');
        assert.equal(Boolean(message.data.partial), scenario === 'failure');
      }
      const destination = path.join(output, scenario, 'usage');
      fs.mkdirSync(destination, { recursive: true, mode: 0o700 });
      fs.writeFileSync(path.join(destination, 'aw_session.jsonl'), text, { mode: 0o600 });
      results.push({ case: scenario, status: 'passed', events: events.length, conversationEvents: conversation.length, callerResults });
      assert.equal(fs.readFileSync(owner, 'utf8'), marker, 'lost task ownership before shared-tree cleanup');
      for (const name of fs.readdirSync(root)) if (name !== '.native-capture-owner') fs.rmSync(path.join(root, name), { recursive: true });
    }
    fs.writeFileSync(path.join(output, 'actions-log.json'), JSON.stringify(adapter, null, 2) + '\n');
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify({ status: 'passed', stage: 'downloaded-artifact-conclusion', verifiedHelpers: Object.keys(manifest).length, entryPoint: 'generate_usage_artifacts.cjs.main()', options: {}, cases: results }, null, 2) + '\n');
    console.log(`Downloaded artifact conclusion acceptance passed: ${path.join(output, 'result.json')}`);
  } finally {
    assert.equal(fs.readFileSync(owner, 'utf8'), marker, 'lost task ownership; refusing shared-tree cleanup');
    fs.rmSync(root, { recursive: true });
  }
}

if (require.main === module) {
  let completed = false;
  process.on('beforeExit', () => {
    if (!completed) {
      console.error('Artifact collection exited before completing its acceptance assertions');
      process.exitCode = 1;
    }
  });
  main().then(() => { completed = true; }).catch(error => { completed = true; console.error(error.stack); process.exitCode = 1; });
}
module.exports = { copyVerifiedHelpers, requireRegularTree };
