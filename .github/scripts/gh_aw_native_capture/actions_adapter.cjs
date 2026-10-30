const assert = require('node:assert/strict');

// A local adapter for the GitHub Actions presentation surface only. It does not
// replace capture readers, parsers, redactors, or session collectors.
function installActionsAdapter() {
  const failures = [];
  const logs = [];
  const summaries = [];
  const summary = new Proxy({}, {
    get(_target, name) {
      if (name === 'write') return async () => summary;
      if (name === 'stringify') return () => summaries.join('\n');
      if (name === 'clear') return () => summary;
      return (...args) => { summaries.push(args.map(String).join(' ')); return summary; };
    },
  });
  global.core = {
    info: message => logs.push(String(message)), warning: message => logs.push(`warning: ${message}`),
    error: message => logs.push(`error: ${message}`), debug: message => logs.push(`debug: ${message}`),
    setFailed: message => { failures.push(String(message)); },
    setOutput: (name, value) => logs.push(`output ${name}: ${value}`), setSecret: () => {}, summary,
  };
  return { failures, logs, summaries };
}

function prepareConclusionEnvironment(scenario) {
  assert(['success', 'failure', 'incomplete'].includes(scenario), 'unknown fixture conclusion scenario');
  // These are the actual conclusion-step bindings, without parser-only secrets
  // or a protectedSessionCapture option absent from the released final caller.
  for (const name of ['GH_AW_AGENT_OUTPUT', 'GH_AW_PROTECTED_SESSION_CAPTURE', 'GH_AW_SECRET_NAMES', 'SECRET_CAPTURE_FIXTURE_SECRET']) delete process.env[name];
  process.env.GH_AW_GRADER_MANIFEST_CHUNKS = '1';
  process.env.GH_AW_GRADER_MANIFEST_0 = '{"version":2,"graders":[]}';
  process.env.GH_AW_JOB_DURATIONS = '{}';
  // The emitted agent execution step has no continue-on-error. Healthy empty
  // detection/safe-output processing can still succeed after a failed agent:
  // detection's no-output guard skips inference, and the no-message handler
  // explicitly succeeds. These statuses describe that bounded no-output caller.
  const jobs = {
    activation: { result: 'success' },
    agent: { result: scenario === 'success' ? 'success' : 'failure' },
    detection: { result: 'success' },
    safe_outputs: { result: 'success' },
  };
  process.env.GH_AW_JOB_RESULTS = JSON.stringify(jobs);
  process.env.GH_AW_JOB_DISPLAY_NAMES = JSON.stringify({ activation: 'activation', agent: 'agent', conclusion: 'conclusion', detection: 'detection', pre_activation: 'pre_activation', safe_outputs: 'safe_outputs' });
  process.env.GH_AW_JOB_TIMING_ELIGIBLE = JSON.stringify({ activation: true, agent: true, conclusion: true, detection: true, pre_activation: true, safe_outputs: true });
  return jobs;
}

function assertConclusionResults(events, scenario, expectedJobs) {
  const observed = Object.fromEntries(events.filter(event => event.type === 'workflow.job_result').map(event => [event.data.jobId, { result: event.data.result }]));
  assert.deepEqual(observed, expectedJobs, 'final workflow job results differ from the actual supplied caller outcomes');
  const expectedAgent = scenario === 'success' ? 'success' : 'failure';
  assert.equal(observed.agent.result, expectedAgent, 'agent execution failure became a successful workflow job');
  const results = events.filter(event => event.type === 'session.result' && event.provenance.component === 'agent');
  if (scenario === 'incomplete') assert.equal(results.length, 0, 'incomplete recording gained a canonical result');
  else assert.equal(results.at(-1)?.data.status, expectedAgent, 'canonical agent result disagrees with the caller job');
  const execution = events.filter(event => event.type === 'agent.execution');
  for (const event of execution) assert.equal(event.data.exitCode, scenario === 'success' ? 0 : scenario === 'failure' ? 1 : 137);
  const workflowResultFromJobs = Object.values(observed).some(job => job.result === 'failure') ? 'failure' : 'success';
  assert.equal(workflowResultFromJobs, expectedAgent);
  return { jobResults: observed, agentResult: expectedAgent, workflowResultFromJobs };
}

module.exports = { installActionsAdapter, prepareConclusionEnvironment, assertConclusionResults };
