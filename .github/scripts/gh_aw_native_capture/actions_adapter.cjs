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

function prepareConclusionEnvironment() {
  // These are the actual conclusion-step bindings, without parser-only secrets
  // or a protectedSessionCapture option absent from the released final caller.
  for (const name of ['GH_AW_AGENT_OUTPUT', 'GH_AW_PROTECTED_SESSION_CAPTURE', 'GH_AW_SECRET_NAMES', 'SECRET_CAPTURE_FIXTURE_SECRET']) delete process.env[name];
  process.env.GH_AW_GRADER_MANIFEST_CHUNKS = '1';
  process.env.GH_AW_GRADER_MANIFEST_0 = '{"version":2,"graders":[]}';
  process.env.GH_AW_JOB_DURATIONS = '{}';
  // No job timing lookup is required for this fixture's three local scenarios.
  process.env.GH_AW_JOB_RESULTS = '';
}

module.exports = { installActionsAdapter, prepareConclusionEnvironment };
