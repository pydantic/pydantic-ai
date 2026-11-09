---
# Shared public CLAI 2 engine with repository workflow policy and Z.ai routing.
# Each workflow declares only engine.model. Keep identity and credentials here:
# gh-aw validates root engine.env before registering imported engines.
#
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/api/anthropic`
# prefix and the upstream returns 404. The `ZAI_API_KEY` secret is injected as
# `ANTHROPIC_API_KEY` by the AWF api-proxy and excluded from the agent container.
# Z.AI Coding Plan exposes an Anthropic-compatible API at https://api.z.ai/api/anthropic.
#
# The public definition installs its published runtime. The repository policy is
# staged outside the checkout and selected through the public PAI_AGENT seam.
#
# Required repo variable:
#   GH_AW_MODEL — unchanged model ID, prefixed with anthropic/ by each workflow.
# Required secret:
#   ZAI_API_KEY — API key injected by the AWF api-proxy.
#
# Usage:
#   imports:
#     - shared/engine-zai.md
imports:
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@764a75c329840605ed6ecced16f102823389c817
runtimes:
  uv: {}
# Keep existing configurable budgets. AWF also has a fixed 10,000-credit cap.
# Pricing here measures API-equivalent usage, not Coding Plan quota or charges.
max-ai-credits: -1
models:
  # USD per million tokens: https://docs.z.ai/guides/overview/pricing (2026-10-15).
  default-ai-credits-pricing:
    input: 0.15
    cache_read: 0.03
    output: 0.50
  providers:
    anthropic:
      models:
        glm-5.3-flash:
          # Provider overlays use USD per token for actual AWF credit accounting.
          cost:
            input: 0.00000015
            cache_read: 0.00000003
            output: 0.00000050
engine:
  id: pydantic-ai
  version: '0.64.0'
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
    PAI_AGENT: pydantic_ai_gh_aw_shim.agent:agent
    PAI_RESULT_METADATA: pydantic_ai_gh_aw_shim.agent:result_metadata
    PAI_OTEL_EXPORT_PATH: /tmp/gh-aw/pydantic-ai-private-otel.jsonl
    # Digest of the staged policy sources, from the `Install repository workflow policy` step.
    LOGFIRE_SERVICE_VERSION: ${{ steps.workflow-policy.outputs.service-version }}
    PYTHONPATH: /tmp/gh-aw/workflow-python
    XDG_CONFIG_HOME: /tmp/gh-aw/clai2-config
    GITHUB_WORKFLOW: ${{ github.workflow }}
    PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
    # A `_KEY` suffix is treated as a secret and removed by the custom-engine sandbox.
    PYDANTIC_AI_TASK_ID: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
    # An outer retry repeats the task and can duplicate investigation or output.
    GH_AW_HARNESS_MAX_RETRIES: "0"
  # One agent run at a time per workflow, on the shared Z.AI account. gh-aw applies this
  # group by default, but drops it for a workflow with `pull_request*` triggers.
  # UI Security runs one review per PR, so reviews of different PRs run in parallel.
  concurrency:
    group: "gh-aw-pydantic-ai-${{ github.workflow }}${{ github.workflow == 'Pydantic AI UI Security Review' && format('-{0}', github.event.pull_request.number || github.ref) || '' }}"
    queue: max
# gh-aw caps the agent job at 60 minutes and detection at 10 by default. Keep GitHub's
# 360-minute job maximum, so each source's engine step budget fits, and a 20-minute detector.
jobs:
  agent:
    timeout-minutes: 360
  detection:
    timeout-minutes: 20
post-steps:
  - name: Forward private workflow spans from the host
    if: always()
    continue-on-error: true
    uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
    env:
      NODE_USE_ENV_PROXY: '1'
    with:
      script: |
        const fs = require('fs');
        const path = require('path');
        const { parseOTLPEndpoints, parseOTLPHeaders } = require(path.join(process.env.RUNNER_TEMP, 'gh-aw', 'actions', 'send_otlp_span.cjs'));
        // Diagnostic input only. It cannot supply health decisions or canonical usage.
        // Keep content-bearing spans out of otel.jsonl and uploaded agent artifacts.
        const source = '/tmp/gh-aw/pydantic-ai-private-otel.jsonl';
        if (!fs.existsSync(source)) return;
        const descriptor = fs.openSync(source, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
        try {
          if (!fs.fstatSync(descriptor).isFile()) throw new Error('Workflow telemetry must be a regular file');
          const endpoints = parseOTLPEndpoints();
          for (const line of fs.readFileSync(descriptor, 'utf8').split('\n').filter(Boolean)) {
            let payload;
            try { payload = JSON.parse(line); }
            catch { throw new Error('Invalid workflow span payload'); }
            if (!payload || !Array.isArray(payload.resourceSpans)) throw new Error('Invalid workflow span payload');
            // Preserve SDK-scrubbed private attributes, including long JSON values.
            // gh-aw's infrastructure sender truncates strings and mirrors by default.
            await Promise.all(endpoints.map(async endpoint => {
              let delivered = false;
              for (let attempt = 0; attempt < 3; attempt++) {
                if (attempt) await new Promise(resolve => setTimeout(resolve, 100 * 2 ** (attempt - 1)));
                try {
                  const response = await fetch(endpoint.url.replace(/\/$/, '') + '/v1/traces', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...parseOTLPHeaders(endpoint.headers || '') },
                    body: line,
                    signal: AbortSignal.timeout(120000),
                  });
                  if (response.ok) { delivered = true; break; }
                  if (response.status === 401 || response.status === 403) break;
                } catch {
                  // Report the outcome without logging private payloads or credentials.
                }
              }
              if (!delivered) core.warning('Private workflow telemetry export failed for a configured collector');
            }));
          }
        } finally {
          fs.closeSync(descriptor);
          fs.unlinkSync(source);
        }
safe-outputs:
  threat-detection:
    # gh-aw runs detection only on its Copilot, Claude, and Codex engines. These post-steps run an
    # OpenCode detector in the engine's place, under the step id gh-aw's conclusion step reads.
    engine: false
    post-steps:
      - name: Check out the OpenCode threat detector
        id: detector_checkout
        if: always() && steps.detection_guard.outputs.run_detection == 'true'
        continue-on-error: true
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.workflow_sha }}
          path: .gh-aw-detector
          sparse-checkout: |
            .github/actions/setup-douwebot-tools
            .github/scripts
          persist-credentials: false
      - name: Set up OpenCode
        id: detector_tools
        if: always() && steps.detector_checkout.outcome == 'success'
        continue-on-error: true
        uses: ./.gh-aw-detector/.github/actions/setup-douwebot-tools
        env:
          # The install has no lockfile. Skip dependency releases younger than 3 days, as gh-aw's Claude CLI install did.
          NPM_CONFIG_MIN_RELEASE_AGE: '3'
      # A missing verdict concludes as a warning. Stop before the 20-minute job limit so it can.
      - name: Run OpenCode threat detection
        id: detection_agentic_execution
        if: always() && steps.detection_setup.outputs.ready == 'true' && steps.threat_detect_install.outcome == 'success' && steps.detector_tools.outcome == 'success'
        continue-on-error: true
        timeout-minutes: 15
        env:
          OPENCODE_API_KEY: ${{ secrets.OPENCODE_API_KEY }}
          PYTHONPATH: ${{ runner.temp }}/douwebot-tools/python
          # Detection only reads, so a failed OpenCode run is retried. An invalid verdict is not.
          GH_AW_HARNESS_MAX_RETRIES: "3"
        run: |
          export PATH="$RUNNER_TEMP/douwebot-tools/bin:$RUNNER_TEMP/douwebot-tools/opencode/node_modules/.bin:$PATH"
          python3 .gh-aw-detector/.github/scripts/gh_aw_opencode_detect.py
      - name: Report detector usage and outcome
        if: always() && steps.detection_guard.outputs.run_detection == 'true'
        continue-on-error: true
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
        env:
          DETECTOR_OUTCOME: ${{ steps.detection_agentic_execution.outcome }}
          DETECTOR_JOB_STATUS: ${{ job.status }}
          OTEL_SERVICE_NAME: gh-aw.threat-detection
        with:
          script: |
            const fs = require('fs');
            const path = require('path');
            const { logSpan } = require(path.join(process.env.RUNNER_TEMP, 'gh-aw', 'actions', 'otlp.cjs'));
            const directory = '/tmp/gh-aw/threat-detection';
            const read = name => fs.existsSync(name) ? fs.readFileSync(name, 'utf8') : '';
            const log = read(path.join(directory, 'detection.log'));
            // One stream per OpenCode attempt. Read only step counters: the streams' text is model-authored.
            const streams = fs.readdirSync(process.env.RUNNER_TEMP).filter(name => /^gh-aw-opencode-detector-\d+\.ndjson$/.test(name));
            // A killed run can leave an unterminated last line; every complete event ends in a newline.
            const steps = streams.flatMap(name => read(path.join(process.env.RUNNER_TEMP, name)).split('\n').slice(0, -1))
              .filter(line => line.includes('"step_finish"')).map(line => JSON.parse(line))
              .filter(event => event.type === 'step_finish').map(event => event.part);
            const marker = log.match(/THREAT_DETECTION_STATUS: reason=(\w+) exit=(\d+)/);
            const verdictText = read(path.join(directory, 'detection_result.json'));
            const verdict = verdictText ? JSON.parse(verdictText) : null;
            const outcome = process.env.DETECTOR_JOB_STATUS === 'cancelled' ? 'cancelled' : process.env.DETECTOR_OUTCOME || 'unknown';
            const valid = verdict && ['prompt_injection', 'secret_leak', 'malicious_patch'].every(key => typeof verdict[key] === 'boolean');
            const failed = outcome !== 'success' || !valid;
            const sum = value => steps.reduce((total, step) => total + value(step), 0);
            const attributes = {
              'gh-aw.telemetry.source': 'detector-evidence',
              'gh-aw.detector.outcome': outcome,
              'gh-aw.detector.reason': marker ? marker[1] : valid ? 'verdict' : 'missing_verdict',
              'gh-aw.detector.verdict': valid ? verdict.prompt_injection || verdict.secret_leak || verdict.malicious_patch ? 'blocked' : 'clean' : 'missing',
              'gh-aw.detector.attempts': streams.length,
              'gh-aw.detector.requests': steps.length,
              'gh-aw.detector.usage_available': steps.length > 0,
              'gh-aw.detector.usage.input_tokens': sum(step => step.tokens?.input ?? 0),
              'gh-aw.detector.usage.output_tokens': sum(step => step.tokens?.output ?? 0),
              'gh-aw.detector.usage.reasoning_tokens': sum(step => step.tokens?.reasoning ?? 0),
              'gh-aw.detector.usage.cache_read_tokens': sum(step => step.tokens?.cache?.read ?? 0),
              'gh-aw.detector.usage.cache_write_tokens': sum(step => step.tokens?.cache?.write ?? 0),
              'gh-aw.detector.cost': sum(step => step.cost ?? 0),
              'gh-aw.detector.log_available': Boolean(log),
            };
            await logSpan('gh-aw.detector.result', attributes, {
              startMs: Number(process.env.GITHUB_AW_OTEL_JOB_START_MS) || Date.now(),
              isError: failed,
              errorMessage: failed ? attributes['gh-aw.detector.reason'] : undefined,
            });
---
