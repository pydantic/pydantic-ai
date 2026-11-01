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
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@5f462569c2b7467e070c7f6c82d5e92bda50fffe
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
sandbox:
  agent:
    token-steering: false
engine:
  id: pydantic-ai
  version: '0.64.0'
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
    PAI_AGENT: pydantic_ai_gh_aw_shim.agent:agent
    PAI_RESULT_METADATA: pydantic_ai_gh_aw_shim.agent:result_metadata
    PAI_OTEL_EXPORT_PATH: /tmp/gh-aw/pydantic-ai-private-otel.jsonl
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
  concurrency:
    group: "gh-aw-pydantic-ai-${{ github.workflow }}"
    queue: max
post-steps:
  - name: Forward private workflow spans from the host
    if: always()
    continue-on-error: true
    uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
    with:
      script: |
        const fs = require('fs');
        const path = require('path');
        const { parseOTLPEndpoints, sendOTLPToAllEndpoints } = require(path.join(process.env.RUNNER_TEMP, 'gh-aw', 'actions', 'send_otlp_span.cjs'));
        // Diagnostic input only. It cannot supply health decisions or canonical usage.
        // Keep content-bearing spans out of otel.jsonl and uploaded agent artifacts.
        const source = '/tmp/gh-aw/pydantic-ai-private-otel.jsonl';
        if (!fs.existsSync(source)) return;
        const descriptor = fs.openSync(source, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
        try {
          if (!fs.fstatSync(descriptor).isFile()) throw new Error('Workflow telemetry must be a regular file');
          const endpoints = parseOTLPEndpoints();
          for (const line of fs.readFileSync(descriptor, 'utf8').split('\n').filter(Boolean)) {
            const payload = JSON.parse(line);
            if (!Array.isArray(payload.resourceSpans)) throw new Error('Invalid workflow span payload');
            await sendOTLPToAllEndpoints(endpoints, payload, { skipJSONL: true });
          }
        } finally {
          fs.closeSync(descriptor);
          fs.unlinkSync(source);
        }
safe-outputs:
  threat-detection:
    # Detection has an independent budget and uses the same verified model pricing.
    max-ai-credits: -1
    post-steps:
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
            const usage = read('/tmp/gh-aw/sandbox/firewall/logs/api-proxy-logs/token-usage.jsonl');
            const requests = usage.split('\n').filter(Boolean).map(line => JSON.parse(line));
            // Keep complete proxy counters separate from native CLI telemetry.
            for (const request of requests) {
              const endMs = Date.parse(request.timestamp);
              await logSpan('gh-aw.detector.request', {
                'gh-aw.telemetry.source': 'awf-proxy',
                'gen_ai.request.model': request.model,
                'gh-aw.detector.usage.input_tokens': request.input_tokens,
                'gh-aw.detector.usage.output_tokens': request.output_tokens,
                'gh-aw.detector.usage.cache_read_tokens': request.cache_read_tokens,
                'gh-aw.detector.usage.cache_write_tokens': request.cache_write_tokens,
                'http.response.status_code': request.status,
                'gh-aw.request.id': request.request_id,
                'gh-aw.request.duration_ms': request.duration_ms,
                'gh-aw.ai_credits': request.ai_credits_this_response,
              }, { startMs: endMs - request.duration_ms, endMs, isError: request.status >= 400 });
            }
            let retries = 0;
            let retryDelayMs = 0;
            const statuses = {};
            for (const line of log.split('\n')) {
              if (!/^\[engine\] \{\s*"type":\s*"system",\s*"subtype":\s*"api_retry"/.test(line)) continue;
              const event = JSON.parse(line.slice('[engine] '.length));
              if (event.type !== 'system' || event.subtype !== 'api_retry') continue;
              retries++;
              if (Number.isFinite(event.retry_delay_ms)) retryDelayMs += event.retry_delay_ms;
              if (Number.isInteger(event.error_status)) {
                statuses[event.error_status] = (statuses[event.error_status] || 0) + 1;
              }
            }
            const marker = log.match(/THREAT_DETECTION_STATUS: reason=(\w+) exit=(\d+)/);
            const verdictText = read(path.join(directory, 'detection_result.json'));
            const verdict = verdictText ? JSON.parse(verdictText) : null;
            const outcome = process.env.DETECTOR_JOB_STATUS === 'cancelled' ? 'cancelled' : process.env.DETECTOR_OUTCOME || 'unknown';
            const valid = verdict && ['prompt_injection', 'secret_leak', 'malicious_patch'].every(key => typeof verdict[key] === 'boolean');
            const failed = outcome !== 'success' || !valid;
            const attributes = {
              'gh-aw.telemetry.source': 'detector-evidence',
              'gh-aw.detector.outcome': outcome,
              'gh-aw.detector.reason': marker ? marker[1] : valid ? 'verdict' : 'missing_verdict',
              'gh-aw.detector.verdict': valid ? verdict.prompt_injection || verdict.secret_leak || verdict.malicious_patch ? 'blocked' : 'clean' : 'missing',
              'gh-aw.detector.requests': requests.length,
              'gh-aw.detector.usage_available': Boolean(usage),
              'gh-aw.detector.retries': retries,
              'gh-aw.detector.retry_delay_ms': retryDelayMs,
              'gh-aw.detector.log_available': Boolean(log),
            };
            for (const [status, count] of Object.entries(statuses)) {
              attributes['gh-aw.detector.retries.http_' + status] = count;
            }
            await logSpan('gh-aw.detector.result', attributes, {
              startMs: Number(process.env.GITHUB_AW_OTEL_JOB_START_MS) || Date.now(),
              isError: failed,
              errorMessage: failed ? attributes['gh-aw.detector.reason'] : undefined,
            });
    # Detection uses the stateful Claude CLI, so it retains normal recovery.
    engine:
      id: claude
      version: '2.1.227'
      model: ${{ vars.GH_AW_MODEL }}
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
        GH_AW_HARNESS_MAX_RETRIES: "3"
        CLAUDE_CODE_ENABLE_TELEMETRY: "1"
        OTEL_LOGS_EXPORTER: otlp
        OTEL_METRICS_EXPORTER: none
        OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf
        OTEL_SERVICE_NAME: gh-aw.threat-detection
---
