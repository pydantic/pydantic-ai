---
runtimes:
  python:
    version: "3.12"
pre-agent-steps:
  - name: Preinstall Pydantic AI coder agent
    run: |
      # This step runs on the host runner with the checkout as its working
      # directory, before the AWF sandbox exists. -P keeps that directory off
      # sys.path, so a repo-local pip.py or pydantic_ai_harness/ cannot be
      # imported in place of the installed packages.
      #
      # CLAI 2.0's published `-p` command is the headless interface used by this
      # engine. The agent hooks and MCP composition require Pydantic AI 2.54.
      python3 -P -m pip install --quiet --user --disable-pip-version-check "pydantic-ai-harness==${GH_AW_ENGINE_VERSION}" "pydantic-clai2==${GH_AW_ENGINE_VERSION}" "pydantic-ai-slim[anthropic,openai,mcp,spec]>=2.54.0"
      # Logfire 4.39.0 is pydantic-ai-slim's compatibility floor. Install it only
      # when gh-aw supplies an OTLP endpoint, so other runs pay no installation cost.
      if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" ]; then
        python3 -P -m pip install --quiet --user --disable-pip-version-check "logfire>=4.39.0"
      fi
      "$HOME/.local/bin/clai2" --help
      python3 -P -c "from pydantic_ai_harness import Coder"
engine:
  id: pydantic-ai
  version: "0.54.0"
  display-name: Pydantic AI
  description: CLAI 2 headless runner for Pydantic AI agents with MCP tool support
  mcp: true
  provider:
    name: github
  behaviors:
    secret-strategy: universal-llm-consumer
    # Repository paths gh-aw treats as this engine's configuration: it protects them
    # from pull-request modification and derives the inline sub-agent and skill
    # directories from the first prefix (.pydantic-ai/agents, .pydantic-ai/skills).
    # The engine itself writes nothing into the checkout.
    manifest:
      files:
        - AGENTS.md
      path-prefixes:
        - .pydantic-ai/
    network:
      defaults:
        - host.docker.internal
        - github.com
        - raw.githubusercontent.com
        - api.github.com
        - objects.githubusercontent.com
        - pypi.org
        - files.pythonhosted.org
      provider-domains:
        copilot: api.githubcopilot.com
        anthropic: api.anthropic.com
        openai: api.openai.com
        codex: api.openai.com
    execution:
      command-name: clai2
      step-name: Execute Pydantic AI CLI
      model-env-var: PAI_MODEL
      write-timestamp: true
      provider-env-mode: universal-llm-consumer
    harness-script: |
      const { spawnSync } = require("child_process");
      const { randomBytes } = require("crypto");
      const { chmodSync, existsSync, mkdtempSync, writeFileSync } = require("fs");
      const { homedir, tmpdir } = require("os");
      const { join } = require("path");
      const { fetchAWFReflect, resolveProviderEndpointFromReflect, deriveBaseUrlFromModelsURL } = require("./awf_reflect.cjs");

      // gh-aw passes `execution.command-name` (or a workflow's `engine.command`)
      // first, then `execution.args`. The name is not spawned -- the CLI is started
      // by the interpreter that owns the install, see LAUNCHER below -- so only the
      // arguments after it are forwarded.
      const commandArgs = process.argv.slice(3);
      const log = message => process.stderr.write(`[pydantic-ai] ${message}\n`);

      // `clai2 -a` takes one target, either an import path or a JSON/YAML agent
      // spec, and the spec format resolves capability names through a closed
      // registry that the harness capabilities are not part of, so the coder
      // composition cannot be expressed as a spec. It is written as a Python
      // module instead: `Coder()` supplies six filesystem and shell tools,
      // repository context and context management.
      //
      // `Coder` acts on the run's workspace, and `LocalWorkspace(".")` makes that
      // the checkout: the launcher runs with the checkout as its working
      // directory. A local workspace hands commands only `PATH`, `HOME` and the
      // locale variables, which is not enough here: AWF's only egress is the
      // proxy named in `HTTPS_PROXY`, and `git commit` needs the identity gh-aw
      // sets in `GIT_AUTHOR_*`. So `env` passes the step's environment on, minus
      // the provider credential variables `Coder`'s shell has always withheld.
      // The AWF sandbox is the isolation boundary.
      //
      // The wrapper loads the gateway's MCP servers with the same public loader
      // and adds them as a dynamic toolset so agent-defined toolsets remain active.
      const AGENT_MODULE = `import importlib
      import json
      import os
      import sys
      from datetime import datetime, timezone
      from fnmatch import fnmatchcase
      from pathlib import Path

      import pydantic_core
      from pydantic_ai import Agent, RunContext
      from pydantic_ai.capabilities import LocalWorkspace
      from pydantic_ai.messages import (
          AgentStreamEvent,
          FunctionToolCallEvent,
          FunctionToolResultEvent,
          OutputToolCallEvent,
          OutputToolResultEvent,
          PartDeltaEvent,
          PartEndEvent,
          PartStartEvent,
          RetryPromptPart,
          TextPart,
          TextPartDelta,
          ThinkingPart,
          ThinkingPartDelta,
      )
      from pydantic_ai.toolsets import CombinedToolset
      from pydantic_ai.usage import RunUsage
      from pydantic_ai_harness import Coder
      from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

      _stdout = sys.__stdout__
      # Remove the transient key before agent code can start child processes.
      _frame_key = os.environ.pop('GH_AW_SESSION_FRAME_KEY')
      _run_id: str | None = None
      _sequence = 0
      _usage: RunUsage | None = None
      _session_id: str | None = None
      _model: str | None = None
      _started = False
      _finished = False
      _partial: dict[int, tuple[str, str]] = {}


      def _json_value(value: object) -> object:
          return pydantic_core.to_jsonable_python(value, bytes_mode='base64', inf_nan_mode='null')


      def _emit(event_type: str, data: dict[str, object]) -> None:
          global _sequence
          event: dict[str, object] = {
              'type': event_type,
              'data': data,
              'parentId': None,
              'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
          }
          if _run_id is not None:
              event['id'] = f'pydantic-ai-{_run_id}-{_sequence}'
              _sequence += 1
          if _session_id is not None:
              event['session_id'] = _session_id
          print(
              f'\\x1eGH-AW-SESSION/{_frame_key} ' + json.dumps(_json_value(event), separators=(',', ':')),
              file=_stdout,
              flush=True,
          )


      class _SessionRecorder:
          def observe(self, ctx: RunContext[object], event: AgentStreamEvent) -> None:
              global _run_id, _usage, _session_id, _model, _started
              if ctx.run_id is None:
                  return
              if _run_id is None:
                  _run_id = ctx.run_id
                  _session_id = ctx.conversation_id
              if ctx.run_id != _run_id:
                  return
              _usage = ctx.usage
              _model = ctx.model_id or ctx.model.model_name
              if not _started:
                  initial: dict[str, object] = {'sourceEngine': 'pydantic-ai', 'model': _model, 'cwd': os.getcwd()}
                  if _session_id is not None:
                      initial['sessionId'] = _session_id
                  _emit('session.init', initial)
                  if ctx.prompt is not None:
                      _emit('user.message', {'content': _json_value(ctx.prompt)})
                  _started = True

              if isinstance(event, PartStartEvent):
                  if isinstance(event.part, TextPart):
                      _partial[event.index] = ('assistant.message', event.part.content)
                  elif isinstance(event.part, ThinkingPart):
                      _partial[event.index] = ('assistant.reasoning', event.part.content)
                  else:
                      _partial.pop(event.index, None)
              elif isinstance(event, PartDeltaEvent):
                  delta = event.delta
                  if isinstance(delta, TextPartDelta):
                      _partial[event.index] = (
                          'assistant.message',
                          _partial.get(event.index, ('', ''))[1] + delta.content_delta,
                      )
                  elif isinstance(delta, ThinkingPartDelta):
                      if delta.content_delta is not None:
                          _partial[event.index] = (
                              'assistant.reasoning',
                              _partial.get(event.index, ('', ''))[1] + delta.content_delta,
                          )
              elif isinstance(event, PartEndEvent):
                  _partial.pop(event.index, None)
                  if isinstance(event.part, TextPart):
                      _emit('assistant.message', {'content': event.part.content})
                  elif isinstance(event.part, ThinkingPart):
                      _emit('assistant.reasoning', {'content': event.part.content})
              elif isinstance(event, (FunctionToolCallEvent, OutputToolCallEvent)):
                  input_value = _json_value(event.part.args)
                  if isinstance(event.part.args, str):
                      try:
                          input_value = json.loads(event.part.args)
                      except json.JSONDecodeError:
                          pass
                  data: dict[str, object] = {
                      'toolCallId': event.tool_call_id,
                      'toolName': event.part.tool_name,
                      'input': input_value,
                  }
                  if isinstance(event, OutputToolCallEvent):
                      data['sourceType'] = 'output_tool'
                  _emit('tool.execution_start', data)
              elif isinstance(event, (FunctionToolResultEvent, OutputToolResultEvent)):
                  part = event.part
                  if isinstance(part, RetryPromptPart):
                      success = False
                      status = 'retry'
                      output = None
                      error = _json_value(part.content)
                      tool_name = part.tool_name
                  else:
                      success = part.outcome == 'success'
                      status = part.outcome
                      output = _json_value(part.content)
                      error = None if success else _json_value(part.content)
                      tool_name = part.tool_name
                  data: dict[str, object] = {
                      'toolCallId': event.tool_call_id,
                      'toolName': tool_name,
                      'success': success,
                      'status': status,
                  }
                  if not isinstance(part, RetryPromptPart):
                      data['output'] = output
                  if error is not None:
                      data['error'] = error
                  if isinstance(event, OutputToolResultEvent):
                      data['sourceType'] = 'output_tool'
                  _emit('tool.execution_complete', data)

          def finish(self, exit_code: int) -> None:
              global _finished
              if _finished:
                  return
              _finished = True
              if _partial and exit_code != 0:
                  for event_type, content in _partial.values():
                      _emit(event_type, {'content': content, 'partial': True})
                  print('[pydantic-ai] preserving partial streamed content after failure', file=sys.stderr, flush=True)
              usage: dict[str, int] | None = None
              if _usage is not None:
                  usage = {'input_tokens': _usage.input_tokens, 'output_tokens': _usage.output_tokens}
                  if _usage.cache_write_tokens:
                      usage['cache_creation_input_tokens'] = _usage.cache_write_tokens
                  if _usage.cache_read_tokens:
                      usage['cache_read_input_tokens'] = _usage.cache_read_tokens
                  if _usage.cache_write_tokens or _usage.cache_read_tokens:
                      usage['input_tokens_include_cache'] = True
              result: dict[str, object] = {'status': 'success' if exit_code == 0 else 'failure', 'sourceType': 'pydantic-ai'}
              if usage is not None:
                  result['usage'] = usage
              _emit('session.result', result)


      _configured_agent = os.environ.get('PAI_AGENT')
      agent: Agent[object, object]
      if not _configured_agent:
          _env: dict[str, str] = {
              name: value
              for name, value in os.environ.items()
              if name != 'GH_AW_SESSION_FRAME_KEY'
              and not any(fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
          }
          agent = Agent(name='coder', capabilities=[LocalWorkspace('.', env=_env), Coder()])
      elif _configured_agent.lower().endswith(('.json', '.yaml', '.yml')):
          agent = Agent.from_file(_configured_agent)
      else:
          sys.path.insert(0, os.getcwd())
          _module_name, _separator, _attribute = _configured_agent.rpartition(':')
          if not _separator:
              _module_name, _separator, _attribute = _configured_agent.rpartition('.')
          if not _separator:
              raise ValueError(
                  f'PAI_AGENT expects MODULE:ATTRIBUTE, MODULE.ATTRIBUTE, or a JSON/YAML file; got {_configured_agent!r}'
              )
          agent = getattr(importlib.import_module(_module_name), _attribute)
          if not isinstance(agent, Agent):
              raise TypeError(f'{_configured_agent} does not refer to a pydantic_ai.Agent')

      _recorder = _SessionRecorder()
      agent.on_event(_recorder.observe)

      _mcp_config = os.environ.get('GH_AW_MCP_CONFIG')
      if _mcp_config and Path(_mcp_config).is_file():
          from pydantic_ai.mcp import load_mcp_toolsets

          _mcp_toolsets = load_mcp_toolsets(_mcp_config)

          @agent.toolset(per_run_step=False)
          def _gateway_toolset(ctx: RunContext[object]) -> CombinedToolset[object]:
              del ctx
              return CombinedToolset(_mcp_toolsets)
      `;
      const DEFAULT_AGENT = "gh_aw_agent:agent";

      // Preload the installed runner before an opt-in PAI_AGENT import adds the
      // checkout to sys.path. The wrapper is cached before CLAI 2 resolves -a,
      // so agent code imports once and import failures retain their traceback.
      const LAUNCHER = `import importlib
      import json
      import os
      import runpy
      import sys
      from datetime import datetime, timezone
      from pathlib import Path

      target, prompt_file, *cli_args = sys.argv[1:]
      frame_key = sys.stdin.readline().rstrip('\\n')
      module, separator, attribute = target.rpartition(':')
      original_stdout = sys.stdout
      stdout_sink = open(os.devnull, 'w')
      sys.stdout = stdout_sink
      exit_code = 0
      agent_module = None
      try:
          importlib.import_module('pydantic_clai2')
          if os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
              import atexit
              import tempfile
              import logfire

              # Ignore checkout configuration and credentials. Exporters shut down
              # before this private directory is removed.
              logfire_dir = tempfile.TemporaryDirectory(prefix='gh-aw-logfire-', dir='/tmp')
              atexit.register(logfire_dir.cleanup)
              logfire.configure(
                  send_to_logfire='if-token-present',
                  console=False,
                  distributed_tracing=True,
                  config_dir=logfire_dir.name,
                  data_dir=logfire_dir.name,
              )
              logfire.instrument_pydantic_ai()
              traceparent = os.environ.get('TRACEPARENT')
              if traceparent:
                  from opentelemetry.context import attach
                  from opentelemetry.propagate import extract

                  attach(extract({'traceparent': traceparent}))

          if not separator:
              raise ValueError(f'Expected MODULE:ATTRIBUTE, got {target!r}')
          os.environ['GH_AW_SESSION_FRAME_KEY'] = frame_key
          agent_module = importlib.import_module(module)
          loaded = getattr(agent_module, attribute)
          from pydantic_ai import Agent

          if not isinstance(loaded, Agent):
              raise TypeError(f'{target} is {type(loaded).__name__}, not pydantic_ai.Agent')
          # Read in-process: workflow context can exceed the OS argv limit.
          sys.argv = ['clai2', *cli_args, '-p', Path(prompt_file).read_text(encoding='utf-8')]
          runpy.run_module('pydantic_clai2', run_name='__main__', alter_sys=True)
      except SystemExit as exc:
          if isinstance(exc.code, int):
              exit_code = exc.code
          elif exc.code is not None:
              exit_code = 1
          raise
      except BaseException:
          exit_code = 1
          raise
      finally:
          sys.stdout = original_stdout
          try:
              result_emitted = False
              try:
                  recorder = getattr(agent_module, '_recorder', None)
                  if recorder is not None:
                      recorder.finish(exit_code)
                      result_emitted = True
              except Exception as exc:
                  print(f'[pydantic-ai] Unable to finish session recording: {exc}', file=sys.stderr)
              if not result_emitted:
                  event = {
                      'type': 'session.result',
                      'data': {'status': 'success' if exit_code == 0 else 'failure', 'sourceType': 'pydantic-ai'},
                      'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  }
                  try:
                      print(
                          '\\x1eGH-AW-SESSION/' + frame_key + ' ' + json.dumps(event),
                          file=original_stdout,
                          flush=True,
                      )
                  except Exception as exc:
                      print(f'[pydantic-ai] Unable to emit the session result: {exc}', file=sys.stderr)
          finally:
              stdout_sink.close()
      `;

      const main = async () => {
        const workspace = process.env.GITHUB_WORKSPACE;
        if (!workspace) throw new Error("GITHUB_WORKSPACE is required");
        const promptFile = process.env.GH_AW_PROMPT;
        if (!promptFile) throw new Error("GH_AW_PROMPT is required");

        // Neither the generated module nor the gateway's MCP config is written into
        // the checkout. A file committed at a path the engine reads is
        // repository-controlled input to a process that runs with the gateway's
        // credentials: an `mcp.json` there can name a stdio server for the CLI to
        // spawn, and a package there shadows an installed one for the whole run. The
        // module goes to a private directory created inside the sandbox; the config
        // adapter writes on the host into the `${RUNNER_TEMP}/gh-aw` tree that the
        // agent step mounts read-only, where gh-aw's own Claude and Codex converters
        // write theirs.
        //
        // The generated module wraps the default agent, an imported Agent, or a
        // JSON/YAML spec so each form gets identical recording and MCP behavior.
        const configuredAgent = process.env.PAI_AGENT;
        const agentTarget = DEFAULT_AGENT;
        const moduleDir = mkdtempSync(join(tmpdir(), "gh-aw-pydantic-ai-"));
        const agentModulePath = join(moduleDir, "gh_aw_agent.py");
        writeFileSync(agentModulePath, AGENT_MODULE, { mode: 0o600 });
        chmodSync(agentModulePath, 0o600);

        const env = { ...process.env };
        const frameKey = randomBytes(16).toString("hex");
        // Linux retains exec environment bytes in /proc even after unsetenv.
        // Deliver the key through stdin so tool subprocesses cannot recover it there.
        delete env.GH_AW_SESSION_FRAME_KEY;
        // `pip install --user` puts `clai2` here. The runner tool cache that holds
        // `uv` and the interpreter's own bin directory is under /opt, which the
        // sandbox exposes read-only, but the home directory is where the CLI and
        // its user site-packages actually live.
        //
        // Which interpreter owns those user site-packages matters: only the one
        // that ran the pre-agent `pip install --user` can import them, and the
        // sandbox prelude prepends every `bin` directory under the runner tool
        // cache (which caches several Python versions), so a bare `python3`
        // there resolves by `find` order rather than to the installing
        // interpreter. `actions/setup-python` names that one in `pythonLocation`;
        // putting its `bin` on PATH also gives the agent's own shell tool a
        // `python3` that can see the installed packages.
        const pythonBin = process.env.pythonLocation ? join(process.env.pythonLocation, "bin") : "";
        const python = pythonBin ? join(pythonBin, "python3") : "python3";
        env.PATH = [join(homedir(), ".local", "bin"), pythonBin, process.env.PATH || ""].filter(Boolean).join(":");
        // The module is reached through PYTHONPATH rather than by importing it as a
        // package, and prepending keeps a caller-supplied PYTHONPATH usable.
        //
        // The wrapper adds the checkout only when resolving an imported PAI_AGENT,
        // after trusted framework packages have loaded.
        env.PYTHONPATH = [moduleDir, process.env.PYTHONPATH || ""].filter(Boolean).join(":");
        if (process.env.OTEL_EXPORTER_OTLP_ENDPOINT) {
          // Traces-only backends return 404 noise for metrics and logs. A workflow
          // can override either default when its backend accepts those signals.
          if (!env.OTEL_METRICS_EXPORTER) env.OTEL_METRICS_EXPORTER = "none";
          if (!env.OTEL_LOGS_EXPORTER) env.OTEL_LOGS_EXPORTER = "none";
        }
        delete env.COPILOT_GITHUB_TOKEN;

        const provider = process.env.GH_AW_LLM_PROVIDER;
        const configuredBaseUrl = process.env.PAI_BASE_URL;

        // The client sends the model name verbatim, minus the provider marker that
        // selects one of its clients. The bare model ID reaches the api-proxy,
        // which steers to the configured provider by the port it is reached on, not
        // by a prefix in the model name: Copilot rejects `copilot/<model>` with
        // `model_not_supported`.
        // Only the first segment is the provider. Stripping greedily would eat an
        // org namespace out of ids like `meta-llama/Llama-3.1`, so this mirrors the
        // `SplitN(model, "/", 2)` gh-aw itself uses to read the provider off.
        if (!env.PAI_MODEL) throw new Error("PAI_MODEL is required");
        const modelProvider = env.PAI_MODEL.split("/", 1)[0].trim().toLowerCase();
        const requestedModel = env.PAI_MODEL.replace(/^[^/]*\//, "");
        // The api-proxy's Anthropic backend forwards the request path to
        // api.anthropic.com unchanged and rewrites Messages-shaped bodies; it does
        // not translate Chat Completions into Messages. So `anthropic/` is addressed
        // with the Messages API: `anthropic:` on `-m`, and ANTHROPIC_BASE_URL for
        // the endpoint. The Copilot and Codex backends are OpenAI-shaped and stay on
        // Chat Completions, and `PAI_BASE_URL` names a Chat Completions endpoint by
        // definition, so it keeps every provider there too.
        const useMessagesAPI = !configuredBaseUrl && modelProvider === "anthropic";
        // The dotted-alias rewrite describes the api-proxy's Copilot backend,
        // which publishes Copilot's Claude models under dotted IDs. Every other
        // destination (the anthropic and openai backends, or an endpoint named
        // by PAI_BASE_URL) gets the id the workflow wrote: a model actually
        // called `claude-sonnet-4-5` there has to arrive as that.
        const model = !configuredBaseUrl && modelProvider === "copilot"
          ? requestedModel.replace(/^(claude-(?:haiku|sonnet|opus)-\d+)-(\d+)$/, "$1.$2")
          : requestedModel;

        // `PAI_BASE_URL` points the engine at an OpenAI-compatible endpoint of the
        // workflow's choosing instead of the AWF api-proxy. Two constraints shape
        // it.
        //
        // It has to be a variable of this definition's own, because AWF sets the
        // backend's own base URL variable on this step itself (OPENAI_BASE_URL, or
        // ANTHROPIC_BASE_URL for the anthropic backend), pointing at the api-proxy
        // on host.docker.internal whenever the firewall is enabled, so its presence
        // cannot carry the workflow's intent, and reading it as intent is what
        // made the pre-#52843 definition pick the wrong endpoint.
        //
        // There is deliberately no matching key knob. gh-aw excludes any
        // `engine.env` value holding a secret from the agent sandbox
        // (`awf --exclude-env`), so a credential cannot be delivered here at all
        // and the API key below stays the placeholder. The endpoint therefore
        // has to accept that placeholder, or be fronted by something upstream of
        // the agent that adds the real credential.
        let baseUrl = configuredBaseUrl || (useMessagesAPI ? process.env.ANTHROPIC_BASE_URL : process.env.OPENAI_BASE_URL);
        if (!configuredBaseUrl) {
          // Only /reflect discovery needs the provider: it selects which of the
          // api-proxy's configured endpoints to use. A caller-supplied base URL
          // names the endpoint outright, so demanding a provider alongside it
          // would reject a complete configuration.
          if (!provider) throw new Error("GH_AW_LLM_PROVIDER is required");
          if (process.env.AWF_REFLECT_ENABLED === "1") {
            const result = await fetchAWFReflect({ logger: log });
            if (!result.ok || !result.reflectData) {
              throw new Error(`Unable to discover the Pydantic AI LLM endpoint from /reflect: ${result.reason || "empty response"}`);
            }
            const endpoint = resolveProviderEndpointFromReflect({
              provider,
              reflectData: result.reflectData,
              logger: log,
            });
            if (!endpoint?.baseUrl) {
              throw new Error(`No configured /reflect endpoint found for provider ${provider}`);
            }
            baseUrl = endpoint.baseUrl;
            const reflectedEndpoint = result.reflectData.endpoints?.find(
              entry => entry?.configured === true && entry.provider === endpoint.endpointProvider
            );
            if (!useMessagesAPI && typeof reflectedEndpoint?.models_url === "string") {
              // `endpoint.baseUrl` is the models-listing origin, while the
              // OpenAI-compatible client posts to `<base>/chat/completions`, so the
              // path prefix carried by models_url (`/v1` on some providers) has to
              // come along. This helper applies the same api-proxy ->
              // host.docker.internal rewrite.
              //
              // The Anthropic client keeps the origin instead: it appends
              // `/v1/messages` itself, so carrying the prefix over would post to
              // `/v1/v1/messages`.
              baseUrl = deriveBaseUrlFromModelsURL(reflectedEndpoint.models_url);
            }
          }
        }
        if (!baseUrl) {
          throw new Error(
            `Pydantic AI requires AWF endpoint discovery, PAI_BASE_URL or ${useMessagesAPI ? "ANTHROPIC_BASE_URL" : "OPENAI_BASE_URL"}`
          );
        }
        // The AWF api-proxy injects the real upstream credentials and ignores the
        // inbound key, but neither client constructs itself without one. Setting it
        // also replaces whatever key this step inherited, so the agent process holds
        // the placeholder rather than a provider credential.
        if (useMessagesAPI) {
          env.ANTHROPIC_BASE_URL = baseUrl;
          env.ANTHROPIC_API_KEY = "awf-anthropic-proxy";
        } else {
          env.OPENAI_BASE_URL = baseUrl;
          env.OPENAI_API_KEY = "awf-copilot-proxy";
        }

        // `-m` is always passed: the composed agent carries no model, and without
        // the flag the CLI falls back to its own default,
        // billing a model the workflow never asked for. gh-aw validates
        // `provider/model` at compile time, so PAI_MODEL is set for every compiled
        // workflow, and the throw above covers any other invocation.
        //
        // An explicit `-m` also replaces the model a loaded agent declares, so a
        // `PAI_AGENT` agent runs on the workflow's `engine.model` whatever it was
        // constructed with. That is what routes it through the endpoint above.
        const cliArgs = [
          ...commandArgs,
          "-a", agentTarget,
          "-m", `${useMessagesAPI ? "anthropic" : "openai-chat"}:${model}`,
        ];
        // The config adapter writes this file only for a workflow that configures
        // MCP tools, so its
        // absence has to mean "no servers" rather than an error. The
        // `RUNNER_TEMP || "/tmp"` fallback is the one gh-aw's own converters use, and
        // the adapter resolves this path by the same expression.
        const mcpConfig = join(process.env.RUNNER_TEMP || "/tmp", "gh-aw", "mcp-config", "mcp-servers.json");
        delete env.GH_AW_MCP_CONFIG;
        if (existsSync(mcpConfig)) env.GH_AW_MCP_CONFIG = mcpConfig;
        // Log only the origin because endpoint userinfo and query parameters can
        // contain credentials, and workflow run logs are not private.
        const otlpEndpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
        let otlpOrigin = "";
        if (otlpEndpoint) {
          try {
            otlpOrigin = new URL(otlpEndpoint).origin;
          } catch {
            otlpOrigin = "(unparsable)";
          }
        }
        log(
          `provider=${configuredBaseUrl ? "(PAI_BASE_URL)" : provider} model=${model} baseUrl=${baseUrl}` +
            (configuredAgent ? ` agent=${configuredAgent}` : "") +
            (otlpOrigin ? ` otlp=${otlpOrigin}` : "")
        );
        // The target is passed twice on purpose: once for LAUNCHER, which imports it
        // and hands the CLI a module already in sys.modules, and once as the `-a`
        // the CLI parses for itself.
        process.stdout.write(`\x1eGH-AW-SESSION-KEY:${frameKey}\x1e\n`);
        const result = spawnSync(python, ["-P", "-c", LAUNCHER, agentTarget, promptFile, ...cliArgs], {
          cwd: workspace, env, input: `${frameKey}\n`, stdio: ["pipe", "inherit", "inherit"],
        });
        if (result.error) throw result.error;
        if (result.status !== 0) {
          const error = new Error(`Pydantic AI execution failed with exit code ${result.status ?? "unknown"}`);
          // Surface the child's own status so the step fails with the same code.
          error.exitCode = typeof result.status === "number" && result.status !== 0 ? result.status : 1;
          throw error;
        }
      };

      main().catch(error => {
        log(error instanceof Error ? error.message : String(error));
        process.exitCode = typeof error?.exitCode === "number" && error.exitCode !== 0 ? error.exitCode : 1;
      });
    mcp:
      config-path: ${RUNNER_TEMP}/gh-aw/mcp-config/mcp-servers.json
      config-adapter: |
        // Renders the MCP gateway's configuration as the Claude-style
        // `mcpServers` document that `pydantic_ai.mcp.load_mcp_toolsets` reads,
        // which the wrapper loads as dynamic toolsets. Only HTTP entries
        // are carried: `load_mcp_toolsets` can host stdio
        // servers too, but the gateway already fronts every configured server
        // over HTTP, and CLI-mounted servers are excluded because the agent
        // reaches those as executables on PATH instead.
        const fs = require("fs");
        const path = require("path");

        const requireEnvVar = name => {
          const value = process.env[name];
          if (!value) throw new Error(`${name} environment variable is required`);
          return value;
        };

        const gatewayOutputPath = requireEnvVar("MCP_GATEWAY_OUTPUT");
        const gatewayDomain = process.env.MCP_GATEWAY_DOMAIN || "host.docker.internal";
        const gatewayPort = requireEnvVar("MCP_GATEWAY_PORT");
        const gatewayURL = `http://${gatewayDomain}:${gatewayPort}`;

        let cliServers;
        try {
          cliServers = new Set(JSON.parse(process.env.GH_AW_MCP_CLI_SERVERS || "[]"));
        } catch (error) {
          throw new Error(`Failed to parse GH_AW_MCP_CLI_SERVERS: ${error instanceof Error ? error.message : String(error)}`);
        }

        const gatewayOutput = JSON.parse(fs.readFileSync(gatewayOutputPath, "utf8"));
        const rawServers = gatewayOutput.mcpServers;
        const servers = rawServers && typeof rawServers === "object" && !Array.isArray(rawServers) ? rawServers : {};

        const mcpServers = {};
        for (const [name, entry] of Object.entries(servers)) {
          if (cliServers.has(name) || !entry || typeof entry !== "object") continue;
          if (typeof entry.url !== "string") {
            console.log(`Skipping MCP server ${name}: the Pydantic AI engine only supports HTTP MCP servers`);
            continue;
          }
          const server = { url: entry.url.replace(/^http:\/\/[^/]+\/mcp\//, `${gatewayURL}/mcp/`) };
          if (entry.headers && typeof entry.headers === "object") server.headers = entry.headers;
          mcpServers[name] = server;
        }

        // This script runs on the host runner, in the Start MCP Gateway step, so it
        // writes where that step already created a directory and where the agent step
        // mounts `${RUNNER_TEMP}/gh-aw` read-only -- the same file the built-in Claude
        // converter produces, which is also the path gh-aw's log redaction scans for
        // the gateway bearer token. The harness script resolves it by the same
        // expression. Keeping it out of the checkout is what stops a committed
        // `mcp.json` from reaching the wrapper; see the harness script.
        const configPath = path.join(process.env.RUNNER_TEMP || "/tmp", "gh-aw", "mcp-config", "mcp-servers.json");
        fs.mkdirSync(path.dirname(configPath), { recursive: true, mode: 0o700 });
        fs.writeFileSync(configPath, JSON.stringify({ mcpServers }, null, 2), { mode: 0o600 });
        fs.chmodSync(configPath, 0o600);
        console.log(`Wrote ${Object.keys(mcpServers).length} MCP server(s) to ${configPath}`);
    log-parser: |
      const { collectAddMaskedValues, redactArtifactMaskedValues } = require('./add_mask_redaction.cjs');

      function parseLog(logContent) {
        const lines = logContent.split('\n');
        const canonicalTypes = new Set([
          'session.init', 'user.message', 'assistant.message', 'assistant.reasoning',
          'tool.execution_start', 'tool.execution_complete', 'session.result',
        ]);
        const header = lines.find(line => /^\x1eGH-AW-SESSION-KEY:[0-9a-f]{32}\x1e$/.test(line));
        const frameKey = header?.match(/^\x1eGH-AW-SESSION-KEY:([0-9a-f]{32})\x1e$/)?.[1];
        const records = [];
        if (frameKey) {
          for (const line of lines) {
            const frame = line.match(/^\x1eGH-AW-SESSION\/([0-9a-f]{32}) (\{.*\})$/);
            if (!frame || frame[1] !== frameKey) continue;
            try {
              const event = JSON.parse(frame[2]);
              if (event && canonicalTypes.has(event.type) && event.data && typeof event.data === 'object' && !Array.isArray(event.data)) {
                records.push(event);
              }
            } catch { /* a malformed record does not discard adjacent events */ }
          }
        }
        const logEntries = JSON.parse(redactArtifactMaskedValues(JSON.stringify(records), collectAddMaskedValues(logContent)));
        const result = logEntries.findLast(event => event.type === 'session.result');
        const usage = result?.data?.usage || {};
        const parts = [];
        if (result?.data?.status) parts.push(`**Status:** ${result.data.status}`);
        if (logEntries.length) {
          const toolCalls = logEntries.filter(event => event.type === 'tool.execution_start').length;
          parts.push(`**Tool calls:** ${toolCalls}`);
        }
        if (usage.input_tokens !== undefined || usage.output_tokens !== undefined) {
          parts.push(`**Tokens:** ${((usage.input_tokens || 0) + (usage.output_tokens || 0)).toLocaleString()}`);
        }
        return { markdown: parts.join(' · ') || 'No recorded agent events.', logEntries, mcpFailures: [], maxTurnsHit: false };
      }
---

<!--
# Pydantic AI

Shared engine definition for CLAI 2 headless execution. Import this file and set
`engine: id: pydantic-ai`; gh-aw v0.91.1 or newer is required for canonical
session capture and schema-valid merged usage.

```yaml
imports:
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: copilot/claude-sonnet-4-5
```

The private wrapper supplies the default `Coder`/`LocalWorkspace` composition or
resolves `PAI_AGENT` once from an imported Agent or JSON/YAML spec. Gateway MCP
tools are added alongside existing tools. The workflow's model and proxy routing
apply to all target forms.

The recorder captures typed assistant, reasoning, tool-call, tool-result, and
usage events. The parser selects its framed records; gh-aw writes
`agent-session.jsonl` and collects `usage/aw_session.jsonl`. Import failures and
interrupted runs keep their actual failure outcome, without inferred turns or
tool completions. See `README.md` for setup, credentials, and observability.
-->
