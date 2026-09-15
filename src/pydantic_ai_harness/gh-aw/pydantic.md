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
      # 2.36.0 is the first pydantic-ai-slim release carrying `pai --mcp-config`,
      # which is how the gateway's MCP servers reach the agent.
      #
      # The anthropic extra is what an `anthropic/` model runs on: that backend of
      # the api-proxy serves the Messages API, not Chat Completions.
      python3 -P -m pip install --quiet --user --disable-pip-version-check "pydantic-ai-harness[cli]==$GH_AW_ENGINE_VERSION" "pydantic-ai-slim[anthropic,openai,mcp]>=2.36.0"
      "$HOME/.local/bin/pai" --version
      python3 -P -c "from pydantic_ai_harness import Coder"
engine:
  id: pydantic-ai
  version: "0.31.0"
  display-name: Pydantic AI
  description: Pydantic AI CLI (pai) running the pydantic-ai-harness coder agent with MCP tool support
  experimental: true
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
      command-name: pai
      step-name: Execute Pydantic AI CLI
      model-env-var: PAI_MODEL
      write-timestamp: true
      provider-env-mode: universal-llm-consumer
    harness-script: |
      const { spawnSync } = require("child_process");
      const { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } = require("fs");
      const { homedir, tmpdir } = require("os");
      const { join } = require("path");
      const { fetchAWFReflect, resolveProviderEndpointFromReflect, deriveBaseUrlFromModelsURL } = require("./awf_reflect.cjs");

      // gh-aw passes `execution.command-name` (or a workflow's `engine.command`)
      // first, then `execution.args`. The name is not spawned -- the CLI is started
      // by the interpreter that owns the install, see LAUNCHER below -- so only the
      // arguments after it are forwarded.
      const commandArgs = process.argv.slice(3);
      const log = message => process.stderr.write(`[pydantic-ai] ${message}\n`);

      // `pai -a` takes one target, either an import path or a JSON/YAML agent
      // spec, and the spec format resolves capability names through a closed
      // registry that the harness capabilities are not part of, so the coder
      // composition cannot be expressed as a spec. It is written as a Python
      // module instead: `Coder()` supplies the filesystem, shell, planning and
      // sub-agent tools.
      //
      // The gateway's MCP servers are deliberately not part of the module.
      // `pai --mcp-config` reads the same Claude-shaped config file through the
      // same `pydantic_ai.mcp.load_mcp_toolsets`, `${VAR}` expansion included, so
      // routing them through the CLI is what lets a `PAI_AGENT` agent receive
      // them on identical terms.
      const AGENT_MODULE = `from pydantic_ai import Agent
      from pydantic_ai_harness import Coder

      agent = Agent(name="coder", capabilities=[Coder()])
      `;
      const DEFAULT_AGENT = "gh_aw_agent:agent";

      // The CLI runs inside the interpreter that owns the install rather than as a
      // separate `pai` process, so the agent module is imported once, in the process
      // that runs it. Three things follow from that.
      //
      // `pai` reduces any failed `-a` load to one line naming the target, so an
      // agent that raises on import would reach the step log without its traceback.
      // The import here happens before the CLI starts, and an unhandled exception is
      // the step's failure, traceback included.
      //
      // `pydantic_ai._cli.load_agent` prepends the working directory -- the checkout
      // -- to sys.path before resolving the target, ahead of PYTHONPATH. A
      // repository file named `gh_aw_agent.py` would therefore be loaded in place of
      // the generated module. Importing the target first settles which file the name
      // means, because an import of a module already in sys.modules does not search
      // the path again.
      //
      // A separate preflight process could do neither: it would import the module in
      // one interpreter and leave the CLI to import it again in another, running any
      // module-level work in the agent twice.
      //
      // The residual is `load_agent`'s insert itself: everything the agent imports
      // after that point still sees the checkout first on sys.path. That is the
      // CLI's documented behavior for its own users, and not something this file
      // can change from the outside.
      //
      // `-P` keeps the `-c` invocation from putting the working directory on
      // sys.path on its own account; PYTHONPATH below is what makes the agent
      // importable. A spec file, and the dotted `module.attribute` form the CLI also
      // accepts, are left to the CLI as before.
      const LAUNCHER = `import runpy
      import sys

      target, *cli_args = sys.argv[1:]
      module, separator, attribute = target.rpartition(":")
      if separator and not target.lower().endswith((".yml", ".yaml", ".json")):
          import importlib

          from pydantic_ai import Agent

          loaded = getattr(importlib.import_module(module), attribute)
          if not isinstance(loaded, Agent):
              raise TypeError(f"{target} is {type(loaded).__name__}, not pydantic_ai.Agent")

      sys.argv = ["pai", *cli_args]
      runpy.run_module("pydantic_ai", run_name="__main__", alter_sys=True)
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
        // `PAI_AGENT` runs an agent the repository defines, in whichever form
        // `pai -a` accepts. The generated module is not written in that case:
        // nothing would load it, and a stale copy on disk is worse than none.
        const configuredAgent = process.env.PAI_AGENT;
        const agentTarget = configuredAgent || DEFAULT_AGENT;
        const moduleDir = configuredAgent ? "" : mkdtempSync(join(tmpdir(), "gh-aw-pydantic-ai-"));
        if (moduleDir) {
          const agentModulePath = join(moduleDir, "gh_aw_agent.py");
          writeFileSync(agentModulePath, AGENT_MODULE, { mode: 0o600 });
          chmodSync(agentModulePath, 0o600);
        }

        const env = { ...process.env };
        // `pip install --user` puts `pai` here. The runner tool cache that holds
        // `uv` and the interpreter's own bin directory is under /opt, which the
        // sandbox exposes read-only, but the home directory is where the CLI and
        // its user site-packages actually live.
        //
        // Which interpreter owns those user site-packages matters: only the one
        // that ran the pre-agent `pip install --user` can import them, and the
        // sandbox prelude prepends every `bin` directory under the runner tool
        // cache — which caches several Python versions — so a bare `python3`
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
        // The checkout itself joins the path only under `PAI_AGENT`. That is the
        // opt-in: it makes repository code importable, which is the whole point
        // of running your own agent, and it is exactly what `-P` on the install
        // step keeps off the path for the default composition.
        env.PYTHONPATH = [moduleDir, configuredAgent ? workspace : "", process.env.PYTHONPATH || ""].filter(Boolean).join(":");
        delete env.COPILOT_GITHUB_TOKEN;

        const provider = process.env.GH_AW_LLM_PROVIDER;
        const configuredBaseUrl = process.env.PAI_BASE_URL;

        // `pai` sends the model name verbatim, minus the provider marker that
        // selects one of its clients, so the bare model ID reaches the api-proxy —
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
        // destination — the anthropic and openai backends, or an endpoint named
        // by PAI_BASE_URL — gets the id the workflow wrote: a model actually
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
              // come along — and this helper applies the same api-proxy ->
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
        // the flag `pai` silently falls back to its own `openai:gpt-5` default,
        // billing a model the workflow never asked for. gh-aw validates
        // `provider/model` at compile time, so PAI_MODEL is set for every compiled
        // workflow, and the throw above covers any other invocation.
        //
        // An explicit `-m` also replaces the model a loaded agent declares, so a
        // `PAI_AGENT` agent runs on the workflow's `engine.model` whatever it was
        // constructed with. That is what routes it through the endpoint above.
        const cliArgs = [...commandArgs, "-a", agentTarget];
        // The config adapter writes this file only for a workflow that configures
        // MCP tools, and `--mcp-config` fails on a path that is not there, so its
        // absence has to mean "no servers" rather than an error. The
        // `RUNNER_TEMP || "/tmp"` fallback is the one gh-aw's own converters use, and
        // the adapter resolves this path by the same expression.
        const mcpConfig = join(process.env.RUNNER_TEMP || "/tmp", "gh-aw", "mcp-config", "mcp-servers.json");
        if (existsSync(mcpConfig)) cliArgs.push("--mcp-config", mcpConfig);
        cliArgs.push("-m", `${useMessagesAPI ? "anthropic" : "openai-chat"}:${model}`, readFileSync(promptFile, "utf8"));
        log(
          `provider=${configuredBaseUrl ? "(PAI_BASE_URL)" : provider} model=${model} baseUrl=${baseUrl}` +
            (configuredAgent ? ` agent=${configuredAgent}` : "")
        );
        // The target is passed twice on purpose: once for LAUNCHER, which imports it
        // and hands the CLI a module already in sys.modules, and once as the `-a`
        // the CLI parses for itself.
        const result = spawnSync(python, ["-P", "-c", LAUNCHER, agentTarget, ...cliArgs], { cwd: workspace, env, stdio: "inherit" });
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
        // which the harness script hands to `pai --mcp-config`. Only HTTP entries
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
        // `mcp.json` from reaching `pai --mcp-config`; see the harness script.
        const configPath = path.join(process.env.RUNNER_TEMP || "/tmp", "gh-aw", "mcp-config", "mcp-servers.json");
        fs.mkdirSync(path.dirname(configPath), { recursive: true, mode: 0o700 });
        fs.writeFileSync(configPath, JSON.stringify({ mcpServers }, null, 2), { mode: 0o600 });
        fs.chmodSync(configPath, 0o600);
        console.log(`Wrote ${Object.keys(mcpServers).length} MCP server(s) to ${configPath}`);
    log-parser: |
      function parseLog(logContent) {
        const lines = logContent.split("\n");
        const logEntries = [];
        const mcpFailures = [];
        let maxTurnsHit = false;
        const AWF_INFRA_RE = /^\[(INFO|WARN|SUCCESS|ERROR|entrypoint|health-check|pydantic-ai)\]|^ (?:Container|Network|Volume) |^Process exiting with code:/;
        let inputTokens = 0;
        let outputTokens = 0;
        let toolCallIndex = 0;
        let turnCount = 0;
        let pendingText = [];

        function flushText() {
          if (pendingText.length === 0) return;
          const text = pendingText.join("\n").trim();
          if (text) {
            logEntries.push({ type: "assistant", message: { content: [{ type: "text", text }] } });
            turnCount++;
          }
          pendingText = [];
        }

        logEntries.push({ type: "system", subtype: "init", model: null, session_id: null });

        for (const line of lines) {
          if (!line.trim()) continue;
          if (AWF_INFRA_RE.test(line)) continue;
          if (/max.?turns|maximum.*turns.*reached|turn limit/i.test(line)) maxTurnsHit = true;
          if (/MCP server .* failed|MCP.*connection.*error|Failed to connect to MCP/i.test(line)) {
            const serverMatch = line.match(/MCP server ['"]?([^\s'"]+)['"]?/i);
            mcpFailures.push(serverMatch ? serverMatch[1] : line.trim());
          }

          let parsed = null;
          try {
            if (line.trim().startsWith("{")) parsed = JSON.parse(line.trim());
          } catch (e) { /* not JSON */ }

          if (parsed) {
            if (parsed.input_tokens) inputTokens += parsed.input_tokens;
            if (parsed.output_tokens) outputTokens += parsed.output_tokens;
            const entryType = parsed.type != null ? String(parsed.type) : "log";
            const msg = parsed.msg || parsed.message || parsed.content || "";

            if (/tool[._]call|tool[._]use/i.test(entryType)) {
              flushText();
              const toolId = `pai_tool_${toolCallIndex++}`;
              const toolName = parsed.tool || parsed.name || entryType;
              logEntries.push({ type: "assistant", message: { content: [{ type: "tool_use", id: toolId, name: toolName, input: {} }] } });
              logEntries.push({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: toolId, content: msg }] } });
            } else if (msg) {
              pendingText.push(msg);
            } else if (!parsed.input_tokens && !parsed.output_tokens) {
              // A JSON line carrying none of the text fields is still assistant output --
              // a reply that is bare JSON, say -- so it is kept as written. A usage record
              // is not: its numbers were just added to the totals.
              pendingText.push(line.trim());
            }
          } else {
            pendingText.push(line.trim());
          }
        }
        flushText();

        const usage = {};
        if (inputTokens) usage.input_tokens = inputTokens;
        if (outputTokens) usage.output_tokens = outputTokens;
        logEntries.push({ type: "result", num_turns: turnCount, usage });
        const parts = [`**Turns:** ${turnCount}`, `**Tool calls:** ${toolCallIndex}`];
        if (inputTokens || outputTokens) parts.push(`**Tokens:** ${((inputTokens ?? 0) + (outputTokens ?? 0)).toLocaleString()}`);
        if (mcpFailures.length) parts.push(`**MCP failures:** ${mcpFailures.length}`);
        if (maxTurnsHit) parts.push("**Max turns reached**");
        return { markdown: parts.join(" · "), logEntries, mcpFailures, maxTurnsHit };
      }
---

<!--
# Pydantic AI

Shared engine definition for the [Pydantic AI](https://ai.pydantic.dev) CLI
(`pai`), running the coder agent from
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness). Import
this file and set `engine: id: pydantic-ai` to use it:

```yaml
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: copilot/claude-sonnet-4-5
```

The agent is a `pydantic_ai.Agent` composed from the harness `Coder`
capability — filesystem, shell, planning, repository context and an explorer
sub-agent, with the harness's own context-management guardrails. `pai -a` accepts
a single target and its JSON agent-spec format cannot name harness capabilities,
so the harness script writes that composition as `gh_aw_agent.py` in a private
directory it creates inside the sandbox, puts that directory on `PYTHONPATH`, and
passes `-a gh_aw_agent:agent`. The module is deliberately not written into the
checkout: a directory the engine puts on `PYTHONPATH` would otherwise let a
package committed to the repository shadow an installed one for the whole run.

The CLI is started by the interpreter that owns the install -- `python -P -c`
importing the target and then `runpy.run_module("pydantic_ai")` -- rather than as
a separate `pai` process. The module is imported exactly once, in the process
that runs it: an agent that raises on import fails the step with its traceback
rather than the one line `pai` prints for a failed `-a` load, and the CLI's own
`load_agent`, which prepends the checkout to `sys.path` before resolving the
target, finds the module already in `sys.modules` instead of a repository file of
the same name. That insert still applies to everything imported after it, which
is the CLI's documented behavior for its own users.

`PAI_AGENT` in `engine.env` replaces that target with an agent the repository
defines, in the same `module:variable` or spec-file form `pai -a` takes. The
generated module is then not written, and `GITHUB_WORKSPACE` joins `PYTHONPATH` so
a module in the repository imports. That is opt-in because it puts repository code
on the import path. `-m` is still passed, so the agent runs on the workflow's
`engine.model` rather than any model it was constructed with. See `README.md` next
to this file.

MCP servers are rendered into `${RUNNER_TEMP}/gh-aw/mcp-config/mcp-servers.json` in
the same `mcpServers` shape Claude Desktop and Cursor use -- written on the host
runner, into the tree the agent step mounts read-only, so a file committed to the
repository cannot stand in for it -- and reach the agent through
`pai --mcp-config`, which loads them with `pydantic_ai.mcp.load_mcp_toolsets`
(including `${VAR}` expansion of header values) and passes the toolsets into the
run. Routing them through the CLI rather than the generated module is what gives a
`PAI_AGENT` agent the same servers. That flag arrived in pydantic-ai 2.36.0, which
is the floor on the install line. Tools are prefixed with their server name, so
safe outputs are reachable as `safeoutputs_create_issue` and the like. Only HTTP
servers are carried over; CLI-mounted servers stay available to the agent's shell
as executables on `PATH`.

`model` must use `provider/model` format. The provider segment selects which of
the AWF api-proxy's backends handles the request; `copilot`, `anthropic`,
`openai` and `codex` are the values gh-aw accepts. Requests are routed through
that proxy, whose endpoint is discovered from `/reflect` at run time, so the
first segment is dropped and the rest of the model ID is passed with `-m`, under
the marker for the wire API that backend serves: `anthropic:<model>` against
`ANTHROPIC_BASE_URL` for `anthropic/`, whose backend forwards the path to
api.anthropic.com unchanged and does not translate Chat Completions into
Messages, and `openai-chat:<model>` against `OPENAI_BASE_URL` for the rest. The
marker selects a Pydantic AI client and is not part of the model name sent
upstream. The two base URLs differ by a segment: the Anthropic client appends
`/v1/messages` to the endpoint's origin, the OpenAI-compatible client appends
`/chat/completions` to the `/v1` prefix the reflected `models_url` carries.
Only the first segment of the model goes, so an ID carrying an org namespace such
as `openai/meta-llama/Llama-3.1` keeps it. When the provider segment is
`copilot`, Claude aliases such as `claude-sonnet-4-5` are normalized to the dotted
model IDs the proxy's Copilot backend exposes, such as `claude-sonnet-4.5`; every
other destination — the `anthropic` and `openai` backends, or a `PAI_BASE_URL`
endpoint — receives the ID as written. `-m` is always passed, because a workflow
that declares no model would otherwise inherit the CLI's own `openai:gpt-5`
default silently.

Setting `PAI_BASE_URL` in `engine.env` sends requests to that URL instead of the
proxy, for any endpoint speaking the OpenAI Chat Completions API. `/reflect`
discovery is skipped and the provider segment of `model` becomes a formality --
including `anthropic/`, which stays on Chat Completions under `PAI_BASE_URL` --
so write `openai/<model-id>` and the bare ID reaches the endpoint. There is no
matching key setting: gh-aw keeps `engine.env` values holding secrets out of the
agent sandbox, so the endpoint has to accept the placeholder bearer token or sit
behind something that adds the real credential. See `README.md` next to this file
for the whole picture.

Responses are streamed. The proxy's aggregated non-streaming body omits
`object` and `choices[].index`, which Pydantic AI rejects during response
validation, so `--no-stream` is deliberately not passed.

`pai` renders its output as Markdown for a terminal and has no structured output
mode today, so the log parser reconstructs turns from that text and reads token
counts only from any JSON lines the run happens to emit.

The CLI and the coder capabilities are installed before the agent runs with
`pip install --user "pydantic-ai-harness[cli]==<engine version>"
"pydantic-ai-slim[anthropic,openai,mcp]>=2.36.0"`, into `~/.local` because the
runner tool cache holding `uv` is not writable from inside the sandbox.
-->
