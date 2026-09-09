# The `pydantic-ai` engine for GitHub Agentic Workflows

`pydantic.md` in this directory is the engine definition that
[GitHub Agentic Workflows](https://github.com/github/gh-aw) (gh-aw) uses for its
`pydantic-ai` engine. It is what gh-aw consumes, read from this repository's `main`
branch: there is no separate published copy, and a merge here reaches every workflow
that recompiles afterwards.

A workflow has to write the `imports:` line itself. gh-aw's engine catalog maps
the `pydantic-ai` id to this path, but only to suggest it: naming the engine
without the import fails to compile with a tip carrying the line to add.

The engine runs the [Pydantic AI](https://ai.pydantic.dev) CLI (`pai`) over an agent
composed from this package's `Coder` capability: filesystem, shell, planning,
repository context and an explorer sub-agent, plus one toolset per MCP server the
gh-aw gateway exposes.

## Quick start

```yaml
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: copilot/claude-sonnet-4-5
---

# Triage

Read the issue and summarize what changed.
```

To freeze the definition instead, import a commit SHA that contains this file, or a
release tag of this repository cut after the definition landed on `main`; gh-aw
resolves the ref at compile time, so `@main` re-resolves on every recompile while a
tag or SHA does not. The ref has to be one that carries the file: `v0.26.0` and
every earlier tag predate it, so an import naming one has nothing to fetch. The two
pins are separate: the ref decides which definition compiles, and `engine.version`
inside it decides which harness release runs. A workflow's own `engine: version:`
overrides the package version the definition pins, because gh-aw applies the
imported definition's version only when the workflow left it empty
(`applyEngineImportDefaults` in `pkg/workflow/compiler_orchestrator_engine.go`).

`model` is required and must be `provider/model`. gh-aw accepts `copilot`,
`anthropic`, `openai` and `codex` as the provider segment; it selects which backend
of the AWF api-proxy serves the request. The proxy holds the credentials, so a
workflow supplies none. `anthropic/` is served over the Anthropic Messages API,
because that backend forwards the request path to `api.anthropic.com` unchanged and
does not translate Chat Completions into Messages; the other three are
OpenAI-shaped and use Chat Completions.

## What actually runs

`pai -a` takes one target and its agent-spec format cannot name harness
capabilities, so the engine writes the composition as `gh_aw_agent.py` in a
private directory it creates inside the sandbox, puts that directory on
`PYTHONPATH`, and passes `-a gh_aw_agent:agent`. The CLI and its
dependencies are installed before the agent starts, with
`pip install --user "pydantic-ai-harness[cli]==<engine version>"
"pydantic-ai-slim[anthropic,openai,mcp]>=2.36.0"`. The pinned harness version is
`engine.version` in `pydantic.md`, and it always names a published release: lint
refuses a pull request whose pin is not on PyPI. The `2.36.0` floor is the first
pydantic-ai release carrying `pai --mcp-config`, and the `anthropic` extra is what
an `anthropic/` model runs on.

The CLI itself is started by the interpreter that owns that install, which imports
the agent target and then runs `pydantic_ai` as `__main__`, rather than by spawning
`pai`. The agent module is therefore imported exactly once, in the process that
runs it. Two things follow. An agent that raises on import fails the step with its
traceback, instead of the single line `pai` prints for a failed `-a` load. And
`load_agent`, which prepends the checkout to `sys.path` before it resolves the
target, finds the module already in `sys.modules`, so a repository file named
`gh_aw_agent.py` cannot stand in for the generated one. That insert still applies to
everything imported after it, which is how the CLI behaves for all of its users.

MCP servers arrive as `${RUNNER_TEMP}/gh-aw/mcp-config/mcp-servers.json` in the
`mcpServers` shape Claude Desktop and Cursor use, and the engine hands that file to
`pai --mcp-config`, which loads it with `pydantic_ai.mcp.load_mcp_toolsets` and
passes the toolsets into the run alongside whatever the agent already carries. Tools
carry their server name as a prefix, so safe outputs are reachable as
`safeoutputs_create_issue` and so on. HTTP servers are carried over; CLI-mounted
servers remain on the agent's `PATH` as executables. gh-aw's config adapter writes
that file in the `Start MCP Gateway` step on the host runner, next to the file the
built-in Claude engine gets, and the agent step mounts `${RUNNER_TEMP}/gh-aw`
read-only.

Neither file is in the checkout, and that is deliberate. A file committed at a path
the engine reads is repository-controlled input to a process that runs with the
gateway's credentials: an MCP config could name a stdio server for the CLI to spawn,
and a package under a directory the engine puts on `PYTHONPATH` would shadow an
installed one for the whole run. Repository code reaches the agent only through
`PAI_AGENT`, below.

## Running your own agent

`PAI_AGENT` in `engine.env` replaces the composed coder agent with one your
repository defines. It takes exactly what `pai -a` takes: a `module:variable`
import path, or a `.yml`, `.yaml` or `.json` agent spec file.

```yaml
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: copilot/claude-sonnet-4-5
  env:
    PAI_AGENT: my_agent:agent
steps:
  - name: Install the agent's dependencies
    run: python3 -P -m pip install --quiet --user --disable-pip-version-check httpx
---

# Triage

Read the issue and summarize what changed.
```

`my_agent.py` lives at the root of your repository:

```python
from pydantic_ai import Agent

agent = Agent(name='triage', instructions='Answer briefly.')
```

Five things to know.

- **The repository joins `PYTHONPATH`.** Only when `PAI_AGENT` is set: making
  repository code importable is the point of running your own agent, and it is what
  the engine otherwise keeps off the import path for its own composition. The
  generated `gh_aw_agent.py` is not written at all in this mode.
- **Dependencies go in a workflow-level `steps:` block.** Those steps run on the
  host runner, after gh-aw's `Setup Python` and before both the engine's install
  step and the agent, so a `--user` install lands in the same `$HOME/.local` the
  sandbox exposes and belongs to the same interpreter (`compiler_yaml_main_job.go`
  emits `generateRuntimeAndWorkspaceSetupSteps` before
  `generateEngineInstallAndPreAgentSteps`). `-P` keeps the checkout off `sys.path`
  for the install itself.
- **MCP tools arrive the same way they do for the coder agent.** The engine passes
  `--mcp-config` whenever the gateway wrote a config, so the gateway's servers are
  added to your agent's own toolsets. Your agent does not load the config itself,
  and does not need to know where it is.
- **The engine always passes `-m`.** An explicit `-m` replaces the model a loaded
  agent declares, so your agent runs on the workflow's `engine.model` whatever it
  was constructed with. Configure the model in the workflow, not in the agent.
- **Endpoint and provider handling are unchanged.** `PAI_BASE_URL`, `/reflect`
  discovery and the `provider/` prefix behave exactly as they do for the coder
  agent, described below.

A `module:variable` target is imported once, by the process that goes on to run the
CLI, so module-level work in your agent runs once too. An agent that fails to import
or is not an `Agent` fails the step with the Python traceback. A spec file, and the
dotted `module.attribute` form `pai` also accepts, are left to the CLI, which reports
its own error.

## gh-aw compatibility

This definition requires the gh-aw action/runtime at
[v0.86.3](https://github.com/github/gh-aw/releases/tag/v0.86.3) or newer. Its
endpoint discovery uses `deriveBaseUrlFromModelsURL`, which that release exports for
converting the reflected `/models` URL into the chat-completions base URL while
preserving the firewall host bridge.

Existing workflows must be recompiled with a compatible gh-aw pin and have their
generated lockfile committed. Installing a newer `gh aw` CLI locally does not alter
an already committed lockfile or the action/runtime it pins.

## Pointing the engine at your own endpoint

`PAI_BASE_URL` in `engine.env` sends requests to any endpoint that speaks the
OpenAI **Chat Completions** API, instead of the AWF api-proxy. The engine skips
`/reflect` discovery and uses the URL verbatim.

```yaml
engine:
  id: pydantic-ai
  model: openai/<model-id>
  env:
    PAI_BASE_URL: https://your-endpoint.example.com/v1
network:
  allowed:
    - your-endpoint.example.com
```

Three things to get right:

- **The URL is used verbatim.** The client posts to `<PAI_BASE_URL>/chat/completions`,
  so include whatever path prefix the endpoint expects, usually `/v1`.
- **Add the host to `network.allowed`.** The agent runs behind an egress firewall
  that denies everything else.
- **Keep the `provider/` prefix on `model`.** gh-aw requires it and only accepts its
  four known providers, but the engine strips it before calling the endpoint. Write
  `openai/<model-id>`, and `<model-id>` is what goes upstream. `openai` here means
  "OpenAI-compatible", not OpenAI the company. `PAI_BASE_URL` keeps Chat Completions
  whichever prefix is written, `anthropic/` included: it names an endpoint of that
  shape by definition.

`PAI_BASE_URL` is a variable of this engine's own rather than `OPENAI_BASE_URL`,
because gh-aw sets `OPENAI_BASE_URL` itself, pointing at the proxy. It is always
present, so it cannot express a choice.

### Credentials, and why there is no `PAI_API_KEY`

gh-aw keeps repository secrets out of the agent sandbox. Any `engine.env` value
containing `${{ secrets.* }}` is stripped from the agent's environment
(`awf --exclude-env`), and the compiler rejects the workflow rather than letting you
believe otherwise. That is deliberate: for the proxy-backed providers, credentials
live in the proxy, outside the agent, and the agent sends a placeholder bearer token
it cannot leak.

The same rule applies here, so this engine has no key setting. Two consequences:

**Endpoints that need no credential from the agent work directly.** A self-hosted
OpenAI-compatible server, a local model runner, or an internal gateway that
authenticates by network position rather than by token:

```yaml
engine:
  id: pydantic-ai
  model: openai/qwen3-coder
  env:
    PAI_BASE_URL: http://models.internal.example.com/v1
network:
  allowed:
    - models.internal.example.com
```

**Keyed providers are reached through a gateway you run.** For a commercial
endpoint that requires an API key (MiniMax, Together, Fireworks and the like), point
`PAI_BASE_URL` at a service you control that holds the key and forwards upstream:

```
agent (placeholder bearer)  ->  your gateway (adds the real key)  ->  provider
```

The gateway is yours to deploy and is out of scope for this repository. What matters
here is the shape: the credential lives on the far side of the sandbox boundary, the
same place gh-aw already keeps them. There is no configuration in this engine that
puts a provider key in the agent's hands, and adding one is not possible without a
change to gh-aw.

## Troubleshooting

Only mechanisms that have been checked against a compiled workflow are listed.

**The request never leaves, or fails to connect.** The host is missing from
`network.allowed`. The agent's egress is default-deny, and `PAI_BASE_URL` does not
open a hole on its own.

**404 from the endpoint.** The base path is wrong. The client appends
`/chat/completions`, so `https://host/v1` produces `https://host/v1/chat/completions`.
An endpoint documented as `https://host/v1/chat/completions` should be given as
`https://host/v1`.

**401 or 403 from the endpoint.** It wanted a credential. The agent sent the
placeholder bearer token, which is all it has. Put a gateway in front that adds the
real key, per the section above.

**`engine.model is required ... must use provider/model format` at compile time.**
The `provider/` prefix is missing. gh-aw validates it before the engine ever runs,
even when `PAI_BASE_URL` makes the provider irrelevant.

**`strict mode: secrets detected in 'engine.env'` at compile time.** A secret was
put in `engine.env`. It would have been removed from the agent's environment anyway;
see the credentials section.

## Changing this file

`pydantic.md` is the published artifact, and merging to `main` is what ships it:
every consumer importing `@main` picks the change up the next time they recompile.
Consumers pinned to a tag or a SHA stay where they are until they move the ref.

A harness release does not move the engine. `engine.version` is an ordinary line in
the file, so a new release reaches consumers only once a pull request bumps it --
and lint refuses a pin that is not on PyPI, so that pull request is green only after
the release is published. The release workflow opens an issue when the tag it just
published and the pinned version differ.
