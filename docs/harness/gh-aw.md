---
title: Run your own Pydantic AI agent as a GitHub Agentic Workflow
description: Run your own Pydantic AI agent headless in GitHub Actions, on issues, pull requests or a schedule, sandboxed, with MCP tools and safe outputs.
---

# Run your own Pydantic AI agent as a GitHub Agentic Workflow

[GitHub Agentic Workflows](https://github.com/github/gh-aw) (gh-aw) runs an agent from a
Markdown file in `.github/workflows/`: it triggers on issues, pull requests or a schedule,
starts the agent in a container behind an egress firewall, hands it MCP tools, and writes
what the agent produces back to GitHub through safe outputs. The
[`pydantic-ai` engine](https://github.com/pydantic/pydantic-ai-harness/tree/main/gh-aw)
points that machinery at a Pydantic AI agent, which can be the
[`Coder`](/ai/harness/coder/) composition it runs by default, another agent the harness
ships, or one your own repository defines. The three are laid out in
[Start from `Coder`, `Researcher`, or your own](#start-from-coder-researcher-or-your-own);
this page then walks through the last of them end to end.

This is the agent running **headless**: on a GitHub runner, on the events you choose, with
nobody at a prompt. Its output is a comment, a pull request or a commit rather than a reply
on a screen, and the only place to see what it did afterwards is the run log and whatever
telemetry you configured, which is why [Observability](#observability) is a section rather
than an aside.

Every model vendor ships an action for this shape, each one running that vendor's agent on
that vendor's models. The difference here is that the agent is yours: your instructions,
your tools, your [capabilities](/ai/capabilities/overview/), and [any
model](/ai/models/overview/) a string swap away, in a repository you can also run from the
terminal, serve over a web UI, or call from your own backend. It is the same `Agent` object
in all of them.

The finished repository is
[dsfaccini/gh-aw-pydantic-ai-demo](https://github.com/dsfaccini/gh-aw-pydantic-ai-demo);
every file and command below comes from it.

Two pieces of gh-aw vocabulary to set aside first. A gh-aw **custom agent** (or "agent
file") is a Markdown prompt in `.github/agents/`, described under
[custom agents](https://github.github.com/gh-aw/reference/copilot-custom-agents/); it is
not what this page builds. Here the agent is a Python
[`Agent`](/ai/core-concepts/agent/) object in a module in your repository, named by the
`PAI_AGENT` environment variable. And `engine: driver:` is gh-aw's mechanism for swapping
the inner driver of its copilot engine (and pi on Node); it has no effect on an
import-based engine like this one, so it is not part of the configuration below.

## Prerequisites

- The [`gh` CLI](https://cli.github.com), authenticated with the `repo` and `workflow`
  scopes: `gh auth login --scopes repo,workflow`.
- The gh-aw extension: `gh extension install github/gh-aw`. Append `@vX.Y.Z` to pin it.
- gh-aw runtime v0.86.3 or newer. The engine definition needs
  `deriveBaseUrlFromModelsURL`, which
  [v0.86.3](https://github.com/github/gh-aw/releases/tag/v0.86.3) is the first release to
  export. An older pin in an already committed lockfile stays in force until you recompile.
- Linux runners. gh-aw's sandbox needs Linux and Docker, so the `macos-*` and `windows-*`
  runner labels are
  [not supported](https://github.github.com/gh-aw/reference/frontmatter/).
- Issues enabled on the repository: the workflow below triggers on the `issues` event and
  posts its result as a comment on an issue. gh-aw also checks the setting at compile time
  when a workflow declares the `create-issue` safe output, and fails compilation if issues
  are off.
- Actions enabled on the repository (see [Repository settings](#repository-settings)).

## Start from `Coder`, `Researcher`, or your own

**Use `Coder` by default.** Omit `PAI_AGENT` to use [`Coder`](/ai/harness/coder/),
with filesystem access and unrestricted shell commands inside the sandbox.
No agent module is needed:

```yaml
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: read
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: openai/gpt-5
safe-outputs:
  add-comment:
---
```

**Another ready-made agent.** The harness exports assembled agents as importable variables,
so `PAI_AGENT` can name one directly and the repository still contains no agent code:

```yaml
engine:
  id: pydantic-ai
  model: openai/gpt-5
  env:
    PAI_AGENT: pydantic_ai_harness.researcher:researcher_agent
steps:
  - name: Install the researcher extra
    run: python3 -P -m pip install --quiet --user --disable-pip-version-check "pydantic-ai-harness[researcher]"
```

[`Researcher`](/ai/harness/researcher/) needs its own extra for the local search and fetch
fallbacks, which is what the `steps:` block installs, and its searches need the hosts it
reaches on the workflow's `network:` allowlist. `pydantic_ai_harness.coder:coder_agent` is
the same shape for the coder agent, and is worth naming explicitly when a workflow wants
the default composition but also wants `PAI_AGENT` set for clarity.

**Your own agent.** The rest of this page. Reach for it when the agent needs its own tools,
its own instructions, or a composition the harness does not ship.

## The agent module

`my_agent.py` at the root of the repository. `PAI_AGENT` names the variable in it, in the
same `module:variable` form the [`pai` CLI](/ai/integrations/cli/) takes for `-a`:

```python
"""The agent this repository's agentic workflow runs.

`PAI_AGENT: my_agent:agent` in `.github/workflows/triage.md` names the `agent`
variable below. The engine passes `-m` from the workflow's `engine.model`, so
the model is not set here.
"""

from pydantic_ai import Agent

LABELS = ('bug', 'documentation', 'enhancement', 'question')

agent = Agent(
    name='triage',
    instructions="""
You triage one GitHub issue. Read the issue in the prompt, then post exactly one
comment with the `safeoutputs_add_comment` tool. The comment has three parts, in order:

1. **Summary.** What the issue reports, in two sentences or fewer.
2. **Suggested label.** One label from `label_catalog()`, and one line saying why.
3. **Question.** What the reporter still needs to tell us before anyone can act.
   Write "No follow-up needed." when the issue is already actionable.

Suggest a label; do not apply one. Do not edit files. Do not open issues.
""",
)


@agent.tool_plain
def label_catalog() -> list[str]:
    """The labels this repository triages with."""
    return list(LABELS)
```

Four things about that module.

- **No model.** The engine always passes `-m` built from the workflow's `engine.model`,
  and an explicit `-m` replaces whatever model a loaded agent declares. Setting a model on
  the `Agent` would be ignored, so the workflow is the one place the model is configured.
- **`safeoutputs_add_comment` is the safe output, not an ordinary tool.** gh-aw fronts
  every MCP server it configures behind a gateway and writes them to
  `${RUNNER_TEMP}/gh-aw/mcp-config/mcp-servers.json` on the host runner, which the agent
  step mounts read-only and the engine passes to `pai --mcp-config`;
  [`load_mcp_toolsets`](/ai/mcp/client/) prefixes each server's tools with its name, so the
  `add-comment` safe output arrives as `safeoutputs_add_comment`. That file is deliberately
  outside the checkout: a file committed at a path the engine reads would be
  repository-controlled input to a process holding the gateway's credentials. The comment
  itself is posted by a separate job after the agent finishes; the agent never holds a
  token that can write to the repository.
- **`label_catalog` is an ordinary [function tool](/ai/tools-toolsets/tools/).** It is
  here to show that repository code is importable and that the agent's own tools work
  alongside the MCP tools gh-aw supplies.
- **No third-party imports.** The engine installs `pydantic-ai-harness[cli]` and
  `pydantic-ai-slim[anthropic,openai,mcp,spec]`, so `pydantic_ai` is importable without any
  setup of your own. Anything else your agent imports is installed by a workflow-level
  `steps:` block (see [Dependencies](#dependencies)).

The module is imported once, by the interpreter that then runs the CLI in the same process,
so module-level work runs once. An agent that raises on import fails the step with its
Python traceback rather than a one-line "could not load agent" message.

## The agent as a spec instead

`PAI_AGENT` takes a `.yml`, `.yaml` or `.json` [agent spec](/ai/core-concepts/agent-spec/)
wherever it takes an import path, so a repository that is already configuring one thing in
YAML can configure the agent the same way, in a file beside the workflow:

```yaml
engine:
  id: pydantic-ai
  model: openai/gpt-5
  env:
    PAI_AGENT: triage_agent.yml
```

```yaml {title="triage_agent.yml"}
name: triage
instructions: |
  You triage one GitHub issue. Read the issue in the prompt, then post exactly one
  comment with the `safeoutputs_add_comment` tool. Suggest a label; do not apply one.
  Do not edit files. Do not open issues.
capabilities:
  - Thinking:
      effort: medium
```

The engine installs the `spec` extra for YAML parsing.

No `model:` in the spec, for the same reason a module carries none: the engine always
passes `-m` built from the workflow's `engine.model`, and an explicit `-m` replaces
whatever a loaded agent declares. The workflow is the one place the model is configured.

The gateway's MCP servers still arrive through `--mcp-config`, so a spec agent gets the
safe outputs and the GitHub tools on the same terms as a module.

Two things a spec cannot do today, both of which send you back to a module:

- **Name a harness capability.** A spec resolves capability names through a closed registry
  that the harness capabilities are not part of, and the CLI passes no
  `custom_capability_types`, so a spec reaches the built-in capabilities and nothing else.
  `Coder`, `Researcher` and the rest are module-only. See
  [pydantic-ai#8334](https://github.com/pydantic/pydantic-ai/issues/8334).
- **Define a function tool.** The `label_catalog` tool above is Python, and there is no
  spec form for it.

Instructions plus built-in capabilities is the shape a spec handles well. Anything past
that is a module.

## The workflow file

`.github/workflows/triage.md`:

```markdown
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: read
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main
engine:
  id: pydantic-ai
  model: openai/gpt-5
  env:
    PAI_AGENT: my_agent:agent
safe-outputs:
  add-comment:
---

# Triage the new issue

The issue that triggered this run, as gh-aw sanitized it:

<issue>
${{ steps.sanitized.outputs.text }}
</issue>

Post your triage as a single comment on that issue.
```

Key by key:

- `on: issues: types: [opened]` is the only trigger. `add-comment` posts to the issue
  that triggered the run, and a `workflow_dispatch` run has none, so the workflow declares
  no manual trigger: `gh aw run` and the Run workflow button are not part of this
  walkthrough.
- `permissions:` is the workflow-level token scope, and gh-aw rejects a write scope here;
  the jobs that write to GitHub get their own narrower scopes in the compiled file.
  `issues: read` is what lets the GitHub MCP tools read an issue. gh-aw loads a default
  GitHub read toolset (`context`, `repos`, `issues`, `pull_requests`, `users`) with no
  `tools:` block at all, so the agent has `github_issue_read` either way, but without the
  scope the call comes back
  `403 Resource not accessible by integration`. Narrow or widen the toolset with
  [`tools: github: toolsets:`](https://github.github.com/gh-aw/reference/github-tools/).
- `imports:` pulls in the engine definition. gh-aw's engine catalog knows the
  `pydantic-ai` id but does not import it for you: naming the engine without this line
  fails to compile.
- `engine: id:` selects the imported engine, and `engine: model:` is required in
  `provider/model` form. The provider segment selects which backend of gh-aw's api-proxy
  serves the request; `copilot`, `anthropic`, `openai` and `codex` are the accepted values.
  It also decides the wire API: `anthropic/` runs over the Anthropic Messages API, because
  that backend forwards the request path to `api.anthropic.com` unchanged, and the other
  three are OpenAI-shaped and use Chat Completions. Under `PAI_BASE_URL` everything stays
  on Chat Completions.
- `engine: env: PAI_AGENT:` is what replaces the engine's composed
  [`Coder`](/ai/harness/coder/) agent with yours. Setting it also puts the checkout on
  `PYTHONPATH`, which is what makes `my_agent` importable.
- `safe-outputs: add-comment:` declares the one write this workflow performs. With no
  `safe-outputs:` section at all, gh-aw enables `create-issue` with a max of 1 instead;
  declaring the section replaces that default, so the triage does not also open an issue
  for every issue it comments on.
- The body after the frontmatter is the prompt. gh-aw prepends its own context block,
  which carries the issue number but not the issue text, so
  `${{ steps.sanitized.outputs.text }}` is what puts the title and body in front of the
  agent. gh-aw computes that value by sanitizing the triggering item's content, and it is
  the form its [templating reference](https://github.github.com/gh-aw/reference/templating/)
  documents for prompts (`.title` and `.body` expose the two halves separately).

The compile error you get from a missing `imports:` line carries a tip naming
`github/gh-aw/.github/workflows/shared/pydantic.md@<version>`. That is gh-aw's own older
copy of the definition. Ignore it and write the `pydantic/pydantic-ai-harness` line above;
the definition in this repository is the one that is maintained.

**Freezing the definition.** `@main` is re-resolved on every compile, so a change to the
definition reaches you the next time you run `gh aw compile`. To hold a fixed version,
import a commit SHA that contains `gh-aw/pydantic.md`, or a release tag cut after the
definition landed on `main`; tags older than the file return a 404 at compile time. That
ref pins the definition. The harness package version is pinned separately, by
`engine: version:` in the definition, and a workflow's own `engine: version:` overrides it.

## Compile and commit

```bash
gh aw compile
```

The compiler writes:

- `.github/workflows/triage.lock.yml`, the GitHub Actions workflow that actually runs.
- `.github/aw/actions-lock.json`, the SHA pins for every action the lock uses.
- `.github/aw/imports/pydantic/pydantic-ai-harness/<sha>/gh-aw_pydantic.md`, a
  byte-identical cache of the imported definition at the resolved SHA.
- `.github/aw/imports/.gitattributes` and a top-level `.gitattributes` marking generated
  files.

Commit all of it, together with the `.md`. The compiled workflow's `Check workflow lock
file` step recomputes the frontmatter hash at run time and fails the run when the lock does
not match the source, so a stale or missing lock stops the workflow rather than running an
old configuration.

Every compile prints `Using experimental engine: Pydantic AI`. The engine definition
declares itself experimental to gh-aw, which is a statement about this definition's
interface with gh-aw rather than about Pydantic AI: the two projects are moving, and the
definition is free to change how it bridges them. Pin the `imports:` ref to a tag or a SHA
when that matters, as described above.

Compiling this workflow for the first time prints a security-review warning listing
`CODEX_API_KEY` and `OPENAI_API_KEY` as new restricted secrets. gh-aw records the secrets,
actions and container images a lock uses in a `gh-aw-manifest` header and asks you to look
at additions before it accepts them. Read the list, then re-run with `--approve`:

```bash
gh aw compile --approve
```

`gh aw compile` also caches the remote import under `.github/aw/imports/` by commit SHA.
gh-aw treats the lock and that cache together as the
[reproducibility guarantee](https://github.github.com/gh-aw/practices/sharing-workflows/)
for shared workflows, so commit it alongside the `.md` and the `.lock.yml`.

## Dependencies

Anything your agent imports beyond `pydantic_ai` is installed by a workflow-level `steps:`
block:

```yaml
steps:
  - name: Install the agent's dependencies
    run: python3 -P -m pip install --quiet --user --disable-pip-version-check httpx
```

`--user` installs into `$HOME/.local`, which is the directory the sandbox exposes and the
one the engine's own install already uses; a system-wide install would land somewhere the
agent cannot read. `-P` keeps the checkout off `sys.path` for the install itself, so a
module in your repository named after a package cannot be imported in place of the real
one.

Order matters, and the compiled lock puts these steps in the right place. In the generated
workflow the sequence is `Setup Python`, then your `steps:`, then the engine's `Preinstall
Pydantic AI coder agent`, then `Execute Pydantic AI CLI`. Your install therefore runs
against the same interpreter the engine later uses.

## Credentials

The provider segment of `engine.model` decides which repository secret the agent step
reads:

| `engine.model` prefix | Secret |
|---|---|
| `copilot/...` | `COPILOT_GITHUB_TOKEN` |
| `anthropic/...` | `ANTHROPIC_API_KEY` |
| `openai/...` | `CODEX_API_KEY`, or `OPENAI_API_KEY` when `CODEX_API_KEY` is unset |
| `codex/...` | `CODEX_API_KEY`, or `OPENAI_API_KEY` when `CODEX_API_KEY` is unset |

Only the secret for the provider you use needs setting. The workflow above uses
`openai/gpt-5`, so it needs `CODEX_API_KEY` or, when that is unset, `OPENAI_API_KEY`. The
examples below set `OPENAI_API_KEY`.

=== "CLI"

    ```bash
    gh aw secrets set OPENAI_API_KEY --value "<key>"
    ```

    The value can also come from an environment variable
    (`--value-from-env OPENAI_KEY`) or from stdin, and `--repo owner/name` targets a
    repository other than the current one. `gh aw secrets bootstrap` reads the compiled
    workflows and walks you through whatever is missing. Plain `gh secret set
    OPENAI_API_KEY` works too; it prompts for the value or reads it from stdin.

=== "Web UI"

    Settings -> Secrets and variables -> Actions -> New repository secret. Name it
    `OPENAI_API_KEY` and paste the key. GitHub documents the screen under
    [using secrets in GitHub Actions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

Where the key comes from, per provider:

- OpenAI and Codex: <https://platform.openai.com/api-keys>.
- Anthropic: <https://console.anthropic.com/settings/keys>.
- Copilot: a fine-grained PAT from
  <https://github.com/settings/personal-access-tokens/new> with the resource owner set to
  your account and Account permissions -> Copilot Requests set to Read. `gho_` OAuth
  tokens are rejected. With an organization Copilot subscription on centralized billing you
  can skip the PAT and add
  [`permissions: copilot-requests: write`](https://github.github.com/gh-aw/reference/auth/#copilot-requests-write-permission)
  instead, which makes the agent step use the run's `github.token`.

**There is no `PAI_API_KEY`.** gh-aw keeps repository secrets out of the agent sandbox: any
`engine.env` value containing `${{ secrets.* }}` is stripped from the agent's environment
via `awf --exclude-env`, and the compiler fails the workflow rather than let you assume
otherwise. The provider credential lives in gh-aw's api-proxy, on the other side of the
sandbox boundary, and the agent sends a placeholder bearer token it cannot leak. To reach a
keyed endpoint of your own, point `PAI_BASE_URL` at a gateway you run that holds the key;
the engine's
[README](https://github.com/pydantic/pydantic-ai-harness/blob/main/gh-aw/README.md)
covers that path.

## Repository settings

**Actions enabled.**

=== "CLI"

    ```bash
    gh api repos/OWNER/REPO/actions/permissions
    gh api -X PUT repos/OWNER/REPO/actions/permissions -F enabled=true -f allowed_actions=all
    ```

    The `PUT` sets the whole policy, so send `allowed_actions` with the value you want
    rather than relying on the current one being kept.

=== "Web UI"

    Settings -> Actions -> General -> Actions permissions.
    [GitHub's guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)
    describes the screen.

**Allowed actions.** Only relevant when `allowed_actions` is `selected`. The compiled lock
uses actions from two owners: `actions/*` (`checkout`, `setup-python`, `setup-node`,
`cache`, `github-script`, `upload-artifact`, `download-artifact`) and
`github/gh-aw-actions/*`. Both need to be in the allowlist. gh-aw's own troubleshooting
page still lists `github/gh-aw@*`, which is not what current versions emit.

=== "CLI"

    ```bash
    gh api repos/OWNER/REPO/actions/permissions/selected-actions
    gh api -X PUT repos/OWNER/REPO/actions/permissions/selected-actions \
      -F github_owned_allowed=true \
      -f 'patterns_allowed[]=github/gh-aw-actions/*'
    ```

    `github_owned_allowed` covers the `actions/*` half. The `selected-actions` endpoint
    only applies while `allowed_actions` is `selected`.

=== "Web UI"

    Settings -> Actions -> General -> Allow select actions -> Allow specified actions and
    reusable workflows.

**Default-branch placement.** GitHub only fires the `issues` event for workflow files on
the repository's default branch. Nothing happens while the workflow sits on a feature branch; this is
[GitHub Actions behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issues),
not a gh-aw setting, and gh-aw's own pages do not mention it.

**Who may trigger.** The compiled workflow gates on the triggering user's repository role,
defaulting to `admin`, `maintainer` and `write`. An issue opened by someone outside that
set does not start the agent. Widen or narrow it with the `roles:` frontmatter key.

**Forks.** gh-aw blocks pull requests from forks unless the `pull_request` trigger names
them in a `forks:` allowlist, and documents that agentic workflows do not run inside a
forked copy of a repository. See
[fork support](https://github.github.com/gh-aw/reference/fork-support/).

Repository-level "Read and write permissions" for the default `GITHUB_TOKEN` is not needed:
the lock declares its scopes per job. "Allow GitHub Actions to create and approve pull
requests" only matters for the `create-pull-request` safe output.

## Run and observe

Merge the workflow to the default branch. Opening an issue then starts a run.

```bash
gh aw status
```

`gh aw status` lists each workflow with its engine, whether the lock is current, and
whether the workflow is enabled:

```
╭────────┬───────────┬────────┬──────┬─────────┬──────┬──────┬──────────╮
│workflow│engine     │compiled│state │remaining│run id│status│conclusion│
├────────┼───────────┼────────┼──────┼─────────┼──────┼──────┼──────────┤
│triage  │pydantic-ai│Yes     │active│N/A      │0     │-     │-         │
╰────────┴───────────┴────────┴──────┴─────────┴──────┴──────┴──────────╯
```

After a run:

```bash
gh aw logs triage --artifacts all
gh aw audit <run-id-or-url>
```

`gh aw logs` downloads runs into per-run folders; `--artifacts all` adds the agent logs,
`agent-stdio.log`, `aw.patch` and `summary.json` to the compact usage artifact it fetches
by default. `gh aw audit` turns one run into a Markdown report under `.github/aw/logs`, and
diffs two or more runs when given more than one id. For the raw step output, including the
engine's own lines, `gh run view <run-id> --log` (add `--attempt N` for an earlier attempt)
is often quicker.

The engine's own step summary is written by the lock's `Parse agent logs for step summary`
step and appears on the Actions run's Summary page, carrying the turn, tool-call and token
counts. `gh aw logs --parse` and `gh aw audit --parse` do not re-render it locally: they
resolve engines through gh-aw's built-in registry (claude, codex, copilot, gemini, pi) and
skip anything else, which is every import-based engine, this one included.

### What a working run looks like

In the `Execute Pydantic AI CLI` step, the engine prints the configuration it resolved
before the agent starts, then a line per tool call:

```text
[pydantic-ai] provider=openai model=gpt-5 baseUrl=http://api-proxy:10000/v1 agent=my_agent:agent
▌ Called tool label_catalog.
▌ Called tool safeoutputs_add_comment.
```

The `agent=` value is what `PAI_AGENT` resolved to, and `baseUrl=` is the api-proxy inside
the sandbox. A line like

```text
[pydantic-ai] awf-reflect: unable to persist reflect payload to /home/runner/work/_temp/awf-reflect.json: EACCES: permission denied
```

appears in successful runs too. It comes from gh-aw's reflect helper trying to cache the
endpoint payload in a directory the sandbox does not let it write, and the discovered
endpoint is used regardless.

The comment lands on the issue with a gh-aw footer naming the workflow and linking its run,
followed by an HTML comment recording the engine, its version and the model.

## Observability

gh-aw exports spans for its own setup and conclusion steps once a workflow configures an
OpenTelemetry backend, and this engine extends that to the agent: the agent run, every
model request and every tool call arrive as Pydantic AI spans in the same trace, with no
Python of your own.

`observability:` names the backend and `network:` lets the export through the egress
firewall:

```yaml
network:
  allowed:
    - logfire-us.pydantic.dev
observability:
  otlp:
    endpoint:
      - url: https://logfire-us.pydantic.dev
        headers:
          Authorization: ${{ secrets.LOGFIRE_TOKEN }}
```

Both halves are needed. The allowlist merges with the domains the engine already permits
(PyPI, GitHub, the model provider) rather than replacing them, and without the entry the
firewall drops the export while the run itself still succeeds, which at the backend looks
the same as a workflow that was never configured.

Set the token as a repository secret, the same way the provider credential is set:

```bash
gh aw secrets set LOGFIRE_TOKEN --value "<write-token>"
```

Use `logfire-eu.pydantic.dev` for a project in the EU region: the host has to match the
region the token belongs to. Logfire's OTLP ingest takes the token bare, with no `Bearer`
prefix, which is the form gh-aw documents for `Authorization` headers generally.

**The telemetry credential does reach the agent.** The model credential does not: it stays
in gh-aw's api-proxy on the other side of the sandbox boundary, which is why there is no
`PAI_API_KEY` (see [Credentials](#credentials)). The OTLP header is different. gh-aw
delivers it to the agent process as `OTEL_EXPORTER_OTLP_HEADERS`, because that is how the
SDKs running inside the sandbox are meant to reach the backend. Use a write token, which
cannot read data back out of the project.

### What the engine does with it

`OTEL_EXPORTER_OTLP_ENDPOINT` is gh-aw's signal that observability is configured, and it is
what the engine keys on. When it is set, and only then:

- **`logfire` is installed** alongside the CLI, so a workflow without observability pays
  nothing for it.
- **The launcher configures and instruments** before it imports your agent, so an agent that
  starts work at import time is already traced. `send_to_logfire` is `'if-token-present'`
  and the console exporter is off: the spans go to the endpoint the workflow configured,
  not to a Logfire project of the engine's choosing, and not into the step log, where they
  would also reach the log parser.
- **A `LOGFIRE_TOKEN` in the agent's environment adds a destination rather than replacing
  one.** `'if-token-present'` means spans go to that project as well as to the OTLP endpoint
  above, prompts and completions included. gh-aw keeps `${{ secrets.* }}` values out of the
  agent's environment when they come from `engine.env`, so this happens only when a workflow
  puts a token in its own `env:` or a `steps:` block, which is the explicit way to ask for
  it. Leave the token out and the workflow's endpoint is the only place anything goes.
- **Configuration and credentials stay outside the checkout.** The launcher passes explicit
  `config_dir` and `data_dir` values pointing to a mode-0700 temporary directory under `/tmp`,
  even if `TMPDIR` points into the checkout. Checkout `pyproject.toml` settings and
  `.logfire/logfire_credentials.json` cannot select a telemetry destination. These arguments
  also override `LOGFIRE_CONFIG_DIR` and `LOGFIRE_CREDENTIALS_DIR`, without disabling an
  environment `LOGFIRE_TOKEN`. The directory lives through the process and is removed at
  interpreter shutdown, after exporter shutdown handlers run.
- **The trace context is attached.** gh-aw publishes the run's W3C trace context in
  `TRACEPARENT` so that an engine can nest its spans under the workflow span, but neither
  Logfire nor the OpenTelemetry SDK reads that variable on its own. Attaching it is what
  keeps the agent's spans in the workflow's trace instead of a second, unrelated one.
- **Traces are the only signal sent.** An OTLP endpoint covers metrics and logs as well, and
  a backend that accepts traces alone answers `404` to those, once per export, in the step
  log. The engine therefore defaults `OTEL_METRICS_EXPORTER` and `OTEL_LOGS_EXPORTER` to
  `none`. Set either in the workflow to turn that signal back on for a backend that takes
  it; token counts are on the spans either way.

The configuration line the engine prints gains an `otlp=` segment, carrying the endpoint's
origin and no more, because userinfo and query parameters in an endpoint are credentials and
a run log is readable by anyone who can read the repository. It is there so that a run log
says whether telemetry was on, and roughly where it went:

```text
[pydantic-ai] provider=openai model=gpt-5 baseUrl=http://api-proxy:10000/v1 agent=my_agent:agent otlp=https://logfire-us.pydantic.dev
```

Pydantic AI's spans are exported straight to the backend and are not mirrored into the
`otel.jsonl` file that `gh aw logs --artifacts agent` downloads; that file carries the spans
gh-aw's own JavaScript helpers emit.

### Finding the runs

gh-aw puts the run's identity in `OTEL_RESOURCE_ATTRIBUTES` and `logfire.configure()` merges
that into the resource, so every span carries it without the engine adding any attributes of
its own:

| Attribute | Value |
|---|---|
| `gh-aw.engine.id` | `pydantic-ai` |
| `gh-aw.workflow.name` | The workflow's name |
| `gh-aw.repository` | `owner/name` |
| `gh-aw.run.id`, `github.run_id` | The Actions run id |

`gh-aw.engine.id = 'pydantic-ai'` is the filter for every run of this engine across every
repository reporting to one backend, and `gh-aw.run.id` joins a trace back to its Actions
run.

### Any OpenTelemetry backend

Nothing above is specific to Logfire beyond the host and the header. `observability.otlp`
takes any endpoint that accepts OTLP over HTTP, and gh-aw's
[OpenTelemetry reference](https://github.github.com/gh-aw/reference/open-telemetry/)
documents a Sentry setup as well as Google Cloud Telemetry through Workload Identity
Federation in place of a static header. Swap the `url`, the `headers` and the host on the
allowlist; the rest of this section is unchanged.

### Configuring it from your own agent

A `PAI_AGENT` module that calls `logfire.configure()` itself runs after the engine's call and
replaces it, which is how to set a service name, scrubbing rules or extra span processors.
It replaces the whole configuration, not the arguments you restate, so carry over the ones
above that are load-bearing:

```python
import atexit
import tempfile

logfire_dir = tempfile.TemporaryDirectory(prefix='gh-aw-logfire-', dir='/tmp')
atexit.register(logfire_dir.cleanup)

import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    console=False,
    distributed_tracing=True,
    config_dir=logfire_dir.name,
    data_dir=logfire_dir.name,
)
logfire.instrument_pydantic_ai()
```

Things to know before you do:

- **Keep both directories private for the process lifetime.** Omitting `config_dir` or
  `data_dir` re-enables checkout configuration or credentials. Register cleanup before
  configuring Logfire so exporter shutdown handlers run first.
- **`console=False` is not optional.** Leaving it out restores logfire's console exporter,
  which writes every span to stderr, which is the stream the engine's log parser reads.
- **`distributed_tracing=True` keeps you in the run's trace.** The engine has already
  attached the context from `TRACEPARENT` by the time your module is imported, and that
  attachment survives, but without the flag logfire warns about it on every run.
- **A bare `logfire.configure()` fails here.** Its `send_to_logfire` default requires a
  `LOGFIRE_TOKEN` in the environment, and the token in the frontmatter above is a header
  value rather than an environment variable. Pass `send_to_logfire='if-token-present'`, as
  the engine does.
- **`logfire` is only installed when an endpoint is configured.** A module that imports it
  unconditionally needs it in the workflow's own `steps:` install as well, or the agent
  fails to import on a run with observability off.

## Troubleshooting

**`error: invalid engine: pydantic-ai. Valid engines are: claude, codex, copilot, gemini,
pi.`** The `imports:` line is missing. Add
`pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main`, not the path the accompanying tip
suggests.

**`error: invalid engine.model for engine 'pydantic-ai': for universal consumer engines,
engine.model must use provider/model format`** The `provider/` prefix is missing from
`engine.model`. gh-aw checks this before the engine runs.

**`error: strict mode: secrets detected in 'engine.env' section are excluded from the agent
sandbox via awf --exclude-env`** A `${{ secrets.* }}` reference is in `engine.env`. It
would have been stripped from the agent's environment anyway; set the provider's secret as
a repository secret instead.

**`warning: safe update mode detected unapproved changes` with a list of new restricted
secrets.** Expected on the first compile, and on any compile that changes which secrets,
actions or images the lock uses. Review the list, then re-run `gh aw compile --approve`.

**`[INFO] API proxy enabled: OpenAI=false, Anthropic=false, ...` then
`[health-check][ERROR] Cannot connect to OpenAI API proxy at http://host.docker.internal:10000`.**
The provider secret is missing or empty, so gh-aw starts no proxy backend and the job fails
before the engine's own script runs. The preceding line says so directly:
`[WARN] API proxy enabled but no API keys found in environment`. Set the secret, then rerun
the run with `gh run rerun <run-id>` so secrets are read again. A run that started before
the secret existed keeps the empty value for its whole life, so re-running the same attempt
is the fix, not waiting.

**`Lock file '...' is outdated! The workflow file '...' frontmatter has changed. Run 'gh aw
compile' to regenerate the lock file.`** The committed `.lock.yml` does not match the
`.md`. Recompile and commit the result.

**The workflow never starts when an issue is opened.** Check that the workflow file is on
the default branch, and that the account that opened the issue has one of the roles in
`roles:`.

## Using a coding agent

A coding agent with shell access can do this setup end to end: it writes both files, and
`gh aw compile` reports precisely what is wrong with the frontmatter when it gets one
wrong. Give it this page and the `module:variable` of the agent you want run.

```
Read https://pydantic.dev/docs/ai/harness/gh-aw/ and set up a gh-aw agentic workflow in
this repository that runs my Pydantic AI agent at my_agent:agent on newly opened issues.
```

The finished repository for this walkthrough is
[dsfaccini/gh-aw-pydantic-ai-demo](https://github.com/dsfaccini/gh-aw-pydantic-ai-demo).
The engine's own reference documentation, covering the default coder agent, custom
endpoints and the credential model, is the
[`gh-aw/README.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/gh-aw/README.md)
in this repository.
