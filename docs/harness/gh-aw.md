---
title: Run your own Pydantic AI agent as a GitHub Agentic Workflow
description: "Run your own Pydantic AI agent in GitHub Agentic Workflows (gh-aw): headless in GitHub Actions on issues, pull requests, or a schedule, with MCP tools."
---

# Run your own Pydantic AI agent as a GitHub Agentic Workflow

[GitHub Agentic Workflows](https://github.com/github/gh-aw) (gh-aw) runs an agent from a
Markdown file in `.github/workflows/`: it triggers on issues, pull requests or a schedule,
starts the agent in a container behind an egress firewall, hands it MCP tools, and writes
what the agent produces back to GitHub through safe outputs. The
[`pydantic-ai` engine](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/gh-aw)
points that machinery at a Pydantic AI agent, which can be the
[`Coder`](coder.md) composition it runs by default, another agent the harness
ships, or one your own repository defines. The three are laid out in
[Start from `Coder`, `Researcher`, or your own](#start-from-coder-researcher-or-your-own);
this page then walks through the last of them end to end.

This is the agent running **headless**: on a GitHub runner, on the events you choose, with
nobody at a prompt. Its output is a comment, a pull request or a commit rather than a reply
on a screen. The engine records the canonical agent session for inspection, and gh-aw can
collect the session and usage artifacts; [Observability](#observability) covers traces.

Every model vendor ships an action for this shape, each one running that vendor's agent on
that vendor's models. The difference here is that the agent is yours: your instructions,
your tools, your [capabilities](../capabilities/overview.md), and [any
model](../models/overview.md) a string swap away, in a repository you can also run from the
terminal, serve over a web UI, or call from your own backend. It is the same `Agent` object
in all of them.

The finished repository is
[dsfaccini/gh-aw-pydantic-ai-demo](https://github.com/dsfaccini/gh-aw-pydantic-ai-demo);
every file and command below comes from it.

Two pieces of gh-aw vocabulary to set aside first. A gh-aw **custom agent** (or "agent
file") is a Markdown prompt in `.github/agents/`, described under
[custom agents](https://github.github.com/gh-aw/reference/copilot-custom-agents/); it is
not what this page builds. Here the agent is a Python
[`Agent`](../agent.md) object in a module in your repository, named by the
`PAI_AGENT` environment variable. And `engine: driver:` is gh-aw's mechanism for swapping
the inner driver of its copilot engine (and pi on Node); it has no effect on an
import-based engine like this one, so it is not part of the configuration below.

![A new issue triggers the compiled gh-aw workflow. Its agent job runs the pydantic-ai engine in a container behind an egress firewall: the engine starts clai2 with your agent, named by PAI_AGENT, plus the workflow model and prompt. Model requests go through gh-aw's api-proxy, which holds the provider key, and the agent's GitHub and safe-output tools, such as github_issue_read and safeoutputs_add_comment, come from gh-aw's MCP gateway. After a threat-detection job checks the output, a separate add-comment job posts the comment with its own token.](../img/harness/gh-aw.svg)

## Prerequisites

- The [`gh` CLI](https://cli.github.com), authenticated with the `repo` and `workflow`
  scopes: `gh auth login --scopes repo,workflow`.
- Use [gh-aw v1.0.1](https://github.com/github/gh-aw/releases/tag/v1.0.1) or newer with
  AWF v0.29.0 or newer. That compiler release defaults to AWF v0.29.0 and includes
  native protected capture, packaged engine parsers, and verified-publication guards.
- Existing workflows need a compatible runtime build and a newly compiled lockfile.
  Installing a newer `gh aw` CLI locally does not change the runtime an existing lockfile
  selects.
- The protected-capture runtime rejects workflows with `evals`, whose secondary engine invocation
  reads `evals.log`. Omit `evals` when using this engine.
- The workflow summary is the canonical success/failure report. Categorical summary labels
  for specific MCP failures and maximum-turn outcomes are removed; MCP startup and
  request-limit errors retain their CLAI 2 diagnostics and nonzero exit status.
- Linux runners. gh-aw's sandbox needs Linux and Docker, so the `macos-*` and `windows-*`
  runner labels are
  [not supported](https://github.github.com/gh-aw/reference/frontmatter/).
- Issues enabled on the repository: the workflow below triggers on the `issues` event and
  posts its result as a comment on an issue. gh-aw also checks the setting at compile time
  when a workflow declares the `create-issue` safe output, and fails compilation if issues
  are off.
- Actions enabled on the repository (see [Repository settings](#repository-settings)).

Install the current stable compiler, check that it is at least v1.0.1, then compile:

```bash
gh extension install github/gh-aw
gh aw version
gh aw compile
```

For an existing installation, use `gh extension upgrade gh-aw` before compiling.
For a fresh installation of the exact tested compiler, use
`gh extension install github/gh-aw --pin v1.0.1`. To upgrade an explicitly pinned
installation, use `gh extension upgrade gh-aw --force`.

## Start from `Coder`, `Researcher`, or your own

**Use `Coder` by default.** Omit `PAI_AGENT` to use [`Coder`](coder.md) in a
[`LocalWorkspace`](../workspace.md) on the checkout, with filesystem access and unrestricted
shell commands inside the sandbox. Commands get the step's environment minus provider
credential variables (`OPENAI_*`, `ANTHROPIC_*` and the like). No agent module is needed:

```yaml
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: read
imports:
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@main
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

[`Researcher`](researcher.md) needs its own extra for the local search and fetch
fallbacks, which is what the `steps:` block installs, and its searches need the hosts it
reaches on the workflow's `network:` allowlist. `pydantic_ai_harness.coder:coder_agent` is
the same shape for the coder agent. It differs from the default composition in one way: its
`LocalWorkspace` passes commands only `PATH`, `HOME` and the locale variables, not the step's
environment.

**Your own agent.** The rest of this page. Reach for it when the agent needs its own tools,
its own instructions, or a composition the harness does not ship. A harness capability that
reads files or runs commands, such as `Coder`, needs a workspace: add `LocalWorkspace('.')`
next to it for the checkout.

## The agent module

`my_agent.py` at the root of the repository. `PAI_AGENT` accepts either a `module:variable`
or dotted `MODULE.ATTRIBUTE` import target:

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
  step mounts read-only. The engine loads the gateway tools and attaches them through a
  public dynamic toolset while preserving the agent's tools, so the
  `add-comment` safe output arrives as `safeoutputs_add_comment`. That file is deliberately
  outside the checkout: a file committed at a path the engine reads would be
  repository-controlled input to a process holding the gateway's credentials. The comment
  itself is posted by a separate job after the agent finishes; the agent never holds a
  token that can write to the repository.
- **`label_catalog` is an ordinary [function tool](../tools.md).** It is
  here to show that repository code is importable and that the agent's own tools work
  alongside the MCP tools gh-aw supplies through the engine's dynamic toolset.
- **No third-party imports.** The engine installs `pydantic-clai2` and
  `pydantic-ai-harness` at the release the definition's `engine: version:` pins, plus
  `pydantic-ai-slim[anthropic,openai,mcp,spec]>=2.59.0`, so `pydantic_ai` is importable
  without setup of your own. Anything else your agent imports is installed by a
  workflow-level `steps:` block (see [Dependencies](#dependencies)).

The engine resolves and imports the module once, then gives the agent to CLAI 2 with
`-a`, the workflow model with `-m`, and the prompt with `-p`. Module-level work therefore
runs once. An import error fails the step with its Python traceback.

## The agent as a spec instead

`PAI_AGENT` takes a `.yml`, `.yaml` or `.json` [agent spec](../agent-spec.md)
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

The engine installs the core `spec` extra for YAML parsing.

No `model:` in the spec, for the same reason a module carries none: the engine always
passes `-m` built from the workflow's `engine.model`, and an explicit `-m` replaces
whatever a loaded agent declares. The workflow is the one place the model is configured.

The engine resolves the spec once and attaches the gateway tools to the resulting agent,
so a spec gets safe outputs and GitHub tools on the same terms as a module.

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
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@main
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
  three are OpenAI-shaped and use Chat Completions. `copilot/` runs on Pydantic AI's
  [`github-copilot:`](../models/github-copilot.md) model, so Copilot's per-family model
  profiles apply and the thinking its Claude and Gemini models return in `reasoning_text`
  is kept. Its requests carry the `agentic-workflows` integration id, the one the api-proxy
  adds to requests that carry none. Under `PAI_BASE_URL` everything stays on the generic
  Chat Completions client.
- `engine: env: PAI_AGENT:` is what replaces the engine's composed
  [`Coder`](coder.md) agent with yours. The engine writes a private wrapper module in every mode;
  when resolving an imported target, it adds the checkout to `sys.path` after loading the
  framework packages.
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

**Freezing the definition.** `@main` is re-resolved on every compile, so a change to the
definition reaches you the next time you run `gh aw compile`. To hold a fixed version,
import a commit SHA that contains `src/pydantic_ai_harness/gh-aw/pydantic.md`, or a release
tag cut after the harness moved into this repository; older tags do not carry the file and
return a 404 at compile time. That ref pins the definition. The harness package version
is pinned separately, by `engine: version:` in the definition, and a workflow's own
`engine: version:` overrides it. An override has to be 0.59.0 or newer: each harness
release pins its own `pydantic-ai-slim`, and the definition installs
`pydantic-ai-slim>=2.59.0` alongside it.

## Compile and commit

```bash
gh aw compile
```

The compiler writes:

- `.github/workflows/triage.lock.yml`, the GitHub Actions workflow that actually runs.
- `.github/aw/actions-lock.json`, the SHA pins for every action the lock uses.
- `.github/aw/imports/pydantic/pydantic-ai/<sha>/src_pydantic_ai_harness_gh-aw_pydantic.md`, a
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

Anything your agent imports beyond the engine's runtime dependencies is installed by a
workflow-level `steps:` block:

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
workflow, `Setup Python` runs before your `steps:`, then the engine's install and agent
steps. Your install therefore runs against the same interpreter the engine later uses.

A step can also end a run that has nothing to do by appending
`{"type":"noop","message":"..."}` to `$GH_AW_SAFE_OUTPUTS`. The engine reads that file
before it starts CLAI 2 and exits with status 0 when it finds a `noop`, so no model request
is made. gh-aw's
[cost management reference](https://github.github.com/gh-aw/reference/cost-management/#skip-the-agent-when-it-is-not-needed)
documents the pattern. When a search query can express the condition, `skip-if-match` or
`skip-if-no-match` also avoids scheduling the agent job.

## Credentials

The provider segment of `engine.model` decides which repository secret the agent step
reads:

| `engine.model` prefix | Secret |
|---|---|
| `copilot/...` | `COPILOT_GITHUB_TOKEN` |
| `anthropic/...` | `ANTHROPIC_API_KEY` |
| `openai/...` | `CODEX_API_KEY`, or `OPENAI_API_KEY` when `CODEX_API_KEY` is unset |
| `codex/...` | `CODEX_API_KEY`, or `OPENAI_API_KEY` when `CODEX_API_KEY` is unset |

Only the secret for the provider you use needs setting, plus whichever one
[threat detection](#threat-detection) reads. The workflow above uses
`openai/gpt-5`, so it needs `CODEX_API_KEY` or, when that is unset, `OPENAI_API_KEY`, plus
`COPILOT_GITHUB_TOKEN` for threat detection unless it adds the `codex` override shown there.
The examples below set `OPENAI_API_KEY`.

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
[README](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/gh-aw/README.md)
covers that path.

### Threat detection

With safe outputs declared, gh-aw runs a
[threat-detection job](https://github.github.com/gh-aw/reference/threat-detection/) over the
agent's output before applying it. That job runs only on gh-aw's built-in engines, and the
definition picks `copilot`, so detection reads `COPILOT_GITHUB_TOKEN`, or the run's
`github.token` under `permissions: copilot-requests: write`, even when the agent uses
another provider. A workflow with neither points detection at the built-in engine whose
key it already has, `codex` for `openai/` and `codex/` models and `claude` for
`anthropic/` ones, or sets `false` to skip the AI analysis. The workflow above has an
OpenAI key, so it would add:

```yaml
safe-outputs:
  add-comment:
  threat-detection:
    engine: codex
```

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

The engine records assistant messages, reasoning, tool calls and results, and usage
from a host-protected capture. Ordinary terminal output and files written by agent tools
cannot supply canonical session records.

The canonical stream in `agent-session.jsonl` includes `session.init`, `user.message`,
`assistant.message`, `assistant.reasoning`, `tool.execution_start`,
`tool.execution_complete`, and `session.result`. The result includes reported usage when
available; startup failures still record status without usage when none is available.
If finalizing the recording fails, the launcher reports the error as terminal output,
preserves the CLI exit status, and attempts a status-only result.

A recording is published only after its authenticity, sequence, and terminal result are
verified. A missing `session.result`, authenticated sequence gap, or authenticated event
after the terminal leaves the authenticated prefix in separate parser diagnostics,
without canonical status or usage. Complete failure recordings retain observed partial
content and their failure result. An abrupt `SIGTERM` or `SIGKILL` can prevent a terminal
result; such recordings remain diagnostic-only, including timeout or cancellation cases.
No tool IDs, completions, or turns are inferred.

gh-aw redacts configured secrets and MCP gateway bearer tokens from session artifacts.
The engine parser treats runtime `::add-mask::` commands as data. gh-aw applies masks
from captured output when publishing artifacts, rejecting masks that would change
event structure, status, or usage.

### What a working run looks like

The agent step starts `clai2 -a <target> -m <provider:model> -p <prompt>` with the
generated `Coder` agent or the `PAI_AGENT` target resolved by the engine. The recorder
captures assistant messages, reasoning, tool calls and results (including native tools),
and usage. Repository Python imported through
`PAI_AGENT` and any caller-supplied `PYTHONPATH` are trusted engine configuration.
Python can import from `PYTHONPATH` before the recorder starts, and `PAI_AGENT` runs in
the recorder interpreter. Point either setting only at code you trust.

The gh-aw conclusion step gathers the agent session and usage into `aw_session.jsonl` and
includes gateway and safe-output activity. The gh-aw usage artifact can be downloaded with
`gh aw logs triage --artifacts all`; raw step output is available through
`gh run view <run-id> --log` (add `--attempt N` for an earlier attempt).

The comment lands on the issue with a gh-aw footer naming the workflow and linking its run,
followed by an HTML comment recording the engine, its version and the model.

## Observability

gh-aw keeps OTLP endpoints and authorization headers on the host. The agent sandbox
does not receive those credentials. Configure `observability.otlp` for the host and
set `PAI_OTEL_EXPORT_PATH` in `engine.env` to a writable shared path when Python
agent spans are required:

```yaml
engine:
  id: pydantic-ai
  env:
    PAI_OTEL_EXPORT_PATH: /tmp/gh-aw/pydantic-ai-private-otel.jsonl
observability:
  otlp:
    endpoint:
      - url: https://logfire-us.pydantic.dev
        headers:
          Authorization: ${{ secrets.LOGFIRE_TOKEN }}
```

When host OTLP observability is configured, the installer adds Logfire and the official
OTLP file exporter. For standalone local-file export without host OTLP configuration,
preinstall `logfire>=4.39.0` and `opentelemetry-exporter-otlp-json-file>=0.64b0` in a host
workflow step. The agent-only export-path setting does not reach the host installer.
The launcher instruments Pydantic AI and writes OTLP JSON trace payloads to the file without network export.
Ended spans are flushed as they finish, including delegate and failed-request spans.
The launcher attaches `TRACEPARENT` and preserves `OTEL_RESOURCE_ATTRIBUTES`, so
the exported spans retain their workflow parentage and identity. A hard process kill
can still lose spans that have not ended.

Add an always-run host `post-steps` entry that sends each line unchanged to the
configured collectors' `/v1/traces` endpoint using their host-held headers. Preserve
long JSON-valued attributes and avoid artifact mirroring. Remove the file afterwards. The repository's
[shared workflow configuration](https://github.com/pydantic/pydantic-ai/blob/main/.github/workflows/shared/engine-zai.md)
contains that forwarding step. It reads the released helper's endpoint configuration
and per-endpoint headers, then sends the private payload without the infrastructure
exporter's attribute truncation. Keep the private span file out of artifact uploads: HTTP and
model instrumentation can contain content intended only for the telemetry backend.
Do not mirror these payloads into gh-aw's uploaded `otel.jsonl`.

The local span file is diagnostic input from the sandbox. It does not supply
canonical session usage, provider-health authority, or safe-output decisions.
Host and proxy telemetry continue to use gh-aw's own exporters independently.

The launcher uses private mode-0700 Logfire configuration and data directories under
`/tmp`, ignoring checkout settings and credentials. `console=False` prevents span
content entering captured engine stdio. In local-file mode, `send_to_logfire=False`
and metrics export is disabled. When an OTLP endpoint is directly available without
local-file mode, the launcher retains direct OTLP export and
`send_to_logfire='if-token-present'`. With neither setting, it does not configure
Logfire.

A `PAI_AGENT` module can add HTTP or MCP instrumentation after the launcher configures
Logfire. Calling `logfire.configure()` again replaces that configuration, including
the local exporter. Preserve the private directories, disabled console output,
distributed tracing and chosen export path if the agent replaces it.

### Authenticated terminal metadata

Set `PAI_RESULT_METADATA` to `MODULE:ATTRIBUTE` to include a trusted agent module's
dictionary in `session.result.data.metadata`. The dictionary must have string keys
and JSON-serializable values. Update the same dictionary during the run; the recorder
serializes it when the process finishes, including handled failures.

The engine signs this metadata with the terminal result. Ordinary stdout and
sandbox-written files cannot substitute for it. Consumers should read the verified
`agent-session.jsonl` artifact and validate their metadata schema and run identity.
Like `PAI_AGENT`, this import target is trusted engine configuration.

## Troubleshooting

**`error: invalid engine: pydantic-ai. Valid engines are: claude, codex, copilot, gemini,
pi.`** The `imports:` line is missing. Add
`pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@main`.

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
[`gh-aw/README.md`](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/gh-aw/README.md)
in this repository.
