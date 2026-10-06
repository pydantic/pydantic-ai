# Track A: clai2 under Logfire fleet control (hackathon)

## Try it (colleagues)

One command. It installs this branch, enrols your clai2 in the Pydantic Logfire project, and uses the AI Gateway:

```zsh
CLAI2_MANAGED_URL=https://logfire-eu.pydantic.info/logfire/clai2 uvx --refresh-package pydantic-clai2 \
  --from 'git+https://github.com/pydantic/pydantic-ai@control-plane#subdirectory=src/pydantic_clai2' \
  --with 'logfire[variables] @ git+https://github.com/pydantic/logfire.git@ca08c9e15746e53a364fdf72d9156ce87492ad3a#subdirectory=logfire' \
  --with 'logfire-sdk @ git+https://github.com/pydantic/logfire.git@ca08c9e15746e53a364fdf72d9156ce87492ad3a#subdirectory=logfire-sdk' \
  --with 'pydantic-handlebars>=0.2.1' \
  clai2
```

What happens:

- **First launch.** clai2 says "Your organization manages clai2 through Logfire (logfire)" and shows a sign-in link.
  After you approve it, one sign-in sets up:
  - traces to logfire/clai2;
  - a personal API key that reads the company config and uses the AI Gateway (90 days);
  - an optional team.
- **Model.** The default becomes `gateway/anthropic:claude-sonnet-5-5`; no model keys needed.
- **Later launches.** No prompt. An expired or revoked key brings the sign-in back. Ctrl-C at the sign-in exits.
- **While managed.** Observability can't be turned off, and the Logfire project can't be switched.
- **Instead of the env var,** IT can drop `url = "…"` in `/Library/Application Support/clai2/managed.toml`
  (macOS) or `/etc/clai2/managed.toml` (Linux). The file wins over the env var. An optional `agent` key or
  `?agent=` query parameter picks the Agent Control name (default `clai2`).

This branch (`control-plane`) is `main` plus Agent Control (#9066), and it lets clai2 take company config from
Logfire managed variables:

- **Company tier: `agent__clai2`.** This is the `AgentConfig` from Agent Control, plus two sections. In detail:
  - Instruction blocks without an `id` are added to the prompt. Give one a `name`
    (`{"name": "run-tests-first", "instructions": "..."}`) to label it in notices and adoption
    (`instruction:<name>`). Blocks with an `id` still override the code block with that id.
  - `skills: [{name, description, instructions}]` become deferred capabilities: the model sees each name and
    description and loads the body with `load_capability`.
  - `mcp_servers: [{name, url, headers}]` become Streamable HTTP MCP toolsets, with tools prefixed by the server
    name. A header value can reference an environment variable as `${env:NAME}`.
  - `model` and `settings` work as Agent Control defines them.
- **Catalog (marketplace): `catalog__clai2`.** It holds `{"items": [{kind, name, description, default, payload}]}`.
  - Items with `default: on` are active unless you opt out; items with `default: off` only after you opt in.
  - `plugin` items need their `module:Class` on the plugin's `allowed_catalog_plugins` list.
  - `/catalog` lists everything, and `/catalog enable|disable NAME` toggles one item for you.
- **Notices.** At session start and at every prompt, clai2 prints what Logfire pushed since you last looked
  (`◆ Added company skill from Logfire: pr-shepherd`). The status row repeats the latest one. Updates arrive
  through the Logfire variables SSE stream, so no restart is needed. A change applies from your next prompt.
- **Identity.** Every span of a run carries your `user.email` (from the `user_tag` setting) and `clai2.team`
  (from `team` or `CLAI2_TEAM`) as baggage. Agent Control targets on the same values, so per-team overrides and percentage rollouts work through
  normal variable targeting.
- **Agent name.** The stock agent is named `clai2`, so Logfire groups every user's runs as one agent.

## Running it against EU staging (logfire/clai2)

```bash
uv run --env-file .env clai2 -m gateway/anthropic:claude-sonnet-5-5
```

You need:

- **An API key that can read variables.** It comes from `LOGFIRE_CLAI2_API_KEY` (or `LOGFIRE_API_KEY`), or from
  the observability plugin's `api_key` setting, which names a `/keys` entry. Without one, fleet control is off and
  clai2 behaves as before.
- **The staging URL.** Set the observability plugin's `base_url` to `https://logfire-eu.pydantic.info`, which
  `/plugins configure observability` does when you set up the staging project, or export `LOGFIRE_BASE_URL`.
- **A write token**, as usual, so traces go to logfire/clai2.
- **Optionally a team:** `/plugins` → observability settings → `team`, or edit the saved settings.

To publish a new version of a variable without the UI, run
`uv run --env-file .env python publish.py agent__clai2 value.json` (the script is in the hackathon scratchpad).

## Attributes on the spans (what the miner and the UI read)

| Where | Attribute | Meaning |
|---|---|---|
| Every span of an agent run | `user.email`, `clai2.team` | Who ran it (baggage) |
| Every span of an agent run | `clai2.prompt.source` | `typed`, `plugin` (automated continuation), `headless` (`-p`), or `subagent` |
| Every span of an agent run | `clai2.fleet.active` | The company and catalog items in force, as sorted `kind:name` keys joined by commas |
| Every span of an agent run | `logfire.variables.agent__clai2`, `logfire.variables.agent__clai2.version` | Label and version of the company config this run used |
| Every span of an agent run | `clai2.policy.version`, `clai2.catalog.version` | The `agent__` and `catalog__` versions the run used |
| Every span of an agent run | `logfire.managed.applied_sections` | Agent Control sections applied, such as `instructions` |
| `prompt submitted` UI record | `kind`, `prompt`, `clai2.prompt.source` (`typed`), `user.email`, `clai2.team`, `agent_session_id` | One typed prompt, joinable without the session root |
| `agent_control_config_hint` | `agent_control.variable_name`, `.agent_name`, `.baseline`, ... | The code baseline, once per process |
| `agent_control_config_hint` | `agent_control.client_features` | `["named_instructions", "catalog"]` for clai2; plain `AgentControl` reports `["named_instructions"]` |
| `Resolve variable agent__clai2` | `name`, `label`, `version`, `reason`, `targeting_key` | Each resolution |

## Policy (`agent__clai2.policy`)

- **Rules** are applied by the harness `PolicyRules` capability (`pydantic_ai_harness.policy`), using the
  `before_tool_execute` hook.
  - `observe` records a `policy decision` span and lets the call run.
  - `enforce` + `deny` skips the call and tells the model why.
  - `enforce` + `ask` asks in clai2's `ask_user` picker. With no terminal, the call is denied.
- **`match.command`** globs match each command segment, split on `&&`, `||`, `;`, `|` and newlines. A glob that
  contains a literal `|` matches the whole command.
- **`monty` rules** run in Monty with `tool_name` and `args` in scope and no host functions, with a 2 s timeout.
  If a rule fails, it lets the call through in `observe` mode and blocks it in `enforce` mode, recording `error`.
- **`mcp.allow`** applies to the user's and the project's MCP servers. Servers Logfire pushes are always allowed.
- **`locked`** keys can't be turned off: `/catalog disable` and `/plugins disable` answer "locked by your
  organization".
- **Telemetry:**
  - `policy decision` spans carry `clai2.policy.rule`, `.mode`, `.action`, `.outcome`, `.subject`,
    `gen_ai.tool.name`, and also `.error` and `.monty_ms` for Monty rules.
  - Every run span carries `clai2.policy.version`, `clai2.catalog.opted_out` and `clai2.policy.locked_ok`.
  - The config hint's `agent_control.client_features` includes `policy`.

## In the terminal

- **Notices:** pushed changes arrive as one "◆ From your organization (config vN)" panel per batch. It lists
  `+`, `~` and `-` items with their description, plus the item's `why` and `pushed_by` when present, and a link to
  the agent in Logfire. Changes show while the prompt is idle, checked every 5 s, or every 2 s with
  `CLAI2_FLEET_DEMO=1`.
- **Consent:** a pushed MCP server or plugin asks once before it turns on, as soon as it arrives:
  "Logfire wants to connect MCP server X at URL and send $Y". The answer is remembered per name, target and
  set of environment variables.
  - `${env:NAME}` in pushed headers expands only for names matching the local `fleet_env_allow` or the pushed
    `policy.env_allow`.
- **`/catalog`:** an interactive picker of "From your organization" items and "Optional add-ons", showing on/off,
  a lock, "new" and "declined". Enter toggles an add-on. `/catalog why NAME` says where an item came from.
- **Policy blocks:** a "◆ Policy" panel with the rule's description, its name and a "Learn more" link. The
  model gets the same text, plus an instruction not to retry and to suggest a safe alternative.
- **Status row:** `Logfire config vN`, plus how many pushed items await your OK.
- **Test sessions:** set `CLAI2_TEST=1` (tags every run `clai2.test=true`) and point `agent_control_name` at
  `clai2_test`, which uses the `agent__clai2_test` and `catalog__clai2_test` variables.
- **Demo reset:** `hackathon/demo_reset.py` prints its plan and diff, and writes only with `--yes`.

## Known limitations

- **The cache prefix breaks.** A skill or instruction pushed mid-conversation changes the instructions and the
  deferred-capability catalog, which busts the provider prompt cache from that turn. The planned fix is to
  deliver the change as a delta message instead (#8188).
- **The read-variables key comes from sign-in.** Setup exchanges the device sign-in for a personal API key,
  using RFC 8693 token exchange at `/api/oauth/token`. The key is scoped to `project:read_variables` and expires
  after 90 days. clai2 doesn't refresh it: run setup again when it expires.
- **No team from groups yet.** Logfire has no group API a user token can read: groups are admin-only in the UI API.
  Defaulting the team from SCIM groups would need a backend change, such as `groups` on `/v1/account/me`.
- **Opt-ins are per machine.** They live in `~/.config/pydantic-clai2/logfire/fleet_state.json`, keyed by email.
