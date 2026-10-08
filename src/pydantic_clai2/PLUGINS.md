# CLAI plugins

A plugin is a Python file that teaches CLAI new tricks: react when something
happens, add a `/command`, give the agent a tool, or change how a tool's output
is shown on screen.

Everything a plugin can do goes through one object, the `PluginHost`. There is no
global registry to import and no magic file to name. You get a `host`, you tell it
what you want, you're done.

CLAI groups its shell implementation into `cli/`, `config/`, `runtime/`,
`models/`, `plugins/`, and `ui/`. Built-in plugin implementations live in
`builtin_plugins/`. These are source directories, not extra plugin APIs:
continue importing `PluginHost` from `pydantic_clai2.plugins` and `Command`
from `pydantic_clai2.commands`.

## Startup

`clai2 --help` parses arguments without loading the agent or plugins. Interactive
startup defers model menus and provider integrations until you open those menus,
log in, or run a prompt. Once the prompt is ready, a background thread imports
them, so the first prompt usually finds them loaded; if it arrives sooner, it waits
for the rest of those imports. Enabled plugins still load before the first prompt;
their initialization contributes to startup time.
`/login` offers both Codex and GitHub Copilot without loading their integrations for
completion. Copilot requests use your saved login through the lazy provider resolver.

## Connect MCP servers

`/mcp` is the front door for MCP servers, modelled on Code Puppy's `/mcp`. The
built-in `mcp` plugin provides it and includes the MCP client; local server
programs and their runtimes (`npx`, `uvx`, ...) still need to be installed
separately.

```text
/mcp                                  status dashboard (also /mcp list, /mcp status)
/mcp install                          add a server in the form below
/mcp start NAME | stop NAME | restart NAME | start-all | stop-all
/mcp status NAME                      target, env references, tools, last error
/mcp tools NAME                       connect and list the tools the agent sees
/mcp logs NAME [LINES]                server stderr and lifecycle events
/mcp auth NAME [logout]               sign in to an OAuth server again, or sign out
/mcp edit NAME                        the same form, prefilled
/mcp remove NAME
/mcp trust [status|accept|revoke]     load this repository's .clai/mcp_servers.json and .mcp.json
/mcp help
```

`/mcp install` opens Code Puppy's custom server form:

```text
 Add Custom MCP Server
 Server Name: docs              | Name   docs
 Server Type: http              | Type   http - Streamable HTTP endpoint ...
 URL: https://mcp.example.com   |
 OAuth sign-in: on              |
 JSON Configuration (valid)     | {"type": "http", "url": "https://...", "auth": "oauth", "timeout": 330}
 Load example for http          |
 Save & Install                 | Configuration is valid
 Cancel                         |
```

- **Server Type** switches between `stdio` (a local program), `http` (Streamable
  HTTP), and `sse` (Server-Sent Events, used by older servers). An untouched
  example follows the type; a configuration you have edited is kept.
- **URL** (for `http` and `sse`) or **Command** (for `stdio`, the program and its
  arguments, split like a shell would but run without one) is typed straight
  into the form and written into the JSON.
- **JSON Configuration** opens `$VISUAL` or `$EDITOR` (default `vi`) on the
  server's JSON, with a one-line input as a fallback when no editor runs. The
  preview says whether it is valid and why not.
- **OAuth sign-in** appears for `http` and `sse` servers. Switching it on sets
  `"auth": "oauth"`, allows 330 seconds for the handshake so there is time to
  sign in, and drops any `Authorization` header. FastMCP runs discovery, dynamic
  client registration, PKCE, and a browser sign-in through a loopback callback
  when the server connects. The access and refresh tokens and the registered
  client go to your OS keyring (the `mcp-NAME` entry under `pydantic-clai2`, or a
  private `credentials-mcp-NAME.json` when no keyring exists), so restarting CLAI
  reuses or refreshes them instead of signing in again. Tokens belong to the URL
  they were issued for: changing the URL signs in again. OAuth needs `https`,
  except for loopback servers.

The dashboard shows each server as `running` (connected), `ready` (enabled;
connects on the next prompt), `stopped`, or `error` (a failed connection, or a
referenced environment variable that is not set). Saved and enabled servers are
available to the agent from the next prompt; `/mcp start` is not required, but
connects now and lists the tools. Once connected, a server stays connected
between prompts. A server that cannot connect is marked `error` and left out of
the prompt instead of failing it; `/mcp logs NAME` shows why. `/mcp stop`
disconnects the server and disables it until `/mcp start`. The model sees tools
prefixed by server name, for example `local_search`.

Only install servers you trust. Stdio servers run programs with your user
permissions, launched as an executable plus arguments without a shell. Remote
servers receive tool arguments and can return untrusted content. HTTP redirects
are rejected; configure the final endpoint URL.

### Where servers are stored

`/mcp install` and `/mcp edit` write `mcp.json` in the CLAI config folder
(`~/.config/pydantic-clai2/`, or under `$XDG_CONFIG_HOME`). The file is created
readable only by you. Do not put secrets in it: write `$VAR` or `${VAR}` in an
`env` or `headers` value and CLAI fills it from its own environment when it
connects, so the file holds only the reference; a server whose variable is
unset shows as `error` until you set it. Stdio server stderr goes to `mcp_logs/NAME.log` next to
`mcp.json`, which `/mcp logs` reads. The file shape is:

```json
{"servers": {"local": {"type": "stdio", "command": "uvx", "args": ["my-mcp-server"], "env": {"TOKEN": "$MY_TOKEN"}},
             "remote": {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer $API_KEY"}}}}
```

Stdio servers also accept `cwd`; `http` and `sse` servers accept `auth: "oauth"`;
every server accepts `timeout` (seconds for the initialize handshake) and
`enabled: false`. Names start with a letter and contain letters, digits, and
hyphens. Underscores are reserved for the separator so server/tool name pairs
cannot produce the same name.

### Project servers and trust

A repository can commit either or both of these project files. Each is found by
looking in the working directory, then each parent up to the git root:

| Path | Shape |
| --- | --- |
| `.clai/mcp_servers.json` | `{"servers": {...}}`, the same shape as your `mcp.json` |
| `.mcp.json` | Claude Code's project scope: `{"mcpServers": {...}}` |

A `.mcp.json` written for Claude Code loads as is in the common case:

```json
{"mcpServers": {"github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                            "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},
                "docs": {"type": "http", "url": "https://example.com/mcp"}}}
```

As in Claude Code, `type` can be left out for a stdio server (an entry with a
`command`); `http` and `sse` servers name their `type`. Each server otherwise takes
the keys described above. Differences from Claude Code: `$VAR` and `${VAR}` are
expanded only in `env` and `headers` values (not in `command`, `args`, or `url`),
`${VAR:-default}` is not supported, server names cannot contain underscores, and
keys CLAI does not know fail loudly with the file's path. Claude Code's user and
local scopes (`~/.claude.json`) are not read.

Because a stdio server runs a program, project servers do not load until you run
`/mcp trust accept`, which accepts every project file found. Trust is
stored in your `mcp.json`, keyed by each file's path and a SHA-256 of its
contents: any change to a file unloads its servers until you accept again, and
a repository cannot trust itself. A symlinked project file or `.clai` folder is
never trusted, so a repository cannot point at a file you trusted elsewhere.
`/mcp stop` on a project server lasts for the
session; edit the project file to change it permanently. When a name exists in
more than one place, your own server wins, then plugin settings, then
`.clai/mcp_servers.json`, then `.mcp.json`.

### Plugin settings and gaps

Servers configured the older way, as plugin settings
(`/plugins add mcp pydantic_clai2.mcp '{"servers": {...}}'`), still load and show
up in `/mcp` with source `plugin`; change or remove them through `/plugins`.
The older `"transport"` key is still read as `"type"`.
`/plugins disable mcp` removes both the tools and the command.

Compared with Code Puppy, CLAI has one agent, so there are no per-agent server
bindings and no `silence-warning` command: every enabled server is offered to
the agent. There is no built-in server catalog, so there is no `/mcp search`.
The plugin adds no telemetry beyond core's tool spans, and does not enable MCP
sampling or native provider-side MCP.

## On-demand authoring help

The default CLAI agent exposes `read_clai_customization_guide`. When you ask for
customization, its instructions tell it to read this guide first. Only the short
hint and tool description are present initially; the bundled text is read on tool
invocation. That hint is ordered after the guidance the plugins contribute and
before the repository instruction file, so it never leads the system prompt. Reading it needs neither a checkout nor a network connection.
This tool needs no arguments and ignores extra arguments supplied by a model.
Other tools keep their existing validation.

The [bundled guide](pydantic_clai2/customization.md) includes examples for
commands, hooks, settings, renderers, custom Termflow menus, and custom model
launchers. It also names current limits: PluginHost does not replace the prompt
editor or change the built-in model catalog's entries. Those need a
custom agent launcher or a source change, as explained in the guide.

Custom agents can opt in with `customization_guide()` from
`pydantic_clai2.customization`. The tool only returns documentation; it does not
write files, activate plugins, or grant permission to execute generated code.
Keep the bundled guide aligned with this contract when changing plugin APIs.

## Default code rendering

The default stream renderer buffers fenced code until the fence closes or the
text part ends, then highlights the whole block. This preserves multiline lexer
context. Unlabelled and Markdown fences stay literal; unknown languages use
plain text. Long code lines wrap to the terminal width. Prose still streams line
by line. A plugin renderer that handles a text event replaces this default
rendering for that event.

## Tool retries

CLAI defaults to three retries per tool call. `/set run.tool_retries N` changes
that default for subsequent turns; `N` must be a non-negative integer, with `0`
disabling retries. Explicit retry limits on a tool or toolset take precedence.
Output-validation and HTTP transport retry budgets are unchanged.

## Credentials

CLAI's `/login openai-codex`, `/login github-copilot`, and the vllm and openrouter connections store tokens
in the configured keyring backend, not plugin settings. Plugins that need an API key, such as
[`posthog`](#posthog-posthog-analytics-signed-in-for-clai), keep it in `/keys` and save only its name. Large token bundles use
multiple entries to fit Windows Credential Manager's size limit. When no keyring
backend exists, credentials go to a per-account `0600` file under the user's CLAI config
directory instead. None of this changes plugin APIs. See
[Codex authentication](README.md#codex-authentication) for storage and security
details.

### GitHub Copilot subscriptions

```bash
uv run clai2
```

Run `/login github-copilot`, then open `/model add` and choose `github-copilot`.
The provider menu also starts login when no credentials exist. No application
registration or client ID configuration is required. CLAI supplies the same
[public Copilot OAuth client ID as Pi](https://github.com/earendil-works/pi/blob/fde38ed7c2f64434beffc6c0ec3b9994cb89ae23/packages/ai/src/auth/oauth/github-copilot.ts#L10-L11)
and requests `read:user` access to your GitHub profile. This identifies the existing
Copilot OAuth application, not a separately registered CLAI application.
`GITHUB_COPILOT_CLIENT_ID` is an optional override for your own device-enabled OAuth
application; unset or blank uses the bundled default. The workspace pins the merged
Pydantic AI device-flow implementation until its release.

Login prints a code and `https://github.com/login/device`, then starts polling.
Open the link on this or another device and approve only your own session's code.
CLAI does not launch a browser, so a text browser cannot block login or take over
an SSH terminal. Ctrl-C stops polling; GitHub controls expiry. There is no localhost callback.
GitHub authorization does not establish Copilot access: the menu queries your
account's catalog and keeps only picker-enabled `/chat/completions` models.
The shared model menu includes details and `Ctrl+S` settings. Subscription and
organization policy still control inference access. A known ID also works with
`/model github-copilot:claude-haiku-4.5`.

The `github-copilot` keyring account is separate from Codex and named API keys.
Without a keyring, CLAI reports the plaintext `credentials-github-copilot.json`
fallback, created with mode `0600`. Tokens and issuance time stay out of settings,
history, and login output. Expiring tokens require another `/login github-copilot`;
there is no automatic refresh. Failed or cancelled authorization preserves the
previous login.

Saved login takes precedence over `GITHUB_COPILOT_API_KEY`,
`GITHUB_COPILOT_API_TOKEN`, and `COPILOT_GITHUB_TOKEN`, checked in that order when
no login is saved. Copilot login does not read `GH_TOKEN`, `GITHUB_TOKEN`, the
`GITHUB_TOKEN` key in `/keys`, or another application's token files; that key
belongs to the separate [`github` plugin](#github-tools-from-githubs-hosted-mcp-server). This is shell-owned authentication, not a plugin API.
Core owns inference and its telemetry; CLAI adds no login-specific spans.
Bare `/login` asks which sign-in to run.

## Herdr integration

The built-in `herdr` plugin (`pydantic_clai2.builtin_plugins.herdr`) starts disabled.
Run `/plugins enable herdr` inside a [herdr](https://herdr.dev) pane. Use
`/plugins disable herdr` to release the pane and stop reporting. No herdr-side
integration install is needed. The plugin requires `HERDR_ENV=1`,
`HERDR_SOCKET_PATH`, and `HERDR_PANE_ID`; optional `HERDR_TAB_ID` enables tab titles.
Outside herdr, or on Windows, it contributes nothing and starts no worker.

It reports `working` while agent runs are in flight, `blocked` while `AskUser`
waits for an answer, and `idle` otherwise. Failed and cancelled runs return to
`idle` too. Nested runs are counted; concurrent question waits are tracked by
request ID. Tool names supply activity text, never tool arguments or results.
User-opened menus do not report `blocked`. Other approval UIs are not tracked:
CLAI2 has no universal approval-wait event. Herdr owns attention notifications;
this plugin sends none. To avoid duplicate desktop alerts, separately disable
CLAI2's `notifications` plugin if you prefer herdr's alerts.

Persisted sessions report their stable conversation ID and SQLite database path,
not a per-run ID. Resume manually with `clai2 --resume SESSION-ID` using the same
CLAI2 config directory. Automatic restoration by herdr is not verified. Hosts
without CLAI2 session persistence still report state and token metadata.

Metadata has a 24-hour TTL and reports `$model`, `$tokens` (retained-history
input plus output tokens), and `$context` (percentage from the compaction
plugin's context events, omitted when unknown). Add those fields to your herdr
sidebar's `rows_by_agent.clai2` configuration to display them. No prompts,
answers, or tool contents are sent; session IDs, database paths, and conversation
titles are sent to the local herdr socket.

The pane title follows the persisted conversation title, including background
naming and manual renames, checked every two seconds. Each metadata update keeps
the current title. Only single-pane tabs are renamed. The original tab label is
restored on session changes or clean unload, but a manually renamed or shared
tab is left alone. An abrupt exit may leave the last tab label in place.

Socket IO uses a plugin-owned daemon worker with bounded, latest-wins mailboxes.
State and session reports take priority over activity and metadata. Requests
retry up to three times with the same sequence number; missing sockets and
server errors are nonfatal. Unloading cancels and drains the title watcher,
discards queued work, and attempts one release with bounded shutdown. A departed
or unresponsive herdr may miss reports; they do not fail the agent turn.

## Desktop notifications

The default-enabled `notifications` plugin (`pydantic_clai2.builtin_plugins.notifications`)
observes `turn_end` for completed and failed turns and `AskUserRequestedEvent`
before the answer picker waits. Cancelled turns do not notify. It registers no
tools or instructions and has no plugin settings. Its title is `CLAI2`; its
messages contain only generic status text, never conversation content or errors.

macOS uses `/usr/bin/osascript`; enable Script Editor notifications in System
Settings > Notifications. Linux uses `/usr/bin/notify-send` when installed and a desktop
notification service is available. OS permissions and Focus settings determine
delivery. Windows, SSH, headless mode, and redirected output are skipped. Local
tmux needs no passthrough because delivery uses the OS, not terminal escapes.
The plugin does not detect focus and submits notifications even in the active
terminal. Submission is awaited with a two-second timeout; missing services,
nonzero exits, and timeouts are nonfatal. Cancellation still propagates. It emits
no notification-specific telemetry.

`/plugins disable notifications` persists an off override. Use
`/plugins enable notifications` to load it again or `/plugins remove notifications`
to restore the built-in default. Normal plugin unloading discards its handlers;
there are no background workers to stop.

## Observability: default agent tracing

The built-in `observability` plugin (`pydantic_clai2.builtin_plugins.logfire`) is enabled by default in
the stock CLI. It registers Pydantic AI's `Instrumentation` capability with an
isolated Logfire instance, not process-wide agent instrumentation or custom tracing
hooks. Agent/model/tool spans include timing, token usage, failures, text content,
and binary image attachments by default, including retained history used by
later turns. This may export source code, file contents, and screenshots; verify
the configured telemetry destination first.

Startup plugin load failures reported in the terminal, including missing optional dependencies,
are recorded with their exception and traceback through this same instance. This does not require
`ui_events`. Reporting waits until startup loading finishes, so failures before the
observability plugin loaded are included. Disabling observability stops this reporting;
the existing terminal messages remain.

Credentials are read from `LOGFIRE_TOKEN` or the SDK's `logfire_credentials.json`
in `$XDG_CONFIG_HOME/pydantic-clai2/logfire/`, defaulting to
`~/.config/pydantic-clai2/logfire/`. SDK configuration is read only from that user
directory too. Repository-local configuration/credentials and the SDK's
`LOGFIRE_CONFIG_DIR`/`LOGFIRE_CREDENTIALS_DIR` overrides are ignored. Relative
`XDG_CONFIG_HOME` values fall back to `~/.config`. A checkout cannot select the
telemetry destination through its own files. Without credentials the default
`if-token-present` mode does not export to Logfire or start interactive setup. Console logging is disabled. Other SDK configuration,
such as explicit OTLP exporters, still applies. The SDKs' own log messages, such as
failed or retried exports, never print to the terminal; a logging handler that a
plugin configures still receives them.

Previously named `logfire`, this plugin keeps existing enabled/disabled choices,
settings, and saved token references. No reconfiguration is needed. Old commands
and project or drop-in declarations using `logfire` refer to the same plugin,
not a second tracing instance. If both names were saved, the `observability`
declaration takes precedence; edits and removal apply to that one shared entry.

Manage it with `/plugins disable observability`, `/plugins enable observability`, or
`/plugins reload observability`. `/plugins configure observability` (or `c` on
`observability` in `/plugins`) opens its settings menu: **Logfire project** runs the
setup described below, and the other rows edit the options listed here. Each edit
saves at once, and the plugin is loaded again when you close the menu, so the next
run uses it. Without a chosen project, that row notes whether `LOGFIRE_TOKEN` or a
credentials file was found. Scripts can replace the declaration instead:

```text
/plugins add observability pydantic_clai2.builtin_plugins.logfire '{"include_content": false, "include_binary_content": false}'
```

Options are `service_name` (default `pydantic-clai2`), `include_content` and
`include_binary_content` (both default `true`), `httpx` (default `false`), and `send_to_logfire` (default
`"if-token-present"`, or `false`). The explicit plugin option takes precedence
over `LOGFIRE_SEND_TO_LOGFIRE`. Tokens are not accepted in plugin settings;
`token` takes only the name of a `/keys` entry (`{"name": "CLAI2_LOGFIRE_TOKEN"}`),
whose write token then replaces `LOGFIRE_TOKEN` and the credential file, so its
project receives the telemetry. If that key is missing, the plugin warns and
exports nothing rather than falling back to another project.
Content flags do not suppress all metadata: tool names and definitions may still
be recorded. Logfire's usual scrubbing is enabled. Set `httpx` to `true` to
instrument `httpx` and `httpx2` requests process-wide while this plugin is loaded.
With `include_content=true`, Logfire captures HTTP headers and request and response
bodies too; with it off, those are not captured. HTTP instrumentation is removed
on unload unless it was already installed by another owner.

`base_url` (an https origin) is the Logfire to send to; unset, the SDK uses
`LOGFIRE_BASE_URL`, else the region the token names. The **Logfire project** row
sets `token`, `base_url`, `account`, and `send_to_logfire` for you (`R` on it
clears `token`, `base_url`, and `account` again): it asks where traces go, runs
Logfire's own device sign-in there (the one behind `logfire auth`, not
`logfire_mcp`'s MCP OAuth, whose tokens only the MCP server accepts), reads your
account's email, lists the projects you can write to, and saves a new write
token for the one you pick in `/keys`. The sign-in token is used only during
setup. The flow lives in `pydantic_clai2.builtin_plugins.logfire_setup`.

Agent runs and UI records nest under a `CLAI session` root whose
`agent_session_id` is the saved conversation ID. `/clear` selects a new root;
`/resume` reuses that conversation's root if this plugin instance already opened
it. Unloading the plugin ends its roots; reloading starts new traces with the
same saved conversation IDs.

`user_tag` (default `logfire-account`) tags each session root, and only the
root, with your email, as a Logfire tag and the `user.email` attribute.
`logfire-account` uses the email in `account`, the account that signed in during
the **Logfire project** setup, while `token` still names the key that setup
saved. A token from elsewhere, a setup made before this setting existed, or a
server that does not report the email leaves roots untagged until setup runs
again. `git-email` uses `git config user.email` (Git is queried only then), and
`false` turns the tag off. Set **User tag** in
`/plugins configure observability`.

`ui_events` (default `true`; set `false` to opt out) also records CLAI's UI interactions on the same
instance, as spans and logs in the `clai2` instrumentation scope, which session
roots and plugin load failures share: menus opened and how they closed, slash
commands, `/set` changes, plugin actions, `/keys` saves and prompts, prompt
submissions, steering, interrupts, completions, and session start, clear, and
resume. Attributes carry names and listed choices, never typed values or
secrets. The exception follows `include_content`, like agent spans: while it is
on, a `prompt submitted` record also carries the prompt's text as `prompt`, cut
to 64,000 characters (`chars` keeps the full length). A `!` line records only
its kind and length: it never reaches the agent, and it can hold a secret such
as `!export KEY=...`. A slash command records only its name, never its
arguments, because any plugin can add a command and its arguments can hold a
secret. The chokepoints live in `pydantic_clai2.ui.telemetry`, and
`run_worker` opens every menu's span, so a new menu is covered without extra
code.
With `ui_events` on, the attributes that only hold names (`command`, `menu`,
`setting`, `key_name`, ...) are exempt from scrubbing, since names like
`OPENAI_API_KEY` or `sessions.naming` would otherwise be redacted. So is
`prompt`, which agent spans already record unscrubbed.

Unload flushes and shuts down only this plugin's providers. Reload creates a new
instance. The supplied agent and global providers are unchanged, and the existing
global propagator is preserved. The SDK may install shared executor propagation
helpers; those hooks are not removed on unload. Core's normal
instrumentation precedence applies: the plugin's explicit per-run capability
wins while enabled; disabling it restores the supplied agent's own tracing
behavior. Custom launchers must pass `builtin_plugins=DEFAULT_PLUGINS` to opt in
to stock built-ins. See [telemetry](README.md#telemetry-and-references).

## Linear: issues and projects

The built-in `linear` plugin (`pydantic_clai2.builtin_plugins.linear`) gives the agent the tools
of Linear's hosted MCP server through harness
[`Linear`](../pydantic_ai_harness/pydantic_ai_harness/linear/README.md). It starts disabled. Turning
it on (`/plugins enable linear`, or Space in `/plugins`) opens its settings menu,
and `/plugins configure linear` (or `c` in `/plugins`) opens it again later:

```text
 Linear settings
 search: (type to filter)

 > Sign-in                  API key from /keys
   API key                  LINEAR_API_KEY
   Access                   Read-only
   Server instructions      Include
   Save & close

 type to filter - Enter edit - R reset - Esc close
```

| Row | Choices | Default |
|---|---|---|
| Sign-in | an API key from `/keys`, or browser sign-in (OAuth) | API key |
| API key | a `/keys` entry, picked from a searchable list, or a new key typed into a masked prompt | `LINEAR_API_KEY` |
| Access | read-only, or read and write | read-only |
| Server instructions | pass Linear's own MCP instructions to the agent, or leave them out | include |

Enter edits a row, R resets it to the default, and Esc backs out of a picker
without changing anything. Each change is saved as you make it, so **Save & close**
and Esc both just leave the menu, and the plugin is reloaded with the new settings.
Read-only is the default because tools that create or change issues act on a
workspace your team shares. These rows are the settings harness `Linear` takes
from a user. Linear's hosted server has a single URL (with a read-only variant)
and takes the workspace from the account you sign in with, so the menu has no
base URL or workspace field.

The API key lives in [`/keys`](#saved-api-keys), not in plugin settings, which
are stored in plaintext. On the API key row, pick any saved key (one entry can
serve several plugins), or choose **Enter a different API key**. A new key is
saved in `/keys` as `LINEAR_API_KEY`; if that name already exists, CLAI asks
before replacing it, since other plugins may use it. The plugin stores only the
key's name and looks the key up at the start of every run, so replacing the value
in `/keys` takes effect on the next run. If the key is missing, loading the plugin
prints a warning and each run fails with an error naming `/plugins configure linear`,
rather than running without Linear. Harness's `LINEAR_ACCESS_TOKEN` environment
variable is not read.

With browser sign-in, the key row is hidden. Tokens go to the keyring the way
`/mcp` OAuth tokens do, and `/linear logout` signs out.

The settings are also plain JSON, for scripts:

```text
/plugins add linear pydantic_clai2.builtin_plugins.linear '{"auth": "oauth", "read_only": false}'
```

## Notion: workspace tools

The built-in `notion` plugin (`pydantic_clai2.builtin_plugins.notion`) starts disabled. It adds
harness `Notion`: the tools of Notion's hosted MCP server, acting with the
permissions of the Notion account it connects as. That includes tools that
change pages.

`/plugins enable notion` loads it and opens its settings menu. To change the
settings later, run `/plugins configure notion` or press `c` on it in
`/plugins`; you never need to reinstall it. The menu is the shared field editor
`/set` uses: type to filter, Enter to edit a row, `R` to reset one, Esc to close.
Each change is saved as soon as you make it, and the plugin loads again when the
menu closes, so the next turn uses the new settings.

| Row | Default | Does |
|---|---|---|
| Key | none | the `/keys` entry to connect with: pick a saved key from a searchable list, or type a new one into a masked field; `R` clears the choice |
| Sign-in | automatic | automatic uses the key when one is chosen and otherwise signs in through the browser; key only never opens a browser; browser always does |
| Tools | read and write | read-only keeps only the tools the server marks as read-only |
| Server instructions | forwarded | whether the Notion server's own instructions reach the agent |

Notion's server has one fixed URL, and the workspace is the one the connected
account belongs to, so there is no URL or workspace row. The settings JSON uses
`auth` (`"key"`, `"oauth"`, or unset), `read_only`, and `include_instructions`:

```text
/plugins add notion pydantic_clai2.builtin_plugins.notion '{"auth": "key", "read_only": true}'
```

### Secrets live in `/keys`

Plugin settings are plaintext SQLite, so they never hold the token, and a
declaration that tries to is rejected. The token lives in the named keystore
you manage with `/keys`:

- A new token typed in the menu is saved in `/keys` as `NOTION_API_KEY`. If
  that name already exists, the menu asks before replacing it, because other
  plugins and connections may share it.
- The plugin keeps only the key's name, in CLAI's credential store. Each run
  looks the key up again, so replacing it in `/keys` reaches every plugin that
  uses it. Several plugins can share one named key, the way the GitHub and
  Copilot integrations can both use `GITHUB_TOKEN`.
- Deleting the key makes Notion runs fail until you restore it or choose
  another. A key Notion uses cannot be renamed.
- `NOTION_API_KEY` is a label in `/keys`, not an environment variable; the
  plugin does not read `NOTION_ACCESS_TOKEN` either. Notion integration tokens do
  not work with the hosted server; use a Notion OAuth access token.

With no key chosen, the first run that connects opens your browser to sign in.
Those OAuth tokens go to the OS keyring, or CLAI's private credential file when
there is no keyring, so later launches reuse and refresh them. `/notion logout`
forgets the browser sign-in and the chosen key name; the key itself stays in
`/keys`.

The browser sign-in needs a browser on the machine CLAI runs on. For headless
runs or remote machines, choose a key and set Sign-in to key only: without a
key, CLAI warns at startup and runs fail with a message instead of waiting for a
sign-in. The plugin emits no telemetry of its own; tool calls appear in core's
spans.
## Logfire MCP: query your telemetry

The built-in `logfire_mcp` plugin (`pydantic_clai2.builtin_plugins.logfire_mcp`) gives the agent
the tools of Logfire's hosted MCP server through harness
[`LogfireMCP`](../../docs/harness/logfire-mcp.md), each named `logfire_` plus the
server's name for it (`logfire_query_run`). It starts disabled.
Turning it on (Space in `/plugins`, or `/plugins enable logfire_mcp`) loads it and
opens its settings menu; reopen the menu any time with
`/plugins configure logfire_mcp` or `c` in `/plugins`.

Every row saves as soon as you change it, so **Save & close** (or Esc) just leaves
the menu, and the plugin loads again with the new settings. Esc backs out of any
picker or text field without changing anything; `R` resets the highlighted row to
its default.

| Row | Setting | Default | Does |
|---|---|---|---|
| API key | `key` | none | name of the `/keys` entry to connect with (see below) |
| Destination | `url` | Logfire US | Logfire US, Logfire EU, or type the `https://` MCP URL of a self-hosted Logfire |
| Tools | `read_only` | read-only | offer only the tools the server marks read-only; "read and write" also allows tools that change Logfire resources |
| Server instructions | `include_instructions` | forwarded | whether the server's instructions, query guidance, and current UTC time reach the agent |
| Browser sign-in | `oauth` | when there is no key | sign in, or sign up, through the browser when no key is chosen, set, or saved; the row shows whether you are signed in |

### Keys live in `/keys`

Plugin settings are stored as plaintext in SQLite, so they hold only the key's
name, never its value. Enter on **API key** opens the same picker as
`/model add`:

- **a saved key**: any entry in [`/keys`](#saved-api-keys). Several plugins and
  connections can name one key, so one Logfire API key saved once serves them all.
- **Enter a different API key**: a masked field. The value is saved in `/keys`
  as `LOGFIRE_API_KEY`, the conventional label; CLAI asks before replacing an
  existing `LOGFIRE_API_KEY`, since other plugins may use it.
- **No API key**: clears the choice.

The plugin uses the first credential that is available:

1. The key chosen in the menu.
2. `LOGFIRE_API_KEY` from the environment.
3. A key saved in `/keys` as `LOGFIRE_API_KEY`, so saving one there is enough.
4. Browser sign-in, when it is on (the default). See below.

With browser sign-in off, the plugin looks `LOGFIRE_API_KEY` up in `/keys` on
every run, so saving it there later connects without a reload.

#### Browser sign-in

Browser sign-in uses the OAuth device flow
([RFC 8628](https://datatracker.ietf.org/doc/html/rfc8628)), as Code Puppy's
Logfire plugin does. The first run with no usable token, or `/logfire_mcp login`
at any time, prints a link and a code and opens the link:

```text
Sign in to Logfire (new users can sign up there): open https://logfire-us.pydantic.dev/auth/oauth-device?code=ABCD-EFGH
Enter code: ABCD-EFGH
Approve only the code shown here. You can open the link on another device.
```

On that Logfire page you sign in, or create an account if you have none, and
approve the code. CLAI waits up to 660 seconds (Logfire's codes last 600). No
local callback server is involved, so this also works over SSH: open the link on
any device.

- **Discovery:** the Logfire server is found from the Destination URL
  ([RFC 9728](https://datatracker.ietf.org/doc/html/rfc9728) resource metadata),
  so self-hosted Logfire works too, including an issuer with a path
  ([RFC 8414](https://datatracker.ietf.org/doc/html/rfc8414)). CLAI registers
  itself as a client ([RFC 7591](https://datatracker.ietf.org/doc/html/rfc7591))
  and uses PKCE. It registers again if the server has forgotten the earlier
  registration.
- **Binding:** every request names the Destination URL as the token's resource
  ([RFC 8707](https://datatracker.ietf.org/doc/html/rfc8707)), and the
  discovered metadata must describe that same URL (and the authorization
  server's metadata its own issuer). A token is only issued for the MCP server
  you configured, so another endpoint that names Logfire as its
  authorization server cannot receive one. Logfire rejects unknown resources
  with `invalid_target`.
- **Scopes:** read-only tools ask only for `project:read`. With Tools set to read
  and write, CLAI asks for every scope the MCP server lists (on Logfire's hosted
  servers that includes `organization:create_project`). Switching Tools to read
  and write signs in again for those scopes. If Logfire grants fewer scopes, CLAI
  says so once and keeps the sign-in, since asking again would get the same
  grant; `/logfire_mcp login` asks again when you want to.
- **Tokens:** kept per Destination URL in the OS keyring (or the private
  credential file) under the `logfire-oauth` account, so restarting CLAI does
  not mean signing in again. An expired or rejected token is refreshed; if that
  fails, the next run signs in again. `/logfire_mcp logout` forgets every
  Logfire sign-in, including one still waiting for approval, and keeps keys in
  `/keys`.
  If the keyring or file refuses to save a sign-in, CLAI says so and keeps it in
  memory for the rest of the session (logout forgets it too); the next session
  asks you to sign in again.

#### Saved keys

A saved key's value is read at the start of each run, like a model connection's.
Replacing it in `/keys` applies from the next turn. `/keys` does not stop you
deleting or renaming a key that plugin settings name; the plugin's runs then fail
with the old name and `/plugins configure logfire_mcp` as the fix, until you pick
a key again.

When the key the plugin depends on is missing from `/keys`, the plugin still
loads, so the menu stays reachable. It prints a warning when the session starts,
and each run fails with that message instead of reaching Logfire.

The label is `LOGFIRE_API_KEY`, the variable `LogfireMCP` reads, and not
`LOGFIRE_TOKEN`. `LOGFIRE_TOKEN` is the write token the
[`observability` plugin](#observability-default-agent-tracing) sends traces with, and it
cannot query the MCP server. The `observability` plugin keeps reading it from the
environment or the Logfire SDK's credential file.

The same settings can be given as JSON, which is validated the same way:

```text
/plugins add logfire_mcp pydantic_clai2.builtin_plugins.logfire_mcp '{"url": "https://logfire-eu.pydantic.dev/mcp"}'
```

The key's own scopes still decide which projects and actions are allowed. If you
enabled `logfire_mcp` from the old harness catalog, that saved declaration still
takes this plugin's place; `/plugins remove logfire_mcp` switches to this one.
The plugin emits no telemetry of its own; tool calls appear in core's spans.

## Where plugins live

Plugins are trusted Python code. Drop-in files execute automatically at startup;
this directory is an executable startup configuration, not a sandbox. The default
Coder runs as your OS user and can modify it, just as it can modify your shell
startup files. Use a separate OS identity or sandbox for untrusted agent work.

Three ways to install one:

1. Drop a `.py` file (or a package folder) into
   `$XDG_CONFIG_HOME/pydantic-clai2/plugins/` (default `~/.config/pydantic-clai2/plugins/`).
   Its name is the file name without `.py`. CLAI creates the folder at startup, so it
   is there to copy into after the first run.
2. Point CLAI at anything importable, from the shell or from inside CLAI:

   ```sh
   clai2 plugins add notify my_package.notify
   /plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": true}'
   ```
3. From inside CLAI, clone a trusted Git repository:

   ```text
   /plugins add https://github.com/your-org/my-plugin.git
   /plugins add git@github.com:your-org/my-plugin.git
   ```

   Git must be installed and available on `PATH`. On Windows, CLAI resolves
   `git.exe` only from absolute `PATH` directories outside the working directory.
   The repository must have an `__init__.py` or `plugin.py`
   at its root that defines one public `Plugin` subclass. If both exist,
   `__init__.py` is used. Relative imports can load other files from the checkout.
   Install the plugin's dependencies in CLAI's Python environment first; this
   command does not install packages or run build scripts.

   CLAI clones the default branch into `plugins/_git/my_plugin/` under its
   configuration directory. The repository name becomes the plugin ID, with
   hyphens and dots replaced by underscores. Names must start with a letter and
   contain only letters, digits, dots, hyphens, or underscores. Existing plugins
   and checkouts are never replaced. Network URLs must use HTTPS or SSH (including
   `git@host:path`); these may also start with `git+`. SCP-style URLs require
   `user@host:path`; use `ssh://host/path` for an SSH alias without an explicit user.
   Local `file://` URLs work too.
   Plaintext `http://` and `git://` transports are rejected because they cannot
   authenticate the plugin code being downloaded.
   Authentication uses your existing Git credentials without terminal prompts.
   CLAI removes failed or cancelled clones. If the operating system prevents
   cleanup, clear the leftover checkout directory before retrying.
   If the plugin itself fails to load,
   its checkout and declaration remain so you can fix it and `/plugins enable my_plugin`.
   To update, run `!git -C <checkout-directory> pull --ff-only`, then
   `/plugins reload my_plugin`. Reloading by itself does not fetch from Git.
   `/plugins remove my_plugin` unloads it and forgets its declaration, but keeps
   the checkout so local changes are not lost. To reinstall, delete the checkout
   directory named in the response, then run `/plugins add GIT_URL` again.

No restart needed when you do it from inside CLAI. A plugin you add or enable is
active for the next prompt; one you disable or remove is gone for the next
prompt. From the shell, `clai2 plugins ...` only saves; it loads on the next
start. Plugins in the folder and the ones you added by name are managed the
same way. Explicit module declarations take precedence over a same-named drop-in
file. Reloading a disabled plugin is rejected; enable it first. Drop-in entry
modules are compiled from current source. Installed modules use `importlib.reload`,
which can retain globals removed from source; initialize plugin state explicitly.

`/reload` refreshes CLAI code, not Harness or core. After changing those packages,
restart with the same launch options and `--resume` to continue the saved session.
Keep the worktree if asked to remove it.

Plugins are trusted code running as you. Only install what you trust.

## Worktree startup

```bash
clai2 --worktree my-task
```

`--worktree` (or `-w`) creates or reopens `<repository-root>/.worktrees/NAME` and changes to
that directory before reading project settings or activating plugins. Relative paths in your plugin, the coding tools,
and `repo_context` therefore refer to that checkout. User plugins and settings
still load from the same database directory, even with a relative `--database`
path. Only committed project files reach the new checkout. After interactive
shutdown and plugin cleanup, CLAI offers to remove the linked worktree, defaulting
to keep. This includes existing linked worktrees. Removal uses Git without
`--force` and keeps the branch; dirty or locked checkouts stay on disk.
Headless runs, piped input, and startup errors do not prompt. `/new`, `/resume`,
and `/reload` keep the checkout in use. Plugins do not own worktree cleanup.
See [Git worktrees](README.md#git-worktrees) for naming and cleanup.

## The built-in plugins

The coding tools are a plugin too, and so are asking you multiple-choice
questions mid-run, reading the repository's instruction file, and keeping the
conversation inside the context window. These five plugins are marked
`(built-in)` and enabled unless you say otherwise. Built-ins load first, in
the order listed below, so `coder`'s guidance leads the system prompt; saved,
drop-in, and project plugins follow in name order. Registration order is also the
order plugin instructions, renderers, and status segments are consulted in.
`/plugins` and `/plugins list` stay alphabetical for scanning. CLAI's plugin
implementations live in `pydantic_clai2.builtin_plugins`. Saved plugin
settings using the former import paths are redirected to the new modules.

| Id | Backed by | Settings | Does |
|---|---|---|---|
| `coder` | `pydantic_clai2.builtin_plugins.coder` | `{"unrestricted_filesystem": true, "repo_context": false, "sub_agents": true, "agent_folders": ["agents"]}` | the file and shell tools, plus task delegation and disk agents |
| `ask_user` | `pydantic_clai2.builtin_plugins.ask_user_menu` | `{}` | the `ask_user_question` tool: multiple-choice questions answered from the terminal |
| `repo_context` | `pydantic_clai2.builtin_plugins.repo_context` | `{}` | reads `CLAUDE.md` or `AGENTS.md` from the launch directory into the instructions |
| `persistence` | `pydantic_clai2.runtime.sessions` | `{}` | Harness step checkpoints for interrupted session recovery |
| `compaction` | `pydantic_clai2.builtin_plugins.compaction` | `{}` | automatic summarisation with a truncation fallback, `/compact`, and the context warning |
| `slack` (off until enabled) | `pydantic_clai2.builtin_plugins.slack` | `{}` | Slack's hosted tools as you, read-only by default; see [below](#slack-your-slack-workspace-as-you) |

[`day_ai`](#day_ai-day-ai-crm-tools) and [`grain`](#grain-meetings-with-a-saved-sign-in)
are built in too, but start disabled because they need your Day AI or Grain
account.

### Other harness capabilities

`/plugins` lists only the built-ins above, plus plugins you or the repository
declared. It does not list every public harness capability for Space-enable:
hosted-MCP integrations such as GitHub, sandboxes, and guardrails need
credentials, extras, or settings that a checkbox cannot supply, so they belong in
CLAI plugins written for them, such as the disabled built-ins
[`day_ai`](#day_ai-day-ai-crm-tools),
[`google_workspace`](#google_workspace-gmail-calendar-and-drive-tools),
[`grain`](#grain-meetings-with-a-saved-sign-in),
[`linear`](#linear-issues-and-projects),
[`logfire_mcp`](#logfire-mcp-query-your-telemetry),
[`notion`](#notion-workspace-tools),
[`ordinal`](#ordinal-social-posts-in-ordinal),
[`posthog`](#posthog-posthog-analytics-signed-in-for-clai),
[`pylon`](#pylon-support-issues-and-accounts-in-pylon), and
[`slack`](#slack-your-slack-workspace-as-you).

To run any other capability, declare it on purpose under an id of your choice,
with JSON constructor settings if it takes them:

```text
/plugins add sliding_window_compaction pydantic_ai_harness.compaction:SlidingWindowCompaction '{"max_messages": 40}'
```

For callbacks, stores, or other Python objects, write a plugin module that builds
the capability in `get_capabilities`. CLAI does not install the capability's
optional dependencies. Avoid enabling overlapping tool providers together, such
as `filesystem` or `shell` alongside `coder`.

Earlier releases listed every harness capability here, disabled. If you enabled
one of those, it was saved as your own declaration, so it keeps loading and now
shows as a saved plugin; `/plugins remove NAME` forgets it. The exception is a
saved copy of an entry that now has its own built-in under the same id, such as
`google_workspace`, `ordinal`, or `slack`: it becomes that built-in, keeping whether it was enabled.

`/plugins disable coder` gives you a chat-only CLAI (a writing or research setup
with `ExaSearch` instead, say); `/plugins enable coder` brings the tools back;
`/plugins remove coder` cannot forget a built-in, so it resets it to its
defaults. `/plugins disable repo_context` stops the instruction file from being
read. To run a built-in with different options, add your own declaration under
the same name and it takes the built-in's place:

```text
/plugins add coder pydantic_clai2.builtin_plugins.coder '{"unrestricted_filesystem": false, "repo_context": false}'
/plugins add repo_context pydantic_clai2.builtin_plugins.repo_context '{"walk_up": true}'
```

Keep `"repo_context": false` on a replacement `coder`: `Coder` bundles its own
`RepoContext`, and with the `repo_context` plugin also on, the instruction file
would reach the model twice. The stock `coder` enables `delegate_task`: a task
can run in a fresh conversation with the same active plugin tools, instructions,
and guardrails. CLAI rebuilds its stock agent before the next prompt when the
active capability snapshot changes, binding those capabilities to the new agent.
Conversation history stays in the session; an existing run keeps its own snapshot.
The exported `DEFAULT_PLUGINS` keeps delegation off for custom-agent launchers;
the CLI uses `STOCK_PLUGINS`, which opts its own rebuildable agent in.

Saved `Coder` declarations that omit `sub_agents` still default to `false` for
compatibility. Set `"sub_agents": true` in `/plugins configure coder` to opt in;
explicit `false` remains an opt-out. **Unrestricted filesystem** in
`/plugins configure coder` decides whether the file tools reach any path on this
machine (`true`, the stock default) or only the launch directory (`false`). It
saves `unrestricted_filesystem` in the `coder` declaration. Supplied agents are not rebuilt: their plugins
are still run-level capabilities, so self-delegation requires binding `Coder` and
the capabilities it should carry when constructing that agent.

`agent_folders` is a JSON list of folder names or paths for disk-defined agents
(Claude `*.md` or Codex `*.toml`). A name searches `.agents/<name>`,
`.claude/<name>`, and `.codex/<name>` in the project, then your home directory;
project definitions win. A path loads exactly that folder; `[]` disables disk
agents. The stock CLI uses `["agents"]`. Saved `coder` declarations that omit
`agent_folders` keep disk agents off, so an upgrade never loads new definitions
without your say. Old `pydantic_ai_harness.coder:Coder` declarations load
through this module and are not rewritten.

Managed tasks are a stock-shell service over harness `DelegationTasks`, not new
host hooks. The shell keeps plugin resources alive until children settle; `/plugins`
changes are refused while managed children run. Exit/reload drains them before
`session_end`. Child questions use `host.full_screen()` and name the child in the picker's title.
Typed delegation lifecycle events supply compact transcript rows; raw child events
update the task inspector rather than entering the parent's transcript. Core hooks
and guardrails bound to the stock agent still run on general-purpose children.
Explore/Plan are separate read-only agents and do not inherit plugin tools.

See [managed tasks](README.md#managed-tasks-in-the-interactive-stock-cli) for `/tasks`,
Ctrl+B, independent histories, explicit resume, completion-report provenance, and
workspace restrictions. Supplied agents and headless calls are unchanged. Do not
hold a parent's `RunContext` or call its event emitter after that run has ended;
background reports go through the task owner's currently attached parent queue.
When a direct background child finishes while the interactive stock CLI is idle,
the shell starts an automated continuation with no user message. Its `turn_start`
and `turn_end` hooks receive empty text. The editor preserves drafts and gives
queued user input priority. Reports arriving during a run stay on its native queue.

Saved settings that need features absent from this build use defaults; see
[Settings that need a feature](#settings-that-need-a-feature-hostsettingsmodel-requires).

`repo_context` wraps harness `RepoContext` with the launch directory as the
workspace and its default filenames. Its settings:

| Key | Default | Does |
|---|---|---|
| `walk_up` | `false` | also load instruction files from every directory between the workspace and your home directory |
| `inventory_tool` | `false` | give the agent `inventory_agent_context`, which maps the repo's `.claude`, `.agents`, `.codex`, and `.grok` assets |
| `nested_traversal` | `false` | when the agent reads or lists a directory, tell it about that directory's instruction file |
| `nested_inject` | `"pointer"` | what nested traversal adds: `"pointer"` (one line naming the file) or `"contents"` |

A repository can declare plugins too, in `.clai/settings.json`; they show as
`(project)` and rank just above the built-ins. They start off, because a
repository must not run code as you just because you opened it: CLAI names the
ones waiting at startup, and `/plugins enable NAME` approves one. See
[Project settings](README.md#project-settings).

Some plugins already include others. `coder` includes task delegation, so while
`coder` is on, a harness `SubAgents` row (such as `subagents`) stays off unless
`coder` has `sub_agents` turned off: `/plugins` shows it greyed out with
`in coder`, `/plugins list` says `included in coder`, and enabling it says to
disable `coder` first. Turning `coder` off loads any of them you had enabled.
`coder` only clears old tool results, so `compaction` runs beside it. Were `coder`
to bind its own history compaction, `compaction` would be greyed out the same way,
so two compaction chains never run together.

`compaction` directly registers harness `FallbackCompaction` with
`max_fraction=threshold`; harness owns the automatic trigger. `/compact` runs the
same chain unconditionally, with the protected tail capped at half the
conversation, and keeps the history unless the result is smaller. Its optional
focus is free text, not shell arguments:
`/compact don't lose the "auth" decisions` preserves the apostrophe and quotes in
the summariser's prompt. Only `ModelAPIError`, `FallbackExceptionGroup`, and
`UsageLimitExceeded` cause summarisation to fall back to truncation; other exceptions
propagate. `/plugins disable compaction` turns automatic compaction,
`/compact`, and its context warning off. `/plugins configure compaction` edits its
settings (`strategy`, `threshold`, `protected_tokens`, `context_window`,
`summarization_model`; see the README) and reloads it for the next turn. A
declaration under the same name changes them too:

```text
/plugins add compaction pydantic_clai2.builtin_plugins.compaction '{"threshold": 0.7, "context_window": 200000}'
```

### `ask_user`: questions answered from the terminal

The second built-in, `ask_user` (`pydantic_clai2.builtin_plugins.ask_user_menu`), gives
the model the harness's `AskUser` capability: one tool, `ask_user_question`, for
asking you one to ten multiple-choice questions when the task is ambiguous. Each
question appears inline above a compact numbered picker, keeping the conversation
visible. The question is pinned with its choices, so output streamed meanwhile (a
delegated task's, say) lands above it instead of between it and the picker. Up/Down moves the highlight; Enter or an option's number selects it.
For multi-select questions, Enter or a number toggles that choice; select `Done`
to submit at least one choice. The title says `question 2 of 3` when there are
several. Esc or Ctrl-C declines the whole request and lets the model continue.
The plugin's `render` replaces the tool's argument dump with a header that lists
the question headers. The picker uses `host.full_screen()` only to flush streaming output and suspend
the editor's input reader. It keeps the live panel's alternate screen. The draft
is restored on exit, and the question and your picks are printed to the transcript afterwards.
`/plugins disable ask_user` takes the tool away.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The capability does not know it is in a terminal. It hands an `AskUserRequest`
to an `Answerer` (one async callable returning an `AskUserResponse`) and waits.
To answer questions somewhere else, a web page or a chat bridge, say, replace
the built-in with your own plugin under the same name that constructs `AskUser`
with a different answerer:

```python
from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer, AskUserRequest, AskUserResponse

from pydantic_clai2.plugins import Plugin


async def ask_over_http(request: AskUserRequest) -> AskUserResponse:
    # POST request.questions to your front end, keyed by request.id, and wait
    # for the reply; return AskUserResponse(cancelled=True) if the user dismisses it.
    picks = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
    return AskUserResponse(answers=tuple(picks))


class AskOverHTTP(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (AskUser(answerer=ask_over_http),)
```

```text
/plugins add ask_user my_ask_user
```

The request and its response are also emitted as `AskUserRequestedEvent` and
`AskUserAnsweredEvent`, so a plugin that only wants to watch (log the question,
show a "waiting for you" state) returns a `Hooks` capability with
`hooks.on.event(EventClass)`, or overrides `render`, without being the answerer.

### `github`: tools from GitHub's hosted MCP server

The `github` built-in (`pydantic_clai2.builtin_plugins.github`) gives the model harness
[`GitHub`](../pydantic_ai_harness/pydantic_ai_harness/github/README.md): the tools of GitHub's hosted
MCP server, acting as the account behind a token. It starts disabled.
`/plugins enable github`, or Space on it in `/plugins`, loads it and opens its
settings menu. To change the settings later, run `/plugins configure github` or
press `c` on it in `/plugins`.
You never need to reinstall it.

The menu is the shared field editor that `/set` uses: type to filter, Enter to
edit a row, `R` to reset one, Esc or **Save & close** to close. Each change is
saved as soon as you make it. When the menu closes, the plugin loads again, so the next turn uses the
new settings.

| Row | Default | Does |
|---|---|---|
| Sign-in | GitHub CLI (browser) | where the token comes from: the GitHub CLI's browser sign-in, or a token saved in `/keys` (pick one from a searchable list, or type a new one into a masked field) |
| GitHub host | github.com | github.com, or GitHub Enterprise Cloud with data residency; the latter asks for `octocorp.ghe.com` or the full MCP URL and connects to `https://copilot-api.octocorp.ghe.com/mcp` |
| Tools | read-only | read-only, or read and write |
| Tool groups | server defaults | `all`, a preset, or your own comma-separated groups such as `repos,issues,actions` |
| Server instructions | forwarded | whether the GitHub server's own instructions reach the agent |

GitHub Enterprise Server has no hosted MCP server, so it is not offered. The
token's own permissions still limit what any of these settings can reach.

#### Signing in through the browser

GitHub's hosted MCP server does not support dynamic client registration, so an
MCP client can only use OAuth with an OAuth App registered to that client. CLAI
doesn't register one. Instead it uses the GitHub CLI (`gh`), which has its own.
Choosing **GitHub CLI (browser)** under Sign-in runs
`gh auth login --hostname HOST --web --clipboard`. The menu shows the one-time
code (`gh` also copies it to your clipboard) and opens
`https://github.com/login/device` in your browser. Once you approve the code,
the screen closes by itself. Enter opens the page again, and Esc stops `gh` and
leaves the setting unchanged. If `gh` already has a login for the host, choosing
it just switches the plugin over. `HOST` is `github.com`, or the ghe.com host for
GitHub Enterprise Cloud.

`gh` keeps the token in your OS keyring. On every run the plugin calls
`gh auth token --hostname HOST`, so `gh auth login`, `gh auth switch`, or
`gh auth logout` in a shell take effect from the next turn. `GH_TOKEN`,
`GITHUB_TOKEN`, and their enterprise variants are removed from the environment
`gh` sees. Without that, `gh` would hand back the variable's token instead of
your login, and it refuses to sign in while one is set. If `gh` is not installed,
the menu says so; install it from <https://cli.github.com> or use `/keys`.

#### Using a token saved in `/keys`

Choosing **Token saved in /keys** stores the token in the named API key store
that `/keys` manages, not in plugin settings, which are plaintext SQLite. A new
token typed in the menu is saved in `/keys` under the key's name. If a key of
that name already exists, the menu asks before replacing it, because other
plugins and connections may share it. `GITHUB_TOKEN` is a label in `/keys`, not
an environment variable. Any plugin that names the same key shares it.

Either way, the plugin's settings hold only which source to use and a key's name,
such as `{"login": "key", "token": {"name": "GITHUB_TOKEN"}}`. A declaration that
tries to hold a token is rejected.

The token is looked up on every run. If none is available when the plugin loads
(no `gh` login for the host, or the named key is missing), the plugin still
loads, prints a warning, and keeps its settings menu available. Until you sign
in or save a key, each run fails with an error saying how to fix it, so the agent
never runs as the wrong account. `/keys` does not stop you renaming or deleting a
key that a plugin uses. Neither sign-in uses your `/login github-copilot` login,
and Copilot does not read the `GITHUB_TOKEN` key. The plugin emits no telemetry
of its own; tool calls appear in core's spans.

### `google_workspace`: Gmail, Calendar, and Drive tools

`google_workspace` (`pydantic_clai2.builtin_plugins.google_workspace`) is a built-in that starts
disabled. It gives the agent the tools of Google's hosted Workspace MCP servers
through harness [`GoogleWorkspace`](../../docs/harness/google-workspace.md). It needs a
Google OAuth access token whose scopes cover the products you select.

Turning it on (Space in `/plugins`, or `/plugins enable google_workspace`) opens
its settings menu. Open it again later with `c` in `/plugins`,
`/plugins configure google_workspace`, or:

```text
/google_workspace
```

The settings menu is full-screen with one row per setting. Up/Down moves, Enter
edits a row, `r` puts a row back to its default, and **Save & close** or Esc
leaves. Each change is saved to the plugin's declaration as soon as you make it and applies
from the next turn, without reloading. Run `/google_workspace` again at any time
to change a setting or pick a different key.

| Row | Stored as | Default | Does |
|---|---|---|---|
| Access token key | the key's name, in the credential store | `GOOGLE_ACCESS_TOKEN` | which `/keys` entry holds the Google OAuth access token |
| Products | `services` | `gmail, calendar, drive` | a searchable checklist of `gmail`, `drive`, `docs`, `sheets`, `slides`, `calendar`, `chat`, `people`; Enter toggles one, and at least one stays on |
| Read-only tools | `read_only` | `true` | keep only the tools Google marks as read-only |
| Server instructions | `include_instructions` | `true` | pass the Google servers' own instructions to the agent |

CLAI runs tools without asking first, so `read_only` defaults to `true`. Set it to
`false` to also get the tools that send, change, and delete.

**The token.** It lives in the [saved API keys](#saved-api-keys) store, never in
plugin settings, which are plain SQLite and reject a `token` or `auth` entry.
Enter on the key row lists your saved key names so you can pick one (type to
filter; Esc leaves the choice unchanged). **Enter a different API key** asks for a
new token without echoing it and saves it in `/keys` as `GOOGLE_ACCESS_TOKEN`,
replacing any value already stored under that name. Only the chosen key's name is
remembered, in the credential store. Several plugins and connections can share one
named key, such as GitHub integrations all using `GITHUB_TOKEN`. While the plugin
refers to a key, `/keys` refuses to rename it; pick another key here first. To
replace the token itself, edit the key in `/keys`.

The plugin loads without a token so that `/google_workspace` is available, and
prints which key it is missing. Each turn looks the key up again, because Google
access tokens expire after about an hour: replacing the value in `/keys` applies
from the next turn. If the key is missing or was deleted, the turn fails with a
message naming it instead of running without the tools. The `GOOGLE_ACCESS_TOKEN`
environment variable is not read; key names are labels, not environment variables.

**Not configurable here.** The token decides the Google account and its OAuth
scopes. `GoogleWorkspace` takes a ready-made access token and has no OAuth client
ID, client secret, or scope settings, so there is nothing about the OAuth client to
store: mint the token with your own OAuth client and save it in `/keys`.

Declarations still work for scripted setups:

```text
/plugins add google_workspace pydantic_clai2.builtin_plugins.google_workspace '{"services": ["gmail", "docs"], "read_only": false}'
```

Earlier versions listed `google_workspace` as a raw harness entry,
`pydantic_ai_harness.google_workspace:GoogleWorkspace`. If you turned that entry on
or off in the menu, CLAI now loads this plugin in its place and keeps your on or off
choice. A declaration you added with its own settings under that factory is kept
as written.

### `pylon`: support issues and accounts in Pylon

`pylon` (`pydantic_clai2.builtin_plugins.pylon`) gives the agent harness
[`Pylon`](../../docs/harness/pylon.md): Pylon's hosted MCP tools for
searching, reading, creating, and updating support issues, looking up and
updating accounts, and looking up contacts. It starts disabled;
`/plugins enable pylon` turns it on. The agent acts as the Pylon user who signed
in, so only Member and Admin users with Pylon's `MCP Access` role can use it.

#### Configuring Pylon

Turning the plugin on opens its settings menu, like any plugin with a
[`configure`](#offer-a-settings-menu-async-def-configureself) menu. `c` in
`/plugins`, `/plugins configure pylon`, and `/pylon` reopen it at any time to
change anything, including the key. Type to filter the rows. Enter edits a row,
`R` restores its default, and **Save & close** or Esc leaves the menu. Each
change is saved to the plugin's settings as soon as you make it and applies from
the next run, with no reinstall.

| Row | Default | Does |
|---|---|---|
| Sign-in (`auth`) | Named key from /keys (`"key"`) | `"key"` connects with a `/keys` entry. `"browser"` signs in through the browser |
| Key (/keys) | not chosen | shown only for `"key"`. Enter opens the saved-key picker, and `R` forgets the choice. Stored as a name, not in settings |
| Read-only tools (`read_only`) | `false` | keep only the tools Pylon's server labels read-only |
| Server instructions (`include_instructions`) | `true` | pass Pylon's own server instructions to the agent |

`/pylon status` prints the current setup without opening the menu, and
`/pylon key` goes straight to the key picker. A declaration can also carry the
settings as JSON, since none of them are secret:

```text
/plugins add pylon pydantic_clai2.builtin_plugins.pylon '{"read_only": true}'
```

Pylon's endpoint (`https://mcp.usepylon.com`) is fixed and there is no
workspace or organization field. The token decides which Pylon organization
and user the agent acts as.

#### Keys: `/keys` holds the secret, Pylon holds its name

The token lives in [`/keys`](#saved-api-keys), not in plugin settings, which are
plaintext SQLite. The Key row uses the shared saved-key picker, which is
searchable, and Esc cancels it. Choose an existing key, or choose **Enter a
different API key** to type a masked token. CLAI saves that token in `/keys` as
`PYLON_ACCESS_TOKEN`, the name harness `Pylon` documents. If that name already
exists, CLAI asks before replacing it, since other connections may use it. The
name is only a label: CLAI does not read an exported `PYLON_ACCESS_TOKEN`.

Only the key's name is saved, in the credential store beside the vLLM and
OpenRouter connections. Each run looks up the key's current value, so replacing
it in `/keys` takes effect on the next run. A key Pylon uses cannot be renamed
in `/keys`. Deleting it makes Pylon runs fail with an error naming the key until
you restore it or choose another. Until a key is chosen, runs get no Pylon
tools. This applies outside a terminal and after cancelling the menu.

Several connections and plugins can share one named key by pointing at the same
name. For example, a GitHub plugin and Copilot tooling can both reference
`GITHUB_TOKEN`, so replacing it once updates both.

Pylon only accepts OAuth access tokens, not its REST API keys. With the
**Browser sign-in** option, CLAI signs in for you: the first run that uses Pylon
opens the browser, as `/mcp` servers with OAuth do, and waits up to five minutes
for you to finish. Those tokens stay in the keyring (account `mcp-plugin_pylon`,
or a private `0600` file when there is no keyring). They are refreshed as
needed and never touch `/keys` or plugin settings.

### `day_ai`: Day AI CRM tools

`day_ai` (`pydantic_clai2.builtin_plugins.day_ai`) starts disabled. It gives the model harness
[`DayAI`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/day_ai/),
the tools of Day AI's hosted MCP server: search and update CRM records, read
meeting context, and draft emails. It needs a paid Day AI Agent tier.

`/plugins enable day_ai` loads it and opens its settings menu. Reopen the menu
at any time with `/plugins configure day_ai`, or `c` on `day_ai` in `/plugins`.
Type to filter the rows, press Enter to change one, `R` to reset it to its
default, and Esc to close. Each change is saved as soon as you make it, and the
plugin loads again with the new settings when the menu closes. The menu has
one row for each option harness `DayAI` takes:

| Row | Setting | Choices |
| --- | --- | --- |
| Sign-in | `auth` | **automatic** (default): `DAY_AI_ACCESS_TOKEN` from `/keys` if it is saved, else a browser sign-in you completed earlier; **choose or enter a key in /keys**; **browser sign-in** |
| Server instructions | `include_instructions` | **forwarded** (default) or **left out**: whether the Day AI server's own instructions reach the agent |

Day AI runs one hosted endpoint (`https://day.ai/api/mcp`), and harness `DayAI`
has no base URL, workspace, or read-only option, so the menu offers none.
Which workspace you reach follows from the token or the account you sign in
with. The server does not mark any tool read-only, so the model gets every
tool your tier and role allow, including ones that change CRM records.

Tokens are kept out of plugin settings, which are stored in plaintext:

- **A token in `/keys`.** Choosing "choose or enter a key in /keys" lists your
  saved keys, searchable by name, so you can pick one, or enter a new token in
  a masked field. A new token is saved in `/keys` as `DAY_AI_ACCESS_TOKEN`, the
  name harness `DayAI` documents, after asking before it replaces a saved one.
  Settings keep only the name, as `{"auth": {"name": "DAY_AI_ACCESS_TOKEN"}}`.
  The name is a `/keys` label only; CLAI does not read the environment
  variable. The token is looked up on every run, so replacing it in `/keys`
  reaches the next run, and deleting it makes runs fail with a message until
  you save it again. Rename, replace, or delete keys in `/keys`. Plugins that
  name the same key share one value.
- **Browser sign-in.** This keeps the tokens in the OS keyring (credential
  `mcp-day_ai`), the way `/mcp` signs in to an OAuth server, so later sessions
  reuse and refresh them. Choosing it saves `{"auth": "oauth"}`, which keeps
  using the browser even when `DAY_AI_ACCESS_TOKEN` is saved. If you are not
  signed in yet, the browser opens right away, behind a waiting screen. Esc
  cancels it, and a failed sign-in is reported in the menu. Either way Day AI
  stays signed out and CLAI keeps working. Loading never opens the browser.
  While browser sign-in is chosen but not finished, the plugin loads without
  Day AI tools and prints how to sign in, so a sign-in you cannot finish never
  holds up a session.

Until you choose one, with no `DAY_AI_ACCESS_TOKEN` and no earlier sign-in,
the plugin loads without Day AI tools and prints how to connect. A key named
in `auth` that is missing from `/keys` never falls back to the browser, and the
menu marks it "missing from /keys".

If you enabled `day_ai` from the earlier harness catalog, your saved
`pydantic_ai_harness.day_ai:DayAI` declaration still takes precedence and reads
`DAY_AI_ACCESS_TOKEN` from the environment. `/plugins remove day_ai` switches to
this plugin.

### `ordinal`: social posts in Ordinal

`ordinal` (`pydantic_clai2.builtin_plugins.ordinal`) gives the model harness
[`Ordinal`](../pydantic_ai_harness/pydantic_ai_harness/ordinal/README.md), which drafts, schedules,
and analyzes social posts through Ordinal's hosted MCP server. It starts disabled;
`/plugins enable ordinal` turns it on. Ordinal MCP needs the Pro plan or higher.
If you had enabled or disabled the former `pydantic_ai_harness.ordinal:Ordinal`
catalog entry, that choice carries over to this plugin.

Set it up in its settings menu. Turning it on (Space in `/plugins`,
`/plugins enable ordinal`, or `/plugins add`) opens the menu, and so do `/plugins configure ordinal` and `c` in `/plugins`
later, so you can change anything without reinstalling. It is the same field
editor `/set` uses: type to filter, Enter edits a row, `R` resets it, and
**Save & close** or Esc closes.
Each change is saved as soon as you make it, and the plugin loads again when the
menu closes.

| Row | Default | Does |
|---|---|---|
| Sign-in | Automatic | which credential runs use: Automatic (a `/keys` entry, else `ORDINAL_ACCESS_TOKEN`, else the browser), or only a saved key, only the environment variable, or only the browser |
| `/keys` entry | none | opens the `/keys` picker: choose a saved key, or type a new token masked; `R` stops using the key |
| Server instructions | Included | pass Ordinal's own server instructions to the model |

Harness `Ordinal` has one endpoint and reaches every workspace the token's user
belongs to, so there is no base URL or workspace to set.

The token never goes in plugin settings, which are plaintext SQLite; the settings
hold only the two options above, and a declaration with any other field, such as
a pasted token, fails to load. A token typed in the menu is saved in `/keys` as
`ORDINAL_ACCESS_TOKEN`, after asking if that would replace an existing key, since
other plugins and connections may share it. CLAI keeps only the chosen key's name
and looks it up on every run. Replacing the key in `/keys` applies to the next
run, and deleting it makes runs fail instead of connecting without it. `/keys`
will not rename a key while Ordinal uses it. Several plugins and providers can
name the same key, as GitHub and Copilot can both use `GITHUB_TOKEN`.

A browser sign-in opens your browser on the first run that uses Ordinal. CLAI
keeps the OAuth tokens in the OS keyring (the private credential file when no
keyring exists), as `/mcp` does for OAuth servers, so later launches reuse them.
It only works on the machine you run CLAI on. With no credential for the chosen
sign-in and no terminal (a headless run from CI, say), the plugin fails to load
with a message naming `/plugins configure ordinal`, rather than adding tools that
cannot connect.

`/ordinal` shows which credential the next run uses. `/ordinal logout` forgets the
saved browser sign-in and drops the one in use, so the next browser run signs in
again; it does not touch a `/keys` entry or the environment variable. Disabling
the plugin does not sign you out.

### `slack`: your Slack workspace, as you

`slack` (`pydantic_clai2.builtin_plugins.slack`) connects harness
[`Slack`](../pydantic_ai_harness/pydantic_ai_harness/slack/README.md) to Slack's hosted MCP server.
It ships disabled. The tools act as the user whose token CLAI connects with, so
anything the agent posts appears under your name. If you enabled or disabled the
earlier raw `pydantic_ai_harness.slack:Slack` catalog row, CLAI loads this plugin
in its place and keeps your choice. A declaration with your own
settings is left as it is.

Turning it on (Space in `/plugins`, `/plugins enable slack`, or `/plugins add`),
or pressing `c` on it in `/plugins`, opens its settings menu. `/plugins configure slack`
reopens it later, with no reinstall. The list is searchable, Enter edits the
highlighted row, `R` resets it, and **Save & close** or Esc closes. Every change is saved as you make it:

| Row | Choices | Saved in |
|---|---|---|
| Sign-in | user token from `/keys` (default) or browser sign-in through your Slack app | plugin settings, as `auth` (`key` or `browser`) |
| User token (`key` only) | a key from `/keys`, or a new user token (`xoxp-`) | `/keys`; only the key's name is kept for Slack, in the credential store |
| Slack app (`browser` only) | creates your CLAI Slack app, then takes its Client ID | plugin settings, as `client_id` (public, not a secret) |
| Browser sign-in (`browser` only) | signs in again, or `R` signs out | the tokens, in the credential store (`slack-oauth`) |
| Tools | read-only (default) or read and write | plugin settings, as `read_only` |
| Server instructions | forwarded (default) or left out | plugin settings, as `include_instructions` |

**User token.** Enter shows the names of your saved keys. Pick one, or, when
`/keys` is empty, paste a token into the masked input. A new token is saved in `/keys` as
`SLACK_USER_TOKEN` (the name harness `Slack` documents, used here only as a label). If that
name already holds a token, CLAI asks before replacing it, because every plugin and
connection that uses the key would change with it. Slack's MCP server acts as a
user and rejects bot tokens, so the menu refuses an `xoxb-` token whether picked
or pasted; there is no bot-token mode. The token also decides the workspace and
user: to use another workspace, save its token under another name, such as
`SLACK_USER_TOKEN_ACME`, and pick that. `R` on this row forgets the choice, which
turns the Slack tools off, and leaves the key in `/keys`. The row notes when no
key is chosen or the chosen key is gone from `/keys`.

**Browser sign-in.** Slack's MCP server has no dynamic client registration and
serves only apps installed in your workspace, so browser sign-in goes through your
own Slack app. CLAI creates it for you. Choose **Browser sign-in** in the Sign-in row,
then press Enter on **Slack app**. CLAI opens Slack's create-app page filled in with
CLAI's manifest. Pick your workspace, click **Create**, and paste the app's **Client
ID** (Basic Information, App Credentials). CLAI then opens Slack's sign-in page and
waits up to five minutes behind a screen that shows the URL, in case no browser
opens (over SSH, for example); Esc cancels. You never copy a token or a client
secret: the manifest enables PKCE, so Slack treats the app as a public client and
neither signing in nor renewing needs a secret. The manifest also turns on the
app's MCP access (`is_mcp_enabled`), which Slack requires, and token rotation.
Access tokens last 12 hours; CLAI renews them before a turn when they are close
to expiring, keeping the rotated refresh token in the credential store. If CLAI
goes unused for 30 days, the refresh token expires and you sign in again.

A read-only sign-in asks Slack for read scopes only, so the token itself cannot
post. After switching Tools to read and write, press Enter on **Browser sign-in**
to sign in again with the write scopes. Until you do, turns have no Slack tools
and CLAI says to sign in again, rather than offering write tools the token cannot
use. The redirect is fixed at
`http://localhost:53118/slack/callback`, because Slack matches the registered URL
exactly. Workspaces that require admin approval for new apps need that approval
first. `R` on Browser sign-in signs out; `R` on Slack app also forgets the app.

Harness `Slack` has no server URL option: it always connects to
`https://mcp.slack.com/mcp`, so the menu has no base URL row. Plugin settings never
hold the token, and a pasted `token` setting is rejected. CLAI does not read the
`SLACK_USER_TOKEN` environment variable either. If it is set and no key is
chosen, CLAI says so and points at the menu.

With a user token, before every turn CLAI reads the chosen key's current value from `/keys`, so
replacing it there takes effect on the next turn with no reload. Until you choose
a key (or, for browser sign-in, set up the Slack app), turns just have no Slack
tools, without a warning; the menu rows say what is missing. When the chosen key
was deleted or the saved choice is invalid, CLAI prints why and that turn has no
Slack tools. While Slack uses a key, `/keys` will not
rename it.

| Key | Default | Does |
|---|---|---|
| `auth` | `key` | `key` connects with the user token chosen from `/keys`; `browser` with the browser sign-in |
| `client_id` | none | your CLAI Slack app's Client ID, for `browser` |
| `read_only` | `true` | keep only the tools Slack marks read-only, so the agent can search and read but not post or edit |
| `include_instructions` | `true` | forward the Slack server's own instructions to the agent |

To let the agent send messages and edit canvases as you, choose **read and
write** in the Tools row. The equivalent typed command is
`/plugins add slack pydantic_clai2.builtin_plugins.slack '{"read_only": false}'`.

### `posthog`: PostHog analytics, signed in for CLAI

`posthog` (`pydantic_clai2.builtin_plugins.posthog`) connects the agent to PostHog's hosted MCP
server through harness [`PostHog`](../../docs/harness/posthog.md). It starts disabled.
`/plugins enable posthog` loads it and opens its settings menu. To change the
settings later, run `/plugins configure posthog` or press `c` on it in
`/plugins`. You never need to reinstall it.

The menu is the shared field editor that `/set` uses: type to filter, Enter to
edit a row, `R` to reset one, **Save & close** or Esc to leave. Each change is
saved as soon as you make it. When the menu closes, the plugin loads again, so the next turn uses the
new settings.

| Row | Default | Does |
|---|---|---|
| API key | none | the `/keys` entry to connect with: pick a saved key from a searchable list, or type a new one into a masked field |
| Sign-in | API key from `/keys` | or browser sign-in, with tokens kept in the keyring |
| Region | US cloud | US (`mcp.posthog.com`), EU (`mcp-eu.posthog.com`), or a typed `https://` URL for a PostHog MCP server you run (`http://` only for localhost) |
| Tools | read-only | read-only, or read and write (for example editing feature flags) |
| Feature groups | every group | a searchable list of PostHog's feature groups; Enter toggles one and saves it |
| Server mode | server default | one `posthog` tool driven by commands, or one tool per operation |
| Project ID | not set | pin every request to one project (`x-posthog-project-id`) |
| Organization ID | not set | pin every request to one organization (`x-posthog-organization-id`) |
| Server instructions | forwarded | whether the PostHog server's own instructions reach the agent |

The key's own scopes, organizations, and projects still limit what any of these
settings can reach. Some PostHog tools use an LLM on PostHog's side and need AI
data processing enabled for your organization.

Secrets are managed in `/keys`, never in plugin settings (plugin settings are
plaintext SQLite, and a declaration that tries to hold a key is rejected: `/plugins add` keeps
no settings that fail validation, and the error does not echo them). A new
key typed in the menu is saved in `/keys` as `POSTHOG_PERSONAL_API_KEY`, the name
harness `PostHog` documents. It is a label only: CLAI does not read or export the
environment variable. If a key of that name exists, the menu asks before
replacing it. CLAI remembers only the key's name, in the credential store beside
the vllm and openrouter connections. Create the key with PostHog's "MCP Server"
preset.

One named key can serve several plugins and connections: pick the same `/keys`
entry for each, the way GitHub integrations can all use one `GITHUB_TOKEN`. The
name is looked up on every request, so replacing the key in `/keys` reaches the
next turn of everything that names it. `/keys` refuses to rename a key while
PostHog uses it. Until a key is chosen, or if the chosen key is deleted, the
plugin still loads with a warning and keeps its menu, and each run fails with an
error naming the fix instead of connecting without the key.

With browser sign-in, the first prompt that uses PostHog opens the browser. The
tokens go to the keyring (the `mcp-posthog_plugin` entry under `pydantic-clai2`),
the same storage `/mcp` uses for OAuth servers, so the sign-in lasts across turns
and launches. Pick the EU region if your account is on the EU instance.

`/posthog` shows the key or sign-in in use; `/posthog logout` forgets the browser
sign-in. The plugin always builds its own connection instead of passing `auth` to
`PostHog`: harness `PostHog`'s key path connects only to the US endpoint, and
`auth='oauth'` keeps browser tokens in memory, so it would sign in on every turn
with a 5-second connect timeout. The plugin emits no telemetry of its own; tool
calls appear in core's spans.

### `grain`: meetings, with a saved sign-in

`grain` (`pydantic_clai2.builtin_plugins.grain`) gives the agent harness's
[`Grain`](../pydantic_ai_harness/pydantic_ai_harness/grain/README.md) capability: search and read
the Grain meetings, transcripts, and notes you can see. It starts disabled;
turning it on (Space in `/plugins`, or `/plugins enable grain`) opens its
settings menu. Until you have opened the menu once or picked a key, loading it
prints a line pointing there.

`/grain`, `c` in `/plugins`, or `/plugins configure grain` opens the settings
menu. Enter edits a row, `r` resets it to its default, and Esc or **Save &
close** leaves the menu. Each change is saved to the plugin's settings at once
and applies to the next prompt. Reopen it any time to change a setting or pick a
different key:

```text
 Grain
> Token                    browser sign-in
  Tools                    read-only
  Server instructions      included
  Save & close
```

| Row | Setting | Default | Does |
|---|---|---|---|
| Token | none (see below) | browser sign-in | where the Grain token comes from |
| Tools | `read_only` | `true` | read-only offers only the tools Grain marks read-only; all tools also lets the agent create clips and tag meetings |
| Server instructions | `include_instructions` | `true` | pass Grain's own instructions for its tools to the agent |

Grain's MCP endpoint is fixed, and the capability has no workspace, base URL, or
project option, so the menu has none either.

No token goes in plugin settings, which are plaintext SQLite. The plugin takes
the first of these that applies:

1. The `GRAIN_ACCESS_TOKEN` environment variable. The Token row shows it and says
   it overrides the choice there while it is set.
2. A [saved API key](#saved-api-keys) picked on the Token row, or with
   `/grain key`. The searchable picker lists your `/keys` entries by name; you can
   also type a token (masked), which is saved in `/keys` as `GRAIN_ACCESS_TOKEN`,
   the name harness's `Grain` documents. CLAI keeps only the key's name and looks
   the key up on every run, so replacing it in `/keys` takes effect on the next
   prompt, deleting it makes Grain fail rather than connect without it, and
   `/keys` refuses to rename it while Grain uses it. Several plugins can share one
   named key, the way `vllm` and `openrouter` connections can. Choose "No API
   key" to go back to the browser sign-in.
3. A browser sign-in. The first prompt that connects opens your browser and
   prints the sign-in URL, in case the browser does not open (over SSH, for
   example). The tokens go to the OS keyring (or CLAI's private credential file
   when there is no keyring), the way `/mcp` OAuth servers keep theirs, so later
   sessions refresh them instead of signing in again. A headless run
   (`clai2 -p`) cannot sign in: with no saved sign-in, its Grain connection fails
   and says so.

`/grain status` says which of the three this session uses. `/grain logout`
forgets the browser sign-in and the tokens the session holds, so the next prompt
that uses Grain signs in again. It cannot revoke `GRAIN_ACCESS_TOKEN` (unset it,
then `/plugins reload grain`) or a `/keys` entry (pick "No API key").

If you enabled `grain` from the old `/plugins` catalog, which saved
`pydantic_ai_harness.grain:Grain` under that id, CLAI loads this plugin in its
place and keeps it enabled or disabled. A declaration you gave
settings with `/plugins add` stays as you wrote it.

The settings can also be given up front:

```text
/plugins add grain pydantic_clai2.builtin_plugins.grain '{"read_only": false, "include_instructions": true}'
```

## Managing plugins

`/plugins` on its own opens a full-screen menu, the same kind Code Puppy uses
for `/agent` and `/mcp`:

```text
 Plugins
 search: (type to filter)

 > ● coder    on      built-in   │ coder
   ● notify   on      drop-in    │ on · built-in
   ○ audit    off     installed  │
   ○ broken   failed  drop-in    │ source   pydantic_clai2.builtin_plugins.coder
   Save & close                  │ provides 1 capability
                                 │ settings press c to configure

 / search · space on/off · c configure · r reload · d remove · esc close
```

The left side lists every plugin with `●` for on and `○` for off, a coloured
status (`on`, `off`, `failed`, or `idle` when enabled but not loaded yet), and
where it came from (`built-in`, `project`, `drop-in`, or `installed`). The
colours follow your `/theme`. The right side shows details for the highlighted
one: a description, its source, what it registered, whether it has a settings
menu, and the last error if loading failed. The description is the first
paragraph of the plugin's docstring: the class's for `module:Class`, otherwise
the module's. CLAI reads it from the source file without importing it, so a
plugin that is off runs no code to describe itself.

Space turns the highlighted plugin on or off; `c` configures it, `r` reloads
it, and `d` removes it. Press `/` to search: while searching, every key you type
filters the list (including `c`, `r`, `d`, `q`, and Space), matching ignores case,
Backspace edits the search, and Up and Down move between matches. Enter ends the
search and keeps the matches, so the keys above act on them; `/` resumes the
search. Esc clears the search, or the matches Enter kept, including when nothing
matches; with no filter, Esc closes. Changes apply immediately; there is no
pending save step, so the **Save & close** row, Enter, `q`, Esc, and Ctrl-C just
close.

Turning a plugin on with Space opens its settings menu straight away when it
offers one (see [`configure`](#offer-a-settings-menu-async-def-configureself)).
When you leave that menu, `/plugins` comes back with the plugin's message in
the details panel. `c` opens the settings menu of a plugin that is already on.
Turning a plugin off never opens a menu, and a plugin without a settings menu
just turns on.
Closing returns to the prompt without printing the plugin list. Use `/plugins list`
to print it.
Adding a plugin needs a name and a module, so that stays a typed command.

At startup, an enabled declaration whose own module cannot be imported (for
example a built-in saved by a newer CLAI) is skipped quietly and listed as
failed. A plugin that is installed but fails to import one of its dependencies
is still reported, and `/plugins enable`, `add`, and `reload` always report
failures.

With arguments `/plugins` is a plain command. The standalone `clai2 plugins ...`
commands edit saved declarations without loading plugin code. Git installation,
configuration menus, and live reloading are available only inside a CLAI session:

| Command | Does |
|---|---|
| `/plugins list` | show every plugin and whether it is on |
| `/plugins add NAME module[:Class] [JSON]` | save it and load it now |
| `/plugins add GIT_URL` | clone a trusted repository and load its plugin (in a CLAI session only); see [where plugins live](#where-plugins-live) |
| `/plugins remove NAME` | forget an installed declaration, keeping any Git checkout on disk; persistently disable a drop-in (delete its file yourself to remove it); reset a built-in or project-declared plugin to its declaration |
| `/plugins enable NAME` / `disable NAME` | load or unload, remembered across restarts; enabling (like `add`) opens the plugin's settings menu if it has one |
| `/plugins configure NAME` | open a loaded plugin's settings menu, if it overrides `configure` (in a CLAI session only); `enable` and `add` open it too |
| `/plugins reload NAME` | re-import the file and load it again (in a CLAI session only; does not fetch Git updates) |
| `/reload` | reload CLAI's own Python modules for development and rebuild the shell without restarting the process |

`/reload` takes no arguments. It uses `importlib.reload`, preserves the conversation,
agent, selected model, and active settings, and reloads enabled plugins against
the refreshed shell types. Each loaded plugin receives `session_end` before reload
and `session_start` when loaded again. Plugin hosts and their registrations are
recreated, but installed module globals not overwritten by the new source can
survive. Initialize mutable state in `__init__`. Disabled and unapproved project
plugins stay off. Failed imports or shell rebuilds restore the
previous module bindings and report the error, but cannot undo import-time side
effects. Reload ordering is planned from current module-scope source imports,
including newly referenced local modules and package initializers. Lazy imports
inside functions and `TYPE_CHECKING` guards do not create eager dependencies;
new modules are imported only when reached by the updated code. Literal guards
and direct platform/version comparisons select their active branch without
executing guard expressions. Other conditions are analyzed conservatively and
may require a restart if their alternatives form a cycle. Invalid source
or a detected import cycle fails before module reloads begin. Restart for changes
to startup code, dynamically loaded dependencies, or agent construction.
Third-party dependencies are not recursively reloaded.

If loading fails or is cancelled after the plugin was built, its `on_session_end`
receives `reason='error'` under cancellation shielding before the plugin is dropped.
It has a five-second cooperative timeout, and an error or timeout is reported
without replacing the load error. Guard cleanup on what was actually acquired;
it may run before `on_session_start` finishes.
Handlers must cooperate with cancellation: blocking code and additional shields
can exceed that timeout. Caller cancellation still propagates after cleanup;
cleanup errors do not replace the original load error.

What "load" and "unload" mean for your plugin:

- Load builds the plugin class, calls its `get_*` methods once, and then runs
  `on_session_start`, so a plugin loaded mid-session sees the same first event as
  one loaded at start.
- Unload runs `on_session_end`, then drops everything the plugin declared:
  commands, capabilities, renderers, status segments, spinners, and model
  providers. Nothing else is touched.
- `/plugins` can load and unload during a turn. The agent run already in
  progress keeps the capabilities it bound when it started, so a plugin's tools,
  instructions, and hooks join or leave the agent on the next prompt. Everything
  else changes at once. A plugin unloaded mid-turn gets `on_session_end` once
  the turn ends, even if it was cancelled, because the run may still use what it
  declared. A reload mid-turn can therefore start the new instance before the
  old one ends, which is one more reason to keep state on the instance.
- A plugin's code that `/plugins` runs mid-turn (`on_session_start`,
  `configure`) can use `host.full_screen()`; the command already owns the
  screen, so the block does not wait for it.
- Drop-in entry modules load from current source. Installed entry modules use
  `importlib.reload`, which retains globals absent from the new source. Keep
  plugin state on the instance, set up in `__init__`.

## The first plugin: give the agent web search

Pydantic AI Harness ships capabilities that are plugins as they are. `ExaSearch`
adds `web_search` and `get_page` tools backed by [Exa](https://exa.ai). Install
the extra and set the key, then add the class by name:

```sh
pip install 'pydantic-ai-harness[exa]'
export EXA_API_KEY=...
```

With `uv tool`, extra packages installed using `--with` must be added again after
`/update`. See [Updating](README.md#updating) for how CLAI installs updates.

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

The JSON is passed to the constructor, so any keyword `ExaSearch` accepts that
JSON can express works here; an option that takes a Python object, like
`client`, needs the file form below. Ask the agent something that needs the web on the next prompt and it has
the tools. `/plugins disable exa` takes them away again.

The same thing as a plugin file, `~/.config/pydantic-clai2/plugins/search.py`,
which is the shape to start from when you want more than one capability, or a
`/command`, or a hook alongside it:

```python
from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import Plugin


class Search(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (ExaSearch(num_results=8, text_summary=True),)
```

A plugin is a subclass of `Plugin`, declared the way a Pydantic AI
[capability](https://pydantic.dev/docs/ai/core-concepts/capabilities/) is: each
thing it contributes has a `get_*` method, and a plugin overrides only the ones it
needs. CLAI builds one instance when the plugin loads, calls each `get_*` method
once, and keeps what they returned until the plugin is unloaded or CLAI quits.
Any Pydantic AI capability goes in `get_capabilities`, so `YouSearch` from
`pydantic_ai_harness.youdotcom`, core's `WebSearch`, or one you wrote yourself
all work the same way.

A module that defines exactly one public `Plugin` subclass can be named on its
own (`/plugins add search my_package.search`). Name the class with
`module:Class` when a module defines several. `module:Class` may also name a
capability class, as `ExaSearch` does above, and the JSON settings become its
constructor's keyword arguments.

## Reacting to a moment

Plugins are not only for tools. This one rings the terminal bell when a turn
finishes, so you can tab away during a long run:

```python
from pydantic_clai2.plugins import Plugin, TurnEnd


class Bell(Plugin):
    async def on_turn_end(self, event: TurnEnd) -> None:
        self.host.console.bell()
```

## What a plugin declares

Everything a plugin can contribute is a method on `Plugin`, with a default that
contributes nothing:

| Method | Contributes |
|---|---|
| `get_capabilities()` | capabilities (or per-run capability functions) added to every agent run |
| `get_commands()` | `/commands` |
| `render(event)` | a drawing for a stream event, or `None` for the default |
| `get_status_segments()` | text appended to the status row |
| `get_spinners()` | working animations offered by `/spinner` |
| `get_model_providers()` | `PREFIX:NAME` models CLAI can run |
| `configure()` | the settings menu `/plugins` opens |
| `on_session_start` / `on_session_end` / `on_turn_start` / `on_turn_end` / `on_plugin_load_failed` | handlers for CLAI's own moments |

The class says what settings it takes, and `self.host` is what it can reach at
runtime: the console, the conversation, the status row, the full screen, and its
saved settings. Set up state in `__init__`; call `super().__init__(host, settings)`
first.

### React to CLAI's moments: `on_session_start`, `on_session_end`, `on_turn_start`, `on_turn_end`, `on_plugin_load_failed`

Five `async` methods fire outside the agent run, in the shell:

| Method | When | Event fields | Can change things? |
|---|---|---|---|
| `on_session_start` | CLAI has started, before the first prompt, or the plugin loaded mid-session | `agent`, `settings` | no |
| `on_session_end` | CLAI is quitting, or the plugin is unloading | `reason`: `exit`, `eof`, or `error` | no |
| `on_turn_start` | you pressed Enter on a prompt | `text` | yes: edit `event.text`, or `event.cancel()` |
| `on_turn_end` | the turn finished, failed, or was interrupted | `text`, `outcome`, `result`, `error` | no |
| `on_plugin_load_failed` | startup loading finished, once per reported plugin failure | `plugin`, `error` | no |

`on_plugin_load_failed` receives a `PluginLoadFailed` event with the failed plugin's
name and original exception. Every successfully loaded plugin receives it, regardless
of load order. It covers startup loading, not individual `/plugins` actions. Declarations
whose own module is not installed stay quiet, as on the terminal; a missing dependency
inside an available plugin is reported. A handler failure is printed without stopping
startup or preventing other handlers from running.

Ctrl-C during an agent run keeps the prompt and captured partial messages in
conversation history for the next turn. Cancellation still reaches the running
tools for cleanup; it does not undo completed side effects or retry the run.
Retained failed turns and restored interrupted sessions are marked interrupted so
core can close unanswered tool calls on the next prompt without replaying them.
A prompt cancelled by `on_turn_start` never starts an agent run and is not retained.
`/fork` fires both handlers for its background run too: an `on_turn_start` that cancels
the prompt refuses the fork, and `on_turn_end` runs when the fork finishes.

Codex token-refresh failures show `/login openai-codex` recovery advice, including
when the SDK wraps them as connection errors. This changes only the terminal
message: `TurnEnd.error` still contains the original exception and its chain.
Headless runs show the same advice on stderr and exit with code 1.

### Hook into the agent run: return a `Hooks` capability

Everything that happens inside an agent run is a Pydantic AI lifecycle hook, so
a plugin hooks it the way any agent does: return a
[`Hooks`](https://pydantic.dev/docs/ai/core-concepts/hooks/) capability, or your
own `AbstractCapability` subclass, from `get_capabilities`.

```python
from collections.abc import Sequence

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from pydantic_clai2.plugins import Plugin


class Audit(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        hooks = Hooks[None]()

        @hooks.on.before_tool_execute
        async def log_call(
            ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
        ) -> ValidatedToolArgs:
            self.host.console.print(f'calling {call.tool_name}')
            return args

        return (hooks,)
```

The ones people reach for:

| `hooks.on.` | When |
|---|---|
| `before_run` / `after_run` | an agent run starts / finishes |
| `before_model_request` | just before the model is called; you can edit the request |
| `before_tool_execute` | a tool is about to run; raise to stop it |
| `after_tool_execute` | a tool has returned |
| `tool_execute_error` | a tool raised |
| `event` | every stream event, or only the classes you name |

The full list and every signature are in the
[hooks reference](https://pydantic.dev/docs/ai/core-concepts/hooks/).

### React to a typed event: `hooks.on.event(EventClass)`

Tools and capabilities emit typed events (a shell started, a file was written).
Name the classes you want:

```python
from collections.abc import Sequence

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks
from pydantic_ai_harness.filesystem import FileWrittenEvent

from pydantic_clai2.plugins import Plugin


class WriteLog(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        hooks = Hooks[None]()

        @hooks.on.event(FileWrittenEvent)
        async def log_write(ctx: RunContext[None], event: FileWrittenEvent) -> None:
            self.host.console.print(f'wrote {event.path}')

        return (hooks,)
```

A capability class of your own can do the same with core's `@on_event(EventClass)`
method decorator. Some events let you say no. If the event has a `cancel()`
method, calling it stops the action before it happens (for example
`FileChangeRequestEvent`).

### Add a `/command`: `get_commands()`

```python
from collections.abc import Sequence

from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin


class Greet(Plugin):
    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='greet',
                description='Say hello',
                handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
            ),
        )
```

The handler gets the arguments as a list of strings and returns the text to show.
Return `''` when there is nothing to report, such as a menu closed without
changes: the shell prints nothing rather than blank lines. A one-line result
ending in ` unchanged.` (for example `GitHub settings unchanged.`) is treated
the same way, so no-op notices stay out of the transcript.
Arguments are split like a shell command line, so quotes group words. Pass
`raw=True` to receive the unsplit argument text as one string instead (an empty
list when there is none); `/fork` does this so prompts keep their apostrophes.
It may be `async`. Add `complete=` to offer Tab suggestions. The registry filters
command names and returned candidates by case-sensitive substring, replacing the
whole typed fragment when selected. Return full candidates, not just suffixes.
Command names are listed exact match first, then names that start with the typed
fragment; the first is highlighted as you type. Candidates keep the order
`complete=` returns them in, and none is highlighted until Tab or Up/down picks one.
Set `available=` to a zero-argument callable returning a boolean to gate dispatch,
help, and completion on live session state. It defaults to always available.
Unavailable commands retain their registered names and ownership, so they still
participate in duplicate checks and are removed on plugin unload.
Names must be unique;
clashing with a built-in is an error at startup, not a silent override.

Path-like input is not dispatched to commands. A slash, dot, or backslash in the
first token after the leading `/` makes it a prompt instead, so screenshot paths
such as `/Users/me/Desktop/Screen Shot.png` appear as follow-ups in the queue.
For path text not converted to an image attachment by the editor, surrounding
whitespace is trimmed and the remaining text reaches the agent without further
rewriting. Quoted paths are prompts too. Unknown command-shaped names such as
`/missing` still report an error. Routing itself does not read files; the editor's
separate image-paste handling can attach existing images before routing. See
[Image input](#image-input) for the resulting hook payloads.

### Give the agent tools or instructions: `get_capabilities()`

```python
from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability, Capability

from pydantic_clai2.plugins import Plugin


class Words(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        tools = Capability[None](instructions='Prefer British spelling.')

        @tools.tool_plain
        def word_count(text: str) -> int:
            return len(text.split())

        return (tools,)
```

An entry may also be a function that takes a `RunContext` and returns a
capability (or `None`), for tools that should only exist in some runs, or that
read settings saved since the plugin loaded.

Markdown in streamed answers and thinking uses OSC 8 hyperlinks for link labels
when writing to a terminal. The URL is also shown as text. Transcript replay keeps
hyperlinks after resize, but does not replay clipboard, title, or palette commands.
Redirected Markdown output does not emit hyperlinks. Destinations longer than
2,048 characters stay visible but do not get clickable metadata.

### Draw an event yourself: `render(event)`


Built-in tool rendering shows one summary line per call by default, clipped to
the terminal width and followed by a blank line. Tool names are pink; arguments
and bullet markers are muted grey. Successful file writes and edits show their
diffs even with `display.tool_output` off. Shell output and completion details and grep results
are hidden from the terminal, not from the model. Set `/set display.tool_output true`
to show those details; `display.shell_lines` and `display.grep_lines` then control
preview lengths (20 lines each by default). This setting does not suppress file
diffs, plugin renderers, or interactive questions. With `/set display.tool_calls grouped`,
calls no renderer claims are counted by tool on one line instead, except `edit_file` and
`write_file`, which still print their summary and diff; `display.tool_output`
has no effect, and anything a renderer draws ends that line.

CLAI shows unknown tool calls as `● tool_name`, with the name in pink. To show something
better, return a Rich renderable (a `str` is fine). Return `None` to say "not mine,
use the default".

```python
from pydantic_ai import AgentStreamEvent
from pydantic_ai_harness.filesystem import FileWrittenEvent
from rich.console import RenderableType

from pydantic_clai2.plugins import Plugin


class Writes(Plugin):
    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        if isinstance(event, FileWrittenEvent):
            return f'wrote {event.path}'
        return None
```

CLAI flushes any streaming text before it prints what you return, so your output
never lands in the middle of a paragraph.

`/resume` shows the restored conversation through the same renderers. It sends
`FunctionToolCallEvent` and `FunctionToolResultEvent` built from the saved
messages, so a renderer for those also draws resumed turns. Events that only
streamed, such as `FileWrittenEvent`, are not saved and are not sent again.

### Take the whole screen mid-run: `async with self.host.full_screen()`

A widget opened from inside a tool call, including the inline `ask_user` picker,
has to wait for streamed text to finish and the editor and status row to
get out of the way. `host.full_screen()` flushes pending output, suspends the
editor's input reader, and restores the editor and its draft when the block exits.
The editor remains active during agent turns. Enter queues a separate turn with
its own `turn_start` and `turn_end` hooks. Alt+Enter (Option+Enter) sends the typed
draft, or with an empty draft the oldest queued follow-up, to the active run
through core's `RunContext.enqueue(priority='asap')`. Accepted steering messages
appear immediately in the transcript and are enqueued for the next model request,
without starting another turn, interrupting the current response, or cancelling
tools. Each press sends one message. Slash commands, `!` shell
commands, and exit signals are never steered: such a draft, or any draft the run
does not accept, is taken as Enter would take it, and such a queued message stays
queued and is not skipped over. When idle, Enter starts a turn. Shift-Enter inserts
a newline when the terminal distinguishes it from Enter. Ctrl-J inserts a newline
on terminals that do not, including GNOME Terminal/VTE on Ubuntu. xterm and Kitty
keyboard reporting is enabled only while the editor owns input and released for
menus and on exit.
Option+Backspace (Alt+Backspace) deletes the word before the cursor, like Ctrl-W,
including trailing whitespace. Spaces, tabs, and newlines separate words. Text
after the cursor is preserved. Your terminal must send Option as Alt/Meta for
this shortcut; legacy and modified-key encodings are supported.

Completion rows remain visible while a replacement lookup runs, but stale results
cannot be selected. Popup height changes reuse available space without adding
blank transcript lines on each key. Completion providers should be read-only.
Their errors are shown in the footer rather than ending the session; on menu
handoff or shutdown, a blocked synchronous lookup may finish in the background
and its result is ignored. A single daemon worker and a latest-only queue bound
this work; a stuck provider delays further lookups, not input or process exit.
While work or turn lifecycle
hooks are active, a `Working` label and spinner appear in the editor's top border.
The spinner uses the same pink `ACCENT` as tool names, while the label and border
stay muted. The indicator appears
without adding an input row or changing the draft. The indicator uses the editor's refresh cycle, adds no
background task, and is hidden while a full-screen interface owns the terminal.
The editor and transcript share a Termflow live cell buffer on the alternate
screen. Only changed cells paint. PageUp/PageDown and mouse-wheel input scroll
output without changing the draft. New output does not move a scrolled view.
A mouse drag selects painted cells, and releasing it copies them to the clipboard.
On exit, the retained transcript prints into native terminal scrollback.
Resize, theme changes, and returning from a menu repaint from the transcript.
Assistant Markdown renders again at the new width and theme; tool and command
output rewraps, and a theme change repaints its colours role by role. Output
coloured with `theme.color(...)` or the theme's diff and syntax colours follows
`/theme`; other colours stay as printed. The transcript includes startup and
plugin lifecycle output. `/reload` preserves it, including when a running session
upgrades from the older scrollback implementation. It keeps styling, not arbitrary
terminal controls.
Use `self.host.full_screen()` for widgets instead of printing cursor-control
sequences into the transcript. `run_worker` temporarily leaves the live panel
for full-screen widgets; output continues into the transcript without painting
until the widget closes. Inline questions keep the panel on screen.
The Termflow smoothing defaults match Code Puppy: responses use 12 ms ticks, a 0.5-second
catch-up window, and at least one character per tick; thinking uses 20 ms ticks,
a 0.4-second window, and at least two characters per tick, and renders as dimmed
Markdown after the `Thinking` heading on the same row. Plugins do not need their own redraw logic. Pending message previews appear
above the editor in execution order (`Follow-up:` for messages, `Command:` for
slash commands), and disappear when consumed. The preview is read-only; clipping
and flattening multiline text for display do not change the submitted text.
Control bytes are escaped in completion labels, queued previews, and prompt echoes
rather than being executed as terminal commands.
Esc in the live editor cancels active work, including turn lifecycle hooks,
without clearing the draft or requesting exit. While a plugin owns the screen,
its menu retains control of Esc.
Slash-command handlers already
run with the editor suspended; tool-driven widgets must take the screen explicitly:

```python
from pydantic_clai2.plugins import Plugin


class Chooser(Plugin):
    async def choose(self) -> str:
        async with self.host.full_screen():
            return await show_my_menu()
```

When no editor or stream is active, taking the screen is a no-op. It only settles
the screen; drawing, and restoring the terminal afterwards, is the widget's job. One
widget owns the screen at a time: a second `full_screen()` (from a parallel tool
call, say) waits for the first block to exit. Do not nest it inside itself.

### Declare your settings: `Plugin[Model]`

The JSON passed to `plugins add` is validated against the model named as the
class's first type parameter, and the result is `self.settings`:

```python
from pydantic import BaseModel

from pydantic_clai2.plugins import Plugin, TurnEnd


class NotifySettings(BaseModel):
    sound: bool = True


class Notify(Plugin[NotifySettings]):
    async def on_turn_end(self, event: TurnEnd) -> None:
        if self.settings.sound:
            self.host.console.bell()
```

A plugin with no type parameter takes no settings (`NoSettings`), and rejects
any it is given. Bad or missing values fail at startup with a message naming
your plugin. A plugin with an agent dependency type names it second, as
`Plugin[NotifySettings, MyDeps]`.

`self.settings` is what the plugin loaded with. To edit settings from inside the
plugin, call `self.host.save_settings(model)` (see
[`configure`](#offer-a-settings-menu-async-def-configureself));
`self.host.settings(Model)` returns them from then on. Read them per run (for
example in a capability function returned from `get_capabilities`) so an edit
also reaches the next turn from a plugin command, without a reload.
`google_workspace` is a worked example. CLAI ignores unknown names in its own
saved settings and preserves their values for other versions or branches. This
does not relax validation of plugin declarations or `self.host.settings(Model)`.

### Settings that need a feature: `host.settings(Model, requires=...)`

Every CLAI on a machine shares one settings database, whatever code it runs: other
worktrees, branches, and installs. When a setting's valid values or meaning depend
on code that other builds may lack, tag it with the feature it needs:

```python
settings = host.settings(Options, requires={'mode': ['fancy-mode']})
```

The contract:

- **Feature names** are lowercase words joined by hyphens, such as
  `stock-bound-delegation`. A build lists the ones it supports in
  `SUPPORTED_FEATURES` (`pydantic_clai2/config/features.py`). Never rename or reuse one.
- **Keys** are the saved names, aliases included. An unknown key or a badly formed
  name raises `ValueError` at activation.
- **Writers attach tags for you.** `host.save_settings`, `/plugins add`,
  `/plugins enable` and `disable`, and the settings menus store them beside the
  declaration, in their own table. Your settings JSON never holds them.
- **A build that lacks a feature, or does not know its name, ignores that one
  setting.** It uses the built-in declaration's value, or your model's default,
  keeps your other settings, and prints one line per plugin:
  `coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.`
  Reading never rewrites the database, and saving from that build keeps the
  ignored value and its tag. A tag goes away only when its value changes.
- **Tag only settings whose default is the safe choice.** Dropping a value means
  using the default, so a default must never be looser than what it replaces.
- **Builds older than tags ignore them** and apply every setting as before.
  Tags cannot protect those builds.

A capability class declared as `module:Class` has no `Plugin` subclass; CLAI lists its
tags in `CAPABILITY_REQUIREMENTS`, keyed by that factory string.
`clai2 plugins add` from the command line imports nothing, so it attaches only
those; a plugin's own tags are attached the next time it saves or is enabled.

If an untagged setting still makes a capability that CLAI built from a
`module:Class` declaration (such as `coder`) raise `UserError` while a run is
set up, that turn fails closed with the plugin named, and CLAI leaves that
capability out of later turns; `/plugins reload NAME` brings it back. Nothing is
retried, so no other capability is set up twice. Errors from the model, from tools, or
from a raising plugin handler still fail the turn as before, and capabilities an
`get_capabilities` returns, or that contain a `Hooks`, are never left out.

### Keep secrets in `/keys`: `KeyReference`, `SavedKey`, `host.save_settings`

Plugin settings are stored in plaintext SQLite, so a token, API key, or client
secret must never be one of them. Keep the secret in the named API key store that
`/keys` manages, and put only its name in your settings. Use the conventional
uppercase variable name as the label, such as `GITHUB_TOKEN` or `SLACK_BOT_TOKEN`,
so plugins that need the same credential share one key; replacing it in `/keys`
reaches all of them. The label does not export or read an environment variable.

```python
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.capabilities import AgentCapability

from pydantic_clai2.config.api_keys import KeyReference, SavedKey
from pydantic_clai2.plugins import Plugin


class MySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token: KeyReference = Field(default_factory=lambda: KeyReference(name='MY_SERVICE_TOKEN'))


class MyPlugin(Plugin[MySettings]):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        auth = SavedKey(name=self.settings.token.name, setup='Add MY_SERVICE_TOKEN in /keys.')
        return (MyService(auth=auth),)  # your capability, taking an `auth` function
```

`SavedKey` is a capability `auth` function. It looks the key up on every run and
raises with `setup` when the key is missing, so a deleted key fails closed rather
than falling back to something else. To let the user choose a key, call `prompt_api_key(prompt=..., label=...)`: it
returns a `KeyReference` to a saved key, a masked new value for you to
`save_key(name=..., value=...)`, or `None` when cancelled. Then call
`self.host.save_settings(settings)` with the new reference. It saves your plugin's
declaration as `plugins add` would, and `self.host.settings(Model)` returns the new
values from then on.

### Offer a settings menu: `async def configure(self)`

Override `configure` with an async method that shows a settings menu and returns
a line to show afterwards. CLAI opens it when the plugin is turned on (Space in
`/plugins`, `/plugins enable`, or `/plugins add`), on `c` in `/plugins`, and on
`/plugins configure NAME`. Save each change with `self.host.save_settings(model)` as
the user makes it; settings are stored in plaintext, so keep secrets in `/keys`
and save only a key's name: let the user pick one with
`prompt_api_key(prompt=..., label=...)` and remember only a `KeyReference` to
it. When the saved settings changed, CLAI loads the plugin again afterwards, so a
fresh instance is built from them.

Build the menu with the shared field editor (`FieldMenu` and `run_flow` from
`pydantic_clai2.ui.menus.field_menu`, run through `run_worker` from
`pydantic_clai2.ui.menus.menu_worker`), the same one `/set` uses. Its last row is
**Save & close**: every edit is already saved, so choosing it, like Esc, just
leaves the menu. A menu you build yourself ends with `save_and_close_item()` and
treats a result as closed when `picked(result)` is `None`.

The built-in `logfire_mcp` plugin is a complete example.

```python
class Notify(Plugin[NotifySettings]):
    async def configure(self) -> str:
        # `NotifySource` is your `FieldSource`: rows, current values, validation, apply, and reset.
        messages = await run_worker(lambda: run_flow(FieldMenu(NotifySource(self.host))))
        return '\n'.join(messages) or 'Notify settings unchanged.'
```

For a row that opens something other than a plain value, pass `submenus` to
`run_flow` to map the row's key to a function that returns messages. When that
function is itself async, such as `api_keys.prompt_api_key`, await
`field_menu.run_flow_async` from the `configure` function instead of running
`run_flow` in a worker: it runs each widget in its own `run_worker`, so the
submenu can open its own.

The built-in `github`, `grain`, `linear`, `notion`, and [`pylon`](#configuring-pylon) plugins are
complete examples; `linear` uses `run_flow_async`, and `pylon` steps out of the menu
worker to run the async key picker.

Plugin settings are plaintext SQLite. Never save a token, API key, or client
secret in them: keep it in `/keys` with `prompt_api_key` and `save_key`, save only
its name as a `KeyReference`, and call `resolve_key` when connecting, so a key
replaced in `/keys` applies and a deleted one fails closed. Name a new key with
the conventional environment-style label (`GITHUB_TOKEN`, `ORDINAL_ACCESS_TOKEN`)
so plugins that need the same credential share it. The label is only a name; it
does not export or read an environment variable.

For a token row, `pydantic_clai2.plugins.keys.choose_key(name=..., label=..., runners=...)`
shows the searchable `/keys` picker, or a masked input when no keys are saved.
It saves a new value under `name` only after confirming a replacement, and
returns a `KeyReference` to persist in place of the secret. Call it through
`plugin_keys.on_loop` from the menu's worker thread. The built-in `slack` plugin
is a complete example.

### Sign in through the browser: `pkce.PKCESignIn`

For a service whose OAuth needs a registered app but accepts PKCE instead of a
client secret (a public client), describe the app once and let CLAI run the
authorization-code flow, keep the tokens, and renew them:

```python
from pydantic_clai2.pkce import PKCESignIn, PublicClient

client = PublicClient(
    authorize_url='https://example.com/oauth/authorize',
    token_url='https://example.com/oauth/token',
    client_id=settings.client_id,  # public: plugin settings may hold it
    redirect_uri='http://localhost:53119/example/callback',  # registered with the app, fixed port
    scopes=('read',),
)
sign_in = PKCESignIn(client=client, account='example-oauth', service='Example', setup='/plugins configure example')
```

`await plugin_keys.browser_sign_in(sign_in, runners)` signs in from a settings
menu behind a waiting screen that shows the URL, and returns `False` when the user
presses Esc. Outside a menu, `await sign_in.sign_in()` opens the browser directly.
Before each run, `await sign_in.token()` returns an access token, refreshing it
first when it expires within five minutes and saving a rotated refresh token. It
raises `UserError` naming `setup` when there is no usable sign-in, so a plugin can
turn its tools off with that message. Refreshes are serialized across CLAI
processes, because a service that rotates refresh tokens invalidates the old one.
Token responses may report errors the standard way or, like Slack, with HTTP 200
and an `error` field; both are handled. The tokens are stored under `account` in
the credential store with the scopes the service granted, tied to the Client ID.
Changing the app, or asking for scopes the sign-in did not grant, means signing in
again. Services with Dynamic Client Registration need none of this: add them as
`/mcp` servers with OAuth.

### Reach the conversation and the status row: `host.conversation`, `host.status`

`host.conversation` is the retained history: `messages` is a snapshot,
`await commit_messages(...)` persists and swaps it between turns, and `resolved_model()` is the
model CLAI or the user selected for the next prompt. A capability that selects a model, such as
Logfire's `AgentControl`, can replace CLAI's default per request; `resolved_model()` and the
status row still name the selected one, and `/compact` and session naming (without a
`sessions.naming_model`) run on it. Local `!command` executions append a user message
with the command, stdout, stderr, and completion status. They do not start an
agent turn or fire turn hooks; the context reaches the model on the next prompt.

`host.session_id` is the current saved conversation ID, following `/clear` and
`/resume`. During a run it identifies that run's conversation, including a
background fork. It is `None` while startup resume is selecting a conversation,
or for a host without session persistence. Custom hosts can supply a
`session_id` callback to read their current ID.

`host.status` is the footer's state: `context_tokens` and `context_window`
render as compact used/max, such as `128k/1m`; `None` renders as `?`. Only set
`context_window` for a known capacity, not an assumed fallback. Set
`context_alert` to paint the figure in the warning colour. The built-in
`compaction` plugin fills these fields from Harness usage events, including an
explicit window override, and clears the window when unloaded. A host built
outside the shell gets an in-memory `Transcript` and a detached `Status`, so
tests need no special case. The status row itself is CLAI's; a plugin adds to it
with `get_status_segments`. While the main run waits on a subagent, the row
shows that subagent's figures instead. These fields keep the main conversation's
values meanwhile and show again when it settles; segments stay on the row.

The double-Esc rewind menu also uses `commit_messages` between turns. It removes
the selected prompt and later history, but does not undo plugin state, file
changes, or other tool side effects. It never replays tools or fires turn hooks.

### Add to the status row: `get_status_segments()`

Each segment is a function that takes no arguments and returns a short string. It
is appended after the built-in figures, painted muted, and dropped when the
plugin unloads. The built-in row accents output-token counts and tool names using
the selected theme; plugin fragments stay muted.

```python
import os
from collections.abc import Sequence

from pydantic_clai2.plugins import Plugin
from pydantic_clai2.ui.rendering.status import StatusSegment


class Where(Plugin):
    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (os.getcwd,)
```

The row is repainted about ten times a second, so `fn` runs that often: keep it
cheap, synchronous, and free of blocking IO. Return `''` to contribute nothing
for a frame. Fragments are truncated from the right on a narrow terminal and are
readable text only; the colours in the row belong to CLAI. A fragment that raises
shows its error name in the row instead, so one broken plugin cannot take the
footer down.

### Offer a spinner: `get_spinners()`

Adds a working animation to `/spinner`. The user selects it there or with
`/set display.spinner NAME`; offering it does not select it.

```python
from collections.abc import Sequence

from pydantic_clai2.plugins import Plugin
from pydantic_clai2.ui.rendering.spinners import Spinner, make_spinner


class Wave(Plugin):
    def get_spinners(self) -> Sequence[Spinner]:
        return (make_spinner('wave', ['~   ', ' ~  ', '  ~ ', '   ~'], interval=0.1, description='a small wave'),)
```

Frames are capped at 40 characters and padded to one terminal width, and
`interval` is clamped to 0.02-1 seconds per frame (0.2 by default). A blank or
spaced name, an empty frame list, or a control character in a frame raises
`ValueError` during activation. A plugin spinner replaces a builtin of the same
name; the user's `spinners.json` replaces both. Unloading the plugin removes it,
and a selected spinner that is gone shows the default `working`.

### Run models under your own prefix: `get_model_providers()`

Makes `PREFIX:NAME` a model CLAI can run, for a Pydantic AI `Model` that no core
provider builds, such as one authenticated with a subscription login:

```python
from collections.abc import Sequence

from pydantic_ai.models import Model

from pydantic_clai2.plugins import ModelProvider, Plugin


def resolve(name: str) -> Model:
    return MyModel(name, provider=MyProvider())


class MyService(Plugin):
    def get_model_providers(self) -> Sequence[ModelProvider]:
        return (ModelProvider(prefix='my-service', resolve=resolve, models=('fast', 'smart')),)
```

`models` are listed under the prefix in `/model add` and completed by `/set model`
and `/model add`; any other `my-service:NAME` can still be typed. `resolve` gets
`NAME` without the prefix. CLAI calls it in a worker thread before every run with
one of these models, so reading the keyring there is fine, and a sign-in made
since the last run applies. Raise `UserError` naming the setup step when it cannot
build the model; the run fails with that message.

The prefix starts with a lowercase letter, followed by lowercase letters, digits,
and hyphens. It cannot be one Pydantic AI or CLAI already runs, aliases included
(`anthropic`, `openai-chat`, `azure`, `openai-codex`, `vllm`, ...): that raises
`ValueError` during activation. Only a name with a colon is looked up, so a bare
`my-service` is never routed to the plugin. When two plugins register one prefix, the later one
wins. Unloading the plugin removes the prefix; a saved model under it stays in
`/model`, and runs with it fail as an unknown provider until the plugin is enabled
again. Users can remove an unused model and its saved settings with **Ctrl+D** or **Delete** in
`/model`, after confirmation. The current model and saved default are protected;
select another model or change the default with `/set model NAME` first. Deleting
a model does not unload its plugin or delete provider credentials.

`/model settings` offers generic controls (max tokens, temperature, custom
parameters) for plugin models. When `resolve` returns a model class of a provider
CLAI knows, such as an `AnthropicModel` subclass, set `settings_from='anthropic'`
(or `'openai'`, `'openai-chat'`, `'google'`) on the `ModelProvider` and these
models get that provider's controls instead, such as Claude's thinking mode and
effort. Any other value raises `ValueError`.

The built-in `/effort [VALUE|reset]` shortcut uses the active model's
`/model settings` effort control, including a plugin provider's `settings_from`
mapping. It saves values under the plugin model identifier, not the mapped provider.
Custom effort body parameters take precedence; `/effort` identifies these overrides
and asks you to remove them before saving a native effort value.

### Add a sign-in to `/login`: `get_logins()`

`/login NAME` signs in to a subscription: `openai-codex` and `github-copilot`
ship with CLAI, named after the model prefix each one unlocks (the earlier `codex`
and `copilot` still work), and bare `/login` asks which one to run. A plugin whose
models need a sign-in adds its own name next to them, ideally its model prefix:

```python
from collections.abc import Sequence

from pydantic_clai2.plugins import ModelProvider, Plugin, PluginLogin

PROVIDER = ModelProvider(prefix='my-service', resolve=resolve, models=('fast', 'smart'))


async def sign_in() -> str:
    ...  # run the OAuth flow, save tokens to the keyring
    return 'Signed in to My Service.'


class MyService(Plugin):
    def get_model_providers(self) -> Sequence[ModelProvider]:
        return (PROVIDER,)

    def get_logins(self) -> Sequence[PluginLogin]:
        return (PluginLogin(name='my-service', handler=sign_in, models=PROVIDER.names),)
```

`/login my-service` awaits `sign_in` and shows the message it returns, and `/login`
completes the name. Raise `UserError` when signing in fails, and keep tokens in the
keyring, never in plugin settings. The name uses the same format as a model prefix
and cannot be one of CLAI's sign-ins, which raises `ValueError`; when two plugins
add one name, the later one wins. Unloading the plugin removes it.

Once the sign-in succeeds, `models` (as `PREFIX:NAME`, such as the provider's
`names`) are added to the saved model list, so `/model` and `/model settings`
offer them without adding them first. A failed sign-in adds nothing.

### Several accounts: `resolve_profile` and `profile_handler`

Users can sign in to more than one account per provider. An account other than
the default is a profile: `/login NAME@PROFILE` signs in to it, and
`PREFIX@PROFILE:MODEL` runs on it. A fallback chain (made in `/model`) can then pool
accounts, moving to the next one when a request fails. To support profiles, set
both optional fields:

```python
PROVIDER = ModelProvider(
    prefix='my-service', resolve=resolve, resolve_profile=resolve_profile, models=('fast', 'smart')
)
PluginLogin(name='my-service', handler=sign_in, profile_handler=sign_in_profile, models=PROVIDER.names)
```

`resolve_profile(NAME, PROFILE)` builds the model for that account, in a worker
thread like `resolve`. `profile_handler(PROFILE)` signs in to it; on success the
`models` are saved with the profile, as `PREFIX@PROFILE:NAME`. Keep each profile's
tokens apart, for example one keyring entry per profile, and keep the default
profile where it was so existing sign-ins still work. A profile name is 1 to 32
lowercase letters, digits, hyphens, or underscores, and never `default`, so you can
keep the default account under that name. CLAI checks it before calling you. Without these fields, a model or sign-in naming a profile fails with a
message saying the plugin has one account.

Each successful `/login NAME` or `/login NAME@PROFILE` puts the account in
`/accounts`, under your first model's prefix (`claude-code`, not `claude`), so
users can add, order, and pick your accounts without typing a profile, and
`PREFIX@*:MODEL` tries them in that order. So does `PREFIX:MODEL` once two or more
are listed, unless the user turns `accounts.pool` off; `PREFIX@default:MODEL`
reaches your `resolve`, never `resolve_profile`. CLAI cannot see your tokens, so it
lists an account from the time it signed in until the user removes it there;
removing one does not sign it out of your plugin, and the menu says so. Offer your
own logout for that.

### Account usage: `usage`

Set `usage` on your `PluginLogin` to show each account's current usage in
`/accounts`, as CLAI does for ChatGPT/Codex and Copilot. CLAI calls
`usage(PROFILE)` (`None` for the default account) in the background while the
menu is open, every account at once, and stops waiting after a few seconds.
Return an `AccountUsage`: its `windows` (each a `UsageWindow` with a short
`label`, `used_percent` from 0 to 100, and an optional timezone-aware
`resets_at`), most pressing first, and its `plan` when you know it. Raise
`UserError` with a short reason when usage is unavailable; the menu shows it.

```python
from pydantic_clai2.plugins import AccountUsage, PluginLogin, UsageWindow


async def usage(profile: str | None) -> AccountUsage:
    limits = await my_service.limits(profile)  # your own client, with your tokens
    return AccountUsage(
        windows=(UsageWindow(label='5h', used_percent=limits.session, resets_at=limits.session_resets),),
    )


PluginLogin(name='my-service', handler=sign_in, models=PROVIDER.names, usage=usage)
```

The row shows the first two windows, such as `5h 92% · 7d 43%`; the details
panel shows a bar and reset time for each.

## Rules that keep plugins predictable

- Handlers are `async`. There is no sync variant of anything.
- `get_*` methods are called once, when the plugin loads. Return what the plugin
  offers; do not do slow or blocking work there. Use `on_session_start` for
  that, but keep it asynchronous: it runs on the event loop, so offload blocking
  work with `anyio.to_thread.run_sync` or a worker.
- CLAI's `on_*` handlers return `None`. Core hooks keep their exact core return
  contracts: for example `before_model_request` must return its `ModelRequestContext`.
  For cancelable host events, edit the event or call `event.cancel()`.
- Raising in `on_turn_start` prevents the turn. Raising in core `before_tool_execute`
  fails the agent run, not only that tool. Use core's documented tool-denial
  mechanisms when the model should recover instead of ending the run, such as
  `pydantic_ai.exceptions.SkipToolExecution` for skipping an individual tool.
- Startup plugins load in alphabetical ID order. CLAI's `on_*` handlers run in
  load order, while `/plugins list` remains alphabetical. Core hook ordering
  follows core composition, including reverse order for `after_*` hooks. The first renderer that returns something wins. A
  plugin loaded later goes to the end of the line.
- Anything you print, print through `self.host.console`, so it stays in step with
  streaming output.

## Testing a plugin

`load_plugin(PluginClass, host)` builds a plugin and collects its contributions,
exactly as the loader does. `PluginHost` is an ordinary object: build one with a
`Console` writing to a `StringIO`, then send hand-made events through
`dispatch`. No terminal, no model, no network.

```python
from io import StringIO

from rich.console import Console

from pydantic_clai2.plugins import PluginHost, TurnEnd, load_plugin

host = PluginHost(name='notify', console=Console(file=StringIO()), settings={})
plugin = load_plugin(Notify, host)
await plugin.dispatch(TurnEnd(text='hi', outcome='completed'))
assert plugin.capabilities == ()
```

`plugin.commands`, `plugin.status_segments`, and the rest hold what the plugin
declared. A plugin that reads the history gets a `Transcript` by default; pass
`conversation=Transcript(messages=[...], model=TestModel())` to seed it.

## vllm connection

Open `/model add`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/model add`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.


## Saved API keys

Startup [`.env` loading](README.md#environment-variables) makes values available to
providers and plugins that read the environment. It does not import credentials
into `/keys` or replace a plugin's saved-key picker.

Open `/keys` to browse and manage saved API keys in a full-screen menu.
Use A to add, Enter to replace a value, R to rename, and D to delete with
confirmation. Values are masked and are not shown in previews. Changes save
immediately; Esc or Ctrl-C closes the menu. Errors appear inside the menu.

The compatibility command `/set api_key` prompts for a name and masked API key. Names are trimmed
and uppercased automatically, so `my_vllm_key` becomes `MY_VLLM_KEY`. Use letters,
numbers, and underscores, starting with a letter or underscore. Saving an existing
name asks before replacing it. Ctrl-C or Ctrl-D cancels without saving. Do not put
the secret on the command line.

Consumers store a key's name, not its value, and look it up each time they
connect, so replacing a value in `/keys` updates every consumer and a deleted key
makes them fail with an error. Renaming a key that vLLM, OpenRouter, or a
plugin uses is refused until they are pointed at another key. Several
consumers can share one entry: name keys with the conventional variable name for
the service, such as `LINEAR_API_KEY` or `GITHUB_TOKEN`, and every plugin for that
service can pick the same entry. The names are labels only; CLAI does not export
them as environment variables.

When saved keys exist, vLLM's token prompt, OpenRouter's **Enter API key** flow,
and plugins such as [`google_workspace`](#google_workspace-gmail-calendar-and-drive-tools),
[`grain`](#grain-meetings-with-a-saved-sign-in), [`linear`](#linear-issues-and-projects),
[`logfire_mcp`](#logfire-mcp-query-your-telemetry), [`notion`](#notion-workspace-tools),
[`pylon`](#pylon-support-issues-and-accounts-in-pylon), and
[`slack`](#slack-your-slack-workspace-as-you) show a
searchable list of names. Choose one, enter a different key privately, or
choose **No API key** for vLLM. Esc closes the picker without connecting. Browser
login flows are unchanged. Select keys only for endpoints you trust.

Named keys use the existing credential backend, separate from provider logins and
SQLite settings. If no OS keyring exists, CLAI warns that it saved them in the
per-user `0600` plaintext file `credentials-api-keys.json` in its config directory.
Key values never appear in the picker or confirmation. Names are labels, not
exported environment variables. Selecting a saved key stores a reference, not a copy. Discovery and each new
turn resolve its current value. Replacing a key updates connections that reference
it. Deleting it makes those connections fail until you restore the same name or
reconfigure them. Keys referenced by saved connections, including Pylon's, cannot be renamed;
a key named in plugin settings, such as `logfire_mcp`'s, can, and that plugin's runs then fail
with the missing name until you pick it again. One named key can serve several connections and
plugins, so built-in plugins look for conventional labels, such as `LOGFIRE_API_KEY`, that other
tools can share. A cross-process lock
serializes key changes and connection saves so concurrent CLAI sessions do not
overwrite each other's key edits. The lock file contains no credentials.

Manage plugin secrets here too: a plugin that needs a token stores a reference to
a named key, never the token itself, because plugin settings are plaintext
SQLite. Several connections can share one key: name it the way its provider's
tools do, such as `SLACK_USER_TOKEN` or `GITHUB_TOKEN`, and choose that name in
each connection. Replacing it then updates all of them.

Existing connections with inline credentials, manually entered connection keys,
and browser logins remain unchanged. To switch an existing connection to a
reference, reconfigure it through `/model add` and select a saved key. Changes do not
alter an already running request or revoke credentials at the provider.

### Adding and selecting models

`/model` is a flat picker of added models. Choose **Add a model...** in it, or
run `/model add`, to open the provider catalog, connection setup, and per-model
settings, including when the saved list is empty. `/model PROVIDER:NAME` selects
any model without the menu: one not yet in the list is added first, without
checking that the provider serves it, so a wrong name fails on the next prompt.
`/model add PROVIDER:NAME` does the same. Adding a model also selects it for the
next prompt. `/model settings [NAME]` edits a model's settings. Tab suggests the
`add` and `settings` subcommands and added models; model names normally start
with a provider, so no real model is called `add` or `settings`. `/model add`
and `/model settings` remain as deprecated spellings.
The list persists across sessions. The currently configured model is retained
when upgrading; `/set model NAME` also saves the model in this list.

### Terminal themes

Theme selection and cancellation do not print status messages. Terminal colour
controls are never replayed as conversation text.

```text
/theme tokyo_night
/set display.theme github_light
```

`/theme` without arguments opens a searchable picker with a sample conversation,
including Markdown, thinking, tool output, code, warnings, errors, and the input
area. Browsing previews colours without applying them; Enter confirms, and Esc
or Ctrl-C cancels. Choices are `default` and `termflow.themes.PALETTES`.
`default` preserves CLAI's existing appearance without changing terminal colours
on startup or exit. `/theme default` restores it after a palette selection.
The picker, `/set`, project settings, and persisted `display.theme` values share
validation. A project override takes precedence again on the next startup.

```python
from rich.console import Console
from pydantic_clai2 import theme

Console().print('Ready for your next prompt.', style=theme.color(theme.INFO))
```

Resolve the `ACCENT`, `INFO`, `WARNING`, `ERROR`, `MUTED`, and `THINKING` roles
through `theme.color(role)` at render time. The constants retain their brand
values; the resolver reads the active palette. `theme.sgr(role)` resolves raw
ANSI surfaces itself. `theme.current()` returns the selected Termflow
`TerminalPalette`, or `None` for the existing default appearance.

Selecting a bundled palette changes terminal foreground, background, and ANSI
slots via Termflow's OSC sequences. CLAI resets them to terminal defaults when
you return to `default` or exit a selected palette, including errors and
cancellation. Redirected output receives no palette-changing sequences.
Unsupported terminals may ignore changes; supported ones may recolour ANSI
scrollback. The early splash retains brand colours. Code uses the terminal
foreground and ANSI syntax colours.
Diff colours stay unchanged in `default`; bundled palettes use Termflow's diff
defaults. Plugins cannot register custom palettes. Theme selection adds no model
requests, hooks, or telemetry.
### Model settings and custom parameters

`/model settings` opens a searchable list of added models. Enter configures a
model without changing the active model. Esc returns from settings to this list;
Esc again closes it. `/model settings PROVIDER:NAME` opens that model directly.
Tab completes added models.
`Ctrl+S` in `/model add` opens the same editor. Edits save immediately and apply
from the model's next request, even in a running turn. `r` resets a field; Esc
or Ctrl-C goes back. Fixed choices
open a picker; numeric fields accept typed values, and empty input resets.

The built-in model catalog and `/set model` completions include
`openai-codex:gpt-6.1-sol`, `openai-codex:gpt-6-sol`, and `openai-codex:gpt-6-luna`.

For `openai-codex` models, open `/model settings openai-codex:gpt-6-astra`
(or your saved Codex model), then **Service Tier / Fast Mode**. Choose
**Fast (priority)** to request fast processing, or **Standard (default)** to
turn it off. [Codex fast mode](https://developers.openai.com/codex/speed)
uses more ChatGPT credits and depends on model and account availability. It does
not lower reasoning effort. Reset restores the existing model default; it does
not enable fast mode. The stored values remain `service_tier=priority` and
`service_tier=default`, so older CLAI versions can read them. A custom
`service_tier` body parameter still takes precedence.

While the active model starts with `openai-codex:`, `/fast` toggles between
priority and standard processing. `/fast on` and `/fast off` select explicitly.
It saves the active model's service tier for subsequent prompts and sessions,
without changing reasoning effort or other preferences. It is absent from help
and Tab completion on other models, and typing it there reports an unknown command.
If a custom `service_tier` parameter is set, `/fast` asks you to remove it first
with `/model settings` rather than saving an ineffective change.

Model preferences are shared across checkouts. Reading saved preferences ignores
unknown fields, so newer settings do not break an older reader with this
compatibility fix. Editing or resetting a known field preserves unknown fields
in the store. New edits still reject unknown keys and invalid values.
An invalid value in a known field stops that turn with a repair message, not the
shell; use `/model settings` to fix or reset it and try again. CLAI does not silently
run with different settings or delete saved preferences.

Older branches must receive this fix too. The minimum read-side backport is
`ModelSettingsForm.model_validate(values, extra='ignore')` in
`model_settings_from_json`; keep the form itself strict. Also backport preservation
of unknown keys on save and the shell's model-settings validation error handler.

The editor offers OpenAI reasoning effort, Responses reasoning context, mode,
summary, and verbosity, and Claude classic/adaptive thinking and effort.
The editor hides generic request fields such as timeouts and penalties.
Reasoning GPT models do not show sampling controls. Previously saved overrides
remain visible so they can be reset. Choices depend on the model and API: Chat Completions does not get Responses
controls. OpenRouter and vLLM GPT routes expose Chat Completions reasoning effort
and service tier, not Responses-only controls. `all_turns` appears only on compatible models, and adaptive Claude
models do not get a token budget. Classic thinking budgets must be at least
1024 and below an explicit `max_tokens`. Without one, Pydantic AI
leaves room for the answer beyond the budget. Other unset fields
use the provider default.
Explicit native thinking settings take precedence over generic `thinking`.
GPT-6 and GPT-5.6 families, including provider-qualified and namespaced names,
default to `thinking=true`, `service_tier=default`, reasoning effort `medium`,
context `all_turns`, mode `standard`, summary `detailed`, and verbosity `low`.
Explicit per-model values win; reset restores the family default without saving
it as an override. Other models keep their existing defaults. Family defaults
follow the model each request uses and sit beneath the settings of
capabilities, a plugin's or your agent's, so a capability can change them; an
agent's own `model_settings` stay beneath the defaults. The values you saved for
the session's model (the one you chose, or CLAI's default) are passed to each run
and win over everything else on requests to that model; a request on a model a
capability selected instead gets only that model's family defaults. Provider-specific
fields are consumed only by APIs that support them; this does not add Responses
controls to Chat Completions or other protocols.

Code Puppy runtime parity is not complete. In particular, its progress-aware
main/sub-agent streaming retries require core recovery support before CLAI can
expose working retry controls. See [the parity audit](MODEL_SETTINGS_AUDIT.md).

Pydantic AI owns adaptive-thinking translation and preserved-thinking replay,
including Fable 5.1's recovery when a changed conversation prefix invalidates a
thinking block. CLAI does not strip thinking or implement a second recovery loop.
See [core's thinking block binding documentation](https://pydantic.dev/docs/ai/models/anthropic/#thinking-block-binding).

For native Anthropic models, **Preserved Thinking** controls
`thinking.block_binding.prefix_mismatch_behavior`: `error` rejects a mismatched
prefix; `drop_block` continues without the mismatched reasoning block. It appears
on adaptive-capable Claude models. Fable 5.1 also exposes **Thinking Display**:
`updates` or `summarized`. Setting either control without a thinking mode selects
adaptive thinking; neither can be combined with disabled thinking. Resetting the
mode clears its budget, display, and binding overrides. **Interleaved Thinking**
adds the beta header on classic Claude 4 models. CLAI adds the display beta when
requesting updates; core adds the block-binding beta. These native controls do
not add Anthropic protocol support to third-party Chat Completions endpoints.

GLM-4.5 and newer expose **Thinking (GLM)** and **Clear Thinking (GLM)**;
GLM-5.2 and newer also expose **Reasoning Effort (GLM)**. Clear Thinking set to
true clears earlier reasoning; false preserves it. These controls send GLM's
native `thinking.type`, `thinking.clear_thinking`, and `reasoning_effort` body
fields. Only explicit overrides are sent. Disabled thinking cannot be combined
with an effort override. A proxy that needs `chat_template_kwargs` instead of
this native shape still needs custom parameters. Custom parameters win over the
generated body on conflict.

Open `custom_params` for Code Puppy-style **Custom Params**. Enter adds or edits
`key = value`; editing the key renames it. `d` deletes a pair and Esc goes back.
Dotted keys nest in the request body's `extra_body`, for example:

```text
chat_template_kwargs.thinking = medium
reasoning.effort = max
```

Values accept JSON booleans, numbers, null, arrays, and objects, or unquoted
text. Quote numeric-looking strings to keep them strings. Parameters are saved
per model and applied last, overriding built-in request fields on conflict.
An extra-body object replaces the corresponding generated object, rather than
deep-merging it. For example, overriding `reasoning.effort` replaces the generated
`reasoning` object; include custom `reasoning.context` too if you need both.
They deliberately bypass the model compatibility checks: the endpoint must
support what you send. This is also the escape hatch for custom endpoints and
provider options not listed in the form. Reset `custom_params` to remove all
pairs. Do not put credentials here: values are stored as plaintext in SQLite.

### Persisting conversation changes

Use `await self.host.conversation.commit_messages(messages)` for between-turn history
changes. It commits to storage before replacing the live history, and rejects
changes while an operation is running. `replace_messages(...)` remains an
in-memory compatibility API; it does not save by itself. `Transcript` implements
`commit_messages` without disk IO for headless plugin tests.

`host.conversation.step_store` is the configured Harness `StepStore`, or `None`
for an in-memory host. The built-in `persistence` plugin binds
`StepPersistence(capture_frontier=True)` to it. Do not register a second recorder
for the same store and run. Removing the plugin removes step capture on subsequent
turns; conversation-head saving is owned by `Session` and continues independently.

Harness exports `SnapshotSaved` from `pydantic_ai_harness.step_persistence`.
Subscribe with a `Hooks` capability's `hooks.on.event(SnapshotSaved)` to observe committed checkpoints.
It carries `persistence_run_id`, `conversation_id`, `step_index`, and `state`.
This is a notification, not the durable source of truth or permission to replay a
tool. A durable replay may notify again. An observer failure cannot roll back the
already committed snapshot. No new CLAI lifecycle hooks are introduced.

Session naming is a shell-owned background service (`runtime/session_naming.py`).
It never writes into the agent transcript or loads plugin code. `/resume` does
not fire plugin load/unload hooks or restore previous plugin approvals. Cross-project
resume keeps the current working directory and the saved conversation's original
project grouping. The
project/session browser is a dedicated Termflow widget: unlike a single-pane
`MenuBuilder`, it has two independently navigable panes and two-line cards. Its
pure frame and scripted-key tests follow the same headless menu conventions.
The selected project stays highlighted while browsing sessions. The focused pane
is labeled **SELECT PROJECT** or **SELECT SESSION**, with matching key hints.

The browser groups existing Git worktrees by repository and labels session cards
with the current branch or detached worktree name. This is display metadata only;
saved workspace paths are unchanged. Selecting a session resumes immediately in
the current directory, without confirmation. See
[Saved sessions](README.md#saved-sessions-and-resume) for fallback behavior.

The resume transcript preview displays at most 24,000 characters of the newest-first
text, with a truncation notice for longer histories. Search is Unicode
case-insensitive and includes text instructions in multimodal prompts.

## Image input

Clipboard and image-path paste are part of the shell's prompt editor, not a plugin
API. Ctrl-V or Alt-V attaches clipboard images; bracketed paste of existing image
paths attaches local files. See [Pasting images](README.md#pasting-images) for
platform requirements and limits.

`turn_start.text` and `turn_end.text` contain the text caption with attachment
markers removed, possibly an empty string for an image-only turn. A `turn_start`
handler may rewrite the caption or cancel the entire turn. Rewriting the text
does not remove the images. Core hooks receive the native multimodal request with
`BinaryContent` image parts. Plugins that inspect or transform image content
should use core hooks rather than parsing terminal markers. There are no new
host lifecycle hooks. Images are persisted with the conversation, including the
accepted request when a turn fails or is cancelled.

## Headless CLI runs

`clai2 -p "PROMPT" [-m PROVIDER:NAME]` runs one saved turn without an editor.
Session and turn hooks still run; stream renderers do not. Host console output
is suppressed, and stdout contains only the final answer. Plugin load failures
abort the run. The `ask_user` plugin is skipped even if saved settings enable or
replace it; this does not change those settings. `host.full_screen()` raises in
headless mode. Plugins must not bypass the host by reading terminal input or
printing directly to stdout. `--resume SESSION-ID` restores history without a
browser or tool replay.

## The stock agent from code

`open_stock_agent` loads only the `coder`, `repo_context`, and `compaction` built-ins,
with `plugin_settings` merged over their stock settings. Saved, drop-in, and project
plugins never load. Each plugin gets `session_start` when the context opens and
`session_end` when it closes, with reason `error` if the block raised. No turn hooks
fire, no `/commands` run, and nothing renders: the host runs the agent. See
[The stock agent in your own code](README.md#the-stock-agent-in-your-own-code).
