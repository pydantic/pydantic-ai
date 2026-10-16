# CLAI web UI README content to fold into the README later

<!-- From #10262 -->
## Web UI

Run `/web` in a session to open it in your browser: a local web app with a sessions sidebar,
streaming Markdown answers, collapsible tool calls (shell output, diffs, searches), thinking
blocks, live token counts while a turn streams, a Monty panel with code mode's `run_code` cells, and a composer with `/` command completion, a
model picker, and a stop button. The
terminal and the browser drive the same session: a prompt typed in either runs once, and both
show it. The page follows `/theme`, and the sidebar's footer picks a theme too. `/web stop`
ends the web UI; `/web status` shows its address.

To use only the browser, start CLAI with `--web` (also `uvx pydantic-clai2 --web`). It prints
the address, opens it, and logs the conversation to the terminal until Ctrl-C. There, each
session runs in its own directory. Opening a session saved elsewhere moves CLAI there before its
next turn, with that directory's `AGENTS.md`, skills, `.clai/settings.json`, and MCP files, as if
CLAI had started there. **New session** asks where the session runs: a project or worktree from
the sidebar, by default the current one, or any directory by its absolute path
(`/workspace DIRECTORY` in the composer does the same). The header shows where the session works.
`clai2 --web --resume ID` starts in that session's directory. Nothing moves while a turn, a
background `/fork`, or a delegated task runs. The launch directory's `.env` stays loaded.

The sidebar lists the sessions `/resume` shows, grouped by project (the Git repository, named
after its main checkout's folder), then by worktree (each checkout, labelled with its branch), with
the current ones first and search across all of them. A folder outside Git is a project of its own.
A session remembers its checkout when it is saved, so one from a worktree you have since removed
stays under its project, marked removed; an older session from a removed folder is placed by
where it was, or listed under Removed folders. Click one to switch to it, or use its menu to
rename, fork, or delete it. A session whose directory is gone, such as a removed worktree, opens
read-only; so does, with `/web` from a terminal, a session saved in another directory, since the
terminal works where it started. Keyboard: Enter sends, Shift+Enter adds a line, Esc interrupts,
Cmd/Ctrl+K searches sessions, Cmd/Ctrl+Shift+O starts a new one, Cmd/Ctrl+B hides the sidebar.

The browser can make the agent run commands, so the server listens on 127.0.0.1 only and
every request must carry a token that is new each launch; the printed link contains it. Keep
the link private. The `web` plugin starts disabled; `/web` works once it is enabled with
`/plugins enable web`, and `--web` loads it for that launch only. See
[the plugin guide](PLUGINS.md#web-this-session-in-your-browser) for settings and limits.

<!-- From #10282 -->
`/monty CODE` runs your own cell in Monty, that sandbox, between turns, with the same
tool functions the model's cells get: `/monty x = 41`, then `/monty print(x)`. It prints
what the cell printed and returned, keeps the state with the session, and stays out
of the conversation; your next prompt tells the model what you ran, since the
variables are in its sandbox now. A cell that fails is reported, not retried. In the
web UI, the Monty panel's input runs `/monty` too.

<!-- From #10442 -->
The **Settings** window (the gear in the sidebar's footer, or Cmd/Ctrl+,) has a section per area
on its left, each with its own address (`#settings/general`, `#settings/plugins`) to link or
reload. **General** holds the settings `/set` changes, grouped as their keys are (Display, Run,
Sessions, ...): each change is checked, saved, and applied at once, and the terminal prints what it
saved; a change made in the terminal shows here too. `/set` or `/settings` typed in the composer
opens it, and `/theme` or `/spinner` opens a small dialog with that one setting, instead of a menu
in the terminal.

<!-- From #10456 -->
**Models** (`#settings/models`, or `/model` typed in the composer) picks the model for
the next prompt, exactly as `/model NAME` does: the session switches, the terminal's status row and
the composer show it, and it is saved as the default. Each saved model and fallback chain has a row
that opens its settings beneath it, the rows `/model settings NAME` edits, so effort and Codex fast
mode changed here are what `/effort` and `/fast` read; the current model's opens first. **Add a
model** asks for a provider and one of its catalog models, then the account when the provider has
several, as `/model add` does; GitHub Copilot, OpenRouter, and vLLM, which sign in or ask for an
address first, are added from the terminal. **Remove a model** deletes one after asking, keeping the
current model and the saved default. Creating, editing, and renaming fallback chains stays in the
terminal's `/model chains`. `/effort` typed in the composer opens a dialog with the active model's
settings, effort among them, and `/fast` still toggles fast mode.

<!-- From #10457 -->
**Keys & accounts** (`#settings/keys`, or `/keys`, `/login`, or `/accounts` in the composer) lists
your saved API keys by name, with what uses each, and your accounts in the order `PROVIDER@*` tries
them. Each row's button asks what to do as the terminal's menus do: add, replace (masked), rename,
or delete a key; sign in again, rename, try earlier or later, or sign out of an account; add one.
A key's value is typed in the page but never sent back to it. Signing in to ChatGPT/Codex or GitHub
Copilot shows its link or code on a waiting screen you can cancel; the ChatGPT/Codex sign-in
finishes when the browser comes back to the machine CLAI runs on. An API-key account picks or saves
a `/keys` entry. Connections (`openrouter`, `vllm`) and plugins' sign-ins ask in the terminal, so
the page says to run `/login NAME` there.

<!-- From #10459 -->
**MCP servers** (`#settings/mcp`, or `/mcp` typed in the composer) lists your MCP servers with
their state: running, ready, off, an error, or needing a sign-in.
**Add server** opens `/mcp install`'s form as dialogs, its JSON in a multi-line field, and a
server's **Manage** starts, stops, or restarts it, shows its tools and recent logs, signs in or out,
edits, or removes it, as `/mcp`'s subcommands do; the terminal prints each result, and a change
made there shows here too. A repository's MCP files show as a banner whose **Review** shows their
contents before you trust them.

<!-- From #10289 -->
The Plugins panel, the Settings window's **Plugins** section (the puzzle button in the sidebar's
footer, or `/plugins` in the composer), turns plugins on and off and edits their settings, saved and
applied at once, as `/plugins` does in the terminal. A plugin that declares its settings menu shows it there, sign-ins
and `/keys` entries included, as dialogs in the page; any other plugin's own menu opens in the
terminal.

<!-- From #10319 -->

```diff
@@ -2384 +2384,2 @@
-default, and opens its settings menu (`/plugins configure linear` reopens it). The
+default, and opens its settings menu (`/plugins configure linear` reopens it, and
+the web UI's plugins panel shows it too). The
```

<!-- From #10328 -->

```diff
@@ -24 +24,2 @@
-`/logfire login`). Reopen it with `/plugins configure logfire_mcp`; see
+`/logfire login`). Reopen it with `/plugins configure logfire_mcp`, or in the web UI's
+plugins panel; see
```

<!-- From #10330 -->

```diff
@@ -17,2 +17,3 @@
-token kept in `/keys`. `/google_workspace` opens its settings menu: the `/keys`
-entry to use (`GOOGLE_ACCESS_TOKEN` by default), products, and read-only tools; see
+token kept in `/keys`. `/google_workspace` opens its settings menu (the web UI's
+plugins panel shows it too): the `/keys` entry to use (`GOOGLE_ACCESS_TOKEN` by
+default), products, and read-only tools; see
```

<!-- From #10333 -->

```diff
@@ -2515,2 +2515,2 @@
-its settings menu. Each option below is a row there; each edit saves at once and
-applies from the next run. The last command replaces the built-in configuration
+its settings menu; the web UI's plugins panel shows the same menu. Each option below
+is a row there; each edit saves at once and applies from the next run. The last command replaces the built-in configuration
```

```diff
@@ -2551,2 +2551,3 @@
-Choose **Logfire project** in the settings menu (`/plugins configure observability`, or
-`c` on `observability` in `/plugins`):
+Choose **Logfire project** in the settings menu (`/plugins configure observability`,
+`c` on `observability` in `/plugins`, or **Change…** on the row in the web UI's plugins
+panel, where each step below is a dialog in the page):
```

```diff
@@ -2559,2 +2560,4 @@
-2. Sign in, or sign up, in the browser. CLAI prints the link too, so it works over SSH.
-3. Pick one of the projects you can write to (`/` to search, Enter to use it).
+2. Sign in, or sign up, in the browser. A waiting screen shows the link too, so it
+   works over SSH; Esc there cancels the sign-in and changes nothing.
+3. Pick one of the projects you can write to (typing filters a list of more than
+   eight; Enter uses it).
```

<!-- From #10334 -->

```diff
@@ -1789 +1789,3 @@
-**Agent folders**. The list saves each change immediately; no JSON editing is
+**Agent folders**, or with its **Change…** button in the
+[web UI](#web-ui)'s plugins panel, where each step below opens as a dialog in
+the page. The list saves each change immediately; no JSON editing is
```

```diff
@@ -1792 +1794 @@
-- **Add directory path** (`a`) accepts one local directory. An unprefixed
+- **Add directory path** accepts one local directory. An unprefixed
```

```diff
@@ -1800,8 +1802,10 @@
-- **Browse local directories** (`b`) includes hidden folders. Type to filter,
-  Enter to open a folder, Left to go to its parent, and **Use this directory**
-  to select it. The choice is saved as an absolute path, used in every project;
-  type a relative path with **Add directory path** to follow each project
-  instead. Ctrl+L opens the full current path for viewing or navigation.
-- Enter on an entry opens its actions. `e` edits its current value; `d` removes
-  it from the search without deleting files. Esc cancels an edit or returns to
-  the previous menu; **Save & close** returns with changes already saved.
+- **Browse local directories** includes hidden folders. Pick a folder to open
+  it, **..** to go to its parent, and **Use this directory** to select it; in
+  the terminal, type to filter a long list. The choice is saved as an absolute
+  path, used in every project; type a relative path with **Add directory path**
+  to follow each project instead. **Go to path** shows the full current path
+  for viewing or navigation.
+- Choosing an entry opens its actions: edit its current value, replace it by
+  browsing, or remove it from the search without deleting files. Esc cancels an
+  edit or returns to the previous menu, and an empty edit changes nothing;
+  **Save & close** returns with changes already saved.
```

<!-- From #10357 -->
Plugins can add a panel on the right, toggled by its button at the top right or Cmd/Ctrl+Shift+L,
resizable by its left edge, and remembered by the browser. The `observability` plugin's shows
Logfire's live view of the current conversation, opened at the latest turn. Logfire does not let
other sites show its pages inside theirs today, so the panel opens it in a window beside CLAI's
instead, and points that window at each new turn. Without a Logfire project the panel says how to
set one up.

<!-- From #10359 -->
When the agent delegates to sub-agents, a **Workers** tree appears under the current session:
one row per worker, under the worker that started it, with its agent, model, elapsed time, and
last action. Click one, or the link on the tool call that started it, to watch its transcript
live; type in its box to steer it (it reads your message at its next model request). Stop,
Background, and Resume do what `/tasks stop`, `background`, and `resume` do. A finished worker
shows what it saved; **← Session**, or the session's row, goes back.

<!-- From #10372 -->
`clai2 --web --port PORT` listens on PORT for that launch (`0` picks a free one). `--token-file PATH`
takes the launch's token from a file only you can read, instead of making one, for a program that
starts CLAI and keeps the token itself: CLAI then listens on 127.0.0.1 only, the printed address
leaves the token out, and no browser opens.

## Workspaces

The `workspaces` plugin makes workspaces: a checkout of a repository on its own branch, with its
own CLAI. Turn it on with `/plugins enable workspaces`, then:

```text
/workspaces create local-worktree ~/code/pydantic-ai main
/workspaces open pydantic-ai-3f2a9c
```

`create KIND [REPO [REF]]` checks out REPO (a local path, or a URL that CLAI clones once and
fetches afterwards) at REF on a new branch, the kind's `branch_prefix` (`clai-` by default) and
the ID. `open ID` starts that workspace's own `clai2 --web` in the checkout and opens it in your
browser; `stop ID` stops it, and `destroy ID` stops it and removes the checkout, unless the kind's
`teardown` is `keep`, keeping the branch either way. `/workspaces` lists them all, from any
CLAI: they are kept in CLAI's state directory, and a workspace's CLAI keeps running when the CLAI
that started it quits. The kinds of workspace are the plugin's settings (`/plugins configure
workspaces`, or the web UI's plugins panel); `local-worktree`, a Git worktree on this machine,
is the one backend so far. See [the plugin guide](PLUGINS.md#workspaces-checkouts-with-their-own-clai).

<!-- From #10374 -->
With `workspaces` on, the web UI's sidebar lists them in a **Workspaces** section, each with its
CLAI's status and an **Open** button that starts it if needed and opens its page in a new tab.

## Orchestrator

The web UI has an agent that does that for you: "clone pydantic-ai off main and fix issue #123".
Click **Orchestrator** in the sidebar, right below **New session**, to open its chat. It is a
separate agent with its own conversation, and the only one with these tools: it makes workspaces,
starts each one's CLAI with a prompt, waits for and reads its turns, steers it with more prompts,
answers its questions, and reports back. It destroys a workspace only after you picked **Destroy
ID** when it asked, in its chat. It needs `workspaces` on, which the chat offers to do in one
click, and uses the session's model. Its conversation is kept across reloads and restarts. The
session's own agent gets none of this. See
[the plugin guide](PLUGINS.md#orchestrator-an-agent-that-runs-workspaces).
