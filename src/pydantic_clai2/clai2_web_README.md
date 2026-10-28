# CLAI web UI README content to fold into the README later

<!-- From #10262 -->
## Web UI

Run `/web` in a session to open it in your browser: a local web app with a sessions sidebar,
streaming Markdown answers, collapsible tool calls (shell output, diffs, searches), thinking
blocks, live token counts while a turn streams, a Monty panel with code mode's `run_code` cells, and a composer with `/` command completion, a
model picker, pasted or dropped images, and a stop button. Scrolling up while output streams keeps your place; **Jump to latest** follows
it again. The terminal and the browser drive the same session: a prompt typed in either runs once, and both
show it. The page follows `/theme`, and the sidebar's footer picks a theme too. `/web stop`
ends the web UI; `/web status` shows its address.

To use only the browser, start CLAI with `--web` (also `uvx pydantic-clai2 --web`). It prints
the address, opens it, and logs the session it started with to the terminal until Ctrl-C, which
ends every session. There, each session you open is live in its own directory, with that
directory's `AGENTS.md`, skills, `.clai/settings.json`, and MCP files, as if CLAI had started
there; CLAI's own working directory never changes. **New session** asks where the session runs: a
project or worktree from the sidebar or any other worktree Git lists for its repository, by default
the current one, or any directory by its absolute path, which it completes as you type, listing each
repository's worktrees. The header shows where the session works.
`/workspace DIRECTORY` or `/resume ID` typed in a session moves that session instead, never while a
turn, a background `/fork`, or a delegated task runs in it. `clai2 --web --resume ID` starts in that
session's directory. A session left idle for an hour, with nothing running or queued and no browser
watching it (a page watches the last five sessions it showed), closes, except the one CLAI started
with; it stays saved, and clicking it resumes it. The launch directory's `.env` stays loaded.

The sidebar lists the sessions `/resume` shows, grouped by project (the Git repository, named
after its main checkout's folder), then by worktree (each checkout, labelled with its branch), with
the current ones first and search across all of them. A folder outside Git is a project of its own.
A session remembers its checkout when it is saved, so one from a worktree you have since removed
stays under its project, marked removed, in one **Removed worktrees** group after the live ones,
closed until you open it; an older session from a removed folder is placed by where it was, or
listed under Removed folders. A worktree's `…` menu archives its sessions at once, except the open
and running ones, and not during a search. Click a session to switch to it, even while other
sessions run turns: each session you open stays live and keeps working while you look at another.
A dot by a live session shows that it runs a turn, pulsing; runs one with input queued behind it,
ringed; or waits for your answer to a question, amber. Use a session's menu to rename, fork,
archive, or delete it. Archiving hides a session from the sidebar and `/resume` without deleting
anything; the archive button beside the search lists archived sessions too, muted, and searches
them, each with **Unarchive** in its menu. The shown session and running ones cannot be archived,
and resuming an archived one, or a new turn in it, unarchives it. A session open in CLAI cannot be
deleted. A session whose directory is gone, such as a removed worktree, opens read-only; so does,
with `/web` from a terminal, a session saved in another directory, since the terminal works where it
started. Keyboard: Enter sends, Shift+Enter adds a line, Esc interrupts the shown session's turn and
no other, Cmd/Ctrl+K searches sessions, Cmd/Ctrl+Shift+O starts a new one, Cmd/Ctrl+B hides the
sidebar. As in the terminal, Up on the composer's first line recalls the session's sent prompts,
newest first, and Down on its last line steps back toward your unsent draft.

While a turn runs, the composer works as the terminal's prompt does. Enter queues your message for
after the turn. Alt+Enter (Option+Enter on a Mac), or the bolt button beside Send, steers the running
turn instead: its model reads the message at its next request, and the transcript shows it then.
Commands are never steered, a message with images attached is queued as Enter queues it, and a
message typed while no turn runs is just sent. Alt+Enter with nothing typed steers the oldest queued
message. Everything waiting is listed above the composer,
whether you queued it here, in another browser, or in the terminal: edit a message in place (Enter
saves, Esc cancels), steer it now, or remove it before it runs. A message that runs while you edit it
is queued again as a new follow-up when you save, as in the terminal.

The browser can make the agent run commands, so the server listens on 127.0.0.1 only and
every request must carry a token that is new each launch, though kept across `/update`; the printed link contains it. Keep
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

<!-- From #11590 -->
The **Changes** button in the header (Cmd/Ctrl+Shift+E) opens a review panel of what the session
changed, with the number of files it changed so far on the button. Pick a scope: **Last turn** and
**This session** come from the session's own `write_file` and `edit_file` calls, so they work
outside Git too, and show each file once, as one diff from before the first change to what is on
disk now. A file the session wrote whole counts as created, unless the last commit has it with
content the session did not write, which it is then compared with. These scopes read the
conversation, so after `/compact` they start from the compacted summary. **Uncommitted** is
`git diff HEAD` in the session's directory (untracked files are left out, as there), and **Branch
vs base** compares the files, uncommitted changes included, with where the branch left its base:
`origin/HEAD`, else `main` or `master`. Each file shows its status (A, M, or D) and added and
removed lines; click it to jump to its diff. Files with many changed lines start collapsed. The
panel asks again when a turn ends, and **Refresh** asks at once. It takes the plugin side panels'
place, so opening one closes the other.

<!-- From #10442 -->
The **Settings** window (the gear in the sidebar's footer, or Cmd/Ctrl+,) has a section per area
on its left, each with its own address (`#settings/general`, `#settings/plugins`) to link or
reload. **General** holds the settings `/set` changes, grouped as their keys are (Display, Run,
Sessions, ...): each change is checked, saved, and applied at once, and the terminal prints what it
saved; a change made in the terminal shows here too. `/set` or `/settings` typed in the composer
opens it, and `/theme` or `/spinner` opens a small dialog with that one setting, instead of a menu
in the terminal. A command's menu (`/model`, `/effort`, `/keys`, ...) acts on the session the page
shows, as the command typed there would: picking a model in session B leaves session A's model as it
was. Settings every session shares, such as `/set display.*`, still change them all.

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

<!-- From #11592 -->
A session in a Git repository shows a git chip in its header: the branch, how many files have
uncommitted changes, and how many commits are not pushed (`3 changed, 1 ahead`). Clicking it opens
**Git**, which lists the changed files, each checked, with a commit message the session's model
drafts from them; edit it, untick files to leave out, and **Commit** stages and commits only those,
showing why Git refused, such as a failing hook. **Push** runs `git push -u origin BRANCH`, and asks you
to confirm pushing the default branch. **Create pull request** asks the model for a title and
description you can edit, pushes the branch, opens it with the GitHub CLI (`gh`), and links to it;
it is disabled, saying why, without a GitHub `origin` or a signed-in `gh`, or on the default branch.
Everything runs in the session's own directory, never in one that no longer exists. The terminal
has the same actions: `/commit [MESSAGE]` commits every change (the model drafts the message when
you give none), `/push [confirm]`, and `/pr [TITLE]`.

<!-- From #10461 -->
**System prompt** in the Settings window (`#settings/system-prompt`, or `/system_prompt` in the
composer) is the menu `/system_prompt` shows in the terminal: **Edit…** opens your instructions in
a multi-line dialog (Cmd/Ctrl+Enter saves; an empty text removes them), **Reset** removes them,
**Append…** adds a paragraph, and **View…** shows the full system prompt the model last received,
read-only. **About** (`#settings/about`, or `/update` in the composer) shows the running CLAI's
version, whether a newer one is out, and the `updates.channel` setting; its **Update…** button asks
before it runs `/update now`.
`/usage` and `/cost` show their figures as a card in the chat: the retained history's cost,
tokens, and requests, and for `/usage` a row per turn. Cmd/Ctrl+Shift+P, or `/help` in the
composer, opens the command palette: every command this CLAI offers, plugins' and skills'
included, with what it does and the arguments it takes. Type to filter; Enter runs one as typing it
bare would, and a command that needs arguments (or Tab) goes into the composer to finish. `/tasks`
in the composer moves to the sidebar's Workers tree, and `/resume` to its session search.

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
live; type in its box to steer it (it reads your message at its next model request). Stop and
Background do what `/tasks stop` and `background` do. A finished worker shows what it saved, and
its box then prompts it on its own, as if it were an agent of its own: it continues from its saved
history, its reply streams into its transcript, and the session and its agent never see it.
`POST /api/tasks/{task_id}/resume` still asks the agent to resume it instead, as `/tasks resume`
does. `Explore` and `Plan` workers are one-shot and take no follow-ups. **← Session**, or the session's
row, goes back.

<!-- From #10372 -->
`clai2 --web --port PORT` listens on PORT for that launch (`0` picks a free one). `--token-file PATH`
takes the launch's token from a file only you can read, instead of making one, for a program that
starts CLAI and keeps the token itself: CLAI then listens on 127.0.0.1 only, the printed address
leaves the token out, and no browser opens.

`/update` works from the page too (typed in the composer, it opens Settings › About, whose
**Update…** asks first): the page shows uv's progress, then
"Restarting CLAI…" while the new build starts on the same address, reconnects by itself to the
same session, and says which version it updated to. A failed install, or nothing to install,
says so and nothing restarts. A terminal session with `/web` running comes back with it running.

## Workspaces

The `workspaces` plugin makes workspaces: a checkout of a repository on its own branch, for
sessions of their own. Turn it on with `/plugins enable workspaces`, then:

```text
/workspaces create local-worktree ~/code/pydantic-ai main
```

`create KIND [REPO [REF]]` checks out REPO (a local path, or a URL that CLAI clones once and
fetches afterwards) at REF on a new branch, the kind's `branch_prefix` (`clai-` by default) and
the ID. The sidebar lists the checkout under its repository, grouped by that branch as any other
worktree, so **New session** there starts a session in it, in this same CLAI. `destroy ID` closes
the sessions open in it and removes the checkout, unless the kind's `teardown` is `keep`, keeping
the branch either way. `/workspaces` lists them all, from any CLAI: they are kept in CLAI's state
directory. The kinds of workspace are the plugin's settings (`/plugins configure workspaces`, or
the web UI's plugins panel); `local-worktree`, a Git worktree on this machine, is the one backend so
far. See [the plugin guide](PLUGINS.md#workspaces-checkouts-for-sessions-of-their-own).

## Orchestrator

The web UI has an agent that does that for you: "clone pydantic-ai off main and fix issue #123".
Click **Orchestrator** in the sidebar, right below **New session**, to open its chat. It is a
separate agent with its own conversation, and the only one with these tools: it makes workspaces,
starts a session in each one with a prompt, waits for and reads its turns, steers it with more
prompts, answers its questions, and reports back. Those sessions are ordinary live sessions, listed
in the sidebar under their repository; a tool card's **Open** shows one in the main pane, where you
can follow it or type too. It destroys a workspace only after you picked **Destroy ID** when it
asked, in its chat. It also finds the saved sessions of every project and cleans them up by
archiving them ("archive every session older than 3 days"), which you can undo. No tool deletes a
session, and archiving removes no folder or worktree. It needs `workspaces` on, which the chat
offers to do in one click, and uses the saved default model, the one a new session starts with
(`/set model NAME` changes it). Its conversation is kept across reloads and restarts. The session's
own agent gets none of this. See [the plugin guide](PLUGINS.md#orchestrator-an-agent-that-runs-workspaces).

## Several live sessions

The web UI (`clai2 --web`, or `/web` from a terminal) keeps every session you open live in the one
CLAI process: each runs its own turns in
its own folder, with its own plugins and input queue, while the others run theirs. Over its one
WebSocket the page watches the last five sessions it showed, keeping each one's transcript current,
so switching back to one shows it at once.

### Wire protocol

Sessions are named by their conversation ID, the `id` the sidebar lists.

- **Per-session messages** (`item`, `text_delta`, and the other item messages, `turn_started`, `turn_ended`,
  `status`, `question`, `question_closed`, `notice`, `title`, `task`, `task_items`, `queue`) carry `"session": "<conversation id>"` and go only to the
  connections watching that session. A `snapshot` names its session in `session.id`.
- **Global messages** (`sessions`, `plugins`, `settings`, `theme`, `update`, `orchestrator`, `live`,
  and the settings menus' `dialog`, `dialog_shown`, and `dialog_closed`) go to every connection and
  carry no `session`; every `snapshot` lists the open dialogs.
- `live` lists the live sessions, sent on connect and whenever one opens, closes, switches
  conversation, or changes state:
  `{"type": "live", "sessions": [{"id": "…", "state": "running" | "queued" | "asking" | "idle", "title": "…" | null, "workspace": "/abs/path"}]}`.
  `queued` means input waits while that session's turn runs; `asking` means a question its turn
  asked waits for the user.
- The page sends `{"type": "watch", "sessions": ["<id>", …]}` on the WebSocket to replace the set it
  watches. The server answers with a `snapshot` for each newly watched live session; the default session
  a connection got on connect counts as watched already, so naming it sends no second one. An ID that is not
  live yet stays watched: its `snapshot` comes once it opens. A connection that never sends `watch`
  (an older page, `web/client.py`) watches the default session, as before, and gets its `snapshot`
  when another session becomes the default. A watched session is never closed for being idle.
- When a live session switches conversation (`/new` or `/resume` typed in it), its watchers get a
  `snapshot` for the new ID whose `replaces` names the conversation it held before (every other
  `snapshot` has `"replaces": null`), and `live` changes.
- `queue` lists the input waiting for a session's running turn or command, oldest first, whenever it
  changes, from the page or the terminal: `{"type": "queue", "queue": [{"id": "…", "text": "…", "kind":
  "prompt" | "command" | "shell"}]}`. Only a `prompt` can steer. A `snapshot` carries the same list as
  `queue`. Steering shows as a `user` item once the run's model receives it.

Routes for one live session, `{id}` its conversation ID; each answers 404 when it is not live:

| Route | Does |
|---|---|
| `GET /api/sessions/{id}/state` | its `snapshot` |
| `GET /api/sessions/{id}/images/{image_id}` | an image of a prompt in its transcript, by an ID its `user` item lists in `images` (the SHA-256 of the bytes); 404 for one it does not hold |
| `GET /api/sessions/{id}/changes?scope=turn\|session\|uncommitted\|branch` | the Changes panel's files: `Changes` with each `FileChange`'s path, status (`A`, `M`, `D`), added and removed lines, and unified diff (shapes in `web-ui/src/protocol.ts`); `session` by default; `diffs=false` leaves the diffs out, for a count (the session scopes then leave the line counts at 0 too) |
| `POST /api/sessions/{id}/input` | enter a prompt or `/command`, body `{"text": "…"}`, as `/api/input` |
| `POST /api/sessions/{id}/interrupt` | interrupt its turn only |
| `POST /api/sessions/{id}/steer` | Alt+Enter: send a prompt to its running turn, body `{"text": "…"}`; answers `{"steered": true, "menu": null}`, or, when nothing takes it (no turn runs, or a `/command`), enters it as `/input` does and answers `{"steered": false, "menu": …}` |
| `PUT /api/sessions/{id}/queue/{input_id}` | rewrite a queued input in place, body `{"text": "…"}`; 404 once it ran or was removed |
| `DELETE /api/sessions/{id}/queue/{input_id}` | remove a queued input; 404 once it ran or was removed |
| `POST /api/sessions/{id}/queue/{input_id}/steer` | send a queued prompt to the running turn, taking it out of the queue; answers `{"steered": false}` and leaves it queued when nothing takes it |
| `GET /api/sessions/{id}/complete?text=…` | Tab completions |
| `GET /api/sessions/{id}/files?q=…` | the composer's `@` picker: the 50 files and folders in the session's folder that best match `q` (Git's tracked and unignored files in a checkout), each a completion of `@q` |
| `POST /api/sessions/{id}/model` | select a model, body `{"name": "…"}` |
| `POST /api/sessions/{id}/switch` | turn a `/set` switch on or off, body `{"key": "…", "on": true}`, as `/api/switch` |
| `POST /api/sessions/{id}/questions/{request_id}` | answer its question |
| `POST /api/sessions/{id}/dialogs/{dialog_id}` | answer a settings menu's dialog, as `/api/dialogs/…` |
| `GET /api/sessions/{id}/commands/{name}/fields[?menu=KEY…]` | the `MenuInfo` of the menu bare `/name` opens in this session, built from its own commands, as `/api/commands/{name}/fields` |
| `POST /api/sessions/{id}/commands/{name}/fields/{key}[?menu=KEY…]` | set a row, body `{"value": "…"}` (`null` or empty resets it); answers `MenuEdited` |
| `POST /api/sessions/{id}/commands/{name}/fields/{key}/action[?menu=KEY…]` | run a row's action, asking through the page's dialogs; answers `MenuEdited` |

`POST /api/sessions/{id}/open` makes a saved session live (resumed in a session of its own, or the
one already live) and answers `{"id": "<conversation id>"}`. `POST /api/sessions/new`, with the same
optional body as before (`{"workspace": "/abs/path"}`), starts a new live session and answers
the same `{"id": "<conversation id>"}`. Opening a session whose folder is gone answers 400. Where CLAI
opens no other sessions (a host built outside CLAI's shell), both switch the one session instead,
with `/resume` or `/new`, and answer its ID. The saved-session routes (`GET /api/sessions`, rename, fork, archive, unarchive,
delete) and `/api/tasks/{task_id}/…` are unchanged. `POST /api/sessions/archive` with
`{"ids": [...]}` archives several sessions at once, and answers 400 naming any it refused once it
has archived the rest.

A command menu (`/model`, `/effort`, `/set`, `/keys`, …) reads and changes the session it is built from:
choosing a model in session B's `/model` menu switches B alone, and B's `/effort` shows B's model. Settings
CLAI shares, such as `display.*`, still reach every session, as typing `/set` does. Plugin menus
(`/api/plugins/…`) and dialogs belong to the page.

The unprefixed routes (`/api/state`, `/api/images/…`, `/api/changes`, `/api/input`, `/api/steer`, `/api/queue/…`, `/api/interrupt`,
`/api/complete`, `/api/files`, `/api/model`, `/api/switch`, `/api/dialogs/…`, `/api/commands/…`) and a `/ws` without `watch` keep working on the
default session: the one this page last opened or started while it stays live, else the most recently opened.
`/api/questions/…` answers any live session's question by its ID, as before. `web/client.py` and the
orchestrators of other CLAI versions use them.
