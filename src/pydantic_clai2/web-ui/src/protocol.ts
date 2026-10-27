// Mirrors pydantic_clai2/builtin_plugins/web/protocol.py.

/** The wire version the page names in the WebSocket URL; a page without it gets the tags from before it. */
export const PROTOCOL = 2

export type ItemKind = 'user' | 'assistant' | 'thinking' | 'tool'
export type ToolState = 'running' | 'done' | 'error' | 'interrupted'

export interface Diff {
  path: string
  diff: string
  truncated: boolean
}

export interface SubCall {
  id: string
  name: string
  args: string
  state: ToolState
  text: string
  output: string
}

export interface Item {
  id: string
  kind: ItemKind
  text: string
  name: string
  args: string
  state: ToolState
  output: string
  exit_code: number | null
  diffs: Diff[]
  /** Tools a `run_code` cell called from its sandbox; live runs only. */
  calls: SubCall[]
  /** A cell the user ran with `/monty`, rather than one the model wrote. */
  by_user: boolean
  /** The IDs of a prompt's images, which the session's `images/ID` route serves. */
  images: string[]
}

export interface Option {
  label: string
  description: string | null
}

export interface QuestionInfo {
  header: string
  question: string
  options: Option[]
  multi_select: boolean
}

/** One value a settings row or a dialog offers. */
export interface Choice {
  value: string
  label: string
  description: string
}

/** What a plugin's settings menu asks; which fields apply depends on `kind`. */
export interface Dialog {
  id: string
  kind: 'choose' | 'text' | 'confirm' | 'wait' | 'view'
  title: string
  description: string
  options: Choice[]
  current: string | null
  label: string
  placeholder: string
  initial: string
  secret: boolean
  /** `text`: take text with line breaks, in a text area. */
  multiline: boolean
  yes: string
  no: string
  /** `wait`: the latest text its task showed, such as a sign-in link. */
  shown: string
  /** `view`: the read-only text to show; any answer closes it. */
  text: string
}

export interface Questions {
  id: string
  questions: QuestionInfo[]
  asker: string | null
}

export interface SessionInfo {
  id: string
  title: string | null
  workspace: string
}

export interface StatusInfo {
  model: string
  activity: string
  context_tokens: number | null
  context_window: number | null
  cost: string | null
  cost_is_lower_bound: boolean
  /** The turn's output so far, estimated from streamed characters. */
  streamed_tokens: number
  /** The last turn's reported output tokens, once it ends. */
  output_tokens: number | null
  /** What the running turn's model requests have reported so far. */
  run: Usage | null
  /** The `run.code_mode` setting; `null` without CLAI's settings. */
  code_mode: boolean | null
  /** The `run.speculative_code_mode` setting, which turns code mode on for its runs too. */
  speculation: boolean | null
}

export type Switch = 'run.code_mode' | 'run.speculative_code_mode'

export interface Usage {
  input_tokens: number
  output_tokens: number
  requests: number
}

/** Tokens, requests, and cost of one turn, or of all of them, as a `/usage` row shows them. */
export interface TurnUsage {
  requests: number
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  cache_write_tokens: number
  /** In dollars, as `/cost` formats it; `null` when nothing could be priced. */
  cost: string | null
}

/** A background `/fork`, as a `/forks` row lists it. */
export interface ForkEntry {
  id: number
  model: string
  prompt: string
  status: 'running' | 'done' | 'failed' | 'cancelled'
  /** Seconds it has run so far, or ran in all once it ended. */
  elapsed: number
  /** The saved conversation `/resume` continues it in, once it has saved one. */
  session_id: string | null
}

/** What `/usage` and `/cost` report about the retained history. */
export interface UsageReport {
  turns: TurnUsage[]
  total: TurnUsage
  /** The models without price data, whose responses no cost counts. */
  unpriced: string[]
  unmeasured_requests: number
}

export interface Palette {
  bg: string
  fg: string
  ansi: string[]
}

/** `/update` installing a newer CLAI, until the install ends or CLAI is back. */
export interface UpdateInfo {
  /** The version, or the short commit of a `main` build. */
  target: string
  /** The last line uv printed. */
  line: string
  /** Installed: CLAI is restarting as the new build. */
  restarting: boolean
}

export interface ThemeInfo {
  /** `default` is Pydantic's own colours, in the browser's light or dark mode. */
  name: string
  names: string[]
  palette: Palette | null
}

export type DelegationOutcome = 'ok' | 'timeout' | 'budget' | 'failed' | 'contained' | 'cancelled' | 'error'

/** A delegated task, as the sidebar's Workers tree shows it. */
export interface TaskEntry {
  id: string
  parent_id: string | null
  agent: string
  model: string | null
  prompt: string
  status: 'running' | 'finished'
  outcome: DelegationOutcome | null
  started_at: string
  finished_at: string | null
  background: boolean
  /** Whether the Background button can let the run that waits for it go on. */
  backgroundable: boolean
  resumable: boolean
  /** Its last action, such as `thinking` or `running: shell`. */
  activity: string
  usage: Usage
}

/** A delegated task with the history it saved. */
export interface TaskTranscript {
  task: TaskEntry
  items: Item[]
}

/** A message that changes one transcript's items, the session's or a task's. */
export type ItemMessage =
  | { type: 'item'; item: Item }
  | { type: 'text_delta'; id: string; text: string }
  | { type: 'tool_output'; id: string; text: string }
  | { type: 'args_delta'; id: string; text: string }
  | { type: 'tool_args'; id: string; args: string }
  | { type: 'sub_call'; id: string; call: SubCall }
  | { type: 'sub_call_output'; id: string; call_id: string; text: string }
  | { type: 'sub_call_done'; id: string; call_id: string; text: string; state: ToolState }
  | { type: 'tool_done'; id: string; text: string; state: ToolState }
  | { type: 'shell_done'; id: string; exit_code: number | null }
  | { type: 'diff'; id: string; diff: Diff }

/** A transcript's item, turn, and question messages, which the session and the orchestrator chat both get. */
export type ChatMessage =
  | ItemMessage
  | { type: 'turn_started'; item: Item | null }
  | { type: 'turn_ended'; outcome: 'completed' | 'failed' | 'cancelled'; error: string | null; usage: Usage | null }
  | { type: 'question'; questions: Questions }
  | { type: 'question_closed'; id: string }

/** The orchestrator chat: sent on connect, as a turn starts, and when it starts over. */
export interface OrchestratorState {
  type: 'orchestrator_state'
  /** Whether the `workspaces` plugin is on; the orchestrator runs only then. */
  enabled: boolean
  running: boolean
  /** The model its next turn runs on: the saved default a new session starts with, not a session's own choice. */
  model: string | null
  items: Item[]
  questions: Questions[]
}

export type OrchestratorEvent = ChatMessage | OrchestratorState

/** `queued`: input waits while the session's turn runs; `asking`: a question of its turn waits for the user. */
export type LiveState = 'running' | 'queued' | 'asking' | 'idle'

/** A session open in this CLAI, named by its conversation ID. */
export interface LiveEntry {
  id: string
  state: LiveState
  title: string | null
  workspace: string
}

/** A prompt can steer the running turn; a `/command` or `!` shell line, queued from the terminal, cannot. */
export type InputKind = 'prompt' | 'command' | 'shell'

/** An input waiting for the session's running turn or command to end, editable until it runs. */
export interface QueuedItem {
  id: string
  text: string
  kind: InputKind
}

/** A message about one live session, sent only to the connections watching it. */
export type SessionMessage = (
  | ChatMessage
  | { type: 'queue'; queue: QueuedItem[] }
  | { type: 'status'; status: StatusInfo }
  | {
      type: 'notice'
      text: string
      level: 'info' | 'error'
      command: string | null
      usage: UsageReport | null
      forks: ForkEntry[] | null
      fork: ForkEntry | null
    }
  | { type: 'title'; title: string | null }
  | { type: 'task'; task: TaskEntry }
  | { type: 'task_items'; id: string; messages: ItemMessage[] }
) & {
  /**
   * The session's conversation ID, as the sidebar and `live` name it. The server always sends it; it is optional here
   * only because `reduce` also applies the orchestrator's chat messages, which have none.
   */
  session?: string
}

/** Sent over the WebSocket: the live sessions this page shows, replacing the ones it watched. */
export interface WatchMessage {
  type: 'watch'
  sessions: string[]
}

/** What `POST /api/sessions/{id}/open` and `POST /api/sessions/new` answer: the live session's conversation ID. */
export interface Opened {
  id: string
}

export type ServerMessage =
  | {
      type: 'snapshot'
      session: SessionInfo
      items: Item[]
      running: boolean
      status: StatusInfo
      questions: Questions[]
      dialogs: Dialog[]
      theme: ThemeInfo
      /** Each session runs in its own directory (`clai2 --web`), so a new one asks where. */
      follows_workspace: boolean
      tasks: TaskEntry[]
      /** Each running task's transcript so far, by task ID. */
      task_items: Record<string, Item[]>
      /** The input waiting for the session's running turn or command, oldest first. */
      queue: QueuedItem[]
      /** The running CLAI's version, or a `main` build's short commit. */
      version: string
      update: UpdateInfo | null
      /** The conversation the session held before, when it switched to this one; its watchers watch this one now. */
      replaces?: string | null
    }
  | SessionMessage
  | { type: 'orchestrator'; message: OrchestratorEvent }
  | { type: 'dialog'; dialog: Dialog }
  | { type: 'dialog_shown'; id: string; text: string }
  | { type: 'dialog_closed'; id: string }
  | { type: 'theme'; theme: ThemeInfo }
  | { type: 'sessions' }
  | { type: 'plugins' }
  | { type: 'settings' }
  | { type: 'update'; update: UpdateInfo | null }
  | { type: 'live'; sessions: LiveEntry[] }

export type SessionState = 'ready' | 'running' | 'busy' | 'completed' | 'failed' | 'cancelled'

export interface SessionEntry {
  id: string
  title: string
  workspace: string
  updated_at: string
  /** `running`: a turn of a session live in this CLAI; `busy`: a turn another CLAI process runs. */
  state: SessionState
  messages: number
  model: string | null
  current: boolean
  /** Hidden from the sidebar unless it lists archived sessions too. */
  archived: boolean
  /** Whether it opens live, as `Transcript.openable` says, so a click opens it without fetching its transcript first. */
  openable: boolean
}

export interface Worktree {
  root: string
  /** Its branch, or its folder's name outside Git or when the branch is unknown. */
  label: string
  current: boolean
  /** The folder is gone, as with a removed worktree. */
  missing: boolean
  sessions: SessionEntry[]
}

export interface Project {
  key: string
  name: string
  current: boolean
  worktrees: Worktree[]
}

export interface Transcript {
  id: string
  title: string
  workspace: string
  items: Item[]
  openable: boolean
  /** CLAI would move to its directory, but that directory is gone. */
  missing: boolean
  /** The process ID of another CLAI running it; opening it then continues in a fork of it. */
  owner_pid: number | null
}

export interface CompletionInfo {
  text: string
  start: number
  display: string
  meta: string | null
}

export interface AnswerBody {
  header: string
  selected: string[]
  custom: string | null
}

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }

export type PluginOrigin = 'built-in' | 'project' | 'drop-in' | 'installed'
export type PluginStatus = 'on' | 'off' | 'idle' | 'failed' | 'included'
export type PluginAction = 'enable' | 'disable' | 'reload'

export interface PluginForm {
  json_schema: Schema
  values: Record<string, JsonValue>
}

export interface PluginRow {
  name: string
  description: string
  origin: PluginOrigin
  source: string
  status: PluginStatus
  included_in: string | null
  error: string | null
  notice: string | null
  provides: string
  /** It has a settings menu. */
  has_menu: boolean
  /** That menu is declared, so this page shows it; otherwise it opens in the terminal. */
  settings_menu: boolean
  /** Whether this CLAI has a terminal for an undeclared menu to open in; `clai2 --web` has none. */
  terminal: boolean
  form: PluginForm | null
  /** The plugin serving this page, which the page must not turn off, reload, or reconfigure. */
  serving: boolean
}

/** A plugin's side panel, as it is now. */
export interface PanelInfo {
  plugin: string
  id: string
  title: string
  icon: 'panel' | 'traces'
  /** The page to show; `null` shows `note` alone, such as how to set the plugin up. */
  url: string | null
  note: string
  /** The page lets this one frame it; otherwise it opens in a window of its own. */
  embed: boolean
}

/** One row of a plugin's declared settings menu, with its current value. */
export interface MenuField {
  key: string
  label: string
  description: string
  /** As the user would type it; never a secret itself. */
  value: string
  default: string
  choices: Choice[]
  /** A value other than `choices` may be typed. */
  allow_custom: boolean
  secret: boolean
  /** Where the value comes from when not from the user. */
  note: string
  /** Choosing it runs an action, with dialogs, instead of editing a value. */
  action: boolean
  /** What the action's button says; when empty, `Change…` for a row with a value or a secret, else `Open…`. */
  button: string
  /** Choosing it opens another menu, which `MenuTarget.path` reaches, instead of editing a value. */
  menu: boolean
}

export interface MenuInfo {
  title: string
  fields: MenuField[]
}

/** What entering input did: `menu` names the bare command whose menu the page shows instead of running it. */
export interface Entered {
  menu: string | null
}

/** What Alt+Enter did: `steered` when the running turn took it, else it was entered as Enter enters it. */
export interface Steering extends Entered {
  steered: boolean
}

export interface MenuEdited {
  messages: string[]
  menu: MenuInfo | null
}

/** The parts of a Pydantic JSON Schema the settings form reads. */
export interface Schema {
  type?: string
  enum?: JsonValue[]
  const?: JsonValue
  anyOf?: Schema[]
  oneOf?: Schema[]
  allOf?: Schema[]
  $ref?: string
  $defs?: Record<string, Schema>
  items?: Schema
  properties?: Record<string, Schema>
  title?: string
  description?: string
  default?: JsonValue
  minimum?: number
  maximum?: number
}

export type ChangeScope = 'turn' | 'session' | 'uncommitted' | 'branch'

/** One changed file in the Changes panel. */
export interface FileChange {
  path: string
  /** Added, modified, or deleted. */
  status: 'A' | 'M' | 'D'
  added: number
  removed: number
  diff: Diff
  /** Why the diff is empty, such as a binary file. */
  note: string | null
}

/** What a session changed in one scope. */
export interface Changes {
  scope: ChangeScope
  files: FileChange[]
  /** The branch `branch` compares with. */
  base: string | null
  /** Why the scope cannot be shown, such as a workspace outside Git. */
  note: string | null
  /** Changed files left out, past the most one answer lists. */
  omitted: number
}
