// Mirrors pydantic_clai2/builtin_plugins/web/protocol.py.

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
  /** A cell the user ran with `/repl`, rather than one the model wrote. */
  by_user: boolean
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
  kind: 'choose' | 'text' | 'confirm' | 'wait'
  title: string
  description: string
  options: Choice[]
  current: string | null
  label: string
  placeholder: string
  initial: string
  secret: boolean
  yes: string
  no: string
  /** `wait`: the latest text its task showed, such as a sign-in link. */
  shown: string
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

export interface Palette {
  bg: string
  fg: string
  ansi: string[]
}

export interface ThemeInfo {
  /** `default` is Pydantic's own colours, in the browser's light or dark mode. */
  name: string
  names: string[]
  palette: Palette | null
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
    }
  | { type: 'item'; item: Item }
  | { type: 'delta'; id: string; text: string }
  | { type: 'tool_output'; id: string; text: string }
  | { type: 'args_delta'; id: string; text: string }
  | { type: 'tool_args'; id: string; args: string }
  | { type: 'call'; id: string; call: SubCall }
  | { type: 'call_output'; id: string; call_id: string; text: string }
  | { type: 'call_done'; id: string; call_id: string; text: string; state: ToolState }
  | { type: 'tool_done'; id: string; text: string; state: ToolState }
  | { type: 'shell_done'; id: string; exit_code: number | null }
  | { type: 'diff'; id: string; diff: Diff }
  | { type: 'turn_start'; item: Item | null }
  | { type: 'turn_end'; outcome: 'completed' | 'failed' | 'cancelled'; error: string | null; usage: Usage | null }
  | { type: 'status'; status: StatusInfo }
  | { type: 'question'; questions: Questions }
  | { type: 'question_closed'; id: string }
  | { type: 'dialog'; dialog: Dialog }
  | { type: 'dialog_shown'; id: string; text: string }
  | { type: 'dialog_closed'; id: string }
  | { type: 'notice'; text: string; level: 'info' | 'error'; command: string | null }
  | { type: 'title'; title: string | null }
  | { type: 'theme'; theme: ThemeInfo }
  | { type: 'sessions' }
  | { type: 'plugins' }

export type SessionState = 'ready' | 'running' | 'busy' | 'completed' | 'failed' | 'cancelled'

export interface SessionEntry {
  id: string
  title: string
  workspace: string
  updated_at: string
  state: SessionState
  messages: number
  model: string | null
  current: boolean
}

export interface Project {
  key: string
  name: string
  current: boolean
  sessions: SessionEntry[]
}

export interface Transcript {
  id: string
  title: string
  workspace: string
  items: Item[]
  openable: boolean
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
}

export interface MenuInfo {
  title: string
  fields: MenuField[]
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
