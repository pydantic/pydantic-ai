// Mirrors pydantic_clai2/builtin_plugins/web/protocol.py.

export type ItemKind = 'user' | 'assistant' | 'thinking' | 'tool'
export type ToolState = 'running' | 'done' | 'error' | 'interrupted'

export interface Diff {
  path: string
  diff: string
  truncated: boolean
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
      theme: ThemeInfo
    }
  | { type: 'item'; item: Item }
  | { type: 'delta'; id: string; text: string }
  | { type: 'tool_output'; id: string; text: string }
  | { type: 'tool_done'; id: string; text: string; state: ToolState }
  | { type: 'shell_done'; id: string; exit_code: number | null }
  | { type: 'diff'; id: string; diff: Diff }
  | { type: 'turn_start'; item: Item | null }
  | { type: 'turn_end'; outcome: 'completed' | 'failed' | 'cancelled'; error: string | null; usage: Usage | null }
  | { type: 'status'; status: StatusInfo }
  | { type: 'question'; questions: Questions }
  | { type: 'question_closed'; id: string }
  | { type: 'notice'; text: string; level: 'info' | 'error'; command: string | null }
  | { type: 'title'; title: string | null }
  | { type: 'theme'; theme: ThemeInfo }
  | { type: 'sessions' }

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
