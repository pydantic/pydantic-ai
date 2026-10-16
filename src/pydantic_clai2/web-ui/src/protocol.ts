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
}

export interface Usage {
  input_tokens: number
  output_tokens: number
  requests: number
}

export type ServerMessage =
  | { type: 'snapshot'; session: SessionInfo; items: Item[]; running: boolean; status: StatusInfo; questions: Questions[] }
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
