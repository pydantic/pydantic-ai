import type { Item, Questions, ServerMessage, SessionInfo, StatusInfo, ThemeInfo, Usage } from './protocol'

export interface NoticeEntry {
  id: string
  kind: 'notice'
  text: string
  level: 'info' | 'error'
  command: string | null
}

export interface TurnSummary {
  id: string
  kind: 'turn'
  outcome: 'completed' | 'failed' | 'cancelled'
  error: string | null
  usage: Usage | null
  /** Seconds the turn took, as this page saw it; `null` when it started before the page connected. */
  seconds: number | null
}

export type Entry = Item | NoticeEntry | TurnSummary

export interface State {
  session: SessionInfo | null
  entries: Entry[]
  running: boolean
  status: StatusInfo | null
  questions: Questions[]
  sessionsVersion: number
  /** When the running turn started, by this page's clock. */
  turnStartedAt: number | null
  theme: ThemeInfo | null
}

export const initial: State = {
  session: null,
  entries: [],
  running: false,
  status: null,
  questions: [],
  sessionsVersion: 0,
  turnStartedAt: null,
  theme: null,
}

let counter = 0
const nextId = (prefix: string) => `${prefix}${++counter}`

function update(entries: Entry[], id: string, change: (item: Item) => Item): Entry[] {
  for (let index = entries.length - 1; index >= 0; index--) {
    const entry = entries[index]
    if (entry && entry.id === id && entry.kind !== 'notice' && entry.kind !== 'turn') {
      const next = entries.slice()
      next[index] = change(entry)
      return next
    }
  }
  return entries
}

/** Apply one server message; every change makes new objects, so memoized rows re-render only when theirs changed. */
export function reduce(state: State, message: ServerMessage): State {
  switch (message.type) {
    case 'snapshot':
      return {
        ...state,
        session: message.session,
        entries: message.items,
        running: message.running,
        status: message.status,
        questions: message.questions,
        turnStartedAt: message.running ? (state.turnStartedAt ?? Date.now()) : null,
        theme: message.theme,
      }
    case 'item':
      return { ...state, entries: [...state.entries, message.item] }
    case 'delta':
      return { ...state, entries: update(state.entries, message.id, (item) => ({ ...item, text: item.text + message.text })) }
    case 'tool_output':
      return {
        ...state,
        entries: update(state.entries, message.id, (item) => ({ ...item, output: item.output + message.text })),
      }
    case 'tool_done':
      return {
        ...state,
        entries: update(state.entries, message.id, (item) => ({ ...item, text: message.text, state: message.state })),
      }
    case 'shell_done':
      return { ...state, entries: update(state.entries, message.id, (item) => ({ ...item, exit_code: message.exit_code })) }
    case 'diff':
      return {
        ...state,
        entries: update(state.entries, message.id, (item) => ({ ...item, diffs: [...item.diffs, message.diff] })),
      }
    case 'turn_start':
      // A turn a finished task woke has no prompt of its own.
      return {
        ...state,
        running: true,
        turnStartedAt: Date.now(),
        entries: message.item ? [...state.entries, message.item] : state.entries,
      }
    case 'turn_end': {
      const entries = state.entries.map((entry) =>
        entry.kind === 'tool' && entry.state === 'running' ? { ...entry, state: 'interrupted' as const } : entry,
      )
      const summary: TurnSummary = {
        id: nextId('turn'),
        kind: 'turn',
        outcome: message.outcome,
        error: message.error,
        usage: message.usage,
        seconds: state.turnStartedAt === null ? null : (Date.now() - state.turnStartedAt) / 1000,
      }
      return { ...state, running: false, turnStartedAt: null, entries: [...entries, summary] }
    }
    case 'status':
      return { ...state, status: message.status }
    case 'question':
      return { ...state, questions: [...state.questions.filter((q) => q.id !== message.questions.id), message.questions] }
    case 'question_closed':
      return { ...state, questions: state.questions.filter((q) => q.id !== message.id) }
    case 'notice': {
      const notice: NoticeEntry = {
        id: nextId('notice'),
        kind: 'notice',
        text: message.text,
        level: message.level,
        command: message.command,
      }
      return { ...state, entries: [...state.entries, notice] }
    }
    case 'title':
      return state.session ? { ...state, session: { ...state.session, title: message.title } } : state
    case 'theme':
      return { ...state, theme: message.theme }
    case 'sessions':
      return { ...state, sessionsVersion: state.sessionsVersion + 1 }
  }
}
