import type {
  Dialog,
  Item,
  ItemMessage,
  OrchestratorEvent,
  Questions,
  ServerMessage,
  SessionInfo,
  StatusInfo,
  TaskEntry,
  ThemeInfo,
  UpdateInfo,
  Usage,
} from './protocol'

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
  /** What open settings menus ask, oldest first. */
  dialogs: Dialog[]
  sessionsVersion: number
  /** Bumped after every command, which may have changed a plugin; an open plugins panel lists them again. */
  pluginsVersion: number
  /** When the running turn started, by this page's clock. */
  turnStartedAt: number | null
  theme: ThemeInfo | null
  /** Whether each session runs in its own directory, so a new one asks where. */
  followsWorkspace: boolean
  /** The conversation's delegated tasks, each parent before its children. */
  tasks: TaskEntry[]
  /** What each running task has done so far, by task ID; a finished task's saved history is fetched instead. */
  taskItems: Record<string, Entry[]>
  /** The CLAI this page talks to, by version; `null` until it connects. */
  version: string | null
  /** `/update` installing a newer CLAI, until the install ends or the new build is back. */
  update: UpdateInfo | null
}

export const initial: State = {
  session: null,
  entries: [],
  running: false,
  status: null,
  questions: [],
  dialogs: [],
  sessionsVersion: 0,
  pluginsVersion: 0,
  turnStartedAt: null,
  theme: null,
  followsWorkspace: false,
  tasks: [],
  taskItems: {},
  version: null,
  update: null,
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

/** Apply one message to a transcript's entries, the session's or a delegated task's. */
export function applyItem(entries: Entry[], message: ItemMessage): Entry[] {
  if (message.type === 'item') return [...entries, message.item]
  const change = (edit: (item: Item) => Item) => update(entries, message.id, edit)
  switch (message.type) {
    case 'delta':
      return change((item) => ({ ...item, text: item.text + message.text }))
    case 'tool_output':
      return change((item) => ({ ...item, output: item.output + message.text }))
    case 'tool_done':
      return change((item) => ({ ...item, text: message.text, state: message.state }))
    case 'args_delta':
      return change((item) => ({ ...item, args: item.args + message.text }))
    case 'tool_args':
      return change((item) => ({ ...item, args: message.args }))
    case 'call':
      return change((item) => ({ ...item, calls: [...item.calls, message.call] }))
    case 'call_output':
      return change((item) => ({
        ...item,
        calls: item.calls.map((call) => (call.id === message.call_id ? { ...call, output: call.output + message.text } : call)),
      }))
    case 'call_done':
      return change((item) => ({
        ...item,
        calls: item.calls.map((call) =>
          call.id === message.call_id ? { ...call, text: message.text, state: message.state } : call,
        ),
      }))
    case 'shell_done':
      return change((item) => ({ ...item, exit_code: message.exit_code }))
    case 'diff':
      return change((item) => ({ ...item, diffs: [...item.diffs, message.diff] }))
  }
}

/** A task's new row replaces its old one, or joins the list; a finished task's live items go, as its saved history serves. */
function taskChanged(state: State, task: TaskEntry): State {
  const found = state.tasks.some((entry) => entry.id === task.id)
  const tasks = found ? state.tasks.map((entry) => (entry.id === task.id ? task : entry)) : [...state.tasks, task]
  if (task.status === 'running' || !(task.id in state.taskItems)) return { ...state, tasks }
  const taskItems = { ...state.taskItems }
  delete taskItems[task.id]
  return { ...state, tasks, taskItems }
}

/** Apply one server message; every change makes new objects, so memoized rows re-render only when theirs changed. */
export function reduce(state: State, message: ServerMessage): State {
  switch (message.type) {
    case 'snapshot': {
      // A CLAI that `/update` restarted comes back as a new version, which the page says once it has reconnected.
      const updated = state.version !== null && message.version !== state.version
      const notice: NoticeEntry = {
        id: nextId('notice'),
        kind: 'notice',
        text: `Updated CLAI to ${message.version}.`,
        level: 'info',
        command: null,
      }
      return {
        ...state,
        session: message.session,
        entries: updated ? [...message.items, notice] : message.items,
        running: message.running,
        status: message.status,
        questions: message.questions,
        dialogs: message.dialogs,
        turnStartedAt: message.running ? state.turnStartedAt : null,
        theme: message.theme,
        followsWorkspace: message.follows_workspace,
        tasks: message.tasks,
        taskItems: message.task_items,
        version: message.version,
        update: message.update,
      }
    }
    case 'update':
      return { ...state, update: message.update }
    case 'item':
    case 'delta':
    case 'tool_output':
    case 'tool_done':
    case 'args_delta':
    case 'tool_args':
    case 'call':
    case 'call_output':
    case 'call_done':
    case 'shell_done':
    case 'diff':
      return { ...state, entries: applyItem(state.entries, message) }
    case 'task':
      return taskChanged(state, message.task)
    case 'task_event': {
      const items = message.messages.reduce(applyItem, state.taskItems[message.id] ?? [])
      return { ...state, taskItems: { ...state.taskItems, [message.id]: items } }
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
        entry.kind === 'tool' && (entry.state === 'running' || entry.calls.some((call) => call.state === 'running'))
          ? {
              ...entry,
              state: entry.state === 'running' ? ('interrupted' as const) : entry.state,
              calls: entry.calls.map((call) => (call.state === 'running' ? { ...call, state: 'interrupted' as const } : call)),
            }
          : entry,
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
    case 'dialog':
      return { ...state, dialogs: [...state.dialogs.filter((d) => d.id !== message.dialog.id), message.dialog] }
    case 'dialog_shown':
      return { ...state, dialogs: state.dialogs.map((d) => (d.id === message.id ? { ...d, shown: message.text } : d)) }
    case 'dialog_closed':
      return { ...state, dialogs: state.dialogs.filter((d) => d.id !== message.id) }
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
    case 'plugins':
      return { ...state, pluginsVersion: state.pluginsVersion + 1 }
    case 'orchestrator':
      // The orchestrator chat keeps its own state, with `reduceOrchestrator`.
      return state
  }
}

/** The orchestrator chat: a transcript of its own, kept by the same reducer as the session's. */
export interface OrchestratorView {
  /** Whether the `workspaces` plugin is on, which the orchestrator needs; `null` until the server says. */
  enabled: boolean | null
  chat: State
}

export const initialOrchestrator: OrchestratorView = { enabled: null, chat: initial }

/** What the orchestrator chat gets on the WebSocket, or `enabled` from asking the server again. */
export type OrchestratorAction = OrchestratorEvent | { type: 'enabled'; enabled: boolean }

export function reduceOrchestrator(view: OrchestratorView, action: OrchestratorAction): OrchestratorView {
  if (action.type === 'enabled') return { ...view, enabled: action.enabled }
  if (action.type !== 'orchestrator_state') return { ...view, chat: reduce(view.chat, action) }
  const chat = view.chat
  return {
    enabled: action.enabled,
    chat: {
      ...chat,
      entries: action.items,
      running: action.running,
      questions: action.questions,
      turnStartedAt: action.running ? chat.turnStartedAt : null,
    },
  }
}
