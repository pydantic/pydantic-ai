import type {
  Dialog,
  Item,
  ItemMessage,
  LiveEntry,
  OrchestratorEvent,
  QueuedItem,
  Questions,
  ServerMessage,
  SessionInfo,
  SessionMessage,
  StatusInfo,
  TaskEntry,
  ThemeInfo,
  UpdateInfo,
  Usage,
  UsageReport,
} from './protocol'

export interface NoticeEntry {
  id: string
  kind: 'notice'
  text: string
  level: 'info' | 'error'
  command: string | null
  /** For `/usage` and `/cost`: the figures their text reports, shown as a card. */
  usage: UsageReport | null
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

/** One conversation as the page shows it: a live session's, or the orchestrator's. */
export interface Chat {
  session: SessionInfo | null
  entries: Entry[]
  running: boolean
  status: StatusInfo | null
  questions: Questions[]
  /** When the running turn started, by this page's clock. */
  turnStartedAt: number | null
  /** The conversation's delegated tasks, each parent before its children. */
  tasks: TaskEntry[]
  /** What each running task has done so far, by task ID; a finished task's saved history is fetched instead. */
  taskItems: Record<string, Entry[]>
  /** The input waiting for the running turn or command, oldest first, as the session's prompt holds it. */
  queue: QueuedItem[]
}

export const initialChat: Chat = {
  session: null,
  entries: [],
  running: false,
  status: null,
  questions: [],
  turnStartedAt: null,
  tasks: [],
  taskItems: {},
  queue: [],
}

export type Snapshot = Extract<ServerMessage, { type: 'snapshot' }>

export interface State {
  /** The live session the main pane shows, by conversation ID; `null` until the first snapshot. */
  shown: string | null
  /** The session the page last asked to watch, until its snapshot comes. */
  awaiting: string | null
  /**
   * Whether a `live` listed `awaiting`, or a reconnect asked for it, so a `live` without it means it closed; one sent
   * before it opened may still arrive after the page asked.
   */
  confirmed: boolean
  /** What showed before `awaiting`, shown again if the awaited session turns out not to be live. */
  previous: string | null
  /**
   * Another session's snapshot that came while waiting: on reconnecting, the default session's, shown if the awaited
   * one is gone.
   */
  stray: Snapshot | null
  /**
   * The shown session's chat, by conversation ID; the page watches no other, so no other chat would stay current.
   * Until a newly shown session's snapshot comes, this is still the previous one's.
   */
  chats: Record<string, Chat>
  /** The sessions live in this CLAI, by conversation ID, with what each is doing. */
  live: Record<string, LiveEntry>
  /** What open settings menus ask, oldest first; they belong to the page, not to one session. */
  dialogs: Dialog[]
  sessionsVersion: number
  /** Bumped after every command, which may have changed a plugin; an open plugins panel lists them again. */
  pluginsVersion: number
  /** Bumped after every command, or edit of a command's menu, which may have changed CLAI's settings. */
  settingsVersion: number
  theme: ThemeInfo | null
  /** Whether each session runs in its own directory, so a new one asks where. */
  followsWorkspace: boolean
  /** The CLAI this page talks to, by version; `null` until it connects. */
  version: string | null
  /** `/update` installing a newer CLAI, until the install ends or the new build is back. */
  update: UpdateInfo | null
}

export const initial: State = {
  shown: null,
  awaiting: null,
  confirmed: false,
  previous: null,
  stray: null,
  chats: {},
  live: {},
  dialogs: [],
  sessionsVersion: 0,
  pluginsVersion: 0,
  settingsVersion: 0,
  theme: null,
  followsWorkspace: false,
  version: null,
  update: null,
}

/**
 * What the page does itself: show a session, after asking the server to watch it, again when `reconnect`ing; stop waiting
 * for one that never answered; or show the default session's snapshot when nothing shows.
 */
export type Action =
  | ServerMessage
  | { type: 'show'; id: string; reconnect: boolean }
  | { type: 'give_up'; id: string }
  | { type: 'fallback'; snapshot: Snapshot }

/** The chat of the session the main pane shows, empty while it is on its way. */
export function shownChat(state: State): Chat {
  return (state.shown === null ? undefined : state.chats[state.shown]) ?? initialChat
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
function taskChanged(chat: Chat, task: TaskEntry): Chat {
  const found = chat.tasks.some((entry) => entry.id === task.id)
  const tasks = found ? chat.tasks.map((entry) => (entry.id === task.id ? task : entry)) : [...chat.tasks, task]
  if (task.status === 'running' || !(task.id in chat.taskItems)) return { ...chat, tasks }
  const taskItems = { ...chat.taskItems }
  delete taskItems[task.id]
  return { ...chat, tasks, taskItems }
}

/**
 * Apply one message about a conversation; every change makes new objects, so memoized rows re-render only when theirs
 * changed.
 */
export function reduceChat(chat: Chat, message: SessionMessage): Chat {
  switch (message.type) {
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
      return { ...chat, entries: applyItem(chat.entries, message) }
    case 'task':
      return taskChanged(chat, message.task)
    case 'task_event': {
      const items = message.messages.reduce(applyItem, chat.taskItems[message.id] ?? [])
      return { ...chat, taskItems: { ...chat.taskItems, [message.id]: items } }
    }
    case 'turn_start':
      // A turn a finished task woke has no prompt of its own.
      return {
        ...chat,
        running: true,
        turnStartedAt: Date.now(),
        entries: message.item ? [...chat.entries, message.item] : chat.entries,
      }
    case 'turn_end': {
      const entries = chat.entries.map((entry) =>
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
        seconds: chat.turnStartedAt === null ? null : (Date.now() - chat.turnStartedAt) / 1000,
      }
      return { ...chat, running: false, turnStartedAt: null, entries: [...entries, summary] }
    }
    case 'status':
      return { ...chat, status: message.status }
    case 'question':
      return { ...chat, questions: [...chat.questions.filter((q) => q.id !== message.questions.id), message.questions] }
    case 'question_closed':
      return { ...chat, questions: chat.questions.filter((q) => q.id !== message.id) }
    case 'notice': {
      const notice: NoticeEntry = {
        id: nextId('notice'),
        kind: 'notice',
        text: message.text,
        level: message.level,
        command: message.command,
        usage: message.usage,
      }
      return { ...chat, entries: [...chat.entries, notice] }
    }
    case 'title':
      return chat.session ? { ...chat, session: { ...chat.session, title: message.title } } : chat
    case 'queue':
      return { ...chat, queue: message.queue }
  }
}

/** Show a snapshot: its session's chat starts over from it, and it carries the page-wide state too. */
function applySnapshot(state: State, message: Snapshot): State {
  const id = message.session.id
  const previous = state.chats[id]
  // A CLAI that `/update` restarted comes back as a new version, which the page says once it has reconnected.
  const updated = state.version !== null && message.version !== state.version
  const notice: NoticeEntry = {
    id: nextId('notice'),
    kind: 'notice',
    text: `Updated CLAI to ${message.version}.`,
    level: 'info',
    command: null,
    usage: null,
  }
  const chat: Chat = {
    session: message.session,
    entries: updated ? [...message.items, notice] : message.items,
    running: message.running,
    status: message.status,
    questions: message.questions,
    turnStartedAt: message.running ? (previous?.turnStartedAt ?? null) : null,
    tasks: message.tasks,
    taskItems: message.task_items,
    queue: message.queue,
  }
  return {
    ...state,
    shown: id,
    awaiting: null,
    confirmed: false,
    previous: null,
    stray: null,
    chats: { [id]: chat },
    dialogs: message.dialogs,
    theme: message.theme,
    followsWorkspace: message.follows_workspace,
    version: message.version,
    update: message.update,
  }
}

/** The awaited session will not answer: show what came instead, or what showed before if it is live, or nothing. */
function giveUp(state: State): State {
  if (state.stray) return applySnapshot(state, state.stray)
  const back = state.previous !== null && state.previous in state.live ? state.previous : null
  return { ...state, shown: back, awaiting: null, confirmed: false, previous: null }
}

/** Apply one server message or page action to the page's state. */
export function reduce(state: State, action: Action): State {
  switch (action.type) {
    case 'show': {
      // Shown and watched already: no new snapshot answers, so there is nothing to wait for.
      if (!action.reconnect && action.id === state.shown && state.awaiting === null) return state
      // A reconnect's first `live` is current, so a session missing from it has closed.
      const confirmed = action.reconnect || action.id in state.live
      const previous = state.previous ?? state.shown
      return { ...state, shown: action.id, awaiting: action.id, confirmed, previous, stray: null }
    }
    case 'snapshot': {
      const id = action.session.id
      // While a watch is unanswered, another session's snapshot is the one a reconnect sends first, kept in case.
      if (id !== state.awaiting && state.awaiting !== null) return { ...state, stray: action }
      // Otherwise it is the shown session's, the first one, or the shown session's new conversation (`/new`).
      return applySnapshot(state, action)
    }
    case 'live': {
      const live = Object.fromEntries(action.sessions.map((entry) => [entry.id, entry]))
      const next = { ...state, live }
      // The shown session closed, as with `/exit` typed in it: nothing shows until the default session does.
      if (state.awaiting === null) return state.shown === null || state.shown in live ? next : { ...next, shown: null }
      if (state.awaiting in live) return { ...next, confirmed: true }
      // Sent before the awaited session opened: its answer is still to come.
      if (!state.confirmed) return next
      // The awaited session closed, so no answer comes.
      return giveUp(next)
    }
    case 'give_up':
      return state.awaiting === action.id ? giveUp(state) : state
    case 'fallback':
      // Only while nothing shows or is on its way: a session picked meanwhile wins.
      return state.shown === null && state.awaiting === null ? applySnapshot(state, action.snapshot) : state
    case 'dialog':
      return { ...state, dialogs: [...state.dialogs.filter((d) => d.id !== action.dialog.id), action.dialog] }
    case 'dialog_shown':
      return { ...state, dialogs: state.dialogs.map((d) => (d.id === action.id ? { ...d, shown: action.text } : d)) }
    case 'dialog_closed':
      return { ...state, dialogs: state.dialogs.filter((d) => d.id !== action.id) }
    case 'update':
      return { ...state, update: action.update }
    case 'theme':
      return { ...state, theme: action.theme }
    case 'sessions':
      return { ...state, sessionsVersion: state.sessionsVersion + 1 }
    case 'plugins':
      return { ...state, pluginsVersion: state.pluginsVersion + 1 }
    case 'settings':
      return { ...state, settingsVersion: state.settingsVersion + 1 }
    case 'orchestrator':
      // The orchestrator chat keeps its own state, with `reduceOrchestrator`.
      return state
    default: {
      // An older CLAI names no session: its messages are the shown one's.
      const id = action.session ?? state.shown
      const chat = id === null ? undefined : state.chats[id]
      // Messages still on their way about a session the page never showed have no chat to change.
      if (id === null || chat === undefined) return state
      return { ...state, chats: { ...state.chats, [id]: reduceChat(chat, action) } }
    }
  }
}

/** The orchestrator chat: a transcript of its own, kept by the same reducer as a session's. */
export interface OrchestratorView {
  /** Whether the `workspaces` plugin is on, which the orchestrator needs; `null` until the server says. */
  enabled: boolean | null
  /** The saved default model, which the orchestrator runs on; `null` when there is none. */
  model: string | null
  chat: Chat
}

export const initialOrchestrator: OrchestratorView = { enabled: null, model: null, chat: initialChat }

/** What the orchestrator chat gets on the WebSocket, or `enabled` and `model` from asking the server again. */
export type OrchestratorAction = OrchestratorEvent | { type: 'refreshed'; enabled: boolean; model: string | null }

export function reduceOrchestrator(view: OrchestratorView, action: OrchestratorAction): OrchestratorView {
  if (action.type === 'refreshed') return { ...view, enabled: action.enabled, model: action.model }
  if (action.type !== 'orchestrator_state') return { ...view, chat: reduceChat(view.chat, action) }
  const chat = view.chat
  return {
    enabled: action.enabled,
    model: action.model,
    chat: {
      ...chat,
      entries: action.items,
      running: action.running,
      questions: action.questions,
      turnStartedAt: action.running ? chat.turnStartedAt : null,
    },
  }
}
