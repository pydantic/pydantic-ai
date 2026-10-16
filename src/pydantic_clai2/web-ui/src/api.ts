import type {
  AnswerBody,
  CompletionInfo,
  JsonValue,
  MenuEdited,
  MenuInfo,
  OrchestratorState,
  PanelInfo,
  PluginAction,
  PluginRow,
  Project,
  ServerMessage,
  Switch,
  TaskTranscript,
  Transcript,
  WorkspacesInfo,
} from './protocol'

// The page's token: the server puts it in a meta tag; `npm run dev` serves its own page, so the address has it.
const token =
  document.querySelector<HTMLMetaElement>('meta[name="clai-token"]')?.content ??
  new URLSearchParams(location.search).get('token') ??
  ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** What is wrong with each rejected setting, by its path, such as `limit` or `token.name`. */
    readonly fields: Record<string, string> = {},
  ) {
    super(message)
  }
}

function problem(detail: unknown): { message: string; fields: Record<string, string> } | null {
  if (!detail || typeof detail !== 'object' || !('message' in detail) || !('errors' in detail)) return null
  const { message, errors } = detail as { message: unknown; errors: unknown }
  if (typeof message !== 'string' || !errors || typeof errors !== 'object') return null
  return { message, fields: Object.fromEntries(Object.entries(errors).map(([key, value]) => [key, String(value)])) }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { 'x-clai-token': token, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    let detail = response.statusText
    let fields: Record<string, string> = {}
    try {
      const data: unknown = await response.json()
      if (data && typeof data === 'object' && 'detail' in data) {
        const value = (data as { detail: unknown }).detail
        const rejected = problem(value)
        if (rejected) ({ message: detail, fields } = rejected)
        else detail = typeof value === 'string' ? value : JSON.stringify(value)
      }
    } catch {
      // Not JSON: keep the status text.
    }
    throw new ApiError(detail, response.status, fields)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export const api = {
  send: (text: string) => call<void>('POST', '/api/input', { text }),
  /** Install the newest CLAI as `/update` does; the `update` messages and the next snapshot's version report it. */
  update: () => call<void>('POST', '/api/input', { text: '/update' }),
  interrupt: () => call<{ interrupted: boolean }>('POST', '/api/interrupt'),
  complete: (text: string) => call<CompletionInfo[]>('GET', `/api/complete?text=${encodeURIComponent(text)}`),
  models: () => call<string[]>('GET', '/api/models'),
  selectModel: (name: string) => call<void>('POST', '/api/model', { name }),
  setSwitch: (key: Switch, on: boolean) => call<void>('POST', '/api/switch', { key, on }),
  answer: (id: string, answers: AnswerBody[] | null) => call<void>('POST', `/api/questions/${id}`, { answers }),
  sessions: (query: string, archived = false) =>
    call<Project[]>('GET', `/api/sessions?q=${encodeURIComponent(query)}${archived ? '&include_archived=true' : ''}`),
  transcript: (id: string) => call<Transcript>('GET', `/api/sessions/${id}`),
  newSession: (workspace?: string) =>
    call<void>('POST', '/api/sessions/new', workspace === undefined ? undefined : { workspace }),
  open: (id: string) => call<void>('POST', `/api/sessions/${id}/open`),
  rename: (id: string, title: string) => call<void>('POST', `/api/sessions/${id}/rename`, { title }),
  fork: (id: string) => call<{ id: string }>('POST', `/api/sessions/${id}/fork`),
  archive: (id: string) => call<void>('POST', `/api/sessions/${id}/archive`),
  unarchive: (id: string) => call<void>('POST', `/api/sessions/${id}/unarchive`),
  remove: (id: string) => call<void>('DELETE', `/api/sessions/${id}`),
  plugins: () => call<PluginRow[]>('GET', '/api/plugins'),
  plugin: (name: string, action: PluginAction) => call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/${action}`),
  pluginSettings: (name: string, changes: Record<string, JsonValue>, reset: string[] = []) =>
    call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/settings`, { changes, reset }),
  pluginMenu: (name: string) => call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/menu`),
  task: (id: string) => call<TaskTranscript>('GET', `/api/tasks/${id}`),
  taskAction: (id: string, action: 'stop' | 'background' | 'resume') => call<void>('POST', `/api/tasks/${id}/${action}`),
  steer: (id: string, text: string) => call<{ steered: boolean }>('POST', `/api/tasks/${id}/steer`, { text }),
  menu: (name: string) => call<MenuInfo>('GET', `/api/plugins/${encodeURIComponent(name)}/fields`),
  setField: (name: string, key: string, value: string | null) =>
    call<MenuEdited>('POST', `/api/plugins/${encodeURIComponent(name)}/fields/${encodeURIComponent(key)}`, { value }),
  runAction: (name: string, key: string) =>
    call<MenuEdited>('POST', `/api/plugins/${encodeURIComponent(name)}/fields/${encodeURIComponent(key)}/action`),
  answerDialog: (id: string, value: string | null) => call<void>('POST', `/api/dialogs/${id}`, { value }),
  panels: () => call<PanelInfo[]>('GET', '/api/panels'),
  workspaces: () => call<WorkspacesInfo>('GET', '/api/workspaces'),
  openWorkspace: (id: string) => call<{ url: string }>('POST', `/api/workspaces/${encodeURIComponent(id)}/open`),
}

/** The orchestrator chat: its own agent and conversation, beside the session. */
export const orchestrator = {
  state: () => call<OrchestratorState>('GET', '/api/orchestrator/state'),
  send: (text: string) => call<void>('POST', '/api/orchestrator/input', { text }),
  interrupt: () => call<{ interrupted: boolean }>('POST', '/api/orchestrator/interrupt'),
  answer: (id: string, answers: AnswerBody[] | null) => call<void>('POST', '/api/orchestrator/answer', { id, answers }),
  clear: () => call<void>('POST', '/api/orchestrator/clear'),
}

/** The bridge `clai2 --desktop`'s window gives the page; absent in a browser. */
interface DesktopWindow {
  pywebview?: { api?: { open_external?: (url: string) => Promise<boolean> } }
}

/** In `clai2 --desktop`'s window, which opens no windows of its own, open `url` in the default browser instead. */
export function openExternal(url: string): boolean {
  const open = (window as Window & DesktopWindow).pywebview?.api?.open_external
  if (!open) return false
  void open(url)
  return true
}

/** Start a workspace's CLAI if needed and open its own page in a new tab. */
export async function openWorkspacePage(id: string): Promise<void> {
  if ((window as Window & DesktopWindow).pywebview) {
    openExternal((await api.openWorkspace(id)).url)
    return
  }
  // Opened now, while the click still counts as one, and pointed at the page once its CLAI answers.
  const tab = window.open('', '_blank')
  try {
    const { url } = await api.openWorkspace(id)
    if (tab) {
      tab.opener = null
      tab.location.href = url
    } else window.location.assign(url)
  } catch (error) {
    tab?.close()
    throw error
  }
}

/** Interrupt the running turn, reporting a failure, such as a server that has gone, through `onError`. */
export async function interrupt(
  onError: (message: string) => void,
  stop: () => Promise<unknown> = api.interrupt,
): Promise<void> {
  try {
    await stop()
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  }
}

export type Connection = 'connecting' | 'open' | 'closed' | 'expired'

/** Keep one WebSocket open, reconnecting with backoff; each connection starts with a snapshot.

A CLAI restarted since this page loaded has a new token: after a few failed attempts this checks, and
reports `expired` instead of retrying forever.
 */
export function connect(onMessage: (message: ServerMessage) => void, onState: (state: Connection) => void): () => void {
  let socket: WebSocket | null = null
  let stopped = false
  let delay = 500
  let failures = 0
  let timer: number | undefined
  const retry = async () => {
    if (++failures >= 3) {
      const check = await fetch('/api/state', { headers: { 'x-clai-token': token } }).catch(() => null)
      if (check?.status === 403) {
        onState('expired')
        return
      }
    }
    if (!stopped) {
      timer = window.setTimeout(open, delay)
      // At most 2 seconds apart, so a page waiting on `/update`'s restart is back soon after CLAI is.
      delay = Math.min(delay * 2, 2000)
    }
  }
  const open = () => {
    onState('connecting')
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${scheme}://${location.host}/ws?token=${encodeURIComponent(token)}`)
    socket.onopen = () => {
      delay = 500
      failures = 0
      onState('open')
    }
    socket.onmessage = (event: MessageEvent<string>) => onMessage(JSON.parse(event.data) as ServerMessage)
    socket.onclose = () => {
      if (stopped) return
      onState('closed')
      void retry()
    }
  }
  open()
  return () => {
    stopped = true
    window.clearTimeout(timer)
    socket?.close()
  }
}
