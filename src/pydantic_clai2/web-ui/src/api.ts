import { createContext } from 'react'
import type { Snapshot } from './state'
import type {
  AnswerBody,
  CompletionInfo,
  Entered,
  JsonValue,
  MenuEdited,
  MenuInfo,
  Opened,
  OrchestratorState,
  PanelInfo,
  PluginAction,
  PluginRow,
  Project,
  ServerMessage,
  Steering,
  Switch,
  TaskTranscript,
  Transcript,
  WatchMessage,
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

// Fetched images as `data:` URLs, by ID. An ID is the digest of the bytes, so every session shares them.
const images = new Map<string, Promise<string>>()

/** An image of a prompt as a `data:` URL, which the page's policy allows where it allows no `blob:` one. */
function image(base: string, id: string): Promise<string> {
  let found = images.get(id)
  if (!found) {
    found = fetch(`${base}/images/${encodeURIComponent(id)}`, { headers: { 'x-clai-token': token } })
      .then(async (response) => {
        if (!response.ok) throw new ApiError(response.statusText, response.status)
        const blob = await response.blob()
        return new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(reader.result as string)
          reader.onerror = () => reject(reader.error ?? new Error('The image could not be read.'))
          reader.readAsDataURL(blob)
        })
      })
      // A failure is not kept, so another session, or a later try, can still find the image.
      .catch((error: unknown) => {
        images.delete(id)
        throw error
      })
    images.set(id, found)
  }
  return found
}

/** A declared settings menu: a plugin's, or the one a command such as `/set` opens. */
export interface MenuTarget {
  kind: 'plugins' | 'commands'
  name: string
  /** The rows chosen in turn to reach a menu one of them opens, such as a saved model's settings from `/model`'s. */
  path?: readonly string[]
}

/** A plugin's menu is page-wide; a command's acts on the session it is typed in, at `base`. */
const fieldsPath = (base: string, target: MenuTarget) =>
  `${target.kind === 'plugins' ? '/api' : base}/${target.kind}/${encodeURIComponent(target.name)}/fields`
const menuQuery = (target: MenuTarget) =>
  target.path?.length ? `?${target.path.map((key) => `menu=${encodeURIComponent(key)}`).join('&')}` : ''

/** The routes of one live session, by conversation ID; `null` is the default session's, before the first snapshot. */
export function sessionApi(id: string | null) {
  const base = id === null ? '/api' : `/api/sessions/${encodeURIComponent(id)}`
  return {
    /** Enter a prompt or `/command`; `images`, base64-encoded, go with a prompt. */
    send: (text: string, images: readonly string[] = []) =>
      call<Entered>('POST', `${base}/input`, images.length ? { text, images } : { text }),
    /** Send a prompt to the running turn now, as Alt+Enter at the prompt does; one that cannot steer is entered instead. */
    steer: (text: string) => call<Steering>('POST', `${base}/steer`, { text }),
    /** Rewrite a queued input in place; a 404 means it already ran or was removed. */
    editQueued: (id: string, text: string) => call<void>('PUT', `${base}/queue/${encodeURIComponent(id)}`, { text }),
    removeQueued: (id: string) => call<void>('DELETE', `${base}/queue/${encodeURIComponent(id)}`),
    /** Send a queued prompt to the running turn now; `steered: false` leaves it queued. */
    steerQueued: (id: string) => call<{ steered: boolean }>('POST', `${base}/queue/${encodeURIComponent(id)}/steer`),
    interrupt: () => call<{ interrupted: boolean }>('POST', `${base}/interrupt`),
    /** One of the images a prompt in this session's transcript lists, as a `data:` URL; fetched once per page. */
    image: (id: string) => image(base, id),
    complete: (text: string) => call<CompletionInfo[]>('GET', `${base}/complete?text=${encodeURIComponent(text)}`),
    selectModel: (name: string) => call<void>('POST', `${base}/model`, { name }),
    answer: (request: string, answers: AnswerBody[] | null) =>
      call<void>('POST', `${base}/questions/${request}`, { answers }),
    /** Flip a run switch as `/set` typed in the session does; a running turn keeps the tools it started with. */
    setSwitch: (key: Switch, on: boolean) => call<void>('POST', `${base}/switch`, { key, on }),
    /** A command's settings menu as this session sees it, such as `/model`'s; a plugin's is page-wide. */
    menu: (target: MenuTarget) => call<MenuInfo>('GET', `${fieldsPath(base, target)}${menuQuery(target)}`),
    setField: (target: MenuTarget, key: string, value: string | null) =>
      call<MenuEdited>('POST', `${fieldsPath(base, target)}/${encodeURIComponent(key)}${menuQuery(target)}`, { value }),
    runAction: (target: MenuTarget, key: string) =>
      call<MenuEdited>('POST', `${fieldsPath(base, target)}/${encodeURIComponent(key)}/action${menuQuery(target)}`),
  }
}

export type SessionApi = ReturnType<typeof sessionApi>

/** The routes of the session the main pane shows, for the composer, its questions, Monty, and command menus. */
export const SessionContext = createContext<SessionApi>(sessionApi(null))

export const api = {
  /** Install the newest CLAI as `/update` does; the `update` messages and the next snapshot's version report it. */
  update: () => call<void>('POST', '/api/input', { text: '/update' }),
  models: () => call<string[]>('GET', '/api/models'),
  /** The default session's snapshot: the terminal's, else the one opened last. */
  state: () => call<Snapshot>('GET', '/api/state'),
  sessions: (query: string, archived = false) =>
    call<Project[]>('GET', `/api/sessions?q=${encodeURIComponent(query)}${archived ? '&include_archived=true' : ''}`),
  transcript: (id: string) => call<Transcript>('GET', `/api/sessions/${id}`),
  /** Start a live session, in `workspace` when given; it answers the new session's conversation ID. */
  newSession: (workspace?: string) =>
    call<Opened>('POST', '/api/sessions/new', workspace === undefined ? undefined : { workspace }),
  /** Make a saved session live, or find the live one; it answers its conversation ID. */
  open: (id: string) => call<Opened>('POST', `/api/sessions/${id}/open`),
  rename: (id: string, title: string) => call<void>('POST', `/api/sessions/${id}/rename`, { title }),
  fork: (id: string) => call<{ id: string }>('POST', `/api/sessions/${id}/fork`),
  archive: (id: string) => call<void>('POST', `/api/sessions/${id}/archive`),
  unarchive: (id: string) => call<void>('POST', `/api/sessions/${id}/unarchive`),
  remove: (id: string) => call<void>('DELETE', `/api/sessions/${id}`),
  /** Answer a settings menu's dialog, which belongs to the page rather than to one session. */
  answerDialog: (id: string, value: string | null) => call<void>('POST', `/api/dialogs/${id}`, { value }),
  plugins: () => call<PluginRow[]>('GET', '/api/plugins'),
  plugin: (name: string, action: PluginAction) => call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/${action}`),
  pluginSettings: (name: string, changes: Record<string, JsonValue>, reset: string[] = []) =>
    call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/settings`, { changes, reset }),
  pluginMenu: (name: string) => call<void>('POST', `/api/plugins/${encodeURIComponent(name)}/menu`),
  task: (id: string) => call<TaskTranscript>('GET', `/api/tasks/${id}`),
  taskAction: (id: string, action: 'stop' | 'background' | 'resume') => call<void>('POST', `/api/tasks/${id}/${action}`),
  steer: (id: string, text: string) => call<{ steered: boolean }>('POST', `/api/tasks/${id}/steer`, { text }),
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

/** Start a workspace's CLAI if needed and open its own page in a new tab. */
export async function openWorkspacePage(id: string): Promise<void> {
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
export async function interrupt(onError: (message: string) => void, stop: () => Promise<unknown>): Promise<void> {
  try {
    await stop()
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  }
}

export type Connection = 'connecting' | 'open' | 'closed' | 'expired'

/** The open WebSocket: `watch` names the live session the page shows, so only its messages come. */
export interface Socket {
  watch: (id: string) => void
  close: () => void
}

/** Keep one WebSocket open, reconnecting with backoff; each connection starts with a snapshot.

Each new connection watches `watching()` again, after calling `onWatch` with it, before any of its messages arrive.
A CLAI restarted since this page loaded has a new token: after a few failed attempts this checks, and
reports `expired` instead of retrying forever.
 */
export function connect(
  onMessage: (message: ServerMessage) => void,
  onState: (state: Connection) => void,
  watching: () => string | null,
  onWatch: (id: string) => void,
): Socket {
  let socket: WebSocket | null = null
  // The session this connection watches, so asking again for the same one sends nothing.
  let watched: string | null = null
  const send = (id: string) => {
    if (socket?.readyState !== WebSocket.OPEN || id === watched) return
    watched = id
    const message: WatchMessage = { type: 'watch', sessions: [id] }
    socket.send(JSON.stringify(message))
  }
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
      watched = null
      const id = watching()
      if (id !== null) {
        onWatch(id)
        send(id)
      }
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
  return {
    watch: send,
    close: () => {
      stopped = true
      window.clearTimeout(timer)
      socket?.close()
    },
  }
}
