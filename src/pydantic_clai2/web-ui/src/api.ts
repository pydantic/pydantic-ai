import type { AnswerBody, CompletionInfo, Project, ServerMessage, Transcript } from './protocol'

// The page's token: the server puts it in a meta tag; `npm run dev` serves its own page, so the address has it.
const token =
  document.querySelector<HTMLMetaElement>('meta[name="clai-token"]')?.content ??
  new URLSearchParams(location.search).get('token') ??
  ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { 'x-clai-token': token, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const data: unknown = await response.json()
      if (data && typeof data === 'object' && 'detail' in data) {
        const value = (data as { detail: unknown }).detail
        detail = typeof value === 'string' ? value : JSON.stringify(value)
      }
    } catch {
      // Not JSON: keep the status text.
    }
    throw new ApiError(detail, response.status)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export const api = {
  send: (text: string) => call<void>('POST', '/api/input', { text }),
  interrupt: () => call<{ interrupted: boolean }>('POST', '/api/interrupt'),
  complete: (text: string) => call<CompletionInfo[]>('GET', `/api/complete?text=${encodeURIComponent(text)}`),
  models: () => call<string[]>('GET', '/api/models'),
  selectModel: (name: string) => call<void>('POST', '/api/model', { name }),
  answer: (id: string, answers: AnswerBody[] | null) => call<void>('POST', `/api/questions/${id}`, { answers }),
  sessions: (query: string) => call<Project[]>('GET', `/api/sessions?q=${encodeURIComponent(query)}`),
  transcript: (id: string) => call<Transcript>('GET', `/api/sessions/${id}`),
  newSession: () => call<void>('POST', '/api/sessions/new'),
  open: (id: string) => call<void>('POST', `/api/sessions/${id}/open`),
  rename: (id: string, title: string) => call<void>('POST', `/api/sessions/${id}/rename`, { title }),
  fork: (id: string) => call<{ id: string }>('POST', `/api/sessions/${id}/fork`),
  remove: (id: string) => call<void>('DELETE', `/api/sessions/${id}`),
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
      delay = Math.min(delay * 2, 8000)
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
