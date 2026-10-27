import { createContext, useEffect, useState } from 'react'
import { duration } from './format'
import type { Item, TaskEntry } from './protocol'

/** The workers a transcript's tool cards link to: those its own run started. */
export interface WorkerLinks {
  tasks: TaskEntry[]
  /** The worker whose transcript this is; `null` for the session's own. */
  parent: string | null
  open: (id: string) => void
}

export const WorkersContext = createContext<WorkerLinks>({ tasks: [], parent: null, open: () => undefined })

/** Each task under its parent, in the order they started. */
export function tree(tasks: TaskEntry[]): { task: TaskEntry; depth: number }[] {
  const ids = new Set(tasks.map((task) => task.id))
  const rows: { task: TaskEntry; depth: number }[] = []
  const visit = (task: TaskEntry, depth: number) => {
    rows.push({ task, depth })
    for (const child of tasks) if (child.parent_id === task.id) visit(child, depth + 1)
  }
  for (const task of tasks) if (task.parent_id === null || !ids.has(task.parent_id)) visit(task, 0)
  return rows
}

/** The current time, ticking each second while `ticking`, for elapsed times. */
export function useNow(ticking: boolean): number {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!ticking) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [ticking])
  return now
}

export function elapsed(task: TaskEntry, now: number): string {
  const end = task.finished_at ? new Date(task.finished_at).getTime() : now
  return duration(Math.max(0, (end - new Date(task.started_at).getTime()) / 1000))
}

/** `anthropic:claude-sonnet-4-5` as `claude-sonnet-4-5`; a model's own name, without a provider, as it is. */
export function shortModel(model: string): string {
  return model.slice(model.indexOf(':') + 1).replace(/^:+/, '') || model
}

/** What a task is doing now, or how it ended. */
export function state(task: TaskEntry): string {
  if (task.status === 'running') return task.activity
  return task.outcome === 'ok' ? 'done' : (task.outcome ?? 'finished')
}

function delegated(name: string, args: string): string | null {
  if (name !== 'delegate_task') return null
  try {
    const value: unknown = JSON.parse(args)
    const task = value && typeof value === 'object' ? (value as Record<string, unknown>).task : null
    return typeof task === 'string' ? task : null
  } catch {
    return null
  }
}

/** The workers a tool call started: named by ID in its result, or, until it has one, given the task it delegates. */
export function startedBy(item: Item, links: WorkerLinks): TaskEntry[] {
  const calls = [item, ...item.calls]
  const results = calls.map((call) => call.text).join('\n')
  // A settled call names its task, so two calls delegating the same text each link only their own.
  const waiting = calls.filter((call) => !call.text.includes('Task '))
  const prompts = new Set(waiting.map((call) => delegated(call.name, call.args)).filter((task) => task !== null))
  if (!prompts.size && !results.includes('Task ')) return []
  return links.tasks.filter(
    (task) => task.parent_id === links.parent && (results.includes(task.id) || prompts.has(task.prompt)),
  )
}
