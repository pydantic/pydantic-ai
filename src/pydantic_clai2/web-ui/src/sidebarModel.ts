import type { LiveEntry, Project, SessionEntry } from './protocol'

/**
 * How the sidebar lists sessions: by project, newest first across projects, only those that need you, or archived.
 *
 * Every view orders rows by when each session was created, so nothing moves on selection, activity, renames, or polls.
 */
export type View = 'projects' | 'recent' | 'attention' | 'archived'

/** A saved session with where it lives, flattened out of the server's project and worktree groups. */
export interface Row extends SessionEntry {
  project: string
  projectName: string
  /** Its worktree's branch, or folder name outside Git. */
  branch: string
  /** Its folder is gone, as with a removed worktree. */
  missing: boolean
}

/** What a row's one glyph slot shows; `null` for a session that is idle and read. */
export type Status = 'running' | 'queued' | 'asking' | 'busy' | 'failed' | 'cancelled' | 'unread' | null

/** The statuses of a turn running, here or in another CLAI. */
export const WORKING: readonly Status[] = ['asking', 'queued', 'running', 'busy']

/** A heading and its rows; `removed` collects the sessions whose folder is gone, folded by default. */
export interface Section {
  key: string
  name: string
  rows: Row[]
  /** Whether the section's rows show their branch, because they come from more than one. */
  branches: boolean
  kind: 'project' | 'list' | 'removed'
}

export const REMOVED = '\0removed'
const LIST = '\0list'

/** Newest created first; the ID breaks ties, so equal times never swap. */
function byCreation(a: Row, b: Row): number {
  return Date.parse(b.created_at) - Date.parse(a.created_at) || a.id.localeCompare(b.id)
}

/** The server's projects as one list of rows, newest created first. */
export function flatten(projects: Project[]): Row[] {
  const rows = projects.flatMap((project) =>
    project.worktrees.flatMap((worktree) =>
      worktree.sessions.map((session) => ({
        ...session,
        project: project.key,
        projectName: project.name,
        branch: worktree.label,
        missing: worktree.missing,
      })),
    ),
  )
  return rows.sort(byCreation)
}

/** What a row's glyph shows: this CLAI's live state first, then the saved one, then whether it is unread. */
export function status(row: Row, live: Record<string, LiveEntry>, unread: boolean): Status {
  const state = live[row.id]?.state
  if (state !== undefined && state !== 'idle') return state
  // An idle live session's row may still say `running` until the list is fetched again.
  const saved = state === 'idle' && row.state === 'running' ? 'ready' : row.state
  if (saved === 'running' || saved === 'busy' || saved === 'failed' || saved === 'cancelled') return saved
  return unread ? 'unread' : null
}

/** The sections a view shows, in an order that depends only on names and creation times. */
export function sections(rows: Row[], view: View, needsYou: (row: Row) => boolean): Section[] {
  const listed = rows.filter((row) => (view === 'archived' ? row.archived : !row.archived))
  if (view === 'attention') {
    const waiting = listed.filter(needsYou)
    return waiting.length ? [list(waiting)] : []
  }
  const present = listed.filter((row) => !row.missing)
  const removed = listed.filter((row) => row.missing)
  const shown = view === 'projects' ? byProject(present) : present.length ? [list(present)] : []
  if (removed.length) {
    shown.push({ key: REMOVED, name: `Removed worktrees (${removed.length})`, rows: removed, branches: true, kind: 'removed' })
  }
  return shown
}

function list(rows: Row[]): Section {
  return { key: LIST, name: '', rows, branches: false, kind: 'list' }
}

/** Projects by name, so a project never moves; each keeps its rows newest created first. */
function byProject(rows: Row[]): Section[] {
  const found = new Map<string, Section>()
  for (const row of rows) {
    let section = found.get(row.project)
    if (section === undefined) {
      section = { key: row.project, name: row.projectName, rows: [], branches: false, kind: 'project' }
      found.set(row.project, section)
    }
    section.rows.push(row)
  }
  for (const section of found.values()) section.branches = new Set(section.rows.map((row) => row.branch)).size > 1
  return [...found.values()].sort(
    (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }) || a.key.localeCompare(b.key),
  )
}

/** The IDs of rows whose title another row of the list shares, which then show a short ID to tell them apart. */
export function ambiguous(rows: Row[]): Set<string> {
  const counts = new Map<string, number>()
  for (const row of rows) counts.set(row.title, (counts.get(row.title) ?? 0) + 1)
  return new Set(rows.filter((row) => (counts.get(row.title) ?? 0) > 1).map((row) => row.id))
}

/** `200` seconds as `3m`: a running row's elapsed time, short enough for its time slot. */
export function short(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  return `${Math.floor(seconds / 3600)}h`
}

/**
 * When this browser last saw each session's content, so one that changed since shows as unread.
 *
 * `since` is when the browser first kept track: sessions last active before it start out read.
 */
export interface Seen {
  since: string
  sessions: Record<string, string>
}

const SEEN_KEY = 'clai-sidebar-seen'
const SEEN_LIMIT = 1000

export function loadSeen(): Seen {
  try {
    const saved = JSON.parse(localStorage.getItem(SEEN_KEY) ?? 'null') as Seen | null
    if (saved && typeof saved.since === 'string' && typeof saved.sessions === 'object' && saved.sessions !== null) {
      return saved
    }
  } catch {
    // A corrupt record starts over.
  }
  const fresh = { since: new Date().toISOString(), sessions: {} }
  keep(SEEN_KEY, fresh)
  return fresh
}

/** `seen` with these sessions marked read as of their last activity; the newest records are kept. */
export function markSeen(seen: Seen, rows: Pick<Row, 'id' | 'updated_at'>[]): Seen {
  const changed = rows.filter((row) => seen.sessions[row.id] !== row.updated_at)
  if (!changed.length) return seen
  const sessions = { ...seen.sessions, ...Object.fromEntries(changed.map((row) => [row.id, row.updated_at])) }
  const kept = Object.entries(sessions)
    .sort(([, a], [, b]) => Date.parse(b) - Date.parse(a))
    .slice(0, SEEN_LIMIT)
  const next = { since: seen.since, sessions: Object.fromEntries(kept) }
  keep(SEEN_KEY, next)
  return next
}

/** Whether a row's content changed since this browser last saw it. */
export function isUnread(row: Row, seen: Seen): boolean {
  return row.messages > 0 && Date.parse(row.updated_at) > Date.parse(seen.sessions[row.id] ?? seen.since)
}

/**
 * A small JSON value kept in `localStorage`, falling back when it is missing, unreadable, or not what `valid` accepts:
 * a value of another shape, written by another version or app, must not break the page on every load.
 */
export function stored<T>(key: string, fallback: T, valid: (value: unknown) => value is T): T {
  try {
    const saved = localStorage.getItem(key)
    const value: unknown = saved === null ? fallback : JSON.parse(saved)
    return valid(value) ? value : fallback
  } catch {
    return fallback
  }
}

const VIEW_NAMES: readonly View[] = ['projects', 'recent', 'attention', 'archived']

export const isView = (value: unknown): value is View => VIEW_NAMES.includes(value as View)

export const isList = (value: unknown): value is Project[] => Array.isArray(value)

export const isFlags = (value: unknown): value is Record<string, boolean> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** Save a small JSON value in `localStorage`; a full or disabled store only loses it. */
export function keep(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Nothing to do: the page works without it.
  }
}
