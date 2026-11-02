import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react'
import { ApiError, api } from '../api'
import { MOD } from '../platform'
import { Logo } from './Logo'
import type { LiveEntry, Project, SessionEntry, TaskEntry, ThemeInfo } from '../protocol'
import {
  REMOVED,
  WORKING,
  ambiguous,
  flatten,
  isFlags,
  isList,
  isView,
  isUnread,
  loadSeen,
  markSeen,
  sections,
  short,
  status,
  stored,
  type Row,
  type Section,
  type Status,
  type View,
  keep,
} from '../sidebarModel'
import { useNow } from '../workers'
import { WorkersTree } from './WorkersTree'
import {
  ArchiveIcon,
  ChevronIcon,
  ComposeIcon,
  FolderIcon,
  MoonIcon,
  MoreIcon,
  OrchestratorIcon,
  PaletteIcon,
  GearIcon,
  PuzzleIcon,
  SearchIcon,
  SidebarIcon,
  SunIcon,
} from './icons'

export interface SidebarHandle {
  focusSearch: () => void
  /** Focus the first row of the Workers tree; `false` when the session has no delegated tasks. */
  focusWorkers: () => boolean
  /** Switch to the sessions that need you, or back to the view before. */
  toggleAttention: () => void
}

interface Props {
  version: number
  /** The live session the main pane shows. */
  shown: string | null
  /** The sessions live in this CLAI, by conversation ID, whose rows show what each is doing. */
  live: Record<string, LiveEntry>
  viewing: string | null
  /** The session being opened: its row is highlighted at once, as the selection. */
  opening: string | null
  theme: 'dark' | 'light'
  connected: boolean
  onTheme: () => void
  themeInfo: ThemeInfo | null
  onPickTheme: (name: string) => void
  /** Show a session, such as a fork just made: live, or read-only when CLAI cannot open it. A row passes all it knows. */
  onView: (session: Pick<SessionEntry, 'id' | 'title'> & Partial<Pick<SessionEntry, 'workspace' | 'openable'>>) => void
  /** Start a new session, asking where it runs when sessions run in their own directories. */
  onNew: () => void
  /** Whether the main pane shows the orchestrator chat, and whether its turn runs. */
  orchestrating: boolean
  orchestratorRunning: boolean
  onOrchestrator: () => void
  onCollapse: () => void
  onPlugins: () => void
  onSettings: () => void
  onError: (message: string) => void
  /** The current session's delegated tasks, shown under it. */
  tasks: TaskEntry[]
  /** The worker open in the main pane. */
  worker: string | null
  onWorker: (id: string) => void
}

export function ago(iso: string): string {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`
  if (seconds < 86400 * 7) return `${Math.floor(seconds / 86400)}d`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

const VIEWS: { view: View; label: string; title: string }[] = [
  { view: 'projects', label: 'Projects', title: 'Sessions by project' },
  { view: 'recent', label: 'Recent', title: 'Every project, newest first' },
  { view: 'attention', label: 'Attention', title: `Running, waiting for you, or unread (${MOD}Alt+U)` },
  { view: 'archived', label: 'Archived', title: 'Archived sessions; nothing archived is deleted' },
]

const LABELS: Record<Exclude<Status, null>, string> = {
  running: 'Running',
  queued: 'Running, with input queued',
  asking: 'Needs input',
  busy: 'Running in another CLAI',
  failed: 'Last turn failed',
  cancelled: 'Last turn interrupted',
  unread: 'Unread',
}

/** A row's one glyph: a spinner while it runs here, a ring while another CLAI runs it, else a dot or nothing. */
function Glyph({ state, pid }: { state: Status; pid: number | null }) {
  if (state === null) return <span className="state" />
  const label = state === 'busy' && pid !== null ? `Running in clai2 pid ${pid}` : LABELS[state]
  if (state === 'running' || state === 'queued') {
    return <span className={`spinner row-spinner ${state === 'queued' ? 'with-queue' : ''}`} role="img" aria-label={label} title={label} />
  }
  return <span className={`state ${state}`} role="img" aria-label={label} title={label} />
}

/** The most urgent glyph among a folded section's rows, so it still shows that one works or waits. */
function strongest(states: Status[]): Status {
  return WORKING.find((state) => states.includes(state)) ?? null
}

const SEARCH_KEYS = new Set(['ArrowDown', 'Enter'])

/**
 * Saved sessions, by project or newest first, in an order only creation changes: activity, renames, polling, and
 * selection never move a row. Each row has one status glyph, and folders that are gone fold away at the bottom.
 */
export const Sidebar = forwardRef<SidebarHandle, Props>(function Sidebar(
  {
    version,
    shown,
    live,
    viewing,
    opening,
    theme,
    connected,
    onTheme,
    themeInfo,
    onPickTheme,
    onView,
    onNew,
    orchestrating,
    orchestratorRunning,
    onOrchestrator,
    onCollapse,
    onPlugins,
    onSettings,
    onError,
    tasks,
    worker,
    onWorker,
  },
  ref,
) {
  const [query, setQuery] = useState('')
  const [view, setView] = useState<View>(() => stored('clai-sidebar-view', 'projects', isView))
  const previous = useRef<View>('projects')
  // The list paints from the last one fetched while it is fetched again.
  const [projects, setProjects] = useState<Project[]>(() => stored('clai-sidebar-cache', [], isList))
  /** The search the listed sessions answer, so Enter never opens a result of an earlier one. */
  const [answered, setAnswered] = useState('')
  /** The search whose first result Enter asked to open before its results arrived. */
  const openFirst = useRef<string | null>(null)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() =>
    stored('clai-sidebar-collapsed', {}, isFlags),
  )
  const [menu, setMenu] = useState<string | null>(null)
  const [menuUp, setMenuUp] = useState(false)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [unsaved, setUnsaved] = useState(false)
  const [seen, setSeen] = useState(loadSeen)
  const search = useRef<HTMLInputElement>(null)
  const aside = useRef<HTMLElement>(null)
  const archived = view === 'archived'

  useEffect(() => keep('clai-sidebar-view', view), [view])
  useEffect(() => keep('clai-sidebar-collapsed', collapsed), [collapsed])

  /** The session picked from Attention, which stays listed there while it opens and shows. */
  const pickedHere = useRef<string | null>(null)
  // What was waiting to open or stay listed belongs to the view the user has left.
  useEffect(() => {
    openFirst.current = null
    pickedHere.current = null
    setMenu(null)
  }, [view])

  const pick = useCallback((next: View) => {
    setView((current) => {
      if (current !== next) previous.current = current
      return next
    })
  }, [])

  useImperativeHandle(
    ref,
    () => ({
      focusSearch: () => search.current?.focus(),
      focusWorkers: () => {
        const first = aside.current?.querySelector<HTMLElement>('.workers .worker')
        first?.scrollIntoView({ block: 'nearest' })
        first?.focus()
        return Boolean(first)
      },
      toggleAttention: () =>
        setView((current) => {
          if (current === 'attention') return previous.current
          previous.current = current
          return 'attention'
        }),
    }),
    [],
  )

  const loading = useRef(0)
  const load = useCallback(async () => {
    // A search and the periodic refresh can overlap; only the latest request's answer is shown.
    const id = ++loading.current
    try {
      const found = await api.sessions(query, archived)
      if (id !== loading.current) return
      setProjects(found)
      setAnswered(query)
      if (!query && !archived) keep('clai-sidebar-cache', found)
    } catch (error) {
      // An Enter waiting on this search's results is dropped with them, so a later refresh opens nothing by itself.
      if (openFirst.current === query) openFirst.current = null
      if (error instanceof ApiError && error.status === 404) setUnsaved(true)
      else onError(error instanceof Error ? error.message : String(error))
    }
  }, [query, archived, onError])

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), query ? 150 : 0)
    return () => window.clearTimeout(timer)
  }, [load, version, query])

  // Other CLAIs save sessions too; list them again now and then. Rows keep their places, so this moves nothing.
  useEffect(() => {
    if (unsaved) return
    const interval = window.setInterval(() => void load(), 15000)
    return () => window.clearInterval(interval)
  }, [load, unsaved])

  useEffect(() => {
    if (!menu) return
    const close = (event: Event) => {
      if (event instanceof KeyboardEvent && event.key !== 'Escape') return
      if (event instanceof MouseEvent && (event.target as HTMLElement).closest('.session-menu, .session-more')) return
      event.preventDefault()
      setMenu(null)
    }
    // Capture phase, so Esc closes the menu before the page's own Esc handler can interrupt a turn.
    window.addEventListener('keydown', close, true)
    window.addEventListener('mousedown', close)
    return () => {
      window.removeEventListener('keydown', close, true)
      window.removeEventListener('mousedown', close)
    }
  }, [menu])

  useEffect(() => {
    // Typing on, or clearing the search, drops the request; only its own results open.
    if (openFirst.current !== query) openFirst.current = null
    if (openFirst.current === null || answered !== query) return
    openFirst.current = null
    aside.current?.querySelector<HTMLElement>('.projects .session-main')?.click()
  }, [projects, answered, query])

  const rows = useMemo(() => flatten(projects), [projects])
  // The row picked last is the selection, highlighted at once, before its session opens.
  const selected = opening ?? viewing ?? shown

  // What the shown or previewed session says is read as it changes, unless the orchestrator, a worker, or an opening covers it.
  const readRow =
    orchestrating || worker !== null || opening !== null
      ? undefined
      : rows.find((row) => row.id === (viewing ?? shown))
  useEffect(() => {
    if (readRow) setSeen((current) => markSeen(current, [readRow]))
  }, [readRow])

  // Unread is kept apart from the glyph, so a failed turn you have not seen is both red and unread.
  const unread = useMemo(
    () => new Set(rows.filter((row) => row.id !== readRow?.id && isUnread(row, seen)).map((row) => row.id)),
    [rows, readRow, seen],
  )
  const states = useMemo(() => {
    const found: Record<string, Status> = {}
    for (const row of rows) found[row.id] = status(row, live, unread.has(row.id))
    return found
  }, [rows, live, unread])
  const needsYou = useCallback(
    (row: Row) => unread.has(row.id) || WORKING.includes(states[row.id] ?? null),
    [states, unread],
  )
  // The row picked from Attention stays there while it opens and shows, though it no longer needs you.
  const shownSections = useMemo(
    () => sections(rows, view, (row) => (row.id === selected && row.id === pickedHere.current) || needsYou(row)),
    [rows, view, selected, needsYou],
  )
  const attention = useMemo(() => rows.filter((row) => !row.archived && needsYou(row)).length, [rows, needsYou])

  // Each running row's elapsed time: from its turn's start, which a turn saves, or else from when this page saw it run.
  // A queued prompt starting its turn, or an answer resuming one, starts the count again, though the row never idled.
  const started = useRef(new Map<string, { at: number; after: number }>())
  const lastStates = useRef(new Map<string, Status>())
  const anyWorking = rows.some((row) => WORKING.includes(states[row.id] ?? null))
  const now = useNow(anyWorking)
  for (const row of rows) {
    const state = states[row.id] ?? null
    const last = lastStates.current.get(row.id)
    lastStates.current.set(row.id, state)
    if (!WORKING.includes(state)) {
      started.current.delete(row.id)
      continue
    }
    let turn = started.current.get(row.id)
    if (turn === undefined || ((last === 'queued' || last === 'asking') && state === 'running')) {
      // Saves before `after` belong to an earlier turn.
      turn = { at: Date.now(), after: turn?.at ?? -Infinity }
      started.current.set(row.id, turn)
    }
    const saved = Date.parse(row.updated_at)
    if ((row.state === 'running' || row.state === 'busy') && saved > turn.after) turn.at = Math.min(turn.at, saved)
  }

  const act = async (action: () => Promise<unknown>) => {
    setMenu(null)
    try {
      await action()
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  /** Whether a row can be archived: not the open session, nor one running here or elsewhere. */
  const archivable = (row: Row) => row.id !== shown && !(row.id in live) && !WORKING.includes(states[row.id] ?? null)
  // A search lists only a group's matching sessions, so it cannot archive all of them.
  const archiveAll = (found: Row[]) =>
    act(() => api.archiveAll(found.filter((row) => !row.archived && archivable(row)).map((row) => row.id)))
  /** Why archiving these at once is off: a search lists only some of them, or none can be archived. */
  const bulkRefusal = (found: Row[]): string | null =>
    query
      ? 'Clear the search to archive them all'
      : found.some((row) => !row.archived && archivable(row))
        ? null
        : 'No session here can be archived'
  const markRead = (found: Row[]) => {
    setMenu(null)
    setSeen((current) => markSeen(current, found))
  }

  const toggle = (key: string, open: boolean) => setCollapsed((current) => ({ ...current, [key]: !open }))
  const isClosed = (section: Section) =>
    section.kind !== 'list' && !query && (collapsed[section.key] ?? section.kind === 'removed')

  const openMenu = (key: string, target: HTMLElement) => {
    // The list scrolls and clips the menu, so a row in its lower half opens the menu upwards.
    const list = target.closest('.projects')?.getBoundingClientRect()
    const rect = target.getBoundingClientRect()
    setMenuUp(list !== undefined && rect.bottom > list.top + list.height / 2)
    setMenu(menu === key ? null : key)
  }

  const row = (session: Row, section: Section, twins: Set<string>) => {
    const state = states[session.id] ?? null
    const current = session.id === selected
    const working = WORKING.includes(state)
    const chip = section.kind === 'project' ? (section.branches ? session.branch : null) : session.projectName
    // Branch labels are unique only within a project, and Removed worktrees spans every project.
    const siblings = section.rows.filter(
      (other) => other.project === session.project && other.branch === session.branch && !other.archived,
    )
    const begun = started.current.get(session.id)?.at
    const time = working && begun !== undefined ? short(Math.max(0, (now - begun) / 1000)) : ago(session.updated_at)
    return (
      <div
        key={session.id}
        className={`session ${current ? 'current' : ''} ${unread.has(session.id) ? 'unread' : ''} ${session.archived ? 'archived' : ''} ${session.missing ? 'missing' : ''}`}
        data-session={session.id}
        aria-busy={session.id === opening || undefined}
      >
        {renaming === session.id ? (
          <input
            className="rename"
            autoFocus
            defaultValue={session.title}
            onBlur={() => setRenaming(null)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                setRenaming(null)
              }
              if (event.key === 'Enter') {
                const title = event.currentTarget.value
                setRenaming(null)
                void act(() => api.rename(session.id, title))
              }
            }}
          />
        ) : (
          <button
            className="session-main"
            onClick={() => {
              // A row picked by hand wins over a search's first result still on its way.
              openFirst.current = null
              pickedHere.current = view === 'attention' ? session.id : null
              onView(session)
            }}
            title={`${session.title}\n${session.workspace}`}
            aria-current={current ? 'true' : undefined}
          >
            {/* A row that needs input says so with its pill instead. */}
            <Glyph state={state === 'asking' ? null : state} pid={session.owner_pid} />
            <span className="session-title">{session.title}</span>
            {chip && <span className="session-chip">{chip}</span>}
            {twins.has(session.id) && <span className="session-chip mono">#{session.id.slice(0, 6)}</span>}
            {/* The pill stays on hover, which hides the time to make room for the menu button. */}
            {state === 'asking' ? (
              <span className="needs-input">Needs input</span>
            ) : (
              <span className="session-time">{time}</span>
            )}
          </button>
        )}
        <button
          className="session-more"
          onClick={(event) => openMenu(session.id, event.currentTarget)}
          title="Session actions"
          aria-label="Session actions"
        >
          <MoreIcon size={14} />
        </button>
        {menu === session.id && (
          <div className={`menu session-menu ${menuUp ? 'up' : ''}`} role="menu">
            <button onClick={() => (setMenu(null), setRenaming(session.id))}>Rename</button>
            <button onClick={() => void act(async () => onView({ id: (await api.fork(session.id)).id, title: session.title, workspace: session.workspace }))}>
              Fork
            </button>
            {unread.has(session.id) && <button onClick={() => markRead([session])}>Mark as read</button>}
            {session.archived ? (
              <button onClick={() => void act(() => api.unarchive(session.id))}>Unarchive</button>
            ) : (
              <>
                <button
                  disabled={!archivable(session)}
                  title={
                    session.id === shown || session.id in live
                      ? 'A session open in CLAI cannot be archived'
                      : !archivable(session)
                        ? 'A running session cannot be archived'
                        : 'Hide it; nothing is deleted'
                  }
                  onClick={() => void act(() => api.archive(session.id))}
                >
                  Archive
                </button>
                {section.branches && siblings.length > 1 && (
                  <button
                    disabled={bulkRefusal(siblings) !== null}
                    onClick={() => void archiveAll(siblings)}
                    title={bulkRefusal(siblings) ?? 'Archive every session of this branch you can'}
                  >
                    Archive all in {session.branch}
                  </button>
                )}
              </>
            )}
            <button
              className="danger"
              disabled={session.id in live || session.id === shown}
              title={session.id in live || session.id === shown ? 'A session open in CLAI cannot be deleted' : undefined}
              onClick={() =>
                window.confirm(`Delete "${session.title}"? This cannot be undone.`) &&
                void act(() => api.remove(session.id))
              }
            >
              Delete
            </button>
          </div>
        )}
      </div>
    )
  }

  // The tree sits under the current session, or above the list while that row is hidden.
  const listed = shownSections.some((section) => !isClosed(section) && section.rows.some((row) => row.id === shown))
  const workers = tasks.length > 0 && <WorkersTree key="workers" tasks={tasks} selected={worker} onOpen={onWorker} />

  const heading = (section: Section, closed: boolean): ReactNode => {
    if (section.kind === 'list') return null
    const key = `section:${section.key}`
    const unreadRows = section.rows.filter((row) => unread.has(row.id))
    const removed = section.key === REMOVED
    return (
      <div className="project-head">
        <button
          className="project-title"
          onClick={() => toggle(section.key, closed)}
          aria-expanded={!closed}
          title={removed ? 'Sessions whose folder is gone' : section.key.includes('\0') ? undefined : section.key}
        >
          {removed ? <ArchiveIcon size={14} /> : <FolderIcon size={14} />}
          <span className="project-name">{section.name}</span>
          {closed && <Glyph state={strongest(section.rows.map((row) => states[row.id] ?? null))} pid={null} />}
          <ChevronIcon size={12} className={`chevron ${closed ? '' : 'down'}`} />
        </button>
        {!archived && (
          <button
            className="session-more"
            onClick={(event) => openMenu(key, event.currentTarget)}
            title={`${section.name} actions`}
            aria-label={`${section.name} actions`}
            >
            <MoreIcon size={14} />
          </button>
        )}
        {menu === key && (
          <div className={`menu session-menu ${menuUp ? 'up' : ''}`} role="menu">
            <button
              disabled={bulkRefusal(section.rows) !== null}
              onClick={() => void archiveAll(section.rows)}
              title={bulkRefusal(section.rows) ?? 'Archive every session here you can'}
            >
              Archive chats
            </button>
            {unreadRows.length > 0 && <button onClick={() => markRead(unreadRows)}>Mark all as read</button>}
          </div>
        )}
      </div>
    )
  }

  /** Up and Down move between headings and rows, Right and Left open and fold a project, Enter opens. */
  const navigate = (event: ReactKeyboardEvent<HTMLElement>) => {
    const target = event.target as HTMLElement
    const items = [...(aside.current?.querySelectorAll<HTMLElement>('.projects .project-title, .projects .session-main') ?? [])]
    const index = items.indexOf(target)
    if (index < 0) return
    const step = { ArrowDown: 1, ArrowUp: -1 }[event.key]
    if (step !== undefined) {
      event.preventDefault()
      const next = index + step
      if (next < 0) search.current?.focus()
      else items[Math.min(next, items.length - 1)]?.focus()
      return
    }
    const expanded = target.getAttribute('aria-expanded')
    if (event.key === 'ArrowRight' && expanded === 'false') {
      event.preventDefault()
      target.click()
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      if (expanded === 'true') target.click()
      else target.closest('section')?.querySelector<HTMLElement>('.project-title')?.focus()
    }
  }

  const empty = unsaved
    ? 'This CLAI does not save sessions'
    : query
      ? 'No matching sessions'
      : view === 'attention'
        ? 'Nothing needs you'
        : archived
          ? 'No archived sessions'
          : 'No saved sessions yet'

  return (
    <aside ref={aside} className={`sidebar ${orchestrating ? 'orchestrating' : ''}`}>
      <div className="sidebar-top">
        <div className="brand">
          <Logo />
          <span className="brand-product">CLAI</span>
          <span className={`connection ${connected ? 'on' : ''}`} title={connected ? 'Connected' : 'Reconnecting…'} />
        </div>
        <button className="icon-button" onClick={onCollapse} title="Hide sidebar">
          <SidebarIcon size={16} />
        </button>
      </div>
      <button className="new-session" onClick={onNew} title={`New session (${MOD}Shift+O)`}>
        <ComposeIcon size={16} />
        <span>New session</span>
      </button>
      <button
        className={`new-session orchestrator-entry ${orchestrating ? 'active' : ''}`}
        onClick={onOrchestrator}
        title="Orchestrator: an agent that runs workspaces for you"
        aria-pressed={orchestrating}
      >
        <OrchestratorIcon size={16} />
        <span>Orchestrator</span>
        {orchestratorRunning && <span className="state running" aria-label="Working" />}
      </button>
      <label className="search">
        <SearchIcon size={14} />
        <input
          ref={search}
          value={query}
          placeholder="Search sessions"
          aria-label="Search sessions"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.preventDefault()
              setQuery('')
            } else if (SEARCH_KEYS.has(event.key)) {
              // Down moves to the first result; Enter opens it, as a command palette does.
              const first = aside.current?.querySelector<HTMLElement>('.projects .session-main')
              if (event.key === 'Enter' && answered !== query) {
                // The results for what was typed are still on their way: open the first once they arrive.
                event.preventDefault()
                openFirst.current = query
              } else if (first) {
                event.preventDefault()
                if (event.key === 'Enter') first.click()
                else first.focus()
              }
            }
          }}
        />
        <kbd>{MOD}K</kbd>
      </label>
      <div className="sidebar-views" role="group" aria-label="Sessions view">
        {VIEWS.map((item) => (
          <button
            key={item.view}
            className={view === item.view ? 'on' : ''}
            aria-pressed={view === item.view}
            title={item.title}
            onClick={() => pick(view === item.view && item.view !== 'projects' ? 'projects' : item.view)}
          >
            {item.label}
            {item.view === 'attention' && attention > 0 && <span className="view-count">{attention}</span>}
          </button>
        ))}
      </div>
      <nav className="projects" onKeyDown={navigate}>
        {!listed && workers}
        {shownSections.length === 0 && <div className="empty-list">{empty}</div>}
        {shownSections.map((section) => {
          const closed = isClosed(section)
          const twins = ambiguous(section.rows)
          return (
            <section key={section.key} className={`project ${section.kind}`}>
              {heading(section, closed)}
              {!closed && section.rows.flatMap((session) => [row(session, section, twins), session.id === shown && workers])}
            </section>
          )
        })}
      </nav>
      <div className="sidebar-foot">
        {themeInfo && (
          <label className="theme-picker" title="The /theme palette, shared with the terminal">
            <PaletteIcon size={14} />
            <select value={themeInfo.name} onChange={(event) => onPickTheme(event.target.value)} aria-label="Theme">
              {themeInfo.names.map((name) => (
                <option key={name} value={name}>
                  {name === 'default' ? 'Pydantic' : name.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="foot-actions">
          <button className="icon-button" onClick={onSettings} title={`Settings (${MOD},)`} aria-label="Settings">
            <GearIcon size={16} />
          </button>
          <button className="icon-button" onClick={onPlugins} title="Plugins" aria-label="Plugins">
            <PuzzleIcon size={16} />
          </button>
          {!themeInfo?.palette && (
            <button className="icon-button" onClick={onTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
              {theme === 'dark' ? <SunIcon size={16} /> : <MoonIcon size={16} />}
            </button>
          )}
        </div>
      </div>
    </aside>
  )
})
