import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { ApiError, api } from '../api'
import { MOD } from '../platform'
import { Logo } from './Logo'
import type { LiveEntry, Project, SessionEntry, TaskEntry, ThemeInfo, Worktree } from '../protocol'
import { WorkersTree } from './WorkersTree'
import {
  ArchiveIcon,
  BranchIcon,
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
}

interface Props {
  version: number
  /** The live session the main pane shows. */
  shown: string | null
  /** The sessions live in this CLAI, by conversation ID, whose rows show what each is doing. */
  live: Record<string, LiveEntry>
  viewing: string | null
  /** The session being opened, which shows a spinner. */
  opening: string | null
  theme: 'dark' | 'light'
  connected: boolean
  onTheme: () => void
  themeInfo: ThemeInfo | null
  onPickTheme: (name: string) => void
  /** Show a session, such as a fork just made: live, or read-only when CLAI cannot open it. */
  onView: (session: Pick<SessionEntry, 'id' | 'title'>) => void
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

type Dot = SessionEntry['state'] | 'queued' | 'asking'

const DOT_LABELS: Partial<Record<Dot, string>> = {
  running: 'Running',
  queued: 'Running, with input queued',
  asking: 'Waiting for your answer',
  busy: 'Running in another CLAI',
  failed: 'Last turn failed',
  cancelled: 'Last turn interrupted',
}

/** What a session's dot shows: a live session's state from this CLAI, else what its saved row says. */
function dot(session: SessionEntry, live: Record<string, LiveEntry>): Dot {
  const state = live[session.id]?.state
  if (state === undefined) return session.state
  if (state !== 'idle') return state
  // An idle live session's row may still say `running` until the list is fetched again.
  return session.state === 'running' ? 'ready' : session.state
}

/** The dots of a session with a turn running, here or in another CLAI, the most urgent first. */
const WORKING: readonly Dot[] = ['asking', 'queued', 'running', 'busy']

/** The most urgent dot among these sessions, so a collapsed group still shows it; `null` when none works. */
function strongest(worktrees: Worktree[], live: Record<string, LiveEntry>): Dot | null {
  const found = new Set(worktrees.flatMap((worktree) => worktree.sessions.map((s) => dot(s, live))))
  return WORKING.find((state) => found.has(state)) ?? null
}

/** A collapsed group's dot, if a session in it works. */
function GroupDot({ state }: { state: Dot | null }) {
  return state === null ? null : <StateDot state={state} />
}

/** A session's status dot, named for screen readers; none for a session that is not doing anything. */
function StateDot({ state }: { state: Dot }) {
  const label = DOT_LABELS[state]
  return <span className={`state ${state}`} role={label ? 'img' : undefined} aria-label={label} title={label} />
}

/** Saved sessions grouped by project, then worktree, as `/resume` lists them, with search and per-session actions. */
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
  /** Whether archived sessions are listed too, muted, and searched. */
  const [archived, setArchived] = useState(false)
  const [projects, setProjects] = useState<Project[]>([])
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [menu, setMenu] = useState<string | null>(null)
  const [menuUp, setMenuUp] = useState(false)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [unsaved, setUnsaved] = useState(false)
  const search = useRef<HTMLInputElement>(null)
  const aside = useRef<HTMLElement>(null)

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
    }),
    [],
  )

  const loading = useRef(0)
  const load = useCallback(async () => {
    // A search and the periodic refresh can overlap; only the latest request's answer is shown.
    const id = ++loading.current
    try {
      const found = await api.sessions(query, archived)
      if (id === loading.current) setProjects(found)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) setUnsaved(true)
      else onError(error instanceof Error ? error.message : String(error))
    }
  }, [query, archived, onError])

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), query ? 150 : 0)
    return () => window.clearTimeout(timer)
  }, [load, version, query])

  // Other CLAIs save sessions too; list them again now and then.
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

  const act = async (action: () => Promise<unknown>) => {
    setMenu(null)
    try {
      await action()
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  const isShown = (session: SessionEntry) => session.id === shown

  const row = (session: SessionEntry) => {
    const state = dot(session, live)
    const current = isShown(session)
    const working = WORKING.includes(state)
    return (
      <div
        key={session.id}
        className={`session ${current ? 'current' : ''} ${viewing === session.id ? 'viewing' : ''} ${opening === session.id ? 'opening' : ''} ${session.archived ? 'archived' : ''}`}
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
          <button className="session-main" onClick={() => onView(session)} title={session.workspace}>
            <StateDot state={state} />
            <span className="session-title">{session.title}</span>
            {session.archived && <span className="worktree-tag">archived</span>}
            {opening === session.id && <span className="spinner" aria-label="Opening" />}
            <span className="session-time">
              {state === 'running' || state === 'queued' || state === 'asking' ? '' : ago(session.updated_at)}
            </span>
          </button>
        )}
        <button
          className="session-more"
          onClick={(event) => {
            // The list scrolls and clips the menu, so a row in its lower half opens the menu upwards.
            const list = event.currentTarget.closest('.projects')?.getBoundingClientRect()
            const rect = event.currentTarget.getBoundingClientRect()
            setMenuUp(list !== undefined && rect.bottom > list.top + list.height / 2)
            setMenu(menu === session.id ? null : session.id)
          }}
          title="Session actions"
          aria-label="Session actions"
        >
          <MoreIcon size={14} />
        </button>
        {menu === session.id && (
          <div className={`menu session-menu ${menuUp ? 'up' : ''}`} role="menu">
            <button onClick={() => (setMenu(null), setRenaming(session.id))}>Rename</button>
            <button onClick={() => void act(async () => onView({ id: (await api.fork(session.id)).id, title: session.title }))}>
              Fork
            </button>
            {session.archived ? (
              <button onClick={() => void act(() => api.unarchive(session.id))}>Unarchive</button>
            ) : (
              <button
                disabled={current || working}
                title={
                  current
                    ? 'The open session cannot be archived'
                    : working
                      ? 'A running session cannot be archived'
                      : 'Hide it; nothing is deleted'
                }
                onClick={() => void act(() => api.archive(session.id))}
              >
                Archive
              </button>
            )}
            <button
              className="danger"
              disabled={session.id in live || current}
              title={session.id in live || current ? 'A session open in CLAI cannot be deleted' : undefined}
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

  // A lone checkout needs no heading of its own, unless it is gone.
  const isNested = (project: Project) =>
    project.worktrees.length > 1 || project.worktrees.some((worktree) => worktree.missing)

  // The tree sits under the current session, or above the list while that row is hidden.
  const listed = projects.some(
    (project) =>
      !(!query && (collapsed[project.key] ?? false)) &&
      project.worktrees.some(
        (worktree) =>
          !(isNested(project) && !query && (collapsed[`${project.key}\0${worktree.root}`] ?? false)) &&
          worktree.sessions.some(isShown),
      ),
  )
  const workers = tasks.length > 0 && <WorkersTree key="workers" tasks={tasks} selected={worker} onOpen={onWorker} />

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
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.preventDefault()
              setQuery('')
            }
          }}
        />
        <button
          className={`archived-toggle ${archived ? 'on' : ''}`}
          onClick={(event) => {
            event.preventDefault()
            setArchived((on) => !on)
          }}
          title={archived ? 'Hide archived sessions' : 'Show and search archived sessions'}
          aria-label="Archived"
          aria-pressed={archived}
        >
          <ArchiveIcon size={14} />
        </button>
        <kbd>{MOD}K</kbd>
      </label>
      <nav className="projects">
        {!listed && workers}
        {projects.length === 0 && (
          <div className="empty-list">
            {unsaved ? 'This CLAI does not save sessions' : query ? 'No matching sessions' : 'No saved sessions yet'}
          </div>
        )}
        {projects.map((project) => {
          // A search shows every match, whatever is collapsed.
          const closed = !query && (collapsed[project.key] ?? false)
          const nested = isNested(project)
          return (
            <section key={project.key} className="project">
              <button
                className="project-title"
                onClick={() => setCollapsed((current) => ({ ...current, [project.key]: !closed }))}
                title={project.key.includes('\0') ? undefined : project.key}
              >
                <FolderIcon size={14} />
                <span className="project-name">{project.name}</span>
                {closed && <GroupDot state={strongest(project.worktrees, live)} />}
                <ChevronIcon size={12} className={`chevron ${closed ? '' : 'down'}`} />
              </button>
              {!closed &&
                project.worktrees.map((worktree) => {
                  const id = `${project.key}\0${worktree.root}`
                  const shut = nested && !query && (collapsed[id] ?? false)
                  return (
                    <div
                      key={worktree.root}
                      className={`worktree ${nested ? 'nested' : ''} ${worktree.sessions.some(isShown) ? 'current' : ''} ${worktree.missing ? 'missing' : ''}`}
                    >
                      {nested && (
                        <button
                          className="worktree-title"
                          onClick={() => setCollapsed((current) => ({ ...current, [id]: !shut }))}
                          title={worktree.missing ? `${worktree.root} (removed)` : worktree.root}
                        >
                          <BranchIcon size={12} />
                          <span className="worktree-name">{worktree.label}</span>
                          {worktree.missing && <span className="worktree-tag">removed</span>}
                          {shut && <GroupDot state={strongest([worktree], live)} />}
                          <ChevronIcon size={10} className={`chevron ${shut ? '' : 'down'}`} />
                        </button>
                      )}
                      {!shut && worktree.sessions.flatMap((session) => [row(session), isShown(session) && workers])}
                    </div>
                  )
                })}
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
