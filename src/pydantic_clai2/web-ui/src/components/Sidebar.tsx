import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { ApiError, api } from '../api'
import { MOD } from '../platform'
import { Logo } from './Logo'
import { WorkspacesSection } from './WorkspacesSection'
import type { Project, SessionEntry, ThemeInfo, Worktree } from '../protocol'
import {
  BranchIcon,
  ChevronIcon,
  ComposeIcon,
  FolderIcon,
  MoonIcon,
  MoreIcon,
  OrchestratorIcon,
  PaletteIcon,
  PuzzleIcon,
  SearchIcon,
  SidebarIcon,
  SunIcon,
} from './icons'

export interface SidebarHandle {
  focusSearch: () => void
}

interface Props {
  version: number
  viewing: string | null
  /** The session being opened, which shows a spinner. */
  opening: string | null
  theme: 'dark' | 'light'
  connected: boolean
  onTheme: () => void
  themeInfo: ThemeInfo | null
  onPickTheme: (name: string) => void
  onView: (session: SessionEntry) => void
  /** Start a new session, asking where it runs when sessions run in their own directories. */
  onNew: () => void
  /** Whether the main pane shows the orchestrator chat, and whether its turn runs. */
  orchestrating: boolean
  orchestratorRunning: boolean
  onOrchestrator: () => void
  onCollapse: () => void
  onPlugins: () => void
  onError: (message: string) => void
}

export function ago(iso: string): string {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`
  if (seconds < 86400 * 7) return `${Math.floor(seconds / 86400)}d`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/** Whether a turn runs in one of these sessions, so a collapsed group still shows it. */
function live(worktrees: Worktree[]): boolean {
  return worktrees.some((worktree) => worktree.sessions.some((s) => s.state === 'running' || s.state === 'busy'))
}

/** Saved sessions grouped by project, then worktree, as `/resume` lists them, with search and per-session actions. */
export const Sidebar = forwardRef<SidebarHandle, Props>(function Sidebar(
  {
    version,
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
    onError,
  },
  ref,
) {
  const [query, setQuery] = useState('')
  const [projects, setProjects] = useState<Project[]>([])
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [menu, setMenu] = useState<string | null>(null)
  const [menuUp, setMenuUp] = useState(false)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [unsaved, setUnsaved] = useState(false)
  const search = useRef<HTMLInputElement>(null)

  useImperativeHandle(ref, () => ({ focusSearch: () => search.current?.focus() }), [])

  const loading = useRef(0)
  const load = useCallback(async () => {
    // A search and the periodic refresh can overlap; only the latest request's answer is shown.
    const id = ++loading.current
    try {
      const found = await api.sessions(query)
      if (id === loading.current) setProjects(found)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) setUnsaved(true)
      else onError(error instanceof Error ? error.message : String(error))
    }
  }, [query, onError])

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

  const row = (session: SessionEntry) => (
    <div
      key={session.id}
      className={`session ${session.current ? 'current' : ''} ${viewing === session.id ? 'viewing' : ''} ${opening === session.id ? 'opening' : ''}`}
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
          <span className={`state ${session.state}`} />
          <span className="session-title">{session.title}</span>
          {opening === session.id && <span className="spinner" aria-label="Opening" />}
          <span className="session-time">{session.state === 'running' ? '' : ago(session.updated_at)}</span>
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
          <button onClick={() => void act(() => api.fork(session.id))}>Fork</button>
          <button
            className="danger"
            disabled={session.current}
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

  return (
    <aside className={`sidebar ${orchestrating ? 'orchestrating' : ''}`}>
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
        <kbd>{MOD}K</kbd>
      </label>
      <nav className="projects">
        <WorkspacesSection version={version} onError={onError} />
        {projects.length === 0 && (
          <div className="empty-list">
            {unsaved ? 'This CLAI does not save sessions' : query ? 'No matching sessions' : 'No saved sessions yet'}
          </div>
        )}
        {projects.map((project) => {
          // A search shows every match, whatever is collapsed.
          const closed = !query && (collapsed[project.key] ?? false)
          // A lone checkout needs no heading of its own, unless it is gone.
          const nested = project.worktrees.length > 1 || project.worktrees.some((worktree) => worktree.missing)
          return (
            <section key={project.key} className="project">
              <button
                className="project-title"
                onClick={() => setCollapsed((current) => ({ ...current, [project.key]: !closed }))}
                title={project.key.includes('\0') ? undefined : project.key}
              >
                <FolderIcon size={14} />
                <span className="project-name">{project.name}</span>
                {closed && live(project.worktrees) && <span className="state running" />}
                <ChevronIcon size={12} className={`chevron ${closed ? '' : 'down'}`} />
              </button>
              {!closed &&
                project.worktrees.map((worktree) => {
                  const id = `${project.key}\0${worktree.root}`
                  const shut = nested && !query && (collapsed[id] ?? false)
                  return (
                    <div
                      key={worktree.root}
                      className={`worktree ${nested ? 'nested' : ''} ${worktree.current ? 'current' : ''} ${worktree.missing ? 'missing' : ''}`}
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
                          {shut && live([worktree]) && <span className="state running" />}
                          <ChevronIcon size={10} className={`chevron ${shut ? '' : 'down'}`} />
                        </button>
                      )}
                      {!shut && worktree.sessions.map(row)}
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
          <button className="icon-button" onClick={onPlugins} title={`Plugins (${MOD},)`} aria-label="Plugins">
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
