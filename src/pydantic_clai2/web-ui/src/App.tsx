import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { api, connect, interrupt, type Connection } from './api'
import { Composer, type ComposerHandle } from './components/Composer'
import { QuestionCard } from './components/QuestionCard'
import { Sidebar, type SidebarHandle } from './components/Sidebar'
import { Switches } from './components/Switches'
import { Transcript } from './components/Transcript'
import { TurnMeter } from './components/TurnMeter'
import { compact } from './format'
import { PluginsPanel } from './components/PluginsPanel'
import { DialogLayer } from './components/DialogLayer'
import { NewSessionDialog } from './components/NewSessionDialog'
import { ReplPanel } from './components/ReplPanel'
import { SidebarIcon, TerminalIcon } from './components/icons'
import type { ServerMessage, SessionEntry, Transcript as Saved } from './protocol'
import { MAC, MOD } from './platform'
import { isLight, paletteVariables } from './palette'
import { initial, reduce } from './state'

type Theme = 'dark' | 'light'

interface Opening {
  id: string
  title: string
}

function initialTheme(): Theme {
  const saved = localStorage.getItem('clai-theme')
  if (saved === 'dark' || saved === 'light') return saved
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function App() {
  const [state, dispatch] = useReducer(reduce, initial)
  const [connection, setConnection] = useState<Connection>('connecting')
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [sidebar, setSidebar] = useState(true)
  const [repl, setRepl] = useState(() => localStorage.getItem('clai-repl') === 'open')
  const [preview, setPreview] = useState<Saved | null>(null)
  // The session the user picked, until it shows: its transcript loads, then the shell resumes it.
  const [opening, setOpening] = useState<Opening | null>(null)
  const picked = useRef(0)
  const [toast, setToast] = useState<{ text: string; level: 'info' | 'error' } | null>(null)
  const [plugins, setPlugins] = useState(false)
  // Asking where a new session runs, when each session runs in its own directory.
  const [choosing, setChoosing] = useState(false)
  const openPlugins = useCallback(() => setPlugins(true), [])
  const closePlugins = useCallback(() => setPlugins(false), [])
  const local = useMemo(() => ({ '/plugins': openPlugins }), [openPlugins])
  const composer = useRef<ComposerHandle>(null)
  const sidebarRef = useRef<SidebarHandle>(null)

  // Batch a burst of streamed messages into one render per frame; hidden tabs get no frames, so a timer.
  useEffect(() => {
    let pending: ServerMessage[] = []
    let scheduled = false
    let frame = 0
    let timer = 0
    const flush = () => {
      scheduled = false
      const batch = pending
      pending = []
      for (const message of batch) dispatch(message)
    }
    const disconnect = connect((message) => {
      pending.push(message)
      if (scheduled) return
      scheduled = true
      if (document.hidden) timer = window.setTimeout(flush, 50)
      else frame = requestAnimationFrame(flush)
    }, setConnection)
    return () => {
      disconnect()
      cancelAnimationFrame(frame)
      window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => localStorage.setItem('clai-repl', repl ? 'open' : 'closed'), [repl])

  // A `/theme` palette paints the page as it paints the terminal; `default` keeps Pydantic's colours in the chosen mode.
  const palette = state.theme?.palette ?? null
  const painted = useRef<string[]>([])
  useEffect(() => {
    const root = document.documentElement
    localStorage.setItem('clai-theme', theme)
    for (const name of painted.current) root.style.removeProperty(name)
    const variables = palette ? paletteVariables(palette) : {}
    for (const [name, value] of Object.entries(variables)) root.style.setProperty(name, value)
    painted.current = Object.keys(variables)
    root.dataset.theme = palette ? (isLight(palette) ? 'light' : 'dark') : theme
    // The browser's own bar follows the page.
    const bar = palette ? palette.bg : theme === 'dark' ? '#1d0214' : '#fdfcf6'
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', bar)
  }, [theme, palette])

  useEffect(() => {
    document.title = state.session?.title ? `${state.session.title} · CLAI` : 'CLAI'
  }, [state.session?.title])

  // A switch of the live session ends any read-only preview, and the wait for it if it is the one picked.
  const sessionId = state.session?.id
  useEffect(() => {
    setPreview(null)
    setOpening((current) => (current?.id === sessionId ? null : current))
  }, [sessionId])

  // The shell may not switch at all (it reports a failed `/resume` in the terminal), so stop waiting eventually.
  useEffect(() => {
    if (!opening) return
    const timer = window.setTimeout(() => setOpening(null), 15000)
    return () => window.clearTimeout(timer)
  }, [opening])

  const showToast = useCallback((text: string, level: 'info' | 'error') => {
    const shown = { text, level }
    setToast(shown)
    window.setTimeout(() => setToast((current) => (current === shown ? null : current)), level === 'error' ? 5000 : 3000)
  }, [])
  const showError = useCallback((message: string) => showToast(message, 'error'), [showToast])
  const showNotice = useCallback((message: string) => showToast(message, 'info'), [showToast])

  const followsWorkspace = state.followsWorkspace
  const newSession = useCallback(() => {
    if (followsWorkspace) setChoosing(true)
    else void api.newSession().catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
  }, [followsWorkspace, showError])

  const open = useCallback(
    async (session: Opening) => {
      setOpening(session)
      try {
        await api.open(session.id)
      } catch (error) {
        setOpening(null)
        showError(error instanceof Error ? error.message : String(error))
      }
    },
    [showError],
  )

  const view = useCallback(
    async (session: SessionEntry) => {
      // Only the last session clicked is shown, however the requests for earlier ones resolve.
      const request = ++picked.current
      if (session.current) {
        setPreview(null)
        setOpening(null)
        return
      }
      setOpening(session)
      try {
        const saved = await api.transcript(session.id)
        if (request !== picked.current) return
        if (saved.openable && !state.running) {
          await open(session)
        } else {
          setPreview(saved)
          setOpening(null)
        }
      } catch (error) {
        if (request !== picked.current) return
        setOpening(null)
        showError(error instanceof Error ? error.message : String(error))
      }
    },
    [state.running, showError, open],
  )

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = MAC ? event.metaKey : event.ctrlKey
      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setSidebar(true)
        window.setTimeout(() => sidebarRef.current?.focusSearch())
      } else if (mod && event.shiftKey && event.key.toLowerCase() === 'o') {
        event.preventDefault()
        newSession()
      } else if (mod && event.key === 'j') {
        event.preventDefault()
        setRepl((open) => !open)
      } else if (mod && event.key === ',') {
        event.preventDefault()
        setPlugins(true)
      } else if (mod && event.key === 'b') {
        event.preventDefault()
        setSidebar((open) => !open)
      } else if (event.key === 'Escape' && state.running && !event.defaultPrevented && !event.isComposing) {
        // Esc in the composer or on the page interrupts, as in the terminal; other fields keep it for themselves.
        const target = event.target as HTMLElement
        const field = target.closest('input, textarea, select')
        if (!field || field.closest('.composer')) void interrupt(showError)
      } else if (event.key === '/' && document.activeElement === document.body) {
        event.preventDefault()
        composer.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [state.running, showError, newSession])

  const status = state.status
  const title = preview ? preview.title : (state.session?.title ?? 'New session')
  const workspace = preview ? preview.workspace : (state.session?.workspace ?? '')
  const entries = preview ? preview.items : state.entries
  const project = (state.session?.workspace ?? '').split('/').filter(Boolean).pop() ?? ''
  const fresh = !preview && entries.length === 0 && state.session !== null
  const composerElement = (
    <Composer
      ref={composer}
      running={state.running}
      meter={<TurnMeter status={status} startedAt={state.turnStartedAt} />}
      controls={<Switches status={status} running={state.running} onError={showError} />}
      model={status?.model ?? '…'}
      disabled={connection !== 'open'}
      placeholder={state.running ? 'Queue a follow-up…' : fresh ? 'Describe a task or ask a question' : 'Ask for follow-up changes'}
      local={local}
      onError={showError}
    />
  )

  return (
    <div className={`app ${sidebar ? '' : 'no-sidebar'} ${repl ? 'with-repl' : ''}`}>
      {sidebar && (
        <Sidebar
          ref={sidebarRef}
          version={state.sessionsVersion}
          viewing={preview?.id ?? null}
          opening={opening?.id ?? null}
          theme={theme}
          connected={connection === 'open'}
          onTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          themeInfo={state.theme}
          onPickTheme={(name) =>
            void api.send(`/theme ${name}`).catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
          }
          onView={(session) => void view(session)}
          onNew={newSession}
          onCollapse={() => setSidebar(false)}
          onPlugins={openPlugins}
          onError={showError}
        />
      )}
      <main className="main">
        <header className="topbar">
          {!sidebar && (
            <button className="icon-button" onClick={() => setSidebar(true)} title={`Show sidebar (${MOD}B)`}>
              <SidebarIcon size={16} />
            </button>
          )}
          <div className="topbar-title">
            <span className="title">{title}</span>
            <span className="workspace" title={workspace}>
              {workspace.replace(/^\/Users\/[^/]+/, '~')}
            </span>
          </div>
          {status && !preview && (
            <div className="topbar-status">
              <span className={`activity ${state.running ? 'busy' : ''}`}>{state.running ? status.activity : 'ready'}</span>
              <span className="context" title="Context used / window">
                {compact(status.context_tokens)} / {compact(status.context_window)}
              </span>
              {status.cost && <span className="cost">${Number(status.cost).toFixed(2)}</span>}
            </div>
          )}
          <button
            className={`icon-button repl-toggle ${repl ? 'active' : ''}`}
            onClick={() => setRepl(!repl)}
            title={`${repl ? 'Hide' : 'Show'} the run_code REPL (${MOD}J)`}
            aria-pressed={repl}
          >
            <TerminalIcon size={16} />
          </button>
        </header>
        {preview && (
          <div className="preview-banner">
            {preview.openable
              ? 'A turn is running. You can switch to this session once it ends.'
              : preview.missing
                ? `Saved in ${preview.workspace}, which no longer exists. You can read it here.`
                : `Saved in ${preview.workspace}. Start CLAI there to continue it.`}
            {preview.openable && !state.running && (
              <button className="primary small" onClick={() => void open(preview)}>
                Continue here
              </button>
            )}
            <button className="ghost small" onClick={() => setPreview(null)}>
              Back to current
            </button>
          </div>
        )}
        {fresh ? (
          <div className="hero">
            <h1>
              What should we build{project ? ' in ' : ''}
              {project && <span className="hero-project">{project}</span>}?
            </h1>
            {state.questions.map((request) => (
              <QuestionCard key={request.id} request={request} onError={showError} />
            ))}
            {composerElement}
            <p className="hero-hint">
              Type <kbd>/</kbd> for commands. The terminal and this page drive the same session.
            </p>
          </div>
        ) : (
          <>
            <Transcript
              entries={entries}
              running={!preview && state.running}
              empty={preview ? <div className="empty">This session has no messages yet.</div> : null}
            />
            <footer className="bottom">
              {!preview &&
                state.questions.map((request) => <QuestionCard key={request.id} request={request} onError={showError} />)}
              {!preview && composerElement}
            </footer>
          </>
        )}
        {opening && (
          <div className="opening-overlay" role="status">
            <span className="spinner" />
            <span>
              Opening <span className="opening-title">{opening.title}</span>…
            </span>
          </div>
        )}
        {connection === 'expired' && (
          <div className="toast expired" role="alert">
            This page belongs to an earlier CLAI. Open the new address it printed.
          </div>
        )}
        {toast && (
          <button
            className={`toast ${toast.level}`}
            role={toast.level === 'error' ? 'alert' : 'status'}
            onClick={() => setToast(null)}
            title="Dismiss"
          >
            {toast.text}
          </button>
        )}
      </main>
      {plugins && (
        <PluginsPanel
          version={state.pluginsVersion}
          running={state.running}
          onClose={closePlugins}
          onError={showError}
          onNotice={showNotice}
        />
      )}
      {repl && (
        <ReplPanel
          entries={entries}
          status={status}
          running={state.running}
          onClose={() => setRepl(false)}
          onError={showError}
        />
      )}
      {choosing && state.session && (
        <NewSessionDialog current={state.session.workspace} onClose={() => setChoosing(false)} />
      )}
      <DialogLayer dialogs={state.dialogs} onError={showError} />
    </div>
  )
}
