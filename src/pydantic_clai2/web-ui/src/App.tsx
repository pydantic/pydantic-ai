import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api, connect, interrupt, type Connection } from './api'
import { Composer, type ComposerHandle } from './components/Composer'
import { QuestionCard } from './components/QuestionCard'
import { Sidebar, type SidebarHandle } from './components/Sidebar'
import { Switches } from './components/Switches'
import { Transcript } from './components/Transcript'
import { TurnMeter } from './components/TurnMeter'
import { compact } from './format'
import { SidebarIcon } from './components/icons'
import type { ServerMessage, SessionEntry, Transcript as Saved } from './protocol'
import { MAC, MOD } from './platform'
import { isLight, paletteVariables } from './palette'
import { initial, reduce } from './state'

type Theme = 'dark' | 'light'

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
  const [preview, setPreview] = useState<Saved | null>(null)
  const [toast, setToast] = useState<string | null>(null)
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

  // A switch of the live session ends any read-only preview.
  useEffect(() => setPreview(null), [state.session?.id])

  const showError = useCallback((message: string) => {
    setToast(message)
    window.setTimeout(() => setToast((current) => (current === message ? null : current)), 5000)
  }, [])

  const view = useCallback(
    async (session: SessionEntry) => {
      if (session.current) {
        setPreview(null)
        return
      }
      try {
        const saved = await api.transcript(session.id)
        if (saved.openable && !state.running) {
          await api.open(session.id)
        } else {
          setPreview(saved)
        }
      } catch (error) {
        showError(error instanceof Error ? error.message : String(error))
      }
    },
    [state.running, showError],
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
        void api.newSession().catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
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
  }, [state.running, showError])

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
      onError={showError}
    />
  )

  return (
    <div className={`app ${sidebar ? '' : 'no-sidebar'}`}>
      {sidebar && (
        <Sidebar
          ref={sidebarRef}
          version={state.sessionsVersion}
          viewing={preview?.id ?? null}
          theme={theme}
          connected={connection === 'open'}
          onTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          themeInfo={state.theme}
          onPickTheme={(name) =>
            void api.send(`/theme ${name}`).catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
          }
          onView={(session) => void view(session)}
          onCollapse={() => setSidebar(false)}
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
        </header>
        {preview && (
          <div className="preview-banner">
            {preview.openable
              ? 'A turn is running. You can switch to this session once it ends.'
              : `Saved in ${preview.workspace}. Start CLAI there to continue it.`}
            {preview.openable && !state.running && (
              <button className="primary small" onClick={() => void api.open(preview.id)}>
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
        {connection === 'expired' && (
          <div className="toast expired" role="alert">
            This page belongs to an earlier CLAI. Open the new address it printed.
          </div>
        )}
        {toast && (
          <button className="toast" role="alert" onClick={() => setToast(null)} title="Dismiss">
            {toast}
          </button>
        )}
      </main>
    </div>
  )
}
