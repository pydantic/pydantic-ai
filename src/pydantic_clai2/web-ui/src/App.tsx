import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type CSSProperties } from 'react'
import { SessionContext, api, connect, interrupt, orchestrator, sessionApi, type Connection, type Socket } from './api'
import type { Attachment } from './attachments'
import { Composer, type ComposerHandle } from './components/Composer'
import { QuestionCard } from './components/QuestionCard'
import { Sidebar, type SidebarHandle } from './components/Sidebar'
import { Switches } from './components/Switches'
import { Transcript } from './components/Transcript'
import { TurnMeter } from './components/TurnMeter'
import { compact } from './format'
import { CommandMenu } from './components/CommandMenu'
import { CommandPalette } from './components/CommandPalette'
import { SettingsWindow, commandSection, linkedSection, type SettingsSection } from './components/SettingsWindow'
import { DialogLayer } from './components/DialogLayer'
import { MontyPanel } from './components/MontyPanel'
import { OrchestratorPane } from './components/OrchestratorPane'
import { NewSessionDialog } from './components/NewSessionDialog'
import { PanelGlyph, SidePanel, followWindows, following, panelKey } from './components/SidePanel'
import { WorkerView } from './components/WorkerView'
import { SidebarIcon, TerminalIcon } from './components/icons'
import type { PanelInfo, ServerMessage, Transcript as Saved } from './protocol'
import { MAC, MOD, STEER_KEY } from './platform'
import { isLight, paletteVariables } from './palette'
import { initial, initialOrchestrator, reduce, reduceOrchestrator, shownChat, type Action } from './state'
import { WorkersContext } from './workers'

type Theme = 'dark' | 'light'
/** What the main pane shows: the session, or the orchestrator chat. */
type Pane = 'session' | 'orchestrator'

interface Opening {
  id: string
  title: string
}

const NO_ATTACHMENTS: readonly Attachment[] = []

function initialTheme(): Theme {
  const saved = localStorage.getItem('clai-theme')
  if (saved === 'dark' || saved === 'light') return saved
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function App() {
  const [state, dispatch] = useReducer(reduce, initial)
  // The live session the main pane shows; the others' messages do not come, and their dots come from `live`.
  const chat = shownChat(state)
  const shown = state.shown
  const live = state.live
  const session = useMemo(() => sessionApi(shown), [shown])
  const [orchestration, dispatchOrchestrator] = useReducer(reduceOrchestrator, initialOrchestrator)
  const [pane, setPane] = useState<Pane>(() =>
    localStorage.getItem('clai-pane') === 'orchestrator' ? 'orchestrator' : 'session',
  )
  const [connection, setConnection] = useState<Connection>('connecting')
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [sidebar, setSidebar] = useState(true)
  const [monty, setMonty] = useState(() => localStorage.getItem('clai-monty') === 'open')
  const [preview, setPreview] = useState<Saved | null>(null)
  // The session the user picked, until it shows: CLAI makes it live, then its snapshot comes.
  const [opening, setOpening] = useState<Opening | null>(null)
  // The delegated task whose transcript the main pane shows instead of the session's.
  const [worker, setWorker] = useState<string | null>(null)
  const picked = useRef(0)
  const [toast, setToast] = useState<{ text: string; level: 'info' | 'error' } | null>(null)
  // The Settings window's open section, which the address keeps as `#settings/ID`; `null` while it is closed.
  const [settings, setSettings] = useState<SettingsSection | null>(() => linkedSection(location.hash))
  // The plugin the Plugins section opens on, when a side panel asks to set its plugin up.
  const [pluginFocus, setPluginFocus] = useState<string | null>(null)
  const openPlugins = useCallback(() => {
    setPluginFocus(null)
    setSettings('plugins')
  }, [])
  const openSettings = useCallback(() => setSettings((open) => open ?? 'general'), [])
  const closeSettings = useCallback(() => setSettings(null), [])
  // The bare command whose menu shows in the page, such as `/theme`.
  const [commandMenu, setCommandMenu] = useState<string | null>(null)
  const closeCommandMenu = useCallback(() => setCommandMenu(null), [])
  const openMenu = useCallback((command: string) => {
    const section = commandSection(command)
    if (section) setSettings(section)
    else setCommandMenu(command)
  }, [])
  useEffect(() => {
    const linked = settings === null ? '' : `#settings/${settings}`
    if (location.hash !== linked) history.replaceState(null, '', `${location.pathname}${location.search}${linked}`)
  }, [settings])
  useEffect(() => {
    const follow = () => setSettings(linkedSection(location.hash))
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  }, [])
  // Asking where a new session runs, when each session runs in its own directory.
  const [choosing, setChoosing] = useState(false)
  const [panels, setPanels] = useState<PanelInfo[]>([])
  // Which plugin side panel is open, and how wide, as this browser last left them.
  const [sidePanel, setSidePanel] = useState(() => localStorage.getItem('clai-side-panel'))
  const [panelWidth, setPanelWidth] = useState(() => Number(localStorage.getItem('clai-side-panel-width')) || 440)
  // The command palette, with every command the registry offers.
  const [commandPalette, setCommandPalette] = useState(false)
  const closeCommandPalette = useCallback(() => setCommandPalette(false), [])
  // Each session's unsent draft, by conversation ID, so switching away and back keeps it.
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const draft = drafts[shown ?? ''] ?? ''
  const setDraft = useCallback((text: string) => setDrafts((all) => ({ ...all, [shown ?? '']: text })), [shown])
  // Each session's pasted images, kept as its draft is.
  const [attached, setAttached] = useState<Record<string, readonly Attachment[]>>({})
  const attachments = attached[shown ?? ''] ?? NO_ATTACHMENTS
  const setAttachments = useCallback(
    (update: (current: readonly Attachment[]) => Attachment[]) =>
      setAttached((all) => ({ ...all, [shown ?? '']: update(all[shown ?? ''] ?? NO_ATTACHMENTS) })),
    [shown],
  )
  const composer = useRef<ComposerHandle>(null)
  // A file dropped beside the composer must not open in the tab, losing every session's draft.
  useEffect(() => {
    const guard = (event: DragEvent) => {
      if (event.dataTransfer?.types.includes('Files')) event.preventDefault()
    }
    window.addEventListener('dragover', guard)
    window.addEventListener('drop', guard)
    return () => {
      window.removeEventListener('dragover', guard)
      window.removeEventListener('drop', guard)
    }
  }, [])
  const sidebarRef = useRef<SidebarHandle>(null)
  const socket = useRef<Socket | null>(null)
  // What a new connection watches again, read when it opens.
  const watching = useRef(shown)
  watching.current = shown

  // Batch a burst of streamed messages into one render per frame; hidden tabs get no frames, so a timer.
  useEffect(() => {
    let pending: (ServerMessage | Action)[] = []
    let scheduled = false
    let frame = 0
    let timer = 0
    const flush = () => {
      scheduled = false
      const batch = pending
      pending = []
      for (const message of batch) {
        if (message.type === 'orchestrator') dispatchOrchestrator(message.message)
        else dispatch(message)
      }
    }
    const queue = (message: ServerMessage | Action) => {
      pending.push(message)
      if (scheduled) return
      scheduled = true
      if (document.hidden) timer = window.setTimeout(flush, 50)
      else frame = requestAnimationFrame(flush)
    }
    // A reconnect watches the shown session again, in order with the messages, so the default session's snapshot
    // that every connection starts with waits aside.
    const rewatch = (id: string) => queue({ type: 'show', id, reconnect: true })
    const opened = connect(queue, setConnection, () => watching.current, rewatch)
    socket.current = opened
    return () => {
      socket.current = null
      opened.close()
      cancelAnimationFrame(frame)
      window.clearTimeout(timer)
    }
  }, [])

  // Watch the shown session: its fresh snapshot comes, then its messages, while the others run on unwatched.
  useEffect(() => {
    if (shown !== null && connection === 'open') socket.current?.watch(shown)
  }, [shown, connection])
  // A session that closed before it could show leaves nothing shown: show the default session again.
  const lostSession = shown === null && state.version !== null
  const liveCount = Object.keys(state.live).length
  useEffect(() => {
    if (!lostSession || connection !== 'open' || liveCount === 0) return
    void api.state().then((snapshot) => dispatch({ type: 'fallback', snapshot }), () => undefined)
  }, [lostSession, connection, liveCount])
  // A session that never answers, as one that closed before any `live` listed it, is not waited for forever.
  const awaiting = state.awaiting
  useEffect(() => {
    if (awaiting === null) return
    const timer = window.setTimeout(() => dispatch({ type: 'give_up', id: awaiting }), 15000)
    return () => window.clearTimeout(timer)
  }, [awaiting])

  useEffect(() => localStorage.setItem('clai-monty', monty ? 'open' : 'closed'), [monty])
  useEffect(() => localStorage.setItem('clai-pane', pane), [pane])

  // A command or the plugins panel may have turned `workspaces` on or off, which the orchestrator needs, or
  // saved another default model, which it runs on.
  const pluginsVersion = state.pluginsVersion
  const settingsVersion = state.settingsVersion
  useEffect(() => {
    if (pluginsVersion === 0 && settingsVersion === 0) return
    void orchestrator
      .state()
      .then((found) => dispatchOrchestrator({ type: 'refreshed', enabled: found.enabled, model: found.model }))
      .catch(() => undefined)
  }, [pluginsVersion, settingsVersion])
  useEffect(() => {
    if (sidePanel) localStorage.setItem('clai-side-panel', sidePanel)
    else localStorage.removeItem('clai-side-panel')
  }, [sidePanel])

  // Panels follow the conversation: ask again on connecting, switching sessions, each turn's start and end, and
  // every command, which may have changed a plugin; while a turn runs and one is shown, every few seconds too.
  const panelRequest = useRef(0)
  const panelsLoading = useRef(false)
  const loadPanels = useCallback(async (poll = false) => {
    if (poll && panelsLoading.current) return
    // Only the latest answer is shown, however the requests resolve.
    const request = ++panelRequest.current
    panelsLoading.current = true
    try {
      const found = await api.panels()
      if (request === panelRequest.current)
        setPanels((current) => (JSON.stringify(current) === JSON.stringify(found) ? current : found))
    } catch {
      // Keep what is shown; the next turn or command asks again.
    } finally {
      if (request === panelRequest.current) panelsLoading.current = false
    }
  }, [])
  const shownPanel = panels.find((panel) => panelKey(panel) === sidePanel) ?? null
  const firstPanel = panels[0]
  useEffect(() => {
    if (connection === 'open') void loadPanels()
  }, [connection, chat.session?.id, chat.running, state.pluginsVersion, loadPanels])
  const panelShown = shownPanel !== null
  useEffect(() => {
    if (!chat.running || panels.length === 0) return
    const timer = window.setInterval(() => {
      if (panelShown || following()) void loadPanels(true)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [chat.running, panels.length, panelShown, loadPanels])
  useEffect(() => followWindows(panels), [panels])
  // With no key, the shortcut: close the open panel, or open the first.
  const shownKey = shownPanel ? panelKey(shownPanel) : null
  const togglePanel = useCallback(
    (key?: string) => {
      if (key !== undefined) setSidePanel(shownKey === key ? null : key)
      else setSidePanel(shownKey || !firstPanel ? null : panelKey(firstPanel))
    },
    [shownKey, firstPanel],
  )

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
    if (pane === 'orchestrator') document.title = 'Orchestrator · CLAI'
    else document.title = chat.session?.title ? `${chat.session.title} · CLAI` : 'CLAI'
  }, [chat.session?.title, pane])

  // The picked session showing ends the wait for it; `show` already closed any preview and worker.
  const sessionId = chat.session?.id
  useEffect(() => {
    setOpening((current) => (current?.id === sessionId ? null : current))
  }, [sessionId])

  // The session may not show at all (it closed meanwhile), so stop waiting eventually.
  useEffect(() => {
    if (!opening) return
    const timer = window.setTimeout(() => setOpening(null), 15000)
    return () => window.clearTimeout(timer)
  }, [opening])

  // A CLAI that does not come back from `/update`'s restart, such as one whose port was taken, leaves the page waiting.
  const restarting = state.update !== null && (state.update.restarting || connection !== 'open')
  const [lost, setLost] = useState(false)
  useEffect(() => {
    setLost(false)
    if (!restarting) return
    const timer = window.setTimeout(() => setLost(true), 60000)
    return () => window.clearTimeout(timer)
  }, [restarting])

  const showToast = useCallback((text: string, level: 'info' | 'error') => {
    const shown = { text, level }
    setToast(shown)
    window.setTimeout(() => setToast((current) => (current === shown ? null : current)), level === 'error' ? 5000 : 3000)
  }, [])
  const showError = useCallback((message: string) => showToast(message, 'error'), [showToast])
  const showNotice = useCallback((message: string) => showToast(message, 'info'), [showToast])

  // Show a live session, by conversation ID; the effect above watches it, and its snapshot replaces what shows.
  const show = useCallback((id: string) => {
    // A session picked earlier and still opening loses to this one.
    picked.current++
    // Its own spinner stays until it shows; one for a session picked before goes.
    setOpening((current) => (current?.id === id ? current : null))
    setPane('session')
    setWorker(null)
    setPreview(null)
    dispatch({ type: 'show', id, reconnect: false })
  }, [])

  // The session shown when a new one was asked for. With `/web` from a terminal, `new` answers that session's ID while
  // it switches to the new conversation, whose snapshot follows by itself.
  const newFrom = useRef<string | null>(null)
  const showNew = useCallback(
    (id: string) => {
      if (id !== newFrom.current) show(id)
      else {
        // The shown session itself switches, so `show` does not run to close what covers it, nor end an earlier pick.
        picked.current++
        setOpening(null)
        setWorker(null)
        setPreview(null)
      }
    },
    [show],
  )

  // Bare commands whose terminal pickers have a place in this page already: the sidebar, or the palette.
  const focusSessions = useCallback(() => {
    setSidebar(true)
    window.setTimeout(() => sidebarRef.current?.focusSearch())
  }, [])
  const focusWorkers = useCallback(() => {
    setSidebar(true)
    window.setTimeout(() => sidebarRef.current?.focusWorkers() || showNotice('This session has no delegated tasks yet.'))
  }, [showNotice])
  const local = useMemo(
    () => ({ '/plugins': openPlugins, '/help': () => setCommandPalette(true), '/tasks': focusWorkers, '/resume': focusSessions }),
    [openPlugins, focusWorkers, focusSessions],
  )
  /** A bare command picked in the palette, run as the composer runs it. */
  const runCommand = useCallback(
    async (name: string) => {
      const text = `/${name}`
      const handle = Object.hasOwn(local, text) ? local[text as keyof typeof local] : undefined
      if (handle) return handle()
      try {
        const entered = await session.send(text)
        if (entered.menu !== null) openMenu(entered.menu)
      } catch (error) {
        showError(error instanceof Error ? error.message : String(error))
      }
    },
    [local, session, openMenu, showError],
  )
  const followsWorkspace = state.followsWorkspace
  const newSession = useCallback(() => {
    setPane('session')
    newFrom.current = shown
    if (followsWorkspace) {
      setChoosing(true)
      return
    }
    // A session picked before still opening loses to the new one, which loses to one picked after.
    const request = ++picked.current
    setOpening(null)
    void api
      .newSession()
      .then(({ id }) => {
        if (request === picked.current) showNew(id)
      })
      .catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
  }, [followsWorkspace, shown, showNew, showError])

  // Closing the preview also drops a transcript still on its way, so it cannot reopen it.
  const closePreview = useCallback(() => {
    picked.current++
    setPreview(null)
    setOpening(null)
  }, [])

  const view = useCallback(
    async (session: Opening) => {
      // Only the last session clicked is shown, however the requests for earlier ones resolve.
      const request = ++picked.current
      setPane('session')
      setWorker(null)
      if (session.id === shown) {
        setPreview(null)
        setOpening(null)
        return
      }
      setOpening(session)
      try {
        // A saved session opens live, whatever the others run, unless CLAI can only show it, as when its
        // directory is gone; a live one opens at once.
        if (!(session.id in live)) {
          const saved = await api.transcript(session.id)
          if (request !== picked.current) return
          if (!saved.openable) {
            setPreview(saved)
            setOpening(null)
            return
          }
        }
        const { id } = await api.open(session.id)
        if (request !== picked.current) return
        // With `/web` from a terminal, the shown session itself switches to it, and its new snapshot follows.
        if (id !== shown) show(id)
        else setPreview(null)
      } catch (error) {
        if (request !== picked.current) return
        setOpening(null)
        showError(error instanceof Error ? error.message : String(error))
      }
    },
    [shown, live, show, showError],
  )

  const openWorker = useCallback((id: string) => {
    setPreview(null)
    setWorker(id)
  }, [])
  const workerTask = worker === null ? null : (chat.tasks.find((task) => task.id === worker) ?? null)
  const links = useMemo(() => ({ tasks: chat.tasks, parent: null, open: openWorker }), [chat.tasks, openWorker])

  // Esc stops the turn of what the main pane shows: the orchestrator's, or the shown session's alone.
  const orchestrating = pane === 'orchestrator'
  const running = orchestrating ? orchestration.chat.running : chat.running
  const stop = orchestrating ? orchestrator.interrupt : session.interrupt
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = MAC ? event.metaKey : event.ctrlKey
      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        focusSessions()
      } else if (mod && event.shiftKey && event.key.toLowerCase() === 'p') {
        event.preventDefault()
        setCommandPalette((open) => !open)
      } else if (mod && event.shiftKey && event.key.toLowerCase() === 'o') {
        event.preventDefault()
        newSession()
      } else if (mod && event.key === 'j') {
        event.preventDefault()
        setMonty((open) => !open)
      } else if (mod && event.shiftKey && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        togglePanel()
      } else if (mod && event.key === ',') {
        event.preventDefault()
        openSettings()
      } else if (mod && event.key === 'b') {
        event.preventDefault()
        setSidebar((open) => !open)
      } else if (
        event.key === 'Escape' &&
        running &&
        (orchestrating || (!workerTask && !preview)) &&
        !event.defaultPrevented &&
        !event.isComposing
      ) {
        // Esc in the composer or on the page interrupts, as in the terminal; other fields keep it for themselves.
        const target = event.target as HTMLElement
        const field = target.closest('input, textarea, select')
        if (!field || field.closest('.composer')) void interrupt(showError, stop)
      } else if (event.key === '/' && document.activeElement === document.body) {
        event.preventDefault()
        composer.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [running, stop, orchestrating, workerTask, preview, showError, togglePanel, openSettings, newSession, focusSessions])

  const status = chat.status
  const title = workerTask
    ? `${workerTask.agent}: ${workerTask.prompt.split('\n')[0]}`
    : preview
      ? preview.title
      : (chat.session?.title ?? 'New session')
  const workspace = preview ? preview.workspace : (chat.session?.workspace ?? '')
  const entries = preview ? preview.items : chat.entries
  const project = (chat.session?.workspace ?? '').split('/').filter(Boolean).pop() ?? ''
  const fresh = !preview && !workerTask && entries.length === 0 && chat.session !== null
  const composerElement = (
    <Composer
      ref={composer}
      running={chat.running}
      meter={<TurnMeter status={status} startedAt={chat.turnStartedAt} />}
      controls={<Switches status={status} running={chat.running} onError={showError} />}
      model={status?.model ?? '…'}
      disabled={connection !== 'open'}
      placeholder={chat.running ? `Queue a follow-up, or steer with ${STEER_KEY}…` : fresh ? 'Describe a task or ask a question' : 'Ask for follow-up changes'}
      local={local}
      onMenu={openMenu}
      draft={draft}
      onDraft={setDraft}
      queue={chat.queue}
      attachments={attachments}
      onAttachments={setAttachments}
      onError={showError}
    />
  )

  const page = (
    <div
      className={`app ${sidebar ? '' : 'no-sidebar'} ${monty && !orchestrating ? 'with-monty' : ''} ${shownPanel && !orchestrating ? 'with-side-panel' : ''}`}
      style={{ '--side-panel-width': `min(${panelWidth}px, 70vw)` } as CSSProperties}
    >
      {sidebar && (
        <Sidebar
          ref={sidebarRef}
          version={state.sessionsVersion}
          shown={shown}
          live={live}
          viewing={preview?.id ?? null}
          opening={opening?.id ?? null}
          theme={theme}
          connected={connection === 'open'}
          onTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          themeInfo={state.theme}
          onPickTheme={(name) =>
            void session
              .send(`/theme ${name}`)
              .catch((error: unknown) => showError(error instanceof Error ? error.message : String(error)))
          }
          onView={(entry) => void view(entry)}
          onNew={newSession}
          orchestrating={orchestrating}
          orchestratorRunning={orchestration.chat.running}
          onOrchestrator={() => {
            setPane('orchestrator')
            setPreview(null)
          }}
          onCollapse={() => setSidebar(false)}
          onPlugins={openPlugins}
          onSettings={openSettings}
          onError={showError}
          tasks={chat.tasks}
          worker={workerTask?.id ?? null}
          onWorker={openWorker}
        />
      )}
      <main className="main">
        {orchestrating ? (
          <OrchestratorPane
            view={orchestration}
            model={orchestration.model ?? 'no model'}
            connected={connection === 'open'}
            onShowSidebar={sidebar ? null : () => setSidebar(true)}
            onOpenSession={show}
            onError={showError}
          />
        ) : (
          <>
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
              {status && !preview && !workerTask && (
                <div className="topbar-status">
                  <span className={`activity ${chat.running ? 'busy' : ''}`}>{chat.running ? status.activity : 'ready'}</span>
                  <span className="context" title="Context used / window">
                    {compact(status.context_tokens)} / {compact(status.context_window)}
                  </span>
                  {status.cost && (
                    <span className="cost">
                      {status.cost_is_lower_bound && '>='}${Number(status.cost).toFixed(2)}
                    </span>
                  )}
                </div>
              )}
              {panels.map((panel) => {
                const key = panelKey(panel)
                const active = shownKey === key
                return (
                  <button
                    key={key}
                    className={`icon-button monty-toggle ${active ? 'active' : ''}`}
                    onClick={() => togglePanel(key)}
                    title={`${active ? 'Hide' : 'Show'} ${panel.title}${panel === firstPanel ? ` (${MOD}⇧L)` : ''}`}
                    aria-pressed={active}
                  >
                    <PanelGlyph panel={panel} size={16} />
                  </button>
                )
              })}
              <button
                className={`icon-button monty-toggle ${monty ? 'active' : ''}`}
                onClick={() => setMonty(!monty)}
                title={`${monty ? 'Hide' : 'Show'} Monty, the run_code sandbox (${MOD}J)`}
                aria-pressed={monty}
              >
                <TerminalIcon size={16} />
              </button>
            </header>
            {preview && (
              <div className="preview-banner">
                {preview.missing
                  ? `Saved in ${preview.workspace}, which no longer exists. You can read it here.`
                  : `Saved in ${preview.workspace}. Start CLAI there to continue it.`}
                <button className="ghost small" onClick={closePreview}>
                  Back to current
                </button>
              </div>
            )}
            {workerTask ? (
              <WorkerView
                task={workerTask}
                tasks={chat.tasks}
                live={chat.taskItems[workerTask.id]}
                onOpen={openWorker}
                onBack={() => setWorker(null)}
                onError={showError}
              />
            ) : fresh ? (
              <div className="hero">
                <h1>
                  What should we build{project ? ' in ' : ''}
                  {project && <span className="hero-project">{project}</span>}?
                </h1>
                {chat.questions.map((request) => (
                  <QuestionCard key={request.id} request={request} onError={showError} />
                ))}
                {composerElement}
                <p className="hero-hint">
                  Type <kbd>/</kbd> for commands, or <kbd>{MOD}⇧P</kbd> to browse them. Your other sessions keep
                  running while you work here.
                </p>
              </div>
            ) : (
              <>
                <WorkersContext.Provider value={links}>
                  <Transcript
                    entries={entries}
                    running={!preview && chat.running}
                    empty={preview ? <div className="empty">This session has no messages yet.</div> : null}
                  />
                </WorkersContext.Provider>
                <footer className="bottom">
                  {!preview &&
                    chat.questions.map((request) => <QuestionCard key={request.id} request={request} onError={showError} />)}
                  {!preview && composerElement}
                </footer>
              </>
            )}
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
        {state.update && (
          <div className="opening-overlay" role="status">
            {!lost && <span className="spinner" />}
            {lost ? (
              <span>CLAI did not come back at this address. Check the terminal it runs in.</span>
            ) : restarting ? (
              <span>Restarting CLAI…</span>
            ) : (
              <span>
                Updating CLAI to <span className="opening-title">{state.update.target}</span>…
                {state.update.line && <span className="update-line mono">{state.update.line}</span>}
              </span>
            )}
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
      {settings && (
        <SettingsWindow
          section={settings}
          onSection={setSettings}
          onClose={closeSettings}
          running={chat.running}
          pluginsVersion={state.pluginsVersion}
          settingsVersion={state.settingsVersion}
          pluginFocus={pluginFocus}
          onError={showError}
          onNotice={showNotice}
        />
      )}
      {commandMenu && (
        <CommandMenu
          command={commandMenu}
          version={state.settingsVersion}
          onClose={closeCommandMenu}
          onError={showError}
          onNotice={showNotice}
        />
      )}
      {monty && !orchestrating && (
        <MontyPanel
          entries={entries}
          status={status}
          running={chat.running}
          onClose={() => setMonty(false)}
          onError={showError}
        />
      )}
      {shownPanel && !orchestrating && (
        <SidePanel
          panel={shownPanel}
          width={panelWidth}
          onResize={(width, done) => {
            setPanelWidth(width)
            if (done) localStorage.setItem('clai-side-panel-width', String(width))
          }}
          onClose={() => setSidePanel(null)}
          onSettings={(plugin) => {
            setPluginFocus(plugin)
            setSettings('plugins')
          }}
          onError={showError}
        />
      )}
      {choosing && chat.session && (
        <NewSessionDialog current={chat.session.workspace} onOpened={showNew} onClose={() => setChoosing(false)} />
      )}
      {commandPalette && (
        <CommandPalette
          onRun={(name) => void runCommand(name)}
          onInsert={(text) => {
            setPane('session')
            setWorker(null)
            closePreview()
            setDraft(text)
            window.setTimeout(() => composer.current?.focus())
          }}
          onClose={closeCommandPalette}
          onError={showError}
        />
      )}
      <DialogLayer dialogs={state.dialogs} onError={showError} />
    </div>
  )
  // The composer, its questions, and Monty talk to the shown session.
  return <SessionContext.Provider value={session}>{page}</SessionContext.Provider>
}
