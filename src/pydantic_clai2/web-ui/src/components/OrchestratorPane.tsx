import { useCallback, useState } from 'react'
import { api, orchestrator } from '../api'
import { MOD } from '../platform'
import type { OrchestratorView } from '../state'
import { WorkspaceLinkContext } from '../workspaceLink'
import { Composer } from './Composer'
import { ComposeIcon, OrchestratorIcon, SidebarIcon } from './icons'
import { QuestionCard } from './QuestionCard'
import { Transcript } from './Transcript'

interface Props {
  view: OrchestratorView
  /** The saved default model, which the orchestrator runs on, never the shown session's own choice. */
  model: string
  connected: boolean
  /** Shows the sidebar again when it is hidden. */
  onShowSidebar: (() => void) | null
  /** Shows a live session, by its conversation ID, in place of the chat. */
  onOpenSession: (id: string) => void
  onError: (message: string) => void
}

const CHAT = { send: orchestrator.send, interrupt: orchestrator.interrupt }

/** The orchestrator chat: its own agent and conversation, which makes workspaces and runs a session in each for you. */
export function OrchestratorPane({ view, model, connected, onShowSidebar, onOpenSession, onError }: Props) {
  const { chat, enabled } = view
  const [enabling, setEnabling] = useState(false)
  // `Composer` leaves its draft to its owner.
  const [draft, setDraft] = useState('')
  const report = useCallback((error: unknown) => onError(error instanceof Error ? error.message : String(error)), [onError])
  const openWorkspace = useCallback(
    (id: string) =>
      void api
        .openWorkspace(id)
        .then((opened) => onOpenSession(opened.id))
        .catch(report),
    [onOpenSession, report],
  )

  const enable = async () => {
    setEnabling(true)
    try {
      await api.plugin('workspaces', 'enable')
    } catch (error) {
      report(error)
    } finally {
      setEnabling(false)
    }
  }

  const enableButton = (
    <button className="primary" onClick={() => void enable()} disabled={enabling || !connected}>
      Enable workspaces
    </button>
  )
  const questions = chat.questions.map((request) => (
    <QuestionCard key={request.id} request={request} asker="The orchestrator" onAnswer={orchestrator.answer} onError={onError} />
  ))
  const fresh = chat.entries.length === 0
  const composer = (
    <Composer
      running={chat.running}
      model={model}
      disabled={!connected}
      placeholder={
        chat.running ? 'The orchestrator is working…' : fresh ? 'Describe the work for your workspaces' : 'Ask for more, or steer it'
      }
      chat={CHAT}
      commands={false}
      label="Message the orchestrator"
      draft={draft}
      onDraft={setDraft}
      onError={onError}
    />
  )

  return (
    <WorkspaceLinkContext.Provider value={openWorkspace}>
      <header className="topbar">
        {onShowSidebar && (
          <button className="icon-button" onClick={onShowSidebar} title={`Show sidebar (${MOD}B)`}>
            <SidebarIcon size={16} />
          </button>
        )}
        <div className="topbar-title">
          <span className="title">Orchestrator</span>
          <span className="workspace">Runs workspaces: checkouts with a session of their own</span>
        </div>
        {enabled !== null && (
          <button
            className="ghost small new-chat"
            onClick={() => void orchestrator.clear().catch(report)}
            disabled={chat.running || fresh}
            title="Start a new orchestrator chat; its workspaces are kept"
          >
            <ComposeIcon size={14} />
            New chat
          </button>
        )}
      </header>
      {enabled === null ? null : enabled === false && fresh && !chat.running ? (
        <div className="hero orchestrator-off">
          <span className="hero-icon">
            <OrchestratorIcon size={28} />
          </span>
          <h1>The orchestrator runs workspaces</h1>
          <p className="hero-hint">
            It makes checkouts of a repository, starts a session in each, gives their agents tasks, watches and steers
            them, and reports back. It needs the workspaces plugin.
          </p>
          {enableButton}
        </div>
      ) : fresh && !chat.running ? (
        <div className="hero">
          <h1>What should your workspaces do?</h1>
          {questions}
          {composer}
          <p className="hero-hint">
            It makes workspaces, gives each one's session a task, and reports back. It asks before destroying one.
          </p>
        </div>
      ) : (
        <>
          <Transcript entries={chat.entries} running={chat.running} empty={null} />
          <footer className="bottom">
            {questions}
            {enabled || chat.running ? (
              composer
            ) : (
              <div className="orchestrator-off-bar">
                <span>The workspaces plugin is off, so the orchestrator cannot work.</span>
                {enableButton}
              </div>
            )}
          </footer>
        </>
      )}
    </WorkspaceLinkContext.Provider>
  )
}
