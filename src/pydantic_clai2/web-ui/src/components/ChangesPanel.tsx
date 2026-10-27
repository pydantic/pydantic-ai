import { useCallback, useEffect, useState } from 'react'
import type { SessionApi } from '../api'
import { MOD } from '../platform'
import type { ChangeScope, Changes, FileChange } from '../protocol'
import { DiffView } from './DiffView'
import { ChevronIcon, CloseIcon, DiffIcon } from './icons'

const SCOPES: { scope: ChangeScope; label: string; empty: string }[] = [
  { scope: 'turn', label: 'Last turn', empty: 'The last turn changed no files.' },
  { scope: 'session', label: 'This session', empty: 'This session has changed no files yet.' },
  { scope: 'uncommitted', label: 'Uncommitted', empty: 'Nothing is changed since the last commit.' },
  { scope: 'branch', label: 'Branch vs base', empty: 'This branch has no changes against its base.' },
]

/** Files with more changed lines than this start collapsed, so one large file does not bury the rest. */
const COLLAPSE_LINES = 400

export const CHANGES_KEY = 'E'

/** The files `session` changed in `scope`, asked again whenever `key` changes, such as when a turn ends; none while
 * `key` is `null`. The last answer stays until the next arrives. */
export function useChanges(session: SessionApi, scope: ChangeScope, key: string | null) {
  const [found, setFound] = useState<{ changes: Changes | null; error: string | null }>({ changes: null, error: null })
  const [version, setVersion] = useState(0)
  useEffect(() => {
    if (key === null) return
    let current = true
    session.changes(scope).then(
      (changes) => current && setFound({ changes, error: null }),
      (error: unknown) => current && setFound({ changes: null, error: error instanceof Error ? error.message : String(error) }),
    )
    return () => {
      current = false
    }
  }, [session, scope, key, version])
  const refresh = useCallback(() => setVersion((value) => value + 1), [])
  return { ...found, refresh }
}

const anchor = (path: string) => `change-${path.replace(/[^A-Za-z0-9_-]/g, '-')}`

function Counts({ change }: { change: Pick<FileChange, 'added' | 'removed'> }) {
  return (
    <span className="stats">
      <span className="added">+{change.added}</span> <span className="removed">−{change.removed}</span>
    </span>
  )
}

function FileDiff({ change, open, onToggle }: { change: FileChange; open: boolean; onToggle: () => void }) {
  return (
    <section className="changes-file" id={anchor(change.path)}>
      <button className="changes-file-head" onClick={onToggle} aria-expanded={open}>
        <ChevronIcon size={13} className={`changes-chevron ${open ? 'open' : ''}`} />
        <span className="changes-path mono">{change.path}</span>
        <Counts change={change} />
      </button>
      {open &&
        (change.note ? (
          <div className="changes-note">{change.note}</div>
        ) : (
          change.diff.diff && <DiffView diff={change.diff} />
        ))}
    </section>
  )
}

/** What the session changed, beside the conversation: a file list and each file's diff, by scope. */
export function ChangesPanel({
  session,
  sessionKey,
  onClose,
}: {
  session: SessionApi
  /** Changes whenever the files may have: the session shown and its turn's end; `null` while a turn runs. */
  sessionKey: string | null
  onClose: () => void
}) {
  const [scope, setScope] = useState<ChangeScope>(() => {
    const saved = localStorage.getItem('clai-changes-scope')
    return SCOPES.find((entry) => entry.scope === saved)?.scope ?? 'session'
  })
  useEffect(() => localStorage.setItem('clai-changes-scope', scope), [scope])
  const { changes, error, refresh } = useChanges(session, scope, sessionKey)
  // Paths opened or closed by hand; the rest follow their size.
  const [toggled, setToggled] = useState<Record<string, boolean>>({})
  const shown = changes?.scope === scope ? changes : null
  const files = shown?.files ?? []
  const isOpen = (change: FileChange) => toggled[change.path] ?? change.added + change.removed <= COLLAPSE_LINES
  const jump = (change: FileChange) => {
    setToggled((all) => ({ ...all, [change.path]: true }))
    window.setTimeout(() => document.getElementById(anchor(change.path))?.scrollIntoView({ block: 'start' }))
  }
  const total = files.reduce((sum, change) => ({ added: sum.added + change.added, removed: sum.removed + change.removed }), {
    added: 0,
    removed: 0,
  })
  return (
    <aside className="side-panel changes" aria-label="Changes">
      <header className="side-panel-head">
        <DiffIcon size={15} />
        <span className="side-panel-title">Changes</span>
        <button className="ghost small changes-refresh" onClick={refresh} title="Ask again">
          Refresh
        </button>
        <button className="icon-button" onClick={onClose} title={`Hide the changes (${MOD}⇧${CHANGES_KEY})`} aria-label="Hide the changes">
          <CloseIcon size={14} />
        </button>
      </header>
      <div className="changes-scopes" role="tablist" aria-label="Which changes">
        {SCOPES.map((entry) => (
          <button
            key={entry.scope}
            role="tab"
            aria-selected={scope === entry.scope}
            className={scope === entry.scope ? 'active' : ''}
            onClick={() => setScope(entry.scope)}
          >
            {entry.label}
          </button>
        ))}
      </div>
      <div className="changes-body">
        {error ? (
          <div className="changes-empty">{error}</div>
        ) : shown === null ? (
          <div className="changes-empty">Loading…</div>
        ) : shown.note ? (
          <div className="changes-empty">{shown.note}</div>
        ) : files.length === 0 ? (
          <div className="changes-empty">{SCOPES.find((entry) => entry.scope === scope)?.empty}</div>
        ) : (
          <>
            <div className="changes-summary">
              <span>
                {files.length} {files.length === 1 ? 'file' : 'files'} changed
                {shown.base && <> against <span className="mono">{shown.base}</span></>}
              </span>
              <Counts change={total} />
            </div>
            <ul className="changes-files" aria-label="Changed files">
              {files.map((change) => (
                <li key={change.path}>
                  <button onClick={() => jump(change)} title={`Show ${change.path}`}>
                    <span className={`changes-status status-${change.status}`}>{change.status}</span>
                    <span className="changes-path mono">{change.path}</span>
                    <Counts change={change} />
                  </button>
                </li>
              ))}
            </ul>
            {files.map((change) => (
              <FileDiff
                key={change.path}
                change={change}
                open={isOpen(change)}
                onToggle={() => setToggled((all) => ({ ...all, [change.path]: !isOpen(change) }))}
              />
            ))}
          </>
        )}
      </div>
    </aside>
  )
}
