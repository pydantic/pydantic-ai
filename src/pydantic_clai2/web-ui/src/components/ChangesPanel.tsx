import { useCallback, useEffect, useRef, useState } from 'react'
import type { SessionApi } from '../api'
import { MOD } from '../platform'
import type { ChangeScope, Changes, FileChange } from '../protocol'
import { DiffView } from './DiffView'
import { ChevronIcon, CloseIcon, DiffIcon } from './icons'

const SCOPES: { scope: ChangeScope; label: string; empty: string }[] = [
  { scope: 'turn', label: 'Last turn', empty: 'The last turn changed no files.' },
  { scope: 'session', label: 'This session', empty: 'This session has changed no files yet.' },
  {
    scope: 'uncommitted',
    label: 'Uncommitted',
    empty: 'No tracked file differs from the last commit. Untracked files are not shown, as in git diff HEAD.',
  },
  {
    scope: 'branch',
    label: 'Branch vs base',
    empty: 'No tracked file differs from where this branch left its base. Untracked files are not shown.',
  },
]

/** Files with more changed lines than this start collapsed, so one large file does not bury the rest. */
const COLLAPSE_LINES = 400
/** The lines each diff mounts until the user asks for the rest, as a file-change card does. */
const PREVIEW_LINES = 60

export const CHANGES_KEY = 'E'

/** The files `session` changed in `scope`. Asks while `enabled`, and again whenever `enabled` stays on.
 * While `hold` is set, as during a turn, an answer already in hand waits; a scope with none still asks once, so opening
 * the panel mid-turn shows what is on disk instead of `Loading…`. That answer stays until the next arrives, and only
 * for the session `owner` it was asked for. Without `diffs`, files come with their counts alone, as for the header's count. */
export function useChanges(
  session: SessionApi,
  scope: ChangeScope,
  owner: string,
  enabled: boolean,
  hold = false,
  diffs = true,
  saved = 0,
) {
  const [found, setFound] = useState<{ owner: string; scope: ChangeScope; changes: Changes | null; error: string | null }>()
  const [version, setVersion] = useState(0)
  const answered = useRef<string | null>(null)
  useEffect(() => {
    if (!enabled) return
    const key = `${owner}\0${scope}\0${String(diffs)}`
    if (hold && answered.current === key) return
    let current = true
    const settle = (changes: Changes | null, error: string | null) => {
      if (!current) return
      answered.current = key
      setFound({ owner, scope, changes, error })
    }
    session.changes(scope, diffs).then(
      (changes) => settle(changes, null),
      (error: unknown) => settle(null, error instanceof Error ? error.message : String(error)),
    )
    return () => {
      current = false
    }
  }, [session, scope, owner, enabled, hold, diffs, version, saved])
  const refresh = useCallback(() => setVersion((value) => value + 1), [])
  const mine = found?.owner === owner && found.scope === scope ? found : undefined
  return { changes: mine?.changes ?? null, error: mine?.error ?? null, refresh }
}

const anchor = (path: string) => `change-${encodeURIComponent(path)}`

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
          change.diff.diff && <DiffView diff={change.diff} preview={PREVIEW_LINES} />
        ))}
    </section>
  )
}

/** What the session changed, beside the conversation: a file list and each file's diff, by scope. */
export function ChangesPanel({
  session,
  owner,
  active,
  running,
  saved,
  onClose,
}: {
  session: SessionApi
  /** The session shown, which the answers belong to. */
  owner: string
  /** Whether the page can ask. Off until the socket is open. */
  active: boolean
  /** A turn is in progress: keep the answer in hand, and still ask once for a scope that has none. */
  running: boolean
  /** Changes whenever the session list does, including once the session has saved a turn; the panel then asks again. */
  saved: number
  onClose: () => void
}) {
  const [scope, setScope] = useState<ChangeScope>(() => {
    const stored = localStorage.getItem('clai-changes-scope')
    return SCOPES.find((entry) => entry.scope === stored)?.scope ?? 'session'
  })
  useEffect(() => localStorage.setItem('clai-changes-scope', scope), [scope])
  const { changes, error, refresh } = useChanges(session, scope, owner, active, running, true, saved)
  // Paths opened or closed by hand; the rest follow their size.
  const [toggled, setToggled] = useState<ReadonlyMap<string, boolean>>(() => new Map())
  const files = changes?.files ?? []
  const isOpen = (change: FileChange) => toggled.get(change.path) ?? change.added + change.removed <= COLLAPSE_LINES
  const jump = (change: FileChange) => {
    setToggled((all) => new Map(all).set(change.path, true))
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
        <button
          className="ghost small changes-refresh"
          onClick={refresh}
          disabled={!active || running}
          title={active && !running ? 'Ask again' : 'Asks again when the turn ends'}
        >
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
        ) : changes === null ? (
          <div className="changes-empty">Loading…</div>
        ) : changes.note ? (
          <div className="changes-empty">{changes.note}</div>
        ) : files.length === 0 ? (
          <div className="changes-empty">{SCOPES.find((entry) => entry.scope === scope)?.empty}</div>
        ) : (
          <>
            <div className="changes-summary">
              <span>
                {files.length} {files.length === 1 ? 'file' : 'files'} changed
                {changes.base && <> against <span className="mono">{changes.base}</span></>}
                {changes.omitted > 0 && `, ${changes.omitted} more not shown`}
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
                onToggle={() => setToggled((all) => new Map(all).set(change.path, !isOpen(change)))}
              />
            ))}
          </>
        )}
      </div>
    </aside>
  )
}
