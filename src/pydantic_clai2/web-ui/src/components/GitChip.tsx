import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { gitApi, gitCounts, type GitApi, type GitStatus } from '../git'
import { BranchIcon, ExternalIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** The header's git chip for the live session `session`: its branch and counts, opening the Commit, Push, and PR dialog.

It shows nothing outside a Git repository. It asks again whenever `refresh` changes, as when a turn ends, and when the
window regains focus, so changes made elsewhere show too.
 */
export function GitChip({
  session,
  refresh,
  onNotice,
}: {
  session: string
  refresh: number
  onNotice: (message: string) => void
}) {
  const git = useMemo(() => gitApi(session), [session])
  const [status, setStatus] = useState<GitStatus | null>(null)
  const [open, setOpen] = useState(false)
  const opened = useRef(false)
  opened.current = open
  // Only the latest answer is shown, however the requests resolve.
  const request = useRef(0)
  const load = useCallback(async () => {
    const mine = ++request.current
    try {
      const found = await git.status()
      if (mine === request.current) setStatus(found)
    } catch {
      // A session that closed, or a folder that is gone: no chip, unless the dialog shows what the user typed.
      if (mine === request.current && !opened.current) setStatus(null)
    }
  }, [git])
  useEffect(() => {
    setStatus(null)
    setOpen(false)
  }, [git])
  useEffect(() => {
    void load()
  }, [load, refresh])
  useEffect(() => {
    const again = () => void load()
    window.addEventListener('focus', again)
    return () => window.removeEventListener('focus', again)
  }, [load])
  const close = useCallback(() => {
    setOpen(false)
    void load()
  }, [load])

  if (!status) return null
  const branch = status.branch ?? 'detached HEAD'
  return (
    <>
      <button
        className="git-chip"
        onClick={() => setOpen(true)}
        title={`${branch}${status.upstream ? ` → ${status.upstream}` : ''}: commit, push, or open a pull request`}
      >
        <BranchIcon size={13} />
        <span className="git-branch">{branch}</span>
        <span className="git-counts">{gitCounts(status)}</span>
      </button>
      {open && (
        <GitDialog
          git={git}
          status={status}
          onStatus={(fresh) => {
            // Newer than any status still being asked for.
            request.current++
            setStatus(fresh)
          }}
          onRefresh={() => void load()}
          onClose={close}
          onNotice={onNotice}
        />
      )}
    </>
  )
}

type View = 'changes' | 'pull-request'

/** Commit the chosen files with a message the model drafts, push the branch, or open a pull request for it. */
function GitDialog({
  git,
  status,
  onStatus,
  onRefresh,
  onClose,
  onNotice,
}: {
  git: GitApi
  status: GitStatus
  onStatus: (status: GitStatus | null) => void
  /** Ask for the repository's status again, as after opening a pull request, which may push. */
  onRefresh: () => void
  onClose: () => void
  onNotice: (message: string) => void
}) {
  const [view, setView] = useState<View>('changes')
  // Whether the pull request view was opened, which drafts it; it stays mounted after.
  const [visited, setVisited] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Registered once, so a re-render during the key press cannot drop it.
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  useEffect(() => {
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      // Esc is the dialog's alone, not the page's, which would interrupt a turn.
      event.preventDefault()
      event.stopImmediatePropagation()
      closeRef.current()
    }
    window.addEventListener('keydown', cancel, true)
    return () => window.removeEventListener('keydown', cancel, true)
  }, [])

  const act = async (action: () => Promise<void>) => {
    setBusy(true)
    setProblem(null)
    setDone(null)
    try {
      await action()
    } catch (error) {
      setProblem(message(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay dialog-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="dialog-box git-dialog" role="dialog" aria-modal="true" aria-label="Git">
        <h2 className="dialog-title">
          <BranchIcon size={15} /> {status.branch ?? 'detached HEAD'}
          <span className="git-counts">{gitCounts(status)}</span>
        </h2>
        {/* Both stay mounted once shown, so switching keeps what was typed and drafts each only once. */}
        <div className="git-view" hidden={view !== 'changes'}>
          <Changes
            git={git}
            status={status}
            busy={busy}
            act={act}
            onStatus={onStatus}
            onDone={(text) => {
              setDone(text)
              onNotice(text)
            }}
            onPullRequest={() => {
              setProblem(null)
              setDone(null)
              setVisited(true)
              setView('pull-request')
            }}
          />
        </div>
        {visited && (
          <div className="git-view" hidden={view !== 'pull-request'}>
            <PullRequest git={git} busy={busy} act={act} onOpened={onRefresh} onBack={() => setView('changes')} />
          </div>
        )}
        {problem && (
          <pre className="setting-error git-problem" role="alert">
            {problem}
          </pre>
        )}
        {done && (
          <p className="git-done" role="status">
            {done}
          </p>
        )}
        <div className="dialog-actions">
          <button className="ghost" type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function Changes({
  git,
  status,
  busy,
  act,
  onStatus,
  onDone,
  onPullRequest,
}: {
  git: GitApi
  status: GitStatus
  busy: boolean
  act: (action: () => Promise<void>) => Promise<void>
  onStatus: (status: GitStatus | null) => void
  onDone: (text: string) => void
  onPullRequest: () => void
}) {
  const paths = useMemo(() => status.files.map((file) => file.path), [status.files])
  const [chosen, setChosen] = useState<ReadonlySet<string>>(() => new Set(paths))
  // Files that appear after the dialog opened start chosen too; ones committed or reverted drop out.
  const known = useRef<ReadonlySet<string>>(new Set(paths))
  useEffect(() => {
    setChosen((current) => new Set(paths.filter((path) => current.has(path) || !known.current.has(path))))
    known.current = new Set(paths)
  }, [paths])
  const [text, setText] = useState('')
  const [drafting, setDrafting] = useState(false)
  const [draftProblem, setDraftProblem] = useState<string | null>(null)
  // The model's draft never replaces what the user typed.
  const typed = useRef(false)
  // Only the latest draft is used: one still on its way when a commit lands is for files already committed.
  const drafts = useRef(0)
  // The files the message in the box was drafted for, so a selection that no longer matches can say so.
  const [draftedFor, setDraftedFor] = useState<ReadonlySet<string> | null>(null)
  const draft = useCallback(
    async (files: string[] | null, shown: readonly string[]) => {
      const mine = ++drafts.current
      setDrafting(true)
      setDraftProblem(null)
      try {
        const { message: drafted } = await git.draftCommit(files)
        if (mine === drafts.current && !typed.current) {
          setText(drafted)
          setDraftedFor(new Set(shown))
        }
      } catch (error) {
        if (mine === drafts.current) setDraftProblem(message(error))
      } finally {
        if (mine === drafts.current) setDrafting(false)
      }
    },
    [git],
  )
  const hasChanges = paths.length > 0
  useEffect(() => {
    if (hasChanges) void draft(null, [...known.current])
  }, [draft, hasChanges])
  const [confirmed, setConfirmed] = useState(false)

  const selected = paths.filter((path) => chosen.has(path))
  // Only an untouched draft is known to describe other files: typing drops `draftedFor`.
  const stale =
    draftedFor !== null &&
    selected.length > 0 &&
    (draftedFor.size !== selected.length || selected.some((path) => !draftedFor.has(path)))
  const commit = () =>
    act(async () => {
      // The files shown, not whatever has changed since.
      const done = await git.commit(text, selected)
      drafts.current++
      setDrafting(false)
      typed.current = false
      setText('')
      setDraftedFor(null)
      onStatus(done.status)
      onDone(done.message)
    })
  const push = () =>
    act(async () => {
      const done = await git.push(confirmed)
      onStatus(done.status)
      onDone(done.message)
    })
  const canPush = status.push === null && (!status.default_branch || confirmed)

  return (
    <>
      <section className="git-section" aria-label="Commit">
        {hasChanges ? (
          <>
            <div className="git-files" role="group" aria-label="Changed files">
              {status.files.map((file) => (
                <label key={`${file.status}:${file.path}`} className="git-file" title={file.original ? `${file.original} → ${file.path}` : file.path}>
                  <input
                    type="checkbox"
                    checked={chosen.has(file.path)}
                    disabled={busy}
                    onChange={(event) =>
                      setChosen((current) => {
                        const next = new Set(current)
                        if (event.target.checked) next.add(file.path)
                        else next.delete(file.path)
                        return next
                      })
                    }
                  />
                  <span className={`git-file-status status-${file.status === '?' ? 'new' : file.status}`}>
                    {file.status === '?' ? 'N' : file.status}
                  </span>
                  <span className="git-file-path mono">{file.path}</span>
                </label>
              ))}
            </div>
            <label className="dialog-label">
              <span className="git-message-label">
                Commit message
                <button
                  className="ghost small"
                  type="button"
                  disabled={busy || drafting || selected.length === 0}
                  onClick={() => {
                    typed.current = false
                    void draft(selected.length === paths.length ? null : selected, selected)
                  }}
                >
                  {drafting ? 'Drafting…' : 'Draft again'}
                </button>
              </span>
              <textarea
                className="setting-input git-message"
                aria-label="Commit message"
                rows={5}
                spellCheck={false}
                value={text}
                placeholder={drafting ? 'The model is drafting a message…' : 'Summarize the change'}
                disabled={busy}
                onChange={(event) => {
                  typed.current = true
                  setDraftedFor(null)
                  setText(event.target.value)
                }}
              />
            </label>
            {stale && !drafting && (
              <p className="dialog-hint">
                This message was drafted for {draftedFor.size} {draftedFor.size === 1 ? 'file' : 'files'}; {selected.length}{' '}
                {selected.length === 1 ? 'is' : 'are'} selected. Draft again to match.
              </p>
            )}
            {draftProblem && <p className="dialog-hint">Could not draft a message: {draftProblem}</p>}
            <div className="dialog-actions">
              <button
                className="primary"
                type="button"
                disabled={busy || selected.length === 0 || !text.trim()}
                onClick={() => void commit()}
              >
                Commit {selected.length === paths.length ? 'all' : `${selected.length} of ${paths.length}`}
              </button>
            </div>
          </>
        ) : (
          <p className="dialog-description">No uncommitted changes.</p>
        )}
      </section>
      <section className="git-section" aria-label="Push and pull request">
        {status.default_branch && status.push === null && (
          <label className="git-confirm">
            <input type="checkbox" checked={confirmed} disabled={busy} onChange={(event) => setConfirmed(event.target.checked)} />
            Push to {status.branch}, the default branch
          </label>
        )}
        {status.push && <p className="dialog-hint">{status.push}</p>}
        {!status.push && status.pull_request && <p className="dialog-hint">{status.pull_request}</p>}
        <div className="dialog-actions git-remote-actions">
          <button
            className="ghost"
            type="button"
            disabled={busy || !canPush}
            title={status.push ?? undefined}
            onClick={() => void push()}
          >
            Push{status.upstream ? '' : ' -u origin'}
          </button>
          <button
            className="ghost"
            type="button"
            disabled={busy || status.pull_request !== null}
            title={status.pull_request ?? undefined}
            onClick={onPullRequest}
          >
            Create pull request…
          </button>
        </div>
      </section>
    </>
  )
}

function PullRequest({
  git,
  busy,
  act,
  onOpened,
  onBack,
}: {
  git: GitApi
  busy: boolean
  act: (action: () => Promise<void>) => Promise<void>
  onOpened: () => void
  onBack: () => void
}) {
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [drafting, setDrafting] = useState(true)
  const [draftProblem, setDraftProblem] = useState<string | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  // The draft fills only the fields not typed in yet.
  const typedTitle = useRef(false)
  const typedBody = useRef(false)
  useEffect(() => {
    let current = true
    git
      .draftPullRequest()
      .then((draft) => {
        if (!current) return
        if (!typedTitle.current) setTitle(draft.title)
        if (!typedBody.current) setBody(draft.body)
      })
      .catch((error: unknown) => current && setDraftProblem(message(error)))
      .finally(() => current && setDrafting(false))
    return () => {
      current = false
    }
  }, [git])

  if (url)
    return (
      <section className="git-section" aria-label="Pull request">
        <p className="dialog-description">
          Opened{' '}
          <a href={url} target="_blank" rel="noreferrer noopener">
            {url} <ExternalIcon size={12} />
          </a>
        </p>
      </section>
    )
  return (
    <section className="git-section" aria-label="Pull request">
      <label className="dialog-label">
        <span>Title</span>
        <input
          className="setting-input"
          value={title}
          placeholder={drafting ? 'The model is drafting a title…' : 'What the pull request does'}
          disabled={busy}
          onChange={(event) => {
            typedTitle.current = true
            setTitle(event.target.value)
          }}
        />
      </label>
      <label className="dialog-label">
        <span>Description</span>
        <textarea
          className="setting-input git-message"
          rows={8}
          value={body}
          disabled={busy}
          onChange={(event) => {
            typedBody.current = true
            setBody(event.target.value)
          }}
        />
      </label>
      {draftProblem && <p className="dialog-hint">Could not draft it: {draftProblem}</p>}
      <div className="dialog-actions">
        <button className="ghost" type="button" disabled={busy} onClick={onBack}>
          Back
        </button>
        <button
          className="primary"
          type="button"
          disabled={busy || !title.trim()}
          onClick={() =>
            void act(async () => {
              setUrl((await git.openPullRequest(title, body)).url)
              onOpened()
            })
          }
        >
          {busy ? 'Opening…' : 'Open pull request'}
        </button>
      </div>
    </section>
  )
}
