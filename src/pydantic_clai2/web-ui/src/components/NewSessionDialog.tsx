import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Project } from '../protocol'
import { CheckIcon, FolderIcon } from './icons'

interface Place {
  path: string
  label: string
  detail: string
  /** In a Git checkout, so New worktree… can check out another of its repository. */
  repo: boolean
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function tilde(path: string): string {
  return path.replace(/^\/Users\/[^/]+/, '~')
}

/** The current directory, then each project's checkouts, including Git worktrees without sessions, leaving out removed ones. */
function places(projects: Project[], current: string): Place[] {
  const found = new Map<string, Place>()
  found.set(current, {
    path: current,
    label: current.split('/').filter(Boolean).pop() ?? current,
    detail: tilde(current),
    repo: false,
  })
  for (const project of projects) {
    for (const worktree of project.worktrees) {
      if (worktree.missing) continue
      const label =
        project.worktrees.length > 1 || worktree.label !== project.name
          ? `${project.name} · ${worktree.label}`
          : project.name
      // Outside Git, a project's key is its folder; in Git, the repository's Git directory.
      const repo = project.key !== worktree.root
      found.set(worktree.root, { path: worktree.root, label, detail: tilde(worktree.root), repo })
    }
  }
  return [...found.values()]
}

/** Where a new session runs: a directory CLAI has sessions in or a worktree of its repository, by default the current one,
 * a new worktree of the highlighted place's repository, or any other directory by its path. */
export function NewSessionDialog({
  current,
  onOpened,
  onClose,
}: {
  current: string
  /** Show the new live session, by its conversation ID. */
  onOpened: (id: string) => void
  onClose: () => void
}) {
  const [projects, setProjects] = useState<Project[]>([])
  const [typed, setTyped] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const shown = useMemo(() => places(projects, current), [projects, current])
  const [cursor, setCursor] = useState(0)
  // The place New worktree… checks out another worktree of, while its form shows.
  const [worktreeOf, setWorktreeOf] = useState<Place | null>(null)
  const [branch, setBranch] = useState('')
  const [base, setBase] = useState('')
  const highlighted = shown[cursor]
  const list = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api
      .checkouts()
      .then(setProjects)
      .catch((error: unknown) => setProblem(message(error)))
  }, [])

  useEffect(() => {
    list.current?.focus()
  }, [])

  // Registered once, so a re-render during the key press cannot drop it. In the New worktree… form, it goes back to the list.
  const close = useRef(onClose)
  close.current = worktreeOf ? () => setWorktreeOf(null) : onClose
  useEffect(() => {
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      // Esc is the dialog's alone, not the page's, which would interrupt a turn.
      event.preventDefault()
      event.stopImmediatePropagation()
      close.current()
    }
    window.addEventListener('keydown', cancel, true)
    return () => window.removeEventListener('keydown', cancel, true)
  }, [])

  /** Start a session in `where`, or in the folder it makes first, as New worktree… does. */
  const start = async (where: string | (() => Promise<string>)) => {
    if (sending) return
    setSending(true)
    setProblem(null)
    try {
      const path = typeof where === 'string' ? where : await where()
      const { id } = await api.newSession(path)
      onOpened(id)
      onClose()
    } catch (error) {
      setProblem(message(error))
      setSending(false)
    }
  }

  const keys = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const step = event.key === 'ArrowDown' ? 1 : -1
      setCursor((index) => (index + step + shown.length) % shown.length)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const place = shown[cursor]
      if (place) void start(place.path)
    }
  }

  return (
    <div className="overlay dialog-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="dialog-box new-session-dialog" role="dialog" aria-modal="true" aria-label="New session">
        <h2 className="dialog-title">New session</h2>
        <p className="dialog-description">
          Where should it run? Its tools, AGENTS.md, and project settings come from there.
        </p>
        <div
          className="dialog-options"
          role="listbox"
          ref={list}
          tabIndex={0}
          aria-activedescendant={`place-${cursor}`}
          onKeyDown={keys}
        >
          {shown.map((place, index) => (
            <button
              key={place.path}
              id={`place-${index}`}
              role="option"
              aria-selected={index === cursor}
              className={`dialog-option ${index === cursor ? 'active' : ''}`}
              disabled={sending}
              onMouseEnter={() => setCursor(index)}
              onClick={() => void start(place.path)}
              title={place.path}
            >
              <span className="dialog-option-label">
                <FolderIcon size={13} />
                {place.label}
                {place.path === current && (
                  <span className="dialog-current">
                    <CheckIcon size={12} /> current
                  </span>
                )}
              </span>
              <span className="dialog-option-description">{place.detail}</span>
            </button>
          ))}
        </div>
        {worktreeOf ? (
          <form
            className="dialog-text"
            aria-label={`New worktree of ${worktreeOf.label}`}
            onSubmit={(event) => {
              event.preventDefault()
              void start(async () => (await api.addWorktree(worktreeOf.path, branch.trim(), base.trim())).root)
            }}
          >
            <label className="dialog-label">
              <span>Branch of the new worktree of {worktreeOf.label}</span>
              <input
                className={`setting-input ${problem ? 'invalid' : ''}`}
                spellCheck={false}
                autoFocus
                value={branch}
                placeholder="Generated, as with clai2 --worktree"
                disabled={sending}
                onChange={(event) => {
                  setBranch(event.target.value)
                  setProblem(null)
                }}
              />
            </label>
            <label className="dialog-label">
              <span>Base</span>
              <input
                className="setting-input"
                spellCheck={false}
                value={base}
                placeholder="HEAD"
                disabled={sending}
                onChange={(event) => {
                  setBase(event.target.value)
                  setProblem(null)
                }}
              />
            </label>
            {problem && (
              <p className="setting-error" role="alert">
                {problem}
              </p>
            )}
            <div className="dialog-actions">
              <button
                className="ghost"
                type="button"
                disabled={sending}
                onClick={() => {
                  setWorktreeOf(null)
                  setProblem(null)
                }}
              >
                Back
              </button>
              <button className="primary" type="submit" disabled={sending}>
                Create and start
              </button>
            </div>
          </form>
        ) : (
          <form
            className="dialog-text"
            onSubmit={(event) => {
              event.preventDefault()
              if (typed.trim()) void start(typed.trim())
            }}
          >
            <label className="dialog-label">
              <span>Another directory</span>
              <input
                className={`setting-input ${problem ? 'invalid' : ''}`}
                spellCheck={false}
                value={typed}
                placeholder="/absolute/path/to/a/directory"
                disabled={sending}
                onChange={(event) => {
                  setTyped(event.target.value)
                  setProblem(null)
                }}
              />
            </label>
            {problem && (
              <p className="setting-error" role="alert">
                {problem}
              </p>
            )}
            <div className="dialog-actions">
              {highlighted?.repo && (
                <button
                  className="ghost"
                  type="button"
                  disabled={sending}
                  title={`Check out a new branch of ${highlighted.label} in a worktree, and start there`}
                  onClick={() => {
                    setWorktreeOf(highlighted)
                    setProblem(null)
                  }}
                >
                  New worktree of {highlighted.label}…
                </button>
              )}
              <button className="ghost" type="button" onClick={onClose}>
                Cancel
              </button>
              <button className="primary" type="submit" disabled={sending || !typed.trim()}>
                Start there
              </button>
            </div>
          </form>
        )}
        <p className="dialog-hint">
          Click a directory, make a new worktree of the highlighted one, or type an absolute path. Esc goes back.
        </p>
      </div>
    </div>
  )
}
