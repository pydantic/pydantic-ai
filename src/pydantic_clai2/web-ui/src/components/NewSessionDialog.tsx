import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Project } from '../protocol'
import { CheckIcon, FolderIcon } from './icons'

interface Place {
  path: string
  label: string
  detail: string
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function tilde(path: string): string {
  return path.replace(/^\/Users\/[^/]+/, '~')
}

/** The current directory if given, then each project's checkouts, including Git worktrees without sessions, leaving out removed ones. */
function places(projects: Project[], current?: string): Place[] {
  const found = new Map<string, Place>()
  if (current !== undefined) {
    found.set(current, {
      path: current,
      label: current.split('/').filter(Boolean).pop() ?? current,
      detail: tilde(current),
    })
  }
  for (const project of projects) {
    for (const worktree of project.worktrees) {
      if (worktree.missing) continue
      const label =
        project.worktrees.length > 1 || worktree.label !== project.name
          ? `${project.name} · ${worktree.label}`
          : project.name
      found.set(worktree.root, { path: worktree.root, label, detail: tilde(worktree.root) })
    }
  }
  return [...found.values()]
}

/** Where a new session runs: a directory CLAI has sessions in or a worktree of its repository, by default the current one, or any other by its path, completed as it is typed. */
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
  // While a path is typed, the directories it completes to, so a repository without sessions and its worktrees are reachable.
  const [completed, setCompleted] = useState<Project[] | null>(null)
  const shown = useMemo(
    () => (completed === null ? places(projects, current) : places(completed)),
    [completed, projects, current],
  )
  const [cursor, setCursor] = useState(0)
  const list = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api
      .checkouts()
      .then(setProjects)
      .catch((error: unknown) => setProblem(message(error)))
  }, [])

  useEffect(() => {
    const path = typed.trim()
    if (!path) {
      setCompleted(null)
      return
    }
    let stale = false
    const timer = window.setTimeout(() => {
      api
        .directories(path)
        .then((found) => !stale && setCompleted(found))
        .catch(() => !stale && setCompleted([]))
    }, 150)
    return () => {
      stale = true
      window.clearTimeout(timer)
    }
  }, [typed])

  useEffect(() => setCursor(0), [shown])

  useEffect(() => {
    list.current?.focus()
  }, [])

  // Registered once, so a re-render during the key press cannot drop it.
  const close = useRef(onClose)
  close.current = onClose
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

  const start = async (path: string) => {
    if (sending) return
    setSending(true)
    setProblem(null)
    try {
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
      if (!shown.length) return
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
            <button className="ghost" type="button" onClick={onClose}>
              Cancel
            </button>
            <button className="primary" type="submit" disabled={sending || !typed.trim()}>
              Start there
            </button>
          </div>
        </form>
        <p className="dialog-hint">Click a directory, or type an absolute path to find others. Esc goes back.</p>
      </div>
    </div>
  )
}
