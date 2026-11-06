import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Project } from '../protocol'
import { BranchIcon, CheckIcon, ChevronIcon, FolderIcon } from './icons'

interface Place {
  /** Unique among the rows: a project's key, or a worktree's root under it. */
  key: string
  path: string
  label: string
  detail: string
  /** In a Git checkout, so New worktree… can check out another of its repository. */
  repo: boolean
  /** On a project's row, how many worktrees it unfolds to, if more than one. */
  worktrees: number
  /** On a worktree's row, its project's key, and the branch it shows beneath the project. */
  parent?: string
  branch?: string
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function tilde(path: string): string {
  return path.replace(/^\/Users\/[^/]+/, '~')
}

/** A row per project, the current one first, which starts in its current checkout, else its main one, else its first;
 * an unfolded project's checkouts, including Git worktrees without sessions, follow it, leaving out removed ones. The
 * current directory comes first on its own when no project holds it. */
function places(projects: Project[], unfolded: (key: string) => boolean, current?: string): Place[] {
  const found: Place[] = []
  const holds = (project: Project) =>
    project.worktrees.some((worktree) => !worktree.missing && worktree.root === current)
  if (current !== undefined && !projects.some(holds)) {
    const label = current.split('/').filter(Boolean).pop() ?? current
    found.push({ key: current, path: current, label, detail: tilde(current), repo: false, worktrees: 0 })
  }
  for (const project of [...projects.filter(holds), ...projects.filter((project) => !holds(project))]) {
    const live = project.worktrees.filter((worktree) => !worktree.missing)
    const start =
      live.find((worktree) => worktree.root === current) ??
      live.find((worktree) => worktree.root === project.folder) ??
      live[0]
    if (!start) continue
    // Outside Git, a project's key is its folder; in Git, the repository's Git directory.
    const repo = project.key !== start.root
    const many = live.length > 1
    found.push({
      key: project.key,
      path: start.root,
      label: project.name,
      detail: tilde(start.root),
      repo,
      worktrees: many ? live.length : 0,
    })
    if (!many || !unfolded(project.key)) continue
    for (const worktree of live) {
      found.push({
        key: `${project.key}\n${worktree.root}`,
        path: worktree.root,
        label: `${project.name} · ${worktree.label}`,
        detail: tilde(worktree.root),
        repo,
        worktrees: 0,
        parent: project.key,
        branch: worktree.label,
      })
    }
  }
  return found
}

/** Where a new session runs: a project CLAI has sessions in, by default the current one, or one of its worktrees, which
 * its chevron unfolds; a new worktree of the highlighted place's repository; or any other directory by its path,
 * completed as it is typed, with each repository's worktrees unfolded. */
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
  // The projects folded the other way from how they start: folded, or unfolded while a path is typed.
  const [toggled, setToggled] = useState<ReadonlySet<string>>(new Set())
  const unfolded = useCallback((key: string) => (completed !== null) !== toggled.has(key), [completed, toggled])
  const shown = useMemo(
    () => (completed === null ? places(projects, unfolded, current) : places(completed, unfolded)),
    [completed, projects, current, unfolded],
  )
  // The highlighted row's key, kept as rows unfold and fold around it; the first row until the arrow keys move it.
  const [cursorKey, setCursorKey] = useState<string | null>(null)
  const cursor = Math.max(
    0,
    shown.findIndex((place) => place.key === cursorKey),
  )
  // The place New worktree… checks out another worktree of, while its form shows.
  const [worktreeOf, setWorktreeOf] = useState<Place | null>(null)
  const [branch, setBranch] = useState('')
  const [base, setBase] = useState('')
  // The worktree a submit made, by what it asked for, so a retry after the session failed to start reuses it.
  const made = useRef<{ asked: string; root: string } | null>(null)
  const highlighted = shown[cursor]
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

  useEffect(() => {
    setCursorKey(null)
    setToggled(new Set())
  }, [completed, projects])

  const fold = (place: Place, open: boolean) => {
    if (unfolded(place.key) === open) return
    setToggled((was) => {
      const next = new Set(was)
      if (!next.delete(place.key)) next.add(place.key)
      return next
    })
  }

  useEffect(() => {
    list.current?.focus()
  }, [])

  // Registered once, so a re-render during the key press cannot drop it. In the New worktree… form, it goes back to the list.
  const close = useRef(onClose)
  // Not while its request runs, as Back is disabled then: its outcome belongs to the form.
  close.current = worktreeOf ? () => !sending && setWorktreeOf(null) : onClose
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

  const chooseWorktree = (place: Place) => {
    setWorktreeOf(place)
    setProblem(null)
  }

  const keys = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      if (!shown.length) return
      const step = event.key === 'ArrowDown' ? 1 : -1
      setCursorKey(shown[(cursor + step + shown.length) % shown.length]!.key)
    } else if (event.key === 'ArrowRight' && highlighted?.worktrees) {
      // As in a tree: → unfolds a folded project, and steps into an unfolded one's first worktree.
      event.preventDefault()
      if (unfolded(highlighted.key)) setCursorKey(shown[cursor + 1]!.key)
      else fold(highlighted, true)
    } else if (event.key === 'ArrowLeft' && highlighted) {
      event.preventDefault()
      if (highlighted.parent !== undefined) setCursorKey(highlighted.parent)
      else if (highlighted.worktrees) fold(highlighted, false)
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
          {shown.map((place, index) => {
            const open = place.worktrees > 0 && unfolded(place.key)
            return (
              // The pointer crosses other rows on its way to the buttons below, so hovering one does not move the
              // keyboard's cursor, which those buttons act on; a row of a Git checkout has its own New worktree… instead:
              // a button in the listbox beside the row's option, out of the Tab order but named for voice control.
              // So has a project of several worktrees its chevron, which unfolds them as → does.
              <div
                key={place.key}
                className={`dialog-option-row ${place.parent !== undefined ? 'nested' : ''} ${index === cursor ? 'active' : ''}`}
                role="none"
              >
                {place.worktrees > 0 ? (
                  <button
                    className="icon-button dialog-option-fold"
                    type="button"
                    tabIndex={-1}
                    aria-expanded={open}
                    aria-label={`${open ? 'Hide' : 'Show'} the worktrees of ${place.label}`}
                    title={`${open ? 'Hide' : 'Show'} its worktrees`}
                    onClick={() => {
                      setCursorKey(place.key)
                      fold(place, !open)
                    }}
                  >
                    <ChevronIcon size={12} className={`chevron ${open ? 'down' : ''}`} />
                  </button>
                ) : (
                  place.parent === undefined && <span className="dialog-option-fold" />
                )}
                <button
                  id={`place-${index}`}
                  role="option"
                  aria-selected={index === cursor}
                  // A worktree is named with its project; a project of several says whether they show, as an option
                  // cannot carry `aria-expanded`.
                  aria-label={
                    place.branch !== undefined
                      ? place.label
                      : place.worktrees > 0
                        ? `${place.label}, ${place.worktrees} worktrees ${open ? 'shown' : 'hidden'}`
                        : undefined
                  }
                  className={`dialog-option ${index === cursor ? 'active' : ''}`}
                  disabled={sending}
                  onClick={() => void start(place.path)}
                  title={place.path}
                >
                  <span className="dialog-option-label">
                    {place.branch === undefined ? <FolderIcon size={13} /> : <BranchIcon size={13} />}
                    {place.branch ?? place.label}
                    {place.worktrees > 0 && <span className="dialog-option-count">{place.worktrees} worktrees</span>}
                    {place.path === current && (
                      <span className="dialog-current">
                        <CheckIcon size={12} /> current
                      </span>
                    )}
                  </span>
                  <span className="dialog-option-description">{place.detail}</span>
                </button>
                {place.repo && !worktreeOf && (
                  <button
                    className="ghost small dialog-option-worktree"
                    type="button"
                    disabled={sending}
                    aria-label={`New worktree of ${place.label}…`}
                    title={`Check out a new branch of ${place.label} in a worktree, and start there`}
                    // For the pointer: the keyboard reaches the one below, for the cursor's row.
                    tabIndex={-1}
                    onClick={() => chooseWorktree(place)}
                  >
                    New worktree…
                  </button>
                )}
              </div>
            )
          })}
        </div>
        {worktreeOf ? (
          <form
            className="dialog-text"
            aria-label={`New worktree of ${worktreeOf.label}`}
            onSubmit={(event) => {
              event.preventDefault()
              const asked = JSON.stringify([worktreeOf.path, branch.trim(), base.trim()])
              void start(async () => {
                if (made.current?.asked !== asked) {
                  const { root } = await api.addWorktree(worktreeOf.path, branch.trim(), base.trim())
                  made.current = { asked, root }
                }
                return made.current.root
              })
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
                className={`setting-input ${problem ? 'invalid' : ''}`}
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
                  onClick={() => chooseWorktree(highlighted)}
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
          Click a project or its New worktree…, or its chevron (→ and ←) to show or hide its worktrees, or type an
          absolute path to find others. Esc goes back.
        </p>
      </div>
    </div>
  )
}
