import { useCallback, useEffect, useState } from 'react'
import { api, openWorkspacePage } from '../api'
import type { WorkspaceRow } from '../protocol'
import { BranchIcon, ChevronIcon, FolderIcon } from './icons'

interface Props {
  /** Bumped when the page hears that sessions or plugins changed, which lists the workspaces again. */
  version: number
  onError: (message: string) => void
}

const STATUS: Record<WorkspaceRow['status'], string> = { ready: 'not started', running: 'running', stopped: 'stopped' }

/** The `workspaces` plugin's workspaces, each with its CLAI's status and a button that opens its own page. */
export function WorkspacesSection({ version, onError }: Props) {
  const [rows, setRows] = useState<WorkspaceRow[] | null>(null)
  const [open, setOpen] = useState(true)
  const [opening, setOpening] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const info = await api.workspaces()
      setRows(info.enabled ? info.workspaces : null)
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }, [onError])

  useEffect(() => void load(), [load, version])

  // Workspaces start and stop in other CLAIs too; list them again now and then while the plugin is on.
  useEffect(() => {
    if (rows === null) return
    const interval = window.setInterval(() => void load(), 5000)
    return () => window.clearInterval(interval)
  }, [load, rows])

  if (rows === null) return null

  const openPage = async (row: WorkspaceRow) => {
    setOpening(row.id)
    try {
      await openWorkspacePage(row.id)
      void load()
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    } finally {
      setOpening(null)
    }
  }

  return (
    <section className="project workspaces">
      <button className="project-title" onClick={() => setOpen(!open)} title="The workspaces plugin's checkouts">
        <FolderIcon size={14} />
        <span className="project-name">Workspaces</span>
        {!open && rows.some((row) => row.status === 'running') && <span className="state running" />}
        <ChevronIcon size={12} className={`chevron ${open ? 'down' : ''}`} />
      </button>
      {open && rows.length === 0 && <div className="empty-list">No workspaces. /workspaces create makes one.</div>}
      {open &&
        rows.map((row) => (
          <div key={row.id} className={`session workspace ${row.status}`} title={`${row.repo}\n${row.path}`}>
            <span className={`state ${row.status === 'running' ? 'running' : ''}`} />
            <span className="workspace-main">
              <span className="session-title">{row.id}</span>
              <span className="workspace-meta">
                <BranchIcon size={10} />
                {row.branch} · {STATUS[row.status]}
              </span>
            </span>
            {opening === row.id && <span className="spinner" aria-label="Starting" />}
            <button
              className="workspace-open"
              disabled={opening !== null}
              onClick={() => void openPage(row)}
              title={row.status === 'running' ? 'Open its page' : 'Start its CLAI and open its page'}
            >
              Open
            </button>
          </div>
        ))}
    </section>
  )
}
