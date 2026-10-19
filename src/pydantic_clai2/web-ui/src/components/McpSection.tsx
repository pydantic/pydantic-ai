import { useCallback, useContext, useEffect, useRef, useState } from 'react'
import { SessionContext, type MenuTarget } from '../api'
import type { MenuField, MenuInfo } from '../protocol'
import { PlusIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

const TARGET: MenuTarget = { kind: 'commands', name: 'mcp' }
const SERVER = 'server.'

/** A server row's value: `/mcp`'s state, or `sign-in` for an OAuth server that is on but signed out. */
const STATES: Record<string, string> = {
  running: 'Running',
  ready: 'Ready',
  stopped: 'Off',
  error: 'Error',
  'sign-in': 'Needs sign-in',
}

/** The trust row's value: the least trusted of the repository's MCP files. */
const TRUST: Record<string, string> = {
  untrusted: 'This repository ships MCP servers. They load once you review and trust its MCP files.',
  changed: 'An MCP file changed since you trusted it, so its servers stay unloaded until you trust it again.',
  trusted: 'Trusted: the servers in these files are loaded.',
}

/** `/mcp`'s declared menu as a list of servers: each one's state, and its actions, which ask in dialogs. */
export function McpSection({
  version,
  running,
  onError,
  onNotice,
}: {
  /** Bumped when a command may have changed the servers; the list is read again. */
  version: number
  /** A turn connects servers, so the list is read again when one ends. */
  running: boolean
  onError: (message: string) => void
  onNotice: (message: string) => void
}) {
  const [menu, setMenu] = useState<MenuInfo | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const request = useRef(0)

  // A command's menu acts on the shown session, so `/model` picks its model, not another session's.
  const session = useContext(SessionContext)
  // The session shown now: an action answered after a switch must not show the last session's servers.
  const shown = useRef(session)
  shown.current = session

  const load = useCallback(async () => {
    const id = ++request.current
    try {
      const found = await session.menu(TARGET)
      if (id === request.current) setMenu(found)
    } catch (error) {
      onError(message(error))
    }
  }, [session, onError])

  useEffect(() => {
    void load()
  }, [load, version, running])

  const act = async (key: string) => {
    setBusy(key)
    const owner = session
    try {
      const result = await owner.runAction(TARGET, key)
      if (owner === shown.current) {
        ++request.current
        setMenu(result.menu)
      }
      const text = result.messages.filter(Boolean).join('\n')
      if (text) onNotice(text)
    } catch (error) {
      onError(message(error))
    } finally {
      setBusy(null)
    }
  }

  if (menu === null) return <p className="plugin-empty">Loading MCP servers…</p>
  const servers = menu.fields.filter((field) => field.key.startsWith(SERVER))
  const trust = menu.fields.find((field) => field.key === 'trust')
  const add = menu.fields.find((field) => field.key === 'add')
  return (
    <div className="settings-groups mcp-section">
      {trust && (
        <div className={`mcp-trust ${trust.value}`}>
          <div className="mcp-trust-text">
            <strong>{trust.label}</strong>
            <span>{TRUST[trust.value] ?? trust.description}</span>
            <code>{trust.note}</code>
          </div>
          <button className="ghost small" disabled={busy !== null} onClick={() => void act(trust.key)}>
            {busy === trust.key ? 'Waiting…' : 'Review…'}
          </button>
        </div>
      )}
      <section className="settings-group">
        <div className="mcp-head">
          <h3 className="settings-group-title">Servers</h3>
          {add && (
            <button
              className="primary small mcp-add"
              disabled={busy !== null}
              title={add.description}
              onClick={() => void act(add.key)}
            >
              <PlusIcon size={13} /> {add.label}
            </button>
          )}
        </div>
        <div className="settings-card">
          {servers.length === 0 ? (
            <p className="mcp-empty">
              No MCP servers yet. Add a local program (stdio), or an http or sse endpoint, and the agent can use its
              tools on your next prompt.
            </p>
          ) : (
            servers.map((field) => (
              <ServerRow
                key={field.key}
                field={field}
                busy={busy === field.key}
                disabled={busy !== null}
                onManage={() => void act(field.key)}
              />
            ))
          )}
        </div>
      </section>
    </div>
  )
}

function ServerRow({
  field,
  busy,
  disabled,
  onManage,
}: {
  field: MenuField
  busy: boolean
  disabled: boolean
  onManage: () => void
}) {
  return (
    <div className="mcp-server">
      <span className={`mcp-dot ${field.value}`} aria-hidden="true" />
      <div className="mcp-info">
        <div className="mcp-name">
          <span>{field.label}</span>
          <span className={`mcp-state ${field.value}`}>{STATES[field.value] ?? field.value}</span>
        </div>
        <code className="mcp-target" title={field.description}>
          {field.description}
        </code>
        {field.note && <p className="setting-note">{field.note}</p>}
      </div>
      <button className="ghost small" disabled={disabled} onClick={onManage} aria-label={`Manage ${field.label}`}>
        {busy ? 'Waiting…' : 'Manage…'}
      </button>
    </div>
  )
}
