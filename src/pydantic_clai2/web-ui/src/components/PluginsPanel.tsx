import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { PluginAction, PluginRow } from '../protocol'
import { MenuForm } from './MenuForm'
import { SettingsForm } from './SettingsForm'
import { SearchIcon, TerminalIcon } from './icons'

const STATUS: Record<PluginRow['status'], string> = {
  on: 'On',
  off: 'Off',
  idle: 'Not loaded',
  failed: 'Failed',
  included: 'Included',
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Whether the plugin's declaration is on, whatever happened when it loaded. */
const enabled = (row: PluginRow) => row.status === 'on' || row.status === 'idle' || row.status === 'failed'

/** `/plugins` in the browser, the Settings window's Plugins section: every plugin with an on/off switch, and a form
for the settings of the one picked. */
export function PluginsPanel({
  version,
  running,
  initial = null,
  onError,
  onNotice,
}: {
  version: number
  running: boolean
  /** The plugin to show first, such as the one a side panel asks to set up. */
  initial?: string | null
  onError: (message: string) => void
  onNotice: (message: string) => void
}) {
  const [rows, setRows] = useState<PluginRow[] | null>(null)
  const [selected, setSelected] = useState<string | null>(initial)
  const [query, setQuery] = useState('')
  const [pending, setPending] = useState<string | null>(null)
  const request = useRef(0)

  const load = useCallback(async () => {
    const id = ++request.current
    try {
      const found = await api.plugins()
      if (id === request.current) setRows(found)
    } catch (error) {
      onError(message(error))
    }
  }, [onError])

  // Every command, from here or the terminal, may change a plugin: list them again.
  useEffect(() => {
    void load()
  }, [load, version])

  const act = async (row: PluginRow, action: PluginAction) => {
    setPending(row.name)
    try {
      await api.plugin(row.name, action)
    } catch (error) {
      onError(message(error))
    } finally {
      setPending(null)
      void load()
    }
  }

  const openMenu = async (row: PluginRow) => {
    try {
      await api.pluginMenu(row.name)
    } catch (error) {
      onError(message(error))
    }
  }

  const needle = query.trim().toLowerCase()
  const shown = (rows ?? []).filter(
    (row) => !needle || row.name.toLowerCase().includes(needle) || row.description.toLowerCase().includes(needle),
  )
  const current = shown.find((row) => row.name === selected) ?? shown[0] ?? null

  return (
    <div className="plugins-panel">
      <div className="plugins-body">
        <aside className="plugin-list">
          <label className="search">
            <SearchIcon size={14} />
            <input value={query} placeholder="Search plugins" onChange={(event) => setQuery(event.target.value)} />
          </label>
          <nav>
            {rows === null && <div className="empty-list">Loading plugins…</div>}
            {rows !== null && shown.length === 0 && <div className="empty-list">No matching plugins</div>}
            {shown.map((row) => {
              const on = enabled(row)
              const locked = row.serving || row.status === 'included' || pending !== null
              return (
                <div key={row.name} className={`plugin-item ${current?.name === row.name ? 'selected' : ''}`}>
                  <button className="plugin-pick" onClick={() => setSelected(row.name)}>
                    <span className={`plugin-dot ${row.status}`} />
                    <span className="plugin-name">{row.name}</span>
                    <span className="plugin-origin">{row.origin}</span>
                  </button>
                  <button
                    className={`toggle small ${on ? 'on' : ''} ${pending === row.name ? 'busy' : ''}`}
                    role="switch"
                    aria-checked={on}
                    aria-label={`${on ? 'Turn off' : 'Turn on'} ${row.name}`}
                    disabled={locked}
                    title={
                      row.serving
                        ? 'Serves this page: turn it off from the terminal'
                        : row.status === 'included'
                          ? `Included in ${row.included_in ?? 'another plugin'}`
                          : on
                            ? 'Turn off'
                            : 'Turn on'
                    }
                    onClick={() => void act(row, on ? 'disable' : 'enable')}
                  >
                    <span />
                  </button>
                </div>
              )
            })}
          </nav>
        </aside>
        <section className="plugin-detail">
          {current ? (
            <Detail
              key={current.name}
              row={current}
              pending={pending === current.name}
              version={version}
              onReload={() => void act(current, 'reload')}
              onMenu={() => void openMenu(current)}
              onError={onError}
              onNotice={onNotice}
            />
          ) : (
            rows !== null && <p className="plugin-empty">No plugins.</p>
          )}
        </section>
      </div>
      {running && (
        <footer className="plugins-footer">
          The running turn keeps the plugins it started with; changes apply from the next prompt.
        </footer>
      )}
    </div>
  )
}

function Detail({
  row,
  pending,
  version,
  onReload,
  onMenu,
  onError,
  onNotice,
}: {
  row: PluginRow
  pending: boolean
  version: number
  onReload: () => void
  onMenu: () => void
  onError: (message: string) => void
  onNotice: (message: string) => void
}) {
  const on = row.status === 'on'
  return (
    <>
      <div className="plugin-title">
        <h3>{row.name}</h3>
        <span className={`plugin-status ${row.status}`}>{STATUS[row.status]}</span>
        <span className="plugin-origin">{row.origin}</span>
        <div className="plugin-actions">
          {on && !row.serving && (
            <button className="ghost small" onClick={onReload} disabled={pending} title="Import it again and load it">
              Reload
            </button>
          )}
          {on && row.has_menu && !row.settings_menu && row.terminal && (
            <button
              className="ghost small"
              onClick={onMenu}
              title={`Opens /plugins configure ${row.name} in the terminal, where the keyboard drives it`}
            >
              <TerminalIcon size={13} /> Open its menu
            </button>
          )}
        </div>
      </div>
      <code className="plugin-source">{row.source}</code>
      {row.description && <p className="plugin-description">{row.description}</p>}
      {row.status === 'included' && (
        <p className="plugin-note">
          {row.included_in} already provides what this plugin does, so it stays off while {row.included_in} is on.
        </p>
      )}
      {row.error && (
        <div className="plugin-note error">
          <strong>Failed to load.</strong> {row.error}
        </div>
      )}
      {row.notice && <p className="plugin-note warning">{row.notice}</p>}
      {on && row.provides && (
        <p className="plugin-provides">
          <span>Provides</span> {row.provides}
        </p>
      )}
      <h4 className="plugin-section">Settings</h4>
      {on && row.settings_menu ? (
        <MenuForm
          target={{ kind: 'plugins', name: row.name }}
          version={version}
          onError={onError}
          onNotice={onNotice}
        />
      ) : row.form ? (
        <>
          {row.serving && (
            <p className="plugin-note">
              This page runs on {row.name}, so change these from the terminal: any change restarts the web UI.
            </p>
          )}
          {row.has_menu && !row.serving && <MenuNote row={row} />}
          <SettingsForm plugin={row.name} form={row.form} readOnly={row.serving} onError={onError} />
        </>
      ) : (
        <>
          {on && row.has_menu ? (
            <MenuNote row={row} />
          ) : (
            <p className="plugin-empty">{on ? 'It has no settings form.' : 'Turn it on to see its settings.'}</p>
          )}
        </>
      )}
    </>
  )
}

/** Where the plugin's own menu, for sign-ins, keys, and the like, opens: the terminal, which `clai2 --web` lacks. */
function MenuNote({ row }: { row: PluginRow }) {
  return (
    <p className="plugin-note">
      {row.terminal ? (
        <>
          Sign-ins, keys, and other steps live in its own menu: <em>Open its menu</em> shows it in the terminal.
        </>
      ) : (
        <>
          Sign-ins, keys, and other steps live in its own menu, which needs CLAI running in a terminal: start CLAI
          there and run <code>/plugins configure {row.name}</code>, or open this page with <code>/web</code>.
        </>
      )}
    </p>
  )
}
