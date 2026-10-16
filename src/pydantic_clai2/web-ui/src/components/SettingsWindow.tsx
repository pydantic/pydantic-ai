import { useEffect, useRef } from 'react'
import type { MenuField } from '../protocol'
import { McpSection } from './McpSection'
import { MenuForm } from './MenuForm'
import { PluginsPanel } from './PluginsPanel'
import { BackIcon, BrainIcon, FileIcon, InfoIcon, KeyIcon, PuzzleIcon, ServerIcon, SlidersIcon } from './icons'

interface Section {
  id: string
  label: string
  icon: typeof SlidersIcon
  hint: string
  commands: readonly string[]
}

/** The Settings window's sections, in the order its navigation lists them; each is `#settings/ID` in the address.

`commands` are the bare commands whose declared menu opens the section, rather than a dialog of its own. */
export const SECTIONS = [
  {
    id: 'general',
    label: 'General',
    icon: SlidersIcon,
    hint: 'The settings /set changes, saved and applied at once.',
    commands: ['set', 'settings'],
  },
  {
    id: 'models',
    label: 'Models',
    icon: BrainIcon,
    hint: 'The model each prompt uses, and each saved model\'s settings, as /model and /model settings change them.',
    commands: ['model'],
  },
  {
    id: 'plugins',
    label: 'Plugins',
    icon: PuzzleIcon,
    hint: 'Turn plugins on and off and edit their settings, as /plugins does.',
    commands: [],
  },
  {
    id: 'keys',
    label: 'Keys & accounts',
    icon: KeyIcon,
    hint: 'Saved API keys and signed-in accounts, as /keys and /accounts manage them. Secrets never leave CLAI.',
    commands: ['keys', 'login', 'accounts'],
  },
  {
    id: 'mcp',
    label: 'MCP servers',
    icon: ServerIcon,
    hint: 'Connect MCP servers whose tools the agent can use, and manage them, as /mcp does.',
    commands: ['mcp'],
  },
  {
    id: 'system-prompt',
    label: 'System prompt',
    icon: FileIcon,
    hint: "Your own instructions, sent after CLAI's, and the full prompt the model last received.",
    commands: ['system_prompt'],
  },
  {
    id: 'about',
    label: 'About',
    icon: InfoIcon,
    hint: 'The CLAI running now, and the channel /update installs from.',
    commands: ['update'],
  },
] as const satisfies readonly Section[]

/** The Keys & accounts section's cards: each the declared menu of a command. */
const KEYS_MENUS = [
  { command: 'keys', title: 'API keys' },
  { command: 'accounts', title: 'Accounts' },
] as const

export type SettingsSection = (typeof SECTIONS)[number]['id']

/** `/model`'s rows in cards: the model for the next prompt, then the saved models with add and remove, then chains. */
const modelGroup = (field: MenuField) =>
  field.key === 'model' ? 'Model' : field.key.startsWith('chain:') ? 'Fallback chains' : 'Saved models'

/** The section a bare command's menu opens, or `null` for one that opens a dialog of its own. */
export function commandSection(command: string): SettingsSection | null {
  return SECTIONS.find((section) => (section.commands as readonly string[]).includes(command))?.id ?? null
}

/** The section `#settings/ID` names, the first for a bare `#settings`, or `null` when the address names none. */
export function linkedSection(hash: string): SettingsSection | null {
  const found = /^#settings(?:\/([\w-]+))?$/.exec(hash)
  if (!found) return null
  return SECTIONS.find((section) => section.id === found[1])?.id ?? SECTIONS[0].id
}

/** Esc closes, except for an open dialog, which backs out of itself, and a field, which undoes its edit first. */
export function useEscape(onClose: () => void) {
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || document.querySelector('.dialog-overlay')) return
      if ((event.target as HTMLElement).closest('input, textarea, select')) return
      // Capture phase, before the page's own Esc handler can interrupt a turn.
      event.preventDefault()
      onClose()
    }
    window.addEventListener('keydown', close, true)
    return () => window.removeEventListener('keydown', close, true)
  }, [onClose])
}

/** CLAI's settings in one window, as the Codex desktop app shows its own: sections on the left, one shown beside. */
export function SettingsWindow({
  section,
  onSection,
  onClose,
  running,
  pluginsVersion,
  settingsVersion,
  pluginFocus,
  onError,
  onNotice,
}: {
  section: SettingsSection
  onSection: (section: SettingsSection) => void
  onClose: () => void
  running: boolean
  pluginsVersion: number
  settingsVersion: number
  /** The plugin the Plugins section opens on, such as the one a side panel asks to set up. */
  pluginFocus: string | null
  onError: (message: string) => void
  onNotice: (message: string) => void
}) {
  const window_ = useRef<HTMLDivElement>(null)
  useEscape(onClose)
  useEffect(() => window_.current?.focus(), [])
  const shown = SECTIONS.find((entry) => entry.id === section) ?? SECTIONS[0]
  return (
    <div className="settings-window" role="dialog" aria-modal="true" aria-label="Settings" tabIndex={-1} ref={window_}>
      <nav className="settings-nav" aria-label="Settings sections">
        <button className="settings-back" onClick={onClose} title="Back to the session (Esc)">
          <BackIcon size={15} /> Back to app
        </button>
        {SECTIONS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`settings-nav-item ${id === section ? 'active' : ''}`}
            aria-current={id === section ? 'page' : undefined}
            onClick={() => onSection(id)}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </nav>
      <main className={`settings-pane ${section}`}>
        <div className="settings-content">
          <header className="settings-header">
            <h1>{shown.label}</h1>
            <p>{shown.hint}</p>
          </header>
          {section === 'general' && (
            <MenuForm
              target={{ kind: 'commands', name: 'set' }}
              version={settingsVersion}
              grouped
              onError={onError}
              onNotice={onNotice}
            />
          )}
          {section === 'models' && (
            <MenuForm
              target={{ kind: 'commands', name: 'model' }}
              version={settingsVersion}
              grouped
              group={modelGroup}
              opened={(menu) => menu.fields.find((field) => field.key === 'model')?.value}
              onError={onError}
              onNotice={onNotice}
            />
          )}
          {section === 'keys' && (
            <div className="settings-groups">
              {KEYS_MENUS.map(({ command, title }) => (
                <section key={command} className="settings-group">
                  <h3 className="settings-group-title">{title}</h3>
                  <div className="settings-card">
                    <MenuForm
                      target={{ kind: 'commands', name: command }}
                      version={settingsVersion}
                      onError={onError}
                      onNotice={onNotice}
                    />
                  </div>
                </section>
              ))}
            </div>
          )}
          {section === 'plugins' && (
            <PluginsPanel
              version={pluginsVersion}
              running={running}
              initial={pluginFocus}
              onError={onError}
              onNotice={onNotice}
            />
          )}
          {section === 'mcp' && (
            <McpSection version={settingsVersion} running={running} onError={onError} onNotice={onNotice} />
          )}
          {section === 'system-prompt' && (
            <div className="settings-card">
              <MenuForm
                target={{ kind: 'commands', name: 'system_prompt' }}
                version={settingsVersion}
                onError={onError}
                onNotice={onNotice}
              />
            </div>
          )}
          {section === 'about' && (
            <div className="settings-card">
              <MenuForm
                target={{ kind: 'commands', name: 'update' }}
                version={settingsVersion}
                onError={onError}
                onNotice={onNotice}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
