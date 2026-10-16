import { useEffect, useRef } from 'react'
import { MenuForm } from './MenuForm'
import { PluginsPanel } from './PluginsPanel'
import { BackIcon, PuzzleIcon, SlidersIcon } from './icons'

/** The Settings window's sections, in the order its navigation lists them; each is `#settings/ID` in the address. */
export const SECTIONS = [
  { id: 'general', label: 'General', icon: SlidersIcon, hint: 'The settings /set changes, saved and applied at once.' },
  {
    id: 'plugins',
    label: 'Plugins',
    icon: PuzzleIcon,
    hint: 'Turn plugins on and off and edit their settings, as /plugins does.',
  },
] as const

export type SettingsSection = (typeof SECTIONS)[number]['id']

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
          {section === 'plugins' && (
            <PluginsPanel
              version={pluginsVersion}
              running={running}
              initial={pluginFocus}
              onError={onError}
              onNotice={onNotice}
            />
          )}
        </div>
      </main>
    </div>
  )
}
