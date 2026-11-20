import { MenuForm } from './MenuForm'
import { useEscape } from './SettingsWindow'
import { CloseIcon } from './icons'

/** The menu a bare `/COMMAND` typed here opens, in the page instead of the terminal: its rows, saved as they change. */
export function CommandMenu({
  command,
  version,
  onClose,
  onError,
  onNotice,
}: {
  command: string
  version: number
  onClose: () => void
  onError: (message: string) => void
  onNotice: (message: string) => void
}) {
  useEscape(onClose)
  return (
    <div className="overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="command-menu" role="dialog" aria-modal="true" aria-label={`/${command}`}>
        <header className="command-menu-header">
          <code>/{command}</code>
          <span className="plugins-hint">Changes apply now and are saved, as in the terminal.</span>
          <button className="icon-button" onClick={onClose} title="Close (Esc)" aria-label="Close">
            <CloseIcon size={16} />
          </button>
        </header>
        <div className="command-menu-body">
          <MenuForm
            target={{ kind: 'commands', name: command }}
            version={version}
            titled
            onError={onError}
            onNotice={onNotice}
          />
        </div>
      </div>
    </div>
  )
}
