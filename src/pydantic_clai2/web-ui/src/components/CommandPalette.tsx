import { useContext, useEffect, useMemo, useRef, useState } from 'react'
import { SessionContext } from '../api'
import type { CompletionInfo } from '../protocol'
import { SearchIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

interface Entry {
  name: string
  description: string
  /** The arguments its description shows after `/NAME`, such as `[@model] PROMPT`. */
  hint: string
  /** Whether it needs one of them, so Enter puts it in the composer instead of running it. */
  needsArguments: boolean
}

/** What a description shows after `/NAME `, such as `[VALUE|reset]` in `/effort [VALUE|reset]`; empty when none.
 *
 * Only arguments count, in brackets or capitals: `add [NAME]` in `/model add [NAME]` names a subcommand. */
export function argumentHint(name: string, description: string): string {
  const hint = new RegExp(`/${name} ([^,;:()]+)`).exec(description)?.[1]?.trim() ?? ''
  return /^[[A-Z]/.test(hint) ? hint : ''
}

/** Whether a hint names an argument outside brackets, as `PROMPT` in `[@model] PROMPT` is. */
const required = (hint: string) => /\b[A-Z][A-Z_]+\b/.test(hint.replace(/\[[^\]]*\]/g, ''))

function entry(completion: CompletionInfo): Entry {
  const description = completion.meta ?? ''
  const hint = argumentHint(completion.text, description)
  return { name: completion.text, description, hint, needsArguments: required(hint) }
}

/** Every command this CLAI offers, from its own registry, filterable: Enter runs one as typing it bare would. */
export function CommandPalette({
  onRun,
  onInsert,
  onClose,
  onError,
}: {
  /** Run a bare command, as the composer runs it: here, in its menu or Settings section, or at the prompt. */
  onRun: (name: string) => void
  /** Put text in the composer, for a command that needs arguments. */
  onInsert: (text: string) => void
  onClose: () => void
  onError: (message: string) => void
}) {
  const [commands, setCommands] = useState<Entry[] | null>(null)
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const list = useRef<HTMLDivElement>(null)
  // The shown session's commands: its plugins' and skills' come from its own folder.
  const session = useContext(SessionContext)

  useEffect(() => {
    let current = true
    session
      .complete('/')
      .then((found) => current && setCommands(found.map(entry).sort((a, b) => a.name.localeCompare(b.name))))
      .catch((error: unknown) => current && onError(message(error)))
    return () => {
      current = false
    }
  }, [session, onError])

  const needle = query.trim().toLowerCase().replace(/^\//, '')
  const shown = useMemo(() => {
    const all = commands ?? []
    if (!needle) return all
    const matches = all.filter(
      (command) => command.name.toLowerCase().includes(needle) || command.description.toLowerCase().includes(needle),
    )
    // Names that start with the query first, then other name matches, then description matches.
    const rank = (command: Entry) =>
      command.name.toLowerCase().startsWith(needle) ? 0 : command.name.toLowerCase().includes(needle) ? 1 : 2
    return matches.sort((a, b) => rank(a) - rank(b))
  }, [commands, needle])

  useEffect(() => {
    list.current?.querySelector<HTMLElement>('.palette-item.active')?.scrollIntoView({ block: 'nearest' })
  }, [cursor, shown])

  // Esc is the palette's alone, before the page's own handler can interrupt a turn.
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || document.querySelector('.dialog-overlay')) return
      event.preventDefault()
      event.stopImmediatePropagation()
      onClose()
    }
    window.addEventListener('keydown', close, true)
    return () => window.removeEventListener('keydown', close, true)
  }, [onClose])

  const choose = (command: Entry | undefined, insert = false) => {
    if (!command) return
    onClose()
    if (insert || command.needsArguments) onInsert(`/${command.name} `)
    else onRun(command.name)
  }

  const keys = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const step = event.key === 'ArrowDown' ? 1 : -1
      setCursor((current) => (shown.length ? (current + step + shown.length) % shown.length : 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(shown[cursor])
    } else if (event.key === 'Tab') {
      event.preventDefault()
      choose(shown[cursor], true)
    }
  }

  return (
    <div className="overlay palette-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Commands" onKeyDown={keys}>
        <label className="palette-search">
          <SearchIcon size={15} />
          <input
            autoFocus
            value={query}
            placeholder="Search commands"
            spellCheck={false}
            aria-controls="palette-list"
            aria-activedescendant={shown[cursor] ? `palette-${shown[cursor].name}` : undefined}
            onChange={(event) => {
              setQuery(event.target.value)
              setCursor(0)
            }}
          />
        </label>
        <div className="palette-list" id="palette-list" role="listbox" ref={list}>
          {commands === null && <div className="empty-list">Loading commands…</div>}
          {commands !== null && shown.length === 0 && <div className="empty-list">No command matches</div>}
          {shown.map((command, index) => (
            <button
              key={command.name}
              id={`palette-${command.name}`}
              role="option"
              aria-selected={index === cursor}
              className={`palette-item ${index === cursor ? 'active' : ''}`}
              tabIndex={-1}
              onMouseMove={() => setCursor(index)}
              onClick={() => choose(command)}
            >
              <span className="palette-name">
                /{command.name}
                {command.hint && <span className="palette-hint">{command.hint}</span>}
              </span>
              <span className="palette-description">{command.description}</span>
            </button>
          ))}
        </div>
        <footer className="palette-foot">
          <span>
            <kbd>↵</kbd> run
          </span>
          <span>
            <kbd>Tab</kbd> edit in the composer
          </span>
          <span>
            <kbd>Esc</kbd> close
          </span>
        </footer>
      </div>
    </div>
  )
}
