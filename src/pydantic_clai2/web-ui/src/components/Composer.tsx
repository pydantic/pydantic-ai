import { forwardRef, useCallback, useContext, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { SessionContext, api, interrupt } from '../api'
import { type Attachment, base64, imageFiles, readImages } from '../attachments'
import type { CompletionInfo, QueuedItem } from '../protocol'
import { ArrowUpIcon, BoltIcon, ChevronIcon, CheckIcon, CloseIcon, StopIcon } from './icons'
import { STEER_KEY } from '../platform'
import { QueuePanel } from './QueuePanel'

export interface ComposerHandle {
  focus: () => void
}

interface Props {
  running: boolean
  /** Controls after the model picker, such as the run switches. */
  controls?: React.ReactNode
  /** Shown in the bar while a turn runs. */
  meter?: React.ReactNode
  model: string
  disabled: boolean
  placeholder: string
  /** Commands the page handles itself instead of the terminal, such as `/plugins`, which opens its panel. */
  local?: Readonly<Record<string, () => void>>
  /** A bare command the server answered by naming its menu, such as `/set`, which the page shows instead. */
  onMenu?: (command: string) => void
  /** The draft, held by the page so it survives the composer moving between the new-session hero and the bottom bar. */
  draft: string
  onDraft: (draft: string) => void
  /** Images pasted or dropped for the next prompt, held by the page as the draft is; without them, only text pastes. */
  attachments?: readonly Attachment[]
  onAttachments?: (update: (current: readonly Attachment[]) => Attachment[]) => void
  /** Where a message goes, and what stops the running turn: the shown session's by default. */
  chat?: { send: (text: string) => Promise<void>; interrupt: () => Promise<unknown> }
  /** Whether `/` completes commands and the model picker changes the model; the orchestrator chat has neither. */
  commands?: boolean
  /**
   * The shown session's queued input, listed above the box. With it, Alt+Enter steers the running turn, as in the
   * terminal; the orchestrator chat has no queue to show or turn to steer.
   */
  queue?: QueuedItem[]
  label?: string
  onError: (message: string) => void
}

/**
 * The prompt box: Enter sends, or queues while a turn runs; Alt+Enter steers the running turn; Shift+Enter adds a line;
 * `/` completes commands from CLAI's own registry; pasted or dropped images go with the prompt.
 */
export const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  {
    running,
    meter,
    controls,
    model,
    disabled,
    placeholder,
    local,
    onMenu,
    draft: text,
    onDraft: setText,
    attachments = [],
    onAttachments,
    chat,
    commands = true,
    queue,
    label = 'Message CLAI',
    onError,
  },
  ref,
) {
  const [suggestions, setSuggestions] = useState<CompletionInfo[]>([])
  const [selected, setSelected] = useState(0)
  const [models, setModels] = useState<string[] | null>(null)
  const area = useRef<HTMLTextAreaElement>(null)
  const request = useRef(0)
  const session = useContext(SessionContext)

  // Another session shown: the last one's completions and model list are not this one's.
  useEffect(() => {
    request.current++
    setSuggestions([])
    setModels(null)
  }, [session])

  useImperativeHandle(ref, () => ({ focus: () => area.current?.focus() }), [])

  // The composer moves from the new-session hero to the bottom bar when the first message lands; keep typing there.
  useEffect(() => area.current?.focus(), [])

  useEffect(() => {
    const element = area.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 240)}px`
  }, [text])

  const complete = useCallback(async (value: string) => {
    const id = ++request.current
    if (!commands || !value.startsWith('/') || value.includes('\n')) {
      setSuggestions([])
      return
    }
    try {
      const found = await session.complete(value)
      if (id === request.current) {
        setSuggestions(found.slice(0, 50))
        setSelected(0)
      }
    } catch {
      if (id === request.current) setSuggestions([])
    }
  }, [commands, session])

  useEffect(() => {
    if (!models) return
    const close = (event: Event) => {
      if (event instanceof KeyboardEvent && event.key !== 'Escape') return
      if (event instanceof MouseEvent && (event.target as HTMLElement).closest('.model-picker')) return
      event.preventDefault()
      setModels(null)
    }
    // Capture phase, so Esc closes the menu before the page's own Esc handler can interrupt a turn.
    window.addEventListener('keydown', close, true)
    window.addEventListener('mousedown', close)
    return () => {
      window.removeEventListener('keydown', close, true)
      window.removeEventListener('mousedown', close)
    }
  }, [models])

  const accept = (suggestion: CompletionInfo) => {
    const next = text.slice(0, text.length - suggestion.start) + suggestion.text + ' '
    setText(next)
    void complete(next)
    area.current?.focus()
  }

  const report = (error: unknown) => onError(error instanceof Error ? error.message : String(error))
  const steers = queue !== undefined && !chat
  const empty = !text.trim() && attachments.length === 0

  const send = async (steer = false) => {
    const value = text.trim()
    if (disabled) return
    if (empty) {
      // As in the terminal: Alt+Enter with nothing typed steers the oldest queued follow-up, which stays if it cannot.
      const head = steer ? queue?.[0] : undefined
      if (head?.kind === 'prompt') await session.steerQueued(head.id).catch(report)
      return
    }
    const images = attachments
    setText('')
    onAttachments?.(() => [])
    setSuggestions([])
    // Own keys only: `constructor` or `toString` typed alone is a prompt, not a local command.
    const handle = images.length === 0 && local && Object.hasOwn(local, value) ? local[value] : undefined
    if (handle) {
      handle()
      return
    }
    try {
      if (chat) await chat.send(value)
      else {
        // A prompt Alt+Enter cannot steer, such as any while no turn runs, is entered as Enter enters it.
        // Steering takes text alone, so a prompt with images is entered as Enter enters it too.
        const entered = await (steer && images.length === 0
          ? session.steer(value)
          : session.send(value, images.map(base64)))
        if (entered.menu !== null) onMenu?.(entered.menu)
      }
    } catch (error) {
      setText(value)
      onAttachments?.((current) => [...images, ...current])
      report(error)
    }
  }

  const attach = async (files: File[]) => {
    if (!onAttachments) return
    try {
      const read = await readImages(files, attachments)
      onAttachments((current) => [...current, ...read])
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
    area.current?.focus()
  }

  const onPaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    if (!onAttachments) return
    const files = imageFiles(event.clipboardData.files)
    if (files.length === 0) return
    // Text copied with a picture of itself, as from a spreadsheet, pastes as text; a copied file's name does not.
    const pasted = event.clipboardData.getData('text/plain').trim()
    if (pasted && !files.some((file) => file.name && pasted.endsWith(file.name))) return
    event.preventDefault()
    void attach(files)
  }

  const onDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    if (onAttachments && event.dataTransfer.types.includes('Files')) event.preventDefault()
  }

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    if (!onAttachments || event.dataTransfer.files.length === 0) return
    // Never let the browser leave the page to open a dropped file.
    event.preventDefault()
    const files = imageFiles(event.dataTransfer.files)
    if (files.length === 0) onError('Only images can be attached.')
    else void attach(files)
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (suggestions.length > 0) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        const step = event.key === 'ArrowDown' ? 1 : -1
        setSelected((current) => (current + step + suggestions.length) % suggestions.length)
        return
      }
      if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey && text.trim().split(/\s+/).length === 1)) {
        const suggestion = suggestions[selected]
        if (suggestion && `/${suggestion.text}` !== text.trim()) {
          event.preventDefault()
          accept(suggestion)
          return
        }
      }
      if (event.key === 'Escape') {
        event.preventDefault()
        setSuggestions([])
        return
      }
    }
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      void send(steers && event.altKey)
    }
  }

  const openModels = async () => {
    if (models) {
      setModels(null)
      return
    }
    try {
      setModels(await api.models())
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  const pickModel = async (name: string) => {
    setModels(null)
    try {
      await session.selectModel(name)
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="composer-wrap">
      {steers && <QueuePanel queue={queue} running={running} onError={onError} />}
      {suggestions.length > 0 && (
        <div className="suggestions" role="listbox" id="suggestions">
          {suggestions.map((suggestion, index) => (
            <button
              key={`${suggestion.text}-${index}`}
              id={`suggestion-${index}`}
              role="option"
              aria-selected={index === selected}
              className={index === selected ? 'active' : ''}
              onMouseEnter={() => setSelected(index)}
              onMouseDown={(event) => {
                event.preventDefault()
                accept(suggestion)
              }}
            >
              <span className="suggestion-name">{suggestion.display}</span>
              {suggestion.meta && <span className="suggestion-meta">{suggestion.meta}</span>}
            </button>
          ))}
        </div>
      )}
      <div className={`composer ${disabled ? 'disabled' : ''}`} onDragOver={onDragOver} onDrop={onDrop}>
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((attachment) => (
              <div className="attachment" key={attachment.id}>
                <img src={attachment.url} alt={attachment.name} title={attachment.name} />
                <button
                  className="attachment-remove"
                  onClick={() => onAttachments?.((current) => current.filter((other) => other.id !== attachment.id))}
                  title="Remove"
                  aria-label={`Remove ${attachment.name}`}
                >
                  <CloseIcon size={10} />
                </button>
              </div>
            ))}
          </div>
        )}
        <textarea
          ref={area}
          rows={1}
          value={text}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(event) => {
            setText(event.target.value)
            void complete(event.target.value)
          }}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          onBlur={() => window.setTimeout(() => setSuggestions([]), 120)}
          aria-label={label}
          role="combobox"
          aria-expanded={suggestions.length > 0}
          aria-controls="suggestions"
          aria-autocomplete="list"
          aria-activedescendant={suggestions.length > 0 ? `suggestion-${selected}` : undefined}
        />
        <div className="composer-bar">
          {commands ? (
            <div className="model-picker">
              <button
                className="pill"
                onClick={() => void openModels()}
                title="Choose a saved model"
                aria-haspopup="menu"
                aria-expanded={models !== null}
              >
                <span className="model-name">{model}</span>
                <ChevronIcon size={12} className="chevron down" />
              </button>
              {models && (
                <div className="menu model-menu" role="menu">
                  {models.length === 0 && <div className="menu-empty">No saved models. Add one with /model add in the terminal.</div>}
                  {models.map((name) => (
                    <button key={name} onClick={() => void pickModel(name)}>
                      <span>{name}</span>
                      {name === model && <CheckIcon size={14} />}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <span className="pill static" title="The orchestrator runs on the saved default model, as a new session starts with">
              <span className="model-name">{model}</span>
            </span>
          )}
          {controls}
          {running ? meter : <span className="hint">{commands ? '/ for commands' : 'Enter to send'}</span>}
          {steers && running && text.trim() && attachments.length === 0 && (
            <button
              className="send steer"
              onClick={() => void send(true)}
              disabled={disabled}
              title={`Steer now: send it to the running turn (${STEER_KEY})`}
              aria-label="Steer now"
            >
              <BoltIcon size={15} />
            </button>
          )}
          {running && empty ? (
            <button
              className="send stop"
              onClick={() => void interrupt(onError, chat?.interrupt ?? session.interrupt)}
              title="Interrupt (Esc)"
              aria-label="Interrupt"
            >
              <StopIcon size={14} />
            </button>
          ) : (
            <button
              className="send"
              onClick={() => void send()}
              disabled={empty || disabled}
              title={running ? 'Queue for after this turn (Enter)' : 'Send (Enter)'}
            >
              <ArrowUpIcon size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
})
