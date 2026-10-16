import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { api, interrupt } from '../api'
import type { CompletionInfo } from '../protocol'
import { ArrowUpIcon, ChevronIcon, CheckIcon, StopIcon } from './icons'

export interface ComposerHandle {
  focus: () => void
}

/** Where a composer's messages go, what completes its `/commands`, and what its model picker changes. */
export interface Chat {
  send: (text: string) => Promise<void>
  interrupt: () => Promise<unknown>
  complete: (text: string) => Promise<CompletionInfo[]>
  selectModel: (name: string) => Promise<void>
}

const SESSION: Chat = { send: api.send, interrupt: api.interrupt, complete: api.complete, selectModel: api.selectModel }

/** A first model picker row that follows another chat's model, checked while it does. */
export interface Follow {
  label: string
  active: boolean
  select: () => Promise<void>
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
  /** The draft, held by the page so it survives the composer moving between the new-session hero and the bottom bar. */
  draft: string
  onDraft: (draft: string) => void
  /** The session's by default. */
  chat?: Chat
  follow?: Follow
  label?: string
  onError: (message: string) => void
}

/** The prompt box: Enter sends, Shift+Enter adds a line, `/` completes commands from CLAI's own registry. */
export const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  {
    running,
    meter,
    controls,
    model,
    disabled,
    placeholder,
    local,
    draft: text,
    onDraft: setText,
    chat = SESSION,
    follow,
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
    if (!value.startsWith('/') || value.includes('\n')) {
      setSuggestions([])
      return
    }
    try {
      const found = await chat.complete(value)
      if (id === request.current) {
        setSuggestions(found.slice(0, 50))
        setSelected(0)
      }
    } catch {
      if (id === request.current) setSuggestions([])
    }
  }, [chat])

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

  const send = async () => {
    const value = text.trim()
    if (!value || disabled) return
    setText('')
    setSuggestions([])
    // Own keys only: `constructor` or `toString` typed alone is a prompt, not a local command.
    const handle = local && Object.hasOwn(local, value) ? local[value] : undefined
    if (handle) {
      handle()
      return
    }
    try {
      await chat.send(value)
    } catch (error) {
      setText(value)
      onError(error instanceof Error ? error.message : String(error))
    }
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
      void send()
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

  const pickModel = async (select: () => Promise<void>) => {
    setModels(null)
    try {
      await select()
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="composer-wrap">
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
      <div className={`composer ${disabled ? 'disabled' : ''}`}>
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
          onBlur={() => window.setTimeout(() => setSuggestions([]), 120)}
          aria-label={label}
          role="combobox"
          aria-expanded={suggestions.length > 0}
          aria-controls="suggestions"
          aria-autocomplete="list"
          aria-activedescendant={suggestions.length > 0 ? `suggestion-${selected}` : undefined}
        />
        <div className="composer-bar">
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
                {follow && (
                  <button onClick={() => void pickModel(follow.select)}>
                    <span>{follow.label}</span>
                    {follow.active && <CheckIcon size={14} />}
                  </button>
                )}
                {models.length === 0 && <div className="menu-empty">No saved models. Add one with /model add in the terminal.</div>}
                {models.map((name) => (
                  <button key={name} onClick={() => void pickModel(() => chat.selectModel(name))}>
                    <span>{name}</span>
                    {name === model && !follow?.active && <CheckIcon size={14} />}
                  </button>
                ))}
              </div>
            )}
          </div>
          {controls}
          {running ? meter : <span className="hint">/ for commands</span>}
          {running && !text.trim() ? (
            <button
              className="send stop"
              onClick={() => void interrupt(onError, chat.interrupt)}
              title="Interrupt (Esc)"
              aria-label="Interrupt"
            >
              <StopIcon size={14} />
            </button>
          ) : (
            <button className="send" onClick={() => void send()} disabled={!text.trim() || disabled} title="Send (Enter)">
              <ArrowUpIcon size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
})
