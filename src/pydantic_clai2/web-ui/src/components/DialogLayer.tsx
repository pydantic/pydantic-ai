import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { MOD } from '../platform'
import type { Dialog } from '../protocol'
import { CheckIcon, SearchIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** What a plugin's settings menu asks, one dialog at a time, above everything else on the page. */
export function DialogLayer({ dialogs, onError }: { dialogs: Dialog[]; onError: (message: string) => void }) {
  const dialog = dialogs[0]
  if (!dialog) return null
  return <DialogBox key={dialog.id} dialog={dialog} onError={onError} />
}

function DialogBox({ dialog, onError }: { dialog: Dialog; onError: (message: string) => void }) {
  const [sending, setSending] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  /** Answer, or back out with `null`; a rejected answer keeps the dialog open with the reason. */
  const answer = async (value: string | null) => {
    if (sending) return
    setSending(true)
    try {
      await api.answerDialog(dialog.id, value)
    } catch (error) {
      if (value === null) onError(message(error))
      else setProblem(message(error))
      setSending(false)
    }
  }

  // Registered once, so a re-render during the key press cannot drop it; it calls the latest `answer`.
  const latest = useRef(answer)
  latest.current = answer
  useEffect(() => {
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      // Esc is the dialog's alone: not the plugins panel's, nor the page's, which would interrupt a turn.
      event.preventDefault()
      event.stopImmediatePropagation()
      void latest.current(null)
    }
    window.addEventListener('keydown', cancel, true)
    return () => window.removeEventListener('keydown', cancel, true)
  }, [])

  return (
    <div className="overlay dialog-overlay">
      <div className={`dialog-box ${dialog.kind} ${dialog.multiline ? 'multiline' : ''}`} role="dialog" aria-modal="true" aria-label={dialog.title}>
        <h2 className="dialog-title">{dialog.title}</h2>
        {dialog.description && dialog.kind !== 'wait' && (
          <Paragraphs className="dialog-description" text={dialog.description} />
        )}
        {dialog.kind === 'choose' && (
          <Choose dialog={dialog} disabled={sending} onAnswer={(value) => void answer(value)} />
        )}
        {dialog.kind === 'text' && (
          <TextAnswer
            dialog={dialog}
            disabled={sending}
            problem={problem}
            onEdit={() => setProblem(null)}
            onAnswer={(value) => void answer(value)}
          />
        )}
        {dialog.kind === 'confirm' && (
          <div className="dialog-actions">
            <button className="ghost" autoFocus disabled={sending} onClick={() => void answer('no')}>
              {dialog.no}
            </button>
            <button className="primary" disabled={sending} onClick={() => void answer('yes')}>
              {dialog.yes}
            </button>
          </div>
        )}
        {dialog.kind === 'view' && (
          <>
            <pre className="dialog-view" tabIndex={0}>
              {dialog.text}
            </pre>
            <div className="dialog-actions">
              <button className="primary" autoFocus disabled={sending} onClick={() => void answer(null)}>
                Close
              </button>
            </div>
          </>
        )}
        {dialog.kind === 'wait' && (
          <>
            <div className="dialog-waiting">
              <span className="spinner" aria-hidden="true" />
              <Paragraphs className="dialog-description" text={dialog.description} />
            </div>
            {dialog.shown && <Paragraphs className="dialog-shown" text={dialog.shown} />}
            <div className="dialog-actions">
              <button className="ghost" autoFocus disabled={sending} onClick={() => void answer(null)}>
                Cancel
              </button>
            </div>
          </>
        )}
        {dialog.kind !== 'wait' && dialog.kind !== 'confirm' && dialog.kind !== 'view' && (
          <p className="dialog-hint">Esc goes back without changing anything.</p>
        )}
      </div>
    </div>
  )
}

function Choose({
  dialog,
  disabled,
  onAnswer,
}: {
  dialog: Dialog
  disabled: boolean
  onAnswer: (value: string) => void
}) {
  const [query, setQuery] = useState('')
  const searchable = dialog.options.length > 8
  const needle = query.trim().toLowerCase()
  const shown = useMemo(
    () => dialog.options.filter((option) => !needle || option.label.toLowerCase().includes(needle)),
    [dialog.options, needle],
  )
  const [cursor, setCursor] = useState(() =>
    Math.max(
      0,
      shown.findIndex((option) => option.value === dialog.current),
    ),
  )
  const list = useRef<HTMLDivElement>(null)

  useEffect(() => {
    list.current?.querySelector<HTMLElement>('.dialog-option.active')?.scrollIntoView({ block: 'nearest' })
  }, [cursor])

  // React's `autoFocus` focuses form controls only, so the list takes the keys once it opens.
  useEffect(() => {
    if (!searchable) list.current?.focus()
  }, [searchable])

  const keys = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const step = event.key === 'ArrowDown' ? 1 : -1
      setCursor((current) => (shown.length ? (current + step + shown.length) % shown.length : 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const option = shown[cursor]
      if (option) onAnswer(option.value)
    }
  }

  return (
    <div className="dialog-choose" onKeyDown={keys}>
      {searchable && (
        <label className="search">
          <SearchIcon size={14} />
          <input
            autoFocus
            value={query}
            placeholder="Filter"
            onChange={(event) => {
              setQuery(event.target.value)
              setCursor(0)
            }}
          />
        </label>
      )}
      <div className="dialog-options" role="listbox" ref={list} tabIndex={searchable ? -1 : 0}>
        {shown.length === 0 && <div className="empty-list">Nothing matches</div>}
        {shown.map((option, index) => (
          <button
            key={option.value}
            role="option"
            aria-selected={index === cursor}
            className={`dialog-option ${index === cursor ? 'active' : ''}`}
            disabled={disabled}
            onMouseEnter={() => setCursor(index)}
            onClick={() => onAnswer(option.value)}
          >
            <span className="dialog-option-label">
              {option.label}
              {option.value === dialog.current && (
                <span className="dialog-current">
                  <CheckIcon size={12} /> current
                </span>
              )}
            </span>
            {option.description && <span className="dialog-option-description">{option.description}</span>}
          </button>
        ))}
      </div>
    </div>
  )
}

function TextAnswer({
  dialog,
  disabled,
  problem,
  onEdit,
  onAnswer,
}: {
  dialog: Dialog
  disabled: boolean
  problem: string | null
  onEdit: () => void
  onAnswer: (value: string) => void
}) {
  const [text, setText] = useState(dialog.initial)
  // Text asked for over several lines, such as instructions, or that starts on several, such as a server's JSON, is
  // edited over several lines; Cmd/Ctrl+Enter saves.
  const multiline = dialog.multiline || dialog.initial.includes('\n')
  const field = {
    autoFocus: true,
    className: `setting-input ${multiline ? 'code' : ''} ${dialog.multiline ? 'dialog-textarea' : ''} ${problem ? 'invalid' : ''}`,
    spellCheck: false,
    value: text,
    placeholder: dialog.placeholder,
    disabled,
    onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setText(event.target.value)
      onEdit()
    },
  }
  return (
    <form
      className="dialog-text"
      onSubmit={(event) => {
        event.preventDefault()
        onAnswer(text)
      }}
    >
      <label className="dialog-label">
        <span>{dialog.label}</span>
        {multiline ? (
          <textarea
            {...field}
            rows={dialog.multiline ? 12 : Math.min(16, Math.max(4, text.split('\n').length))}
            onKeyDown={(event) => {
              if (event.key !== 'Enter' || !(event.metaKey || event.ctrlKey)) return
              event.preventDefault()
              onAnswer(text)
            }}
          />
        ) : (
          <input
            {...field}
            type={dialog.secret ? 'password' : 'text'}
            autoComplete={dialog.secret ? 'off' : undefined}
          />
        )}
      </label>
      {problem && (
        <p className="setting-error" role="alert">
          {problem}
        </p>
      )}
      <div className="dialog-actions">
        {multiline && <span className="dialog-keys">{MOD}Enter saves</span>}
        <button className="primary" type="submit" disabled={disabled}>
          Save
        </button>
      </div>
    </form>
  )
}

const LINK = /(https?:\/\/[^\s<>"')\]]+)/g

/** Text as paragraphs, with web addresses as links that open in a new tab. */
function Paragraphs({ className, text }: { className: string; text: string }) {
  return (
    <div className={className}>
      {text.split(/\n{2,}/).map((paragraph, index) => (
        <p key={index}>
          {paragraph.split(LINK).map((part, position) =>
            position % 2 === 1 ? (
              <a key={position} href={part} target="_blank" rel="noreferrer noopener">
                {part}
              </a>
            ) : (
              part
            ),
          )}
        </p>
      ))}
    </div>
  )
}
