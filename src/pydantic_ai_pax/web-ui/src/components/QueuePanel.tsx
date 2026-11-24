import { useContext, useEffect, useRef, useState } from 'react'
import { ApiError, SessionContext } from '../api'
import { STEER_KEY } from '../platform'
import type { QueuedItem } from '../protocol'
import { BoltIcon, CloseIcon, PencilIcon } from './icons'

const LABELS = { prompt: 'Follow-up', command: 'Command', shell: 'Shell' } as const

interface Props {
  queue: QueuedItem[]
  /** Whether a turn runs to steer; a queued prompt can steer only then. */
  running: boolean
  onError: (message: string) => void
}

/**
 * What waits for the running turn, as the terminal lists it above its prompt: each input can be edited, sent to the
 * turn now, or removed until it runs. The terminal and every other browser see the same queue.
 */
export function QueuePanel({ queue, running, onError }: Props) {
  const session = useContext(SessionContext)
  // The input being edited, kept by ID: once the session takes it, saving queues the edit as a new follow-up.
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null)
  const area = useRef<HTMLTextAreaElement>(null)

  // Focus once per edit, not per keystroke.
  const editingId = editing?.id
  useEffect(() => {
    if (editingId !== undefined) area.current?.focus()
  }, [editingId])

  const report = (error: unknown) => onError(error instanceof Error ? error.message : String(error))

  const save = async () => {
    if (!editing) return
    const { id, text } = editing
    setEditing(null)
    try {
      if (!text.trim()) await session.removeQueued(id)
      else await session.editQueued(id, text)
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 404)) {
        setEditing({ id, text })
        report(error)
      } else if (text.trim()) {
        // It ran while you edited it, as in the terminal: the edit becomes a follow-up of its own.
        await session.send(text).catch(report)
      }
    }
  }

  const steer = async (item: QueuedItem) => {
    try {
      const { steered } = await session.steerQueued(item.id)
      if (!steered) onError('The turn could not take it just now, so it stays queued.')
    } catch (error) {
      report(error)
    }
  }

  const remove = (item: QueuedItem) => void session.removeQueued(item.id).catch(report)

  const gone = editing !== null && !queue.some((item) => item.id === editing.id)
  if (queue.length === 0 && !gone) return null
  const editor = (label: string) =>
    editing && (
      <div className="queued editing">
        <span className="queued-label">{label}</span>
        <textarea
          ref={area}
          rows={1}
          value={editing.text}
          aria-label="Edit queued message"
          onChange={(event) => setEditing({ ...editing, text: event.target.value })}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.preventDefault()
              setEditing(null)
            } else if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault()
              void save()
            }
          }}
        />
        <span className="queued-keys">Enter saves · Esc cancels</span>
      </div>
    )

  return (
    <div className="queue" role="list" aria-label="Queued messages">
      <div className="queue-head">
        <span>Queued · runs after this turn</span>
        {running && <span className="queue-hint">Enter queues · {STEER_KEY} steers now</span>}
      </div>
      {queue.map((item) =>
        editing?.id === item.id ? (
          <div key={item.id} role="listitem">
            {editor(LABELS[item.kind])}
          </div>
        ) : (
          <div key={item.id} className="queued" role="listitem">
            <span className="queued-label">{LABELS[item.kind]}</span>
            <span className="queued-text" title={item.text}>
              {item.text.split(/\s+/).join(' ')}
            </span>
            <span className="queued-actions">
              {item.kind !== 'shell' && (
                <button
                  className="icon-button"
                  title="Edit"
                  aria-label="Edit queued message"
                  onClick={() => setEditing({ id: item.id, text: item.text })}
                >
                  <PencilIcon size={13} />
                </button>
              )}
              {running && item.kind === 'prompt' && (
                <button
                  className="icon-button"
                  title="Steer now: send it to the running turn"
                  aria-label="Steer now"
                  onClick={() => void steer(item)}
                >
                  <BoltIcon size={13} />
                </button>
              )}
              <button className="icon-button" title="Remove" aria-label="Remove queued message" onClick={() => remove(item)}>
                <CloseIcon size={13} />
              </button>
            </span>
          </div>
        ),
      )}
      {gone && <div role="listitem">{editor('Sent · saving queues it again')}</div>}
    </div>
  )
}
