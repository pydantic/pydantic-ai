import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { TaskEntry } from '../protocol'
import type { Entry } from '../state'
import { WorkersContext, elapsed, shortModel, state, useNow } from '../workers'
import { ArrowUpIcon } from './icons'
import { Transcript } from './Transcript'
import { WorkerMark } from './WorkersTree'

const message = (error: unknown) => (error instanceof Error ? error.message : String(error))

interface Props {
  task: TaskEntry
  tasks: TaskEntry[]
  /** What the running task has done so far; a finished one's saved history is fetched instead. */
  live: Entry[] | undefined
  onOpen: (id: string) => void
  onBack: () => void
  onError: (message: string) => void
}

/** One worker's transcript, live while it runs, with its controls and a box that steers it, or prompts it once it has finished. */
export function WorkerView({ task, tasks, live, onOpen, onBack, onError }: Props) {
  const finished = task.status === 'finished'
  const [saved, setSaved] = useState<Entry[] | null>(null)
  const now = useNow(!finished)

  // A finished worker shows what it saved, read again when the one shown finishes.
  useEffect(() => {
    setSaved(null)
    if (!finished) return
    let current = true
    api.task(task.id).then(
      (found) => current && setSaved(found.items),
      (error: unknown) => current && onError(message(error)),
    )
    return () => {
      current = false
    }
  }, [task.id, finished, onError])

  const act = async (action: 'stop' | 'background') => {
    try {
      await api.taskAction(task.id, action)
    } catch (error) {
      onError(message(error))
    }
  }

  const links = useMemo(() => ({ tasks, parent: task.id, open: onOpen }), [tasks, task.id, onOpen])
  const entries = finished ? (saved ?? []) : (live ?? [])
  return (
    <>
      <div className="worker-banner">
        <button className="ghost small" onClick={onBack} title="Back to the session">
          ← Session
        </button>
        <span className="worker-banner-state">
          <WorkerMark task={task} />
          <span className="worker-agent">{task.agent}</span>
          {task.model && <span className="worker-model">{shortModel(task.model)}</span>}
          <span className="worker-time">{elapsed(task, now)}</span>
          <span className="worker-activity">
            {task.background && !finished ? 'background · ' : ''}
            {state(task)}
          </span>
        </span>
        {task.backgroundable && (
          <button
            className="ghost small"
            onClick={() => void act('background')}
            title="Let the session go on without waiting"
          >
            Background
          </button>
        )}
        {!finished && (
          <button
            className="ghost small stop-worker"
            onClick={() => void act('stop')}
            title="Stop it and the workers it started"
          >
            Stop
          </button>
        )}
      </div>
      <WorkersContext.Provider value={links}>
        <Transcript
          entries={entries}
          running={!finished}
          empty={<div className="empty">{finished && saved === null ? 'Loading…' : 'Nothing yet.'}</div>}
        />
      </WorkersContext.Provider>
      <footer className="bottom">
        {finished && !task.resumable ? (
          <div className="worker-note">This worker has finished. It is one-shot, so it takes no follow-ups.</div>
        ) : (
          <TaskBox key={task.id} task={task} onError={onError} />
        )}
      </footer>
    </>
  )
}

/**
 * Enter sends the worker a message, Shift+Enter adds a line. While it runs, it reads the message at its next
 * model request; once it has finished, the message prompts it on its own, apart from the session.
 */
function TaskBox({ task, onError }: { task: TaskEntry; onError: (message: string) => void }) {
  const finished = task.status === 'finished'
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const area = useRef<HTMLTextAreaElement>(null)

  useEffect(() => area.current?.focus(), [])
  useEffect(() => {
    const element = area.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 240)}px`
  }, [text])

  const send = async () => {
    const value = text.trim()
    if (!value || sending) return
    setSending(true)
    try {
      if (finished) {
        await api.promptTask(task.id, value)
        setText('')
      } else if ((await api.steer(task.id, value)).steered) {
        setText('')
      } else {
        // A one-shot worker's composer gives way to its note once the row shows it finished.
        onError(
          task.resumable
            ? 'This worker has just finished. Send your message again to prompt it on its own.'
            : 'This worker has just finished. It is one-shot, so it takes no follow-ups.',
        )
      }
    } catch (error) {
      onError(message(error))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={area}
          rows={1}
          value={text}
          placeholder={finished ? `Message ${task.agent}…` : `Steer ${task.agent}…`}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault()
              void send()
            }
          }}
          aria-label={`Message ${task.agent}`}
        />
        <div className="composer-bar">
          <span className="hint">
            {finished
              ? 'Continues this worker on its own; the session does not see it.'
              : 'It reads this at its next model request.'}
          </span>
          <button
            className="send"
            onClick={() => void send()}
            disabled={!text.trim() || sending}
            title="Send (Enter)"
          >
            <ArrowUpIcon size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
