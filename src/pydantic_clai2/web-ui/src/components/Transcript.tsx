import { memo, useEffect, useLayoutEffect, useRef } from 'react'
import type { Entry } from '../state'
import { Markdown } from './Markdown'
import { Thinking } from './Thinking'
import { ToolCard } from './ToolCard'

const EntryView = memo(function EntryView({ entry, live }: { entry: Entry; live: boolean }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div className="user-row">
          <div className="user-bubble">{entry.text}</div>
        </div>
      )
    case 'assistant':
      return <Markdown className={`assistant ${live ? 'streaming' : ''}`} text={entry.text} />
    case 'thinking':
      return <Thinking text={entry.text} live={live} />
    case 'tool':
      return <ToolCard item={entry} />
    case 'notice':
      return (
        <div className={`notice ${entry.level}`}>
          {entry.command && <span className="notice-command">{entry.command}</span>}
          <pre>{entry.text}</pre>
        </div>
      )
    case 'turn':
      if (entry.outcome === 'completed') return null
      return (
        <div className={`turn-note ${entry.outcome}`}>
          {entry.outcome === 'cancelled' ? 'Interrupted' : `Failed: ${entry.error ?? 'unknown error'}`}
        </div>
      )
  }
})

/** The conversation; it follows new output while the reader is at the bottom, and stays put once they scroll up. */
export function Transcript({ entries, running, empty }: { entries: Entry[]; running: boolean; empty: React.ReactNode }) {
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)
  useEffect(() => {
    const element = scroller.current
    if (!element) return
    const onScroll = () => {
      pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80
    }
    element.addEventListener('scroll', onScroll, { passive: true })
    return () => element.removeEventListener('scroll', onScroll)
  }, [])
  useLayoutEffect(() => {
    const element = scroller.current
    if (element && pinned.current) element.scrollTop = element.scrollHeight
  }, [entries])
  const last = entries[entries.length - 1]
  return (
    <div className="transcript" ref={scroller}>
      <div className="column">
        {entries.length === 0 ? empty : entries.map((entry) => <EntryView key={entry.id} entry={entry} live={running && entry === last} />)}
        {running && last?.kind !== 'assistant' && last?.kind !== 'thinking' && (
          <div className="working">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}
      </div>
    </div>
  )
}
