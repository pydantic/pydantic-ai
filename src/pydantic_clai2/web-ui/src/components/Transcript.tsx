import { memo, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { OpenCardsContext } from '../openCards'
import type { Item } from '../protocol'
import type { Entry } from '../state'
import { cell } from '../code'
import { compact, duration } from '../format'
import type { TurnSummary } from '../state'
import { ChevronIcon } from './icons'
import { Markdown } from './Markdown'
import { Thinking } from './Thinking'
import { ToolCard, codeCalls } from './ToolCard'

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
      if (entry.outcome === 'completed') return <TurnUsage turn={entry} />
      return (
        <div className={`turn-note ${entry.outcome}`}>
          {entry.outcome === 'cancelled' ? 'Interrupted' : `Failed: ${entry.error ?? 'unknown error'}`}
        </div>
      )
  }
})

/** What a finished turn used, in one quiet line. */
function TurnUsage({ turn }: { turn: TurnSummary }) {
  const { usage, seconds } = turn
  if (!usage && seconds === null) return null
  const parts = [
    usage && `↑ ${compact(usage.input_tokens)}`,
    usage && `↓ ${compact(usage.output_tokens)}`,
    usage && `${usage.requests} ${usage.requests === 1 ? 'request' : 'requests'}`,
    seconds !== null && duration(seconds),
  ].filter(Boolean)
  return <div className="turn-usage">{parts.join(' · ')}</div>
}

const EDITS = new Set(['edit_file', 'write_file'])
const COMMANDS = new Set(['shell', 'run_code'])
const EXPLORING = new Set(['read_file', 'list_files', 'grep'])

/** Consecutive tool calls other than edits fold into one group, as a desktop agent shows its work. */
type Row = { kind: 'entry'; entry: Entry } | { kind: 'group'; id: string; tools: Item[] }

function rows(entries: Entry[]): Row[] {
  const result: Row[] = []
  for (const entry of entries) {
    const last = result[result.length - 1]
    const groupable = entry.kind === 'tool' && !EDITS.has(entry.name)
    if (groupable && last?.kind === 'group') last.tools.push(entry)
    else if (groupable) result.push({ kind: 'group', id: `group-${entry.id}`, tools: [entry] })
    else result.push({ kind: 'entry', entry })
  }
  return result.flatMap((row) =>
    row.kind === 'group' && row.tools.length === 1 ? [{ kind: 'entry', entry: row.tools[0] as Item }] : [row],
  )
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`
}

/** The names of what each call did; a code-mode script counts as the harness calls inside it. */
function actions(tool: Item): string[] {
  if (tool.name !== 'run_code') return [tool.name]
  if (tool.calls.length) return tool.calls.map((call) => call.name)
  const calls = codeCalls(cell(tool.args).code).map((call) => call.name)
  return calls.length ? calls : ['run_code']
}

export function groupSummary(tools: Item[]): string {
  const names = tools.flatMap(actions)
  const explored = names.filter((name) => EXPLORING.has(name)).length
  const commands = names.filter((name) => COMMANDS.has(name)).length
  const edited = names.filter((name) => EDITS.has(name)).length
  const other = names.length - explored - commands - edited
  return [
    explored && `Explored ${plural(explored, 'file', 'files')}`,
    commands && `Ran ${plural(commands, 'command', 'commands')}`,
    edited && `Edited ${plural(edited, 'file', 'files')}`,
    other && `Called ${plural(other, 'tool', 'tools')}`,
  ]
    .filter(Boolean)
    .join(' · ')
}

const ToolGroup = memo(function ToolGroup({ tools }: { tools: Item[] }) {
  const cards = useContext(OpenCardsContext)
  const running = tools.some((tool) => tool.state === 'running')
  const failed = tools.filter((tool) => tool.state === 'error' || (tool.exit_code ?? 0) !== 0).length
  const interrupted = tools.filter((tool) => tool.state === 'interrupted').length
  const [open, setOpen] = useState<boolean | null>(null)
  // Open while a call runs, or while the user has one of its cards open, until they close the group.
  const shown = open ?? (running || tools.some((tool) => cards.open.has(tool.id)))
  return (
    <div className={`tool-group ${shown ? 'open' : ''}`}>
      <button className="row-toggle" onClick={() => setOpen(!shown)} aria-expanded={shown}>
        <span className={running ? 'shimmer' : ''}>{groupSummary(tools)}</span>
        {failed > 0 && <span className="badge error">{plural(failed, 'failed', 'failed')}</span>}
        {interrupted > 0 && <span className="badge">{plural(interrupted, 'interrupted', 'interrupted')}</span>}
        <ChevronIcon size={12} className="chevron" />
      </button>
      {shown && (
        <div className="tool-group-body">
          {tools.map((tool) => (
            <ToolCard key={tool.id} item={tool} />
          ))}
        </div>
      )}
    </div>
  )
})

/** The conversation; it follows new output while the reader is at the bottom, and stays put once they scroll up. */
export function Transcript({ entries, running, empty }: { entries: Entry[]; running: boolean; empty: React.ReactNode }) {
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set())
  const cards = useMemo(
    () => ({
      open,
      toggle: (id: string) =>
        setOpen((current) => {
          const next = new Set(current)
          if (!next.delete(id)) next.add(id)
          return next
        }),
    }),
    [open],
  )
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
    <OpenCardsContext.Provider value={cards}>
    <div className="transcript" ref={scroller}>
      <div className="column">
        {entries.length === 0
          ? empty
          : rows(entries).map((row) =>
              row.kind === 'group' ? (
                <ToolGroup key={row.id} tools={row.tools} />
              ) : (
                <EntryView key={row.entry.id} entry={row.entry} live={running && row.entry === last} />
              ),
            )}
        {running && last?.kind !== 'assistant' && last?.kind !== 'thinking' && (
          <div className="working">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}
      </div>
    </div>
    </OpenCardsContext.Provider>
  )
}
