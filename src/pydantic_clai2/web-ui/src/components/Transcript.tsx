import { memo, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { SessionContext } from '../api'
import { OpenCardsContext } from '../openCards'
import { JumpToLatest, useStickToBottom } from '../stickToBottom'
import { WorkspaceLinkContext } from '../workspaceLink'
import type { Item } from '../protocol'
import type { Entry } from '../state'
import { cell } from '../code'
import { compact, duration } from '../format'
import type { TurnSummary } from '../state'
import { ChevronIcon } from './icons'
import { Markdown } from './Markdown'
import { Thinking } from './Thinking'
import { ForkResult, ForksCard } from './ForksCard'
import { ToolCard, codeCalls, isWorkspaceTool } from './ToolCard'
import { UsageCard } from './UsageCard'

/** One image of a prompt, fetched from the session once it scrolls into view. */
function Thumbnail({ id }: { id: string }) {
  const session = useContext(SessionContext)
  const tile = useRef<HTMLDivElement>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    const element = tile.current
    if (!element) return
    let current = true
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return
      observer.disconnect()
      session.image(id).then(
        (found) => current && setUrl(found),
        () => current && setFailed(true),
      )
    })
    observer.observe(element)
    return () => {
      current = false
      observer.disconnect()
    }
  }, [id, session])
  return (
    <div className={`attachment ${failed ? 'missing' : ''}`} ref={tile} title={failed ? 'Image not available' : undefined}>
      {url && <img src={url} alt="Image sent with the prompt" />}
    </div>
  )
}

const EntryView = memo(function EntryView({ entry, live }: { entry: Entry; live: boolean }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div className="user-row">
          <div className="user-bubble">
            {entry.images.length > 0 && (
              <div className={`user-images ${entry.text ? 'captioned' : ''}`}>
                {entry.images.map((id, index) => (
                  <Thumbnail key={`${index}-${id}`} id={id} />
                ))}
              </div>
            )}
            {entry.text}
          </div>
        </div>
      )
    case 'assistant':
      return <Markdown className={`assistant ${live ? 'streaming' : ''}`} text={entry.text} />
    case 'thinking':
      return <Thinking text={entry.text} live={live} />
    case 'tool':
      return <ToolCard item={entry} />
    case 'notice':
      if (entry.usage && entry.level === 'info') return <UsageCard notice={entry} usage={entry.usage} />
      if (entry.forks) return <ForksCard notice={entry} forks={entry.forks} />
      if (entry.fork) return <ForkResult notice={entry} fork={entry.fork} />
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

/** Consecutive tool calls fold into one group, as a desktop agent shows its work.
 * Edits stand alone, and so do the orchestrator's calls in its chat. */
type Row = { kind: 'entry'; entry: Entry } | { kind: 'group'; id: string; tools: Item[] }

function rows(entries: Entry[], orchestrating: boolean): Row[] {
  const result: Row[] = []
  for (const entry of entries) {
    const last = result[result.length - 1]
    const groupable = entry.kind === 'tool' && !EDITS.has(entry.name) && !(orchestrating && isWorkspaceTool(entry.name))
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
  const shown = open ?? (running || tools.some((tool) => cards.open.get(tool.id) === true))
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
  const [open, setOpen] = useState<ReadonlyMap<string, boolean>>(() => new Map())
  const cards = useMemo(
    () => ({
      open,
      toggle: (id: string, shown: boolean) => setOpen((current) => new Map(current).set(id, !shown)),
    }),
    [open],
  )
  const { scroller, behind, jump } = useStickToBottom<HTMLDivElement>(
    entries,
    entries.findLast((entry) => entry.kind === 'user'),
  )
  const last = entries[entries.length - 1]
  const orchestrating = useContext(WorkspaceLinkContext) !== null
  return (
    <OpenCardsContext.Provider value={cards}>
    <div className="transcript" ref={scroller}>
      <div className="column">
        {entries.length === 0
          ? empty
          : rows(entries.filter((entry) => !(entry.kind === 'tool' && entry.by_user)), orchestrating).map((row) =>
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
      <JumpToLatest shown={behind} onJump={jump} />
    </div>
    </OpenCardsContext.Provider>
  )
}
