import { memo, useEffect, useRef, useState } from 'react'
import type { SessionApi } from '../api'
import { MOD } from '../platform'
import type { BoardItemEntry, TeamMemberEntry, TeamState } from '../protocol'
import type { Entry } from '../state'
import { shortModel } from '../workers'
import { SidebarIcon } from './icons'

/** The board's columns, in order; an item of a status not listed gets a column of its own after them. */
const COLUMNS: [string, string][] = [
  ['pending', 'Pending'],
  ['in_progress', 'In progress'],
  ['blocked', 'Blocked'],
  ['completed', 'Done'],
  ['cancelled', 'Cancelled'],
]
const SETTLED = new Set(['completed', 'cancelled'])
/** The member states with a colour of their own; any other shows as a plain badge. */
const COLOURED = new Set(['working', 'idle', 'stopped', 'failed'])

/** How many teammates are in each state, such as `2 working · 1 idle`. */
export function teamSummary(team: TeamState): string {
  const counts = new Map<string, number>()
  for (const member of team.members) counts.set(member.state, (counts.get(member.state) ?? 0) + 1)
  return [...counts].map(([state, count]) => `${count} ${state}`).join(' · ')
}

export function StateBadge({ state }: { state: string }) {
  return <span className={`team-state ${COLOURED.has(state) ? state : ''}`}>{state}</span>
}

const message = (error: unknown) => (error instanceof Error ? error.message : String(error))

interface Props {
  team: TeamState
  /** The session's title, which the team works for. */
  title: string
  /** Each working teammate's transcript so far, by task ID, for the tiles the page shows. */
  taskItems: Record<string, Entry[]>
  session: SessionApi
  /** Opens a teammate's transcript, by task ID, with a box that messages it. */
  onOpen: (taskId: string) => void
  /** The task IDs of the teammates whose tiles are in view, whose transcripts the page then asks for. */
  onVisible: (taskIds: string[]) => void
  onShowSidebar: (() => void) | null
  onError: (message: string) => void
}

/** The team the shown session leads: a tile per teammate, the shared board, and the messages between them. */
export function TeamPane({ team, title, taskItems, session, onOpen, onVisible, onShowSidebar, onError }: Props) {
  const items = new Map(team.board.map((item) => [item.id, item]))
  const statuses = [...new Set(team.board.map((item) => item.status))].filter(
    (status) => !COLUMNS.some(([known]) => known === status),
  )
  const columns: [string, string][] = [...COLUMNS, ...statuses.map((status): [string, string] => [status, status])]
  const act = (action: () => Promise<unknown>) => void action().catch((error: unknown) => onError(message(error)))
  return (
    <>
      <header className="topbar">
        {onShowSidebar && (
          <button className="icon-button" onClick={onShowSidebar} title={`Show sidebar (${MOD}B)`}>
            <SidebarIcon size={16} />
          </button>
        )}
        <div className="topbar-title">
          <span className="title">Team</span>
          <span className="workspace">{title}</span>
        </div>
      </header>
      <div className="team-pane">
        <section aria-label="Teammates">
          <h2 className="team-heading">Teammates {team.members.length > 0 && <span>{teamSummary(team)}</span>}</h2>
          {team.members.length === 0 ? (
            <p className="team-empty">No teammates yet. The lead starts them, or type /team spawn NAME PROMPT.</p>
          ) : (
            <Tiles team={team} items={items} taskItems={taskItems} onOpen={onOpen} onVisible={onVisible} />
          )}
        </section>
        <section aria-label="Board">
          <h2 className="team-heading">Board</h2>
          <div className="team-board">
            {columns.map(([status, label]) => (
              <div key={status} className="team-column">
                <h3>{label}</h3>
                {team.board
                  .filter((item) => item.status === status)
                  .map((item) => (
                    <BoardCard
                      key={item.id}
                      item={item}
                      members={team.members}
                      onAssign={(owner) => act(() => session.assign(item.id, owner))}
                      onUnclaim={() => act(() => session.unclaim(item.id))}
                      onCancel={() => act(() => session.cancelItem(item.id))}
                    />
                  ))}
              </div>
            ))}
          </div>
        </section>
        <section aria-label="Messages">
          <h2 className="team-heading">Messages</h2>
          {team.messages.length === 0 ? (
            <p className="team-empty">No messages yet.</p>
          ) : (
            <ol className="team-messages">
              {team.messages.map((entry) => (
                <li key={entry.id}>
                  <span className="team-route">
                    {entry.sender} → {entry.recipient}:
                  </span>{' '}
                  {entry.text}
                  {entry.truncated && '…'}
                </li>
              ))}
            </ol>
          )}
        </section>
      </div>
    </>
  )
}

/** The roster's tiles; each one in view reports itself, so only those teammates' transcripts stream to the page. */
function Tiles({
  team,
  items,
  taskItems,
  onOpen,
  onVisible,
}: {
  team: TeamState
  items: Map<string, BoardItemEntry>
  taskItems: Record<string, Entry[]>
  onOpen: (taskId: string) => void
  onVisible: (taskIds: string[]) => void
}) {
  const grid = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState<Set<string>>(new Set())
  const ids = team.members.map((member) => member.task_id).join(' ')
  useEffect(() => {
    const element = grid.current
    if (!element) return
    const shown = new Set<string>()
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const id = (entry.target as HTMLElement).dataset.task ?? ''
        if (entry.isIntersecting) shown.add(id)
        else shown.delete(id)
      }
      setVisible(new Set(shown))
    })
    for (const tile of element.querySelectorAll('[data-task]')) observer.observe(tile)
    return () => observer.disconnect()
  }, [ids])
  useEffect(() => onVisible([...visible].sort()), [visible, onVisible])
  // Leaving the pane shows no tile.
  useEffect(() => () => onVisible([]), [onVisible])
  return (
    <div ref={grid} className="team-tiles">
      {team.members.map((member) => (
        <Tile
          key={member.name}
          member={member}
          item={member.current_task === null ? undefined : items.get(member.current_task)}
          tail={visible.has(member.task_id) ? taskItems[member.task_id] : undefined}
          onOpen={onOpen}
        />
      ))}
    </div>
  )
}

/** The last lines of a transcript, as a tile shows it: the end of the text, or the tool called. */
function tailLines(entries: Entry[]): string[] {
  return entries.slice(-3).map((entry) => {
    if (entry.kind === 'tool') return `⚙ ${entry.name}`
    if (entry.kind === 'notice' || entry.kind === 'turn') return ''
    const text = entry.text.replace(/\s+/g, ' ').trim()
    return `${entry.kind === 'user' ? '› ' : ''}${text.length > 160 ? `…${text.slice(-160)}` : text}`
  })
}

const Tile = memo(function Tile({
  member,
  item,
  tail,
  onOpen,
}: {
  member: TeamMemberEntry
  item: BoardItemEntry | undefined
  tail: Entry[] | undefined
  onOpen: (taskId: string) => void
}) {
  const lines = tail ? tailLines(tail).filter(Boolean) : []
  return (
    <button
      className="team-tile"
      data-task={member.task_id}
      onClick={() => onOpen(member.task_id)}
      title={`Open ${member.name}'s transcript`}
    >
      <span className="team-tile-head">
        <span className="worker-agent">{member.name}</span>
        <span className="worker-model">
          {member.agent} · {member.model ? shortModel(member.model) : 'session model'}
        </span>
        <StateBadge state={member.state} />
      </span>
      {member.current_task !== null && (
        <span className="team-tile-task">
          #{member.current_task} {item?.content ?? ''}
        </span>
      )}
      {lines.length > 0 && (
        <span className="team-tile-tail mono">
          {lines.map((line, index) => (
            <span key={index}>{line}</span>
          ))}
        </span>
      )}
    </button>
  )
})

function BoardCard({
  item,
  members,
  onAssign,
  onUnclaim,
  onCancel,
}: {
  item: BoardItemEntry
  members: TeamMemberEntry[]
  onAssign: (owner: string) => void
  onUnclaim: () => void
  onCancel: () => void
}) {
  return (
    <div className="team-card">
      <span className="team-card-text">
        <span className="team-card-id">#{item.id}</span> {item.content}
      </span>
      <span className="team-card-meta">
        {item.owner && <span className="badge">{item.owner}</span>}
        {item.blocked_by.length > 0 && (
          <span className="team-blockers" title="Waits for these items">
            ⛓ {item.blocked_by.map((id) => `#${id}`).join(', ')}
          </span>
        )}
      </span>
      {!SETTLED.has(item.status) && (
        <span className="team-card-actions">
          {item.status === 'pending' && members.length > 0 && (
            <select
              aria-label={`Assign #${item.id}`}
              value=""
              onChange={(event) => event.target.value && onAssign(event.target.value)}
            >
              <option value="">Assign…</option>
              {members.map((member) => (
                <option key={member.name} value={member.name}>
                  {member.name}
                </option>
              ))}
            </select>
          )}
          {item.status === 'in_progress' && (
            <button className="ghost small" onClick={onUnclaim} title="Hand it back, so any teammate can claim it">
              Unclaim
            </button>
          )}
          <button className="ghost small" onClick={onCancel} title="Cancel it, unblocking what waits only on it">
            Cancel
          </button>
        </span>
      )}
    </div>
  )
}
