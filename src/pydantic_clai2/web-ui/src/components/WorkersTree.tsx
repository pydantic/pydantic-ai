import { useContext } from 'react'
import type { Item, TaskEntry } from '../protocol'
import { WorkersContext, elapsed, shortModel, state, startedBy, tree, useNow } from '../workers'

/** A task's spinner while it runs, else a dot coloured by how it ended. */
export function WorkerMark({ task }: { task: TaskEntry }) {
  if (task.status === 'running') return <span className="spinner worker-spinner" aria-label="Running" />
  const ended = task.outcome === 'ok' ? 'completed' : task.outcome === 'cancelled' ? 'cancelled' : 'failed'
  return <span className={`state ${ended}`} aria-label={task.outcome ?? 'finished'} />
}

/** Links from a tool card to the workers its call started. */
export function WorkerChips({ item }: { item: Item }) {
  const links = useContext(WorkersContext)
  const started = startedBy(item, links)
  if (!started.length) return null
  return (
    <div className="worker-chips">
      {started.map((task) => (
        <button key={task.id} className="worker-chip" onClick={() => links.open(task.id)} title={task.prompt}>
          <WorkerMark task={task} />
          <span>{task.agent}</span>
          <span className="worker-activity">{state(task)}</span>
        </button>
      ))}
    </div>
  )
}

/** The session's delegated tasks, each under the one that started it; a click opens its transcript. */
export function WorkersTree({
  tasks,
  selected,
  onOpen,
}: {
  tasks: TaskEntry[]
  selected: string | null
  onOpen: (id: string) => void
}) {
  const now = useNow(tasks.some((task) => task.status === 'running'))
  const running = tasks.filter((task) => task.status === 'running').length
  return (
    <div className="workers" role="tree" aria-label="Workers">
      <div className="workers-title">
        <span>Workers</span>
        <span className="workers-count">{running ? `${running} running` : tasks.length}</span>
      </div>
      {tree(tasks).map(({ task, depth }) => (
        <button
          key={task.id}
          role="treeitem"
          aria-level={depth + 1}
          aria-selected={selected === task.id}
          className={`worker ${selected === task.id ? 'selected' : ''} ${task.status}`}
          style={{ paddingLeft: 14 + depth * 14 }}
          onClick={() => onOpen(task.id)}
          title={task.prompt}
        >
          <span className="worker-line">
            <WorkerMark task={task} />
            <span className="worker-agent">{task.agent}</span>
            {task.model && <span className="worker-model">{shortModel(task.model)}</span>}
            <span className="worker-time">{elapsed(task, now)}</span>
          </span>
          <span className="worker-activity">{state(task)}</span>
        </button>
      ))}
    </div>
  )
}
