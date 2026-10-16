import { memo } from 'react'
import type { Diff } from '../protocol'

/** A diff's lines, with its `---`/`+++` file headers left out: only before its first hunk, so a
 * removed line that itself starts with `--` (a SQL comment, a Markdown rule) is kept. */
function body(diff: string): string[] {
  const lines = diff.replace(/\n$/, '').split('\n')
  const first = lines.findIndex((line) => line.startsWith('@@ '))
  if (first < 0) return lines
  return lines.filter((line, index) => index >= first || !(line.startsWith('--- ') || line.startsWith('+++ ')))
}

export function diffStats(diff: string): { added: number; removed: number } {
  let added = 0
  let removed = 0
  for (const line of body(diff)) {
    if (line.startsWith('+')) added++
    else if (line.startsWith('-')) removed++
  }
  return { added, removed }
}

/** A unified diff with old and new line numbers, additions and removals tinted. */
export const DiffView = memo(function DiffView({ diff }: { diff: Diff }) {
  const rows: { kind: string; old: string; new: string; text: string }[] = []
  let oldLine = 0
  let newLine = 0
  let numbered = false
  for (const line of body(diff.diff)) {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)/.exec(line)
    if (hunk) {
      oldLine = Number(hunk[1])
      newLine = Number(hunk[2])
      numbered = hunk[1] !== '0' || hunk[2] !== '1' || rows.length > 0
      if (numbered) rows.push({ kind: 'hunk', old: '', new: '', text: line })
    } else if (line.startsWith('+')) {
      rows.push({ kind: 'add', old: '', new: String(newLine++), text: line.slice(1) })
    } else if (line.startsWith('-')) {
      rows.push({ kind: 'del', old: String(oldLine++), new: '', text: line.slice(1) })
    } else if (line.startsWith('\\')) {
      continue
    } else if (line || rows.length) {
      rows.push({ kind: 'ctx', old: String(oldLine++), new: String(newLine++), text: line.slice(1) })
    }
  }
  return (
    <div className="diff">
      <table>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={`diff-${row.kind}`}>
              {numbered && <td className="ln">{row.old}</td>}
              {numbered && <td className="ln">{row.new}</td>}
              <td className="sign">{row.kind === 'add' ? '+' : row.kind === 'del' ? '−' : ''}</td>
              <td className="code">{row.text || ' '}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {diff.truncated && <div className="diff-note">Diff truncated</div>}
    </div>
  )
})
