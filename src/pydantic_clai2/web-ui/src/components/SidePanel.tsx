import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { MOD } from '../platform'
import type { PanelInfo } from '../protocol'
import { CloseIcon, ExternalIcon, PanelIcon, TracesIcon } from './icons'

const MIN_WIDTH = 280

/** A panel's key, as the page remembers which one is open. */
export const panelKey = (panel: PanelInfo) => `${panel.plugin}/${panel.id}`

export function PanelGlyph({ panel, size }: { panel: PanelInfo; size: number }) {
  return panel.icon === 'traces' ? <TracesIcon size={size} /> : <PanelIcon size={size} />
}

interface Opened {
  window: Window
  url: string
}

/** The windows opened for panels whose pages refuse to be framed, by panel key, for as long as this page lives. */
const opened = new Map<string, Opened>()

/** Point an opened window at `url`; `false` once it is closed or the browser refuses, so the next click opens one. */
function navigate(key: string, existing: Opened, url: string): boolean {
  if (!existing.window.closed) {
    try {
      if (existing.url !== url) existing.window.location.href = url
      existing.url = url
      return true
    } catch {
      // The browser refused, such as after the page cut its tie to this one: forget the window.
    }
  }
  opened.delete(key)
  return false
}

/** Open `panel`'s page in a window beside this one, or point the one already open at it and bring it forward. */
function openBeside(panel: PanelInfo, url: string): boolean {
  const key = panelKey(panel)
  const existing = opened.get(key)
  if (existing && navigate(key, existing, url)) {
    existing.window.focus()
    return true
  }
  const width = Math.max(480, Math.min(1200, Math.round(screen.availWidth * 0.45)))
  const availLeft = (screen as Screen & { availLeft?: number }).availLeft ?? 0
  const screenRight = availLeft + screen.availWidth
  const right = window.screenX + window.outerWidth
  // Right of this window, else left of it, else against the screen's right edge.
  const left =
    right + width <= screenRight
      ? right
      : window.screenX - width >= availLeft
        ? window.screenX - width
        : screenRight - width
  const features = `popup,width=${width},height=${window.outerHeight},left=${left},top=${window.screenY}`
  // It keeps this page as its opener, the one way to point it at each new turn. Across origins, the page it shows
  // cannot read this one or its address, which carries the access token, and the page sends no referrer.
  const child = window.open(url, `clai-${key.replace(/[^A-Za-z0-9]/g, '-')}`, features)
  if (!child) return false
  opened.set(key, { window: child, url })
  child.focus()
  return true
}

/** Whether a window opened for a panel is still open, so the page keeps asking where it should point. */
export const following = () => [...opened.values()].some((entry) => !entry.window.closed)

/** Point every window opened for a panel at the panel's current page, such as the trace of a turn that started. */
export function followWindows(panels: PanelInfo[]) {
  for (const panel of panels) {
    const key = panelKey(panel)
    const existing = opened.get(key)
    if (existing && panel.url && !panel.embed) navigate(key, existing, panel.url)
  }
}

/** A plugin's page beside the conversation: framed when the page allows it, otherwise opened in a window beside this one. */
export function SidePanel({
  panel,
  width,
  onResize,
  onClose,
  onSettings,
  onError,
}: {
  panel: PanelInfo
  width: number
  onResize: (width: number, done: boolean) => void
  onClose: () => void
  onSettings: (plugin: string) => void
  onError: (message: string) => void
}) {
  const [resizing, setResizing] = useState(false)
  const start = useRef({ x: 0, width })

  const grab = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    start.current = { x: event.clientX, width }
    setResizing(true)
  }
  const drag = (event: ReactPointerEvent<HTMLDivElement>, done: boolean) => {
    if (!resizing) return
    const next = start.current.width + start.current.x - event.clientX
    onResize(Math.round(Math.max(MIN_WIDTH, Math.min(next, window.innerWidth * 0.7))), done)
    if (done) setResizing(false)
  }

  const url = panel.url
  // The server sends only well-formed http and https addresses; a throw here would blank the page.
  const host = url && URL.canParse(url) ? new URL(url).host : (url ?? '')
  return (
    <aside className={`side-panel ${resizing ? 'resizing' : ''}`} aria-label={panel.title}>
      <div
        className="side-panel-resize"
        role="separator"
        aria-orientation="vertical"
        title="Drag to resize"
        onPointerDown={grab}
        onPointerMove={(event) => drag(event, false)}
        onPointerUp={(event) => drag(event, true)}
      />
      <header className="side-panel-head">
        <PanelGlyph panel={panel} size={15} />
        <span className="side-panel-title">{panel.title}</span>
        {url && <span className="badge">{panel.embed ? host : 'own window'}</span>}
        {url && panel.embed && (
          <a className="icon-button" href={url} target="_blank" rel="noopener noreferrer" title="Open in a new tab">
            <ExternalIcon size={14} />
          </a>
        )}
        <button
          className="icon-button"
          onClick={onClose}
          title={`Hide the panel (${MOD}⇧L)`}
          aria-label="Hide the panel"
        >
          <CloseIcon size={14} />
        </button>
      </header>
      {url && panel.embed ? (
        <iframe
          className="side-panel-frame"
          src={url}
          title={panel.title}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox"
          referrerPolicy="no-referrer"
        />
      ) : (
        <div className="side-panel-body">
          {panel.note && <p>{panel.note}</p>}
          {url ? (
            <>
              <p>
                {host} does not let other sites show it inside their pages, so it opens in a window of its own beside
                this one. That window follows each new turn.
              </p>
              <div>
                <button
                  className="primary small"
                  onClick={() => {
                    if (!openBeside(panel, url)) onError('The browser blocked the window: allow pop-ups for this page.')
                  }}
                >
                  <ExternalIcon size={14} /> Open {panel.title}
                </button>
              </div>
              <div className="side-panel-link">
                <span className="side-panel-label">Current link</span>
                <a href={url} target="_blank" rel="noopener noreferrer">
                  {url}
                </a>
              </div>
            </>
          ) : (
            <div>
              <button className="ghost small" onClick={() => onSettings(panel.plugin)}>
                Open {panel.plugin} settings
              </button>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}
