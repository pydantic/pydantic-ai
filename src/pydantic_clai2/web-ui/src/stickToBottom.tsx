import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowDownIcon } from './components/icons'

/** How near the bottom, in pixels, the reader's own scroll must end to follow new output again. */
const NEAR = 4
const UP_KEYS = new Set(['ArrowUp', 'PageUp', 'Home'])

/** Keeps a scroller at its bottom as `content` changes, until the reader scrolls up, and follows again once they
 * scroll back to the bottom, `jump`, or `restart` changes (a new prompt, or another session's transcript).
 *
 * Only the reader's input decides: a wheel, touch, or key scrolling up stops following at once, before the browser
 * scrolls, so new output arriving every few milliseconds cannot snap the view back; the scrolls this makes itself are
 * ignored. `behind` says output arrived while the reader was away from the bottom. */
export function useStickToBottom<T extends HTMLElement>(content: unknown, restart: unknown) {
  const scroller = useRef<T>(null)
  const following = useRef(true)
  // Where the last scroll left the scroller, or where this put it; a scroll event elsewhere is the reader's.
  const top = useRef(0)
  const [behind, setBehind] = useState(false)

  const follow = useCallback(() => {
    following.current = true
    setBehind(false)
  }, [])

  /** Takes in a scroll made since the last look: one up stops following, one down to the bottom follows again. */
  const settle = useCallback(
    (element: T) => {
      const moved = element.scrollTop - top.current
      top.current = element.scrollTop
      const gap = element.scrollHeight - element.scrollTop - element.clientHeight
      // Upwards, unless content shrank and the browser clamped the scroller to its new bottom.
      if (moved < 0 && gap > 1) following.current = false
      else if (moved > 0 && gap <= NEAR) follow()
    },
    [follow],
  )

  useEffect(() => {
    const element = scroller.current
    if (!element) return
    const gap = () => element.scrollHeight - element.scrollTop - element.clientHeight
    /** Stops following for input that scrolls this scroller up, not a scroller inside it or one already at its top. */
    const leave = (target: EventTarget | null) => {
      if (element.scrollTop <= 0) return
      let node = target instanceof Element && element.contains(target) ? target : null
      for (; node && node !== element; node = node.parentElement) {
        if (node.scrollTop > 0 && node.scrollHeight > node.clientHeight) return
      }
      following.current = false
    }
    let touchY = 0
    // Whether the reader last clicked in this scroller, so the keys they press with nothing focused scroll it.
    let clicked = false
    const onScroll = () => settle(element)
    const onWheel = (event: WheelEvent) => {
      // Pinch-zoom and sideways swipes are not scrolls up.
      if (event.ctrlKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return
      if (event.deltaY < 0) leave(event.target)
      else if (event.deltaY > 0 && gap() <= NEAR) follow()
    }
    const onTouchStart = (event: TouchEvent) => {
      touchY = event.touches[0]?.clientY ?? 0
    }
    const onTouchMove = (event: TouchEvent) => {
      // A finger moving down from the highest point it reached drags the content down, scrolling up.
      const y = event.touches[0]?.clientY ?? 0
      if (y > touchY + NEAR) leave(event.target)
      touchY = Math.min(touchY, y)
    }
    const onPointerDown = (event: PointerEvent) => {
      clicked = element.contains(event.target as Node)
      // Grabbing the scrollbar, which lies outside the client area.
      if (event.target === element && event.offsetX >= element.clientWidth) leave(element)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      const ours = element.contains(event.target as Node) || (event.target === document.body && clicked)
      if (ours && (UP_KEYS.has(event.key) || (event.key === ' ' && event.shiftKey))) leave(event.target)
    }
    element.addEventListener('scroll', onScroll, { passive: true })
    element.addEventListener('wheel', onWheel, { passive: true })
    element.addEventListener('touchstart', onTouchStart, { passive: true })
    element.addEventListener('touchmove', onTouchMove, { passive: true })
    document.addEventListener('pointerdown', onPointerDown, true)
    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      element.removeEventListener('scroll', onScroll)
      element.removeEventListener('wheel', onWheel)
      element.removeEventListener('touchstart', onTouchStart)
      element.removeEventListener('touchmove', onTouchMove)
      document.removeEventListener('pointerdown', onPointerDown, true)
      document.removeEventListener('keydown', onKeyDown, true)
    }
  }, [follow, settle])

  useLayoutEffect(() => {
    follow()
    // A kept transcript shown again starts here too: a scroll its browser reset while hidden is not the reader's.
    if (scroller.current) top.current = scroller.current.scrollTop
  }, [restart, follow])

  useLayoutEffect(() => {
    const element = scroller.current
    if (!element) return
    // A scroll up the browser has not reported yet, such as a find-in-page jump, must not be overwritten.
    settle(element)
    if (following.current) {
      element.scrollTop = element.scrollHeight
      top.current = element.scrollTop
    } else if (element.scrollHeight - element.scrollTop - element.clientHeight > NEAR) setBehind(true)
  }, [content, restart, settle])

  const jump = useCallback(() => {
    const element = scroller.current
    if (!element) return
    follow()
    element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' })
  }, [follow])

  return { scroller, behind, jump }
}

/** The button that scrolls back to new output, floating at the bottom of the scroller it is the last child of. */
export function JumpToLatest({ shown, onJump }: { shown: boolean; onJump: () => void }) {
  return (
    <div className="jump-dock">
      {shown && (
        <button className="jump-to-latest" onClick={onJump} title="Jump to latest">
          <ArrowDownIcon size={14} />
          Jump to latest
        </button>
      )}
    </div>
  )
}
