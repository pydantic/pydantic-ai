import { useCallback, useEffect, useRef, useState } from 'react'
import type { Attachment } from './attachments'

interface Draft {
  id: string
  text: string
  attachments: readonly Attachment[]
}

const EMPTY: readonly Attachment[] = []

/** Persist the page's existing conversation drafts, without making the database another editor state. */
export function useDraftStorage(
  drafts: Record<string, string>,
  attached: Record<string, readonly Attachment[]>,
  restore: (drafts: Record<string, string>, attached: Record<string, readonly Attachment[]>) => void,
) {
  const [ready, setReady] = useState(false)
  const [unsaved, setUnsaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const db = useRef<IDBDatabase | null>(null)
  const pending = useRef(new Map<string, Draft>())
  const writing = useRef(false)
  const timer = useRef(0)
  const previous = useRef({ drafts, attached })

  const flush = useCallback(function flush() {
    window.clearTimeout(timer.current)
    if (!db.current || writing.current || pending.current.size === 0) return
    const batch = [...pending.current.values()]
    const failed = () => {
      writing.current = false
      setError('Drafts could not be saved in this browser. Keep this page open or copy your draft before leaving.')
    }
    let transaction: IDBTransaction | null = null
    try {
      transaction = db.current.transaction('drafts', 'readwrite')
      const store = transaction.objectStore('drafts')
      writing.current = true
      transaction.onabort = failed
      transaction.oncomplete = () => {
        writing.current = false
        for (const draft of batch) {
          // A completed write only saves its own revision, never edits made while it was in flight.
          if (pending.current.get(draft.id) === draft) pending.current.delete(draft.id)
        }
        setUnsaved(pending.current.size > 0)
        if (db.current) setError(null)
        if (pending.current.size > 0) flush()
      }
      for (const draft of batch) {
        if (draft.text === '' && draft.attachments.length === 0) store.delete(draft.id)
        else store.put(draft)
      }
    } catch {
      transaction?.abort()
      failed()
    }
  }, [])

  useEffect(() => {
    let stopped = false
    let failed = false
    let opened: IDBDatabase | null = null
    const fail = () => {
      if (stopped) return
      failed = true
      opened?.close()
      db.current = null
      setError('Saved drafts could not be loaded. You can keep editing, but new drafts will stay only in this page.')
      setReady(true)
    }
    try {
      const request = window.indexedDB.open('clai-drafts', 1)
      request.onupgradeneeded = () => request.result.createObjectStore('drafts', { keyPath: 'id' })
      request.onerror = fail
      request.onblocked = fail
      request.onsuccess = () => {
        opened = request.result
        if (stopped || failed) {
          opened.close()
          return
        }
        opened.onversionchange = () => {
          opened?.close()
          db.current = null
          setError('Draft storage changed in another page. Reload to restore automatic saving.')
        }
        try {
          const transaction = opened.transaction('drafts')
          const read = transaction.objectStore('drafts').getAll()
          transaction.onabort = fail
          transaction.oncomplete = () => {
            if (stopped) return
            const found = (read.result as Draft[]).map((draft) => ({
              ...draft,
              // New attachments use a page-local numeric counter. Restored IDs must not collide with it.
              attachments: draft.attachments.map((image, index) => ({ ...image, id: `restored-${draft.id}-${index}` })),
            }))
            previous.current = {
              drafts: Object.fromEntries(found.map((draft) => [draft.id, draft.text])),
              attached: Object.fromEntries(found.map((draft) => [draft.id, draft.attachments])),
            }
            db.current = opened
            restore(previous.current.drafts, previous.current.attached)
            setReady(true)
          }
        } catch {
          fail()
        }
      }
    } catch {
      fail()
    }
    return () => {
      stopped = true
      window.clearTimeout(timer.current)
      opened?.close()
      db.current = null
    }
  }, [restore])

  useEffect(() => {
    if (!ready) return
    const before = previous.current
    previous.current = { drafts, attached }
    let changed = false
    let immediate = false
    const ids = new Set([...Object.keys(drafts), ...Object.keys(attached), ...Object.keys(before.drafts), ...Object.keys(before.attached)])
    for (const id of ids) {
      if (!id) continue
      const text = drafts[id] ?? ''
      const images = attached[id] ?? EMPTY
      const changedImages = images !== (before.attached[id] ?? EMPTY)
      if (text === (before.drafts[id] ?? '') && !changedImages) continue
      pending.current.set(id, { id, text, attachments: images })
      changed = true
      immediate ||= changedImages || (text === '' && images.length === 0)
    }
    if (!changed) return
    setUnsaved(true)
    window.clearTimeout(timer.current)
    if (immediate) flush()
    else timer.current = window.setTimeout(flush, 300)
  }, [drafts, attached, ready, flush])

  useEffect(() => {
    const hide = () => {
      if (document.hidden) flush()
    }
    document.addEventListener('visibilitychange', hide)
    return () => document.removeEventListener('visibilitychange', hide)
  }, [flush])

  useEffect(() => {
    if (!unsaved) return
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = true
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [unsaved])

  return { ready, unsaved, error, flush }
}
