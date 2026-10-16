import { createContext } from 'react'
import { token } from './api'

/** An image attached in the composer, read as a data URL: its thumbnail, and what is sent. */
export interface Attachment {
  id: string
  name: string
  url: string
}

/** The terminal's limit for one image; the server checks every limit again, and normalizes the image. */
const MAX_BYTES = 10 * 1024 * 1024

let attached = 0

function read(file: File): Promise<Attachment> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve({ id: `image-${++attached}`, name: file.name, url: String(reader.result) })
    reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${file.name}.`))
    reader.readAsDataURL(file)
  })
}

/** Read the image files among `files`, with a message for those left out; images only, as the terminal pastes. */
export async function readImages(files: Iterable<File>): Promise<{ images: Attachment[]; problem: string | null }> {
  const all = [...files]
  const images = all.filter((file) => file.type.startsWith('image/'))
  const fitting = images.filter((file) => file.size <= MAX_BYTES)
  const problem =
    fitting.length < images.length
      ? 'Images may be at most 10 MiB each.'
      : images.length < all.length
        ? 'Only images are supported.'
        : null
  return { images: await Promise.all(fitting.map(read)), problem }
}

/** What the server takes: the base64 data of each image. */
export const encoded = (images: Attachment[]) => images.map((image) => image.url.slice(image.url.indexOf(',') + 1))

/** The conversation whose transcript is shown, whose prompts' images `mediaUrl` serves; `null` for none. */
export const ConversationContext = createContext<string | null>(null)

export const mediaUrl = (conversation: string, id: string) =>
  `/media/${encodeURIComponent(conversation)}/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`
