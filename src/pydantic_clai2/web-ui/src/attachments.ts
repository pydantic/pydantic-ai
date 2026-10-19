/** An image pasted or dropped into the composer, waiting to be sent with the next prompt. */
export interface Attachment {
  id: string
  name: string
  /** A `data:` URL, which both the preview shows (the page's policy allows no `blob:` images) and the server reads. */
  url: string
}

/** The server's limit on each image; it converts every image to PNG and checks again. */
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
let attached = 0

/** The images among pasted or dropped files. */
export function imageFiles(files: Iterable<File>): File[] {
  return Array.from(files).filter((file) => file.type.startsWith('image/'))
}

/** Read `files` for sending, rejecting one over the size limit before reading any. */
export async function readImages(files: readonly File[]): Promise<Attachment[]> {
  const large = files.find((file) => file.size > MAX_IMAGE_BYTES)
  if (large) throw new Error(`${large.name || 'The image'} is over the 10 MiB attachment limit.`)
  return Promise.all(
    files.map(
      (file) =>
        new Promise<Attachment>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve({ id: String(++attached), name: file.name || 'Pasted image', url: String(reader.result) })
          reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${file.name || 'the image'}.`))
          reader.readAsDataURL(file)
        }),
    ),
  )
}

/** What the server takes for an image: its bytes, base64-encoded. */
export const base64 = (attachment: Attachment) => attachment.url.slice(attachment.url.indexOf(',') + 1)
