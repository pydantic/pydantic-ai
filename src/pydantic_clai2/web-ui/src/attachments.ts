/** An image pasted or dropped into the composer, waiting to be sent with the next prompt. */
export interface Attachment {
  id: string
  name: string
  /** A `data:` URL, which both the preview shows (the page's policy allows no `blob:` images) and the server reads. */
  url: string
}

/** The server's limit on each image; it converts every image to PNG and checks again. */
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
/**
 * Larger images are scaled down to this longest side first, as models scale them about as far anyway: the server
 * converts each image to PNG, and even an uncompressed one this size stays under its 10 MiB limit.
 */
const MAX_SIDE = 1568
/** The server's limit on the images sent together. */
const MAX_BATCH_BYTES = 32 * 1024 * 1024
/** The formats the server reads; anything else the browser can draw is sent as PNG. */
const SERVER_TYPES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff'])
let count = 0

/** The images among pasted or dropped files. */
export function imageFiles(files: Iterable<File>): File[] {
  return Array.from(files).filter((file) => file.type.startsWith('image/'))
}

/**
 * Read `files` for sending beside the images already `attached`: each scaled down, or drawn as PNG, when the server
 * would otherwise refuse it.
 */
export async function readImages(files: readonly File[], attached: readonly Attachment[]): Promise<Attachment[]> {
  const read = await Promise.all(files.map(readImage))
  const bytes = (attachment: Attachment) => (base64(attachment).length * 3) / 4
  const large = read.find((attachment) => bytes(attachment) > MAX_IMAGE_BYTES)
  if (large) throw new Error(`${large.name} is over the 10 MiB attachment limit.`)
  if ([...attached, ...read].reduce((total, attachment) => total + bytes(attachment), 0) > MAX_BATCH_BYTES) {
    throw new Error('These images are over the 32 MiB limit on images sent together.')
  }
  return read
}

async function readImage(file: File): Promise<Attachment> {
  const name = file.name || 'Pasted image'
  const picture = await decode(file)
  if (picture) {
    const scale = Math.min(1, MAX_SIDE / Math.max(picture.width, picture.height))
    if (scale < 1 || !SERVER_TYPES.has(file.type)) return { id: String(++count), name, url: draw(picture, scale, file.type) }
  } else if (!SERVER_TYPES.has(file.type)) {
    throw new Error(`${name} is not an image CLAI can read. Use PNG, JPEG, GIF, WebP, BMP, or TIFF.`)
  }
  return { id: String(++count), name, url: await dataUrl(file) }
}

/** `file` as the browser draws it, or `null` when it cannot; an SVG, which `createImageBitmap` refuses, through an `<img>`. */
async function decode(file: File): Promise<{ image: CanvasImageSource; width: number; height: number } | null> {
  try {
    const bitmap = await createImageBitmap(file)
    return { image: bitmap, width: bitmap.width, height: bitmap.height }
  } catch {
    try {
      const image = new Image()
      image.src = await dataUrl(file)
      await image.decode()
      return { image, width: image.naturalWidth, height: image.naturalHeight }
    } catch {
      return null
    }
  }
}

/** The picture scaled by `scale`: a JPEG stays one, as it has no transparency to keep and is far smaller; the rest are PNG. */
function draw(picture: { image: CanvasImageSource; width: number; height: number }, scale: number, type: string): string {
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(picture.width * scale))
  canvas.height = Math.max(1, Math.round(picture.height * scale))
  canvas.getContext('2d')?.drawImage(picture.image, 0, 0, canvas.width, canvas.height)
  if (picture.image instanceof ImageBitmap) picture.image.close()
  return type === 'image/jpeg' ? canvas.toDataURL('image/jpeg', 0.92) : canvas.toDataURL('image/png')
}

function dataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${file.name || 'the image'}.`))
    reader.readAsDataURL(file)
  })
}

/** What the server takes for an image: its bytes, base64-encoded. */
export const base64 = (attachment: Attachment) => attachment.url.slice(attachment.url.indexOf(',') + 1)
