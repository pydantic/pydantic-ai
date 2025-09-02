import * as path from '@std/path'
import { exists } from '@std/fs/exists'
import { contentType } from '@std/media-types'
import { type McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js'
import z from 'zod'
import { encodeBase64 } from '@std/encoding/base64'

/**
 * Returns the temporary directory in the local filesystem for file persistence.
 */
export function createRootDir(): string {
  return Deno.makeTempDirSync({ prefix: 'mcp_run_python' })
}

/**
 * Register file related functions to the MCP server.
 * @param server The MCP Server
 * @param rootDir Directory in the local file system to read/write to.
 */
export function registerFileFunctions(server: McpServer, rootDir: string) {
  // File Upload
  server.registerTool(
    'upload_file_from_uri',
    {
      title: 'Upload file from URI',
      description: 'Ingest a file by URI and store it. Returns a canonical URL.',
      inputSchema: {
        uri: z.string().url().describe('file:// or https:// style URL'),
        filename: z
          .string()
          .describe('The name of the file to write.'),
      },
    },
    async ({ uri, filename }: { uri: string; filename: string }) => {
      const absPath = path.join(rootDir, filename)
      const fileResponse = await fetch(uri)
      if (fileResponse.body) {
        const file = await Deno.open(absPath, { write: true, create: true })
        await fileResponse.body.pipeTo(file.writable)
      }
      return {
        content: [{
          type: 'resource_link',
          uri: `file:///${filename}`,
          name: filename,
          mimeType: contentType(path.extname(absPath)),
        }],
      }
    },
  )

  // Register all the files in the local directory as resources
  server.registerResource(
    'read-file',
    new ResourceTemplate('file:///{filename}', {
      list: async (_extra) => {
        const resources = []
        for await (const dirEntry of Deno.readDir(rootDir)) {
          if (!dirEntry.isFile) continue
          resources.push({
            uri: `file:///${dirEntry.name}`,
            name: dirEntry.name,
            mimeType: contentType(path.extname(dirEntry.name)),
          })
        }
        return { resources: resources }
      },
    }),
    {
      title: 'Read file.',
      description: 'Read file from persistent storage',
    },
    async (uri, { filename }) => {
      const absPath = path.join(rootDir, ...(Array.isArray(filename) ? filename : [filename]))
      const mime = contentType(path.extname(absPath))
      const fileBytes = await Deno.readFile(absPath)

      // Check if it's text-based
      if (mime && /^(text\/|.*\/json$|.*\/csv$|.*\/javascript$|.*\/xml$)/.test(mime.split(';')[0])) {
        const text = new TextDecoder().decode(fileBytes)
        return { contents: [{ uri: uri.href, mimeType: mime, text: text }] }
      } else {
        const base64 = encodeBase64(fileBytes)
        return { contents: [{ uri: uri.href, mimeType: mime, blob: base64 }] }
      }
    },
  )

  // This functions only checks if the file exits
  // Download happens through the registered resource
  server.registerTool('retrieve_file', {
    title: 'Retrieve a file',
    description: 'Retrieve a file from the persistent file store.',
    inputSchema: { filename: z.string().describe('The name of the file to read.') },
  }, async ({ filename }) => {
    const absPath = path.join(rootDir, filename)
    if (await exists(absPath, { isFile: true })) {
      return {
        content: [{
          type: 'resource_link',
          uri: `file:///${filename}`,
          name: filename,
          mimeType: contentType(path.extname(absPath)),
        }],
      }
    } else {
      return {
        content: [{ 'type': 'text', 'text': `Failed to retrieve file ${filename}. File not found.` }],
        isError: true,
      }
    }
  })

  // File deletion
  server.registerTool('delete_file', {
    title: 'Delete a file',
    description: 'Delete a file from the persistent file store.',
    inputSchema: { filename: z.string().describe('The name of the file to delete.') },
  }, async ({ filename }) => {
    const absPath = path.join(rootDir, filename)
    if (await exists(absPath, { isFile: true })) {
      await Deno.remove(absPath)
      return {
        content: [{
          type: 'text',
          text: `${filename} deleted successfully`,
        }],
      }
    } else {
      return {
        content: [{ 'type': 'text', 'text': `Failed to delete file ${filename}. File not found.` }],
        isError: true,
      }
    }
  })
}
