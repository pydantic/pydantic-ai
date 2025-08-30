/// <reference types="npm:@types/node@22.12.0" />

import './polyfill.ts'
import http from 'node:http'
import { randomUUID } from 'node:crypto'
import { parseArgs } from '@std/cli/parse-args'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js'
import { type LoggingLevel, SetLevelRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'
import { asXml, getRootDir, runCode } from './runCode.ts'
import { Buffer } from 'node:buffer'
import * as path from 'node:path'

const VERSION = '0.0.13'

export async function main() {
  // Parse global flags once, then branch on subcommand
  const flags = parseArgs(Deno.args, {
    string: ['port'],
    default: { port: '3001', mount: false },
  })
  const mode = (flags._[0] as string | undefined) ?? ''
  const port = parseInt(flags.port as string)
  const mount = flags.mount as string | boolean

  if (mode === 'stdio') {
    await runStdio(mount)
  } else if (mode === 'streamable_http') {
    runStreamableHttp(port, mount)
  } else if (mode === 'sse') {
    runSse(port, mount)
  } else if (mode === 'warmup') {
    await warmup(mount)
  } else {
    console.error(
      `\
Invalid arguments.

Usage: deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python [stdio|streamable_http|sse|warmup] [--port <port>] [--mount [dir]]

options:
  --port <port>   Port to run the SSE/HTTP server on (default: 3001)
  --mount [dir]   Relative or absolute directory path or boolean. If omitted: false; if provided without value: true`,
    )
    Deno.exit(1)
  }
}

/*
 * Resolve a mountDir cli option to a specific directory
 */
export function resolveMountDir(mountDir: string): string {
  // Base dir created by emscriptem
  // See https://emscripten.org/docs/api_reference/Filesystem-API.html#file-system-api
  const baseDir = '/home/web_user'

  if (mountDir.trim() === '') {
    return path.join(baseDir, 'persistent')
  }

  if (path.isAbsolute(mountDir)) {
    return mountDir
  }

  // relative path
  return path.join(baseDir, mountDir)
}

/*
 * Ensure and cleanup the root directory used by the MCP server
 */
function ensureRootDir() {
  Deno.mkdirSync(getRootDir(), { recursive: true })
}

function cleanupRootDir() {
  try {
    Deno.removeSync(getRootDir(), { recursive: true })
  } catch (err) {
    if (!(err instanceof Deno.errors.NotFound)) throw err
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(mount: string | boolean): McpServer {
  const server = new McpServer(
    {
      name: 'MCP Run Python',
      version: VERSION,
    },
    {
      instructions: 'Call the "run_python_code" tool with the Python code to run.',
      capabilities: {
        logging: {},
      },
    },
  )

  let mountDirDescription: string
  let mountDir: string | null
  if (mount !== false) {
    // Create temporary directory
    ensureRootDir()
    // Resolve mounted directory
    mountDir = resolveMountDir(typeof mount === 'string' ? mount : '')
    mountDirDescription = `To store files permanently use the directory at: ${mountDir}\n`
  } else {
    mountDir = null
    mountDirDescription = ''
  }

  const toolDescription = `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return value.

The code will be executed with Python 3.12.
${mountDirDescription}
Dependencies may be defined via PEP 723 script metadata, e.g. to install "pydantic", the script should start
with a comment of the form:

# /// script
# dependencies = ['pydantic']
# ///
print('python code here')
`

  let setLogLevel: LoggingLevel = 'emergency'

  server.server.setRequestHandler(SetLevelRequestSchema, (request) => {
    setLogLevel = request.params.level
    return {}
  })

  server.registerTool(
    'run_python_code',
    {
      title: 'Run Python Code',
      description: toolDescription,
      inputSchema: { python_code: z.string().describe('Python code to run') },
    },
    async ({ python_code }: { python_code: string }) => {
      const logPromises: Promise<void>[] = []
      const result = await runCode(
        [
          {
            name: 'main.py',
            content: python_code,
            active: true,
          },
        ],
        (level, data) => {
          if (LogLevels.indexOf(level) >= LogLevels.indexOf(setLogLevel)) {
            logPromises.push(server.server.sendLoggingMessage({ level, data }))
          }
        },
        mountDir,
      )
      await Promise.all(logPromises)
      return {
        content: [{ type: 'text', text: asXml(result) }],
      }
    },
  )
  return server
}

/*
 * Define some QOL functions for both the SSE and Streamable HTTP server implementation
 */
function httpGetUrl(req: http.IncomingMessage): URL {
  return new URL(req.url ?? '', `http://${req.headers.host ?? 'unknown'}`)
}

function httpGetBody(req: http.IncomingMessage): Promise<JSON> {
  // https://nodejs.org/en/learn/modules/anatomy-of-an-http-transaction#request-body
  return new Promise((resolve) => {
    // deno-lint-ignore no-explicit-any
    const bodyParts: any[] = []
    let body
    req
      .on('data', (chunk) => {
        bodyParts.push(chunk)
      })
      .on('end', () => {
        body = Buffer.concat(bodyParts).toString()
        resolve(JSON.parse(body))
      })
  })
}

function httpSetTextResponse(
  res: http.ServerResponse,
  status: number,
  text: string,
) {
  res.setHeader('Content-Type', 'text/plain')
  res.statusCode = status
  res.end(`${text}\n`)
}

function httpSetJsonResponse(
  res: http.ServerResponse,
  status: number,
  text: string,
  code: number,
) {
  res.setHeader('Content-Type', 'application/json')
  res.statusCode = status
  res.write(
    JSON.stringify({
      jsonrpc: '2.0',
      error: {
        code: code,
        message: text,
      },
      id: null,
    }),
  )
  res.end()
}

/*
 * Run the MCP server using the Streamable HTTP transport
 */
function runStreamableHttp(port: number, mount: string | boolean) {
  // https://github.com/modelcontextprotocol/typescript-sdk?tab=readme-ov-file#with-session-management
  const mcpServer = createServer(mount)
  const transports: { [sessionId: string]: StreamableHTTPServerTransport } = {}

  const server = http.createServer(async (req, res) => {
    const url = httpGetUrl(req)
    let pathMatch = false
    function match(method: string, path: string): boolean {
      if (url.pathname === path) {
        pathMatch = true
        return req.method === method
      }
      return false
    }

    // Reusable handler for GET and DELETE requests
    async function handleSessionRequest() {
      const sessionId = req.headers['mcp-session-id'] as string | undefined
      if (!sessionId || !transports[sessionId]) {
        httpSetTextResponse(res, 400, 'Invalid or missing session ID')
        return
      }

      const transport = transports[sessionId]
      await transport.handleRequest(req, res)
    }

    // Handle different request methods and paths
    if (match('POST', '/mcp')) {
      // Check for existing session ID
      const sessionId = req.headers['mcp-session-id'] as string | undefined
      let transport: StreamableHTTPServerTransport

      const body = await httpGetBody(req)

      if (sessionId && transports[sessionId]) {
        // Reuse existing transport
        transport = transports[sessionId]
      } else if (!sessionId && isInitializeRequest(body)) {
        // New initialization request
        transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomUUID(),
          onsessioninitialized: (sessionId) => {
            // Store the transport by session ID
            transports[sessionId] = transport
          },
        })

        // Clean up transport when closed
        transport.onclose = () => {
          if (transport.sessionId) {
            delete transports[transport.sessionId]
          }
        }

        await mcpServer.connect(transport)
      } else {
        httpSetJsonResponse(
          res,
          400,
          'Bad Request: No valid session ID provided',
          -32000,
        )
        return
      }

      // Handle the request
      await transport.handleRequest(req, res, body)
    } else if (match('GET', '/mcp')) {
      // Handle server-to-client notifications via SSE
      await handleSessionRequest()
    } else if (match('DELETE', '/mcp')) {
      // Handle requests for session termination
      await handleSessionRequest()
    } else if (pathMatch) {
      httpSetTextResponse(res, 405, 'Method not allowed')
    } else {
      httpSetTextResponse(res, 404, 'Page not found')
    }
  })

  // Cleanup root dir on server close
  server.on('close', () => {
    cleanupRootDir()
  })

  server.listen(port, () => {
    console.log(
      `Running MCP Run Python version ${VERSION} with Streamable HTTP transport on port ${port}`,
    )
  })
}

/*
 * Run the MCP server using the SSE transport, e.g. over HTTP.
 */
function runSse(port: number, mount: string | boolean) {
  const mcpServer = createServer(mount)
  const transports: { [sessionId: string]: SSEServerTransport } = {}

  const server = http.createServer(async (req, res) => {
    const url = httpGetUrl(req)
    let pathMatch = false
    function match(method: string, path: string): boolean {
      if (url.pathname === path) {
        pathMatch = true
        return req.method === method
      }
      return false
    }

    if (match('GET', '/sse')) {
      const transport = new SSEServerTransport('/messages', res)
      transports[transport.sessionId] = transport
      res.on('close', () => {
        delete transports[transport.sessionId]
      })
      await mcpServer.connect(transport)
    } else if (match('POST', '/messages')) {
      const sessionId = url.searchParams.get('sessionId') ?? ''
      const transport = transports[sessionId]
      if (transport) {
        await transport.handlePostMessage(req, res)
      } else {
        httpSetTextResponse(
          res,
          400,
          `No transport found for sessionId '${sessionId}'`,
        )
      }
    } else if (pathMatch) {
      httpSetTextResponse(res, 405, 'Method not allowed')
    } else {
      httpSetTextResponse(res, 404, 'Page not found')
    }
  })

  // Cleanup root dir on server close
  server.on('close', () => {
    cleanupRootDir()
  })

  server.listen(port, () => {
    console.log(
      `Running MCP Run Python version ${VERSION} with SSE transport on port ${port}`,
    )
  })
}

/*
 * Run the MCP server using the Stdio transport.
 */
async function runStdio(mount: string | boolean) {
  const mcpServer = createServer(mount)
  const transport = new StdioServerTransport()

  // Cleanup root dir on transport close
  transport.onclose = () => {
    cleanupRootDir()
  }

  await mcpServer.connect(transport)
}

/*
 * Run pyodide to download packages which can otherwise interrupt the server
 */
async function warmup(mount?: string | boolean) {
  console.error(
    `Running warmup script for MCP Run Python version ${VERSION}...` +
      (mount ? ` (mount: ${typeof mount === 'string' ? mount : 'enabled'})` : ''),
  )
  const code = `
import numpy
a = numpy.array([1, 2, 3])
print('numpy array:', a)
a
`
  const result = await runCode(
    [
      {
        name: 'warmup.py',
        content: code,
        active: true,
      },
    ],
    (level, data) =>
      // use warn to avoid recursion since console.log is patched in runCode
      console.error(`${level}: ${data}`),
    null,
  )
  console.log('Tool return value:')
  console.log(asXml(result))
  console.log('\nwarmup successful 🎉')
}

// list of log levels to use for level comparison
const LogLevels: LoggingLevel[] = [
  'debug',
  'info',
  'notice',
  'warning',
  'error',
  'critical',
  'alert',
  'emergency',
]

await main()
