/// <reference types="npm:@types/node@22.12.0" />

import './polyfill.ts'
import http from 'node:http'
import { randomUUID } from 'node:crypto'
import { parseArgs } from '@std/cli/parse-args'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { asXml, registerCodeFunctions, runCode } from './runCode.ts'
import { Buffer } from 'node:buffer'
import { createRootDir, registerFileFunctions } from './files.ts'

const VERSION = '0.0.13'

export async function main() {
  // Parse global flags once, then branch on subcommand
  const flags = parseArgs(Deno.args, {
    string: ['port', 'mount'],
    default: { port: '3001' },
  })
  const mode = (flags._[0] as string | undefined) ?? ''
  const port = parseInt(flags.port as string)
  const rawMount = flags.mount as string | undefined
  const mount: string | boolean = rawMount === undefined ? false : rawMount === '' ? true : rawMount

  if (mode === 'stdio') {
    await runStdio(mount)
  } else if (mode === 'streamable_http') {
    runStreamableHttp(port, mount)
  } else if (mode === 'sse') {
    runSse(port, mount)
  } else if (mode === 'warmup') {
    await warmup()
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
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(rootDir: string | null, mount: string | boolean): McpServer {
  const server = new McpServer(
    {
      name: 'MCP Run Python',
      version: VERSION,
    },
    {
      instructions: 'Call the "run_python_code" tool with the Python code to run.' +
        (rootDir != null ? ` Persistent storage is mounted at: "${rootDir}".` : ''),
      capabilities: {
        resources: {},
        tools: {},
        logging: {},
      },
    },
  )

  registerCodeFunctions(server, rootDir, mount)
  if (rootDir != null) {
    registerFileFunctions(server, rootDir)
  }

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

function addDirCleanupCallback(server: http.Server | StdioServerTransport, dir: string) {
  let cleaned = false
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    try {
      Deno.removeSync(dir, { recursive: true })
    } catch {
      // ignore
    }
  }
  if (server instanceof http.Server) {
    server.on('close', cleanup)
  } else {
    server.onclose = cleanup
  }
  const handleSig = () => {
    try {
      server.close(() => {})
    } catch {
      // ignore
    }
    cleanup()
    Deno.exit()
  }
  Deno.addSignalListener('SIGINT', handleSig)
  Deno.addSignalListener('SIGTERM', handleSig)
  addEventListener('unload', cleanup)
}

/*
 * Run the MCP server using the Streamable HTTP transport
 */
function runStreamableHttp(port: number, mount: string | boolean) {
  const rootDir = mount !== false ? createRootDir() : null

  // https://github.com/modelcontextprotocol/typescript-sdk?tab=readme-ov-file#with-session-management
  const mcpServer = createServer(rootDir, mount)
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

  // Cleanup root dir on server close and on process signals
  if (rootDir != null) {
    addDirCleanupCallback(server, rootDir)
  }

  server.listen(port, () => {
    console.log(
      `Running MCP Run Python version ${VERSION} with SSE transport on port ${port}.`,
    )
  })
}

/*
 * Run the MCP server using the SSE transport, e.g. over HTTP.
 */
function runSse(port: number, mount: string | boolean) {
  const rootDir = mount !== false ? createRootDir() : null

  const mcpServer = createServer(rootDir, mount)
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

  // Cleanup root dir on server close and on process signals
  if (rootDir != null) {
    addDirCleanupCallback(server, rootDir)
  }

  server.listen(port, () => {
    console.log(
      `Running MCP Run Python version ${VERSION} with SSE transport on port ${port}.`,
    )
  })
}

/*
 * Run the MCP server using the Stdio transport.
 */
async function runStdio(mount: string | boolean) {
  const rootDir = mount !== false ? createRootDir() : null
  const mcpServer = createServer(rootDir, mount)
  const transport = new StdioServerTransport()

  // Cleanup root dir on server close and on process signals
  if (rootDir != null) {
    addDirCleanupCallback(transport, rootDir)
  }

  await mcpServer.connect(transport)
}

/*
 * Run pyodide to download packages which can otherwise interrupt the server
 */
async function warmup() {
  console.error(
    `Running warmup script for MCP Run Python version ${VERSION}...`,
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
    null,
  )
  console.log('Tool return value:')
  console.log(asXml(result))
  console.log('\nwarmup successful 🎉')
}

await main()
