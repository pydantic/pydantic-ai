/// <reference types="npm:@types/node@22.12.0" />

import './polyfill.ts'
import http from 'node:http'
import { parseArgs } from '@std/cli/parse-args'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { type LoggingLevel, SetLevelRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'

import { asXml, runCode } from './runCode.ts'

const VERSION = '0.0.11'

export async function main() {
  const { args } = Deno
  if (args.length === 1 && args[0] === 'stdio') {
    await runStdio()
  } else if (args.length >= 1 && args[0] === 'sse') {
    const flags = parseArgs(Deno.args, {
      string: ['port'],
      default: { port: '3001' },
    })
    const port = parseInt(flags.port)
    runSse(port)
  } else if (args.length === 1 && args[0] === 'warmup') {
    await warmup()
  } else {
    console.error(
      `\
Invalid arguments.

Usage: deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python [stdio|sse|warmup]

options:
  --port <port>  Port to run the SSE server on (default: 3001)`,
    )
    Deno.exit(1)
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(): McpServer {
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

  const toolDescription = `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return value.

The code will be executed with Python 3.12.

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

  server.tool(
    'run_python_code',
    toolDescription,
    { python_code: z.string().describe('Python code to run') },
    async ({ python_code }: { python_code: string }) => {
      const logPromises: Promise<void>[] = []
      const result = await runCode([{
        name: 'main.py',
        content: python_code,
        active: true,
      }], (level, data) => {
        if (LogLevels.indexOf(level) >= LogLevels.indexOf(setLogLevel)) {
          logPromises.push(server.server.sendLoggingMessage({ level, data }))
        }
      })
      await Promise.all(logPromises)
      return {
        content: [{ type: 'text', text: asXml(result) }],
      }
    },
  )
  return server
}

/*
 * Run the MCP server using the SSE transport, e.g. over HTTP.
 */
function runSse(port: number) {
  const mcpServer = createServer()
  const transports: { [sessionId: string]: SSEServerTransport } = {}

  const server = http.createServer(async (req, res) => {
    const url = new URL(
      req.url ?? '',
      `http://${req.headers.host ?? 'unknown'}`,
    )
    let pathMatch = false
    function match(method: string, path: string): boolean {
      if (url.pathname === path) {
        pathMatch = true
        return req.method === method
      }
      return false
    }
    function textResponse(status: number, text: string) {
      res.setHeader('Content-Type', 'text/plain')
      res.statusCode = status
      res.end(`${text}\n`)
    }
    // console.log(`${req.method} ${url}`)

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
        textResponse(400, `No transport found for sessionId '${sessionId}'`)
      }
    } else if (pathMatch) {
      textResponse(405, 'Method not allowed')
    } else {
      textResponse(404, 'Page not found')
    }
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
async function runStdio() {
  const mcpServer = createServer()
  const transport = new StdioServerTransport()
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
  const result = await runCode([{
    name: 'warmup.py',
    content: code,
    active: true,
  }], (level, data) =>
    // use warn to avoid recursion since console.log is patched in runCode
    console.error(`${level}: ${data}`))
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
