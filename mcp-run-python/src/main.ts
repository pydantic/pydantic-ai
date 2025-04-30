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

const VERSION = '0.0.13'

export async function main() {
  const { args } = Deno
  const flags = parseArgs(args, {
    string: ['port', 'callbacks'],
    default: { port: '3001' },
  })
  const { _: [task], callbacks, port } = flags
  if (task === 'stdio') {
    await runStdio(callbacks)
  } else if (task === 'sse' || task === 'http') {
    runSse(parseInt(port), callbacks)
  } else if (task === 'warmup') {
    await warmup(callbacks)
  } else {
    console.error(
      `\
Invalid arguments.

Usage:
  deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto \\
    jsr:@pydantic/mcp-run-python [stdio|sse|warmup]

options:
  --port <port>                    Port to run the SSE server on (default: 3001).
  --callbacks <python-signatures>  Python code representing the signatures of client functions the server can call.`,
    )
    Deno.exit(1)
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(callbacks?: string): McpServer {
  const functions = _extractFunctions(callbacks)
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

  let toolDescription = `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return.

The code will be executed with Python 3.12.

Dependencies may be defined via PEP 723 script metadata.

To make HTTP requests, you must use the "httpx" library in async mode.

For example:

\`\`\`python
# /// script
# dependencies = ['httpx']
# ///
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get('https://example.com')
# return the text of the page
response.text
\`\`\`
`
  if (callbacks) {
    toolDescription += `
The following functions are already defined globally and available to call from within your code:

\`\`\`python
${callbacks}
\`\`\`
    `
  }

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
      const mainPy = {
        name: 'main.py',
        content: python_code,
        active: true,
      }
      const codeLog = (level: LoggingLevel, data: string) => {
        if (LogLevels.indexOf(level) >= LogLevels.indexOf(setLogLevel)) {
          logPromises.push(server.server.sendLoggingMessage({ level, data }))
        }
      }
      async function clientCallback(func: string, args?: string, kwargs?: string) {
        const { content } = await server.server.createMessage({
          messages: [],
          maxTokens: 0,
          systemPrompt: '',
          metadata: { pydantic_custom_use: '__python_function_call__', func, args, kwargs },
        })
        if (content.type !== 'text') {
          throw new Error('Expected return content type to be "text"')
        } else {
          return content.text
        }
      }

      const result = await runCode([mainPy], codeLog, functions, clientCallback)
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
function runSse(port: number, callbacks?: string) {
  const mcpServer = createServer(callbacks)
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
async function runStdio(callbacks?: string) {
  const mcpServer = createServer(callbacks)
  const transport = new StdioServerTransport()
  await mcpServer.connect(transport)
}

/*
 * Run pyodide to download packages which can otherwise interrupt the server
 */
async function warmup(callbacks?: string) {
  if (callbacks) {
    const functions = _extractFunctions(callbacks)
    console.error(`Functions extracted from callbacks: ${JSON.stringify(functions)}`)
  }
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

function _extractFunctions(callbacks?: string): string[] {
  return callbacks ? [...callbacks.matchAll(/^async def (\w+)/g).map(([, f]) => f)] : []
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
