import express, { Request, Response } from 'express'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { SetLevelRequestSchema, LoggingLevel } from '@modelcontextprotocol/sdk/types.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'

import { runCode, asXml } from './runCode.js'

export async function main() {
  const args = process.argv.slice(2)
  if (args.length === 1 && args[0] === 'stdio') {
    await runStdio()
  } else if (args.length === 1 && args[0] === 'sse') {
    runSse()
  } else if (args.length === 1 && args[0] === 'warmup') {
    await warmup()
  } else {
    console.error('Usage: npx @pydantic/mcp-run-python [stdio|sse|warmup]')
    process.exit(1)
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(): McpServer {
  const server = new McpServer(
    {
      name: 'MCP Run Python',
      version: '0.0.1',
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
`

  let setLogLevel: LoggingLevel = 'emergency'

  server.server.setRequestHandler(SetLevelRequestSchema, async (request) => {
    setLogLevel = request.params.level
    return {}
  })

  server.tool(
    'run_python_code',
    toolDescription,
    { python_code: z.string().describe('Python code to run') },
    async ({ python_code }: { python_code: string }) => {
      const logPromises: Promise<void>[] = []
      const result = await runCode([{ name: 'main.py', content: python_code, active: true }], (level, data) => {
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
function runSse() {
  const mcpServer = createServer()
  const app = express()
  const transports: { [sessionId: string]: SSEServerTransport } = {}

  app.get('/sse', async (_: Request, res: Response) => {
    const transport = new SSEServerTransport('/messages', res)
    transports[transport.sessionId] = transport
    res.on('close', () => {
      delete transports[transport.sessionId]
    })
    await mcpServer.connect(transport)
  })

  app.post('/messages', async (req: Request, res: Response) => {
    const sessionId = req.query.sessionId as string
    const transport = transports[sessionId]
    if (transport) {
      await transport.handlePostMessage(req, res)
    } else {
      res.status(400).send(`No transport found for sessionId '${sessionId}'`)
    }
  })

  const port = process.env.PORT ? parseInt(process.env.PORT) : 3001
  const host = process.env.HOST || 'localhost'
  console.log(`Running MCP server with SSE transport on ${host}:${port}`)
  app.listen(port, host)
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
  console.error('Running warmup script...')
  const code = `
import numpy
a = numpy.array([1, 2, 3])
print('numpy array:', a)
a
`
  const result = await runCode([{ name: 'warmup.py', content: code, active: true }], (level, data) =>
    // use warn to avoid recursion since console.log is patched in runCode
    console.error(`${level}: ${data}`),
  )
  console.log('Tool return value:')
  console.log(asXml(result))
  console.log('\nwarmup successful 🎉')
}

// list of log levels to use for level comparison
const LogLevels: LoggingLevel[] = ['debug', 'info', 'notice', 'warning', 'error', 'critical', 'alert', 'emergency']
