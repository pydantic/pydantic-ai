import express, { Request, Response } from 'express'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'

import { runCode, asXml } from './runCode.js'

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

  server.tool(
    'run_python_code',
    toolDescription,
    { python_code: z.string().describe('Python code to run') },
    async ({ python_code }: { python_code: string }) => {
      const result = await runCode([{ name: 'main.py', content: python_code, active: true }])
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
function runSse(mcpServer: McpServer) {
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
async function runStdio(mcpServer: McpServer) {
  const transport = new StdioServerTransport()
  // using console.error to print to stderr to avoid conflicts with the stdio transport
  console.error(`Running MCP server with Stdio transport`)
  await mcpServer.connect(transport)
}

const args = process.argv.slice(2)
const mcpServer = createServer()
if (args.length === 1 && args[0] === 'stdio') {
  await runStdio(mcpServer)
} else if (args.length === 1 && args[0] === 'sse') {
  runSse(mcpServer)
} else {
  console.error('Usage: node cli.js [stdio|sse]')
  process.exit(1)
}
