/// <reference types="npm:@types/node@22.12.0" />

import './polyfill.ts'
import http, { type IncomingMessage, type ServerResponse } from 'node:http'
import { parseArgs } from '@std/cli/parse-args'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { type LoggingLevel, SetLevelRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'

import { asXml, runCode, runCodeWithToolInjection, type ToolInjectionConfig } from './runCode.ts'

const VERSION = '0.0.14'

export async function main() {
  const args = globalThis.Deno?.args || []
  if (args.length === 1 && args[0] === 'stdio') {
    await runStdio()
  } else if (args.length >= 1 && args[0] === 'sse') {
    const flags = parseArgs(args, {
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
    globalThis.Deno?.exit(1)
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(): McpServer {
  const server = new McpServer(
    {
      name: 'MCP Run Python with Tool Injection',
      version: VERSION,
    },
    {
      instructions: 'Call the "run_python_code" tool with the Python code to run.',
      capabilities: {
        logging: {},
        elicitation: {},
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

TOOL INJECTION: When 'tools' parameter is provided, the specified tool functions become available directly in Python's global namespace. You can call them directly like any other function. For example, if 'web_search' is provided as a tool, you can call it directly:

result = web_search("search query")
print(result)

The tools are injected into the global namespace automatically - no discovery functions needed.
`

  let setLogLevel: LoggingLevel = 'emergency'

  server.server.setRequestHandler(SetLevelRequestSchema, (request) => {
    setLogLevel = request.params.level
    return {}
  })

  server.tool(
    'run_python_code',
    toolDescription,
    {
      python_code: z.string().describe('Python code to run'),
      tools: z
        .array(z.string())
        .optional()
        .describe('List of available tools for injection (enables tool injection when provided)'),
    },
    async ({
      python_code,
      tools = [],
    }: {
      python_code: string
      tools?: string[]
    }) => {
      const logPromises: Promise<void>[] = []

      // Check if tools are provided
      if (tools.length > 0) {
        // Create elicitation callback
        // deno-lint-ignore no-explicit-any
        const elicitationCallback = async (elicitationRequest: any) => {
          // Convert Python dict to JavaScript object if needed
          let jsRequest
          if (elicitationRequest && typeof elicitationRequest === 'object' && elicitationRequest.toJs) {
            jsRequest = elicitationRequest.toJs()
          } else if (elicitationRequest && typeof elicitationRequest === 'object') {
            // Handle Python dict-like objects
            jsRequest = {
              message: elicitationRequest.message || elicitationRequest.get?.('message'),
              requestedSchema: elicitationRequest.requestedSchema || elicitationRequest.get?.('requestedSchema'),
            }
          } else {
            jsRequest = elicitationRequest
          }

          try {
            const elicitationResult = await server.server.request(
              {
                method: 'elicitation/create',
                params: {
                  message: jsRequest.message,
                  requestedSchema: jsRequest.requestedSchema,
                },
              },
              z.object({
                action: z.enum(['accept', 'decline', 'cancel']),
                content: z.optional(z.record(z.string(), z.unknown())),
              }),
            )

            return elicitationResult
          } catch (error) {
            logPromises.push(
              server.server.sendLoggingMessage({
                level: 'error',
                data: `Elicitation error: ${error}`,
              }),
            )
            throw error
          }
        }

        // Use tool injection mode
        const result = await runCodeWithToolInjection(
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
          {
            enableToolInjection: true,
            availableTools: tools,
            timeoutSeconds: 30,
            elicitationCallback,
          } as ToolInjectionConfig,
        )

        await Promise.all(logPromises)

        return {
          content: [{ type: 'text', text: asXml(result) }],
        }
      } else {
        // Use basic mode without tool injection
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
        )
        await Promise.all(logPromises)
        return {
          content: [{ type: 'text', text: asXml(result) }],
        }
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

  const server = http.createServer(async (req: IncomingMessage, res: ServerResponse) => {
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
  const result = await runCode(
    [
      {
        name: 'warmup.py',
        content: code,
        active: true,
      },
    ],
    (level, data) => console.error(`${level}: ${data}`),
  )
  console.log(asXml(result))

  // Test tool injection functionality
  console.error('Testing tool injection framework...')
  const toolCode = `
# Test tool injection - directly call an injected tool
result = web_search("test query")
print(f"Tool result: {result}")
"tool_test_complete"
`

  try {
    const toolResult = await runCodeWithToolInjection(
      [
        {
          name: 'tool_test.py',
          content: toolCode,
          active: true,
        },
      ],
      (level, data) => console.error(`${level}: ${data}`),
      {
        enableToolInjection: true,
        availableTools: ['web_search', 'send_email'],
        timeoutSeconds: 30,
        // deno-lint-ignore no-explicit-any require-await
        elicitationCallback: async (_elicitationRequest: any) => {
          // Mock callback for warmup test
          return {
            action: 'accept',
            content: {
              result: '{"status": "mock success"}',
            },
          }
        },
      } as ToolInjectionConfig,
    )
    console.log('Tool injection result:')
    console.log(asXml(toolResult))
  } catch (error) {
    console.error('Tool injection test failed:', error)
  }

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
