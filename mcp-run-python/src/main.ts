import http from "node:http";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  type LoggingLevel,
  SetLevelRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import { asXml, runCode } from "./runCode.ts";

export async function main() {
  const { args } = Deno;
  if (args.length === 1 && args[0] === "stdio") {
    await runStdio();
  } else if (args.length === 1 && args[0] === "sse") {
    runSse();
  } else if (args.length === 1 && args[0] === "warmup") {
    await warmup();
  } else {
    console.error("Usage: npx @pydantic/mcp-run-python [stdio|sse|warmup]");
    Deno.exit(1);
  }
}

/*
 * Create an MCP server with the `run_python_code` tool registered.
 */
function createServer(): McpServer {
  const server = new McpServer(
    {
      name: "MCP Run Python",
      version: "0.0.1",
    },
    {
      instructions:
        'Call the "run_python_code" tool with the Python code to run.',
      capabilities: {
        logging: {},
      },
    },
  );

  const toolDescription =
    `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return value.

The code will be executed with Python 3.12.

Dependencies may be defined via PEP 723 script metadata, e.g. to install "pydantic", the script should start
with a comment of the form:

# /// script
# dependencies = ['pydantic']
# ///
`;

  let setLogLevel: LoggingLevel = "emergency";

  server.server.setRequestHandler(SetLevelRequestSchema, (request) => {
    setLogLevel = request.params.level;
    return {};
  });

  server.tool(
    "run_python_code",
    toolDescription,
    { python_code: z.string().describe("Python code to run") },
    async ({ python_code }: { python_code: string }) => {
      const logPromises: Promise<void>[] = [];
      const result = await runCode([{
        name: "main.py",
        content: python_code,
        active: true,
      }], (level, data) => {
        if (LogLevels.indexOf(level) >= LogLevels.indexOf(setLogLevel)) {
          logPromises.push(server.server.sendLoggingMessage({ level, data }));
        }
      });
      await Promise.all(logPromises);
      return {
        content: [{ type: "text", text: asXml(result) }],
      };
    },
  );
  return server;
}

/*
 * Run the MCP server using the SSE transport, e.g. over HTTP.
 */
function runSse() {
  const mcpServer = createServer();
  const transports: { [sessionId: string]: SSEServerTransport } = {};

  const server = http.createServer(async (req, res) => {
    const url = new URL(
      req.url ?? "",
      `http://${req.headers.host ?? "unknown"}`,
    );
    console.log({ url, method: req.method });
    let pathMatch = false;
    function match(methods: string[], path: string): boolean {
      if (url.pathname === path) {
        pathMatch = true;
        return methods.includes(req.method!);
      }
      return false;
    }

    if (match(["GET", "POST"], "/sse")) {
      const transport = new SSEServerTransport("/sse", res);
      transports[transport.sessionId] = transport;
      res.on("close", () => {
        delete transports[transport.sessionId];
      });
      await mcpServer.connect(transport);
    } else if (match(["POST"], "/messages")) {
      const sessionId = url.searchParams.get("sessionId") ?? "";
      const transport = transports[sessionId];
      if (transport) {
        await transport.handlePostMessage(req, res);
      } else {
        res.setHeader("Content-Type", "text/plain");
        res.statusCode = 400;
        res.end(`No transport found for sessionId '${sessionId}'`);
      }
    } else {
      res.setHeader("Content-Type", "text/plain");
      res.statusCode = pathMatch ? 405 : 404;
      res.end(pathMatch ? "Method not allowed\n" : "Page not found\n");
    }
  });

  // const port = Deno.env.PORT ? parseInt(Deno.env.PORT) : 3001;
  const port = 3001;
  server.listen(port, () => {
    console.log(`Running MCP server with SSE transport on port ${port}`);
  });
}

/*
 * Run the MCP server using the Stdio transport.
 */
async function runStdio() {
  const mcpServer = createServer();
  const transport = new StdioServerTransport();
  await mcpServer.connect(transport);
}

/*
 * Run pyodide to download packages which can otherwise interrupt the server
 */
async function warmup() {
  console.error("Running warmup script...");
  const code = `
import numpy
a = numpy.array([1, 2, 3])
print('numpy array:', a)
a
`;
  const result = await runCode([{
    name: "warmup.py",
    content: code,
    active: true,
  }], (level, data) =>
    // use warn to avoid recursion since console.log is patched in runCode
    console.error(`${level}: ${data}`));
  console.log("Tool return value:");
  console.log(asXml(result));
  console.log("\nwarmup successful 🎉");
}

// list of log levels to use for level comparison
const LogLevels: LoggingLevel[] = [
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
  "alert",
  "emergency",
];

await main();
