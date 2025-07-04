# Use official Deno image as base
FROM denoland/deno:alpine-2.3.7

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Expose port for SSE mode (optional, only needed for SSE)
EXPOSE 3101

# Default command: run MCP Run Python server in SSE mode with health check endpoint
CMD ["run", "-N", "-R=node_modules", "-W=node_modules", "--node-modules-dir=auto", "jsr:@pydantic/mcp-run-python", "sse", "--port=3101"]
