FROM denoland/deno:1.44.0

WORKDIR /app

COPY . .

RUN deno task build

RUN deno cache src/main.ts

EXPOSE 3001

CMD [
  "deno", "run",
  "-N",
  "-R=node_modules",
  "-W=node_modules",
  "--node-modules-dir=auto",
  "src/main.ts",
  "sse",
  "--port=3001"
]
