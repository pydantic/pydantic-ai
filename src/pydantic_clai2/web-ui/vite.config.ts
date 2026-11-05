import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `npm run dev` proxies to a running `/web` server: set CLAI_WEB to its address, such as
// http://127.0.0.1:8765, and open the dev server with the same `?token=` the plugin printed.
const target = process.env.CLAI_WEB ?? 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../pydantic_clai2/builtin_plugins/web/static',
    emptyOutDir: true,
    // Stable names keep the committed bundle's diffs readable.
    rollupOptions: {
      output: {
        entryFileNames: 'assets/app.js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
    chunkSizeWarningLimit: 800,
  },
  server: {
    proxy: {
      '/api': { target, changeOrigin: true },
      '/ws': { target, ws: true, changeOrigin: true, headers: { origin: target } },
    },
  },
})
