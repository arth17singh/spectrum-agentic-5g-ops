import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

const API = process.env.SPECTRUM_API ?? 'http://127.0.0.1:8787'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: {
    // this laptop only: proxied calls reach the API from 127.0.0.1, which needs no access token
    // (phones use `make demo`)
    host: 'localhost',
    port: 5173,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/ws': { target: API.replace('http', 'ws'), ws: true },
    },
  },
})
