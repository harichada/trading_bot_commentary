import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/gf/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gf/, ''),
      },
      '/gf-api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gf-api/, '/api'),
      },
      '/id-api': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/id-api/, '/api'),
      },
      '/gf-ws': {
        target: 'ws://localhost:8003',
        ws: true,
        rewrite: (path) => path.replace(/^\/gf-ws/, '/ws'),
      },
      '/id-ws': {
        target: 'ws://localhost:8004',
        ws: true,
        rewrite: (path) => path.replace(/^\/id-ws/, '/ws'),
      },
    },
  },
})
