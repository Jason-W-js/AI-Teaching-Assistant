import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (/\/(react|react-dom|react-router|scheduler|zustand)\//.test(id)) {
            return 'react-vendor'
          }
          if (/\/(react-markdown|remark-|rehype-|unified|katex|micromark|mdast|hast)/.test(id)) {
            return 'content-vendor'
          }
          if (/\/(antd|@ant-design|rc-|@rc-component)\//.test(id)) {
            return 'ui-vendor'
          }
          if (id.includes('/lucide-react/')) return 'icons-vendor'
          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
