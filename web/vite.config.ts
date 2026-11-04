import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [tailwindcss(), react(), viteSingleFile()],
  server: {
    port: 3000,
    proxy: {
      '/v1': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/profile': 'http://127.0.0.1:8000',
      '/experts': 'http://127.0.0.1:8000',
    }
  }
});
