import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'X-Content-Type-Options': 'nosniff',
};
const ortCdnShim = fileURLToPath(new URL('./src/ort-cdn-shim.js', import.meta.url));
const demucsLowMemory = fileURLToPath(new URL('./src/demucs-low-memory.js', import.meta.url));

export default defineConfig({
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  resolve: {
    alias: [
      { find: /^onnxruntime-web\/webgpu$/, replacement: ortCdnShim },
      { find: /^onnxruntime-web$/, replacement: ortCdnShim },
      { find: /^demucs-web$/, replacement: demucsLowMemory },
    ],
  },
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 2300,
  },
});