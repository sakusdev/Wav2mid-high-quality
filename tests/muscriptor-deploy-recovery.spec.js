import { expect, test } from '@playwright/test';

import { findStaleMuScriptorChunkUrl } from '../src/muscriptor-deploy-recovery.js';

test.describe('MuScriptor deploy chunk recovery', () => {
  test('recognizes a same-origin stale Vite MuScriptor core chunk through diagnostic causes', () => {
    const previousLocation = globalThis.location;
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: new URL('https://wav2mid.example/app'),
    });
    try {
      const root = new TypeError(
        'Failed to fetch dynamically imported module: https://wav2mid.example/assets/muscriptor-browser-core-DKuX3vSb.js',
      );
      const wrapped = new Error('[MUSCRIPTOR_ORT_RUNTIME_FETCH] model-loader: wrapped');
      wrapped.cause = root;
      expect(findStaleMuScriptorChunkUrl(wrapped)).toBe(
        'https://wav2mid.example/assets/muscriptor-browser-core-DKuX3vSb.js',
      );
    } finally {
      if (previousLocation === undefined) delete globalThis.location;
      else Object.defineProperty(globalThis, 'location', { configurable: true, value: previousLocation });
    }
  });

  test('does not confuse ORT JSEP or cross-origin chunks with an app deploy rollover', () => {
    const previousLocation = globalThis.location;
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: new URL('https://wav2mid.example/'),
    });
    try {
      expect(findStaleMuScriptorChunkUrl(new Error(
        'Failed to fetch dynamically imported module: https://wav2mid.example/ort-wasm/ort-wasm-simd-threaded.jsep.mjs',
      ))).toBeNull();
      expect(findStaleMuScriptorChunkUrl(new Error(
        'Failed to fetch dynamically imported module: https://cdn.example/assets/muscriptor-browser-core-old.js',
      ))).toBeNull();
    } finally {
      if (previousLocation === undefined) delete globalThis.location;
      else Object.defineProperty(globalThis, 'location', { configurable: true, value: previousLocation });
    }
  });
});
