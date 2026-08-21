const LOADER_KEY = '__WAV2MID_MUSCRIPTOR_MODEL_LOADER__';
const RELOAD_MARKER_KEY = '__wav2mid_muscriptor_stale_chunk_reload__';
const RELOAD_QUERY_KEY = '__wav2mid_deploy';
const CORE_CHUNK_RE = /\/assets\/muscriptor-browser-core-[A-Za-z0-9_-]+\.js(?:[?#]|$)/;

installMuScriptorDeployRecovery();

export function findStaleMuScriptorChunkUrl(error) {
  let current = error;
  for (let depth = 0; current && depth < 8; depth += 1) {
    const text = String(current?.message || current || '');
    if (/failed to fetch dynamically imported module/i.test(text)) {
      const match = text.match(/https?:\/\/[^\s)]+/i);
      if (match) {
        try {
          const url = new URL(match[0]);
          if (url.origin === globalThis.location?.origin && CORE_CHUNK_RE.test(url.pathname)) {
            return url.href;
          }
        } catch {
          // Keep walking the cause chain.
        }
      }
    }
    current = current?.cause;
  }
  return null;
}

function installMuScriptorDeployRecovery() {
  cleanupReloadQuery();
  const loader = globalThis[LOADER_KEY];
  if (typeof loader !== 'function' || loader.__wav2midDeployRecovery) return;

  const wrapped = async (...args) => {
    try {
      const model = await loader(...args);
      clearReloadMarker();
      return model;
    } catch (error) {
      const staleChunkUrl = findStaleMuScriptorChunkUrl(error);
      if (!staleChunkUrl) throw error;

      if (readReloadMarker()) {
        clearReloadMarker();
        const recoveryError = new Error(
          `[MUSCRIPTOR_APP_CHUNK_FETCH] model-loader: 現行deployへ自動再読み込み後もMuScriptor chunkを取得できません: ${staleChunkUrl}`,
        );
        recoveryError.code = 'MUSCRIPTOR_APP_CHUNK_FETCH';
        recoveryError.cause = error;
        surfaceReloadMessage('MuScriptorアプリ本体の取得に失敗しました。通信状態を確認して再読み込みしてください。');
        throw recoveryError;
      }

      writeReloadMarker(staleChunkUrl);
      surfaceReloadMessage('本番更新を検出しました。最新バージョンへ自動再読み込みします…');
      reloadCurrentDeployment();
      return new Promise(() => {});
    }
  };

  Object.defineProperty(wrapped, '__wav2midDeployRecovery', { value: true });
  globalThis[LOADER_KEY] = wrapped;
}

function reloadCurrentDeployment() {
  const next = new URL(globalThis.location.href);
  next.searchParams.set(RELOAD_QUERY_KEY, Date.now().toString(36));
  globalThis.location.replace(next.href);
}

function cleanupReloadQuery() {
  try {
    const current = new URL(globalThis.location?.href || '');
    if (!current.searchParams.has(RELOAD_QUERY_KEY)) return;
    current.searchParams.delete(RELOAD_QUERY_KEY);
    globalThis.history?.replaceState?.(globalThis.history.state, '', current.href);
  } catch {
    // Non-browser test environments do not need URL cleanup.
  }
}

function readReloadMarker() {
  try {
    return globalThis.sessionStorage?.getItem(RELOAD_MARKER_KEY) || '';
  } catch {
    return '';
  }
}

function writeReloadMarker(value) {
  try {
    globalThis.sessionStorage?.setItem(RELOAD_MARKER_KEY, String(value));
  } catch {
    // Reload still works without storage; worst case the browser shows the original error.
  }
}

function clearReloadMarker() {
  try {
    globalThis.sessionStorage?.removeItem(RELOAD_MARKER_KEY);
  } catch {
    // no-op
  }
}

function surfaceReloadMessage(message) {
  const panel = document.getElementById('progressPanel');
  const text = document.getElementById('progressText');
  const hint = document.getElementById('progressHint');
  const state = document.getElementById('ultraState');
  if (panel) panel.hidden = false;
  if (text) text.textContent = message;
  if (hint) hint.textContent = 'デプロイ更新で旧Vite chunkが消えた場合にだけ発生するため、モデルやORTの故障ではありません。';
  if (state) state.textContent = 'reload';
}
