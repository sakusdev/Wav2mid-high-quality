const MODEL_PREFIX = '/models/muscriptor-small/';
const STREAM_MAP_PATH = `${MODEL_PREFIX}stream-map.json`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!env.ASSETS) return new Response('Static asset binding unavailable', { status: 500 });

    const isModelRequest = url.pathname.startsWith(MODEL_PREFIX);
    if (request.method === 'GET' || request.method === 'HEAD') {
      const streamed = await lookupStreamedAsset(request, env, url.pathname);
      if (streamed) return streamed;
    }

    if (isModelRequest) {
      const response = await env.ASSETS.fetch(request);
      if (isSpaFallback(response)) return modelUnavailable(url.pathname, request.method);
      return response;
    }

    return env.ASSETS.fetch(request);
  },
};

async function lookupStreamedAsset(request, env, pathname) {
  let mapResponse;
  try {
    mapResponse = await assetFetch(env, request.url, STREAM_MAP_PATH);
  } catch {
    return null;
  }
  if (!mapResponse.ok || isSpaFallback(mapResponse)) return null;

  let map;
  try {
    map = await mapResponse.json();
  } catch {
    return null;
  }
  const partManifestPath = map?.models?.[pathname];
  if (!partManifestPath) return null;

  const manifestResponse = await assetFetch(env, request.url, partManifestPath);
  if (!manifestResponse.ok || isSpaFallback(manifestResponse)) {
    return new Response('Model part manifest unavailable', { status: 502 });
  }
  const manifest = await manifestResponse.json();
  const parts = Array.isArray(manifest?.parts) ? manifest.parts : [];
  if (!parts.length || !Number.isFinite(Number(manifest.bytes))) {
    return new Response('Invalid model part manifest', { status: 502 });
  }

  const headers = new Headers({
    'Content-Type': manifest.contentType || 'application/octet-stream',
    'Content-Length': String(manifest.bytes),
    'Cache-Control': 'public, max-age=31536000, immutable',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'X-Content-Type-Options': 'nosniff',
    'Accept-Ranges': 'none',
  });
  if (manifest.sha256) headers.set('ETag', `"sha256-${manifest.sha256}"`);

  if (request.method === 'HEAD') return new Response(null, { status: 200, headers });

  const body = new ReadableStream({
    async start(controller) {
      try {
        for (const part of parts) {
          const response = await assetFetch(env, request.url, part.url);
          if (!response.ok || !response.body || isSpaFallback(response)) {
            throw new Error(`model part unavailable: ${part.url} (${response.status})`);
          }
          const reader = response.body.getReader();
          try {
            while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              controller.enqueue(value);
            }
          } finally {
            reader.releaseLock();
          }
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });

  return new Response(body, { status: 200, headers });
}

function isSpaFallback(response) {
  return String(response?.headers?.get('content-type') || '').toLowerCase().includes('text/html');
}

function modelUnavailable(pathname, method) {
  const body = JSON.stringify({
    error: 'MUSCRIPTOR_MODEL_NOT_DEPLOYED',
    path: pathname,
    message: 'MuScriptor model assets are missing from this deployment.',
  });
  const headers = new Headers({
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  return new Response(method === 'HEAD' ? null : body, { status: 503, headers });
}

function assetFetch(env, requestUrl, pathname) {
  const url = new URL(pathname, requestUrl);
  return env.ASSETS.fetch(new Request(url, { method: 'GET' }));
}
