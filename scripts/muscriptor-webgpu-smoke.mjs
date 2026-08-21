#!/usr/bin/env node
import { chromium } from '@playwright/test';

const baseURL = process.env.MUSCRIPTOR_SMOKE_URL || 'http://127.0.0.1:4173';

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,VulkanFromANGLE',
    '--use-angle=swiftshader',
    '--disable-gpu-sandbox',
  ],
});

try {
  const page = await browser.newPage();
  page.on('console', msg => console.log(`[browser:${msg.type()}] ${msg.text()}`));
  page.on('pageerror', error => console.error('[browser:error]', error));
  page.on('crash', () => console.error('[browser:crash] renderer crashed'));
  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });

  const result = await page.evaluate(async () => {
    // GitHub's hosted Linux runner currently exposes Chrome with navigator.gpu
    // but may not provide any usable adapter (SwiftShader/Vulkan initialization
    // can fail before our model is involved). Treat only that infrastructure
    // condition as a skip. Once an adapter exists, every model/runtime mismatch
    // below remains a hard failure.
    if (!navigator.gpu) {
      return { status: 'skipped', reason: 'navigator-gpu-unavailable' };
    }
    let adapter;
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch (error) {
      return {
        status: 'skipped',
        reason: 'webgpu-adapter-request-failed',
        detail: String(error?.message || error),
      };
    }
    if (!adapter) {
      return { status: 'skipped', reason: 'webgpu-adapter-unavailable' };
    }

    const ort = await import('/src/ort-cdn-shim.js');
    const response = await fetch('/models/muscriptor-small/manifest.json');
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    const manifest = await response.json();
    const arch = manifest.architecture;
    const smoke = manifest.browserSmoke;
    if (!smoke) throw new Error('manifest is missing browserSmoke fixture metadata');

    const conditionerDtype = arch.conditionerActivationType || arch.activationType || 'float16';
    const decoderDtype = arch.decoderActivationType || arch.activationType || 'float16';
    if (!['float16', 'float32'].includes(conditionerDtype)) {
      throw new Error(`unsupported conditioner dtype ${conditionerDtype}`);
    }
    if (!['float16', 'float32'].includes(decoderDtype)) {
      throw new Error(`unsupported decoder dtype ${decoderDtype}`);
    }
    if (conditionerDtype !== decoderDtype) {
      throw new Error(`conditioner/decoder dtype mismatch ${conditionerDtype} -> ${decoderDtype}`);
    }

    const adapterInfo = adapter.info ? {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
    } : null;
    const device = await adapter.requestDevice();
    const ep = { name: 'webgpu', device, storageBufferCacheMode: 'simple' };

    const conditioner = await ort.InferenceSession.create(manifest.files.conditioner.url, {
      executionProviders: [ep],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: { prefix_embeddings: 'gpu-buffer' },
    });
    const preferred = { logits: 'cpu' };
    for (let i = 0; i < arch.layers; i += 1) {
      preferred[`new_k_${i}`] = 'gpu-buffer';
      preferred[`new_v_${i}`] = 'gpu-buffer';
    }
    const decoder = await ort.InferenceSession.create(manifest.files.decoder.url, {
      executionProviders: [ep],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: preferred,
    });

    const logMelShape = smoke.logMelShape || [1, 501, arch.melBins];
    const logMelElements = logMelShape.reduce((a, b) => a * b, 1);
    const logMel = new ort.Tensor('float32', new Float32Array(logMelElements), logMelShape);
    const instrumentIds = new ort.Tensor(
      'int64',
      BigInt64Array.from(smoke.instrumentEmbedIds || [1], BigInt),
      [1, (smoke.instrumentEmbedIds || [1]).length],
    );

    let prefix;
    let outputs;
    const cacheBuffers = [];
    const cacheTensors = [];
    try {
      const conditioned = await conditioner.run({
        log_mel: logMel,
        instrument_embed_ids: instrumentIds,
      });
      prefix = conditioned.prefix_embeddings;
      const prefixLen = smoke.prefixTokens || manifest.parity?.pytorch?.prefix_tokens || 503;
      if (prefix?.dims?.[1] !== prefixLen || prefix?.dims?.[2] !== arch.dim) {
        throw new Error(`conditioner prefix shape ${prefix?.dims} != [1,${prefixLen},${arch.dim}]`);
      }
      if (prefix.type !== decoderDtype) {
        throw new Error(`conditioner prefix dtype ${prefix.type} != decoder ${decoderDtype}`);
      }

      const dim = arch.dim;
      const heads = arch.heads;
      const headDim = dim / heads;
      const cacheRows = arch.maxCache;
      const bytesPerElement = decoderDtype === 'float16' ? 2 : 4;
      const cacheBytes = cacheRows * heads * headDim * bytesPerElement;
      const tokenIds = new ort.Tensor('int64', BigInt64Array.of(BigInt(arch.card)), [1, 1]);
      const pastLen = new ort.Tensor('int64', BigInt64Array.of(0n), []);
      const feeds = {
        prefix_embeddings: prefix,
        token_ids: tokenIds,
        past_len: pastLen,
      };

      for (let i = 0; i < arch.layers; i += 1) {
        for (const kind of ['k', 'v']) {
          const buffer = device.createBuffer({
            size: cacheBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const tensor = ort.Tensor.fromGpuBuffer(buffer, {
            dataType: decoderDtype,
            dims: [1, cacheRows, heads, headDim],
          });
          cacheBuffers.push(buffer);
          cacheTensors.push(tensor);
          feeds[`cache_${kind}_${i}`] = tensor;
        }
      }

      outputs = await decoder.run(feeds);
      const logits = outputs.logits.data;
      let argmax = 0;
      let best = -Infinity;
      let finite = true;
      for (let i = 0; i < logits.length; i += 1) {
        const value = Number(logits[i]);
        if (!Number.isFinite(value)) finite = false;
        if (value > best) { best = value; argmax = i; }
      }
      const expectedQuery = prefixLen + 1;
      const firstK = outputs.new_k_0;
      if (!finite) throw new Error('WebGPU decoder produced non-finite logits');
      if (logits.length !== arch.card) throw new Error(`logits length ${logits.length} != ${arch.card}`);
      if (firstK?.dims?.[1] !== expectedQuery) {
        throw new Error(`new_k_0 query length ${firstK?.dims?.[1]} != ${expectedQuery}`);
      }
      if (argmax !== smoke.expectedFirstToken) {
        throw new Error(`WebGPU first token ${argmax} != CPU ORT ${smoke.expectedFirstToken}`);
      }

      return {
        status: 'ok',
        conditioner: manifest.files.conditioner.name,
        decoder: manifest.files.decoder.name,
        conditionerBytes: manifest.files.conditioner.bytes,
        decoderBytes: manifest.files.decoder.bytes,
        dtype: decoderDtype,
        firstToken: argmax,
        logits: logits.length,
        prefixTokens: prefixLen,
        queryTokens: expectedQuery,
        layers: arch.layers,
        cacheMiB: Math.round((cacheBytes * arch.layers * 2) / 1048576 * 10) / 10,
        adapterInfo,
      };
    } finally {
      if (outputs) {
        outputs.logits?.dispose?.();
        for (let i = 0; i < arch.layers; i += 1) {
          outputs[`new_k_${i}`]?.dispose?.();
          outputs[`new_v_${i}`]?.dispose?.();
        }
      }
      prefix?.dispose?.();
      logMel.dispose?.();
      instrumentIds.dispose?.();
      for (const tensor of cacheTensors) tensor.dispose?.();
      for (const buffer of cacheBuffers) buffer.destroy();
      await device.queue.onSubmittedWorkDone();
      device.destroy?.();
    }
  });

  console.log(JSON.stringify(result, null, 2));
  if (result.status === 'skipped') {
    console.warn(`MuScriptor WebGPU model smoke skipped: ${result.reason}`);
  }
} finally {
  await browser.close();
}
