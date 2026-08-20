import './ultra.css';

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8223';
const CLIENT_ID_KEY = 'wav2mid.muscriptorClientId';
const ENDPOINT_KEY = 'wav2mid.muscriptorEndpoint';

let currentFile = null;
let ultraResult = null;

initWhenReady();

function initWhenReady() {
  const neuralOption = document.getElementById('neuralOption');
  const analyzeBtn = document.getElementById('analyzeBtn');
  if (!neuralOption || !analyzeBtn) {
    requestAnimationFrame(initWhenReady);
    return;
  }

  const block = document.createElement('div');
  block.className = 'ultra-block';
  block.innerHTML = `
    <label class="ultra-option" id="ultraOption">
      <input id="ultraToggle" type="checkbox" />
      <span class="ultra-check" aria-hidden="true"></span>
      <span class="ultra-copy">
        <strong>MuScriptor ULTRA <span>NC</span></strong>
        <small>非商用 · localhost bridge · multi-instrument transformer</small>
      </span>
      <span class="ultra-state" id="ultraState">offline</span>
    </label>
    <div class="ultra-config" id="ultraConfig" hidden>
      <label>
        <span>Bridge</span>
        <input id="muscriptorEndpoint" type="url" inputmode="url" spellcheck="false" />
      </label>
      <button type="button" class="secondary ultra-probe" id="muscriptorProbe">CHECK</button>
      <p>PCで <code>python tools/muscriptor_bridge.py --model small</code> を実行。モデル重みはCC BY-NC 4.0で、音源はlocalhostから外へ送信しません。</p>
    </div>
  `;
  neuralOption.after(block);

  const ultraToggle = document.getElementById('ultraToggle');
  const ultraOption = document.getElementById('ultraOption');
  const ultraConfig = document.getElementById('ultraConfig');
  const endpointInput = document.getElementById('muscriptorEndpoint');
  const probeBtn = document.getElementById('muscriptorProbe');
  const state = document.getElementById('ultraState');
  const neuralToggle = document.getElementById('neuralToggle');
  const qualityGroup = document.getElementById('qualityGroup');
  const modeNote = document.getElementById('modeNote');
  const progressHint = document.getElementById('progressHint');
  const fileInput = document.getElementById('fileInput');
  const dropZone = document.getElementById('dropZone');

  endpointInput.value = localStorage.getItem(ENDPOINT_KEY) || DEFAULT_ENDPOINT;
  endpointInput.addEventListener('change', () => {
    endpointInput.value = normalizeEndpoint(endpointInput.value);
    localStorage.setItem(ENDPOINT_KEY, endpointInput.value);
    state.textContent = 'unchecked';
  });

  fileInput.addEventListener('change', () => {
    currentFile = fileInput.files?.[0] || null;
    clearUltraResult();
  });
  dropZone.addEventListener('drop', event => {
    currentFile = event.dataTransfer?.files?.[0] || null;
    clearUltraResult();
  });

  ultraToggle.addEventListener('change', () => {
    if (ultraToggle.checked && neuralToggle?.checked) {
      neuralToggle.checked = false;
      neuralToggle.dispatchEvent(new Event('change', { bubbles: true }));
    }
    updateUltraUi();
  });

  neuralToggle?.addEventListener('change', () => {
    if (neuralToggle.checked && ultraToggle.checked) {
      ultraToggle.checked = false;
      updateUltraUi();
    }
  });

  probeBtn.addEventListener('click', async () => {
    probeBtn.disabled = true;
    state.textContent = 'checking';
    try {
      const response = await localFetch(`${endpoint()}/health`, { method: 'GET' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const info = await response.json();
      state.textContent = info?.status === 'ok' ? 'ready' : 'online';
    } catch (error) {
      state.textContent = 'offline';
      console.warn('MuScriptor bridge check failed.', error);
    } finally {
      probeBtn.disabled = false;
    }
  });

  analyzeBtn.addEventListener('click', async event => {
    if (!ultraToggle.checked) {
      ultraResult = null;
      document.getElementById('results')?.removeAttribute('data-engine');
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();

    const file = currentFile || fileInput.files?.[0];
    if (!file) return;
    analyzeBtn.disabled = true;
    ultraResult = null;
    document.getElementById('results').hidden = true;
    showProgress(0.01, 'MuScriptor bridgeへ接続中…');

    const started = performance.now();
    try {
      const parsed = await transcribeWithBridge(file, progress => {
        if (progress.total > 0) {
          const ratio = progress.completed / progress.total;
          showProgress(0.06 + ratio * 0.86, `MuScriptor ${progress.completed} / ${progress.total} chunks`);
        }
      });
      ultraResult = {
        ...parsed,
        file,
        endpoint: endpoint(),
        elapsedSeconds: (performance.now() - started) / 1000,
      };
      renderUltraResult(ultraResult);
      state.textContent = 'ready';
      showProgress(1, `完了 · ${parsed.notes.length} notes`);
    } catch (error) {
      console.error(error);
      state.textContent = 'error';
      showProgress(0, `MuScriptor失敗: ${error?.message || error}`);
      progressHint.textContent = 'localhost bridge、Hugging Face認証、ブラウザのローカルネットワーク権限を確認してください。';
    } finally {
      analyzeBtn.disabled = false;
    }
  }, true);

  document.getElementById('midiBtn')?.addEventListener('click', event => {
    if (!ultraResult || document.getElementById('results')?.dataset.engine !== 'muscriptor') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    downloadBlob(new Blob([ultraResult.midiBytes], { type: 'audio/midi' }), `${basename(ultraResult.file.name)}.muscriptor.mid`);
  }, true);

  document.getElementById('jsonBtn')?.addEventListener('click', event => {
    if (!ultraResult || document.getElementById('results')?.dataset.engine !== 'muscriptor') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const payload = {
      format: 'wav2mid-muscriptor-bridge/v1',
      engine: 'MuScriptor',
      license: 'CC BY-NC 4.0 model weights',
      endpoint: ultraResult.endpoint,
      beatGrid: ultraResult.beatGrid,
      notes: ultraResult.notes,
      drums: ultraResult.drums,
    };
    downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }), `${basename(ultraResult.file.name)}.muscriptor.json`);
  }, true);

  function endpoint() {
    return normalizeEndpoint(endpointInput.value || DEFAULT_ENDPOINT);
  }

  function updateUltraUi() {
    const active = ultraToggle.checked;
    ultraOption.classList.toggle('active', active);
    ultraConfig.hidden = !active;
    qualityGroup?.classList.toggle('disabled', active || Boolean(neuralToggle?.checked));
    if (active) {
      modeNote.textContent = 'MuScriptor transformer · multi-instrument · 非商用。推論はlocalhost bridgeで実行。';
      progressHint.textContent = 'MuScriptor ULTRAはPC上のlocalhost bridgeへ音源を渡します。外部サーバーへは送信しません。';
    }
  }
}

async function transcribeWithBridge(file, onProgress) {
  const starts = new Map();
  const rawNotes = [];
  const form = new FormData();
  form.append('file', file, file.name);
  form.append('detect_tempo', 'best-effort');

  const response = await localFetch(`${normalizeEndpoint(localStorage.getItem(ENDPOINT_KEY) || DEFAULT_ENDPOINT)}/transcribe`, {
    method: 'POST',
    body: form,
    headers: { 'X-Client-ID': clientId() },
  });
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json())?.detail || ''; } catch {}
    throw new Error(detail || `bridge HTTP ${response.status}`);
  }
  if (!response.body) throw new Error('bridge response has no readable stream');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let midiBytes = null;
  let beatGrid = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, '\n');
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.type === 'start') {
          starts.set(Number(payload.index), payload);
        } else if (payload.type === 'end') {
          const start = starts.get(Number(payload.start_event_index));
          if (!start) continue;
          rawNotes.push({
            pitchMidi: Number(start.pitch),
            startTimeSeconds: Number(start.start_time),
            durationSeconds: Math.max(0.001, Number(payload.end_time) - Number(start.start_time)),
            instrument: String(start.instrument || 'unknown'),
          });
          starts.delete(Number(payload.start_event_index));
        } else if (payload.type === 'progress') {
          onProgress?.({ completed: Number(payload.completed || 0), total: Number(payload.total || 0) });
        } else if (payload.type === 'transcription_complete') {
          midiBytes = base64ToBytes(payload.data);
          beatGrid = payload.beat_grid || null;
        }
      }
    }
    if (done) break;
  }

  if (!midiBytes) throw new Error('bridge finished without MIDI data');
  const onsetDelay = Number(beatGrid?.onset_delay || 0);
  const normalized = rawNotes.map(note => ({
    ...note,
    startTimeSeconds: Math.max(0, note.startTimeSeconds - onsetDelay),
  }));
  const drums = normalized.filter(note => isDrum(note.instrument)).map(note => ({
    midi: note.pitchMidi,
    name: note.instrument,
    time: note.startTimeSeconds,
    duration: note.durationSeconds,
    velocity: 0.8,
  }));
  const notes = normalized.filter(note => !isDrum(note.instrument));
  return { notes, drums, rawNotes: normalized, midiBytes, beatGrid };
}

function renderUltraResult(data) {
  const results = document.getElementById('results');
  results.hidden = false;
  results.dataset.engine = 'muscriptor';
  document.getElementById('resultTitle').textContent = data.file.name;
  document.getElementById('statNotes').textContent = data.notes.length.toLocaleString();
  document.getElementById('statDrums').textContent = data.drums.length.toLocaleString();
  document.getElementById('statTempo').textContent = data.beatGrid?.bpm ? `${Math.round(data.beatGrid.bpm)} BPM` : '—';
  document.getElementById('statKey').textContent = '—';
  document.getElementById('statPoly').textContent = maxPolyphony(data.notes);
  document.getElementById('statEnsemble').textContent = 'NC';
  const pitches = data.notes.map(note => note.pitchMidi);
  document.getElementById('statRange').textContent = pitches.length ? `${midiName(Math.min(...pitches))} – ${midiName(Math.max(...pitches))}` : '—';
  document.getElementById('rollLegend').textContent = `MuScriptor ULTRA (NC) · ${data.elapsedSeconds.toFixed(1)} sec processing`;
  document.getElementById('pipelineBackend').textContent = `LOCAL · ${data.endpoint}`;
  document.getElementById('pipelineList').innerHTML = [
    'MuScriptor transformer',
    '5 s autoregressive chunks',
    'instrument-aware MIDI',
    'localhost bridge',
    'CC BY-NC 4.0 weights',
  ].map(text => `<span class="pipeline-chip">${escapeHtml(text)}</span>`).join('');

  const instruments = [...new Set(data.rawNotes.map(note => note.instrument))].sort();
  document.getElementById('cleanupSummary').innerHTML = `
    <div><span>Tonal notes</span><strong>${data.notes.length.toLocaleString()}</strong></div>
    <div><span>Drum events</span><strong>${data.drums.length.toLocaleString()}</strong></div>
    <div><span>Instruments</span><strong>${instruments.length}</strong></div>
    <div><span>Tempo grid</span><strong>${data.beatGrid?.bpm ? 'detected' : 'none'}</strong></div>
    <div><span>Engine</span><strong>MuScriptor</strong></div>
    <div><span>License</span><strong>NC</strong></div>
  `;
  document.getElementById('chordList').innerHTML = '<p class="muted">MuScriptor bridge mode does not emit chord labels.</p>';
  document.getElementById('backendLabel').textContent = 'backend: MUSCRIPTOR';
  drawPianoRoll(data.notes, data.file);
  requestAnimationFrame(() => results.scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

function drawPianoRoll(notes, file) {
  const canvas = document.getElementById('pianoRoll');
  const wrap = canvas.parentElement;
  const duration = Math.max(1, ...notes.map(note => note.startTimeSeconds + note.durationSeconds));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssWidth = Math.max(wrap.clientWidth, Math.min(6200, 900 + duration * 8));
  const cssHeight = 430;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#0b0b0c';
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const pitches = notes.map(note => note.pitchMidi);
  const minPitch = pitches.length ? Math.max(21, Math.min(...pitches) - 3) : 48;
  const maxPitch = pitches.length ? Math.min(108, Math.max(...pitches) + 3) : 84;
  const span = Math.max(12, maxPitch - minPitch + 1);
  for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
    const y = cssHeight - ((pitch - minPitch + 1) / span) * cssHeight;
    ctx.strokeStyle = pitch % 12 === 0 ? '#303033' : '#1b1b1e';
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssWidth, y); ctx.stroke();
  }
  for (const note of notes) {
    const x = (note.startTimeSeconds / duration) * cssWidth;
    const w = Math.max(1.5, (note.durationSeconds / duration) * cssWidth);
    const y = cssHeight - ((note.pitchMidi - minPitch + 1) / span) * cssHeight;
    const h = Math.max(3, cssHeight / span - 1);
    ctx.fillStyle = '#dddddf';
    ctx.fillRect(x, y + 1, w, h);
  }
  canvas.dataset.source = file.name;
}

function showProgress(value, text) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  document.getElementById('progressPanel').hidden = false;
  document.getElementById('progressText').textContent = text;
  document.getElementById('progressPct').textContent = `${pct}%`;
  document.getElementById('progressBar').style.width = `${pct}%`;
}

function clearUltraResult() {
  ultraResult = null;
  document.getElementById('results')?.removeAttribute('data-engine');
}

function localFetch(url, init = {}) {
  const options = { mode: 'cors', ...init };
  if (/^http:\/\/(?:127(?:\.\d+){3}|localhost|\[::1\])(?::\d+)?(?:\/|$)/i.test(url)) {
    options.targetAddressSpace = 'loopback';
  }
  return fetch(url, options);
}

function normalizeEndpoint(value) {
  const fallback = DEFAULT_ENDPOINT;
  try {
    const url = new URL(String(value || fallback));
    if (!['http:', 'https:'].includes(url.protocol)) return fallback;
    return url.href.replace(/\/$/, '');
  } catch {
    return fallback;
  }
}

function clientId() {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = globalThis.crypto?.randomUUID?.() || `wav2mid-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

function base64ToBytes(value) {
  const binary = atob(String(value || ''));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function maxPolyphony(notes) {
  const events = [];
  for (const note of notes) {
    events.push([note.startTimeSeconds, 1]);
    events.push([note.startTimeSeconds + note.durationSeconds, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let active = 0;
  let maximum = 0;
  for (const [, delta] of events) {
    active += delta;
    maximum = Math.max(maximum, active);
  }
  return maximum;
}

function isDrum(instrument) {
  return /drum|percussion|cymbal|hi.?hat|kick|snare/i.test(String(instrument || ''));
}

function midiName(midi) {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

function basename(name) {
  return String(name || 'transcription').replace(/\.[^.]+$/, '') || 'transcription';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}
