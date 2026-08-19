import './style.css';
import { configureBackend, exportJson, exportMidi, midiName, MODES, transcribe } from './transcribe.js';

const app = document.querySelector('#app');
app.innerHTML = `
  <section class="hero shell">
    <div>
      <p class="eyebrow">LOCAL AUDIO → MIDI</p>
      <h1>Wav2mid <span>HQ</span></h1>
      <p class="lead">音源はアップロードされません。ブラウザ内のTensorFlow.js / WASMで解析し、MIDIまで生成します。</p>
    </div>
    <div class="backend-pill"><span class="dot"></span><span id="backendLabel">backend: loading…</span></div>
  </section>

  <section class="shell workspace">
    <div class="panel input-panel">
      <label id="dropZone" class="drop-zone" for="fileInput">
        <input id="fileInput" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" hidden />
        <div class="drop-icon">↥</div>
        <strong>音源をドロップ / 選択</strong>
        <span>WAV / MP3 / M4A / OGG など、ブラウザが再生できる形式</span>
      </label>
      <audio id="audioPreview" controls hidden></audio>
      <div id="fileInfo" class="file-info muted">まだ音源がありません</div>
    </div>

    <div class="panel settings-panel">
      <div class="panel-title"><h2>解析設定</h2><span id="privacyBadge">LOCAL ONLY</span></div>
      <div class="setting-row">
        <label>Quality</label>
        <div class="segmented" id="qualityGroup">
          <button data-mode="fast">FAST</button>
          <button data-mode="pro" class="active">PRO</button>
          <button data-mode="insane">INSANE</button>
        </div>
      </div>
      <div class="setting-row">
        <label for="backendSelect">Compute</label>
        <select id="backendSelect">
          <option value="auto">Auto</option>
          <option value="wasm">WASM</option>
          <option value="webgl">WebGL</option>
        </select>
      </div>
      <div class="setting-row slider-row">
        <label for="sensitivity">Sensitivity <output id="sensitivityOut">1.00×</output></label>
        <input id="sensitivity" type="range" min="0.65" max="1.5" value="1" step="0.05" />
      </div>
      <div class="range-grid">
        <label>Lowest note <input id="minPitch" type="number" min="0" max="127" value="21" /></label>
        <label>Highest note <input id="maxPitch" type="number" min="0" max="127" value="108" /></label>
      </div>
      <button id="analyzeBtn" class="primary" disabled>ANALYZE AUDIO</button>
    </div>
  </section>

  <section class="shell panel progress-panel" id="progressPanel" hidden>
    <div class="progress-head"><strong id="progressText">解析準備中…</strong><span id="progressPct">0%</span></div>
    <div class="progress-track"><div id="progressBar"></div></div>
    <p class="muted small" id="progressHint">初回はモデルの読み込みがあるため時間がかかります。</p>
  </section>

  <section class="shell results" id="results" hidden>
    <div class="result-header">
      <div><p class="eyebrow">ANALYSIS COMPLETE</p><h2 id="resultTitle">Result</h2></div>
      <div class="actions"><button id="jsonBtn" class="secondary">JSON</button><button id="midiBtn" class="primary compact">DOWNLOAD MIDI</button></div>
    </div>

    <div class="stat-grid">
      <div class="stat"><span>NOTES</span><strong id="statNotes">—</strong></div>
      <div class="stat"><span>TEMPO</span><strong id="statTempo">—</strong></div>
      <div class="stat"><span>KEY</span><strong id="statKey">—</strong></div>
      <div class="stat"><span>MAX POLY</span><strong id="statPoly">—</strong></div>
      <div class="stat"><span>RANGE</span><strong id="statRange">—</strong></div>
    </div>

    <div class="panel roll-panel">
      <div class="panel-title"><h2>Piano roll</h2><span id="rollLegend" class="muted small"></span></div>
      <div class="canvas-wrap"><canvas id="pianoRoll"></canvas></div>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-title"><h2>Chord timeline</h2><span class="muted small">推定</span></div>
        <div id="chordList" class="chord-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title"><h2>Cleanup</h2><span class="muted small">post-processing</span></div>
        <div id="cleanupSummary" class="cleanup"></div>
      </div>
    </div>
  </section>

  <footer class="shell">Wav2mid HQ · runs locally in your browser · Apache-2.0 model dependency: Spotify Basic Pitch</footer>
`;

const el = id => document.getElementById(id);
let selectedFile = null;
let audioBuffer = null;
let audioUrl = null;
let result = null;
let qualityMode = 'pro';
let currentBackendPreference = 'auto';

const backendLabel = el('backendLabel');
const analyzeBtn = el('analyzeBtn');
const fileInput = el('fileInput');
const dropZone = el('dropZone');
const audioPreview = el('audioPreview');

initializeBackend();

async function initializeBackend(pref = 'auto') {
  backendLabel.textContent = 'backend: loading…';
  try {
    const backend = await configureBackend(pref);
    backendLabel.textContent = `backend: ${backend.toUpperCase()}`;
  } catch (error) {
    backendLabel.textContent = 'backend: unavailable';
    console.error(error);
  }
}

fileInput.addEventListener('change', () => handleFile(fileInput.files?.[0]));
for (const type of ['dragenter', 'dragover']) {
  dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add('drag'); });
}
for (const type of ['dragleave', 'drop']) {
  dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove('drag'); });
}
dropZone.addEventListener('drop', event => handleFile(event.dataTransfer?.files?.[0]));

async function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  result = null;
  el('results').hidden = true;
  analyzeBtn.disabled = true;
  el('fileInfo').textContent = `${file.name} · ${formatBytes(file.size)} · decoding…`;

  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(file);
  audioPreview.src = audioUrl;
  audioPreview.hidden = false;

  try {
    const bytes = await file.arrayBuffer();
    const ctx = new AudioContext();
    audioBuffer = await ctx.decodeAudioData(bytes.slice(0));
    await ctx.close();
    el('fileInfo').textContent = `${file.name} · ${formatBytes(file.size)} · ${formatTime(audioBuffer.duration)} · ${audioBuffer.sampleRate.toLocaleString()} Hz`;
    analyzeBtn.disabled = false;
  } catch (error) {
    audioBuffer = null;
    el('fileInfo').textContent = 'この音源はブラウザでデコードできませんでした。WAV/MP3へ変換して再試行してください。';
    console.error(error);
  }
}

el('qualityGroup').addEventListener('click', event => {
  const button = event.target.closest('button[data-mode]');
  if (!button) return;
  qualityMode = button.dataset.mode;
  [...el('qualityGroup').querySelectorAll('button')].forEach(b => b.classList.toggle('active', b === button));
});

el('backendSelect').addEventListener('change', async event => {
  currentBackendPreference = event.target.value;
  analyzeBtn.disabled = true;
  await initializeBackend(currentBackendPreference);
  analyzeBtn.disabled = !audioBuffer;
});

el('sensitivity').addEventListener('input', event => {
  el('sensitivityOut').textContent = `${Number(event.target.value).toFixed(2)}×`;
});

analyzeBtn.addEventListener('click', async () => {
  if (!audioBuffer || !selectedFile) return;
  analyzeBtn.disabled = true;
  el('progressPanel').hidden = false;
  el('results').hidden = true;
  updateProgress(0, 'モデル準備中…');

  try {
    const start = performance.now();
    result = await transcribe(audioBuffer, {
      mode: qualityMode,
      sensitivity: Number(el('sensitivity').value),
      minPitch: Number(el('minPitch').value),
      maxPitch: Number(el('maxPitch').value),
    }, ({ stage, value }) => {
      const labels = { infer: 'ニューラル解析中…', decode: 'ノートへ変換中…', clean: '倍音・ゴーストを整理中…', done: '完了' };
      updateProgress(value, labels[stage] ?? '解析中…');
    });
    result.elapsedSeconds = (performance.now() - start) / 1000;
    renderResult();
  } catch (error) {
    console.error(error);
    updateProgress(0, `解析失敗: ${error?.message ?? error}`);
    el('progressHint').textContent = '開発者コンソールに詳細を出力しました。別の音源形式やCompute backendも試せます。';
  } finally {
    analyzeBtn.disabled = false;
  }
});

el('midiBtn').addEventListener('click', () => {
  if (!result) return;
  exportMidi(result, `${basename(selectedFile.name)}.mid`);
});
el('jsonBtn').addEventListener('click', () => {
  if (!result) return;
  exportJson(result, `${basename(selectedFile.name)}.analysis.json`);
});

function updateProgress(value, text) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  el('progressText').textContent = text;
  el('progressPct').textContent = `${pct}%`;
  el('progressBar').style.width = `${pct}%`;
}

function renderResult() {
  el('results').hidden = false;
  el('resultTitle').textContent = selectedFile.name;
  el('statNotes').textContent = result.stats.noteCount.toLocaleString();
  el('statTempo').textContent = `${result.tempo} BPM`;
  el('statKey').textContent = result.key;
  el('statPoly').textContent = result.stats.maxPolyphony;
  el('statRange').textContent = `${midiName(result.stats.lowestNote)} – ${midiName(result.stats.highestNote)}`;
  el('rollLegend').textContent = `${MODES[qualityMode].label} · ${result.elapsedSeconds.toFixed(1)} sec processing`;

  const removed = Math.max(0, result.rawNotes.length - result.notes.length);
  el('cleanupSummary').innerHTML = `
    <div><span>Raw notes</span><strong>${result.rawNotes.length}</strong></div>
    <div><span>Final notes</span><strong>${result.notes.length}</strong></div>
    <div><span>Filtered / merged</span><strong>${removed}</strong></div>
    <div><span>Backend</span><strong>${backendLabel.textContent.replace('backend: ', '')}</strong></div>
  `;

  const chordList = el('chordList');
  chordList.innerHTML = result.chords.length
    ? result.chords.slice(0, 80).map(chord => `
      <div class="chord-row"><span class="time">${formatTime(chord.start)}</span><strong>${escapeHtml(chord.name)}</strong><span>${Math.round(chord.confidence * 100)}%</span></div>
    `).join('')
    : '<p class="muted">明確なコード候補を検出できませんでした。</p>';

  drawPianoRoll(result.notes, audioBuffer.duration);
  requestAnimationFrame(() => el('results').scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

function drawPianoRoll(notes, duration) {
  const canvas = el('pianoRoll');
  const wrap = canvas.parentElement;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssWidth = Math.max(wrap.clientWidth, Math.min(5200, 900 + duration * 7));
  const cssHeight = 420;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const minPitch = Math.max(21, Math.min(...notes.map(n => n.pitchMidi), 60) - 3);
  const maxPitch = Math.min(108, Math.max(...notes.map(n => n.pitchMidi), 72) + 3);
  const pitchSpan = Math.max(12, maxPitch - minPitch + 1);

  ctx.fillStyle = '#0b0d12';
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.lineWidth = 1;
  for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
    const y = cssHeight - ((pitch - minPitch + 1) / pitchSpan) * cssHeight;
    ctx.strokeStyle = pitch % 12 === 0 ? '#303643' : '#1b1f28';
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssWidth, y); ctx.stroke();
  }
  const secondsStep = duration > 300 ? 30 : duration > 120 ? 10 : 5;
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#697386';
  for (let sec = 0; sec <= duration; sec += secondsStep) {
    const x = (sec / duration) * cssWidth;
    ctx.strokeStyle = '#202631';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cssHeight); ctx.stroke();
    ctx.fillText(`${sec}s`, x + 3, 13);
  }

  for (const note of notes) {
    const x = (note.startTimeSeconds / duration) * cssWidth;
    const w = Math.max(1.5, (note.durationSeconds / duration) * cssWidth);
    const y = cssHeight - ((note.pitchMidi - minPitch + 1) / pitchSpan) * cssHeight;
    const h = Math.max(3, cssHeight / pitchSpan - 1);
    const alpha = 0.42 + note.amplitude * 0.58;
    ctx.fillStyle = `rgba(235, 239, 255, ${alpha.toFixed(3)})`;
    ctx.fillRect(x, y + 1, w, h);
  }
}

window.addEventListener('resize', () => { if (result) drawPianoRoll(result.notes, audioBuffer.duration); });

function basename(name) { return name.replace(/\.[^.]+$/, '') || 'transcription'; }
function formatBytes(bytes) { return bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`; }
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}
