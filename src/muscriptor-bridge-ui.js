import './ultra.css';

let currentFile = null;
let ultraResult = null;
let modelPromise = null;

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
        <small>非商用 · Browser WebGPU · multi-instrument transformer</small>
      </span>
      <span class="ultra-state" id="ultraState">lazy</span>
    </label>
    <div class="ultra-config" id="ultraConfig" hidden>
      <p><strong>端末内推論。</strong> 音源はサーバーへ送信しません。初回解析時だけMuScriptor-smallのモデルを取得し、WebGPUで実行します。モデル重みはCC BY-NC 4.0です。</p>
    </div>
  `;
  neuralOption.after(block);

  const ultraToggle = document.getElementById('ultraToggle');
  const ultraOption = document.getElementById('ultraOption');
  const ultraConfig = document.getElementById('ultraConfig');
  const state = document.getElementById('ultraState');
  const neuralToggle = document.getElementById('neuralToggle');
  const qualityGroup = document.getElementById('qualityGroup');
  const modeNote = document.getElementById('modeNote');
  const progressHint = document.getElementById('progressHint');
  const fileInput = document.getElementById('fileInput');
  const dropZone = document.getElementById('dropZone');

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
    state.textContent = 'loading';
    showProgress(0.01, 'MuScriptor WebGPUを準備中…');

    const started = performance.now();
    let audioContext = null;
    try {
      if (!navigator.gpu && !globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__) {
        throw new Error('WebGPUが利用できません。Chromium系ブラウザと対応GPUを使用してください。');
      }

      const model = await loadBrowserModel();
      showProgress(0.04, '音声をデコード中…');
      audioContext = new AudioContext();
      const audioBytes = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(audioBytes.slice(0));
      const parsed = await model.transcribe(audioBuffer, {}, progress => {
        const value = Number(progress?.value);
        if (Number.isFinite(value)) {
          const detail = progress?.detail ? ` · ${progress.detail}` : '';
          showProgress(0.05 + Math.max(0, Math.min(1, value)) * 0.93, `MuScriptor WebGPU${detail}`);
        }
      });

      const tempo = estimateTempo(parsed.notes || []);
      ultraResult = {
        ...parsed,
        file,
        tempo,
        elapsedSeconds: (performance.now() - started) / 1000,
      };
      renderUltraResult(ultraResult);
      state.textContent = 'ready';
      showProgress(1, `完了 · ${parsed.notes.length} notes`);
    } catch (error) {
      console.error(error);
      state.textContent = 'error';
      showProgress(0, `MuScriptor失敗: ${error?.message || error}`);
      progressHint.textContent = 'WebGPU、GPUメモリ、モデル配信先への接続を確認してください。音源自体は外部へ送信されません。';
    } finally {
      if (audioContext) await audioContext.close().catch(() => {});
      analyzeBtn.disabled = false;
    }
  }, true);

  document.getElementById('midiBtn')?.addEventListener('click', async event => {
    if (!ultraResult || document.getElementById('results')?.dataset.engine !== 'muscriptor') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    await downloadUltraMidi(ultraResult);
  }, true);

  document.getElementById('jsonBtn')?.addEventListener('click', event => {
    if (!ultraResult || document.getElementById('results')?.dataset.engine !== 'muscriptor') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const payload = {
      format: 'wav2mid-muscriptor-browser/v1',
      engine: 'MuScriptor small',
      backend: 'WebGPU',
      license: 'CC BY-NC 4.0 model weights',
      tempo: ultraResult.tempo,
      model: ultraResult.model,
      notes: ultraResult.notes,
      drums: ultraResult.drums,
    };
    downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }), `${basename(ultraResult.file.name)}.muscriptor.json`);
  }, true);

  function updateUltraUi() {
    const active = ultraToggle.checked;
    ultraOption.classList.toggle('active', active);
    ultraConfig.hidden = !active;
    qualityGroup?.classList.toggle('disabled', active || Boolean(neuralToggle?.checked));
    if (active) {
      modeNote.textContent = 'MuScriptor transformer · multi-instrument · 非商用 · Browser WebGPU。';
      progressHint.textContent = '初回解析時にモデルを取得します。音源はブラウザ内だけで処理され、外部サーバーへ送信されません。';
    }
  }
}

async function loadBrowserModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      if (globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__) {
        return globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__();
      }
      const { loadMuScriptorBrowserModel } = await import('./muscriptor-browser-core.js');
      return loadMuScriptorBrowserModel();
    })().catch(error => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

async function downloadUltraMidi(data) {
  const { Midi } = await import('@tonejs/midi');
  const midi = new Midi();
  midi.header.setTempo(Number.isFinite(data.tempo) ? data.tempo : 120);

  const tracks = new Map();
  const ensureTrack = program => {
    if (tracks.has(program)) return tracks.get(program);
    const track = midi.addTrack();
    track.name = programName(program);
    track.instrument.number = Math.max(0, Math.min(127, program));
    tracks.set(program, track);
    return track;
  };

  for (const note of data.notes || []) {
    const program = Number.isFinite(note.program) ? note.program : 0;
    ensureTrack(program).addNote({
      midi: note.pitchMidi,
      time: note.startTimeSeconds,
      duration: Math.max(0.01, note.durationSeconds),
      velocity: Math.max(0.01, Math.min(1, note.amplitude ?? note.confidence ?? 0.8)),
    });
  }

  if ((data.drums || []).length) {
    const drumTrack = midi.addTrack();
    drumTrack.name = 'Drums';
    drumTrack.channel = 9;
    for (const drum of data.drums) {
      drumTrack.addNote({
        midi: drum.midi,
        time: drum.time,
        duration: Math.max(0.01, drum.duration || 0.05),
        velocity: Math.max(0.01, Math.min(1, drum.velocity ?? 0.8)),
      });
    }
  }

  downloadBlob(new Blob([midi.toArray()], { type: 'audio/midi' }), `${basename(data.file.name)}.muscriptor.mid`);
}

function renderUltraResult(data) {
  const results = document.getElementById('results');
  results.hidden = false;
  results.dataset.engine = 'muscriptor';
  document.getElementById('resultTitle').textContent = data.file.name;
  document.getElementById('statNotes').textContent = data.notes.length.toLocaleString();
  document.getElementById('statDrums').textContent = data.drums.length.toLocaleString();
  document.getElementById('statTempo').textContent = Number.isFinite(data.tempo) ? `${Math.round(data.tempo)} BPM` : '—';
  document.getElementById('statKey').textContent = '—';
  document.getElementById('statPoly').textContent = maxPolyphony(data.notes);
  document.getElementById('statEnsemble').textContent = 'NC';
  const pitches = data.notes.map(note => note.pitchMidi);
  document.getElementById('statRange').textContent = pitches.length ? `${midiName(Math.min(...pitches))} – ${midiName(Math.max(...pitches))}` : '—';
  document.getElementById('rollLegend').textContent = `MuScriptor ULTRA (NC) · ${data.elapsedSeconds.toFixed(1)} sec processing`;
  document.getElementById('pipelineBackend').textContent = 'WEBGPU · DEVICE';
  const activation = data.model?.architecture?.decoderActivationType || data.model?.architecture?.activationType || 'unknown';
  document.getElementById('pipelineList').innerHTML = [
    'MuScriptor transformer',
    '5 s autoregressive chunks',
    'instrument-aware MIDI',
    `WebGPU · ${activation}`,
    'INT4 weight-only decoder',
    'CC BY-NC 4.0 weights',
  ].map(text => `<span class="pipeline-chip">${escapeHtml(text)}</span>`).join('');

  const instruments = [...new Set((data.rawNotes || []).map(note => note.instrument))].sort();
  document.getElementById('cleanupSummary').innerHTML = `
    <div><span>Tonal notes</span><strong>${data.notes.length.toLocaleString()}</strong></div>
    <div><span>Drum events</span><strong>${data.drums.length.toLocaleString()}</strong></div>
    <div><span>Instruments</span><strong>${instruments.length}</strong></div>
    <div><span>Chunks</span><strong>${Number(data.chunks || 0).toLocaleString()}</strong></div>
    <div><span>Engine</span><strong>MuScriptor</strong></div>
    <div><span>Backend</span><strong>WebGPU</strong></div>
  `;
  document.getElementById('chordList').innerHTML = '<p class="muted">MuScriptor browser mode does not emit chord labels.</p>';
  document.getElementById('backendLabel').textContent = 'backend: MUSCRIPTOR · WEBGPU';
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

function estimateTempo(notes) {
  if (!notes || notes.length < 4) return 120;
  const onsets = [...new Set(notes
    .map(note => Math.round(Number(note.startTimeSeconds) * 100) / 100)
    .filter(Number.isFinite))]
    .sort((a, b) => a - b)
    .slice(0, 600);
  const votes = new Map();
  for (let i = 0; i < onsets.length; i += 1) {
    for (let j = i + 1; j < Math.min(onsets.length, i + 8); j += 1) {
      const dt = onsets[j] - onsets[i];
      if (dt < 0.18 || dt > 2) continue;
      let bpm = 60 / dt;
      while (bpm < 70) bpm *= 2;
      while (bpm > 180) bpm /= 2;
      const rounded = Math.round(bpm);
      votes.set(rounded, (votes.get(rounded) || 0) + 1 / (j - i));
    }
  }
  if (!votes.size) return 120;
  return [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
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

function programName(program) {
  return `MuScriptor · Program ${program}`;
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
