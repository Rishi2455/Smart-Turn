// ============================================================
// EOU Detector — Frontend Logic
// ============================================================

const API_BASE = '';
let ws = null;
let wsReconnectTimer = null;
let rtDebounceTimer = null;

// ============================================================
// Tab Navigation
// ============================================================

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        const tabId = `tab-${tab.dataset.tab}`;
        document.getElementById(tabId).classList.add('active');

        // Connect WebSocket when switching to real-time tab
        if (tab.dataset.tab === 'realtime') {
            connectWebSocket();
        }
    });
});

// ============================================================
// Init — Check Model Status
// ============================================================

async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();

        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('.status-text');
        const deviceInfo = document.getElementById('deviceInfo');

        if (data.is_loaded) {
            dot.classList.add('loaded');
            dot.classList.remove('error');
            text.textContent = 'Model Loaded';
            deviceInfo.textContent = data.device || '';
            loadModelInfo(data);
        } else {
            dot.classList.add('error');
            dot.classList.remove('loaded');
            text.textContent = 'Model Not Loaded';
        }
    } catch (e) {
        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('.status-text');
        dot.classList.add('error');
        text.textContent = 'Server Unreachable';
    }
}

function loadModelInfo(data) {
    const el = document.getElementById('modelInfo');
    const config = data.config || {};

    el.innerHTML = `
        <div class="info-row">
            <span class="info-key">Status</span>
            <span class="info-val">${data.is_loaded ? '✅ Loaded' : '❌ Not Loaded'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Device</span>
            <span class="info-val">${data.device || 'N/A'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Model</span>
            <span class="info-val">${config.model_name || 'N/A'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Threshold</span>
            <span class="info-val">${data.threshold || 'N/A'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Max Length</span>
            <span class="info-val">${config.max_length || 'N/A'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Aux Features</span>
            <span class="info-val">${config.use_aux_features ? 'Yes' : 'No'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Test F1</span>
            <span class="info-val">${config.test_f1 ? config.test_f1.toFixed(4) : 'N/A'}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Model Dir</span>
            <span class="info-val">${data.model_dir || 'N/A'}</span>
        </div>
    `;

    // Update threshold slider
    if (data.threshold) {
        document.getElementById('thresholdSlider').value = Math.round(data.threshold * 100);
        document.getElementById('thresholdDisplay').textContent = data.threshold.toFixed(2);
    }
}

// ============================================================
// Single Prediction
// ============================================================

async function predictSingle() {
    const text = document.getElementById('textInput').value.trim();

    if (!text) {
        showToast('Please enter some text', 'error');
        return;
    }

    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await res.json();
        displayResult(data);

    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Utterance';
    }
}

function displayResult(data) {
    const card = document.getElementById('resultCard');
    card.classList.remove('hidden', 'complete', 'incomplete');
    card.classList.add(data.is_complete ? 'complete' : 'incomplete');

    // Badge
    const badge = document.getElementById('resultBadge');
    badge.className = `result-badge ${data.is_complete ? 'complete' : 'incomplete'}`;
    badge.innerHTML = data.is_complete
        ? '✅ Complete Utterance'
        : '⏳ Incomplete Utterance';

    // Inference time
    document.getElementById('inferenceTime').textContent =
        `${data.inference_time_ms}ms`;

    // Probability bars
    const compPct = (data.complete_probability * 100).toFixed(1);
    const incPct = (data.incomplete_probability * 100).toFixed(1);

    document.getElementById('completeProb').textContent = `${compPct}%`;
    document.getElementById('incompleteProb').textContent = `${incPct}%`;
    document.getElementById('completeBar').style.width = `${compPct}%`;
    document.getElementById('incompleteBar').style.width = `${incPct}%`;

    // Threshold
    document.getElementById('thresholdValue').textContent =
        data.threshold.toFixed(2);

    // Features
    const featureGrid = document.getElementById('featureGrid');
    if (data.features) {
        featureGrid.innerHTML = Object.entries(data.features).map(([name, val]) => {
            const isActive = val > 0;
            return `
                <div class="feature-item">
                    <span class="feature-name">${name}</span>
                    <span class="feature-value ${isActive ? 'active' : 'inactive'}">
                        ${typeof val === 'number' && val % 1 !== 0 ? val.toFixed(3) : val}
                    </span>
                </div>
            `;
        }).join('');
    }

    // Scroll into view
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================
// Audio Recording & Upload (Deepgram ASR)
// ============================================================

let audioWs = null;
let audioContext = null;
let mediaStream = null;
let audioProcessor = null;
let isRecording = false;

async function toggleRecording() {
    const btn = document.getElementById('recordBtn');
    const status = document.getElementById('recordingStatus');

    if (isRecording) {
        // Stop recording
        stopLiveAudio();
        btn.innerHTML = '<span class="btn-icon">🎤</span> Start Recording';
        btn.classList.remove('recording');
        status.textContent = 'Processing final audio...';
        return;
    }

    try {
        // Connect WS
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/audio`;

        audioWs = new WebSocket(wsUrl);

        audioWs.onopen = async () => {
            // Request Mic
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Set up Web Audio API to capture raw audio
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000 // Deepgram Nova-2 prefers 16kHz
            });
            const source = audioContext.createMediaStreamSource(mediaStream);

            // Create a ScriptProcessorNode with a bufferSize of 4096 and a single input and output channel
            audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            source.connect(audioProcessor);
            audioProcessor.connect(audioContext.destination);

            audioProcessor.onaudioprocess = (e) => {
                if (!isRecording || audioWs.readyState !== WebSocket.OPEN) return;

                const float32Array = e.inputBuffer.getChannelData(0);
                // Convert float32 to int16 for Deepgram
                const int16Array = new Int16Array(float32Array.length);
                for (let i = 0; i < float32Array.length; i++) {
                    let s = Math.max(-1, Math.min(1, float32Array[i]));
                    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                audioWs.send(int16Array.buffer);
            };

            isRecording = true;
            btn.innerHTML = '<span class="btn-icon">⏹️</span> Stop Recording';
            btn.classList.add('recording');
            status.textContent = 'Recording live (streaming)...';

            const box = document.getElementById('transcriptionBox');
            box.classList.remove('hidden');
            document.getElementById('transcriptionText').textContent = "Listening...";
        };

        audioWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.error) {
                console.error("Audio WS Error:", data.error);
                showToast(data.error, "error");
                stopLiveAudio();
                return;
            }

            const textEl = document.getElementById('transcriptionText');

            if (data.type === "interim") {
                textEl.textContent = data.text + " ...";
            } else if (data.type === "final") {
                textEl.textContent = data.text;
                document.getElementById('textInput').value = data.text;
                if (data.prediction) {
                    displayResult(data.prediction);
                }

                // We stop capturing after a final snippet for a single turn, but could keep going.
                // For a "turn", we'll stop after the first final result.
                stopLiveAudio();
                btn.innerHTML = '<span class="btn-icon">🎤</span> Start Recording';
                btn.classList.remove('recording');
                status.textContent = 'Recording complete.';
            }
        };

        audioWs.onclose = () => {
            stopLiveAudio();
            status.textContent = 'Disconnected.';
            btn.innerHTML = '<span class="btn-icon">🎤</span> Start Recording';
            btn.classList.remove('recording');
        };

        audioWs.onerror = (err) => {
            console.error('Audio WS error:', err);
            stopLiveAudio();
            showToast("Audio connection error.", "error");
        };

    } catch (err) {
        console.error("Microphone access denied or error:", err);
        showToast("Microphone access denied or unavailable.", "error");
    }
}

function stopLiveAudio() {
    isRecording = false;
    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioWs) {
        if (audioWs.readyState === WebSocket.OPEN) {
            audioWs.close();
        }
        audioWs = null;
    }
}

async function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('fileName').textContent = file.name;

    // Send to backend
    await processAudio(file, file.name);

    // Reset input
    event.target.value = '';
}

async function processAudio(audioBlob, filename) {
    const formData = new FormData();
    formData.append("file", audioBlob, filename);

    showToast("Processing audio...", "info");

    try {
        const res = await fetch(`${API_BASE}/api/predict/audio`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Audio processing failed');
        }

        const data = await res.json();

        // Show transcription
        const box = document.getElementById('transcriptionBox');
        const textEl = document.getElementById('transcriptionText');
        textEl.textContent = data.text || "(No speech detected)";
        box.classList.remove('hidden');

        // Fill the text input with transcription
        document.getElementById('textInput').value = data.text;

        // Display EOU result
        displayResult(data.prediction);
        showToast("Audio analyzed successfully!", "success");

    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

// ============================================================
// WebSocket — Real-time Prediction
// ============================================================

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/predict`;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            updateWsStatus(true);
            showToast('WebSocket connected', 'success');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.error) {
                console.warn('WS error:', data.error);
                return;
            }
            displayRealtimeResult(data);
        };

        ws.onclose = () => {
            updateWsStatus(false);
            // Auto-reconnect after 3 seconds
            wsReconnectTimer = setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            updateWsStatus(false);
        };
    } catch (e) {
        console.error('WebSocket connection failed:', e);
        updateWsStatus(false);
    }
}

function updateWsStatus(connected) {
    const statusEl = document.getElementById('wsStatus');
    const dot = statusEl.querySelector('.ws-dot');
    dot.className = `ws-dot ${connected ? 'connected' : 'disconnected'}`;
    statusEl.innerHTML = `
        <span class="ws-dot ${connected ? 'connected' : 'disconnected'}"></span>
        WebSocket: ${connected ? 'Connected ✅' : 'Disconnected ❌'}
    `;
}

function displayRealtimeResult(data) {
    const indicator = document.getElementById('rtIndicator');
    indicator.className = `rt-indicator ${data.is_complete ? 'complete' : 'incomplete'}`;
    indicator.querySelector('.rt-label').textContent =
        data.is_complete ? '✅ Complete' : '⏳ Incomplete';

    // Confidence bar
    const confBar = document.getElementById('rtConfBar');
    const pct = data.complete_probability * 100;
    confBar.style.width = `${pct}%`;
    confBar.style.background = data.is_complete
        ? 'linear-gradient(90deg, #22c55e, #4ade80)'
        : 'linear-gradient(90deg, #f87171, #fca5a5)';

    // Details
    document.getElementById('rtDetails').innerHTML = `
        P(Complete): ${(data.complete_probability * 100).toFixed(1)}% |
        Confidence: ${(data.confidence * 100).toFixed(1)}% |
        ${data.inference_time_ms}ms
    `;
}

// Real-time input handler with debounce
document.addEventListener('DOMContentLoaded', () => {
    const rtInput = document.getElementById('rtTextInput');
    if (rtInput) {
        rtInput.addEventListener('input', () => {
            clearTimeout(rtDebounceTimer);
            rtDebounceTimer = setTimeout(() => {
                const text = rtInput.value.trim();

                if (text && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        text: text,
                        request_id: Date.now().toString(),
                    }));
                }
            }, 200); // 200ms debounce
        });
    }

    // Threshold slider
    const slider = document.getElementById('thresholdSlider');
    if (slider) {
        slider.addEventListener('input', () => {
            const val = (slider.value / 100).toFixed(2);
            document.getElementById('thresholdDisplay').textContent = val;
        });
    }

    // Enter key to predict
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                predictSingle();
            }
        });
    }

    // Check status on load
    checkStatus();
});

// ============================================================
// Batch Prediction
// ============================================================

async function predictBatch() {
    const input = document.getElementById('batchInput').value.trim();
    if (!input) {
        showToast('Please enter some text (one utterance per line)', 'error');
        return;
    }

    const lines = input.split('\n').filter(l => l.trim());
    if (lines.length === 0) {
        showToast('No valid utterances found', 'error');
        return;
    }

    if (lines.length > 50) {
        showToast('Maximum 50 utterances per batch', 'error');
        return;
    }

    try {
        const items = lines.map(text => ({ text: text.trim() }));

        const res = await fetch(`${API_BASE}/api/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Batch prediction failed');
        }

        const data = await res.json();
        displayBatchResults(data.results);

    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

function displayBatchResults(results) {
    const container = document.getElementById('batchResults');
    container.classList.remove('hidden');

    // Stats
    const total = results.length;
    const complete = results.filter(r => r.is_complete).length;
    const incomplete = total - complete;

    document.getElementById('batchStats').innerHTML = `
        <div class="batch-stat total">
            <span class="stat-number">${total}</span>
            <span class="stat-label">Total</span>
        </div>
        <div class="batch-stat complete-stat">
            <span class="stat-number">${complete}</span>
            <span class="stat-label">Complete</span>
        </div>
        <div class="batch-stat incomplete-stat">
            <span class="stat-number">${incomplete}</span>
            <span class="stat-label">Incomplete</span>
        </div>
    `;

    // Table
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = results.map((r, i) => `
        <tr>
            <td>${i + 1}</td>
            <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
                title="${r.text}">${r.text}</td>
            <td>
                <span class="result-tag ${r.is_complete ? 'complete' : 'incomplete'}">
                    ${r.is_complete ? '✅ Complete' : '⏳ Incomplete'}
                </span>
            </td>
            <td>${(r.confidence * 100).toFixed(1)}%</td>
            <td>${(r.complete_probability * 100).toFixed(1)}%</td>
            <td>${r.inference_time_ms}</td>
        </tr>
    `).join('');

    container.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// Settings
// ============================================================

async function updateThreshold() {
    const value = document.getElementById('thresholdSlider').value / 100;

    try {
        const res = await fetch(`${API_BASE}/api/threshold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold: value }),
        });

        if (res.ok) {
            const data = await res.json();
            showToast(
                `Threshold updated: ${data.old_threshold.toFixed(2)} → ${data.new_threshold.toFixed(2)}`,
                'success'
            );
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function loadNewModel() {
    const modelPath = document.getElementById('modelPathInput').value.trim();
    if (!modelPath) {
        showToast('Please enter a model directory path', 'error');
        return;
    }

    showToast('Loading model...', 'info');

    try {
        const res = await fetch(`${API_BASE}/api/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_dir: modelPath }),
        });

        if (res.ok) {
            const data = await res.json();
            showToast(`Model loaded in ${data.load_time_seconds}s!`, 'success');
            checkStatus();
        } else {
            const err = await res.json();
            throw new Error(err.detail);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

// ============================================================
// Toast Notifications
// ============================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}