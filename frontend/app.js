// ═══════════════════════════════════════════════════════════════════════════
// AI Face Tracking — WebGPU Client-Side Inference
// All AI runs in-browser via ONNX Runtime Web (WebGPU).
// Backend is a thin REST API for face DB (FAISS) only.
// ═══════════════════════════════════════════════════════════════════════════

const API = window.location.origin && window.location.origin !== 'null'
    ? window.location.origin
    : 'http://localhost:8000';

const DOM = {
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('gpuCanvas'),
    status: document.getElementById('connectionStatus'),
    fps: document.getElementById('fpsCounter'),
    yoloStatus: document.getElementById('yoloStatus'),
    faceNetStatus: document.getElementById('faceNetStatus'),
    trackName: document.getElementById('trackName'),
    trackId: document.getElementById('trackId'),
    activeSession: document.getElementById('activeSessionDisplay'),
    btnNew: document.getElementById('btnNewSession'),
    btnLoad: document.getElementById('btnLoadSession'),
    btnDelete: document.getElementById('btnDeleteSession'),
    btnPause: document.getElementById('btnPauseTracking'),
    btnUploadTrigger: document.getElementById('btnUploadImageTrigger'),
    imageUpload: document.getElementById('imageUpload'),
    inputSession: document.getElementById('sessionInput'),
    inputName: document.getElementById('nameInput'),
    inputReid: document.getElementById('reidInput'),
    btnUseClosest: document.getElementById('btnUseClosest'),
    btnUpdateName: document.getElementById('btnUpdateName'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingStatus: document.getElementById('loadingStatus'),
    loadingDetail: document.getElementById('loadingDetail'),
    progressBar: document.getElementById('progressBar'),
    debugConsole: document.getElementById('debugConsole'),
    modelSelector: document.getElementById('modelSelector'),
    // Crop Modal Elements
    cropModal: document.getElementById('cropModal'),
    cropContainer: document.getElementById('cropContainer'),
    cropImage: document.getElementById('cropImage'),
    cropBox: document.getElementById('cropBox'),
    cropNameInput: document.getElementById('cropNameInput'),
    btnCancelCrop: document.getElementById('btnCancelCrop'),
    btnSaveIdentity: document.getElementById('btnSaveIdentity')
};

const ctx = DOM.canvas.getContext('2d');
const extractCanvas = document.createElement('canvas');
const extractCtx = extractCanvas.getContext('2d', { willReadFrequently: true });

// ─── State ─────────────────────────────────────────────────────────────────

// Add ONNX Runtime Web configuration for WASM execution to prevent 3 minute freezes
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
ort.env.wasm.simd = true;

let currentSessionId = null;
let yoloSession = null;
let faceSession = null;
let executionProvider = 'wasm'; // will be 'webgpu' if available
let currentModelName = 'edgeface_xs_gamma_06'; // default EdgeFace model
let isPaused = false;

// Multi-face tracking state: Map<track_id, {box, reid, name, track_id, embeddingAge}>
let trackedFaces = new Map();
let closestTrackId = null; // track_id of largest face (shown in side panel)

// Simple IOU tracker state
let nextTrackId = 1;
let tracks = []; // [{id, box, age}]
const MAX_TRACK_AGE = 30; // from botsort.yaml track_buffer
const IOU_THRESHOLD = 0.15; // from botsort.yaml track_low_thresh

// Embedding throttle per face
const EMBEDDING_INTERVAL = 5; // query DB every N inference runs per track

// ─── Crop Modal State ──────────────────────────────────────────────────────
let cropState = {
    active: false,
    uploadedImage: null,
    isDragging: false,
    isResizing: false,
    resizeHandle: null,
    startX: 0,
    startY: 0,
    boxX: 0,
    boxY: 0,
    boxW: 100,
    boxH: 100,
    imgX: 0,
    imgY: 0,
    imgW: 0,
    imgH: 0
};

// ─── Loading / Boot ────────────────────────────────────────────────────────

function setLoading(status, detail, pct) {
    DOM.loadingStatus.textContent = status;
    DOM.loadingDetail.textContent = detail;
    DOM.progressBar.style.width = pct + '%';
}

function logDebug(msg) {
    if (!DOM.debugConsole) return;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const el = document.createElement('div');
    el.textContent = `[${time}] ${msg}`;
    DOM.debugConsole.appendChild(el);
    DOM.debugConsole.scrollTop = DOM.debugConsole.scrollHeight;
}

function resetTrackingState() {
    trackedFaces.clear();
    tracks = [];
    closestTrackId = null;
    nextTrackId = 1;
    updateSidePanel();
}

function setPaused(paused) {
    isPaused = paused;
    DOM.btnPause.textContent = isPaused ? 'Resume' : 'Pause';
    DOM.btnPause.classList.toggle('active', isPaused);

    if (isPaused) {
        resetTrackingState();
        DOM.status.textContent = 'Paused';
        DOM.status.className = 'status-indicator paused';
        DOM.fps.textContent = 'FPS: Paused';
        logDebug('Recognition paused');
    } else {
        DOM.status.textContent = `Running (${executionProvider.toUpperCase()})`;
        DOM.status.className = 'status-indicator connected';
        logDebug('Recognition resumed');
    }
}

function drawPauseOverlay() {
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.45)';
    ctx.fillRect(0, 0, DOM.canvas.width, DOM.canvas.height);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 18px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Paused', DOM.canvas.width / 2, DOM.canvas.height / 2);
    ctx.restore();
}

async function apiFetch(path, options = {}) {
    const res = await fetch(`${API}${path}`, options);
    let data = {};
    try {
        data = await res.json();
    } catch (_) {
        // Keep data as an empty object for non-JSON error responses.
    }
    if (!res.ok || data.status === 'error') {
        throw new Error(data.detail || data.message || `Request failed (${res.status})`);
    }
    return data;
}

function modelQueryParam() {
    return `model_name=${encodeURIComponent(currentModelName)}`;
}

async function boot() {
    try {
        // 1. Wait for backend
        setLoading('Connecting to backend...', 'Checking server health', 5);
        await waitForBackend();

        // 2. Load YOLO
        setLoading('Loading YOLO face detector...', 'Downloading ~38 MB ONNX model (WebGPU)', 20);
        await loadYOLO();

        // 3. Load face embedding model
        currentModelName = DOM.modelSelector.value;
        setLoading('Loading EdgeFace model...', `Loading ${currentModelName} (WebGPU)`, 55);
        await loadFaceNet(currentModelName);

        // 4. Init network / session
        setLoading('Setting up session...', 'Creating session on backend', 80);
        await initNetwork();

        // 5. Init camera
        setLoading('Starting camera...', 'Requesting webcam access', 90);
        await initCamera();

        // Done
        setLoading('Ready!', `AI running via ${executionProvider.toUpperCase()}`, 100);
        await new Promise(r => setTimeout(r, 500));

        DOM.loadingOverlay.classList.add('fade-out');
        DOM.loadingOverlay.addEventListener('transitionend', () => {
            DOM.loadingOverlay.style.display = 'none';
        }, { once: true });

        DOM.status.textContent = `Running (${executionProvider.toUpperCase()})`;
        DOM.status.className = 'status-indicator connected';

    } catch (e) {
        setLoading('Error!', e.message, 0);
        console.error('Boot failed:', e);
    }
}

async function waitForBackend() {
    while (true) {
        try {
            const res = await fetch(`${API}/api/health`, { cache: 'no-store' });
            if (res.ok) return;
        } catch (_) { }
        await new Promise(r => setTimeout(r, 800));
    }
}

// ─── ONNX Model Loading ───────────────────────────────────────────────────

async function loadYOLO() {
    const providers = ['webgpu', 'webgl', 'wasm'];
    for (const ep of providers) {
        try {
            yoloSession = await ort.InferenceSession.create('./models/yolo-face.onnx', {
                executionProviders: [ep],
                graphOptimizationLevel: 'all',
            });
            executionProvider = ep;
            console.log(`YOLO loaded with ${ep} provider`);
            DOM.yoloStatus.textContent = ep.toUpperCase();
            return;
        } catch (e) {
            console.warn(`YOLO failed with ${ep}:`, e.message);
        }
    }
    throw new Error('Failed to load YOLO model with any execution provider');
}

async function loadFaceNet(modelName) {
    const modelPath = `./models/${modelName}.onnx`;
    const providers = ['webgpu', 'webgl', 'wasm'];
    for (const ep of providers) {
        try {
            // Disable optimizations for WebGPU to dodge graph fusion bugs
            faceSession = await ort.InferenceSession.create(modelPath, {
                executionProviders: [ep],
                graphOptimizationLevel: ep === 'webgpu' ? 'none' : 'all',
            });
            console.log(`EdgeFace (${modelName}) loaded with ${ep} provider`);
            logDebug(`EdgeFace ${modelName} on ${ep.toUpperCase()}`);
            DOM.faceNetStatus.textContent = ep.toUpperCase();

            // If EdgeFace falls back to WASM, update the overall status
            if (ep !== 'webgpu' && executionProvider === 'webgpu') {
                executionProvider = 'wasm';
            }
            return;
        } catch (e) {
            console.warn(`EdgeFace (${modelName}) failed with ${ep}:`, e.message);
            logDebug(`EdgeFace fail ${modelName} on ${ep}: ${e.message}`);
        }
    }
    throw new Error(`Failed to load EdgeFace model ${modelName}`);
}

// ─── YOLO Preprocessing ───────────────────────────────────────────────────

function preprocessYOLO(imageData, srcW, srcH) {
    // Letterbox resize to 640x640 preserving aspect ratio
    const modelW = 640, modelH = 640;
    const scale = Math.min(modelW / srcW, modelH / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = (modelW - newW) / 2;
    const padY = (modelH - newH) / 2;

    // Draw letterboxed image
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = modelW;
    tmpCanvas.height = modelH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.fillStyle = '#808080';
    tmpCtx.fillRect(0, 0, modelW, modelH);
    tmpCtx.drawImage(extractCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const pixels = tmpCtx.getImageData(0, 0, modelW, modelH).data;

    // HWC RGBA -> NCHW float32 normalized [0,1]
    const float32 = new Float32Array(3 * modelW * modelH);
    for (let i = 0; i < modelW * modelH; i++) {
        float32[i] = pixels[i * 4] / 255.0; // R
        float32[i + modelW * modelH] = pixels[i * 4 + 1] / 255.0; // G
        float32[i + 2 * modelW * modelH] = pixels[i * 4 + 2] / 255.0; // B
    }

    return { tensor: new ort.Tensor('float32', float32, [1, 3, modelH, modelW]), scale, padX, padY };
}

// ─── YOLO Postprocessing ──────────────────────────────────────────────────

function postprocessYOLO(output, scale, padX, padY, srcW, srcH, confThreshold = 0.55) {
    // output shape: [1, D, 8400] where D = 4 + numClasses
    // Face-only model: D=5 (4 box + 1 face conf)
    // COCO model: D=84 (4 box + 80 class confs)
    const data = output.data;
    const outputDims = output.dims[1]; // auto-detect from model output
    const numPreds = output.dims[2];
    const numClasses = outputDims - 4;

    const boxes = [];

    for (let i = 0; i < numPreds; i++) {
        const cx = data[0 * numPreds + i];
        const cy = data[1 * numPreds + i];
        const w = data[2 * numPreds + i];
        const h = data[3 * numPreds + i];

        // Find best class confidence
        let maxConf = 0;
        for (let c = 0; c < numClasses; c++) {
            const conf = data[(4 + c) * numPreds + i];
            if (conf > maxConf) maxConf = conf;
        }

        if (maxConf < confThreshold) continue;

        // Convert from letterbox coords to original image coords
        const x1 = ((cx - w / 2) - padX) / scale;
        const y1 = ((cy - h / 2) - padY) / scale;
        const x2 = ((cx + w / 2) - padX) / scale;
        const y2 = ((cy + h / 2) - padY) / scale;

        // Clamp to source dimensions
        const bx1 = Math.max(0, Math.min(x1, srcW));
        const by1 = Math.max(0, Math.min(y1, srcH));
        const bx2 = Math.max(0, Math.min(x2, srcW));
        const by2 = Math.max(0, Math.min(y2, srcH));

        if (bx2 - bx1 < 25 || by2 - by1 < 25) continue;

        boxes.push({ x1: bx1, y1: by1, x2: bx2, y2: by2, conf: maxConf });
    }

    // NMS
    boxes.sort((a, b) => b.conf - a.conf);
    const keep = [];
    const suppressed = new Set();
    for (let i = 0; i < boxes.length; i++) {
        if (suppressed.has(i)) continue;
        keep.push(boxes[i]);
        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed.has(j)) continue;
            if (computeIOU(boxes[i], boxes[j]) > 0.4) suppressed.add(j);
        }
    }

    return keep;
}

function computeIOU(a, b) {
    const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
    const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const bA = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aA + bA - inter + 1e-6);
}

// ─── Face Embedding ───────────────────────────────────────────────────────

async function getFaceEmbedding(faceImageData) {
    // Resize face crop to 112x112. Input: ImageData from canvas
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 112;
    tmpCanvas.height = 112;
    const tmpCtx = tmpCanvas.getContext('2d');

    // We need to draw the crop to a temp canvas at 112x112 without stretching
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = faceImageData.width;
    srcCanvas.height = faceImageData.height;
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.putImageData(faceImageData, 0, 0);

    const targetDim = 112;
    const scale = Math.min(targetDim / faceImageData.width, targetDim / faceImageData.height);
    const newW = faceImageData.width * scale;
    const newH = faceImageData.height * scale;
    const padX = (targetDim - newW) / 2;
    const padY = (targetDim - newH) / 2;

    tmpCtx.fillStyle = '#000000'; // Black padding for letterbox
    tmpCtx.fillRect(0, 0, targetDim, targetDim);
    tmpCtx.drawImage(srcCanvas, 0, 0, faceImageData.width, faceImageData.height, padX, padY, newW, newH);

    const pixels = tmpCtx.getImageData(0, 0, 112, 112).data;

    // InsightFace models (like buffalo_l / w600k_r50) strictly require:
    // 1. RGB channel order
    // 2. Normalization: (value - 127.5) / 127.5
    // Canvas getImageData() returns RGBA where index 0=R, 1=G, 2=B.
    const float32 = new Float32Array(3 * 112 * 112);
    for (let i = 0; i < 112 * 112; i++) {
        float32[i] = (pixels[i * 4] - 127.5) / 127.5;         // R → ch0
        float32[i + 112 * 112] = (pixels[i * 4 + 1] - 127.5) / 127.5; // G → ch1
        float32[i + 2 * 112 * 112] = (pixels[i * 4 + 2] - 127.5) / 127.5; // B → ch2
    }

    const inputTensor = new ort.Tensor('float32', float32, [1, 3, 112, 112]);
    const inputName = faceSession.inputNames[0];
    const results = await faceSession.run({ [inputName]: inputTensor });
    const outputName = faceSession.outputNames[0];
    const rawEmbed = Array.from(results[outputName].data);

    // ── Diagnostic: log embedding health once every 60 calls ──────────────────
    if (!getFaceEmbedding._callCount) getFaceEmbedding._callCount = 0;
    if (++getFaceEmbedding._callCount % 60 === 1) {
        const rawNorm = Math.sqrt(rawEmbed.reduce((s, v) => s + v * v, 0));
        const rawMin = Math.min(...rawEmbed.slice(0, 64));
        const rawMax = Math.max(...rawEmbed.slice(0, 64));
        const variance = rawEmbed.slice(0, 64).reduce((s, v) => s + v * v, 0) / 64;
        console.log(`[FaceEmbed] norm=${rawNorm.toFixed(4)} min=${rawMin.toFixed(4)} max=${rawMax.toFixed(4)} var=${variance.toFixed(6)}`);
        logDebug(`[FaceEmbed] norm=${rawNorm.toFixed(3)} var=${variance.toFixed(5)} — if norm~0 model is broken`);
    }

    // L2 normalize to unit sphere
    const norm = Math.sqrt(rawEmbed.reduce((s, v) => s + v * v, 0));
    return rawEmbed.map(v => v / (norm + 1e-10));
}

// ─── Simple IOU Tracker ───────────────────────────────────────────────────

function updateTracker(detections) {
    // Age out all tracks
    for (const t of tracks) t.age++;

    // Match detections to existing tracks using IOU
    const matched = new Set();
    const matchedTracks = new Set();

    for (let di = 0; di < detections.length; di++) {
        let bestIOU = 0, bestTi = -1;
        for (let ti = 0; ti < tracks.length; ti++) {
            if (matchedTracks.has(ti)) continue;
            const iou = computeIOU(detections[di], tracks[ti].box);
            if (iou > bestIOU) { bestIOU = iou; bestTi = ti; }
        }
        if (bestIOU > IOU_THRESHOLD && bestTi >= 0) {
            // Apply EMA smoothing to the bounding box to reduce jitter during expression changes
            const alpha = 0.4;
            const oldBox = tracks[bestTi].box;
            const newBox = detections[di];
            const smoothBox = {
                x1: oldBox.x1 * (1 - alpha) + newBox.x1 * alpha,
                y1: oldBox.y1 * (1 - alpha) + newBox.y1 * alpha,
                x2: oldBox.x2 * (1 - alpha) + newBox.x2 * alpha,
                y2: oldBox.y2 * (1 - alpha) + newBox.y2 * alpha,
                conf: newBox.conf,
                track_id: tracks[bestTi].id
            };

            tracks[bestTi].box = smoothBox;
            tracks[bestTi].age = 0;
            detections[di] = smoothBox; // pass smooth box forward to embedding logic
            matched.add(di);
            matchedTracks.add(bestTi);
        }
    }

    // Create new tracks for unmatched detections
    for (let di = 0; di < detections.length; di++) {
        if (matched.has(di)) continue;
        const id = nextTrackId++;
        tracks.push({ id, box: detections[di], age: 0 });
        detections[di].track_id = id;
    }

    // Remove old tracks
    tracks = tracks.filter(t => t.age < MAX_TRACK_AGE);

    return detections;
}

// ─── Network / Session ─────────────────────────────────────────────────────

async function initNetwork() {
    const data = await apiFetch(`/api/session/new?${modelQueryParam()}`);
    currentSessionId = data.session_id;
    DOM.activeSession.textContent = currentSessionId;

    DOM.btnNew.addEventListener('click', async () => {
        try {
            const data = await apiFetch(`/api/session/new?${modelQueryParam()}`);
            currentSessionId = data.session_id;
            DOM.activeSession.textContent = currentSessionId;
            resetTrackingState();
        } catch (e) {
            alert(`Failed to create session: ${e.message}`);
        }
    });

    DOM.btnLoad.addEventListener('click', async () => {
        const id = DOM.inputSession.value.trim();
        if (!id) return alert('Enter a Session ID first');
        try {
            const data = await apiFetch(`/api/session/load/${encodeURIComponent(id)}?${modelQueryParam()}`);
            currentSessionId = id;
            DOM.activeSession.textContent = currentSessionId;
            resetTrackingState();
            alert(`Session loaded (${data.model_name}).`);
        } catch (e) {
            alert(`Failed to load session: ${e.message}`);
        }
    });

    DOM.btnDelete.addEventListener('click', async () => {
        if (!currentSessionId) return alert('No active session to delete.');
        if (!confirm('Are you sure you want to permanently delete this session? This action cannot be undone.')) return;

        try {
            await apiFetch(`/api/session/${encodeURIComponent(currentSessionId)}`, { method: 'DELETE' });
            alert('Session deleted successfully.');
            currentSessionId = null;
            DOM.activeSession.textContent = 'None';
            resetTrackingState();
        } catch (e) {
            console.error(e);
            alert(`Error deleting session: ${e.message}`);
        }
    });

    DOM.btnPause.addEventListener('click', () => {
        setPaused(!isPaused);
    });

    DOM.btnUseClosest.addEventListener('click', () => {
        const face = closestTrackId !== null ? trackedFaces.get(closestTrackId) : null;
        if (face && face.reid !== null) {
            DOM.inputReid.value = face.reid;
        } else {
            alert('No face is currently being tracked.');
        }
    });

    DOM.btnUpdateName.addEventListener('click', async () => {
        const name = DOM.inputName.value.trim();
        const reidVal = DOM.inputReid.value.trim();

        const face = closestTrackId !== null ? trackedFaces.get(closestTrackId) : null;
        const reid = reidVal !== '' ? parseInt(reidVal) : (face ? face.reid : null);

        if (!name) return alert('Please enter a name.');
        if (!currentSessionId) return alert('No active session.');
        if (reid === null || reid === undefined || isNaN(reid)) return alert('Enter a ReID or wait for a face to be tracked.');
        try {
            await apiFetch('/api/face/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, model_name: currentModelName, reid, name }),
            });
            // Update name in all tracked faces with this reid
            for (const [, f] of trackedFaces) {
                if (f.reid === reid) f.name = name;
            }
            updateSidePanel();
            DOM.inputName.value = '';
            DOM.inputReid.value = '';
        } catch (e) {
            alert(`Update failed: ${e.message}`);
        }
    });

    DOM.btnUploadTrigger.addEventListener('click', () => {
        DOM.imageUpload.click();
    });

    DOM.imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!currentSessionId) {
            alert("Please start or load a session first!");
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => processUploadedImage(img);
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function updateSidePanel() {
    const face = closestTrackId !== null ? trackedFaces.get(closestTrackId) : null;
    if (face) {
        DOM.trackId.textContent = face.track_id;
        let name = face.name || 'Unknown';
        if (face.distance !== null && face.distance !== undefined) {
            name += ` [D: ${face.distance.toFixed(2)}]`;
        }
        DOM.trackName.textContent = name;
    } else {
        DOM.trackId.textContent = '-';
        DOM.trackName.textContent = 'None';
    }
}

// ─── Camera ────────────────────────────────────────────────────────────────

async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    DOM.webcam.srcObject = stream;
    DOM.webcam.style.display = 'none';

    await new Promise((resolve) => {
        if (DOM.webcam.readyState >= 2) resolve();
        else DOM.webcam.addEventListener('loadeddata', resolve, { once: true });
        DOM.webcam.play().catch(() => { });
    });

    extractCanvas.width = 640;
    extractCanvas.height = 480;
    handleResize();
    startRenderLoop();
}

// ─── Main Render + Inference Loop ──────────────────────────────────────────

let prevTime = performance.now();
let frameCount = 0;
let inferenceFrame = 0;
let isInferring = false; // Prevent overlapping inference calls

function startRenderLoop() {
    async function frame() {
        // FPS
        const now = performance.now();
        frameCount++;
        if (now - prevTime >= 1000) {
            DOM.fps.textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            prevTime = now;
        }

        // Draw camera feed
        ctx.drawImage(DOM.webcam, 0, 0, DOM.canvas.width, DOM.canvas.height);

        if (isPaused) {
            drawPauseOverlay();
            requestAnimationFrame(frame);
            return;
        }

        // Draw bounding boxes + labels for ALL tracked faces
        const scaleX = DOM.canvas.width / 640;
        const scaleY = DOM.canvas.height / 480;

        for (const [tid, face] of trackedFaces) {
            if (!face.box) continue;
            const { x1, y1, x2, y2 } = face.box;
            const bx = x1 * scaleX, by = y1 * scaleY;
            const bw = (x2 - x1) * scaleX, bh = (y2 - y1) * scaleY;

            // Calculate the actual bounding box sent to FaceNet (1.5x expansion, shifted UP by 15% to align eyes)
            let rawSide = Math.max(bw, bh);
            let side = rawSide * 1.1;
            const shiftY = side * 0.05;
            const cx = bx + bw / 2;
            const cy = by + bh / 2;

            const eBx = Math.max(0, cx - side / 2);
            const eBy = Math.max(0, (cy - shiftY) - side / 2);
            const maxW = DOM.canvas.width - eBx;
            const maxH = DOM.canvas.height - eBy;
            const eBw = Math.min(side, maxW);
            const eBh = Math.min(side, maxH);

            const isClosest = tid === closestTrackId;
            const color = isClosest ? '#00ffff' : '#a855f7'; // cyan for closest, purple for others

            // Neon bounding box (now representing the full FaceNet extraction)
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = isClosest ? 3 : 2;
            ctx.shadowColor = color;
            ctx.shadowBlur = isClosest ? 12 : 6;
            ctx.strokeRect(eBx, eBy, eBw, eBh);
            ctx.restore();

            // Label: "Name [ReID:X TID:Y]"
            const dispName = face.name || 'Unknown';
            const reidStr = face.reid !== null ? face.reid : '?';
            const tidStr = face.track_id !== null ? face.track_id : '?';
            const hasDist = face.distance !== null && face.distance !== undefined;
            const distStr = hasDist ? ` D:${face.distance.toFixed(2)}` : '';
            const statusTag = face.isProcessing ? ' [⏳ Processing]' : '';
            const label = `${dispName}${distStr} [R:${reidStr} T:${tidStr}]${statusTag}`;

            ctx.font = 'bold 13px Inter, sans-serif';
            const textW = ctx.measureText(label).width + 16;
            const labelH = 24;
            let labelY = eBy - labelH - 4;
            if (labelY < 0) labelY = eBy + 4;

            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(eBx, labelY, textW, labelH);
            ctx.fillStyle = color;
            ctx.fillRect(eBx, labelY, 3, labelH);
            ctx.fillText(label, eBx + 10, labelY + 16);
        }

        // Run inference every 3rd frame to keep render smooth
        inferenceFrame++;
        if (inferenceFrame % 3 === 0 && !isInferring) {
            isInferring = true;
            runInference().finally(() => { isInferring = false; });
        }

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

async function runInference() {
    try {
        if (isPaused) return;

        // Capture current frame
        extractCtx.drawImage(DOM.webcam, 0, 0, 640, 480);

        // Run YOLO
        const { tensor, scale, padX, padY } = preprocessYOLO(null, 640, 480);
        const yoloInputName = yoloSession.inputNames[0];
        const yoloResults = await yoloSession.run({ [yoloInputName]: tensor });
        if (isPaused) return;

        const yoloOutputName = yoloSession.outputNames[0];
        const output = yoloResults[yoloOutputName];

        // Postprocess: get face bounding boxes
        let detections = postprocessYOLO(output, scale, padX, padY, 640, 480, 0.55);

        // Track faces
        detections = updateTracker(detections);

        // Update tracked faces from detections
        const activeTrackIds = new Set();
        let closestDet = null;
        let maxArea = 0;

        for (const det of detections) {
            const tid = det.track_id;
            activeTrackIds.add(tid);

            // Area for finding closest
            const area = (det.x2 - det.x1) * (det.y2 - det.y1);
            if (area > maxArea) { maxArea = area; closestDet = det; }

            // Get or create face entry
            if (!trackedFaces.has(tid)) {
                trackedFaces.set(tid, { box: det, reid: null, name: null, track_id: tid, embeddingAge: 0, isProcessing: false });
            } else {
                trackedFaces.get(tid).box = det;
            }
        }

        closestTrackId = closestDet ? closestDet.track_id : null;

        // Remove faces no longer tracked
        for (const [tid] of trackedFaces) {
            if (!activeTrackIds.has(tid)) trackedFaces.delete(tid);
        }

        // Compute embeddings for each tracked face (throttled per-track)
        for (const [tid, face] of trackedFaces) {
            if (isPaused) return;

            // Throttle queries to avoid saturating GPU/Inference
            if (face.embeddingAge % EMBEDDING_INTERVAL !== 0) {
                face.embeddingAge++;
                continue;
            }
            face.embeddingAge++;

            const cx = (face.box.x1 + face.box.x2) / 2;
            const cy = (face.box.y1 + face.box.y2) / 2;
            let bw = face.box.x2 - face.box.x1;
            let bh = face.box.y2 - face.box.y1;

            // Expand strictly to a square by 1.1x (gives some facial context without excessive background)
            let side = Math.max(bw, bh) * 1.1;

            // Heuristic Alignment: InsightFace expects eyes to be in the upper half.
            const shiftY = side * 0.05; // Shift crop window UP by 5%

            const x1 = cx - side / 2;
            const y1 = (cy - shiftY) - side / 2;
            const w = side;
            const h = side;

            const cx1 = Math.floor(x1);
            const cy1 = Math.floor(y1);
            const cw = Math.ceil(w);
            const ch = Math.ceil(h);

            if (cw > 20 && ch > 20) {
                try {
                    face.isProcessing = true;
                    // We extract using the EXACT intended square, even if cx1/cy1 is negative
                    // drawImage handles out-of-bounds by treating it as transparent black
                    const tmpSquare = document.createElement('canvas');
                    tmpSquare.width = cw;
                    tmpSquare.height = ch;
                    const tmpSqCtx = tmpSquare.getContext('2d');

                    // Fill with black so out-of-bounds areas aren't transparent/random
                    tmpSqCtx.fillStyle = '#000000';
                    tmpSqCtx.fillRect(0, 0, cw, ch);

                    // Draw what we can from the webcam
                    tmpSqCtx.drawImage(extractCanvas, cx1, cy1, cw, ch, 0, 0, cw, ch);

                    const faceImg = tmpSqCtx.getImageData(0, 0, cw, ch);

                    // Allow UI to render the [⏳ Processing] tag before the thread is blocked
                    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

                    const embedding = await getFaceEmbedding(faceImg);

                    const data = await apiFetch('/api/face/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            embedding,
                            session_id: currentSessionId,
                            model_name: currentModelName,
                            track_id: tid,
                            known_reid: face.reid,
                            allow_new_identity: true,
                            allow_profile_expansion: false
                        }),
                    });
                    if (trackedFaces.has(tid)) {
                        face.reid = data.reid;
                        face.name = data.name;
                        face.distance = data.distance;
                        logDebug(`ReID Map | Track ID ${tid} -> ReID ${face.reid} (${face.name})`);
                    }
                } catch (err) {
                    console.warn(`Query ${tid} fail:`, err);
                    logDebug(`Embedding failed for track ${tid}: ${err.message}`);
                } finally {
                    if (trackedFaces.has(tid)) {
                        trackedFaces.get(tid).isProcessing = false;
                    }
                }
            }
        }

        updateSidePanel();

    } catch (e) {
        console.error('Inference error:', e);
    }
}

// ─── Resize ────────────────────────────────────────────────────────────────

function handleResize() {
    const rect = DOM.canvas.parentElement.getBoundingClientRect();
    DOM.canvas.width = rect.width;
    DOM.canvas.height = rect.height;
}
window.addEventListener('resize', handleResize);

// ─── Model Hot-Swap ────────────────────────────────────────────────────────

DOM.modelSelector.addEventListener('change', async () => {
    const newModel = DOM.modelSelector.value;
    if (newModel === currentModelName) return;

    logDebug(`Switching model: ${currentModelName} → ${newModel}`);
    DOM.faceNetStatus.textContent = 'Loading...';

    try {
        // Release old session
        if (faceSession) {
            await faceSession.release();
            faceSession = null;
        }

        currentModelName = newModel;
        await loadFaceNet(currentModelName);

        // New model = new embedding space, must reset session
        const data = await apiFetch(`/api/session/new?${modelQueryParam()}`);
        currentSessionId = data.session_id;
        DOM.activeSession.textContent = currentSessionId;
        resetTrackingState();

        logDebug(`Model switched to ${currentModelName}. New session: ${currentSessionId}`);
    } catch (e) {
        console.error('Model switch failed:', e);
        logDebug(`Model switch error: ${e.message}`);
        DOM.faceNetStatus.textContent = 'ERROR';
    }
});

// ─── Start ─────────────────────────────────────────────────────────────────

// ─── Image Upload Identification Pipeline ────────────────────────────────────

async function processUploadedImage(img) {
    if (!yoloSession || !faceSession) {
        alert("Models are still loading!");
        return;
    }

    try {
        DOM.status.textContent = "Processing Image...";

        // 1. Draw to canvas (scale down if too large, maintain AR)
        const maxWidth = 1024;
        const scale = img.width > maxWidth ? maxWidth / img.width : 1;
        const cw = Math.round(img.width * scale);
        const ch = Math.round(img.height * scale);

        const c = document.createElement('canvas');
        c.width = cw;
        c.height = ch;
        const cx = c.getContext('2d');
        cx.drawImage(img, 0, 0, cw, ch);

        // 2. Preprocess YOLO
        const { tensor, scale: yoloScale, padX, padY } = preprocessYOLO_custom(c, cw, ch);

        // 3. Run YOLO
        const yoloInputName = yoloSession.inputNames[0];
        const results = await yoloSession.run({ [yoloInputName]: tensor });
        const yoloOutputName = yoloSession.outputNames[0];
        const output = results[yoloOutputName];

        // 4. Postprocess YOLO
        const detections = postprocessYOLO(output, yoloScale, padX, padY, cw, ch, 0.45);

        let initialBox = null;

        if (detections.length > 0) {
            // Get largest face
            let largestFace = detections[0];
            let maxArea = 0;
            for (const det of detections) {
                const area = (det.x2 - det.x1) * (det.y2 - det.y1);
                if (area > maxArea) {
                    maxArea = area;
                    largestFace = det;
                }
            }
            initialBox = {
                x1: largestFace.x1 / scale,
                y1: largestFace.y1 / scale,
                x2: largestFace.x2 / scale,
                y2: largestFace.y2 / scale
            };
        }

        // 5. Open Modal instead of automatic embedding
        openCropModal(img, initialBox);

    } catch (e) {
        console.error(e);
        alert("Error processing image: " + e.message);
        DOM.status.textContent = `Running (${executionProvider.toUpperCase()})`;
    } finally {
        DOM.imageUpload.value = ''; // reset file input
    }
}

// ─── Crop Modal Logic ──────────────────────────────────────────────────────

function openCropModal(img, faceBox) {
    cropState.active = true;
    cropState.uploadedImage = img;
    DOM.cropNameInput.value = '';

    // Show modal
    DOM.cropModal.classList.remove('hidden');

    const initCropMath = () => {
        const containerRect = DOM.cropContainer.getBoundingClientRect();
        const imgRect = DOM.cropImage.getBoundingClientRect();

        // Save image rendered boundaries relative to container
        cropState.imgX = imgRect.left - containerRect.left;
        cropState.imgY = imgRect.top - containerRect.top;
        cropState.imgW = imgRect.width;
        cropState.imgH = imgRect.height;

        const scaleX = cropState.imgW / img.width;
        const scaleY = cropState.imgH / img.height;

        if (faceBox) {
            // Convert YOLO face box to rendered modal coords
            const fCx = (faceBox.x1 + faceBox.x2) / 2;
            const fCy = (faceBox.y1 + faceBox.y2) / 2;
            let bw = faceBox.x2 - faceBox.x1;
            let bh = faceBox.y2 - faceBox.y1;

            let side = Math.max(bw, bh) * 1.1; // Match 1.1x scaling of webcam feed exactly
            const shiftY = side * 0.05;

            cropState.boxW = side * scaleX;
            cropState.boxH = side * scaleY;
            cropState.boxX = cropState.imgX + (fCx * scaleX) - (cropState.boxW / 2);
            cropState.boxY = cropState.imgY + ((fCy - shiftY) * scaleY) - (cropState.boxH / 2);
        } else {
            // Default center crop
            cropState.boxW = Math.min(cropState.imgW, cropState.imgH) * 0.5;
            cropState.boxH = cropState.boxW;
            cropState.boxX = cropState.imgX + (cropState.imgW / 2) - (cropState.boxW / 2);
            cropState.boxY = cropState.imgY + (cropState.imgH / 2) - (cropState.boxH / 2);
        }

        clampCropBox();
        updateCropBoxUI();
    };

    // Ensure image is fully rendered before calculating sizes
    DOM.cropImage.src = img.src;
    if (DOM.cropImage.complete) {
        setTimeout(initCropMath, 10);
    } else {
        DOM.cropImage.onload = initCropMath;
    }
}

function updateCropBoxUI() {
    DOM.cropBox.style.left = `${cropState.boxX}px`;
    DOM.cropBox.style.top = `${cropState.boxY}px`;
    DOM.cropBox.style.width = `${cropState.boxW}px`;
    DOM.cropBox.style.height = `${cropState.boxH}px`;
}

function clampCropBox() {
    const maxSide = Math.min(cropState.imgW, cropState.imgH);
    if (cropState.boxW > maxSide) cropState.boxW = maxSide;
    if (cropState.boxH > maxSide) cropState.boxH = maxSide;

    if (cropState.boxW < 40) cropState.boxW = 40;

    // Force square for EdgeFace
    cropState.boxH = cropState.boxW;

    // Prevent dragging out of bounds
    if (cropState.boxX < cropState.imgX) cropState.boxX = cropState.imgX;
    if (cropState.boxY < cropState.imgY) cropState.boxY = cropState.imgY;
    if (cropState.boxX + cropState.boxW > cropState.imgX + cropState.imgW) {
        cropState.boxX = cropState.imgX + cropState.imgW - cropState.boxW;
    }
    if (cropState.boxY + cropState.boxH > cropState.imgY + cropState.imgH) {
        cropState.boxY = cropState.imgY + cropState.imgH - cropState.boxH;
    }
}

// Mouse/Touch Events for Crop Box
DOM.cropContainer.addEventListener('mousedown', startCropInteraction);
window.addEventListener('mousemove', moveCropInteraction);
window.addEventListener('mouseup', endCropInteraction);

DOM.cropContainer.addEventListener('touchstart', startCropInteraction, { passive: false });
window.addEventListener('touchmove', moveCropInteraction, { passive: false });
window.addEventListener('touchend', endCropInteraction);

function getEventPos(e) {
    if (e.touches && e.touches.length > 0) {
        return { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    return { x: e.clientX, y: e.clientY };
}

function startCropInteraction(e) {
    if (!cropState.active) return;

    const target = e.target;
    const pos = getEventPos(e);

    if (target.classList.contains('crop-handle')) {
        cropState.isResizing = true;
        cropState.resizeHandle = target.className.split(' ').find(c => ['nw', 'ne', 'sw', 'se'].includes(c));
        e.preventDefault();
    } else if (target.closest('#cropBox')) {
        cropState.isDragging = true;
        e.preventDefault();
    } else {
        return;
    }

    cropState.startX = pos.x;
    cropState.startY = pos.y;
    cropState.initialBoxX = cropState.boxX;
    cropState.initialBoxY = cropState.boxY;
    cropState.initialBoxW = cropState.boxW;
    cropState.initialBoxH = cropState.boxH;
}

function moveCropInteraction(e) {
    if (!cropState.isDragging && !cropState.isResizing) return;
    e.preventDefault(); // Stop scrolling while dragging

    const pos = getEventPos(e);
    const dx = pos.x - cropState.startX;
    const dy = pos.y - cropState.startY;

    if (cropState.isDragging) {
        cropState.boxX = cropState.initialBoxX + dx;
        cropState.boxY = cropState.initialBoxY + dy;
    } else if (cropState.isResizing) {
        // Uniform diagonal scaling from opposite corner
        let delta;
        if (cropState.resizeHandle === 'se') {
            delta = (dx + dy) / 2;
            cropState.boxW = cropState.initialBoxW + delta;
        } else if (cropState.resizeHandle === 'nw') {
            delta = -(dx + dy) / 2;
            cropState.boxW = cropState.initialBoxW + delta;
            cropState.boxX = cropState.initialBoxX - delta;
            cropState.boxY = cropState.initialBoxY - delta;
        } else if (cropState.resizeHandle === 'ne') {
            delta = (dx - dy) / 2;
            cropState.boxW = cropState.initialBoxW + delta;
            cropState.boxY = cropState.initialBoxY - delta;
        } else if (cropState.resizeHandle === 'sw') {
            delta = (-dx + dy) / 2;
            cropState.boxW = cropState.initialBoxW + delta;
            cropState.boxX = cropState.initialBoxX - delta;
        }
    }

    clampCropBox();
    updateCropBoxUI();
}

function endCropInteraction() {
    cropState.isDragging = false;
    cropState.isResizing = false;
    cropState.resizeHandle = null;
}

DOM.btnCancelCrop.addEventListener('click', () => {
    DOM.cropModal.classList.add('hidden');
    cropState.active = false;
    DOM.status.textContent = `Running (${executionProvider.toUpperCase()})`;
});

DOM.btnSaveIdentity.addEventListener('click', async () => {
    const name = DOM.cropNameInput.value.trim();
    if (!name) return alert('Please enter a name for this identity.');

    DOM.btnSaveIdentity.disabled = true;
    DOM.btnSaveIdentity.textContent = 'Saving...';

    try {
        // Extract crop region
        const scaleX = cropState.uploadedImage.width / cropState.imgW;
        const scaleY = cropState.uploadedImage.height / cropState.imgH;

        const srcX = (cropState.boxX - cropState.imgX) * scaleX;
        const srcY = (cropState.boxY - cropState.imgY) * scaleY;
        const srcW = cropState.boxW * scaleX;
        const srcH = cropState.boxH * scaleY;

        // Draw to FaceNet canvas directly (112x112)
        const faceCanvas = document.createElement('canvas');
        faceCanvas.width = 112;
        faceCanvas.height = 112;
        const faceCtx = faceCanvas.getContext('2d');
        faceCtx.fillStyle = '#000000';
        faceCtx.fillRect(0, 0, 112, 112);

        faceCtx.drawImage(
            cropState.uploadedImage,
            srcX, srcY, srcW, srcH,
            0, 0, 112, 112
        );

        const faceImgData = faceCtx.getImageData(0, 0, 112, 112);
        const embedding = await getFaceEmbedding(faceImgData);

        // Save to DB
        const data = await apiFetch('/api/face/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embedding,
                session_id: currentSessionId,
                model_name: currentModelName,
                track_id: "upload",
                known_reid: null,
                allow_new_identity: true,
                allow_profile_expansion: false
            }),
        });

        // Apply Name
        if (data.reid !== null && data.reid !== undefined) {
            const updateData = await apiFetch('/api/face/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, model_name: currentModelName, reid: data.reid, name: name }),
            });
            if (updateData.status === 'success') {
                alert(`Identity Verified & Saved!\nName: ${name}\nReID: ${data.reid}`);

                // Update live side panel if this reid is currently on screen
                for (const [, f] of trackedFaces) {
                    if (f.reid === data.reid) f.name = name;
                }
                updateSidePanel();
            }
        }

        DOM.cropModal.classList.add('hidden');
        cropState.active = false;
    } catch (e) {
        console.error(e);
        alert('Failed to save identity: ' + e.message);
    } finally {
        DOM.btnSaveIdentity.disabled = false;
        DOM.btnSaveIdentity.textContent = 'Save Identity';
        DOM.status.textContent = `Running (${executionProvider.toUpperCase()})`;
    }
});

// Needed because extractCanvas is hardcoded in original preprocessYOLO
function preprocessYOLO_custom(sourceCanvas, srcW, srcH) {
    const modelW = 640, modelH = 640;
    const scale = Math.min(modelW / srcW, modelH / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = (modelW - newW) / 2;
    const padY = (modelH - newH) / 2;

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = modelW;
    tmpCanvas.height = modelH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.fillStyle = '#808080';
    tmpCtx.fillRect(0, 0, modelW, modelH);
    tmpCtx.drawImage(sourceCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const pixels = tmpCtx.getImageData(0, 0, modelW, modelH).data;

    const float32 = new Float32Array(3 * modelW * modelH);
    for (let i = 0; i < modelW * modelH; i++) {
        float32[i] = pixels[i * 4] / 255.0; // R
        float32[i + modelW * modelH] = pixels[i * 4 + 1] / 255.0; // G
        float32[i + 2 * modelW * modelH] = pixels[i * 4 + 2] / 255.0; // B
    }

    return { tensor: new ort.Tensor('float32', float32, [1, 3, modelH, modelW]), scale, padX, padY };
}

boot();
