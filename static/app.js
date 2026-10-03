// =========================================================
// FaceRecog Enterprise Vision Engine — UI Client v6
// Complete rewrite: MediaPipe + session timers + demo polish
// =========================================================

// ── DOM refs ──────────────────────────────────────────────
const video   = document.getElementById('video');
const canvas  = document.getElementById('overlay-canvas');
const ctx     = canvas.getContext('2d');

// ── State ─────────────────────────────────────────────────
let ws            = null;
let isStreaming   = false;
let currentMode   = 'recognize';
let frameCount    = 0;
let lastFpsTime   = Date.now();
let alertShowing  = false;

const SERVER_W = 480;
const SERVER_H = 360;

// Server-side face positions (for name labels)
let _targetFaces = [];

// ── Session timer per user (2hr limit) ────────────────────
const SESSION_LIMIT_MS = 2 * 60 * 60 * 1000; // 2 hours
const _userSessions = {}; // { name: { startMs, warningShown } }

function _updateSessionTimers() {
  const now = Date.now();
  Object.keys(_userSessions).forEach(name => {
    const s = _userSessions[name];
    const elapsed = now - s.startMs;
    const remaining = SESSION_LIMIT_MS - elapsed;
    if (remaining <= 0 && !s.limitShown) {
      s.limitShown = true;
      showToast(`⏰ ${name}'s 2-hour session limit has been reached!`, 'danger');
    } else if (remaining <= 5 * 60 * 1000 && !s.warningShown) {
      // 5 min warning
      s.warningShown = true;
      showToast(`⚠ ${name} has 5 minutes left in today's session.`, 'info');
    }
  });
}
setInterval(_updateSessionTimers, 10000);

// ── Welcome toast logic ───────────────────────────────────
const _welcomedNames = new Set();
function _checkWelcome(faces) {
  faces.forEach(f => {
    if (f.name && f.name !== 'Unknown' && f.name !== 'Scanning…' && f.name !== 'Person') {
      if (!_welcomedNames.has(f.name)) {
        _welcomedNames.add(f.name);
        showToast(`👋 Welcome, ${f.name}!`, 'success');
        // Start session timer
        if (!_userSessions[f.name]) {
          _userSessions[f.name] = { startMs: Date.now(), warningShown: false, limitShown: false };
        }
      }
    }
  });
}

// =========================================================
// ── MediaPipe FaceMesh (Zero-Lag Tracking) ───────────────
// Replaces face-api.js entirely. MediaPipe runs true ~60fps
// using the browser GPU with no async queuing lag.
// =========================================================

let _mpFaces = []; // Current local MediaPipe detections
let _mpReady = false;
let _faceMesh = null;

async function initMediaPipe() {
  try {
    if (typeof FaceMesh === 'undefined') {
      console.warn('[MediaPipe] Not loaded — falling back to server tracking.');
      return;
    }
    _faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`
    });
    _faceMesh.setOptions({
      maxNumFaces: 5,
      refineLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.4
    });
    _faceMesh.onResults(onMpResults);
    await _faceMesh.initialize();
    _mpReady = true;
    console.log('[MediaPipe] FaceMesh ready — zero-lag tracking active!');
  } catch (e) {
    console.warn('[MediaPipe] init failed:', e.message);
  }
}

let _mpLoopRunning = false;
async function mpTrackLoop() {
  if (!isStreaming || !_mpReady || !_faceMesh) return;
  try {
    await _faceMesh.send({ image: video });
  } catch(e) {}
  requestAnimationFrame(mpTrackLoop);
}

let _retainFrames = 0;
function onMpResults(results) {
  if (!isStreaming) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const sx = SERVER_W / vw, sy = SERVER_H / vh;
  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    _retainFrames = 0;
    _mpFaces = results.multiFaceLandmarks.map(landmarks => {
      // Compute bounding box from landmarks
      let minX = 1, minY = 1, maxX = 0, maxY = 0;
      landmarks.forEach(l => {
        if (l.x < minX) minX = l.x;
        if (l.y < minY) minY = l.y;
        if (l.x > maxX) maxX = l.x;
        if (l.y > maxY) maxY = l.y;
      });
      const x = minX * vw * sx, y = minY * vh * sy;
      const w = (maxX - minX) * vw * sx, h = (maxY - minY) * vh * sy;

      // Match name from server
      let name = 'Scanning…', confidence = 0;
      const lf = { x, y, w, h };
      if (_targetFaces.length === 1 && _mpFaces.length === 0) {
        name = _targetFaces[0].name;
        confidence = _targetFaces[0].confidence;
      } else {
        let best = null, bestScore = 0;
        _targetFaces.forEach(tf => {
          const s = _iou(lf, tf);
          if (s > 0.01 && s > bestScore) { bestScore = s; best = tf; }
        });
        if (best) { name = best.name; confidence = best.confidence; }
        // If only 1 face total on both sides, assume same
        else if (_targetFaces.length === 1) { name = _targetFaces[0].name; confidence = _targetFaces[0].confidence; }
      }

      return { x, y, w, h, name, confidence, visuallyX: x, visuallyY: y, visuallyW: w, visuallyH: h };
    });
  } else {
    // Tracking retention: hold last known position for ~20 frames before clearing
    _retainFrames++;
    if (_retainFrames > 20) _mpFaces = [];
  }
}

// =========================================================
// ── IOU helper ───────────────────────────────────────────
// =========================================================
function _iou(a, b) {
  const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(a.x+a.w, b.x+b.w), iy2 = Math.min(a.y+a.h, b.y+b.h);
  const inter = Math.max(0, ix2-ix1) * Math.max(0, iy2-iy1);
  const union = a.w*a.h + b.w*b.h - inter;
  return union > 0 ? inter / union : 0;
}

// =========================================================
// ── Render Loop (60 FPS) ─────────────────────────────────
// =========================================================
function masterRenderLoop() {
  if (!isStreaming) return;

  // Use MediaPipe positions (instant) if available, else server (fallback)
  const facesToDraw = (_mpReady && _mpFaces.length > 0) ? _mpFaces : _targetFaces.map(t => ({
    ...t, visuallyX: t.x, visuallyY: t.y, visuallyW: t.w, visuallyH: t.h
  }));

  drawOverlay(facesToDraw);
  requestAnimationFrame(masterRenderLoop);
}

// =========================================================
// ── Theme Management ─────────────────────────────────────
// =========================================================
function initTheme() {
  const saved = localStorage.getItem('fr-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  _syncThemeIcon(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('fr-theme', next);
  _syncThemeIcon(next);
}

function _syncThemeIcon(theme) {
  document.getElementById('theme-icon-moon').style.display = theme === 'dark' ? 'none' : 'block';
  document.getElementById('theme-icon-sun').style.display  = theme === 'dark' ? 'block' : 'none';
}

// =========================================================
// ── Fullscreen ───────────────────────────────────────────
// =========================================================
function toggleFullscreen() {
  const feed = document.querySelector('.feed-area');
  if (!document.fullscreenElement) {
    feed.requestFullscreen().catch(() => {});
    showToast('Fullscreen mode — press Esc to exit.', 'info');
  } else {
    document.exitFullscreen();
  }
}

// =========================================================
// ── Demo Reset ───────────────────────────────────────────
// =========================================================
async function resetDemo() {
  if (!confirm('Reset the demo? This will clear all activity logs and welcome history.')) return;
  try {
    await fetch('/api/attendance/clear', { method: 'DELETE' });
    _welcomedNames.clear();
    Object.keys(_userSessions).forEach(k => delete _userSessions[k]);
    loadAttendance();
    showToast('✅ Demo reset complete! Ready for next presenter.', 'success');
  } catch { showToast('Failed to reset demo.', 'danger'); }
}

// =========================================================
// ── WebSocket ────────────────────────────────────────────
// =========================================================
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/stream`);

  ws.onopen = () => {
    _setWsStatus(true);
    if (isStreaming) sendFrame();
  };
  ws.onclose = () => {
    _setWsStatus(false);
    setTimeout(connectWebSocket, 3000);
  };

  ws.onmessage = ({ data }) => {
    if (!isStreaming) return;
    const payload = JSON.parse(data);

    if (payload.error === 'LIMIT_REACHED') {
      stopCamera();
      showToast('⚠ Daily AI Processing Limit Reached (3/3 hours). Please come back tomorrow.', 'danger');
      return;
    }

    const faces = payload.faces || [];

    // Update server positions (used for name labels)
    _targetFaces = faces.map(f => ({ ...f, visuallyX: f.x, visuallyY: f.y, visuallyW: f.w, visuallyH: f.h }));

    updateStats(faces);
    _checkWelcome(faces);

    // FPS counter (server round-trips)
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
      document.getElementById('fps-badge').textContent = `${frameCount} FPS`;
      frameCount = 0;
      lastFpsTime = now;
    }
    sendFrame();
  };
}

function _setWsStatus(online) {
  const dot  = document.querySelector('.ws-dot');
  const text = document.getElementById('ws-text');
  dot.className  = `ws-dot ${online ? 'online' : 'offline'}`;
  text.textContent = online ? 'System Online' : 'Reconnecting…';
}

// =========================================================
// ── Camera ───────────────────────────────────────────────
// =========================================================
async function toggleCamera() {
  isStreaming ? stopCamera() : await startCamera();
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });

    video.srcObject = stream;
    document.getElementById('camera-off-state').style.display = 'none';
    document.getElementById('live-badge').style.display = 'flex';

    const btn = document.getElementById('cam-btn');
    btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop Feed`;
    btn.className = 'btn btn-danger btn-full';

    video.onloadedmetadata = async () => {
      const syncCanvas = () => {
        canvas.width  = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
      };
      syncCanvas();
      new ResizeObserver(syncCanvas).observe(canvas);

      isStreaming = true;

      // Start 60fps render loop immediately
      requestAnimationFrame(masterRenderLoop);

      // Start WebSocket server ping-pong
      if (ws && ws.readyState === WebSocket.OPEN) sendFrame();

      // Initialize MediaPipe in the background (non-blocking)
      initMediaPipe().then(() => {
        if (_mpReady && isStreaming) {
          requestAnimationFrame(mpTrackLoop);
        }
      });

      showToast('Live feed active. Recognition engaged.', 'success');
    };
  } catch (err) {
    showToast('Camera access denied — check browser permissions.', 'danger');
  }
}

function stopCamera() {
  isStreaming = false;
  _mpFaces = [];
  _targetFaces = [];

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  document.getElementById('camera-off-state').style.display = 'flex';
  document.getElementById('live-badge').style.display = 'none';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('fps-badge').textContent = '-- FPS';
  updateStats([]);

  const btn = document.getElementById('cam-btn');
  btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Engage Live Feed`;
  btn.className = 'btn btn-primary btn-full';
}

function sendFrame() {
  if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
  const tmp = document.createElement('canvas');
  tmp.width  = SERVER_W;
  tmp.height = SERVER_H;
  tmp.getContext('2d').drawImage(video, 0, 0, SERVER_W, SERVER_H);
  ws.send(JSON.stringify({ mode: currentMode, frame: tmp.toDataURL('image/jpeg', 0.5) }));
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode-recognize').classList.toggle('active', mode === 'recognize');
  document.getElementById('mode-detect').classList.toggle('active', mode === 'detect');

  if (mode === 'detect') {
    loadCocoModel().then(() => {
      if (isStreaming) requestAnimationFrame(cocoDetectionLoop);
    });
    _targetFaces = [];
    _mpFaces = [];
  }
}

// =========================================================
// ── Canvas Overlay ───────────────────────────────────────
// =========================================================
function drawOverlay(faces) {
  const scaleX = canvas.width  / SERVER_W;
  const scaleY = canvas.height / SERVER_H;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

  const scanMesh = document.querySelector('.scan-mesh');
  const scanRing = document.querySelector('.scan-ring');
  const scanPct  = document.getElementById('scan-pct-text');

  faces.forEach((face, idx) => {
    const isUnknown = face.name === 'Unknown';
    const isDetect  = face.name === 'Person';

    let strokeColor = isDark ? '#10b981' : '#059669';
    if (isUnknown) strokeColor = isDark ? '#ef4444' : '#dc2626';
    if (isDetect)  strokeColor = isDark ? '#3b82f6' : '#2563eb';
    if (face.name === 'Scanning…') strokeColor = '#f59e0b'; // amber while loading

    const fw = face.visuallyW * scaleX;
    const fh = face.visuallyH * scaleY;
    const fy = face.visuallyY * scaleY;
    const fx = canvas.width - (face.visuallyX * scaleX) - fw;

    // Lock scan illusion to first detected face
    if (typeof _scanRunning !== 'undefined' && _scanRunning && idx === 0) {
      const cx = fx + fw / 2;
      const cy = fy + fh / 2;
      if (scanMesh) { scanMesh.style.left = `${cx}px`; scanMesh.style.top = `${cy}px`; }
      if (scanRing) { scanRing.style.left = `${cx}px`; scanRing.style.top = `${cy}px`; }
      if (scanPct)  { scanPct.style.left  = `${cx + fw/2 + 10}px`; scanPct.style.top = `${cy}px`; }
      return;
    }

    // Glow
    if (isDark) { ctx.shadowColor = strokeColor; ctx.shadowBlur = 16; }

    // Oval face outline instead of box
    ctx.beginPath();
    ctx.ellipse(fx + fw/2, fy + fh/2, fw/2, fh/2, 0, 0, 2 * Math.PI);
    ctx.lineWidth   = 2.5;
    ctx.strokeStyle = strokeColor;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Corner accents at ellipse bounds (top-left, top-right, bottom-left, bottom-right)
    const cs = 12;
    ctx.lineWidth = 3;
    if (isDark) { ctx.shadowColor = strokeColor; ctx.shadowBlur = 10; }
    // Top-left
    ctx.beginPath(); ctx.moveTo(fx, fy + cs); ctx.lineTo(fx, fy); ctx.lineTo(fx + cs, fy); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(fx + fw - cs, fy); ctx.lineTo(fx + fw, fy); ctx.lineTo(fx + fw, fy + cs); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(fx, fy + fh - cs); ctx.lineTo(fx, fy + fh); ctx.lineTo(fx + cs, fy + fh); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(fx + fw - cs, fy + fh); ctx.lineTo(fx + fw, fy + fh); ctx.lineTo(fx + fw, fy + fh - cs); ctx.stroke();
    ctx.shadowBlur = 0;

    // Bigger, bolder label for demo visibility
    const label = face.confidence > 0 ? `${face.name}  ${face.confidence}%` : face.name;
    ctx.font = 'bold 14px "Inter", sans-serif';
    const tw = ctx.measureText(label).width;
    const ph = 28, pw = tw + 24;
    const lx = fx;
    const ly = fy - ph - 6;

    ctx.fillStyle = strokeColor;
    _roundRect(ctx, lx, ly < 0 ? fy + 4 : ly, pw, ph, 6);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 12, (ly < 0 ? fy + 4 : ly) + ph / 2);
  });
}

function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function updateStats(faces) {
  let known = 0, unknown = 0;
  faces.forEach(f => {
    if (f.name !== 'Unknown' && f.name !== 'Person' && f.name !== 'Scanning…') known++;
    else if (f.name === 'Unknown') unknown++;
  });
  document.getElementById('stat-faces').textContent   = faces.length;
  document.getElementById('stat-known').textContent   = known;
  document.getElementById('stat-unknown').textContent = unknown;
}

// =========================================================
// ── Enrollment ───────────────────────────────────────────
// =========================================================
let capturedBlob = null;

function previewImage(input) {
  if (!input.files[0]) return;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('preview-filename').textContent = input.files[0].name;
    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('upload-preview').style.display = 'flex';
  };
  reader.readAsDataURL(input.files[0]);
}

function setEnrollTab(mode) {
  document.getElementById('enroll-mode').value = mode;
  document.querySelectorAll('.enroll-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.enroll-pane').forEach(p => p.style.display = 'none');
  event.target.classList.add('active');
  document.getElementById('pane-' + mode).style.display = 'block';
  capturedBlob = null;
  document.getElementById('upload-preview').style.display = 'none';
  if (mode === 'upload') document.getElementById('upload-zone').style.display = 'block';
}

function captureSnapshot() {
  if (!isStreaming) { showToast('Turn on the Camera Feed first.', 'danger'); return; }
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  c.toBlob(blob => {
    capturedBlob = blob;
    document.getElementById('preview-img').src = URL.createObjectURL(blob);
    document.getElementById('preview-filename').textContent = 'Live Snapshot Captured';
    document.getElementById('upload-preview').style.display = 'flex';
    showToast('Snapshot captured.', 'success');
  }, 'image/jpeg', 0.9);
}

async function registerFace(e) {
  e.preventDefault();
  const name = document.getElementById('reg-name').value.trim();
  const mode = document.getElementById('enroll-mode').value;
  const btn  = document.getElementById('reg-btn');
  const formData = new FormData();
  formData.append('name', name);
  formData.append('mode', mode);

  if (mode === 'upload') {
    const file = document.getElementById('reg-file').files[0];
    if (!file) { showToast('Please select a photo.', 'danger'); return; }
    formData.append('file', file);
  } else if (mode === 'camera') {
    if (!capturedBlob) { showToast('Please capture a snapshot first.', 'danger'); return; }
    formData.append('file', capturedBlob, 'snapshot.jpg');
  } else if (mode === 'url') {
    const url = document.getElementById('reg-url').value.trim();
    if (!url) { showToast('Please enter an image URL.', 'danger'); return; }
    formData.append('url', url);
  } else if (mode === 'path') {
    const path = document.getElementById('reg-path').value.trim();
    if (!path) { showToast('Please enter a local file path.', 'danger'); return; }
    formData.append('path', path);
  }

  btn.disabled = true;
  btn.innerHTML = 'Processing…';
  try {
    const res  = await fetch('/api/register', { method: 'POST', body: formData });
    const data = await res.json();
    if (res.ok) {
      showToast(`${data.name} enrolled successfully.`, 'success');
      document.getElementById('reg-form').reset();
      document.getElementById('upload-preview').style.display = 'none';
      document.getElementById('upload-zone').style.display = 'block';
      loadPeople();
    } else {
      showToast(data.detail || 'Enrollment failed.', 'danger');
    }
  } catch {
    showToast('Network error — server unreachable.', 'danger');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Enroll Subject`;
  }
}

// =========================================================
// ── Data Loaders ─────────────────────────────────────────
// =========================================================

// _appMode is set on initApp() from /api/config
let _appMode = 'admin';

async function loadPeople() {
  const grid = document.getElementById('people-grid');
  try {
    const data = await (await fetch('/api/people')).json();
    document.getElementById('db-count').textContent = data.length;
    if (!data.length) {
      grid.innerHTML = `<div class="empty-hint"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg><p>No subjects enrolled</p></div>`;
      return;
    }
    grid.innerHTML = '';
    data.forEach(p => {
      const row = document.createElement('div');
      row.className = 'person-row';
      const img = p.image_path ? `${p.image_path}?t=${Date.now()}` : '';
      // Show delete button only in admin mode
      const delBtn = _appMode === 'admin'
        ? `<button class="person-del" onclick="deletePerson('${p.name}')" title="Revoke">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
           </button>`
        : '';
      row.innerHTML = `
        <img class="person-avatar" src="${img}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22 style=%22background:%23333%22><text y=%22.9em%22 font-size=%2280%22>👤</text></svg>'" />
        <div class="person-info">
          <div class="person-name">${p.name}</div>
          <div class="person-meta">ID·${btoa(p.name).substring(0,8).toUpperCase()}</div>
        </div>${delBtn}`;
      grid.appendChild(row);
    });
  } catch { showToast('Failed to load identity database.', 'danger'); }
}

async function deletePerson(name) {
  if (!confirm(`Revoke access for "${name}"?`)) return;
  try {
    await fetch(`/api/people/${encodeURIComponent(name)}`, { method: 'DELETE' });
    showToast(`${name} removed from database.`, 'success');
    loadPeople();
  } catch { showToast('Failed to delete subject.', 'danger'); }
}

async function clearActivityLog() {
  if (!confirm('Clear all activity log records for today?')) return;
  try {
    await fetch('/api/attendance/clear', { method: 'DELETE' });
    showToast('Activity log cleared.', 'success');
    loadAttendance();
  } catch { showToast('Failed to clear log.', 'danger'); }
}

async function loadAttendance() {
  const list = document.getElementById('attendance-list');
  try {
    const data = await (await fetch('/api/attendance')).json();
    document.getElementById('att-count').textContent = data.length;
    try {
      const stats = await (await fetch('/api/stats')).json();
      document.getElementById('total-today').textContent  = stats.total_recognitions_today;
      document.getElementById('unique-today').textContent = stats.unique_people_today;
    } catch {}

    if (!data.length) {
      list.innerHTML = `<div class="empty-hint"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>Awaiting activity…</p></div>`;
      return;
    }

    list.innerHTML = '';
    data.forEach(r => {
      const d    = new Date(r.seen_at + 'Z');
      const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const row  = document.createElement('div');
      row.className = 'log-entry';
      row.innerHTML = `
        <div class="log-av">${r.name.charAt(0).toUpperCase()}</div>
        <div class="log-info">
          <div class="log-name">${r.name}</div>
          <div class="log-time">${time}</div>
          <span class="log-conf">${(r.confidence||0).toFixed(1)}%</span>
        </div>`;
      list.appendChild(row);
    });
  } catch { /* silent */ }
}

function exportAttendance() {
  window.location.href = '/api/attendance/export';
  showToast('Downloading telemetry export…', 'success');
}

setInterval(() => { if (isStreaming) loadAttendance(); }, 8000);

// =========================================================
// ── "FaceID" Style Auto-Scan Illusion ────────────────────
// =========================================================
let _scanBlobs = [];
let _scanRunning = false;

async function startFaceScan() {
  if (!isStreaming) { showToast('Turn on the Camera Feed first.', 'danger'); return; }
  const name = document.getElementById('reg-name').value.trim();
  if (!name) { showToast('Enter a Subject Name first.', 'danger'); return; }
  if (_scanRunning) return;

  _scanRunning = true;
  _scanBlobs = [];

  document.getElementById('scan-btn').style.display = 'none';
  const overlay = document.getElementById('scan-illusion-overlay');
  const pctText = document.getElementById('scan-pct-text');
  overlay.style.display = 'flex';
  document.getElementById('scan-status').textContent = 'Hold still... Scanning biometric mesh.';

  const duration = 3500;
  let elapsed = 0;
  let lastTime = Date.now();
  const captureTimes = [500, 1000, 1800, 2500, 3200];
  let capturesDone = 0;

  return new Promise(resolve => {
    function tick() {
      if (!_scanRunning) return;
      const now = Date.now();
      const dt = now - lastTime;
      lastTime = now;

      // Pause if face lost
      const hasFace = (_mpReady ? _mpFaces.length > 0 : _targetFaces.length > 0);
      if (!hasFace) {
        document.getElementById('scan-status').textContent = '⚠ Face lost! Please look at the camera...';
        pctText.style.color = '#ef4444';
        requestAnimationFrame(tick);
        return;
      }

      document.getElementById('scan-status').textContent = 'Hold still... Scanning biometric mesh.';
      elapsed += dt;

      const pct = Math.min(100, Math.floor((elapsed / duration) * 100));
      pctText.textContent = `${pct}%`;

      // Colour transition: Red → Blue → Green
      const mesh = document.querySelector('.scan-mesh');
      const ring = document.querySelector('.scan-ring');
      let color = '#10b981';
      if (pct < 20) color = '#ef4444';
      else if (pct < 60) color = '#3b82f6';

      pctText.style.color = color;
      pctText.style.textShadow = `0 0 12px ${color}`;
      if (mesh) mesh.style.borderColor = color;
      if (ring) {
        ring.style.borderTopColor = color;
        ring.style.borderBottomColor = color;
        ring.style.boxShadow = `0 0 15px ${color}`;
      }

      // Capture frames
      if (capturesDone < captureTimes.length && elapsed > captureTimes[capturesDone]) {
        const tmp = document.createElement('canvas');
        tmp.width = video.videoWidth; tmp.height = video.videoHeight;
        tmp.getContext('2d').drawImage(video, 0, 0);
        tmp.toBlob(b => { if (b) _scanBlobs.push(b); }, 'image/jpeg', 0.9);
        capturesDone++;
      }

      if (elapsed < duration) {
        requestAnimationFrame(tick);
      } else {
        finishScan(name);
        resolve();
      }
    }
    requestAnimationFrame(tick);
  });
}

async function finishScan(name) {
  document.getElementById('scan-illusion-overlay').style.display = 'none';
  document.getElementById('scan-status').textContent = '✅ Scan complete! Enrolling subject...';
  await submitMultiScan(name);
  _scanRunning = false;
  _scanBlobs = [];
  document.getElementById('scan-btn').style.display = 'flex';
  document.getElementById('scan-status').textContent = 'Keep your face in the center of the camera.';
}

async function submitMultiScan(name) {
  let lastError = '';
  for (let i = 0; i < _scanBlobs.length; i++) {
    const fd = new FormData();
    fd.append('name', name);
    fd.append('mode', 'camera');
    fd.append('file', _scanBlobs[i], `scan_${i}.jpg`);
    try {
      const res = await fetch('/api/register', { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok) {
        showToast(`✅ ${name} successfully enrolled from 3D scan.`, 'success');
        document.getElementById('reg-name').value = '';
        loadPeople();
        return;
      }
      lastError = data.detail || 'No clear face detected';
    } catch (e) { lastError = 'Network error'; }
  }
  showToast(`⚠ Scan failed: ${lastError}. Try better lighting.`, 'danger');
}

// =========================================================
// ── COCO-SSD Object Detection ─────────────────────────────
// =========================================================
let _cocoModel = null;
let _cocoLoading = false;

async function loadCocoModel() {
  if (_cocoModel || _cocoLoading) return;
  _cocoLoading = true;
  try {
    _cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    console.log('[COCO-SSD] Model loaded.');
  } catch(e) { console.error('[COCO-SSD] Load failed:', e); }
}

async function cocoDetectionLoop() {
  if (!isStreaming || currentMode !== 'detect') return;
  if (!_cocoModel) { requestAnimationFrame(cocoDetectionLoop); return; }
  try {
    const preds = await _cocoModel.detect(video);
    drawCocoOverlay(preds);
  } catch(e) {}
  requestAnimationFrame(cocoDetectionLoop);
}

function drawCocoOverlay(preds) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / vw, scaleY = canvas.height / vh;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316'];
  preds.forEach((pred, idx) => {
    const [bx, by, bw, bh] = pred.bbox;
    const fx = canvas.width - (bx * scaleX) - (bw * scaleX);
    const fy = by * scaleY, fw = bw * scaleX, fh = bh * scaleY;
    const color = colors[idx % colors.length];
    const label = `${pred.class} ${Math.round(pred.score * 100)}%`;
    if (isDark) { ctx.shadowColor = color; ctx.shadowBlur = 12; }
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(fx, fy, fw, fh);
    ctx.shadowBlur = 0;
    ctx.font = 'bold 11px JetBrains Mono, monospace';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    _roundRect(ctx, fx, fy - 22, tw + 16, 20, 4);
    ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
    ctx.fillText(label, fx + 8, fy - 12);
  });
}

// =========================================================
// ── Alert & Toasts ───────────────────────────────────────
// =========================================================
let _alertTimer = null;

function showAlert() {
  if (alertShowing) return;
  alertShowing = true;
  const el = document.getElementById('alert-overlay');
  el.classList.add('show');
  el.setAttribute('aria-hidden', 'false');
  clearTimeout(_alertTimer);
  _alertTimer = setTimeout(closeAlert, 4000);
}

function closeAlert() {
  alertShowing = false;
  const el = document.getElementById('alert-overlay');
  el.classList.remove('show');
  el.setAttribute('aria-hidden', 'true');
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transition = 'opacity 0.3s';
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

// =========================================================
// ── Auth ─────────────────────────────────────────────────
// =========================================================
async function submitLogin(e) {
  e.preventDefault();
  const pass = document.getElementById('login-pass').value;
  const btn = document.getElementById('login-btn');
  btn.disabled = true; btn.textContent = 'Verifying...';
  const fd = new FormData();
  fd.append('password', pass);
  try {
    const res = await fetch('/api/login', { method: 'POST', body: fd });
    if (res.ok) {
      document.getElementById('login-overlay').style.display = 'none';
      applyAdminUI();
      connectWebSocket();
      loadPeople();
      loadAttendance();
    } else {
      showToast('Invalid authorization code.', 'danger');
      btn.disabled = false; btn.textContent = 'Authenticate';
    }
  } catch(e) {
    showToast('Network error.', 'danger');
    btn.disabled = false; btn.textContent = 'Authenticate';
  }
}

// =========================================================
// ── App Mode (controls admin vs public UI) ────────────────
// =========================================================
function applyAdminUI() {
  // Reveal all admin-only elements
  document.querySelectorAll('.admin-only').forEach(el => el.style.display = '');
}

async function initApp() {
  initTheme();
  try {
    const cfg = await (await fetch('/api/config')).json();
    _appMode = cfg.mode || 'admin';
  } catch(e) { _appMode = 'admin'; }

  if (_appMode === 'admin') {
    // Require login first — show screen, wait for auth cookie
    const statsRes = await fetch('/api/stats').catch(() => null);
    if (!statsRes || statsRes.status === 401) {
      document.getElementById('login-overlay').style.display = 'flex';
      return; // Wait for user to log in via submitLogin()
    }
    // Already logged in
    applyAdminUI();
  }

  // Both modes: connect and load data
  connectWebSocket();
  loadPeople();
  loadAttendance();
  setTimeout(() => loadCocoModel(), 3000);
}

// =========================================================
// ── Init ─────────────────────────────────────────────────
// =========================================================
window.onload = () => initApp();
