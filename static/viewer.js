import {
  palettes, colorAt, preset, viewTransform, screenPoint, resizeCanvas,
  createPointRenderer, cameraProjection, projectWorldPoint, imagePoint,
} from './point-cloud.js?v=6';

if (!Object.hasOwn(HTMLElement.prototype, 'popover')) {
  import('https://unpkg.com/@oddbird/popover-polyfill@latest/dist/popover.min.js').catch(() => {});
}

if (!('switch' in HTMLInputElement.prototype)) {
  document.documentElement.classList.add('no-native-switch');
}

const $ = (id) => document.getElementById(id);
const cloud = $('cloud');
const camera = $('camera');
const minimap = $('minimap');
const renderer = createPointRenderer(cloud);
const cameraContext = camera.getContext('2d');
const minimapContext = minimap.getContext('2d');
const lightProbe = document.createElement('canvas');
lightProbe.width = 8;
lightProbe.height = 8;
const lightProbeContext = lightProbe.getContext('2d', { willReadFrequently: true });

const emptyPoints = new Float32Array();
const emptyFloats = new Float32Array();
const imageTypes = {
  image_02: ['Forward camera', 'Image 02 · RGB stereo'],
  image_01: ['Forward camera', 'Image 01 · Monochrome'],
  ambient: ['Ambient panorama', 'OS1-128 · 360° near-IR'],
  reflec: ['Reflectivity panorama', 'OS1-128 · 360° surface'],
};
const viewLabels = {
  chase: '3D CHASE',
  top: 'BEV RADAR',
  ego: 'ROOF HOOD',
  front: 'FRONT LOOK',
  orbit: '3D ORBIT',
};
const routeAliases = {
  DurLAR_20210716_S: 'Jul 16 · Summer Curb Crawl',
  DurLAR_20210901_S: 'Sep 01 · Autumn Campus Drift',
  DurLAR_20211012_S: 'Oct 12 · Leaf Dodger Run',
  DurLAR_20211208_S: 'Dec 08 · Chilly Twizy Lap',
  DurLAR_20211209_S: 'Dec 09 · Winter Photon Hunt',
  sample_durlar_sequence: 'Sim Loop · Fake Street',
};
const sliders = {
  range: ' m', 'min-height': ' m', 'max-height': ' m', intensity: '%',
  'point-size': ' px', opacity: '%', 'overlay-size': ' px',
};
const modeBuckets = Object.fromEntries(
  Object.keys(palettes).map((mode) => [mode, Array.from({ length: 32 }, (_, i) => colorAt(i / 31, mode))]),
);
const state = {
  info: null, frame: null, requestedIndex: 0, sequenceRequest: null, frameRequest: null,
  prefetch: null, points: emptyPoints, selected: -1, drawing: false, timer: null, warmupTimer: null,
  playing: false, playbackRun: 0, stageMode: 'video', view: preset('chase'),
  transform: null, imageTransform: null, speedUnit: 'kmh',
};
const pointCache = new Map();
const perceptionCache = new Map();
let settings;

function syncSliderVars(slider) {
  if (!slider) return;
  slider.style.setProperty('--attr-min', slider.min || 0);
  slider.style.setProperty('--attr-max', slider.max || 100);
  slider.style.setProperty('--control-value', slider.value || 0);
}

function syncAllSliders() {
  for (const id of ['timeline', ...Object.keys(sliders)]) {
    syncSliderVars($(id));
  }
}

function decodeFloat32B64(b64) {
  if (!b64) return emptyFloats;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.byteLength % 4 === 0 ? new Float32Array(bytes.buffer) : emptyFloats;
}

function setStatus(message) {
  $('status').textContent = message;
  $('status').hidden = !message;
}

function readSettings() {
  const number = (id) => Number($(id).value);
  const maxIntensity = Math.max(1, state.frame?.p98Intensity || state.frame?.maxIntensity || 1);
  const fsdAi = $('fsd-toggle')?.getAttribute('aria-pressed') === 'true';
  settings = {
    mode: $('color').value, maxRange: number('range'), minHeight: number('min-height'),
    maxHeight: number('max-height'), minIntensity: (number('intensity') / 100) * maxIntensity,
    maxIntensity, pointSize: number('point-size'),
    stride: number('stride'), opacity: number('opacity') / 100, overlaySize: number('overlay-size'),
    grid: $('grid').checked, vehicle: $('vehicle').checked, axes: $('axes').checked,
    fsdAi,
    fsdRoad: fsdAi && ($('fsd-road')?.checked ?? true),
    fsdDepthFill: fsdAi && ($('fsd-depth')?.checked ?? true),
    fsdObjects: fsdAi && ($('fsd-objects')?.checked ?? true),
  };
  for (const [id, unit] of Object.entries(sliders)) {
    const out = document.querySelector(`output[for="${id}"]`);
    if (out) out.value = $(id).value + unit;
    syncSliderVars($(id));
  }
  const legends = {
    range: ['Distance', '0 m', `${settings.maxRange} m`],
    height: ['Height', `${settings.minHeight} m`, `${settings.maxHeight} m`],
    intensity: ['Reflectivity', '0', Number(settings.maxIntensity.toPrecision(3)).toLocaleString()],
  };
  const [title, min, max] = legends[settings.mode];
  $('legend-title').textContent = title;
  $('legend-min').textContent = min;
  $('legend-max').textContent = max;
  $('legend-ramp').style.background = `linear-gradient(90deg, ${palettes[settings.mode].join(', ')})`;
  const dots = $('camera-legend').querySelectorAll('span');
  if (dots.length >= 2) {
    dots[0].style.background = palettes[settings.mode][0];
    dots[1].style.background = palettes[settings.mode][2];
  }
}

const perceptionPalette = {
  person: { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.88)', boxFill: 'rgba(251, 146, 60, 0.11)', kind: 'person' },
  dog: { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.88)', boxFill: 'rgba(251, 146, 60, 0.11)', kind: 'person' },
  bicycle: { stroke: '#22d3ee', fill: 'rgba(34, 211, 238, 0.88)', boxFill: 'rgba(34, 211, 238, 0.10)', kind: 'cycle' },
  motorcycle: { stroke: '#22d3ee', fill: 'rgba(34, 211, 238, 0.88)', boxFill: 'rgba(34, 211, 238, 0.10)', kind: 'cycle' },
  'traffic light': { stroke: '#facc15', fill: 'rgba(250, 204, 21, 0.88)', boxFill: 'rgba(250, 204, 21, 0.09)', kind: 'sign' },
  'stop sign': { stroke: '#facc15', fill: 'rgba(250, 204, 21, 0.88)', boxFill: 'rgba(250, 204, 21, 0.09)', kind: 'sign' },
};
const defaultPerceptionStyle = {
  stroke: '#60a5fa', fill: 'rgba(96, 165, 250, 0.84)', boxFill: 'rgba(56, 189, 248, 0.10)', kind: 'vehicle',
};
const perceptionStyle = (label) => perceptionPalette[label] || defaultPerceptionStyle;

function updateDetectionPills(perc) {
  const wrap = $('hud-detections');
  if (!wrap) return;
  const objects = settings?.fsdAi && settings?.fsdObjects ? perc?.objects : null;
  if (!objects?.length) {
    wrap.hidden = true;
    wrap.replaceChildren();
    return;
  }
  const counts = new Map();
  for (const obj of objects) {
    counts.set(obj.label, (counts.get(obj.label) || 0) + 1);
  }
  wrap.replaceChildren(...Array.from(counts.entries()).map(([label, n]) => {
    const pill = document.createElement('span');
    pill.className = 'det-pill';
    const dot = document.createElement('i');
    dot.style.background = perceptionStyle(label).stroke;
    pill.append(dot, `${n} ${label.toUpperCase()}${n > 1 && !label.endsWith('s') ? 'S' : ''}`);
    return pill;
  }));
  wrap.hidden = false;
}

function updateRouteSubtitle() {
  const raw = state.frame?.raw || emptyPoints;
  const aiChip = $('hud-ai-chip');
  if (!raw.length) {
    $('hud-route-sub').textContent = 'OS1-128 · Waiting for returns';
    if (aiChip) aiChip.hidden = true;
    return;
  }
  const returnsCount = (state.points.length / 4).toLocaleString();
  const perc = settings.fsdAi ? state.frame?.perception : null;
  if (!perc) {
    $('hud-route-sub').textContent = `OS1-128 · ${returnsCount} filtered returns`;
    if (aiChip) aiChip.hidden = true;
    return;
  }
  const dev = (perc.device || 'ai').split(' ')[0].toUpperCase();
  const timing = perc.cached ? 'CACHED' : `${Math.round(perc.inference_ms || 0)}MS`;
  const count = perc.objects?.length || 0;
  if (aiChip) {
    aiChip.textContent = `${dev} · ${timing}`;
    aiChip.hidden = false;
  }
  $('hud-route-sub').textContent = `${returnsCount} pts · ${count} 3D object${count === 1 ? '' : 's'}`;
}

function filterPoints() {
  const raw = state.frame?.raw || emptyPoints;
  const { stride, maxRange, minHeight, maxHeight, minIntensity } = settings;
  const points = new Float32Array(Math.ceil(raw.length / (4 * stride)) * 4);
  let n = 0;
  for (let i = 0; i < raw.length; i += 4 * stride) {
    const x = raw[i], y = raw[i + 1], z = raw[i + 2], intensity = raw[i + 3];
    if (!Number.isFinite(x + y + z + intensity) || z < minHeight || z > maxHeight || intensity < minIntensity) continue;
    const squaredRange = x * x + y * y + z * z;
    if (squaredRange === 0 || squaredRange > maxRange * maxRange) continue;
    points[n++] = x;
    points[n++] = y;
    points[n++] = z;
    points[n++] = intensity;
  }
  state.points = points.subarray(0, n);
  state.selected = -1;
  renderer.upload(state.points);
  const returnsCount = (n / 4).toLocaleString();
  const totalCount = (raw.length / 4).toLocaleString();
  const sampling = state.frame?.streamStride > 1 ? ` · 1/${state.frame.streamStride}` : '';
  $('point-count').textContent = raw.length ? `${returnsCount} / ${totalCount} pts${sampling}` : 'No returns';
  updateRouteSubtitle();
  $('cloud-empty').hidden = n > 0;
  $('cloud-empty').textContent = raw.length ? 'No points match the current filters.' : 'No laser returns for this frame.';
}

function mark(id, point) {
  const marker = $(id);
  marker.hidden = !point;
  if (!point) return;
  marker.style.left = `${point[0]}px`;
  marker.style.top = `${point[1]}px`;
}

function drawCloud() {
  const { width, height, ratio } = resizeCanvas(cloud);
  if (!width || !height) return;
  state.transform = viewTransform(state.view, width, height);
  renderer.draw(state.transform, settings, ratio);
  mark('cloud-marker', state.selected < 0 ? null : screenPoint(state.points, state.selected, state.transform));
  const scale = $('scene-scale');
  scale.hidden = state.stageMode !== '3d' || state.view.name !== 'top';
  if (!scale.hidden) {
    const meters = state.view.distance < 25 ? 5 : state.view.distance < 105 ? 10 : 20;
    scale.style.width = `${meters * height / state.view.distance}px`;
    scale.textContent = `${meters} m`;
  }
}

function drawMinimap() {
  const { width, height, ratio } = resizeCanvas(minimap);
  if (!width || !height) return;
  const ctx = minimapContext;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  // Subtle tactical sonar radial background
  const bgGrad = ctx.createRadialGradient(width * 0.5, height * 0.68, 4, width * 0.5, height * 0.68, height * 0.8);
  bgGrad.addColorStop(0, '#0c1624');
  bgGrad.addColorStop(1, '#060a11');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, width, height);

  const cx = width * 0.5;
  const cy = height * 0.70;
  const pxPerMeter = height / 44;

  // Forward camera ~62° horizontal FOV frustum wedge
  const fovHalf = (31 * Math.PI) / 180;
  const fovRadius = 34 * pxPerMeter;
  const fovGrad = ctx.createRadialGradient(cx, cy, 4, cx, cy, fovRadius);
  fovGrad.addColorStop(0, 'rgba(56, 189, 248, 0.20)');
  fovGrad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
  ctx.fillStyle = fovGrad;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, fovRadius, -Math.PI / 2 - fovHalf, -Math.PI / 2 + fovHalf);
  ctx.closePath();
  ctx.fill();

  // Concentric metric radar rings
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  ctx.lineWidth = 1;
  ctx.font = '600 8px ui-monospace, SFMono-Regular, monospace';
  ctx.fillStyle = 'rgba(148, 163, 184, 0.42)';
  for (const r of [10, 20, 30]) {
    const rad = r * pxPerMeter;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.stroke();
    if (cy - rad > 8) {
      ctx.fillText(`${r}m`, cx + 3, cy - rad + 8);
    }
  }

  // Ego trajectory beam
  const grad = ctx.createLinearGradient(cx, cy, cx, cy - 28 * pxPerMeter);
  grad.addColorStop(0, 'rgba(59, 130, 246, 0.58)');
  grad.addColorStop(1, 'rgba(59, 130, 246, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(cx - 0.62 * pxPerMeter, cy - 28 * pxPerMeter, 1.24 * pxPerMeter, 28 * pxPerMeter);

  const pts = state.points;
  const step = Math.max(1, Math.floor(pts.length / 4 / 3800)) * 4;
  for (let i = 0; i < pts.length; i += step) {
    const x = pts[i], y = pts[i + 1], z = pts[i + 2];
    const mx = cx - y * pxPerMeter;
    const my = cy - x * pxPerMeter;
    if (mx < 0 || mx >= width || my < 0 || my >= height) continue;
    ctx.fillStyle = z > -1.32 ? 'rgba(91, 247, 204, 0.78)' : 'rgba(71, 85, 105, 0.32)';
    ctx.fillRect(mx, my, 1.4, 1.4);
  }

  const objects = settings.fsdAi && settings.fsdObjects ? state.frame?.perception?.objects : null;
  if (objects?.length) {
    for (const obj of objects) {
      if (!obj.center) continue;
      const mx = cx - obj.center[1] * pxPerMeter;
      const my = cy - obj.center[0] * pxPerMeter;
      if (mx < -20 || mx >= width + 20 || my < -20 || my >= height + 20) continue;
      const st = perceptionStyle(obj.label);
      if (st.kind === 'person') {
        ctx.fillStyle = st.fill;
        ctx.beginPath();
        ctx.arc(mx, my, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(251, 146, 60, 0.52)';
        ctx.beginPath();
        ctx.arc(mx, my, 5.5, 0, Math.PI * 2);
        ctx.stroke();
      } else if (st.kind === 'sign') {
        ctx.fillStyle = st.fill;
        ctx.fillRect(mx - 2, my - 2, 4, 4);
      } else {
        const [len = 4.4, wid = 1.9] = obj.size || [];
        const bw = Math.max(4, wid * pxPerMeter);
        const bh = Math.max(6, len * pxPerMeter);
        ctx.save();
        ctx.translate(mx, my);
        ctx.rotate(-(obj.yaw || 0));
        ctx.fillStyle = st.fill;
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.fillRect(-bw / 2, -bh / 2, bw, bh);
        ctx.strokeRect(-bw / 2, -bh / 2, bw, bh);
        ctx.restore();
      }
    }
  }

  // Ego vehicle chevron
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.moveTo(cx, cy - 7);
  ctx.lineTo(cx + 5, cy + 5);
  ctx.lineTo(cx, cy + 2.2);
  ctx.lineTo(cx - 5, cy + 5);
  ctx.closePath();
  ctx.fill();
}

function drawRoundedPill(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.rect(x, y, w, h);
  }
}

function drawCameraDetections(ctx, perception, ox, oy, scale, canvasW, canvasH, isMain) {
  ctx.save();
  ctx.font = '650 10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';

  for (const obj of perception.objects) {
    const box = obj.box2d;
    if (!box || box.length < 4) continue;
    const rawX1 = ox + box[0] * scale;
    const rawY1 = oy + box[1] * scale;
    const rawX2 = ox + box[2] * scale;
    const rawY2 = oy + box[3] * scale;
    if (rawX2 < 2 || rawY2 < 2 || rawX1 > canvasW - 2 || rawY1 > canvasH - 2) continue;

    // Clamp visible box inside viewport so cover-cropped edge detections never clip off-screen
    const x1 = Math.max(4, rawX1);
    const y1 = Math.max(4, rawY1);
    const x2 = Math.min(canvasW - 4, rawX2);
    const y2 = Math.min(canvasH - 4, rawY2);
    const w = x2 - x1;
    const h = y2 - y1;
    if (w < 6 || h < 6) continue;

    const st = perceptionStyle(obj.label);
    const arm = Math.max(5, Math.min(14, w * 0.24, h * 0.24));

    if (isMain) {
      ctx.fillStyle = st.boxFill;
      ctx.fillRect(x1, y1, w, h);
    }

    ctx.strokeStyle = st.stroke;
    ctx.lineWidth = isMain ? 1.9 : 1.3;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + arm); ctx.lineTo(x1, y1); ctx.lineTo(x1 + arm, y1);
    ctx.moveTo(x1 + w - arm, y1); ctx.lineTo(x1 + w, y1); ctx.lineTo(x1 + w, y1 + arm);
    ctx.moveTo(x1 + w, y1 + h - arm); ctx.lineTo(x1 + w, y1 + h); ctx.lineTo(x1 + w - arm, y1 + h);
    ctx.moveTo(x1 + arm, y1 + h); ctx.lineTo(x1, y1 + h); ctx.lineTo(x1, y1 + h - arm);
    ctx.stroke();

    if (!isMain && w < 28) continue;
    const dist = obj.center ? Math.hypot(obj.center[0], obj.center[1]) : null;
    const tag = dist !== null ? `${obj.label.toUpperCase()} · ${dist.toFixed(1)}m` : obj.label.toUpperCase();
    const tw = ctx.measureText(tag).width + 12;
    const tx = Math.max(8, Math.min(canvasW - tw - 8, x1));
    const ty = y1 > 26 ? y1 - 21 : Math.min(canvasH - 22, y1 + 4);

    ctx.fillStyle = 'rgba(6, 11, 18, 0.88)';
    drawRoundedPill(ctx, tx, ty, tw, 17, 5);
    ctx.fill();
    ctx.strokeStyle = st.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(tag, tx + 6, ty + 9);
  }
  ctx.restore();
}

function drawCameraPathRibbon(ctx, projection, ox, oy, scale) {
  const left = [];
  const right = [];
  const rungs = [];
  const zGround = -1.68;
  for (let x = 2.6; x <= 30.0; x += 2.6) {
    const pL = projectWorldPoint(x, 0.58, zGround, projection);
    const pR = projectWorldPoint(x, -0.58, zGround, projection);
    if (!pL || !pR) continue;
    const sL = [ox + pL[0] * scale, oy + pL[1] * scale];
    const sR = [ox + pR[0] * scale, oy + pR[1] * scale];
    left.push(sL);
    right.push(sR);
    if (Math.abs(x - 7.8) < 0.5 || Math.abs(x - 15.6) < 0.5 || Math.abs(x - 23.4) < 0.5) {
      rungs.push([sL, sR]);
    }
  }
  if (left.length < 2) return;

  const first = left[0], last = left[left.length - 1];
  const grad = ctx.createLinearGradient(first[0], first[1], last[0], last[1]);
  grad.addColorStop(0, 'rgba(37, 99, 235, 0.40)');
  grad.addColorStop(0.6, 'rgba(56, 189, 248, 0.22)');
  grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

  ctx.save();
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < left.length; i++) ctx.lineTo(left[i][0], left[i][1]);
  for (let i = right.length - 1; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
  ctx.closePath();
  ctx.fill();

  // Subtle AR distance cross-rungs along corridor
  ctx.strokeStyle = 'rgba(125, 211, 252, 0.26)';
  ctx.lineWidth = 1;
  for (const [a, b] of rungs) {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(96, 165, 250, 0.62)';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < left.length; i++) ctx.lineTo(left[i][0], left[i][1]);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(right[0][0], right[0][1]);
  for (let i = 1; i < right.length; i++) ctx.lineTo(right[i][0], right[i][1]);
  ctx.stroke();
  ctx.restore();
}

function drawCamera() {
  const { width, height, ratio } = resizeCanvas(camera);
  const frame = state.frame;
  const image = frame?.image;
  const projection = frame?.projection;
  const canProject = !!(image && projection && frame.raw.length);
  const overlay = $('overlay');
  overlay.disabled = !canProject;
  $('camera-legend').hidden = !(canProject && overlay.checked && state.points.length);
  camera.classList.toggle('pickable', canProject && overlay.checked);
  $('image-size').hidden = !image;
  $('camera-empty').hidden = !!image;
  $('projection-hint').textContent = canProject
    ? (overlay.checked ? 'Projected lasers follow active palette & filters.' : 'Enable Lasers to overlay 3D returns.')
    : !image ? 'Camera feed unavailable.'
    : !frame.modality.startsWith('image') ? '360° cylindrical scan captured directly by roof LiDAR.'
    : !frame.raw.length ? 'No laser returns for this frame.' : 'Camera calibration unavailable.';
  cameraContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  cameraContext.clearRect(0, 0, width, height);
  state.imageTransform = null;
  mark('camera-marker', null);
  if (!image || !width || !height) return null;

  const isMainVideo = $('camera-scene').parentElement === $('main-stage');
  const useCover = isMainVideo && frame.modality.startsWith('image');
  const scale = useCover
    ? Math.max(width / image.width, height / image.height)
    : Math.min(width / image.width, height / image.height);
  const x = (width - image.width * scale) / 2;
  const y = (height - image.height * scale) / 2;
  state.imageTransform = canProject ? { x, y, scale } : null;
  cameraContext.drawImage(image, x, y, image.width * scale, image.height * scale);
  $('image-size').textContent = `${image.width} × ${image.height}`;
  if (!canProject) return null;

  if (settings.vehicle) {
    drawCameraPathRibbon(cameraContext, projection, x, y, scale);
  }

  if (overlay.checked && state.points.length) {
    const { points } = state;
    const { mode, maxRange, minHeight, maxHeight, maxIntensity, opacity } = settings;
    const size = settings.overlaySize * (isMainVideo ? 1 : 0.8);
    const half = size / 2;
    const sample = Math.max(1, Math.ceil(points.length / 4 / 36000)) * 4;
    const palette = modeBuckets[mode];
    const groundBuckets = Array.from({ length: 32 }, () => []);
    const objectBuckets = Array.from({ length: 32 }, () => []);
    const hSpan = Math.max(0.001, maxHeight - minHeight);

    for (let i = 0; i < points.length; i += sample) {
      const point = imagePoint(points, i, projection, image.width, image.height);
      if (!point) continue;
      const px = points[i], py = points[i + 1], pz = points[i + 2], pw = points[i + 3];
      let val = Math.hypot(px, py, pz) / maxRange;
      if (mode === 'height') val = (pz - minHeight) / hSpan;
      else if (mode === 'intensity') val = Math.sqrt(Math.max(0, Math.min(1, pw / maxIntensity)));
      const bucketIdx = Math.max(0, Math.min(31, Math.floor(Math.max(0, Math.min(1, val)) * 31)));
      const sx = x + point[0] * scale - half;
      const sy = y + point[1] * scale - half;
      const target = mode !== 'intensity' && pz < -1.32 ? groundBuckets[bucketIdx] : objectBuckets[bucketIdx];
      target.push(sx, sy);
    }

    if (mode !== 'intensity') {
      cameraContext.globalAlpha = opacity * 0.34;
      for (let b = 0; b < 32; b++) {
        const list = groundBuckets[b];
        if (!list.length) continue;
        cameraContext.fillStyle = palette[b];
        for (let k = 0; k < list.length; k += 2) {
          cameraContext.fillRect(list[k], list[k + 1], size, size);
        }
      }
    }

    cameraContext.globalAlpha = opacity;
    for (let b = 0; b < 32; b++) {
      const list = objectBuckets[b];
      if (!list.length) continue;
      cameraContext.fillStyle = palette[b];
      for (let k = 0; k < list.length; k += 2) {
        cameraContext.fillRect(list[k], list[k + 1], size, size);
      }
    }
    cameraContext.globalAlpha = 1;
  }

  if (settings.fsdAi && settings.fsdObjects && frame.modality === 'image_02' && frame.perception?.objects?.length) {
    drawCameraDetections(cameraContext, frame.perception, x, y, scale, width, height, isMainVideo);
  }

  const selected = state.selected < 0 ? null : imagePoint(state.points, state.selected, projection, image.width, image.height);
  const marker = selected && [x + selected[0] * scale, y + selected[1] * scale];
  mark('camera-marker', marker);
  return marker;
}

function drawMeasurement(projected) {
  const i = state.selected;
  const active = i >= 0;
  $('trip-card').dataset.inspecting = String(active);
  $('measurement').hidden = !active;
  $('clear-point').hidden = !active;
  $('projection-hint').hidden = !active;
  $('inspection-title').textContent = active ? 'Selected Return' : 'Laser Telemetry';
  if (!active) {
    $('inspection-help').textContent = 'Click any dot in camera or 3D view to measure metric coordinates.';
    return;
  }
  const [x, y, z, intensity] = state.points.subarray(i, i + 4);
  $('measure-range').textContent = `${Math.hypot(x, y, z).toFixed(2)} m`;
  $('measure-forward').textContent = `${x.toFixed(2)} m`;
  $('measure-left').textContent = `${y.toFixed(2)} m`;
  $('measure-height').textContent = `${z.toFixed(2)} m`;
  $('measure-intensity').textContent = intensity.toFixed(1);
  $('inspection-help').textContent = projected ? 'Matched across both forward camera and 3D views.'
    : state.imageTransform ? 'Return lies outside the forward camera field of view.' : 'Camera projection unavailable for panoramic sensor.';
}

function draw() {
  if (state.drawing) return;
  state.drawing = true;
  requestAnimationFrame(() => {
    state.drawing = false;
    drawCloud();
    drawMinimap();
    drawMeasurement(drawCamera());
  });
}

function pick(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  const { points, frame } = state;
  const isCloud = canvas === cloud;
  const image = state.imageTransform;
  if (!isCloud && (!image || !$('overlay').checked)) return;
  if (!state.transform) return;
  let nearest = -1, best = 10 ** 2, closestDepth = Infinity;
  const step = isCloud ? 4 : Math.max(1, Math.ceil(points.length / 4 / 36000)) * 4;
  for (let i = 0; i < points.length; i += step) {
    const point = isCloud ? screenPoint(points, i, state.transform)
      : imagePoint(points, i, frame.projection, frame.image.width, frame.image.height);
    if (!point) continue;
    const px = isCloud ? point[0] : image.x + point[0] * image.scale;
    const py = isCloud ? point[1] : image.y + point[1] * image.scale;
    const distance = (px - x) ** 2 + (py - y) ** 2;
    if (distance < best || (distance === best && point[2] < closestDepth)) {
      best = distance;
      closestDepth = point[2];
      nearest = i;
    }
  }
  stop();
  state.selected = nearest;
  draw();
}

function updatePipBadge() {
  const badge = $('pip-mode-badge');
  if (!badge) return;
  badge.textContent = state.stageMode === 'video'
    ? (viewLabels[state.view.name] || '3D VIEW')
    : 'CAM PIP';
}

function applyStageDom(name) {
  const mainStage = $('main-stage');
  const pipSlot = $('pip-slot');
  const cloudScene = $('cloud-scene');
  const cameraScene = $('camera-scene');

  if (name === 'video') {
    state.stageMode = 'video';
    if (cameraScene.parentElement !== mainStage) mainStage.prepend(cameraScene);
    if (cloudScene.parentElement !== pipSlot) pipSlot.prepend(cloudScene);
  } else {
    state.stageMode = '3d';
    state.view = preset(name);
    if (cloudScene.parentElement !== mainStage) mainStage.prepend(cloudScene);
    if (cameraScene.parentElement !== pipSlot) pipSlot.prepend(cameraScene);
  }

  const activeButton = state.stageMode === 'video' ? 'video' : state.view.name;
  document.querySelectorAll('[data-view]').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.view === activeButton));
  });
  document.querySelector('.cockpit').dataset.stage = state.stageMode;
  $('orientation').hidden = state.stageMode !== '3d' || state.view.name !== 'top';
  updatePipBadge();
  draw();
}

function setView(name, routeFocus = false) {
  const nextStage = name === 'video' ? 'video' : '3d';
  const stageSwapping = nextStage !== state.stageMode;

  if (!stageSwapping || !document.startViewTransition) {
    applyStageDom(name);
    if (routeFocus) $('cockpit-main')?.focus({ preventScroll: true });
    return;
  }

  const transition = document.startViewTransition(() => {
    applyStageDom(name);
  });
  transition.finished.finally(() => {
    if (routeFocus) $('cockpit-main')?.focus({ preventScroll: true });
  });
}

let drag = null;
cloud.addEventListener('pointerdown', (event) => {
  if (!event.isPrimary || event.button !== 0) return;
  drag = { x: event.clientX, y: event.clientY, moved: false };
  cloud.setPointerCapture(event.pointerId);
});
cloud.addEventListener('pointermove', (event) => {
  if (!drag || !event.isPrimary || !state.transform) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 3) return;
  drag = { x: event.clientX, y: event.clientY, moved: true };
  cloud.classList.add('dragging');
  const view = state.view;
  if (view.name === 'top') {
    const scale = view.distance / state.transform.height;
    view.target[0] += dy * scale;
    view.target[1] += dx * scale;
  } else {
    view.yaw += dx * .007;
    view.pitch = Math.max(-1.4, Math.min(1.4, view.pitch + dy * .007));
  }
  drawCloud();
});
cloud.addEventListener('pointerup', (event) => {
  if (!event.isPrimary || !drag) return;
  if (!drag.moved) pick(cloud, event);
  drag = null;
  cloud.classList.remove('dragging');
});
cloud.addEventListener('lostpointercapture', () => {
  drag = null;
  cloud.classList.remove('dragging');
});
cloud.addEventListener('wheel', (event) => {
  event.preventDefault();
  const distance = state.view.distance * Math.exp(Math.max(-120, Math.min(120, event.deltaY)) * .002);
  state.view.distance = Math.max(3, Math.min(240, distance));
  drawCloud();
}, { passive: false });
camera.addEventListener('click', (event) => pick(camera, event));

async function openResource(path, signal) {
  const offline = document.getElementById('sample-data');
  if (!offline) return fetch(path, { signal });
  const data = offlineResources()[path];
  if (!data) return new Response(null, { status: 404 });
  signal?.throwIfAborted();
  const [type, content] = data;
  if (type === 'application/json') {
    return new Response(JSON.stringify(content), { headers: { 'Content-Type': type } });
  }
  const bytes = Uint8Array.from(atob(content), (char) => char.charCodeAt(0));
  if (type === 'application/octet-stream') {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream, { headers: { 'Content-Type': type } });
  }
  return new Response(bytes, { headers: { 'Content-Type': type } });
}

let embeddedResources;
function offlineResources() {
  embeddedResources ??= JSON.parse(document.getElementById('sample-data').textContent);
  return embeddedResources;
}

async function request(path, signal) {
  const response = await openResource(path, signal);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response;
}

async function loadPoints(base, signal) {
  if (pointCache.has(base)) {
    const result = pointCache.get(base);
    pointCache.delete(base);
    pointCache.set(base, result);
    return result;
  }
  const response = await request(`${base}/points_bin`, signal);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength % 16) throw new Error('Incomplete point data');
  const raw = new Float32Array(bytes);
  let maxIntensity = 1;
  const sampleIntensities = [];
  const step = Math.max(1, Math.floor(raw.length / 4 / 2000)) * 4;
  for (let i = 3; i < raw.length; i += 4) {
    const val = raw[i];
    if (Number.isFinite(val)) {
      maxIntensity = Math.max(maxIntensity, val);
      if ((i - 3) % step === 0 && val > 0) sampleIntensities.push(val);
    }
  }
  let p98Intensity = maxIntensity;
  if (sampleIntensities.length > 20) {
    sampleIntensities.sort((a, b) => a - b);
    p98Intensity = Math.max(1, sampleIntensities[Math.floor(sampleIntensities.length * 0.98)]);
  }
  const streamStride = Number(response.headers.get('X-Point-Stride')) || 1;
  const result = { raw, maxIntensity, p98Intensity, streamStride };
  if (!signal.aborted) {
    pointCache.set(base, result);
    if (pointCache.size > 12) pointCache.delete(pointCache.keys().next().value);
  }
  return result;
}

async function loadImage(base, modality, signal) {
  const response = await openResource(`${base}/image/${modality}`, signal);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return createImageBitmap(await response.blob());
}

function estimateImageLight(image) {
  if (!image) return null;
  try {
    lightProbeContext.drawImage(image, 0, 0, 8, 8);
    const data = lightProbeContext.getImageData(0, 0, 8, 8).data;
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    }
    const avg = sum / 64;
    return avg > 120 ? 'BRIGHT' : avg > 55 ? 'OVERCAST' : 'LOW LIGHT';
  } catch {
    return null;
  }
}

function updateDriveHud(frame) {
  const gps = frame?.telemetry?.gps;
  const lux = frame?.telemetry?.lux?.lux;
  const speedEl = $('hud-speed');
  const unitEl = $('hud-speed-unit');
  const badgeEl = $('hud-speed-badge');
  speedEl.removeAttribute('data-alert');

  if (Number.isFinite(gps?.speed_kmh)) {
    const speed = state.speedUnit === 'mph' ? gps.speed_kmh * 0.621371 : gps.speed_kmh;
    speedEl.textContent = Math.round(speed);
    unitEl.textContent = state.speedUnit === 'mph' ? 'MPH · LIVE GPS' : 'KM/H · LIVE GPS';
    if (badgeEl) badgeEl.textContent = '128 LASERS';
  } else {
    // Smart fallback for recorded LiDAR drives without GPS: highlight nearest 3D object or live beam density
    const objs = settings?.fsdAi && settings?.fsdObjects ? frame?.perception?.objects : null;
    let nearest = null;
    if (objs?.length) {
      for (const obj of objs) {
        if (!obj.center) continue;
        const d = Math.hypot(obj.center[0], obj.center[1]);
        if (!nearest || d < nearest.dist) nearest = { dist: d, label: obj.label };
      }
    }
    if (nearest) {
      speedEl.textContent = `${nearest.dist.toFixed(1)}m`;
      if (nearest.dist < 10) speedEl.dataset.alert = 'near';
      unitEl.textContent = `NEAREST · ${nearest.label.toUpperCase()}`;
      if (badgeEl) badgeEl.textContent = 'OS1-128 · 3D FUSION';
    } else if (state.points.length) {
      const kPts = (state.points.length / 4 / 1000).toFixed(1);
      speedEl.textContent = `${kPts}k`;
      unitEl.textContent = 'ACTIVE RETURNS';
      if (badgeEl) badgeEl.textContent = 'OS1-128 · 360°';
    } else {
      speedEl.textContent = '—';
      unitEl.textContent = 'READY';
      if (badgeEl) badgeEl.textContent = 'OS1-128 · 360°';
    }
  }

  if (Number.isFinite(lux)) {
    $('hud-env').textContent = lux > 20000 ? 'SUNNY' : lux > 1000 ? 'DAYLIGHT' : 'LOW LIGHT';
  } else {
    const est = estimateImageLight(frame?.image);
    $('hud-env').textContent = est || '128-CH';
  }

  const total = state.info?.frames.length || 0;
  const progress = $('frame-progress');
  progress.max = Math.max(1, total);
  progress.value = frame ? frame.index + 1 : 0;
  $('hud-clock').textContent = frame ? `+${(frame.index * 0.1).toFixed(1)}s · 10 Hz` : '10 Hz';
}

async function fetchPerceptionData(base, signal) {
  const response = await openResource(`${base}/perception`, signal);
  if (response.status === 404 || !response.ok) return null;
  const data = await response.json();
  const parsed = {
    frame_id: data.frame_id,
    device: data.device,
    inference_ms: data.inference_ms,
    cached: data.cached,
    objects: Array.isArray(data.objects) ? data.objects : [],
    denseCount: data.dense_count || 0,
    roadMesh: decodeFloat32B64(data.road_mesh_b64),
    densePoints: decodeFloat32B64(data.dense_points_b64),
  };
  perceptionCache.set(base, parsed);
  if (perceptionCache.size > 80) perceptionCache.delete(perceptionCache.keys().next().value);
  return parsed;
}

async function loadPerception(base, signal) {
  if (document.getElementById('sample-data') || !settings?.fsdAi) return null;
  if (perceptionCache.has(base)) {
    const cached = perceptionCache.get(base);
    perceptionCache.delete(base);
    perceptionCache.set(base, cached);
    return cached;
  }
  if (state.prefetch?.base === base && !state.prefetch.signal.aborted) {
    const pending = state.prefetch;
    state.prefetch = null;
    const forwardAbort = () => pending.controller.abort();
    signal?.addEventListener('abort', forwardAbort, { once: true });
    try {
      return await pending.promise;
    } finally {
      signal?.removeEventListener('abort', forwardAbort);
    }
  }
  return fetchPerceptionData(base, signal);
}

function schedulePrefetch(sequence, frames, nextIndex) {
  if (document.getElementById('sample-data') || !settings?.fsdAi) return;
  if (!frames || nextIndex < 0 || nextIndex >= frames.length) return;
  const nextBase = `/api/sequence/${encodeURIComponent(sequence)}/frame/${encodeURIComponent(frames[nextIndex])}`;
  if (perceptionCache.has(nextBase) || state.prefetch?.base === nextBase) return;
  state.prefetch?.controller.abort();
  const controller = new AbortController();
  state.prefetch = {
    base: nextBase,
    controller,
    signal: controller.signal,
    promise: fetchPerceptionData(nextBase, controller.signal).catch(() => null),
  };
}

function applyPerceptionToScene(frame) {
  const perc = settings?.fsdAi ? frame?.perception : null;
  if (perc) {
    renderer.uploadPerception(perc.roadMesh, perc.densePoints, perc.objects);
  } else {
    renderer.uploadPerception(emptyFloats, emptyFloats, []);
  }
  updateDetectionPills(perc);
  updateRouteSubtitle();
}

function stopWarmupPolling() {
  clearInterval(state.warmupTimer);
  state.warmupTimer = null;
}

async function refreshPerceptionStatus(seqId = state.info?.id) {
  const el = $('warmup-status');
  const btn = $('warmup-btn');
  if (!el) return null;
  if (!seqId || document.getElementById('sample-data')) {
    el.textContent = 'Offline bundle (AI disabled)';
    if (btn) btn.disabled = true;
    return null;
  }
  try {
    const response = await openResource(`/api/sequence/${encodeURIComponent(seqId)}/perception/status`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const st = await response.json();
    const dev = (st.device || 'cpu').toUpperCase();
    if (st.read_only) {
      el.textContent = `${st.cached_frames} / ${st.total_frames} cached · read-only`;
      if (btn) btn.disabled = true;
    } else {
      el.textContent = `${st.cached_frames} / ${st.total_frames} cached (${dev})`;
      if (btn && !state.warmupTimer) btn.disabled = false;
    }
    return st;
  } catch {
    el.textContent = 'AI perception unavailable';
    return null;
  }
}

function replaceFrame(frame) {
  state.frame?.image?.close();
  state.frame = frame;
  readSettings();
  filterPoints();
  applyPerceptionToScene(frame);
  updateTelemetry(frame?.telemetry);
  updateDriveHud(frame);
  const modality = frame?.modality || $('modality').value;
  $('camera-title').textContent = imageTypes[modality][0];
  $('camera-subtitle').textContent = imageTypes[modality][1];
  $('camera-empty').textContent = frame ? 'No image for this sensor in this frame.' : 'Loading camera…';
  $('frame-id').value = frame?.id || '······';
  $('frame-position').textContent = `${frame ? frame.index + 1 : 0} / ${state.info?.frames.length || 0}`;
  syncSliderVars($('timeline'));
  draw();
}

async function selectFrame(index) {
  if (!state.info || index < 0 || index >= state.info.frames.length) return false;
  const { id: sequence, frames, calibration } = state.info;
  const id = frames[index];
  const modality = $('modality').value;
  state.frameRequest?.abort();
  const controller = new AbortController();
  const { signal } = controller;
  state.frameRequest = controller;
  state.requestedIndex = index;
  $('timeline').value = index;
  $('timeline').setAttribute('aria-valuetext', `Frame ${id}, ${index + 1} of ${frames.length}`);
  syncSliderVars($('timeline'));
  setStatus('');
  const base = `/api/sequence/${encodeURIComponent(sequence)}/frame/${encodeURIComponent(id)}`;
  if (state.prefetch?.base && state.prefetch.base !== base) {
    state.prefetch.controller.abort();
    state.prefetch = null;
  }
  const reusable = state.frame?.id === id && state.frame.modality === modality && state.frame.image;
  const [points, image, telemetry, perception] = await Promise.allSettled([
    loadPoints(base, signal),
    reusable ? Promise.resolve(null) : loadImage(base, modality, signal),
    request(`${base}/telemetry`, signal).then((response) => response.json()),
    loadPerception(base, signal),
  ]);
  if (signal.aborted) {
    image.value?.close();
    return false;
  }
  const frameImage = reusable || image.value || null;
  if (reusable) state.frame.image = null;
  replaceFrame({
    id, index, modality, ...points.value, raw: points.value?.raw || emptyPoints,
    image: frameImage,
    telemetry: telemetry.value,
    perception: perception.status === 'fulfilled' ? perception.value : null,
    projection: cameraProjection(calibration, modality, frameImage),
  });
  const failures = [
    points.status === 'rejected' && 'laser returns',
    image.status === 'rejected' && 'camera image',
    telemetry.status === 'rejected' && 'telemetry',
  ].filter(Boolean);
  if (failures.length) setStatus(`Couldn’t load ${failures.join(', ')} for frame ${id}.`);
  updateControls();
  schedulePrefetch(sequence, frames, index + 1);
  return true;
}

function updateControls() {
  const total = state.info?.frames.length || 0;
  const index = state.requestedIndex;
  $('timeline').disabled = total < 2;
  $('play').disabled = total < 2;
  $('previous').disabled = index <= 0;
  $('next').disabled = index >= total - 1;
}

async function selectSequence(id) {
  stop();
  stopWarmupPolling();
  state.sequenceRequest?.abort();
  state.frameRequest?.abort();
  state.prefetch?.controller.abort();
  state.prefetch = null;
  const controller = new AbortController();
  state.sequenceRequest = controller;
  state.info = null;
  state.requestedIndex = 0;
  pointCache.clear();
  perceptionCache.clear();
  replaceFrame(null);
  updateControls();
  $('cloud-empty').textContent = 'Waking up the roof lasers…';
  $('source').hidden = true;
  $('timeline').value = 0;
  syncSliderVars($('timeline'));
  setStatus('');
  try {
    const info = await request(`/api/sequence/${encodeURIComponent(id)}/info`, controller.signal).then((response) => response.json());
    if (controller.signal.aborted) return;
    state.info = info;
    $('source').textContent = id === 'sample_durlar_sequence'
      ? 'Sim Data'
      : /^DurLAR_\d{8}_S$/.test(id) ? 'Real Drive' : 'Local Route';
    $('source').hidden = false;
    $('timeline').max = Math.max(0, info.frames.length - 1);
    syncSliderVars($('timeline'));
    updateControls();
    refreshPerceptionStatus(id);
    if (info.frames.length) await selectFrame(0);
    else setStatus('This route has no frames.');
  } catch (error) {
    if (!controller.signal.aborted) setStatus('Couldn’t open this route.');
  }
}

function updateTelemetry(data) {
  const fixed = (value, digits, unit) => Number.isFinite(value) ? value.toFixed(digits) + unit : '';
  const vector = (sensor, keys, digits, unit) => {
    const values = keys.map((key) => sensor?.[key]);
    return values.every(Number.isFinite) ? values.map((value) => value.toFixed(digits)).join(', ') + unit : '';
  };
  const rows = [
    ['Speed', fixed(data?.gps?.speed_kmh, 1, ' km/h')],
    ['Heading', fixed(data?.gps?.heading, 1, '°')],
    ['Latitude', fixed(data?.gps?.latitude, 5, '°')],
    ['Longitude', fixed(data?.gps?.longitude, 5, '°')],
    ['Altitude', fixed(data?.gps?.altitude, 1, ' m')],
    ['Ambient light', fixed(data?.lux?.lux, 1, ' lux')],
    ['Linear accel (x, y, z)', vector(data?.imu, ['accel_x', 'accel_y', 'accel_z'], 2, ' m/s²')],
    ['Angular rate (x, y, z)', vector(data?.imu, ['gyro_x', 'gyro_y', 'gyro_z'], 3, ' rad/s')],
  ].filter(([, value]) => value);
  $('telemetry').replaceChildren(...(rows.length ? rows : [
    ['Sensor mode', 'Ouster OS1-128 + MultiSense S21'],
    ['GNSS / IMU', 'Not recorded in DurLAR_S exemplar subset'],
  ]).map(([label, value]) => {
    const field = document.createElement('div');
    const name = document.createElement('dt');
    const reading = document.createElement('dd');
    name.textContent = label;
    reading.textContent = value;
    field.append(name, reading);
    return field;
  }));
}

function stop() {
  state.playbackRun++;
  state.playing = false;
  clearTimeout(state.timer);
  $('play').textContent = '▶';
  $('play').setAttribute('aria-label', 'Play (Space)');
  $('play').title = 'Play (Space)';
}

function schedule(delay, run) {
  state.timer = setTimeout(async () => {
    if (!state.playing || run !== state.playbackRun || !state.info) return;
    const hasNext = state.requestedIndex + 1 < state.info.frames.length;
    if (!hasNext && $('loop').getAttribute('aria-pressed') === 'false') {
      stop();
      return;
    }
    const started = performance.now();
    const committed = await selectFrame(hasNext ? state.requestedIndex + 1 : 0);
    if (committed && run === state.playbackRun) {
      schedule(Math.max(0, 100 / Number($('speed').value) - (performance.now() - started)), run);
    }
  }, delay);
}

function seek(index) {
  stop();
  selectFrame(index);
  updateControls();
}

function togglePlay() {
  if (state.playing) {
    stop();
    return;
  }
  if ($('play').disabled) return;
  state.playing = true;
  $('play').textContent = 'Ⅱ';
  $('play').setAttribute('aria-label', 'Pause (Space)');
  $('play').title = 'Pause (Space)';
  schedule(100 / Number($('speed').value), ++state.playbackRun);
}

function toggleFsdAi() {
  const button = $('fsd-toggle');
  if (!button) return;
  const enabled = button.getAttribute('aria-pressed') === 'false';
  button.setAttribute('aria-pressed', String(enabled));
  readSettings();
  if (!enabled) {
    state.prefetch?.controller.abort();
    state.prefetch = null;
    applyPerceptionToScene(state.frame);
    updateDriveHud(state.frame);
    draw();
    return;
  }
  if (state.frame && !state.frame.perception && !document.getElementById('sample-data')) {
    selectFrame(state.requestedIndex);
  } else {
    applyPerceptionToScene(state.frame);
    updateDriveHud(state.frame);
    draw();
  }
}

async function loadSequences() {
  try {
    const { sequences } = await request('/api/sequences').then((response) => response.json());
    if (!sequences.length) {
      $('sequence').replaceChildren(new Option('No routes found'));
      setStatus('No routes found under data/. Add a sequence or run generate_sample_data.py.');
      $('cloud-empty').textContent = 'No route loaded.';
      $('camera-empty').textContent = 'No route loaded.';
      return;
    }
    $('sequence').replaceChildren(...sequences.map((source) => new Option(
      routeAliases[source.id] || source.id.replace(/durlar/gi, 'SpinnyHat'),
      source.id,
    )));
    $('sequence').disabled = false;
    const recorded = sequences.find((source) => /^DurLAR_\d{8}_S$/.test(source.id));
    if (recorded) $('sequence').value = recorded.id;
    await selectSequence($('sequence').value);
  } catch (error) {
    setStatus('Couldn’t reach the viewer server. Start it with ./run_viewer.sh and reload.');
    $('cloud-empty').textContent = 'No route loaded.';
    $('camera-empty').textContent = 'No route loaded.';
  }
}

for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => setView(button.dataset.view));
}
$('speed-toggle')?.addEventListener('click', () => {
  state.speedUnit = state.speedUnit === 'kmh' ? 'mph' : 'kmh';
  updateDriveHud(state.frame);
});
$('reset-view').addEventListener('click', () => {
  if (state.stageMode === 'video') {
    state.view = preset('chase');
    updatePipBadge();
    draw();
  } else {
    setView(state.view.name);
  }
});
$('minimap-toggle').addEventListener('click', () => {
  setView(state.stageMode === '3d' && state.view.name === 'top' ? 'video' : 'top');
});
$('swap-stage').addEventListener('click', () => {
  setView(state.stageMode === 'video' ? state.view.name : 'video', true);
});
$('expand-camera').addEventListener('click', () => {
  const card = $('camera-card');
  const expanded = card.classList.toggle('expanded');
  const button = $('expand-camera');
  button.setAttribute('aria-pressed', String(expanded));
  button.textContent = expanded ? '⤡' : '⤢';
  button.setAttribute('aria-label', expanded ? 'Shrink picture-in-picture view' : 'Expand picture-in-picture view');
});
$('sequence').addEventListener('change', () => selectSequence($('sequence').value));
$('modality').addEventListener('change', () => seek(state.requestedIndex));
$('overlay').addEventListener('change', draw);
$('clear-point').addEventListener('click', () => { state.selected = -1; draw(); });
$('timeline').addEventListener('input', () => {
  syncSliderVars($('timeline'));
  seek(Number($('timeline').value));
});
$('previous').addEventListener('click', () => seek(state.requestedIndex - 1));
$('next').addEventListener('click', () => seek(state.requestedIndex + 1));
$('play').addEventListener('click', togglePlay);
$('loop').addEventListener('click', () => {
  const button = $('loop');
  button.setAttribute('aria-pressed', String(button.getAttribute('aria-pressed') === 'false'));
});
$('fsd-toggle')?.addEventListener('click', toggleFsdAi);
$('warmup-btn')?.addEventListener('click', async () => {
  if (!state.info || document.getElementById('sample-data')) return;
  const seqId = state.info.id;
  const btn = $('warmup-btn');
  const statusEl = $('warmup-status');
  btn.disabled = true;
  statusEl.textContent = 'Starting 60-frame CoreML warmup…';
  try {
    const resp = await fetch(`/api/sequence/${encodeURIComponent(seqId)}/perception/warmup?start=${state.requestedIndex}&count=60`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const initial = await refreshPerceptionStatus(seqId);
    let prevCached = initial?.cached_frames ?? 0;
    let stale = 0;
    let ticks = 0;
    stopWarmupPolling();
    state.warmupTimer = setInterval(async () => {
      ticks++;
      if (state.info?.id !== seqId || ticks > 90) {
        stopWarmupPolling();
        btn.disabled = false;
        return;
      }
      const cur = await refreshPerceptionStatus(seqId);
      if (!cur) {
        stopWarmupPolling();
        btn.disabled = false;
        return;
      }
      stale = cur.cached_frames === prevCached ? stale + 1 : 0;
      prevCached = cur.cached_frames;
      if (cur.cached_frames >= cur.total_frames || stale >= 5) {
        stopWarmupPolling();
        btn.disabled = false;
      }
    }, 1200);
  } catch {
    statusEl.textContent = 'Warmup unavailable';
    btn.disabled = false;
  }
});
$('settings-popover')?.addEventListener('toggle', (event) => {
  if (event.newState === 'open' && state.info?.id) refreshPerceptionStatus(state.info.id);
});
for (const id of ['range', 'min-height', 'max-height', 'intensity', 'stride', 'color', 'point-size', 'opacity', 'overlay-size', 'grid', 'vehicle', 'axes', 'fsd-road', 'fsd-depth', 'fsd-objects']) {
  $(id)?.addEventListener('input', () => {
    readSettings();
    if (['range', 'min-height', 'max-height', 'intensity', 'stride'].includes(id) && state.frame) filterPoints();
    if (['fsd-objects'].includes(id)) {
      updateDetectionPills(settings.fsdAi ? state.frame?.perception : null);
      updateDriveHud(state.frame);
    }
    draw();
  });
}
$('reset-settings').addEventListener('click', () => {
  for (const id of Object.keys(sliders)) $(id).value = $(id).defaultValue;
  for (const id of ['grid', 'vehicle', 'axes', 'fsd-road', 'fsd-depth', 'fsd-objects']) {
    if ($(id)) $(id).checked = $(id).defaultChecked;
  }
  $('fsd-toggle')?.setAttribute('aria-pressed', 'true');
  $('stride').selectedIndex = 0;
  $('color').selectedIndex = 0;
  readSettings();
  if (state.frame && !state.frame.perception && !document.getElementById('sample-data')) {
    selectFrame(state.requestedIndex);
    return;
  }
  applyPerceptionToScene(state.frame);
  filterPoints();
  updateDriveHud(state.frame);
  draw();
});
window.addEventListener('keydown', (event) => {
  if (event.target.matches('button, input, select, summary, a') || !state.info?.frames.length) return;
  if (event.code === 'Space') {
    event.preventDefault();
    if (!event.repeat) togglePlay();
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    seek(state.requestedIndex + (event.key === 'ArrowLeft' ? -1 : 1));
  } else if (event.key >= '1' && event.key <= '5') {
    const map = ['video', 'chase', 'top', 'ego', 'front'];
    setView(map[Number(event.key) - 1], true);
  } else if (event.key === 'f' || event.key === 'F') {
    toggleFsdAi();
  }
});
const resize = new ResizeObserver(draw);
resize.observe($('cloud-scene'));
resize.observe($('camera-scene'));
cloud.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  setStatus('The point cloud lost its graphics context. Reload the page to restore it.');
});
readSettings();
syncAllSliders();
updatePipBadge();
renderer.upload(emptyPoints);
renderer.uploadPerception(emptyFloats, emptyFloats, []);
loadSequences();
