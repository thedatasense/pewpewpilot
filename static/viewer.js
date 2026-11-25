import {
  palettes, colorAt, preset, viewTransform, screenPoint, resizeCanvas,
  createPointRenderer, cameraProjection, projectWorldPoint, imagePoint,
} from './point-cloud.js?v=4';

if (!Object.hasOwn(HTMLElement.prototype, 'popover')) {
  import('https://unpkg.com/@oddbird/popover-polyfill@latest/dist/popover.min.js').catch(() => {});
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
const imageTypes = {
  image_02: ['Forward camera', 'Image 02 · color'],
  image_01: ['Forward camera', 'Image 01 · monochrome'],
  ambient: ['Ambient panorama', 'Roof lidar · 360° ambient light'],
  reflec: ['Reflectivity panorama', 'Roof lidar · 360° surface returns'],
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
  points: emptyPoints, selected: -1, drawing: false, timer: null, playing: false, playbackRun: 0,
  stageMode: 'video', view: preset('chase'), transform: null, imageTransform: null, speedUnit: 'kmh',
};
const pointCache = new Map();
let settings;

function setStatus(message) {
  $('status').textContent = message;
  $('status').hidden = !message;
}

function readSettings() {
  const number = (id) => Number($(id).value);
  const maxIntensity = Math.max(1, state.frame?.p98Intensity || state.frame?.maxIntensity || 1);
  settings = {
    mode: $('color').value, maxRange: number('range'), minHeight: number('min-height'),
    maxHeight: number('max-height'), minIntensity: (number('intensity') / 100) * maxIntensity,
    maxIntensity, pointSize: number('point-size'),
    stride: number('stride'), opacity: number('opacity') / 100, overlaySize: number('overlay-size'),
    grid: $('grid').checked, vehicle: $('vehicle').checked, axes: $('axes').checked,
  };
  for (const [id, unit] of Object.entries(sliders)) {
    document.querySelector(`output[for="${id}"]`).value = $(id).value + unit;
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
  $('hud-route-sub').textContent = raw.length ? `OS1-128 · ${returnsCount} returns` : 'OS1-128 · Waiting for returns';
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
  ctx.fillStyle = '#080d14';
  ctx.fillRect(0, 0, width, height);

  const cx = width * 0.5;
  const cy = height * 0.68;
  const pxPerMeter = height / 44;

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  ctx.lineWidth = 1;
  for (const r of [10, 20, 30]) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * pxPerMeter, 0, Math.PI * 2);
    ctx.stroke();
  }

  const grad = ctx.createLinearGradient(cx, cy, cx, cy - 28 * pxPerMeter);
  grad.addColorStop(0, 'rgba(43, 127, 255, 0.55)');
  grad.addColorStop(1, 'rgba(43, 127, 255, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(cx - 0.65 * pxPerMeter, cy - 28 * pxPerMeter, 1.3 * pxPerMeter, 28 * pxPerMeter);

  const pts = state.points;
  const step = Math.max(1, Math.floor(pts.length / 4 / 3500)) * 4;
  for (let i = 0; i < pts.length; i += step) {
    const x = pts[i], y = pts[i + 1], z = pts[i + 2];
    const mx = cx - y * pxPerMeter;
    const my = cy - x * pxPerMeter;
    if (mx < 0 || mx >= width || my < 0 || my >= height) continue;
    ctx.fillStyle = z > -1.32 ? 'rgba(91, 247, 204, 0.72)' : 'rgba(71, 85, 105, 0.35)';
    ctx.fillRect(mx, my, 1.5, 1.5);
  }

  ctx.fillStyle = '#fb923c';
  ctx.beginPath();
  ctx.moveTo(cx, cy - 7);
  ctx.lineTo(cx + 5, cy + 5);
  ctx.lineTo(cx, cy + 2);
  ctx.lineTo(cx - 5, cy + 5);
  ctx.closePath();
  ctx.fill();
}

function drawCameraPathRibbon(ctx, projection, ox, oy, scale) {
  const left = [];
  const right = [];
  const zGround = -1.68;
  for (let x = 2.6; x <= 30.0; x += 2.6) {
    const pL = projectWorldPoint(x, 0.58, zGround, projection);
    const pR = projectWorldPoint(x, -0.58, zGround, projection);
    if (!pL || !pR) continue;
    left.push([ox + pL[0] * scale, oy + pL[1] * scale]);
    right.push([ox + pR[0] * scale, oy + pR[1] * scale]);
  }
  if (left.length < 2) return;

  const first = left[0], last = left[left.length - 1];
  const grad = ctx.createLinearGradient(first[0], first[1], last[0], last[1]);
  grad.addColorStop(0, 'rgba(43, 127, 255, 0.38)');
  grad.addColorStop(0.65, 'rgba(56, 189, 248, 0.22)');
  grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

  ctx.save();
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < left.length; i++) ctx.lineTo(left[i][0], left[i][1]);
  for (let i = right.length - 1; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = 'rgba(96, 165, 250, 0.55)';
  ctx.lineWidth = 1.5;
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
    ? (overlay.checked ? 'Projected lasers follow active palette & filters.' : 'Enable Project lasers to overlay 3D returns.')
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

  const selected = state.selected < 0 ? null : imagePoint(state.points, state.selected, projection, image.width, image.height);
  const marker = selected && [x + selected[0] * scale, y + selected[1] * scale];
  mark('camera-marker', marker);
  return marker;
}

function drawMeasurement(projected) {
  const i = state.selected;
  $('measurement').hidden = i < 0;
  $('clear-point').hidden = i < 0;
  if (i < 0) {
    $('inspection-help').textContent = 'Click any dot in the scan to inspect distance & match it across views.';
    return;
  }
  const [x, y, z, intensity] = state.points.subarray(i, i + 4);
  $('measure-range').textContent = `${Math.hypot(x, y, z).toFixed(2)} m`;
  $('measure-forward').textContent = `${x.toFixed(2)} m`;
  $('measure-left').textContent = `${y.toFixed(2)} m`;
  $('measure-height').textContent = `${z.toFixed(2)} m`;
  $('measure-intensity').textContent = intensity.toFixed(1);
  $('inspection-help').textContent = projected ? 'Selected return matches across both camera and 3D views.'
    : state.imageTransform ? 'Selected return falls outside the forward camera FOV.' : 'Camera projection is unavailable for this sensor view.';
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

function setView(name) {
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
  draw();
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
  if (Number.isFinite(gps?.speed_kmh)) {
    const speed = state.speedUnit === 'mph' ? gps.speed_kmh * 0.621371 : gps.speed_kmh;
    $('hud-speed').textContent = Math.round(speed);
    $('hud-speed-unit').textContent = state.speedUnit === 'mph' ? 'MPH' : 'KM/H';
  } else {
    $('hud-speed').textContent = '—';
    $('hud-speed-unit').textContent = 'NO GPS';
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

function replaceFrame(frame) {
  state.frame?.image?.close();
  state.frame = frame;
  readSettings();
  filterPoints();
  updateTelemetry(frame?.telemetry);
  updateDriveHud(frame);
  const modality = frame?.modality || $('modality').value;
  $('camera-title').textContent = imageTypes[modality][0];
  $('camera-subtitle').textContent = imageTypes[modality][1];
  $('camera-empty').textContent = frame ? 'No image for this sensor in this frame.' : 'Loading camera…';
  $('frame-id').value = frame?.id || '······';
  $('frame-position').textContent = `${frame ? frame.index + 1 : 0} / ${state.info?.frames.length || 0}`;
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
  setStatus('');
  const base = `/api/sequence/${encodeURIComponent(sequence)}/frame/${encodeURIComponent(id)}`;
  const reusable = state.frame?.id === id && state.frame.modality === modality && state.frame.image;
  const [points, image, telemetry] = await Promise.allSettled([
    loadPoints(base, signal),
    reusable ? Promise.resolve(null) : loadImage(base, modality, signal),
    request(`${base}/telemetry`, signal).then((response) => response.json()),
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
    projection: cameraProjection(calibration, modality, frameImage),
  });
  const failures = [
    points.status === 'rejected' && 'laser returns',
    image.status === 'rejected' && 'camera image',
    telemetry.status === 'rejected' && 'telemetry',
  ].filter(Boolean);
  if (failures.length) setStatus(`Couldn’t load ${failures.join(', ')} for frame ${id}.`);
  updateControls();
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
  state.sequenceRequest?.abort();
  state.frameRequest?.abort();
  const controller = new AbortController();
  state.sequenceRequest = controller;
  state.info = null;
  state.requestedIndex = 0;
  pointCache.clear();
  replaceFrame(null);
  updateControls();
  $('cloud-empty').textContent = 'Waking up the roof lasers…';
  $('source').hidden = true;
  $('timeline').value = 0;
  setStatus('');
  try {
    const info = await request(`/api/sequence/${encodeURIComponent(id)}/info`, controller.signal).then((response) => response.json());
    if (controller.signal.aborted) return;
    state.info = info;
    $('source').textContent = id === 'sample_durlar_sequence'
      ? 'Fake Sim'
      : /^DurLAR_\d{8}_S$/.test(id) ? 'Real Lasers' : 'Local Route';
    $('source').hidden = false;
    $('timeline').max = Math.max(0, info.frames.length - 1);
    updateControls();
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
    ['Light level', fixed(data?.lux?.lux, 1, ' lux')],
    ['Acceleration (x, y, z)', vector(data?.imu, ['accel_x', 'accel_y', 'accel_z'], 2, ' m/s²')],
    ['Angular velocity (x, y, z)', vector(data?.imu, ['gyro_x', 'gyro_y', 'gyro_z'], 3, ' rad/s')],
  ].filter(([, value]) => value);
  $('telemetry').replaceChildren(...(rows.length ? rows : [['No GPS/IMU/Lux telemetry in this exemplar sequence.', 'Frame & 128-beam LiDAR stats stay live in the HUD.']]).map(([label, value]) => {
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
$('reset-view').addEventListener('click', () => {
  if (state.stageMode === 'video') {
    state.view = preset('chase');
    draw();
  } else {
    setView(state.view.name);
  }
});
$('minimap-toggle').addEventListener('click', () => {
  setView(state.stageMode === '3d' && state.view.name === 'top' ? 'video' : 'top');
});
$('swap-stage').addEventListener('click', () => {
  setView(state.stageMode === 'video' ? state.view.name : 'video');
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
$('timeline').addEventListener('input', () => seek(Number($('timeline').value)));
$('previous').addEventListener('click', () => seek(state.requestedIndex - 1));
$('next').addEventListener('click', () => seek(state.requestedIndex + 1));
$('play').addEventListener('click', togglePlay);
$('loop').addEventListener('click', () => {
  const button = $('loop');
  button.setAttribute('aria-pressed', String(button.getAttribute('aria-pressed') === 'false'));
});
for (const id of ['range', 'min-height', 'max-height', 'intensity', 'stride', 'color', 'point-size', 'opacity', 'overlay-size', 'grid', 'vehicle', 'axes']) {
  $(id).addEventListener('input', () => {
    readSettings();
    if (['range', 'min-height', 'max-height', 'intensity', 'stride'].includes(id) && state.frame) filterPoints();
    draw();
  });
}
$('reset-settings').addEventListener('click', () => {
  for (const id of Object.keys(sliders)) $(id).value = $(id).defaultValue;
  for (const id of ['grid', 'vehicle', 'axes']) $(id).checked = $(id).defaultChecked;
  $('stride').selectedIndex = 0;
  $('color').selectedIndex = 0;
  readSettings();
  filterPoints();
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
renderer.upload(emptyPoints);
loadSequences();
