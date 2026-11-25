const radians = Math.PI / 180;
const near = 0.15;
const far = 500;
const fieldOfView = 54 * radians;
const ground = -1.68;

export const palettes = {
  range: ['#5bf7cc', '#38bdf8', '#6366f1'],
  height: ['#64748b', '#38bdf8', '#fbbf24'],
  intensity: ['#2563eb', '#c084fc', '#fde047'],
};

export function colorAt(value, mode) {
  const colors = palettes[mode];
  const level = Math.min(2, Math.max(0, value * 2));
  const start = Math.min(1, Math.floor(level));
  const rgb = [1, 3, 5].map((offset) => {
    const a = parseInt(colors[start].slice(offset, offset + 2), 16);
    const b = parseInt(colors[start + 1].slice(offset, offset + 2), 16);
    return Math.round(a + (b - a) * (level - start));
  });
  return `rgb(${rgb.join(',')})`;
}

const normalize = (p) => {
  const length = Math.hypot(...p);
  return p.map((v) => v / length);
};
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function preset(name) {
  const views = {
    chase: { target: [10.5, 0, -0.15], distance: 20.5, yaw: Math.PI, pitch: 18 * radians },
    orbit: { target: [12, 0, -0.35], distance: 25, yaw: 162 * radians, pitch: 22 * radians },
    top: { target: [14, 0, 0], distance: 56, yaw: Math.PI, pitch: Math.PI / 2 },
    ego: { target: [26, 0, -0.4], distance: 24, yaw: Math.PI, pitch: 2 * radians },
    front: { target: [2, 0, -0.4], distance: 24, yaw: 0, pitch: 12 * radians },
  };
  const selected = views[name] || views.chase;
  return { ...selected, name: views[name] ? name : 'chase' };
}

export function viewTransform(view, width, height) {
  const { yaw, pitch, target, distance } = view;
  const outward = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
  const eye = target.map((v, i) => v + outward[i] * distance);
  const forward = outward.map((v) => -v);
  const right = normalize(cross(forward, [0, 0, 1]));
  const up = cross(right, forward);
  const aspect = width / height;
  const orthographic = view.name === 'top';
  const scaleY = orthographic ? 2 / distance : 1 / Math.tan(fieldOfView / 2);
  return { eye, right, up, forward, scaleX: scaleY / aspect, scaleY, orthographic, width, height };
}

export function screenPoint(points, i, view, clip = true) {
  const dx = points[i] - view.eye[0], dy = points[i + 1] - view.eye[1], dz = points[i + 2] - view.eye[2];
  const { forward, right, up, scaleX, scaleY, width, height } = view;
  const depth = dx * forward[0] + dy * forward[1] + dz * forward[2];
  if (depth <= near || depth >= far) return null;
  const w = view.orthographic ? 1 : depth;
  const x = scaleX * (dx * right[0] + dy * right[1] + dz * right[2]) / w;
  const y = scaleY * (dx * up[0] + dy * up[1] + dz * up[2]) / w;
  if (clip && (Math.abs(x) > 1 || Math.abs(y) > 1)) return null;
  return [(x + 1) * width / 2, (1 - y) * height / 2, depth];
}

export function resizeCanvas(canvas) {
  const { clientWidth: width, clientHeight: height } = canvas;
  const ratio = Math.min(devicePixelRatio, 2);
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  return { width, height, ratio };
}

function lineGeometry() {
  const grid = [], vehicle = [], axes = [];
  function line(target, from, to, color) {
    target.push(...from, ...color, ...to, ...color);
  }
  for (let i = -120; i <= 120; i += 4) {
    const major = i % 20 === 0;
    const color = major ? [.14, .21, .31] : [.08, .12, .18];
    line(grid, [i, -120, ground], [i, 120, ground], color);
    line(grid, [-120, i, ground], [120, i, ground], color);
  }
  // Forward distance arcs on the road plane (10m, 20m, 35m, 50m)
  for (const radius of [10, 20, 35, 50]) {
    const segs = 40;
    const arcColor = radius === 20 ? [.18, .36, .58] : [.11, .19, .30];
    for (let s = 0; s < segs; s++) {
      const a0 = -Math.PI * 0.48 + (Math.PI * 0.96) * (s / segs);
      const a1 = -Math.PI * 0.48 + (Math.PI * 0.96) * ((s + 1) / segs);
      line(
        grid,
        [Math.cos(a0) * radius, Math.sin(a0) * radius, ground + 0.01],
        [Math.cos(a1) * radius, Math.sin(a1) * radius, ground + 0.01],
        arcColor,
      );
    }
  }

  // Subtle lane rail guides flanking the forward path corridor
  line(vehicle, [1.2, -0.62, ground + 0.03], [32, -0.62, ground + 0.03], [0.16, 0.56, 0.98]);
  line(vehicle, [1.2, 0.62, ground + 0.03], [32, 0.62, ground + 0.03], [0.16, 0.56, 0.98]);

  // Sensor axes
  line(axes, [0, 0, 0], [3.5, 0, 0], [1, .38, .38]);
  line(axes, [0, 0, 0], [0, 3.5, 0], [.36, .95, .58]);
  line(axes, [0, 0, 0], [0, 0, 3.5], [.38, .68, 1]);
  return { grid, vehicle, axes };
}

function egoAndPathMeshes() {
  const verts = [];
  function quad(p0, p1, p2, p3, c0, c1 = c0, c2 = c0, c3 = c0) {
    verts.push(
      ...p0, ...c0, ...p1, ...c1, ...p2, ...c2,
      ...p0, ...c0, ...p2, ...c2, ...p3, ...c3,
    );
  }
  function box(x0, x1, y0, y1, z0, z1, topCol, sideCol, backCol, frontCol = sideCol) {
    // Top, Bottom
    quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], topCol);
    // Left (+Y), Right (-Y)
    quad([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1], sideCol);
    quad([x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0], sideCol);
    // Rear (-X), Front (+X)
    quad([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1], backCol);
    quad([x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0], frontCol);
  }

  // Dark metallic EV silhouette at origin (inspired by compact EV / FSD blue-slate car)
  const bodyTop = [0.18, 0.30, 0.52, 1.0];
  const bodySide = [0.11, 0.19, 0.34, 1.0];
  const bodyRear = [0.14, 0.24, 0.42, 1.0];
  const glass = [0.05, 0.08, 0.13, 1.0];
  const wheel = [0.05, 0.06, 0.08, 1.0];
  const roofLidar = [0.22, 0.74, 0.98, 1.0];
  const tailLight = [1.0, 0.18, 0.28, 1.0];
  const headLight = [0.85, 0.95, 1.0, 1.0];
  const zBase = ground + 0.14;

  // Lower chassis body
  box(-1.32, 1.38, -0.68, 0.68, zBase, zBase + 0.52, bodyTop, bodySide, bodyRear, bodySide);
  // Sloped hood & bumper accent
  box(0.55, 1.42, -0.64, 0.64, zBase + 0.44, zBase + 0.58, bodyTop, bodySide, bodyRear, bodySide);
  // Upper glasshouse / cabin
  box(-0.78, 0.62, -0.54, 0.54, zBase + 0.52, zBase + 1.18, bodyTop, glass, glass, glass);
  // Roof LiDAR puck ("Spinny Hat" OS1-128 on roof)
  box(-0.14, 0.14, -0.14, 0.14, zBase + 1.18, zBase + 1.38, roofLidar, [0.15, 0.22, 0.32, 1.0], [0.15, 0.22, 0.32, 1.0]);
  // Wheels
  for (const wx of [-0.88, 0.88]) {
    for (const wy of [-0.70, 0.58]) {
      box(wx - 0.25, wx + 0.25, wy, wy + 0.12, ground + 0.02, zBase + 0.34, wheel, wheel, wheel);
    }
  }
  // Glowing LED taillights on rear (-X)
  quad([-1.33, -0.60, zBase + 0.36], [-1.33, -0.26, zBase + 0.36], [-1.33, -0.26, zBase + 0.46], [-1.33, -0.60, zBase + 0.46], tailLight);
  quad([-1.33, 0.26, zBase + 0.36], [-1.33, 0.60, zBase + 0.36], [-1.33, 0.60, zBase + 0.46], [-1.33, 0.26, zBase + 0.46], tailLight);
  // Subtle front headlights (+X)
  quad([1.43, -0.58, zBase + 0.34], [1.43, -0.28, zBase + 0.34], [1.43, -0.28, zBase + 0.44], [1.43, -0.58, zBase + 0.44], headLight);
  quad([1.43, 0.28, zBase + 0.34], [1.43, 0.58, zBase + 0.34], [1.43, 0.58, zBase + 0.44], [1.43, 0.28, zBase + 0.44], headLight);

  // Subtle dark shadow pad under vehicle
  quad([-1.55, -0.82, ground + 0.015], [1.58, -0.82, ground + 0.015], [1.58, 0.82, ground + 0.015], [-1.55, 0.82, ground + 0.015], [0.02, 0.03, 0.05, 0.88]);

  const carVertexCount = verts.length / 7;

  // Glowing translucent FSD blue forward trajectory ribbon
  const segs = 24;
  const startX = 1.35;
  const endX = 34.0;
  for (let i = 0; i < segs; i++) {
    const t0 = i / segs;
    const t1 = (i + 1) / segs;
    const x0 = startX + (endX - startX) * t0;
    const x1 = startX + (endX - startX) * t1;
    const a0 = 0.64 * Math.pow(1 - t0, 1.35);
    const a1 = 0.64 * Math.pow(1 - t1, 1.35);
    const c0 = [0.16, 0.56, 1.0, a0];
    const c1 = [0.16, 0.56, 1.0, a1];
    quad([x0, -0.58, ground + 0.025], [x1, -0.58, ground + 0.025], [x1, 0.58, ground + 0.025], [x0, 0.58, ground + 0.025], c0, c1, c1, c0);
  }

  return {
    data: new Float32Array(verts),
    carCount: carVertexCount,
    totalCount: verts.length / 7,
  };
}

const backdropVertex = `
  attribute vec2 pos;
  varying vec2 ndc;
  void main() {
    ndc = pos;
    gl_Position = vec4(pos, 0.999, 1.0);
  }
`;

const backdropFragment = `
  precision mediump float;
  uniform vec3 forward, right, up;
  uniform vec2 scale;
  uniform float isTop;
  varying vec2 ndc;
  void main() {
    if (isTop > 0.5) {
      gl_FragColor = vec4(0.035, 0.05, 0.075, 1.0);
      return;
    }
    vec3 ray = normalize(forward + right * (ndc.x / max(0.001, scale.x)) + up * (ndc.y / max(0.001, scale.y)));
    float elev = ray.z;
    vec3 skyTop = vec3(0.035, 0.055, 0.095);
    vec3 horizon = vec3(0.105, 0.165, 0.245);
    vec3 asphalt = vec3(0.04, 0.055, 0.08);
    vec3 col = elev >= 0.0
      ? mix(horizon, skyTop, smoothstep(0.0, 0.42, elev))
      : mix(horizon, asphalt, smoothstep(0.0, 0.16, -elev));
    gl_FragColor = vec4(col, 1.0);
  }
`;

const meshVertex = `
  attribute vec3 pos;
  attribute vec4 color;
  uniform vec3 eye, right, up, forward;
  uniform vec2 scale;
  uniform float isTop;
  varying vec4 vColor;
  void main() {
    vec3 relative = pos - eye;
    float z = dot(relative, forward);
    vec2 xy = vec2(dot(relative, right), dot(relative, up)) * scale;
    float w = mix(z, 1.0, isTop);
    float depth = mix(1.00060018 * z - .30009003, (z - .15) / 499.85 * 2.0 - 1.0, isTop);
    gl_Position = vec4(xy, depth, w);
    vColor = color;
  }
`;

const meshFragment = `
  precision mediump float;
  varying vec4 vColor;
  void main() { gl_FragColor = vColor; }
`;

const denseVertex = `
  attribute vec3 pos;
  attribute vec3 rgb;
  uniform vec3 eye, right, up, forward;
  uniform vec2 scale;
  uniform float isTop, pointSize;
  varying vec3 vColor;
  void main() {
    vec3 relative = pos - eye;
    float z = dot(relative, forward);
    vec2 xy = vec2(dot(relative, right), dot(relative, up)) * scale;
    float w = mix(z, 1.0, isTop);
    float depth = mix(1.00060018 * z - .30009003, (z - .15) / 499.85 * 2.0 - 1.0, isTop);
    gl_Position = vec4(xy, depth, w);
    gl_PointSize = pointSize;
    vColor = rgb;
  }
`;

const denseFragment = `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float alpha = smoothstep(0.5, 0.22, d) * 0.72;
    gl_FragColor = vec4(vColor, alpha);
  }
`;

function buildObjectMeshes(objects) {
  if (!objects || !objects.length) return new Float32Array(0);
  const verts = [];

  function pushTri(a, b, c, color) {
    verts.push(...a, ...color, ...b, ...color, ...c, ...color);
  }
  function pushQuad(a, b, c, d, color) {
    pushTri(a, b, c, color);
    pushTri(a, c, d, color);
  }
  function rotBox(cx, cy, cz, l, w, h, yaw, topCol, sideCol, backCol) {
    const cs = Math.cos(yaw), sn = Math.sin(yaw);
    const hl = l * 0.5, hw = w * 0.5, hh = h * 0.5;
    const pt = (lx, ly, lz) => [cx + lx * cs - ly * sn, cy + lx * sn + ly * cs, cz + lz];
    const p000 = pt(-hl, -hw, -hh), p100 = pt(hl, -hw, -hh);
    const p110 = pt(hl, hw, -hh), p010 = pt(-hl, hw, -hh);
    const p001 = pt(-hl, -hw, hh), p101 = pt(hl, -hw, hh);
    const p111 = pt(hl, hw, hh), p011 = pt(-hl, hw, hh);
    pushQuad(p001, p101, p111, p011, topCol);
    pushQuad(p010, p110, p111, p011, sideCol);
    pushQuad(p000, p001, p101, p100, sideCol);
    pushQuad(p000, p010, p011, p001, backCol);
    pushQuad(p100, p101, p111, p110, backCol);
  }
  function haloRing(cx, cy, z, rx, ry, innerCol, outerCol) {
    const segs = 20;
    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const p0i = [cx + Math.cos(a0) * rx * 0.72, cy + Math.sin(a0) * ry * 0.72, z];
      const p1i = [cx + Math.cos(a1) * rx * 0.72, cy + Math.sin(a1) * ry * 0.72, z];
      const p1o = [cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, z];
      const p0o = [cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, z];
      verts.push(...p0i, ...innerCol, ...p1i, ...innerCol, ...p1o, ...outerCol);
      verts.push(...p0i, ...innerCol, ...p1o, ...outerCol, ...p0o, ...outerCol);
    }
  }

  for (const obj of objects) {
    const [cx, cy, cz] = obj.center;
    const [l, w, h] = obj.size;
    const yaw = obj.yaw || 0;
    const zBase = cz - h * 0.5;

    if (obj.label === 'person' || obj.label === 'dog') {
      const torsoTop = [0.22, 0.85, 0.96, 0.92];
      const torsoSide = [0.14, 0.62, 0.82, 0.88];
      const headCol = [0.90, 0.96, 1.0, 0.96];
      const bodyH = h * 0.72;
      rotBox(cx, cy, zBase + bodyH * 0.5, 0.36, 0.46, bodyH, yaw, torsoTop, torsoSide, torsoSide);
      rotBox(cx, cy, zBase + bodyH + h * 0.14, 0.28, 0.28, h * 0.24, yaw, headCol, headCol, headCol);
      haloRing(cx, cy, ground + 0.03, 0.62, 0.62, [0.22, 0.85, 0.96, 0.55], [0.22, 0.85, 0.96, 0.0]);
    } else if (obj.label === 'car' || obj.label === 'bus' || obj.label === 'truck') {
      const isHeavy = obj.label !== 'car';
      const topCol = isHeavy ? [0.68, 0.76, 0.86, 0.92] : [0.80, 0.86, 0.93, 0.92];
      const sideCol = isHeavy ? [0.46, 0.54, 0.66, 0.88] : [0.56, 0.64, 0.75, 0.88];
      const glassCol = [0.10, 0.16, 0.24, 0.94];
      const lowH = h * 0.46;
      const cabH = h * 0.44;
      rotBox(cx, cy, zBase + lowH * 0.5 + 0.08, l, w, lowH, yaw, topCol, sideCol, sideCol);
      rotBox(cx, cy, zBase + lowH + cabH * 0.5 + 0.08, l * (isHeavy ? 0.88 : 0.56), w * 0.82, cabH, yaw, topCol, glassCol, glassCol);
      haloRing(cx, cy, ground + 0.025, l * 0.60, w * 0.72, [0.38, 0.68, 1.0, 0.34], [0.38, 0.68, 1.0, 0.0]);
    } else if (obj.label === 'bicycle' || obj.label === 'motorcycle') {
      rotBox(cx, cy, zBase + h * 0.35, l, 0.38, h * 0.55, yaw, [0.98, 0.75, 0.24, 0.90], [0.76, 0.54, 0.16, 0.85], [0.76, 0.54, 0.16, 0.85]);
      haloRing(cx, cy, ground + 0.03, 0.95, 0.55, [0.98, 0.75, 0.24, 0.45], [0.98, 0.75, 0.24, 0.0]);
    } else {
      rotBox(cx, cy, cz, l, w, h, yaw, [0.98, 0.36, 0.42, 0.92], [0.78, 0.22, 0.28, 0.88], [0.78, 0.22, 0.28, 0.88]);
    }
  }
  return new Float32Array(verts);
}

const vertexSource = `
  attribute vec4 point;
  attribute vec3 lineColor;
  uniform vec3 eye, right, up, forward;
  uniform vec2 scale;
  uniform vec3 lowColor, midColor, highColor;
  uniform float isTop, pointSize, maxRange, minHeight, maxHeight, maxIntensity;
  uniform int mode;
  uniform bool lines;
  varying vec3 color;
  varying float isLine;
  void main() {
    vec3 relative = point.xyz - eye;
    float z = dot(relative, forward);
    vec2 xy = vec2(dot(relative, right), dot(relative, up)) * scale;
    float w = mix(z, 1.0, isTop);
    float depth = mix(1.00060018 * z - .30009003, (z - .15) / 499.85 * 2.0 - 1.0, isTop);
    gl_Position = vec4(xy, depth, w);
    gl_PointSize = pointSize;
    if (lines) {
      color = lineColor;
      isLine = 1.0;
      return;
    }
    isLine = 0.0;
    float value = length(point.xyz) / maxRange;
    if (mode == 1) value = (point.z - minHeight) / max(.001, maxHeight - minHeight);
    if (mode == 2) value = sqrt(clamp(point.w / maxIntensity, 0.0, 1.0));
    float level = clamp(value, 0.0, 1.0) * 2.0;
    vec3 base = level < 1.0 ? mix(lowColor, midColor, level) : mix(midColor, highColor, level - 1.0);
    // Subtle dark asphalt separation for ground returns so obstacles & vehicles pop in chase view
    float groundMask = 1.0 - smoothstep(-1.42, -1.24, point.z);
    if (mode != 2) {
      base = mix(base, vec3(0.14, 0.20, 0.29), groundMask * 0.62);
    } else {
      base = mix(base, base * 0.55 + vec3(0.05, 0.07, 0.11), groundMask * 0.45);
    }
    color = base;
  }
`;

const fragmentSource = `
  precision mediump float;
  varying vec3 color;
  varying float isLine;
  void main() {
    if (isLine > 0.5) {
      gl_FragColor = vec4(color, 1.0);
      return;
    }
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float alpha = smoothstep(0.5, 0.24, d);
    gl_FragColor = vec4(color, alpha);
  }
`;

function makeProgram(gl, vertSrc, fragSrc) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertSrc], [gl.FRAGMENT_SHADER, fragSrc]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

const rgb = (color) => [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16) / 255);

export function createPointRenderer(canvas) {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: true });
  if (!gl) return createCanvasRenderer(canvas);

  const backdropProg = makeProgram(gl, backdropVertex, backdropFragment);
  const backdropUniform = Object.fromEntries(
    ['forward', 'right', 'up', 'scale', 'isTop'].map((n) => [n, gl.getUniformLocation(backdropProg, n)]),
  );
  const backdropPos = gl.getAttribLocation(backdropProg, 'pos');
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);

  const meshProg = makeProgram(gl, meshVertex, meshFragment);
  const meshUniform = Object.fromEntries(
    ['eye', 'right', 'up', 'forward', 'scale', 'isTop'].map((n) => [n, gl.getUniformLocation(meshProg, n)]),
  );
  const meshPosAttr = gl.getAttribLocation(meshProg, 'pos');
  const meshColorAttr = gl.getAttribLocation(meshProg, 'color');
  const egoMesh = egoAndPathMeshes();
  const meshBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
  gl.bufferData(gl.ARRAY_BUFFER, egoMesh.data, gl.STATIC_DRAW);

  const roadBuf = gl.createBuffer();
  let roadCount = 0;
  const objBuf = gl.createBuffer();
  let objCount = 0;

  const denseProg = makeProgram(gl, denseVertex, denseFragment);
  const denseUniform = Object.fromEntries(
    ['eye', 'right', 'up', 'forward', 'scale', 'isTop', 'pointSize'].map((n) => [n, gl.getUniformLocation(denseProg, n)]),
  );
  const densePosAttr = gl.getAttribLocation(denseProg, 'pos');
  const denseRgbAttr = gl.getAttribLocation(denseProg, 'rgb');
  const denseBuf = gl.createBuffer();
  let denseCount = 0;

  const program = makeProgram(gl, vertexSource, fragmentSource);
  const uniform = Object.fromEntries([
    'eye', 'right', 'up', 'forward', 'scale', 'isTop', 'pointSize', 'maxRange', 'minHeight',
    'maxHeight', 'maxIntensity', 'mode', 'lines', 'lowColor', 'midColor', 'highColor',
  ].map((name) => [name, gl.getUniformLocation(program, name)]));
  const point = gl.getAttribLocation(program, 'point');
  const lineColor = gl.getAttribLocation(program, 'lineColor');
  const buffer = gl.createBuffer();
  const guides = Object.entries(lineGeometry()).map(([name, data]) => {
    const lineBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    return { name, buffer: lineBuffer, count: data.length / 6 };
  });
  let count = 0;
  gl.clearColor(0.035, 0.05, 0.075, 1);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  return {
    upload(points) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
      count = points.length / 4;
    },
    uploadPerception(roadMesh, densePoints, objects) {
      gl.bindBuffer(gl.ARRAY_BUFFER, roadBuf);
      if (roadMesh && roadMesh.length) {
        gl.bufferData(gl.ARRAY_BUFFER, roadMesh, gl.DYNAMIC_DRAW);
        roadCount = roadMesh.length / 7;
      } else {
        roadCount = 0;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, denseBuf);
      if (densePoints && densePoints.length) {
        gl.bufferData(gl.ARRAY_BUFFER, densePoints, gl.DYNAMIC_DRAW);
        denseCount = densePoints.length / 6;
      } else {
        denseCount = 0;
      }

      const objVerts = buildObjectMeshes(objects);
      gl.bindBuffer(gl.ARRAY_BUFFER, objBuf);
      if (objVerts.length) {
        gl.bufferData(gl.ARRAY_BUFFER, objVerts, gl.DYNAMIC_DRAW);
        objCount = objVerts.length / 7;
      } else {
        objCount = 0;
      }
    },
    draw(view, settings, ratio) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // 1. Atmospheric sky/horizon/asphalt backdrop
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.useProgram(backdropProg);
      for (const key of ['forward', 'right', 'up']) gl.uniform3fv(backdropUniform[key], view[key]);
      gl.uniform2f(backdropUniform.scale, view.scaleX, view.scaleY);
      gl.uniform1f(backdropUniform.isTop, Number(view.orthographic));
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(backdropPos);
      gl.vertexAttribPointer(backdropPos, 2, gl.FLOAT, false, 8, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.disableVertexAttribArray(backdropPos);

      // 1b. Smooth filled FSD road & terrain mesh
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      if (settings.fsdRoad && roadCount > 0) {
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1.0, 1.0);
        gl.useProgram(meshProg);
        for (const key of ['eye', 'right', 'up', 'forward']) gl.uniform3fv(meshUniform[key], view[key]);
        gl.uniform2f(meshUniform.scale, view.scaleX, view.scaleY);
        gl.uniform1f(meshUniform.isTop, Number(view.orthographic));
        gl.bindBuffer(gl.ARRAY_BUFFER, roadBuf);
        gl.enableVertexAttribArray(meshPosAttr);
        gl.enableVertexAttribArray(meshColorAttr);
        gl.vertexAttribPointer(meshPosAttr, 3, gl.FLOAT, false, 28, 0);
        gl.vertexAttribPointer(meshColorAttr, 4, gl.FLOAT, false, 28, 12);
        gl.drawArrays(gl.TRIANGLES, 0, roadCount);
        gl.disableVertexAttribArray(meshPosAttr);
        gl.disableVertexAttribArray(meshColorAttr);
        gl.disable(gl.POLYGON_OFFSET_FILL);
      }

      // 2. Grid, distance arcs, axes, and LiDAR point cloud
      gl.useProgram(program);
      for (const key of ['eye', 'right', 'up', 'forward']) gl.uniform3fv(uniform[key], view[key]);
      gl.uniform2f(uniform.scale, view.scaleX, view.scaleY);
      gl.uniform1f(uniform.isTop, Number(view.orthographic));
      gl.uniform1f(uniform.pointSize, settings.pointSize * ratio);
      for (const key of ['maxRange', 'minHeight', 'maxHeight', 'maxIntensity']) gl.uniform1f(uniform[key], settings[key]);
      gl.uniform1i(uniform.mode, Object.keys(palettes).indexOf(settings.mode));
      const [low, mid, high] = palettes[settings.mode].map(rgb);
      gl.uniform3fv(uniform.lowColor, low);
      gl.uniform3fv(uniform.midColor, mid);
      gl.uniform3fv(uniform.highColor, high);
      gl.enableVertexAttribArray(point);
      gl.enableVertexAttribArray(lineColor);
      gl.uniform1i(uniform.lines, 1);
      for (const guide of guides) {
        if (!settings[guide.name]) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, guide.buffer);
        gl.vertexAttribPointer(point, 3, gl.FLOAT, false, 24, 0);
        gl.vertexAttribPointer(lineColor, 3, gl.FLOAT, false, 24, 12);
        gl.drawArrays(gl.LINES, 0, guide.count);
      }
      gl.disableVertexAttribArray(lineColor);
      gl.uniform1i(uniform.lines, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.vertexAttribPointer(point, 4, gl.FLOAT, false, 16, 0);
      gl.drawArrays(gl.POINTS, 0, count);
      gl.disableVertexAttribArray(point);

      // 2b. Dense Depth-Anything RGB-colored foreground completion points
      if (settings.fsdDepthFill && denseCount > 0) {
        gl.useProgram(denseProg);
        for (const key of ['eye', 'right', 'up', 'forward']) gl.uniform3fv(denseUniform[key], view[key]);
        gl.uniform2f(denseUniform.scale, view.scaleX, view.scaleY);
        gl.uniform1f(denseUniform.isTop, Number(view.orthographic));
        gl.uniform1f(denseUniform.pointSize, Math.max(2.2, settings.pointSize * 1.45) * ratio);
        gl.bindBuffer(gl.ARRAY_BUFFER, denseBuf);
        gl.enableVertexAttribArray(densePosAttr);
        gl.enableVertexAttribArray(denseRgbAttr);
        gl.vertexAttribPointer(densePosAttr, 3, gl.FLOAT, false, 24, 0);
        gl.vertexAttribPointer(denseRgbAttr, 3, gl.FLOAT, false, 24, 12);
        gl.drawArrays(gl.POINTS, 0, denseCount);
        gl.disableVertexAttribArray(densePosAttr);
        gl.disableVertexAttribArray(denseRgbAttr);
      }

      // 3. Detected 3D FSD Objects + Ego Car & Translucent Blue Forward Path Ribbon
      if ((settings.fsdObjects && objCount > 0) || settings.vehicle) {
        gl.useProgram(meshProg);
        for (const key of ['eye', 'right', 'up', 'forward']) gl.uniform3fv(meshUniform[key], view[key]);
        gl.uniform2f(meshUniform.scale, view.scaleX, view.scaleY);
        gl.uniform1f(meshUniform.isTop, Number(view.orthographic));
        gl.enableVertexAttribArray(meshPosAttr);
        gl.enableVertexAttribArray(meshColorAttr);

        if (settings.fsdObjects && objCount > 0) {
          gl.bindBuffer(gl.ARRAY_BUFFER, objBuf);
          gl.vertexAttribPointer(meshPosAttr, 3, gl.FLOAT, false, 28, 0);
          gl.vertexAttribPointer(meshColorAttr, 4, gl.FLOAT, false, 28, 12);
          gl.drawArrays(gl.TRIANGLES, 0, objCount);
        }

        if (settings.vehicle) {
          gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
          gl.vertexAttribPointer(meshPosAttr, 3, gl.FLOAT, false, 28, 0);
          gl.vertexAttribPointer(meshColorAttr, 4, gl.FLOAT, false, 28, 12);
          gl.drawArrays(gl.TRIANGLES, 0, egoMesh.carCount);
          gl.depthMask(false);
          gl.drawArrays(gl.TRIANGLES, egoMesh.carCount, egoMesh.totalCount - egoMesh.carCount);
          gl.depthMask(true);
        }
        gl.disableVertexAttribArray(meshPosAttr);
        gl.disableVertexAttribArray(meshColorAttr);
      }
    },
  };
}

function createCanvasRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  const guides = lineGeometry();
  let points = new Float32Array();
  return {
    upload(values) { points = values; },
    uploadPerception() {},
    draw(view, settings, ratio) {
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.fillStyle = '#0a0f17';
      ctx.fillRect(0, 0, view.width, view.height);
      for (const [name, data] of Object.entries(guides)) {
        if (!settings[name]) continue;
        for (let i = 0; i < data.length; i += 12) {
          const a = screenPoint(data, i, view, false), b = screenPoint(data, i + 6, view, false);
          if (!a || !b) continue;
          ctx.strokeStyle = `rgb(${data.slice(i + 3, i + 6).map((v) => Math.round(v * 255)).join(',')})`;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      }
      const { minHeight, maxHeight, maxRange, maxIntensity, mode, pointSize } = settings;
      const colors = Array.from({length: 64}, (_, i) => colorAt(i / 63, mode));
      for (let i = 0; i < points.length; i += 4) {
        const p = screenPoint(points, i, view);
        if (!p) continue;
        const range = Math.hypot(points[i], points[i + 1], points[i + 2]);
        const values = {
          range: range / maxRange,
          height: (points[i + 2] - minHeight) / Math.max(.001, maxHeight - minHeight),
          intensity: Math.sqrt(Math.max(0, Math.min(1, points[i + 3] / maxIntensity))),
        };
        ctx.fillStyle = colors[Math.max(0, Math.min(63, Math.round(63 * values[mode])))];
        ctx.fillRect(p[0] - pointSize / 2, p[1] - pointSize / 2, pointSize, pointSize);
      }
    },
  };
}

export function cameraProjection(calibration, modality, image) {
  if (!calibration?.projectable_images?.includes(modality)) return null;
  const p = calibration[modality === 'image_02' ? 'P_rect_02' : 'P_rect_01'];
  const r = calibration.R_lidar_to_cam;
  const rect = calibration.R_rect_reference;
  const t = calibration.T_lidar_to_cam;
  const transform = rect.map((row) => [
    ...[0, 1, 2].map((col) => dot(row, r.map((source) => source[col]))),
    dot(row, t),
  ]);
  const size = calibration.projection_sizes?.[modality];
  const scale = size && image ? [image.width / size[0], image.height / size[1], 1] : [1, 1, 1];
  return p.map((row, index) => [
    ...[0, 1, 2].map((col) => dot(row, transform.map((source) => source[col]))),
    dot(row, transform.map((source) => source[3])) + row[3],
  ].map((value) => value * scale[index]));
}

export function projectWorldPoint(x, y, z, projection) {
  if (!projection) return null;
  const transform = (row) => row[0] * x + row[1] * y + row[2] * z + row[3];
  const depth = transform(projection[2]);
  if (depth <= near) return null;
  return [transform(projection[0]) / depth, transform(projection[1]) / depth, depth];
}

export function imagePoint(points, i, projection, width, height) {
  const projected = projectWorldPoint(points[i], points[i + 1], points[i + 2], projection);
  if (!projected) return null;
  const [u, v, depth] = projected;
  return u >= 0 && v >= 0 && u < width && v < height ? [u, v, depth] : null;
}
