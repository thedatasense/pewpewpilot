use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use crate::loader::{Calibration, DurLARSequence};

const DET_SIZE: usize = 640;
const DEP_SIZE: usize = 518;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject3D {
    pub label: String,
    pub score: f64,
    pub box2d: [f64; 4],
    pub center: [f64; 3],
    pub size: [f64; 3],
    pub yaw: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionResult {
    pub frame_id: String,
    pub device: String,
    pub inference_ms: f64,
    pub cached: bool,
    pub objects: Vec<DetectedObject3D>,
    pub road_mesh_b64: String,
    pub dense_points_b64: String,
    #[serde(default)]
    pub dense_count: usize,
}

struct VisionSessions {
    device: &'static str,
    detector: Mutex<Session>,
    depth: Mutex<Session>,
}

static ENGINE: OnceLock<Result<VisionSessions, String>> = OnceLock::new();

pub fn device_name(models_dir: &Path) -> &'static str {
    let det_exists = models_dir.join("rtdetr_r50vd.onnx").is_file();
    let dep_exists = models_dir.join("depth_anything_v2_small.onnx").is_file();
    if !det_exists || !dep_exists {
        if cfg!(target_os = "macos") {
            return "coreml (models not exported)";
        }
        return "cpu (models not exported)";
    }
    if let Some(Ok(sessions)) = ENGINE.get() {
        return sessions.device;
    }
    if cfg!(target_os = "macos") {
        "coreml"
    } else {
        "cpu"
    }
}

fn get_or_init_sessions(models_dir: &Path) -> Result<&'static VisionSessions, String> {
    let res = ENGINE.get_or_init(|| {
        let det_path = models_dir.join("rtdetr_r50vd.onnx");
        let dep_path = models_dir.join("depth_anything_v2_small.onnx");
        if !det_path.is_file() || !dep_path.is_file() {
            return Err(format!(
                "ONNX models missing in {}. Run `python3 export_onnx_models.py` first.",
                models_dir.display()
            ));
        }

        let env = ort::init()
            .build()
            .map_err(|e| format!("Failed to init ONNX Runtime environment: {e}"))?;

        let cache_root = models_dir.join(".coreml_cache");
        let det_cache = cache_root.join("rtdetr");
        let dep_cache = cache_root.join("depth");
        let _ = fs::create_dir_all(&det_cache);
        let _ = fs::create_dir_all(&dep_cache);

        let (det_session, used_coreml_det) =
            build_session(&env, &det_path, false, &det_cache)?;
        let (dep_session, used_coreml_dep) =
            build_session(&env, &dep_path, true, &dep_cache)?;

        let device = if used_coreml_det || used_coreml_dep {
            "coreml"
        } else {
            "cpu"
        };

        Ok(VisionSessions {
            device,
            detector: Mutex::new(det_session),
            depth: Mutex::new(dep_session),
        })
    });

    res.as_ref().map_err(|e| e.clone())
}

pub fn init_engine_in_background(models_dir: PathBuf) {
    if models_dir.join("rtdetr_r50vd.onnx").is_file()
        && models_dir.join("depth_anything_v2_small.onnx").is_file()
    {
        std::thread::spawn(move || {
            let _ = get_or_init_sessions(&models_dir);
        });
    }
}

fn build_session(
    env: &ort::environment::Environment,
    model_path: &Path,
    use_ml_program: bool,
    cache_dir: &Path,
) -> Result<(Session, bool), String> {
    #[cfg(target_os = "macos")]
    {
        let fmt = if use_ml_program {
            ort::ep::coreml::ModelFormat::MLProgram
        } else {
            ort::ep::coreml::ModelFormat::NeuralNetwork
        };
        let coreml_ep = ort::ep::CoreML::default()
            .with_model_format(fmt)
            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
            .with_static_input_shapes(true)
            .with_model_cache_dir(cache_dir.to_string_lossy())
            .build();

        if let Ok(builder) = Session::builder(env) {
            let builder = builder
                .with_optimization_level(GraphOptimizationLevel::All)
                .unwrap_or_else(|e| e.recover());
            if let Ok(mut with_ep) = builder.with_execution_providers([coreml_ep]) {
                if let Ok(session) = with_ep.commit_from_file(model_path) {
                    return Ok((session, true));
                }
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (use_ml_program, cache_dir);

    let builder = Session::builder(env).map_err(|e| format!("Session builder error: {e}"))?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap_or_else(|e| e.recover());
    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| format!("Failed to load {}: {e}", model_path.display()))?;
    Ok((session, false))
}

pub fn perception_cache_dir(seq: &DurLARSequence) -> PathBuf {
    seq.path.join("perception")
}

/// Fast-path check: reads cached perception JSON directly from disk without decoding base64 buffers.
pub fn read_cached_perception_json(cache_dir: &Path, frame_id: &str) -> Option<String> {
    let cache_path = cache_dir.join(format!("{frame_id}.json"));
    if !cache_path.is_file() {
        return None;
    }
    let raw = fs::read_to_string(&cache_path).ok()?;
    if raw.contains("\"cached\":false") {
        Some(raw.replacen("\"cached\":false", "\"cached\":true", 1))
    } else if raw.contains("\"cached\": false") {
        Some(raw.replacen("\"cached\": false", "\"cached\": true", 1))
    } else {
        Some(raw)
    }
}

pub fn run_frame_perception_json(
    seq: &DurLARSequence,
    frame_id: &str,
    cache_dir: &Path,
    models_dir: &Path,
) -> Result<String, String> {
    if let Some(cached) = read_cached_perception_json(cache_dir, frame_id) {
        return Ok(cached);
    }

    let Some(img_path) = seq.image_path("image_02", frame_id) else {
        let empty = PerceptionResult {
            frame_id: frame_id.to_string(),
            device: device_name(models_dir).to_string(),
            inference_ms: 0.0,
            cached: false,
            objects: Vec::new(),
            road_mesh_b64: String::new(),
            dense_points_b64: String::new(),
            dense_count: 0,
        };
        return serde_json::to_string(&empty).map_err(|e| e.to_string());
    };

    let sessions = get_or_init_sessions(models_dir)?;
    let t0 = Instant::now();

    let rgb_img = image::open(&img_path)
        .map_err(|e| format!("Failed to decode image {}: {e}", img_path.display()))?
        .to_rgb8();
    let img_w = rgb_img.width() as usize;
    let img_h = rgb_img.height() as usize;
    let img_raw = rgb_img.as_raw();
    let (r_l2c, t_l2c, r_rect, p_rect) = calibration_matrices(&seq.calibration, img_w, img_h);

    // Run RT-DETR, Depth Anything V2, and 262k-point LiDAR loading/projection concurrently
    let (det_res, dep_res, lidar_data) = std::thread::scope(|s| {
        let det_handle = s.spawn(|| -> Result<Vec<RawDet2D>, String> {
            let det_input = resize_rgb_to_chw(
                img_raw,
                img_w,
                img_h,
                DET_SIZE,
                DET_SIZE,
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            );
            let det_tensor =
                Tensor::from_array(([1usize, 3, DET_SIZE, DET_SIZE], det_input.into_boxed_slice()))
                    .map_err(|e| format!("RT-DETR tensor error: {e}"))?;
            let mut det_sess = sessions
                .detector
                .lock()
                .map_err(|_| "RT-DETR mutex poisoned".to_string())?;
            let outputs = det_sess
                .run(ort::inputs![det_tensor])
                .map_err(|e| format!("RT-DETR inference failed: {e}"))?;
            let (_, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract logits: {e}"))?;
            let (_, boxes) = outputs["pred_boxes"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract pred_boxes: {e}"))?;
            Ok(decode_rtdetr_detections(
                logits,
                boxes,
                img_w as f64,
                img_h as f64,
            ))
        });

        let dep_handle = s.spawn(|| -> Result<Vec<f32>, String> {
            let dep_input = resize_rgb_to_chw(
                img_raw,
                img_w,
                img_h,
                DEP_SIZE,
                DEP_SIZE,
                [0.485, 0.456, 0.406],
                [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
            );
            let dep_tensor =
                Tensor::from_array(([1usize, 3, DEP_SIZE, DEP_SIZE], dep_input.into_boxed_slice()))
                    .map_err(|e| format!("Depth tensor error: {e}"))?;
            let mut dep_sess = sessions
                .depth
                .lock()
                .map_err(|_| "Depth mutex poisoned".to_string())?;
            let outputs = dep_sess
                .run(ort::inputs![dep_tensor])
                .map_err(|e| format!("Depth inference failed: {e}"))?;
            let (shape, dep_small) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract predicted_depth: {e}"))?;
            let (sh, sw) = if shape.len() == 3 {
                (shape[1] as usize, shape[2] as usize)
            } else {
                (DEP_SIZE, DEP_SIZE)
            };
            Ok(resize_f32_bilinear(dep_small, sw, sh, img_w, img_h))
        });

        let pts = seq.load_points(frame_id, 1);
        let (uvz, in_bounds) = project_lidar(&pts, &r_l2c, &t_l2c, &r_rect, &p_rect, img_w, img_h);

        let d1 = det_handle
            .join()
            .map_err(|_| "RT-DETR worker panicked".to_string())
            .and_then(|r| r);
        let d2 = dep_handle
            .join()
            .map_err(|_| "Depth worker panicked".to_string())
            .and_then(|r| r);
        (d1, d2, (pts, uvz, in_bounds))
    });

    let raw_detections = det_res?;
    let rel_disp = dep_res?;
    let (points, uvz, in_bounds) = lidar_data;

    let metric_depth = align_metric_depth(&rel_disp, img_w, img_h, &uvz, &in_bounds);
    let road_mesh = build_road_mesh(
        &points,
        img_raw,
        img_w,
        img_h,
        &metric_depth,
        &r_l2c,
        &t_l2c,
        &r_rect,
        &p_rect,
    );
    let dense_points = build_dense_points(
        img_raw,
        img_w,
        img_h,
        &metric_depth,
        &r_l2c,
        &t_l2c,
        &r_rect,
        &p_rect,
    );
    let dense_count = dense_points.len() / 6;
    let objects = extract_3d_objects(
        &raw_detections,
        &points,
        &uvz,
        &in_bounds,
        &metric_depth,
        &r_l2c,
        &t_l2c,
        &r_rect,
        &p_rect,
        img_w,
        img_h,
    );

    let elapsed_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
    let result = PerceptionResult {
        frame_id: frame_id.to_string(),
        device: sessions.device.to_string(),
        inference_ms: elapsed_ms,
        cached: false,
        objects,
        road_mesh_b64: encode_f32_slice_b64(&road_mesh),
        dense_points_b64: encode_f32_slice_b64(&dense_points),
        dense_count,
    };

    let json_str = serde_json::to_string(&result).map_err(|e| e.to_string())?;

    // Persist atomically to perception cache directory
    if fs::create_dir_all(cache_dir).is_ok() {
        let cache_path = cache_dir.join(format!("{frame_id}.json"));
        let tmp_path = cache_dir.join(format!("{frame_id}.json.tmp"));
        if fs::write(&tmp_path, &json_str).is_ok() {
            let _ = fs::rename(&tmp_path, &cache_path);
        }
    }

    Ok(json_str)
}

fn encode_f32_slice_b64(floats: &[f32]) -> String {
    if floats.is_empty() {
        return String::new();
    }
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for &v in floats {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    BASE64.encode(&bytes)
}

fn resize_rgb_to_chw(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mean: [f32; 3],
    inv_std: [f32; 3],
) -> Vec<f32> {
    let plane = dst_w * dst_h;
    let mut out = vec![0.0f32; 3 * plane];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let inv_255 = 1.0f32 / 255.0;

    for dy in 0..dst_h {
        let sy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        let wy0 = 1.0 - wy;

        for dx in 0..dst_w {
            let sx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            let wx0 = 1.0 - wx;

            let i00 = (y0 * src_w + x0) * 3;
            let i01 = (y0 * src_w + x1) * 3;
            let i10 = (y1 * src_w + x0) * 3;
            let i11 = (y1 * src_w + x1) * 3;
            let dst_idx = dy * dst_w + dx;

            for c in 0..3 {
                let top = rgb[i00 + c] as f32 * wx0 + rgb[i01 + c] as f32 * wx;
                let bot = rgb[i10 + c] as f32 * wx0 + rgb[i11 + c] as f32 * wx;
                let val = (top * wy0 + bot * wy) * inv_255;
                out[c * plane + dst_idx] = (val - mean[c]) * inv_std[c];
            }
        }
    }
    out
}

fn resize_f32_bilinear(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_w * dst_h];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let sy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        let wy0 = 1.0 - wy;

        for dx in 0..dst_w {
            let sx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            let wx0 = 1.0 - wx;

            let v00 = src[y0 * src_w + x0];
            let v01 = src[y0 * src_w + x1];
            let v10 = src[y1 * src_w + x0];
            let v11 = src[y1 * src_w + x1];

            let top = v00 * wx0 + v01 * wx;
            let bot = v10 * wx0 + v11 * wx;
            out[dy * dst_w + dx] = top * wy0 + bot * wy;
        }
    }
    out
}

struct RawDet2D {
    label: &'static str,
    prior: [f64; 3],
    score: f64,
    box2d: [f64; 4],
}

fn driving_class(class_id: usize) -> Option<(&'static str, [f64; 3])> {
    match class_id {
        0 => Some(("person", [0.65, 0.65, 1.72])),
        1 => Some(("bicycle", [1.75, 0.70, 1.40])),
        2 => Some(("car", [4.20, 1.85, 1.50])),
        3 => Some(("motorcycle", [2.00, 0.80, 1.40])),
        5 => Some(("bus", [8.50, 2.50, 3.10])),
        7 => Some(("truck", [6.50, 2.35, 2.85])),
        9 => Some(("traffic light", [0.45, 0.45, 0.95])),
        11 => Some(("stop sign", [0.65, 0.25, 0.75])),
        16 => Some(("dog", [0.75, 0.40, 0.65])),
        _ => None,
    }
}

fn decode_rtdetr_detections(
    logits: &[f32],
    pred_boxes: &[f32],
    img_w: f64,
    img_h: f64,
) -> Vec<RawDet2D> {
    let num_classes = 80usize;
    let num_queries = logits.len() / num_classes;
    let mut candidates = Vec::new();

    for q in 0..num_queries {
        if (q + 1) * 4 > pred_boxes.len() {
            break;
        }
        // Take highest-scoring class for this object query (focal loss sigmoid)
        let row = &logits[q * num_classes..(q + 1) * num_classes];
        let mut best_cls = 0usize;
        let mut best_logit = f32::NEG_INFINITY;
        for (c, &val) in row.iter().enumerate() {
            if val > best_logit {
                best_logit = val;
                best_cls = c;
            }
        }
        let score = 1.0 / (1.0 + (-best_logit as f64).exp());
        if score < 0.45 {
            continue;
        }
        let Some((label, prior)) = driving_class(best_cls) else {
            continue;
        };

        let cx = pred_boxes[q * 4] as f64;
        let cy = pred_boxes[q * 4 + 1] as f64;
        let bw = pred_boxes[q * 4 + 2] as f64;
        let bh = pred_boxes[q * 4 + 3] as f64;

        let x1 = ((cx - 0.5 * bw) * img_w).clamp(0.0, img_w);
        let y1 = ((cy - 0.5 * bh) * img_h).clamp(0.0, img_h);
        let x2 = ((cx + 0.5 * bw) * img_w).clamp(0.0, img_w);
        let y2 = ((cy + 0.5 * bh) * img_h).clamp(0.0, img_h);
        if x2 - x1 < 2.0 || y2 - y1 < 2.0 {
            continue;
        }

        candidates.push(RawDet2D {
            label,
            prior,
            score,
            box2d: [x1, y1, x2, y2],
        });
    }

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<RawDet2D> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let duplicate = kept
            .iter()
            .any(|k| k.label == cand.label && box_iou(&k.box2d, &cand.box2d) > 0.72);
        if !duplicate {
            kept.push(cand);
        }
    }
    kept
}

fn box_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter).max(1e-6)
}

fn calibration_matrices(
    calib: &Calibration,
    img_w: usize,
    img_h: usize,
) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3], [[f64; 4]; 3]) {
    let r_l2c = calib.r_lidar_to_cam;
    let t_l2c = calib.t_lidar_to_cam;
    let r_rect = calib.r_rect_reference;
    let mut p_rect = calib.p_rect_02;

    if let Some(&[sw, sh]) = calib.projection_sizes.get("image_02") {
        if sw > 0.0 && sh > 0.0 && ((sw - img_w as f64).abs() > 1e-3 || (sh - img_h as f64).abs() > 1e-3) {
            let sx = img_w as f64 / sw;
            let sy = img_h as f64 / sh;
            for v in &mut p_rect[0] {
                *v *= sx;
            }
            for v in &mut p_rect[1] {
                *v *= sy;
            }
        }
    } else if img_w != 2048 || img_h != 1088 {
        if (p_rect[0][2] * 2.0 - img_w as f64).abs() > 10.0 {
            let sx = img_w as f64 / (p_rect[0][2] * 2.0).max(1.0);
            let sy = img_h as f64 / (p_rect[1][2] * 2.0).max(1.0);
            for v in &mut p_rect[0] {
                *v *= sx;
            }
            for v in &mut p_rect[1] {
                *v *= sy;
            }
        }
    }

    (r_l2c, t_l2c, r_rect, p_rect)
}

#[inline]
fn project_single_point(
    xyz: [f64; 3],
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
    img_w: f64,
    img_h: f64,
) -> ([f64; 3], bool) {
    let mut raw_cam = [0.0f64; 3];
    for i in 0..3 {
        raw_cam[i] = r_l2c[i][0] * xyz[0] + r_l2c[i][1] * xyz[1] + r_l2c[i][2] * xyz[2] + t_l2c[i];
    }
    let mut cam = [0.0f64; 3];
    for i in 0..3 {
        cam[i] = r_rect[i][0] * raw_cam[0] + r_rect[i][1] * raw_cam[1] + r_rect[i][2] * raw_cam[2];
    }
    let depth = cam[2];
    let valid_d = depth > 0.8;
    let px = p_rect[0][0] * cam[0] + p_rect[0][1] * cam[1] + p_rect[0][2] * cam[2] + p_rect[0][3];
    let py = p_rect[1][0] * cam[0] + p_rect[1][1] * cam[1] + p_rect[1][2] * cam[2] + p_rect[1][3];
    let denom = if valid_d { depth } else { 1.0 };
    let u = px / denom;
    let v = py / denom;
    let in_bounds = valid_d && u >= 0.0 && u < img_w && v >= 0.0 && v < img_h;
    ([u, v, depth], in_bounds)
}

fn project_lidar(
    points: &[[f32; 4]],
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
    img_w: usize,
    img_h: usize,
) -> (Vec<[f64; 3]>, Vec<bool>) {
    let mut uvz = Vec::with_capacity(points.len());
    let mut in_bounds = Vec::with_capacity(points.len());
    let w = img_w as f64;
    let h = img_h as f64;

    for &[x, y, z, _] in points {
        let (p, ok) = project_single_point([x as f64, y as f64, z as f64], r_l2c, t_l2c, r_rect, p_rect, w, h);
        uvz.push(p);
        in_bounds.push(ok);
    }
    (uvz, in_bounds)
}

#[inline]
fn unproject_to_lidar_point(
    u: f64,
    v: f64,
    z_cam: f64,
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
) -> [f32; 3] {
    let fx = p_rect[0][0];
    let fy = p_rect[1][1];
    let cx = p_rect[0][2];
    let cy = p_rect[1][2];
    let tx = p_rect[0][3];
    let ty = p_rect[1][3];

    let x_c = ((u - cx) * z_cam - tx) / fx;
    let y_c = ((v - cy) * z_cam - ty) / fy;
    let cam_rect = [x_c, y_c, z_cam];

    let mut cam_raw = [0.0f64; 3];
    for i in 0..3 {
        cam_raw[i] = r_rect[0][i] * cam_rect[0]
            + r_rect[1][i] * cam_rect[1]
            + r_rect[2][i] * cam_rect[2]
            - t_l2c[i];
    }

    let mut lidar = [0.0f32; 3];
    for i in 0..3 {
        lidar[i] = (r_l2c[0][i] * cam_raw[0]
            + r_l2c[1][i] * cam_raw[1]
            + r_l2c[2][i] * cam_raw[2]) as f32;
    }
    lidar
}

fn fit_line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut var_x = 0.0;
    let mut cov_xy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        var_x += dx * dx;
        cov_xy += dx * (y - my);
    }
    let a = if var_x > 1e-12 { cov_xy / var_x } else { 0.0 };
    let b = my - a * mx;
    (a, b)
}

fn percentile_f64(vals: &mut [f64], pct: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let pos = (pct / 100.0) * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    vals[lo] * (1.0 - frac) + vals[hi] * frac
}

fn align_metric_depth(
    rel_disp: &[f32],
    w: usize,
    h: usize,
    uvz: &[[f64; 3]],
    in_bounds: &[bool],
) -> Vec<f32> {
    let mut valid_pts = Vec::new();
    for (p, &ok) in uvz.iter().zip(in_bounds.iter()) {
        if ok && p[2] > 1.2 && p[2] < 55.0 {
            valid_pts.push(*p);
        }
    }

    if valid_pts.len() < 40 {
        let mut d_min = f32::INFINITY;
        let mut d_max = f32::NEG_INFINITY;
        for &v in rel_disp {
            if v < d_min {
                d_min = v;
            }
            if v > d_max {
                d_max = v;
            }
        }
        let span = (d_max - d_min).max(1e-5);
        return rel_disp
            .iter()
            .map(|&v| {
                let norm = (v - d_min) / span;
                1.0 / (0.02 + 0.35 * norm).clamp(1.0 / 60.0, 1.0)
            })
            .collect();
    }

    let step = if valid_pts.len() > 4000 {
        valid_pts.len() / 4000
    } else {
        1
    };
    let mut xs = Vec::with_capacity(valid_pts.len() / step + 1);
    let mut ys = Vec::with_capacity(valid_pts.len() / step + 1);
    for p in valid_pts.iter().step_by(step) {
        let ui = (p[0] as isize).clamp(0, (w - 1) as isize) as usize;
        let vi = (p[1] as isize).clamp(0, (h - 1) as isize) as usize;
        xs.push(rel_disp[vi * w + ui] as f64);
        ys.push(1.0 / p[2]);
    }

    let (mut a, mut b) = fit_line(&xs, &ys);
    let mut resid: Vec<f64> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (y - (a * x + b)).abs())
        .collect();
    let mut resid_copy = resid.clone();
    let mad = percentile_f64(&mut resid_copy, 50.0).max(1e-4);

    let mut in_x = Vec::new();
    let mut in_y = Vec::new();
    for i in 0..xs.len() {
        if resid[i] <= 2.2 * mad {
            in_x.push(xs[i]);
            in_y.push(ys[i]);
        }
    }
    if in_x.len() >= 20 {
        (a, b) = fit_line(&in_x, &in_y);
    }

    if a <= 1e-6 {
        resid.clear();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            resid.push(y / x.max(1e-4));
        }
        let scale = percentile_f64(&mut resid, 50.0);
        a = scale.max(1e-4);
        b = 0.0;
    }

    rel_disp
        .iter()
        .map(|&d| {
            let inv_d = (a * d as f64 + b).clamp(1.0 / 65.0, 1.0 / 0.8);
            (1.0 / inv_d) as f32
        })
        .collect()
}

fn build_road_mesh(
    points: &[[f32; 4]],
    img_raw: &[u8],
    img_w: usize,
    img_h: usize,
    metric_depth: &[f32],
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
) -> Vec<f32> {
    const NX: usize = 35;
    const NY: usize = 21;
    let x0 = -16.0f64;
    let x1 = 52.0f64;
    let y0 = -18.0f64;
    let y1 = 18.0f64;
    let dx = (x1 - x0) / (NX - 1) as f64;
    let dy = (y1 - y0) / (NY - 1) as f64;

    let mut z_sum = [[0.0f64; NY]; NX];
    let mut z_cnt = [[0u32; NY]; NX];

    for &[px, py, pz, _] in points {
        let x = px as f64;
        let y = py as f64;
        let z = pz as f64;
        if (-2.3..=-1.10).contains(&z) && (x0..=x1).contains(&x) && (y0..=y1).contains(&y) {
            let ix = (((x - x0) / dx).round() as isize).clamp(0, (NX - 1) as isize) as usize;
            let iy = (((y - y0) / dy).round() as isize).clamp(0, (NY - 1) as isize) as usize;
            z_sum[ix][iy] += z;
            z_cnt[ix][iy] += 1;
        }
    }

    let mut z_grid = [[f64::NAN; NY]; NX];
    for ix in 0..NX {
        for iy in 0..NY {
            if z_cnt[ix][iy] > 0 {
                z_grid[ix][iy] = z_sum[ix][iy] / z_cnt[ix][iy] as f64;
            }
        }
    }

    for _ in 0..3 {
        let prev = z_grid;
        let mut any_nan = false;
        for ix in 0..NX {
            for iy in 0..NY {
                if !prev[ix][iy].is_nan() {
                    continue;
                }
                any_nan = true;
                let mut acc = 0.0;
                let mut cnt = 0.0;
                for ox in -1isize..=1 {
                    for oy in -1isize..=1 {
                        let sx = (ix as isize + ox).clamp(0, (NX - 1) as isize) as usize;
                        let sy = (iy as isize + oy).clamp(0, (NY - 1) as isize) as usize;
                        let v = prev[sx][sy];
                        if !v.is_nan() {
                            acc += v;
                            cnt += 1.0;
                        }
                    }
                }
                if cnt > 0.0 {
                    z_grid[ix][iy] = acc / cnt;
                }
            }
        }
        if !any_nan {
            break;
        }
    }

    let default_ground = -1.68;
    for row in &mut z_grid {
        for val in row {
            if val.is_nan() {
                *val = default_ground;
            }
        }
    }

    let mut verts = [[[0.0f32; 7]; NY]; NX];
    let w_f = img_w as f64;
    let h_f = img_h as f64;

    for ix in 0..NX {
        let gx = x0 + ix as f64 * dx;
        let ix_m = ix.saturating_sub(1);
        let ix_p = (ix + 1).min(NX - 1);
        for iy in 0..NY {
            let gy = y0 + iy as f64 * dy;
            let iy_m = iy.saturating_sub(1);
            let iy_p = (iy + 1).min(NY - 1);
            let z_smooth = (4.0 * z_grid[ix][iy]
                + z_grid[ix_m][iy]
                + z_grid[ix_p][iy]
                + z_grid[ix][iy_m]
                + z_grid[ix][iy_p])
                / 8.0;

            let dist = gx.hypot(gy);
            let fade = (1.0 - (dist - 28.0).max(0.0) / 26.0).clamp(0.08, 0.85) as f32;
            let mut r_col = 0.055f32;
            let mut g_col = 0.075f32;
            let mut b_col = 0.11f32;
            let mut a_col = fade;

            let (uvz, in_fov) =
                project_single_point([gx, gy, z_smooth], r_l2c, t_l2c, r_rect, p_rect, w_f, h_f);
            if in_fov {
                let ui = (uvz[0] as isize).clamp(0, (img_w - 1) as isize) as usize;
                let vi = (uvz[1] as isize).clamp(0, (img_h - 1) as isize) as usize;
                let md = metric_depth[vi * img_w + ui] as f64;
                if (md - uvz[2]).abs() < 5.0 {
                    let p_idx = (vi * img_w + ui) * 3;
                    let pr = img_raw[p_idx] as f32 / 255.0;
                    let pg = img_raw[p_idx + 1] as f32 / 255.0;
                    let pb = img_raw[p_idx + 2] as f32 / 255.0;
                    let luma = 0.2126 * pr + 0.7152 * pg + 0.0722 * pb;
                    r_col = 0.05 + 0.18 * pr + 0.04 * luma;
                    g_col = 0.065 + 0.20 * pg + 0.04 * luma;
                    b_col = 0.095 + 0.22 * pb + 0.05 * luma;
                    a_col = fade.max(0.82);
                }
            }

            verts[ix][iy] = [
                gx as f32,
                gy as f32,
                z_smooth as f32,
                r_col,
                g_col,
                b_col,
                a_col,
            ];
        }
    }

    let mut tris = Vec::with_capacity((NX - 1) * (NY - 1) * 6 * 7);
    for ix in 0..(NX - 1) {
        for iy in 0..(NY - 1) {
            let v00 = &verts[ix][iy];
            let v10 = &verts[ix + 1][iy];
            let v11 = &verts[ix + 1][iy + 1];
            let v01 = &verts[ix][iy + 1];
            for v in [v00, v10, v11, v00, v11, v01] {
                tris.extend_from_slice(v);
            }
        }
    }
    tris
}

fn build_dense_points(
    img_raw: &[u8],
    img_w: usize,
    img_h: usize,
    metric_depth: &[f32],
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
) -> Vec<f32> {
    let stride = if img_w >= 1000 { 6 } else { 4 };
    let mut out = Vec::new();
    let inv_255 = 1.0f32 / 255.0;

    for v in (0..img_h).step_by(stride) {
        for u in (0..img_w).step_by(stride) {
            let z = metric_depth[v * img_w + u] as f64;
            if z <= 1.2 || z >= 48.0 {
                continue;
            }
            let xyz = unproject_to_lidar_point(u as f64, v as f64, z, r_l2c, t_l2c, r_rect, p_rect);
            if xyz[2] <= -1.35 || xyz[2] >= 7.5 {
                continue;
            }
            let idx = (v * img_w + u) * 3;
            out.extend_from_slice(&[
                xyz[0],
                xyz[1],
                xyz[2],
                img_raw[idx] as f32 * inv_255,
                img_raw[idx + 1] as f32 * inv_255,
                img_raw[idx + 2] as f32 * inv_255,
            ]);
        }
    }
    out
}

fn extract_3d_objects(
    detections: &[RawDet2D],
    points: &[[f32; 4]],
    uvz: &[[f64; 3]],
    in_bounds: &[bool],
    metric_depth: &[f32],
    r_l2c: &[[f64; 3]; 3],
    t_l2c: &[f64; 3],
    r_rect: &[[f64; 3]; 3],
    p_rect: &[[f64; 4]; 3],
    img_w: usize,
    img_h: usize,
) -> Vec<DetectedObject3D> {
    let mut lidar_pts = Vec::new();
    for (i, &ok) in in_bounds.iter().enumerate() {
        if ok && i < points.len() {
            lidar_pts.push((uvz[i], points[i]));
        }
    }

    let mut objects = Vec::with_capacity(detections.len());
    for det in detections {
        let [x1, y1, x2, y2] = det.box2d;
        let bw = (x2 - x1).max(1.0);
        let bh = (y2 - y1).max(1.0);
        let ix1 = x1 + 0.15 * bw;
        let ix2 = x2 - 0.15 * bw;
        let iy1 = y1 + 0.05 * bh;
        let iy2 = y2 - 0.05 * bh;

        let [prior_l, prior_w, prior_h] = det.prior;
        let cx_px = 0.5 * (x1 + x2);
        let cy_px = 0.5 * (y1 + y2);

        let in_box: Vec<([f64; 3], [f32; 4])> = lidar_pts
            .iter()
            .copied()
            .filter(|(p, _)| {
                p[0] >= ix1 && p[0] <= ix2 && p[1] >= iy1 && p[1] <= iy2 && p[2] > 1.0 && p[2] < 60.0
            })
            .collect();

        let mut source = "depth";
        let mut yaw = 0.0f64;
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        let mut size = [prior_l, prior_w, prior_h];

        if in_box.len() >= 5 {
            let mut d_vals: Vec<f64> = in_box.iter().map(|(p, _)| p[2]).collect();
            let d_anchor = percentile_f64(&mut d_vals, 30.0);
            let tol = if matches!(det.label, "person" | "bicycle" | "motorcycle" | "dog") {
                1.6
            } else {
                3.2
            };
            let cluster: Vec<[f64; 3]> = in_box
                .iter()
                .filter(|(p, _)| (p[2] - d_anchor).abs() <= tol)
                .map(|(_, xyz)| [xyz[0] as f64, xyz[1] as f64, xyz[2] as f64])
                .collect();

            if cluster.len() >= 4 {
                source = "lidar";
                let mut xs: Vec<f64> = cluster.iter().map(|c| c[0]).collect();
                let mut ys: Vec<f64> = cluster.iter().map(|c| c[1]).collect();
                let mut zs: Vec<f64> = cluster.iter().map(|c| c[2]).collect();
                let mut med_x = percentile_f64(&mut xs, 50.0);
                let mut med_y = percentile_f64(&mut ys, 50.0);
                let z_min = percentile_f64(&mut zs, 5.0);
                let z_max = percentile_f64(&mut zs, 98.0);
                let est_h = (z_max - z_min).clamp(prior_h * 0.88, prior_h * 1.25);

                if matches!(det.label, "car" | "bus" | "truck") {
                    if cluster.len() >= 8 {
                        let n = cluster.len() as f64;
                        let mean_x = cluster.iter().map(|c| c[0]).sum::<f64>() / n;
                        let mean_y = cluster.iter().map(|c| c[1]).sum::<f64>() / n;
                        let mut cxx = 0.0;
                        let mut cxy = 0.0;
                        let mut cyy = 0.0;
                        for c in &cluster {
                            let dx = c[0] - mean_x;
                            let dy = c[1] - mean_y;
                            cxx += dx * dx;
                            cxy += dx * dy;
                            cyy += dy * dy;
                        }
                        let trace = cxx + cyy;
                        let diff = cxx - cyy;
                        let disc = (diff * diff + 4.0 * cxy * cxy).sqrt();
                        let eig_max = 0.5 * (trace + disc);
                        let span = (eig_max.max(0.0) / n).sqrt() * 3.0;
                        if span > 1.3 {
                            let (vx, vy) = if cxy.abs() > 1e-9 {
                                (eig_max - cyy, cxy)
                            } else if cxx >= cyy {
                                (1.0, 0.0)
                            } else {
                                (0.0, 1.0)
                            };
                            yaw = vy.atan2(vx);
                        } else if bw > bh * 1.5 {
                            yaw = med_y.atan2(med_x) + PI / 2.0;
                        }
                    } else if bw > bh * 1.5 {
                        yaw = med_y.atan2(med_x) + PI / 2.0;
                    }

                    let ray_n = med_x.hypot(med_y).max(1e-3);
                    let inward = if bw > bh * 1.4 {
                        prior_w * 0.45
                    } else {
                        prior_l * 0.42
                    };
                    med_x += (med_x / ray_n) * inward;
                    med_y += (med_y / ray_n) * inward;
                }

                cx = med_x;
                cy = med_y;
                cz = if matches!(det.label, "traffic light" | "stop sign") {
                    0.5 * (z_min + z_max)
                } else {
                    -1.68 + est_h / 2.0
                };
                size = [prior_l, prior_w, est_h];
            }
        }

        if source == "depth" {
            let px1 = (ix1 as isize).clamp(0, (img_w - 1) as isize) as usize;
            let px2 = ((ix2 + 1.0) as isize).clamp((px1 + 1) as isize, img_w as isize) as usize;
            let py1 = (iy1 as isize).clamp(0, (img_h - 1) as isize) as usize;
            let py2 = ((iy2 + 1.0) as isize).clamp((py1 + 1) as isize, img_h as isize) as usize;
            let mut patch = Vec::with_capacity((px2 - px1) * (py2 - py1));
            for py in py1..py2 {
                for px in px1..px2 {
                    patch.push(metric_depth[py * img_w + px] as f64);
                }
            }
            let z_cam = percentile_f64(&mut patch, 35.0);
            let xyz = unproject_to_lidar_point(cx_px, cy_px, z_cam, r_l2c, t_l2c, r_rect, p_rect);
            cx = xyz[0] as f64;
            cy = xyz[1] as f64;
            cz = if matches!(det.label, "traffic light" | "stop sign") {
                xyz[2] as f64
            } else {
                -1.68 + prior_h / 2.0
            };
            if matches!(det.label, "car" | "bus" | "truck") && bw > bh * 1.5 {
                yaw = cy.atan2(cx) + PI / 2.0;
            }
            size = [prior_l, prior_w, prior_h];
        }

        objects.push(DetectedObject3D {
            label: det.label.to_string(),
            score: round_n(det.score, 3),
            box2d: [round_n(x1, 1), round_n(y1, 1), round_n(x2, 1), round_n(y2, 1)],
            center: [round_n(cx, 2), round_n(cy, 2), round_n(cz, 2)],
            size: [round_n(size[0], 2), round_n(size[1], 2), round_n(size[2], 2)],
            yaw: round_n(yaw, 3),
            source: source.to_string(),
        });
    }

    objects
}

#[inline]
fn round_n(v: f64, decimals: i32) -> f64 {
    let f = 10f64.powi(decimals);
    (v * f).round() / f
}
