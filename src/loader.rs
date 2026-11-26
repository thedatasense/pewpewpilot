use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::ImageFormat;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    #[serde(rename = "R_lidar_to_cam")]
    pub r_lidar_to_cam: [[f64; 3]; 3],
    #[serde(rename = "T_lidar_to_cam")]
    pub t_lidar_to_cam: [f64; 3],
    #[serde(rename = "P_rect_02")]
    pub p_rect_02: [[f64; 4]; 3],
    #[serde(rename = "P_rect_01")]
    pub p_rect_01: [[f64; 4]; 3],
    pub image_width: u32,
    pub image_height: u32,
    pub lidar_channels: u32,
    #[serde(rename = "R_rect_reference")]
    pub r_rect_reference: [[f64; 3]; 3],
    pub projectable_images: Vec<String>,
    pub projection_sizes: BTreeMap<String, [f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointStats {
    pub num_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub intensity_min: f64,
    pub intensity_max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryResponse {
    pub frame_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gps: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imu: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lux: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub frame_count: usize,
    pub first_frame: Option<String>,
    pub last_frame: Option<String>,
    pub has_points: bool,
    pub has_camera: bool,
    pub has_ambient: bool,
    pub has_reflec: bool,
    pub has_telemetry: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Modalities {
    pub lidar: &'static str,
    pub camera_left: &'static str,
    pub camera_right: &'static str,
    pub panoramic_ambient: &'static str,
    pub panoramic_reflec: &'static str,
    pub telemetry: &'static str,
}

impl Default for Modalities {
    fn default() -> Self {
        Self {
            lidar: "Ouster OS1-128 (128 channels, 360° FOV, KITTI bin)",
            camera_left: "Carnegie Robotics MultiSense S21 RGB (2048x1088)",
            camera_right: "Carnegie Robotics MultiSense S21 Grayscale (2048x1088)",
            panoramic_ambient: "Ouster Panoramic Near-IR Ambient (2048x128)",
            panoramic_reflec: "Ouster Panoramic Surface Reflectivity (2048x128)",
            telemetry: "OxTS RT3000v3 GNSS/INS + Yocto Lux Meter",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceInfo {
    pub id: String,
    pub frames: Vec<String>,
    pub calibration: Calibration,
    pub modalities: Modalities,
}

#[derive(Debug, Clone)]
pub struct DurLARSequence {
    pub path: PathBuf,
    pub name: String,
    pub frames: Vec<String>,
    pub frame_set: HashSet<String>,
    pub calibration: Calibration,
}

impl DurLARSequence {
    pub fn load(sequence_path: &Path) -> Self {
        let path = fs::canonicalize(sequence_path).unwrap_or_else(|_| sequence_path.to_path_buf());
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let frames = Self::discover_frames(&path);
        let frame_set = frames.iter().cloned().collect();
        let calibration = Self::load_calibration(&path);
        Self {
            path,
            name,
            frames,
            frame_set,
            calibration,
        }
    }

    pub fn has_frame(&self, frame_id: &str) -> bool {
        self.frame_set.contains(frame_id)
    }

    pub fn summary(&self, id: &str, read_only: bool) -> SequenceSummary {
        SequenceSummary {
            id: id.to_string(),
            path: if read_only {
                None
            } else {
                Some(self.path.to_string_lossy().into_owned())
            },
            frame_count: self.frames.len(),
            first_frame: self.frames.first().cloned(),
            last_frame: self.frames.last().cloned(),
            has_points: self.path.join("ouster_points").is_dir(),
            has_camera: self.path.join("image_02").is_dir(),
            has_ambient: self.path.join("ambient").is_dir(),
            has_reflec: self.path.join("reflec").is_dir(),
            has_telemetry: self.path.join("gps").is_dir(),
        }
    }

    pub fn info(&self) -> SequenceInfo {
        SequenceInfo {
            id: self.name.clone(),
            frames: self.frames.clone(),
            calibration: self.calibration.clone(),
            modalities: Modalities::default(),
        }
    }

    fn discover_frames(base_path: &Path) -> Vec<String> {
        let pts_data = base_path.join("ouster_points").join("data");
        let pts_dir = if pts_data.is_dir() {
            pts_data
        } else {
            base_path.join("ouster_points")
        };

        if pts_dir.is_dir() {
            return Self::sorted_stems_with_ext(&pts_dir, "bin");
        }

        for sub in &[
            Path::new("image_02").join("data"),
            PathBuf::from("image_02"),
            Path::new("ambient").join("data"),
            PathBuf::from("ambient"),
        ] {
            let test_dir = base_path.join(sub);
            if test_dir.is_dir() {
                return Self::sorted_stems_with_ext(&test_dir, "png");
            }
        }

        Vec::new()
    }

    fn sorted_stems_with_ext(dir: &Path, ext: &str) -> Vec<String> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut names = Vec::new();
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|e| e.to_str()) == Some(ext) && file_path.is_file() {
                if let Some(stem) = file_path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
        names.sort();
        names
    }

    fn load_calibration(base_path: &Path) -> Calibration {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut calib = Calibration {
            r_lidar_to_cam: identity,
            t_lidar_to_cam: [0.0, 0.0, 0.0],
            p_rect_02: [
                [1850.0, 0.0, 1024.0, 0.0],
                [0.0, 1850.0, 544.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            p_rect_01: [
                [1850.0, 0.0, 1024.0, -388.5],
                [0.0, 1850.0, 544.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            image_width: 2048,
            image_height: 1088,
            lidar_channels: 128,
            r_rect_reference: identity,
            projectable_images: Vec::new(),
            projection_sizes: BTreeMap::new(),
        };

        let extrinsic = Self::calibration_fields(base_path, "calib_lidar_to_cam.txt");
        let camera = Self::calibration_fields(base_path, "calib_cam_to_cam.txt");

        let rotation = extrinsic.get("R");
        let translation = extrinsic.get("T");
        let has_extrinsic = matches!((rotation, translation), (Some(r), Some(t)) if r.len() == 9 && t.len() == 3);
        if let (true, Some(r), Some(t)) = (has_extrinsic, rotation, translation) {
            calib.r_lidar_to_cam = reshape_3x3(r);
            calib.t_lidar_to_cam = [t[0], t[1], t[2]];
        }

        let rectification = ["R_rect_00", "R0_rect", "R_rect_02"]
            .iter()
            .find_map(|key| camera.get(*key));
        let rectification_valid = rectification.map_or(true, |r| r.len() == 9);
        if let Some(r) = rectification {
            if r.len() == 9 {
                calib.r_rect_reference = reshape_3x3(r);
            }
        }

        for suffix in ["01", "02"] {
            let p_key = format!("P_rect_{suffix}");
            let s_key = format!("S_rect_{suffix}");
            if let Some(size) = camera.get(&s_key) {
                if size.len() == 2 && size[0] > 0.0 && size[1] > 0.0 {
                    calib
                        .projection_sizes
                        .insert(format!("image_{suffix}"), [size[0], size[1]]);
                }
            }
            if let Some(proj) = camera.get(&p_key) {
                if proj.len() == 12 {
                    let mat = reshape_3x4(proj);
                    if suffix == "01" {
                        calib.p_rect_01 = mat;
                    } else {
                        calib.p_rect_02 = mat;
                    }
                    if has_extrinsic && rectification_valid {
                        calib.projectable_images.push(format!("image_{suffix}"));
                    }
                }
            }
        }

        calib
    }

    fn calibration_fields(base_path: &Path, filename: &str) -> HashMap<String, Vec<f64>> {
        let mut folders = vec![base_path.to_path_buf()];
        if let Some(parent) = base_path.parent() {
            folders.push(parent.to_path_buf());
        }

        for folder in folders {
            let path = folder.join(filename);
            if !path.is_file() {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let mut fields = HashMap::new();
            for line in text.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let parsed: Result<Vec<f64>, _> =
                    value.split_whitespace().map(|s| s.parse::<f64>()).collect();
                let Ok(values) = parsed else {
                    continue;
                };
                let clean = if values.iter().all(|v| v.is_finite()) {
                    values
                } else {
                    Vec::new()
                };
                fields.insert(key.trim().to_string(), clean);
            }
            return fields;
        }
        HashMap::new()
    }

    pub fn point_cloud_path(&self, frame_id: &str) -> Option<PathBuf> {
        let candidates = [
            self.path
                .join("ouster_points")
                .join("data")
                .join(format!("{frame_id}.bin")),
            self.path
                .join("ouster_points")
                .join(format!("{frame_id}.bin")),
        ];
        candidates.into_iter().find(|p| p.is_file())
    }

    /// Reads raw binary float32 [x, y, z, intensity] records with optional striding.
    /// Returns `(bytes, point_count)`.
    pub fn read_points_binary(&self, frame_id: &str, stride: usize) -> (Vec<u8>, usize) {
        let stride = stride.max(1);
        let Some(bin_path) = self.point_cloud_path(frame_id) else {
            return (Vec::new(), 0);
        };
        let Ok(mut raw) = fs::read(&bin_path) else {
            return (Vec::new(), 0);
        };
        let valid_len = raw.len() - (raw.len() % 16);
        raw.truncate(valid_len);
        let total_points = valid_len / 16;

        if stride == 1 {
            return (raw, total_points);
        }

        let out_points = total_points.div_ceil(stride);
        let mut out = Vec::with_capacity(out_points * 16);
        for chunk in raw.chunks_exact(16).step_by(stride) {
            out.extend_from_slice(chunk);
        }
        (out, out_points)
    }

    /// Loads [x, y, z, intensity] points as `[f32; 4]` records.
    pub fn load_points(&self, frame_id: &str, stride: usize) -> Vec<[f32; 4]> {
        let stride = stride.max(1);
        let Some(bin_path) = self.point_cloud_path(frame_id) else {
            return Vec::new();
        };
        let Ok(raw) = fs::read(&bin_path) else {
            return Vec::new();
        };
        let total_points = raw.len() / 16;
        let out_points = total_points.div_ceil(stride);
        let mut pts = Vec::with_capacity(out_points);
        for chunk in raw.chunks_exact(16).step_by(stride) {
            pts.push([
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            ]);
        }
        pts
    }

    pub fn point_stats(&self, frame_id: &str) -> PointStats {
        let pts = self.load_points(frame_id, 1);
        if pts.is_empty() {
            return PointStats {
                num_points: 0,
                x_min: 0.0,
                x_max: 0.0,
                y_min: 0.0,
                y_max: 0.0,
                z_min: 0.0,
                z_max: 0.0,
                intensity_min: 0.0,
                intensity_max: 0.0,
            };
        }

        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;
        let mut i_min = f32::INFINITY;
        let mut i_max = f32::NEG_INFINITY;

        for &[x, y, z, intensity] in &pts {
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
            if y < y_min {
                y_min = y;
            }
            if y > y_max {
                y_max = y;
            }
            if z < z_min {
                z_min = z;
            }
            if z > z_max {
                z_max = z;
            }
            if intensity < i_min {
                i_min = intensity;
            }
            if intensity > i_max {
                i_max = intensity;
            }
        }

        PointStats {
            num_points: pts.len(),
            x_min: x_min as f64,
            x_max: x_max as f64,
            y_min: y_min as f64,
            y_max: y_max as f64,
            z_min: z_min as f64,
            z_max: z_max as f64,
            intensity_min: i_min as f64,
            intensity_max: i_max as f64,
        }
    }

    pub fn image_path(&self, modality: &str, frame_id: &str) -> Option<PathBuf> {
        for ext in ["png", "jpg", "tiff", "tif"] {
            let data_candidate = self
                .path
                .join(modality)
                .join("data")
                .join(format!("{frame_id}.{ext}"));
            if data_candidate.is_file() {
                return Some(data_candidate);
            }
            let direct_candidate = self.path.join(modality).join(format!("{frame_id}.{ext}"));
            if direct_candidate.is_file() {
                return Some(direct_candidate);
            }
        }
        None
    }

    pub fn read_image_bytes(&self, modality: &str, frame_id: &str) -> Option<(Vec<u8>, &'static str)> {
        let img_path = self.image_path(modality, frame_id)?;
        let ext = img_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "tif" || ext == "tiff" {
            let decoded = image::open(&img_path).ok()?;
            let mut buf = Cursor::new(Vec::new());
            decoded.write_to(&mut buf, ImageFormat::Png).ok()?;
            return Some((buf.into_inner(), "image/png"));
        }

        let bytes = fs::read(&img_path).ok()?;
        let content_type = match ext.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            _ => "image/png",
        };
        Some((bytes, content_type))
    }

    pub fn telemetry(&self, frame_id: &str) -> TelemetryResponse {
        TelemetryResponse {
            frame_id: frame_id.to_string(),
            gps: self.read_csv_record("gps", frame_id),
            imu: self.read_csv_record("imu", frame_id),
            lux: self.read_csv_record("lux", frame_id),
        }
    }

    fn read_csv_record(&self, folder: &str, frame_id: &str) -> Option<BTreeMap<String, Value>> {
        let candidates = [
            self.path
                .join(folder)
                .join("data")
                .join(format!("{frame_id}.csv")),
            self.path.join(folder).join(format!("{frame_id}.csv")),
        ];
        let csv_path = candidates.into_iter().find(|p| p.is_file())?;
        let text = fs::read_to_string(csv_path).ok()?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header_line = lines.next()?;
        let value_line = lines.next()?;

        let headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();
        let values: Vec<&str> = value_line.split(',').map(|s| s.trim()).collect();
        if headers.is_empty() || values.is_empty() {
            return None;
        }

        let mut record = BTreeMap::new();
        for (h, v) in headers.into_iter().zip(values.into_iter()) {
            if h.is_empty() {
                continue;
            }
            let json_val = if let Ok(num) = v.parse::<f64>() {
                if num.is_finite() {
                    serde_json::Number::from_f64(num)
                        .map(Value::Number)
                        .unwrap_or_else(|| Value::String(v.to_string()))
                } else {
                    Value::Null
                }
            } else {
                Value::String(v.to_string())
            };
            record.insert(h.to_string(), json_val);
        }

        if record.is_empty() {
            None
        } else {
            Some(record)
        }
    }
}

pub fn scan_for_sequences(base_dir: &Path) -> BTreeMap<String, Arc<DurLARSequence>> {
    let mut sequences = BTreeMap::new();
    if !base_dir.is_dir() {
        return sequences;
    }

    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let full_path = entry.path();
            if full_path.is_dir() {
                let seq = DurLARSequence::load(&full_path);
                if !seq.frames.is_empty() {
                    sequences.insert(name, Arc::new(seq));
                }
            }
        }
    }

    let root_seq = DurLARSequence::load(base_dir);
    if !root_seq.frames.is_empty() {
        sequences.insert(root_seq.name.clone(), Arc::new(root_seq));
    }

    sequences
}

fn reshape_3x3(vals: &[f64]) -> [[f64; 3]; 3] {
    [
        [vals[0], vals[1], vals[2]],
        [vals[3], vals[4], vals[5]],
        [vals[6], vals[7], vals[8]],
    ]
}

fn reshape_3x4(vals: &[f64]) -> [[f64; 4]; 3] {
    [
        [vals[0], vals[1], vals[2], vals[3]],
        [vals[4], vals[5], vals[6], vals[7]],
        [vals[8], vals[9], vals[10], vals[11]],
    ]
}
