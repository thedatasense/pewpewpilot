mod loader;
mod perception;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::loader::{DurLARSequence, scan_for_sequences};
use crate::perception::{
    device_name, init_engine_in_background, perception_cache_dir, read_cached_perception_json,
    run_frame_perception_json,
};

const DEFAULT_HF_REPO_ID: &str = "l1997i/DurLAR_S";

#[derive(Parser, Debug)]
#[command(name = "durlar-server", about = "DurLAR LiDAR & Camera Web Viewer API Server")]
struct Args {
    #[arg(long, env = "HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "PORT", default_value_t = 8000)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    root_dir: PathBuf,
    data_dir: PathBuf,
    models_dir: PathBuf,
    read_only: bool,
    default_stride: usize,
    sequences: Arc<RwLock<BTreeMap<String, Arc<DurLARSequence>>>>,
}

impl AppState {
    fn refresh_sequences(&self) -> BTreeMap<String, Arc<DurLARSequence>> {
        let scanned = scan_for_sequences(&self.data_dir);
        if let Ok(mut guard) = self.sequences.write() {
            *guard = scanned.clone();
        }
        scanned
    }

    fn get_sequence(&self, seq_id: &str) -> Result<Arc<DurLARSequence>, Response> {
        if let Ok(guard) = self.sequences.read() {
            if let Some(seq) = guard.get(seq_id) {
                return Ok(Arc::clone(seq));
            }
        }
        let refreshed = self.refresh_sequences();
        if let Some(seq) = refreshed.get(seq_id) {
            return Ok(Arc::clone(seq));
        }
        let available: Vec<&String> = refreshed.keys().collect();
        Err(api_error(
            StatusCode::NOT_FOUND,
            format!("Sequence '{seq_id}' not found. Available: {available:?}"),
        ))
    }

    fn get_frame_sequence(&self, seq_id: &str, frame_id: &str) -> Result<Arc<DurLARSequence>, Response> {
        let seq = self.get_sequence(seq_id)?;
        if seq.has_frame(frame_id) {
            return Ok(seq);
        }
        // Rescan once in case frames were just downloaded
        let refreshed = self.refresh_sequences();
        if let Some(updated) = refreshed.get(seq_id) {
            if updated.has_frame(frame_id) {
                return Ok(Arc::clone(updated));
            }
        }
        Err(api_error(StatusCode::NOT_FOUND, "Frame not found"))
    }
}

fn api_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": detail.into() }))).into_response()
}

fn resolve_project_root() -> PathBuf {
    if let Ok(explicit) = env::var("DURLAR_ROOT_DIR") {
        let p = PathBuf::from(explicit);
        if p.is_dir() {
            return p;
        }
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("static").is_dir() {
        return cwd;
    }
    if let Ok(exe) = env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            if ancestor.join("static").is_dir() {
                return ancestor.to_path_buf();
            }
        }
    }
    cwd
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let read_only = env::var("DURLAR_READ_ONLY").as_deref() == Ok("1");
    let default_stride: usize = env::var("DURLAR_POINT_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    if !(1..=10).contains(&default_stride) {
        panic!("DURLAR_POINT_STRIDE must be between 1 and 10.");
    }

    let root_dir = resolve_project_root();
    let data_dir = env::var("DURLAR_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root_dir.join("data"));
    let static_dir = env::var("DURLAR_STATIC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root_dir.join("static"));
    let models_dir = env::var("DURLAR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root_dir.join("models"));

    let _ = fs::create_dir_all(&data_dir);
    let _ = fs::create_dir_all(&static_dir);

    if !read_only {
        init_engine_in_background(models_dir.clone());
    }

    let initial_sequences = scan_for_sequences(&data_dir);
    let state = AppState {
        root_dir,
        data_dir,
        models_dir,
        read_only,
        default_stride,
        sequences: Arc::new(RwLock::new(initial_sequences)),
    };

    let mut router = axum::Router::new()
        .route("/health", get(health))
        .route("/api/sequences", get(list_sequences))
        .route("/api/sequence/{seq_id}/info", get(sequence_info))
        .route(
            "/api/sequence/{seq_id}/frame/{frame_id}/points_bin",
            get(get_points_binary),
        )
        .route(
            "/api/sequence/{seq_id}/frame/{frame_id}/stats",
            get(get_points_stats),
        )
        .route(
            "/api/sequence/{seq_id}/frame/{frame_id}/image/{modality}",
            get(get_image),
        )
        .route(
            "/api/sequence/{seq_id}/frame/{frame_id}/telemetry",
            get(get_telemetry),
        )
        .route(
            "/api/sequence/{seq_id}/frame/{frame_id}/perception",
            get(get_frame_perception),
        )
        .route(
            "/api/sequence/{seq_id}/perception/status",
            get(get_perception_status),
        );

    if !read_only {
        router = router
            .route("/api/generate_sample", post(trigger_generate_sample))
            .route("/api/hf/download_sample", post(trigger_hf_download))
            .route(
                "/api/sequence/{seq_id}/perception/warmup",
                post(trigger_perception_warmup),
            );
    }

    let mut app: axum::Router = router
        .fallback_service(ServeDir::new(&static_dir).append_index_html_on_directories(true))
        .layer(middleware::from_fn(static_no_cache))
        .with_state(state);

    if !read_only {
        app = app.layer(CorsLayer::permissive());
    }

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid host/port");
    println!("DurLAR Rust server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");
    axum::serve(listener, app).await.expect("Server error");
}

async fn static_no_cache(req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let mut response = next.run(req).await;
    if path == "/"
        || path.ends_with(".html")
        || path.ends_with(".css")
        || path.ends_with(".js")
    {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    response
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn list_sequences(State(state): State<AppState>) -> Json<serde_json::Value> {
    let seqs = state.refresh_sequences();
    let result: Vec<_> = seqs
        .iter()
        .map(|(name, seq)| seq.summary(name, state.read_only))
        .collect();
    Json(json!({ "sequences": result }))
}

async fn sequence_info(
    State(state): State<AppState>,
    AxumPath(seq_id): AxumPath<String>,
) -> Result<Json<loader::SequenceInfo>, Response> {
    // Refresh on sequence selection so newly downloaded frames appear immediately
    let seqs = state.refresh_sequences();
    let Some(seq) = seqs.get(&seq_id) else {
        let available: Vec<&String> = seqs.keys().collect();
        return Err(api_error(
            StatusCode::NOT_FOUND,
            format!("Sequence '{seq_id}' not found. Available: {available:?}"),
        ));
    };
    Ok(Json(seq.info()))
}

#[derive(Deserialize)]
struct StrideQuery {
    stride: Option<isize>,
}

async fn get_points_binary(
    State(state): State<AppState>,
    AxumPath((seq_id, frame_id)): AxumPath<(String, String)>,
    Query(query): Query<StrideQuery>,
) -> Result<Response, Response> {
    let stride_val = query.stride.unwrap_or(state.default_stride as isize);
    if !(1..=10).contains(&stride_val) {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "stride must be between 1 and 10",
        ));
    }
    let stride = stride_val as usize;
    let seq = state.get_frame_sequence(&seq_id, &frame_id)?;

    let (bytes, count) = tokio::task::spawn_blocking(move || seq.read_points_binary(&frame_id, stride))
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut resp = (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        bytes,
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&count.to_string()) {
        resp.headers_mut().insert("X-Point-Count", v);
    }
    if let Ok(v) = HeaderValue::from_str(&stride.to_string()) {
        resp.headers_mut().insert("X-Point-Stride", v);
    }
    Ok(resp)
}

async fn get_points_stats(
    State(state): State<AppState>,
    AxumPath((seq_id, frame_id)): AxumPath<(String, String)>,
) -> Result<Json<loader::PointStats>, Response> {
    let seq = state.get_frame_sequence(&seq_id, &frame_id)?;
    let stats = tokio::task::spawn_blocking(move || seq.point_stats(&frame_id))
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(stats))
}

async fn get_image(
    State(state): State<AppState>,
    AxumPath((seq_id, frame_id, modality)): AxumPath<(String, String, String)>,
) -> Result<Response, Response> {
    if !matches!(
        modality.as_str(),
        "image_02" | "image_01" | "ambient" | "reflec"
    ) {
        return Err(api_error(StatusCode::NOT_FOUND, "Image modality not found"));
    }
    let seq = state.get_frame_sequence(&seq_id, &frame_id)?;
    let mod_clone = modality.clone();
    let fid_clone = frame_id.clone();

    let img_opt = tokio::task::spawn_blocking(move || seq.read_image_bytes(&mod_clone, &fid_clone))
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let Some((bytes, content_type)) = img_opt else {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            format!("Image {modality}/{frame_id} not found"),
        ));
    };

    Ok((StatusCode::OK, [(header::CONTENT_TYPE, content_type)], bytes).into_response())
}

async fn get_telemetry(
    State(state): State<AppState>,
    AxumPath((seq_id, frame_id)): AxumPath<(String, String)>,
) -> Result<Json<loader::TelemetryResponse>, Response> {
    let seq = state.get_frame_sequence(&seq_id, &frame_id)?;
    let telem = tokio::task::spawn_blocking(move || seq.telemetry(&frame_id))
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(telem))
}

async fn get_frame_perception(
    State(state): State<AppState>,
    AxumPath((seq_id, frame_id)): AxumPath<(String, String)>,
) -> Result<Response, Response> {
    let seq = state.get_frame_sequence(&seq_id, &frame_id)?;
    let cache_dir = perception_cache_dir(&seq);

    if let Some(cached_json) = read_cached_perception_json(&cache_dir, &frame_id) {
        return Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            cached_json,
        )
            .into_response());
    }

    if state.read_only {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "Perception not precomputed in read-only mode",
        ));
    }

    let models_dir = state.models_dir.clone();
    let json_res = tokio::task::spawn_blocking(move || {
        run_frame_perception_json(&seq, &frame_id, &cache_dir, &models_dir)
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match json_res {
        Ok(json_body) => Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            json_body,
        )
            .into_response()),
        Err(err) => Err(api_error(StatusCode::SERVICE_UNAVAILABLE, err)),
    }
}

#[derive(Serialize)]
struct PerceptionStatus {
    sequence: String,
    cached_frames: usize,
    total_frames: usize,
    device: &'static str,
    read_only: bool,
}

async fn get_perception_status(
    State(state): State<AppState>,
    AxumPath(seq_id): AxumPath<String>,
) -> Result<Json<PerceptionStatus>, Response> {
    let seq = state.get_sequence(&seq_id)?;
    let cache_dir = perception_cache_dir(&seq);
    let mut cached = 0usize;
    if cache_dir.is_dir() {
        cached = seq
            .frames
            .iter()
            .filter(|fid| cache_dir.join(format!("{fid}.json")).is_file())
            .count();
    }

    Ok(Json(PerceptionStatus {
        sequence: seq_id,
        cached_frames: cached,
        total_frames: seq.frames.len(),
        device: device_name(&state.models_dir),
        read_only: state.read_only,
    }))
}

#[derive(Deserialize)]
struct WarmupQuery {
    start: Option<isize>,
    count: Option<isize>,
}

async fn trigger_perception_warmup(
    State(state): State<AppState>,
    AxumPath(seq_id): AxumPath<String>,
    Query(query): Query<WarmupQuery>,
) -> Result<Json<serde_json::Value>, Response> {
    let start_raw = query.start.unwrap_or(0);
    let count_raw = query.count.unwrap_or(60);
    if start_raw < 0 || !(1..=600).contains(&count_raw) {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid start or count parameters",
        ));
    }
    let start = start_raw as usize;
    let count = count_raw as usize;

    let seq = state.get_sequence(&seq_id)?;
    let total = seq.frames.len();
    let actual_count = count.min(total.saturating_sub(start));
    let models_dir = state.models_dir.clone();
    let dev = device_name(&models_dir);

    let seq_bg = Arc::clone(&seq);
    tokio::task::spawn_blocking(move || {
        let cache_dir = perception_cache_dir(&seq_bg);
        let end = (start + count).min(seq_bg.frames.len());
        for fid in &seq_bg.frames[start..end] {
            let _ = run_frame_perception_json(&seq_bg, fid, &cache_dir, &models_dir);
        }
    });

    Ok(Json(json!({
        "status": "queued",
        "sequence": seq_id,
        "start": start,
        "count": actual_count,
        "device": dev,
    })))
}

#[derive(Deserialize)]
struct GenerateSampleQuery {
    frames: Option<isize>,
}

async fn trigger_generate_sample(
    State(state): State<AppState>,
    Query(query): Query<GenerateSampleQuery>,
) -> Result<Json<serde_json::Value>, Response> {
    let frames = query.frames.unwrap_or(10);
    if !(1..=50).contains(&frames) {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "frames must be between 1 and 50",
        ));
    }

    let script = state.root_dir.join("generate_sample_data.py");
    if !script.is_file() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "generate_sample_data.py not found",
        ));
    }

    let status = Command::new("python3")
        .current_dir(&state.root_dir)
        .arg("-c")
        .arg(format!(
            "import generate_sample_data; generate_sample_data.main(num_frames={frames})"
        ))
        .status()
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !status.success() {
        return Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Sample generation failed",
        ));
    }

    state.refresh_sequences();
    Ok(Json(json!({
        "status": "success",
        "message": format!("Generated {frames} sample DurLAR frames"),
    })))
}

#[derive(Deserialize)]
struct HfDownloadQuery {
    repo_id: Option<String>,
}

async fn trigger_hf_download(
    State(state): State<AppState>,
    Query(query): Query<HfDownloadQuery>,
) -> Result<Json<serde_json::Value>, Response> {
    let repo_id = query
        .repo_id
        .unwrap_or_else(|| DEFAULT_HF_REPO_ID.to_string());
    let script = state.root_dir.join("download_durlar.py");
    if !script.is_file() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "download_durlar.py not found",
        ));
    }

    let data_dir: &Path = &state.data_dir;
    let data_dir_str = data_dir.to_string_lossy().into_owned();
    let root_dir = state.root_dir.clone();
    let repo_clone = repo_id.clone();
    let state_bg = state.clone();

    tokio::spawn(async move {
        let status = Command::new("python3")
            .current_dir(&root_dir)
            .arg("download_durlar.py")
            .arg("--repo-id")
            .arg(&repo_clone)
            .arg("--local-dir")
            .arg(&data_dir_str)
            .status()
            .await;
        if matches!(status, Ok(s) if s.success()) {
            state_bg.refresh_sequences();
        }
    });

    Ok(Json(json!({
        "status": "queued",
        "message": format!("Initiated download of sample frames from Hugging Face: {repo_id}"),
    })))
}
