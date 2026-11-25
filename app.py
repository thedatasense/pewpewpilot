#!/usr/bin/env python3
"""
DurLAR Web Viewer Server
FastAPI application providing streaming endpoints for 128-channel LiDAR point clouds,
panoramic ambient & reflectivity imagery, stereo camera feeds, and telemetry.
"""

import argparse
import os
import io
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from durlar_loader import scan_for_sequences, DurLARSequence
import download_durlar
import generate_sample_data
import perception

app = FastAPI(title="DurLAR LiDAR Dataset Viewer", version="1.0.0")

READ_ONLY = os.getenv("DURLAR_READ_ONLY") == "1"
DEFAULT_POINT_STRIDE = int(os.getenv("DURLAR_POINT_STRIDE", "1"))
if not 1 <= DEFAULT_POINT_STRIDE <= 10:
    raise ValueError("DURLAR_POINT_STRIDE must be between 1 and 10.")

if not READ_ONLY:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BASE_DATA_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Cache loaded sequences
_sequences_cache = {}

def get_sequences():
    global _sequences_cache
    _sequences_cache = scan_for_sequences(BASE_DATA_DIR)
    return _sequences_cache

def get_seq(seq_id: str) -> DurLARSequence:
    seqs = get_sequences()
    if seq_id not in seqs:
        raise HTTPException(status_code=404, detail=f"Sequence '{seq_id}' not found. Available: {list(seqs.keys())}")
    return seqs[seq_id]

def get_frame_sequence(seq_id: str, frame_id: str) -> DurLARSequence:
    seq = get_seq(seq_id)
    if frame_id not in seq.frames:
        raise HTTPException(status_code=404, detail="Frame not found")
    return seq

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/sequences")
def list_sequences():
    seqs = get_sequences()
    result = []
    for name, seq in seqs.items():
        result.append({
            "id": name,
            **({} if READ_ONLY else {"path": seq.path}),
            "frame_count": len(seq.frames),
            "first_frame": seq.frames[0] if seq.frames else None,
            "last_frame": seq.frames[-1] if seq.frames else None,
            "has_points": os.path.isdir(os.path.join(seq.path, "ouster_points")),
            "has_camera": os.path.isdir(os.path.join(seq.path, "image_02")),
            "has_ambient": os.path.isdir(os.path.join(seq.path, "ambient")),
            "has_reflec": os.path.isdir(os.path.join(seq.path, "reflec")),
            "has_telemetry": os.path.isdir(os.path.join(seq.path, "gps")),
        })
    return {"sequences": result}

@app.get("/api/sequence/{seq_id}/info")
def sequence_info(seq_id: str):
    seq = get_seq(seq_id)
    return {
        "id": seq.name,
        "frames": seq.frames,
        "calibration": seq.calibration,
        "modalities": {
            "lidar": "Ouster OS1-128 (128 channels, 360° FOV, KITTI bin)",
            "camera_left": "Carnegie Robotics MultiSense S21 RGB (2048x1088)",
            "camera_right": "Carnegie Robotics MultiSense S21 Grayscale (2048x1088)",
            "panoramic_ambient": "Ouster Panoramic Near-IR Ambient (2048x128)",
            "panoramic_reflec": "Ouster Panoramic Surface Reflectivity (2048x128)",
            "telemetry": "OxTS RT3000v3 GNSS/INS + Yocto Lux Meter"
        }
    }

@app.get("/api/sequence/{seq_id}/frame/{frame_id}/points_bin")
def get_points_binary(seq_id: str, frame_id: str, stride: int = Query(DEFAULT_POINT_STRIDE, ge=1, le=10)):
    """
    Returns point cloud as binary float32 buffer [x, y, z, intensity].
    Ideal for direct WebGL ArrayBuffer consumption.
    """
    seq = get_frame_sequence(seq_id, frame_id)
    points, _ = seq.get_point_cloud(frame_id, stride=stride)
    return Response(
        content=points.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Point-Count": str(len(points)),
            "X-Point-Stride": str(stride)
        }
    )

@app.get("/api/sequence/{seq_id}/frame/{frame_id}/stats")
def get_points_stats(seq_id: str, frame_id: str):
    seq = get_frame_sequence(seq_id, frame_id)
    _, stats = seq.get_point_cloud(frame_id, stride=1)
    return stats

@app.get("/api/sequence/{seq_id}/frame/{frame_id}/image/{modality}")
def get_image(seq_id: str, frame_id: str, modality: str):
    """modality: 'image_02', 'image_01', 'ambient', 'reflec'"""
    if modality not in {"image_02", "image_01", "ambient", "reflec"}:
        raise HTTPException(status_code=404, detail="Image modality not found")
    seq = get_frame_sequence(seq_id, frame_id)
    img_path = seq.get_image_path(modality, frame_id)
    if not img_path or not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail=f"Image {modality}/{frame_id} not found")
    if img_path.lower().endswith((".tif", ".tiff")):
        buffer = io.BytesIO()
        with Image.open(img_path) as image:
            image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    return FileResponse(img_path)

@app.get("/api/sequence/{seq_id}/frame/{frame_id}/telemetry")
def get_telemetry_endpoint(seq_id: str, frame_id: str):
    seq = get_frame_sequence(seq_id, frame_id)
    return seq.get_telemetry(frame_id)


def _perception_cache_dir(seq: DurLARSequence) -> str:
    return os.path.join(seq.path, "perception")


@app.get("/api/sequence/{seq_id}/frame/{frame_id}/perception")
def get_frame_perception(seq_id: str, frame_id: str):
    seq = get_frame_sequence(seq_id, frame_id)
    cache_dir = _perception_cache_dir(seq)
    if READ_ONLY and not os.path.isfile(os.path.join(cache_dir, f"{frame_id}.json")):
        raise HTTPException(status_code=404, detail="Perception not precomputed in read-only mode")
    return perception.run_frame_perception(seq, frame_id, cache_dir=cache_dir)


@app.get("/api/sequence/{seq_id}/perception/status")
def get_perception_status(seq_id: str):
    seq = get_seq(seq_id)
    cache_dir = _perception_cache_dir(seq)
    cached = 0
    if os.path.isdir(cache_dir):
        cached = sum(1 for fid in seq.frames if os.path.isfile(os.path.join(cache_dir, f"{fid}.json")))
    return {
        "sequence": seq_id,
        "cached_frames": cached,
        "total_frames": len(seq.frames),
        "device": perception.get_device(),
        "read_only": READ_ONLY,
    }


def _warmup_sequence(seq: DurLARSequence, start_index: int, count: int):
    cache_dir = _perception_cache_dir(seq)
    for fid in seq.frames[start_index : start_index + count]:
        try:
            perception.run_frame_perception(seq, fid, cache_dir=cache_dir)
        except Exception:
            continue


def trigger_perception_warmup(
    seq_id: str,
    background_tasks: BackgroundTasks,
    start: int = Query(0, ge=0),
    count: int = Query(60, ge=1, le=600),
):
    seq = get_seq(seq_id)
    background_tasks.add_task(_warmup_sequence, seq, start, count)
    return {
        "status": "queued",
        "sequence": seq_id,
        "start": start,
        "count": min(count, max(0, len(seq.frames) - start)),
        "device": perception.get_device(),
    }


def trigger_generate_sample(frames: int = Query(10, ge=1, le=50)):
    generate_sample_data.main(num_frames=frames)
    get_sequences()
    return {"status": "success", "message": f"Generated {frames} sample DurLAR frames"}

def trigger_hf_download(background_tasks: BackgroundTasks, repo_id: str = download_durlar.DEFAULT_REPO_ID):
    background_tasks.add_task(download_durlar.download_exemplar, repo_id, BASE_DATA_DIR)
    return {"status": "queued", "message": f"Initiated download of sample frames from Hugging Face: {repo_id}"}

if not READ_ONLY:
    app.post("/api/generate_sample")(trigger_generate_sample)
    app.post("/api/hf/download_sample")(trigger_hf_download)
    app.post("/api/sequence/{seq_id}/perception/warmup")(trigger_perception_warmup)

# Serve static web viewer files
@app.middleware("http")
async def static_no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache"
    return response

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local DurLAR viewer")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
