"""
Local Apple Silicon (MPS) / CPU Real-Time FSD Perception Engine
Runs RT-DETR (object detection) + Depth Anything V2 (dense monocular depth)
fused with calibrated 128-channel Ouster LiDAR scans to output:
1. 3D bounding boxes / positions for vehicles, pedestrians, cyclists, signs
2. Smooth filled 3D road & terrain mesh from LiDAR ground + metric depth
3. Dense RGB-colored depth-completed foreground points
"""

import base64
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, RTDetrForObjectDetection

DETECTOR_ID = "PekingU/rtdetr_r50vd"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"

DRIVING_CLASSES = {
    "person": (0.65, 0.65, 1.72),
    "bicycle": (1.75, 0.70, 1.40),
    "car": (4.20, 1.85, 1.50),
    "motorcycle": (2.00, 0.80, 1.40),
    "bus": (8.50, 2.50, 3.10),
    "truck": (6.50, 2.35, 2.85),
    "traffic light": (0.45, 0.45, 0.95),
    "stop sign": (0.65, 0.25, 0.75),
    "dog": (0.75, 0.40, 0.65),
}

_MODEL_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_models():
    with _MODEL_LOCK:
        if _MODELS:
            return _MODELS
        device = get_device()
        det_proc = AutoImageProcessor.from_pretrained(DETECTOR_ID)
        det_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval()
        dep_proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
        dep_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()
        _MODELS.update({
            "device": device,
            "det_proc": det_proc,
            "det_model": det_model,
            "dep_proc": dep_proc,
            "dep_model": dep_model,
        })
        return _MODELS


def _calibration_matrices(calib: Dict, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_l2c = np.array(calib["R_lidar_to_cam"], dtype=np.float64)
    t_l2c = np.array(calib["T_lidar_to_cam"], dtype=np.float64)
    r_rect = np.array(calib.get("R_rect_reference", np.eye(3)), dtype=np.float64)
    p_rect = np.array(calib["P_rect_02"], dtype=np.float64).copy()

    proj_sizes = calib.get("projection_sizes", {})
    if "image_02" in proj_sizes:
        sw, sh = proj_sizes["image_02"]
        if sw > 0 and sh > 0 and (sw != img_w or sh != img_h):
            p_rect[0, :] *= img_w / sw
            p_rect[1, :] *= img_h / sh
    elif img_w != 2048 or img_h != 1088:
        # Fallback check for default calibration size if projection_sizes absent
        if abs(p_rect[0, 2] * 2 - img_w) > 10:
            p_rect[0, :] *= img_w / max(1.0, p_rect[0, 2] * 2)
            p_rect[1, :] *= img_h / max(1.0, p_rect[1, 2] * 2)

    return r_l2c, t_l2c, r_rect, p_rect


def _project_lidar(points: np.ndarray, r_l2c: np.ndarray, t_l2c: np.ndarray, r_rect: np.ndarray, p_rect: np.ndarray, img_w: int, img_h: int):
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=bool)
    xyz = points[:, :3].astype(np.float64)
    cam = (r_rect @ (r_l2c @ xyz.T + t_l2c[:, None])).T  # (N, 3)
    depth = cam[:, 2]
    valid_d = depth > 0.8
    homo = np.hstack([cam, np.ones((len(cam), 1))])
    proj = (p_rect @ homo.T).T
    u = proj[:, 0] / np.where(valid_d, depth, 1.0)
    v = proj[:, 1] / np.where(valid_d, depth, 1.0)
    in_bounds = valid_d & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return np.stack([u, v, depth], axis=1), in_bounds


def _unproject_to_lidar(u: np.ndarray, v: np.ndarray, z_cam: np.ndarray, r_l2c: np.ndarray, t_l2c: np.ndarray, r_rect: np.ndarray, p_rect: np.ndarray) -> np.ndarray:
    fx = p_rect[0, 0]
    fy = p_rect[1, 1]
    cx = p_rect[0, 2]
    cy = p_rect[1, 2]
    tx = p_rect[0, 3]
    ty = p_rect[1, 3]

    x_c = ((u - cx) * z_cam - tx) / fx
    y_c = ((v - cy) * z_cam - ty) / fy
    cam_rect = np.stack([x_c, y_c, z_cam], axis=1)  # (N, 3)
    # p_c = R_rect (R X_L + T)  =>  X_L = R^T (R_rect^T p_c - T)
    cam_raw = (r_rect.T @ cam_rect.T).T - t_l2c[None, :]
    lidar_xyz = (r_l2c.T @ cam_raw.T).T
    return lidar_xyz.astype(np.float32)


def _align_metric_depth(rel_disp: np.ndarray, uvz: np.ndarray, in_bounds: np.ndarray) -> np.ndarray:
    """Fits metric inverse-depth (1/Z = a * rel_disp + b) using calibrated LiDAR returns."""
    h, w = rel_disp.shape
    pts = uvz[in_bounds]
    pts = pts[(pts[:, 2] > 1.2) & (pts[:, 2] < 55.0)]

    if len(pts) < 40:
        # Rough monotonic fallback if too few LiDAR points overlap camera
        norm = (rel_disp - rel_disp.min()) / max(1e-5, rel_disp.max() - rel_disp.min())
        return 1.0 / np.clip(0.02 + 0.35 * norm, 1.0 / 60.0, 1.0 / 1.0)

    ui = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    vi = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    x = rel_disp[vi, ui].astype(np.float64)
    y = 1.0 / pts[:, 2].astype(np.float64)

    # Subsample for fast robust 2-stage trimmed least squares
    if len(x) > 4000:
        step = len(x) // 4000
        x = x[::step]
        y = y[::step]

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.abs(y - (a * x + b))
    mad = max(1e-4, float(np.median(resid)))
    inliers = resid <= 2.2 * mad
    if np.sum(inliers) >= 20:
        a, b = np.linalg.lstsq(A[inliers], y[inliers], rcond=None)[0]

    if a <= 1e-6:
        scale = float(np.median(y / np.maximum(1e-4, x)))
        a, b = max(1e-4, scale), 0.0

    inv_depth = np.clip(a * rel_disp.astype(np.float64) + b, 1.0 / 65.0, 1.0 / 0.8)
    return (1.0 / inv_depth).astype(np.float32)


def _build_road_mesh(
    points: np.ndarray,
    img_arr: np.ndarray,
    metric_depth: np.ndarray,
    r_l2c: np.ndarray,
    t_l2c: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
) -> np.ndarray:
    """
    Constructs a smooth filled 3D road/terrain surface mesh (x, y, z, r, g, b, a)
    covering [-16..52m] x [-18..18m] around the vehicle using LiDAR ground returns
    and camera color sampling.
    """
    nx, ny = 35, 21  # 34 x 20 quads = 1,360 triangles
    xs = np.linspace(-16.0, 52.0, nx)
    ys = np.linspace(-18.0, 18.0, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    z_sum = np.zeros((nx, ny), dtype=np.float64)
    z_cnt = np.zeros((nx, ny), dtype=np.float64)

    if len(points):
        g_mask = (
            (points[:, 2] >= -2.3)
            & (points[:, 2] <= -1.10)
            & (points[:, 0] >= xs[0])
            & (points[:, 0] <= xs[-1])
            & (points[:, 1] >= ys[0])
            & (points[:, 1] <= ys[-1])
        )
        gp = points[g_mask]
        if len(gp):
            ix = np.clip(np.round((gp[:, 0] - xs[0]) / dx).astype(np.int32), 0, nx - 1)
            iy = np.clip(np.round((gp[:, 1] - ys[0]) / dy).astype(np.int32), 0, ny - 1)
            np.add.at(z_sum, (ix, iy), gp[:, 2])
            np.add.at(z_cnt, (ix, iy), 1.0)

    z_grid = np.where(z_cnt > 0, z_sum / np.maximum(1.0, z_cnt), np.nan)

    # Smoothly fill sparse/occluded cells from neighbors
    default_ground = -1.68
    for _ in range(3):
        nan_mask = np.isnan(z_grid)
        if not nan_mask.any():
            break
        padded = np.pad(z_grid, 1, mode="edge")
        acc = np.zeros_like(z_grid)
        cnt = np.zeros_like(z_grid)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                sub = padded[1 + ox : 1 + ox + nx, 1 + oy : 1 + oy + ny]
                valid = ~np.isnan(sub)
                acc[valid] += sub[valid]
                cnt[valid] += 1.0
        filled = np.where(cnt > 0, acc / np.maximum(1.0, cnt), np.nan)
        z_grid[nan_mask] = filled[nan_mask]
    z_grid = np.nan_to_num(z_grid, nan=default_ground)

    # Light smoothing pass so triangles don't spike on single curb outliers
    padded = np.pad(z_grid, 1, mode="edge")
    z_smooth = (
        4.0 * z_grid
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 8.0

    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), z_smooth.ravel()], axis=1)

    h, w, _ = img_arr.shape
    uvz, in_fov = _project_lidar(grid_pts, r_l2c, t_l2c, r_rect, p_rect, w, h)

    # Base dark FSD asphalt color, subtly modulated by camera RGB inside forward FOV
    colors = np.zeros((len(grid_pts), 4), dtype=np.float32)
    dist = np.hypot(grid_pts[:, 0], grid_pts[:, 1])
    fade = np.clip(1.0 - np.maximum(0.0, dist - 28.0) / 26.0, 0.08, 0.85)
    # Dark asphalt base
    colors[:, 0] = 0.055
    colors[:, 1] = 0.075
    colors[:, 2] = 0.11
    colors[:, 3] = fade

    if in_fov.any():
        ui = np.clip(uvz[in_fov, 0].astype(np.int32), 0, w - 1)
        vi = np.clip(uvz[in_fov, 1].astype(np.int32), 0, h - 1)
        rgb = img_arr[vi, ui].astype(np.float32) / 255.0
        # Keep asphalt dark & cohesive while showing real pavement/grass/curb tone
        luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        # Only apply RGB texture if metric depth agrees closely with ground (avoids sky/tree color bleeding onto distant floor)
        d_diff = np.abs(metric_depth[vi, ui] - uvz[in_fov, 2])
        ground_match = (d_diff < 5.0)[:, None]
        tinted = np.stack([
            0.05 + 0.18 * rgb[:, 0] + 0.04 * luma,
            0.065 + 0.20 * rgb[:, 1] + 0.04 * luma,
            0.095 + 0.22 * rgb[:, 2] + 0.05 * luma,
        ], axis=1)
        colors[in_fov, :3] = np.where(ground_match, tinted, colors[in_fov, :3])
        colors[in_fov, 3] = np.where(ground_match[:, 0], np.maximum(fade[in_fov], 0.82), fade[in_fov])

    verts = np.hstack([grid_pts.astype(np.float32), colors]).reshape(nx, ny, 7)
    # Build triangle list: two triangles per quad
    v00 = verts[:-1, :-1].reshape(-1, 7)
    v10 = verts[1:, :-1].reshape(-1, 7)
    v11 = verts[1:, 1:].reshape(-1, 7)
    v01 = verts[:-1, 1:].reshape(-1, 7)
    tris = np.concatenate([v00, v10, v11, v00, v11, v01], axis=1).reshape(-1, 7)
    return tris.astype(np.float32)


def _build_dense_points(
    img_arr: np.ndarray,
    metric_depth: np.ndarray,
    r_l2c: np.ndarray,
    t_l2c: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
) -> np.ndarray:
    """
    Back-projects non-sky Depth Anything pixels into RGB-colored 3D LiDAR points
    [x, y, z, r, g, b] to fill gaps between laser rings on buildings, trees, and curbs.
    """
    h, w, _ = img_arr.shape
    stride = 6 if w >= 1000 else 4
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    u_flat = uu.ravel().astype(np.float64)
    v_flat = vv.ravel().astype(np.float64)
    z_flat = metric_depth[vv, uu].ravel().astype(np.float64)

    keep_d = (z_flat > 1.2) & (z_flat < 48.0)
    if not keep_d.any():
        return np.zeros((0, 6), dtype=np.float32)

    xyz = _unproject_to_lidar(u_flat[keep_d], v_flat[keep_d], z_flat[keep_d], r_l2c, t_l2c, r_rect, p_rect)
    # Keep above-ground structures & skip sky/overly tall outliers
    keep_z = (xyz[:, 2] > -1.35) & (xyz[:, 2] < 7.5)
    xyz = xyz[keep_z]
    if not len(xyz):
        return np.zeros((0, 6), dtype=np.float32)

    rgb = (img_arr[vv.ravel()[keep_d][keep_z], uu.ravel()[keep_d][keep_z]].astype(np.float32)) / 255.0
    return np.hstack([xyz, rgb]).astype(np.float32)


def _extract_3d_objects(
    detections: Dict,
    id2label: Dict[int, str],
    points: np.ndarray,
    uvz: np.ndarray,
    in_bounds: np.ndarray,
    metric_depth: np.ndarray,
    r_l2c: np.ndarray,
    t_l2c: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[Dict]:
    objects = []
    lidar_uvz = uvz[in_bounds]
    lidar_xyz = points[in_bounds, :3] if len(points) else np.zeros((0, 3), dtype=np.float32)

    for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        label = id2label[int(label_id)]
        if label not in DRIVING_CLASSES:
            continue
        conf = float(score)
        if conf < 0.45:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        ix1 = x1 + 0.15 * bw
        ix2 = x2 - 0.15 * bw
        iy1 = y1 + 0.05 * bh
        iy2 = y2 - 0.05 * bh

        prior_l, prior_w, prior_h = DRIVING_CLASSES[label]
        cx_px = 0.5 * (x1 + x2)
        cy_px = 0.5 * (y1 + y2)

        in_box = (
            (lidar_uvz[:, 0] >= ix1)
            & (lidar_uvz[:, 0] <= ix2)
            & (lidar_uvz[:, 1] >= iy1)
            & (lidar_uvz[:, 1] <= iy2)
            & (lidar_uvz[:, 2] > 1.0)
            & (lidar_uvz[:, 2] < 60.0)
        )

        source = "depth"
        yaw = 0.0
        if np.sum(in_box) >= 5:
            pts_box = lidar_xyz[in_box]
            d_box = lidar_uvz[in_box, 2]
            d_anchor = float(np.percentile(d_box, 30))
            tol = 1.6 if label in ("person", "bicycle", "motorcycle", "dog") else 3.2
            cluster = pts_box[np.abs(d_box - d_anchor) <= tol]
            if len(cluster) >= 4:
                source = "lidar"
                med_x = float(np.median(cluster[:, 0]))
                med_y = float(np.median(cluster[:, 1]))
                z_min = float(np.percentile(cluster[:, 2], 5))
                z_max = float(np.percentile(cluster[:, 2], 98))
                est_h = float(np.clip(z_max - z_min, prior_h * 0.88, prior_h * 1.25))
                if label in ("car", "bus", "truck"):
                    # Estimate vehicle orientation from LiDAR xy spread when visible side is broad
                    if len(cluster) >= 8:
                        xy = cluster[:, :2] - np.mean(cluster[:, :2], axis=0)
                        cov = xy.T @ xy
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        major = eigvecs[:, 1]
                        span = float(np.sqrt(max(0.0, eigvals[1]) / len(cluster))) * 3.0
                        if span > 1.3:
                            yaw = float(np.arctan2(major[1], major[0]))
                        elif bw > bh * 1.5:
                            yaw = float(np.arctan2(med_y, med_x) + np.pi / 2.0)
                    elif bw > bh * 1.5:
                        yaw = float(np.arctan2(med_y, med_x) + np.pi / 2.0)

                    # Push visible surface center inward slightly along viewing ray
                    ray_xy = np.array([med_x, med_y], dtype=np.float64)
                    ray_n = max(1e-3, float(np.linalg.norm(ray_xy)))
                    inward = prior_w * 0.45 if bw > bh * 1.4 else prior_l * 0.42
                    med_x += (ray_xy[0] / ray_n) * inward
                    med_y += (ray_xy[1] / ray_n) * inward
                cx, cy = med_x, med_y
                cz = -1.68 + est_h / 2.0 if label not in ("traffic light", "stop sign") else 0.5 * (z_min + z_max)
                size = [prior_l, prior_w, est_h]
            else:
                source = "depth"

        if source == "depth":
            px1 = int(np.clip(ix1, 0, img_w - 1))
            px2 = int(np.clip(ix2 + 1, px1 + 1, img_w))
            py1 = int(np.clip(iy1, 0, img_h - 1))
            py2 = int(np.clip(iy2 + 1, py1 + 1, img_h))
            patch = metric_depth[py1:py2, px1:px2]
            z_cam = float(np.percentile(patch, 35))
            xyz = _unproject_to_lidar(
                np.array([cx_px]), np.array([cy_px]), np.array([z_cam]),
                r_l2c, t_l2c, r_rect, p_rect,
            )[0]
            cx, cy, cz = float(xyz[0]), float(xyz[1]), float(xyz[2])
            if label not in ("traffic light", "stop sign"):
                cz = -1.68 + prior_h / 2.0
            if label in ("car", "bus", "truck") and bw > bh * 1.5:
                yaw = float(np.arctan2(cy, cx) + np.pi / 2.0)
            size = [prior_l, prior_w, prior_h]

        objects.append({
            "label": label,
            "score": round(conf, 3),
            "box2d": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round(cx, 2), round(cy, 2), round(cz, 2)],
            "size": [round(s, 2) for s in size],
            "yaw": round(yaw, 3),
            "source": source,
        })

    return objects


def run_frame_perception(seq, frame_id: str, cache_dir: Optional[str] = None) -> Dict:
    """Runs or loads cached RT-DETR + Depth Anything V2 + LiDAR fusion for a single frame."""
    cache_path = os.path.join(cache_dir, f"{frame_id}.json") if cache_dir else None
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r") as handle:
            cached = json.load(handle)
        cached["cached"] = True
        return cached

    img_path = seq.get_image_path("image_02", frame_id)
    if not img_path or not os.path.isfile(img_path):
        return {
            "frame_id": frame_id,
            "objects": [],
            "road_mesh_b64": "",
            "dense_points_b64": "",
            "device": get_device(),
            "inference_ms": 0.0,
            "cached": False,
        }

    models = _load_models()
    device = models["device"]
    det_proc = models["det_proc"]
    det_model = models["det_model"]
    dep_proc = models["dep_proc"]
    dep_model = models["dep_model"]

    t0 = time.perf_counter()
    with Image.open(img_path) as im:
        rgb_img = im.convert("RGB")
    img_w, img_h = rgb_img.size
    img_arr = np.asarray(rgb_img)

    with torch.no_grad():
        det_inputs = det_proc(images=rgb_img, return_tensors="pt").to(device)
        det_outputs = det_model(**det_inputs)
        detections = det_proc.post_process_object_detection(
            det_outputs, target_sizes=torch.tensor([(img_h, img_w)]), threshold=0.45
        )[0]

        dep_inputs = dep_proc(images=rgb_img, return_tensors="pt").to(device)
        dep_out = dep_model(**dep_inputs).predicted_depth
        dep_resized = F.interpolate(
            dep_out.unsqueeze(1), size=(img_h, img_w), mode="bilinear", align_corners=False
        ).squeeze()
        if device == "mps":
            torch.mps.synchronize()
        rel_disp = dep_resized.detach().cpu().numpy()

    points, _ = seq.get_point_cloud(frame_id, stride=1)
    r_l2c, t_l2c, r_rect, p_rect = _calibration_matrices(seq.calibration, img_w, img_h)
    uvz, in_bounds = _project_lidar(points, r_l2c, t_l2c, r_rect, p_rect, img_w, img_h)

    metric_depth = _align_metric_depth(rel_disp, uvz, in_bounds)
    road_mesh = _build_road_mesh(points, img_arr, metric_depth, r_l2c, t_l2c, r_rect, p_rect)
    dense_points = _build_dense_points(img_arr, metric_depth, r_l2c, t_l2c, r_rect, p_rect)
    objects = _extract_3d_objects(
        detections, det_model.config.id2label, points, uvz, in_bounds,
        metric_depth, r_l2c, t_l2c, r_rect, p_rect, img_w, img_h,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    result = {
        "frame_id": frame_id,
        "device": device,
        "inference_ms": elapsed_ms,
        "cached": False,
        "objects": objects,
        "road_mesh_b64": base64.b64encode(road_mesh.tobytes()).decode("ascii"),
        "dense_points_b64": base64.b64encode(dense_points.tobytes()).decode("ascii"),
        "dense_count": int(len(dense_points)),
    }

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache_path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(result, handle, separators=(",", ":"))
        os.replace(tmp, cache_path)

    return result
