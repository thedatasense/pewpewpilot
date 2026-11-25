"""
DurLAR Dataset Loader & Parser
Handles loading, caching, calibration parsing, and binary streaming of DurLAR frames.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class DurLARSequence:
    def __init__(self, sequence_path: str):
        self.path = os.path.abspath(sequence_path)
        self.name = os.path.basename(self.path)
        self.frames = self._discover_frames()
        self.calibration = self._load_calibration()
        
    def _discover_frames(self) -> List[str]:
        # Frames can be in ouster_points/data/XXXXXX.bin
        pts_dir = os.path.join(self.path, "ouster_points", "data")
        if not os.path.isdir(pts_dir):
            # check directly under ouster_points
            pts_dir = os.path.join(self.path, "ouster_points")
            
        if os.path.isdir(pts_dir):
            bin_files = sorted(glob.glob(os.path.join(pts_dir, "*.bin")))
            return [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
            
        # Fallback to image_02 or any folder
        for sub in ["image_02/data", "image_02", "ambient/data", "ambient"]:
            test_dir = os.path.join(self.path, *sub.split("/"))
            if os.path.isdir(test_dir):
                img_files = sorted(glob.glob(os.path.join(test_dir, "*.png")))
                return [os.path.splitext(os.path.basename(f))[0] for f in img_files]
                
        return []
        
    def _load_calibration(self) -> Dict:
        calib = {
            "R_lidar_to_cam": np.eye(3).tolist(),
            "T_lidar_to_cam": [0.0, 0.0, 0.0],
            "P_rect_02": [
                [1850.0, 0.0, 1024.0, 0.0],
                [0.0, 1850.0, 544.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ],
            "P_rect_01": [
                [1850.0, 0.0, 1024.0, -388.5],
                [0.0, 1850.0, 544.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ],
            "image_width": 2048,
            "image_height": 1088,
            "lidar_channels": 128
        }
        
        extrinsic = self._calibration_fields("calib_lidar_to_cam.txt")
        camera = self._calibration_fields("calib_cam_to_cam.txt")
        rotation = extrinsic.get("R")
        translation = extrinsic.get("T")
        has_extrinsic = rotation is not None and len(rotation) == 9 and translation is not None and len(translation) == 3
        if has_extrinsic:
            calib["R_lidar_to_cam"] = np.array(rotation).reshape(3, 3).tolist()
            calib["T_lidar_to_cam"] = translation

        rectification = next(
            (camera[key] for key in ("R_rect_00", "R0_rect", "R_rect_02") if key in camera),
            None,
        )
        rectification_valid = rectification is None or len(rectification) == 9
        calib["R_rect_reference"] = np.array(rectification).reshape(3, 3).tolist() if rectification and rectification_valid else np.eye(3).tolist()
        calib["projectable_images"] = []
        calib["projection_sizes"] = {}
        for suffix in ("01", "02"):
            key = f"P_rect_{suffix}"
            projection = camera.get(key)
            size = camera.get(f"S_rect_{suffix}", [])
            if len(size) == 2 and min(size) > 0:
                calib["projection_sizes"][f"image_{suffix}"] = size
            if projection is not None and len(projection) == 12:
                calib[key] = np.array(projection).reshape(3, 4).tolist()
                if has_extrinsic and rectification_valid:
                    calib["projectable_images"].append(f"image_{suffix}")

        return calib
        
    def _calibration_fields(self, filename: str) -> Dict[str, List[float]]:
        for folder in (self.path, os.path.dirname(self.path)):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            fields = {}
            with open(path) as handle:
                for line in handle:
                    key, separator, value = line.partition(":")
                    if not separator:
                        continue
                    try:
                        values = [float(item) for item in value.split()]
                    except ValueError:
                        continue
                    fields[key.strip()] = values if np.isfinite(values).all() else []
            return fields
        return {}

    def get_point_cloud(self, frame_id: str, stride: int = 1) -> Tuple[np.ndarray, Dict]:
        """Loads [N, 4] point cloud array (x, y, z, intensity) as float32."""
        candidates = [
            os.path.join(self.path, "ouster_points", "data", f"{frame_id}.bin"),
            os.path.join(self.path, "ouster_points", f"{frame_id}.bin")
        ]
        bin_path = None
        for c in candidates:
            if os.path.isfile(c):
                bin_path = c
                break
                
        if not bin_path:
            return np.zeros((0, 4), dtype=np.float32), {}
            
        points = np.fromfile(bin_path, dtype="<f4").reshape(-1, 4)
        if stride > 1:
            points = points[::stride]
            
        # Compute summary stats
        stats = {
            "num_points": int(points.shape[0]),
            "x_min": float(points[:, 0].min()) if len(points) else 0.0,
            "x_max": float(points[:, 0].max()) if len(points) else 0.0,
            "y_min": float(points[:, 1].min()) if len(points) else 0.0,
            "y_max": float(points[:, 1].max()) if len(points) else 0.0,
            "z_min": float(points[:, 2].min()) if len(points) else 0.0,
            "z_max": float(points[:, 2].max()) if len(points) else 0.0,
            "intensity_min": float(points[:, 3].min()) if len(points) else 0.0,
            "intensity_max": float(points[:, 3].max()) if len(points) else 0.0,
        }
        return points, stats
        
    def get_image_path(self, modality: str, frame_id: str) -> Optional[str]:
        """Modality: 'image_02', 'image_01', 'ambient', 'reflec'"""
        candidates = [
            os.path.join(self.path, modality, "data", f"{frame_id}.png"),
            os.path.join(self.path, modality, f"{frame_id}.png"),
            os.path.join(self.path, modality, "data", f"{frame_id}.jpg"),
            os.path.join(self.path, modality, f"{frame_id}.jpg"),
            os.path.join(self.path, modality, "data", f"{frame_id}.tiff"),
            os.path.join(self.path, modality, f"{frame_id}.tiff"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None
        
    def get_telemetry(self, frame_id: str) -> Dict:
        res = {"frame_id": frame_id}
        
        # GPS
        gps_files = [
            os.path.join(self.path, "gps", "data", f"{frame_id}.csv"),
            os.path.join(self.path, "gps", f"{frame_id}.csv")
        ]
        for gf in gps_files:
            if os.path.isfile(gf):
                try:
                    df = pd.read_csv(gf)
                    res["gps"] = df.to_dict(orient="records")[0]
                except Exception:
                    pass
                break
                
        # IMU
        imu_files = [
            os.path.join(self.path, "imu", "data", f"{frame_id}.csv"),
            os.path.join(self.path, "imu", f"{frame_id}.csv")
        ]
        for imf in imu_files:
            if os.path.isfile(imf):
                try:
                    df = pd.read_csv(imf)
                    res["imu"] = df.to_dict(orient="records")[0]
                except Exception:
                    pass
                break
                
        # Lux
        lux_files = [
            os.path.join(self.path, "lux", "data", f"{frame_id}.csv"),
            os.path.join(self.path, "lux", f"{frame_id}.csv")
        ]
        for lf in lux_files:
            if os.path.isfile(lf):
                try:
                    df = pd.read_csv(lf)
                    res["lux"] = df.to_dict(orient="records")[0]
                except Exception:
                    pass
                break
                
        return res

def scan_for_sequences(base_dir: str) -> Dict[str, DurLARSequence]:
    """Scans base_dir for valid DurLAR sequence directories."""
    sequences = {}
    if not os.path.isdir(base_dir):
        return sequences
        
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and not entry.startswith("."):
            seq = DurLARSequence(full_path)
            if len(seq.frames) > 0:
                sequences[entry] = seq
                
    # Also check base_dir itself
    root_seq = DurLARSequence(base_dir)
    if len(root_seq.frames) > 0:
        sequences[os.path.basename(base_dir)] = root_seq
        
    return sequences
