"""Package a local DurLAR sequence and viewer as a single offline HTML file."""

import argparse
import base64
import csv
import glob
import gzip
import json
import mimetypes
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLE = ROOT / "data" / "sample_durlar_sequence"


class LocalSequence:
    """Self-contained DurLAR sequence reader for standalone HTML bundle export."""

    def __init__(self, sequence_path: str):
        self.path = os.path.abspath(sequence_path)
        self.name = os.path.basename(self.path)
        self.frames = self._discover_frames()
        self.calibration = self._load_calibration()

    def _discover_frames(self):
        pts_dir = os.path.join(self.path, "ouster_points", "data")
        if not os.path.isdir(pts_dir):
            pts_dir = os.path.join(self.path, "ouster_points")
        if os.path.isdir(pts_dir):
            bin_files = sorted(glob.glob(os.path.join(pts_dir, "*.bin")))
            return [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
        for sub in ("image_02/data", "image_02", "ambient/data", "ambient"):
            test_dir = os.path.join(self.path, *sub.split("/"))
            if os.path.isdir(test_dir):
                img_files = sorted(glob.glob(os.path.join(test_dir, "*.png")))
                return [os.path.splitext(os.path.basename(f))[0] for f in img_files]
        return []

    def _calibration_fields(self, filename: str):
        for folder in (self.path, os.path.dirname(self.path)):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            fields = {}
            with open(path) as handle:
                for line in handle:
                    key, sep, value = line.partition(":")
                    if not sep:
                        continue
                    try:
                        values = [float(item) for item in value.split()]
                    except ValueError:
                        continue
                    fields[key.strip()] = values if np.isfinite(values).all() else []
            return fields
        return {}

    def _load_calibration(self):
        calib = {
            "R_lidar_to_cam": np.eye(3).tolist(),
            "T_lidar_to_cam": [0.0, 0.0, 0.0],
            "P_rect_02": [[1850.0, 0.0, 1024.0, 0.0], [0.0, 1850.0, 544.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            "P_rect_01": [[1850.0, 0.0, 1024.0, -388.5], [0.0, 1850.0, 544.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            "image_width": 2048,
            "image_height": 1088,
            "lidar_channels": 128,
        }
        extrinsic = self._calibration_fields("calib_lidar_to_cam.txt")
        camera = self._calibration_fields("calib_cam_to_cam.txt")
        rotation = extrinsic.get("R")
        translation = extrinsic.get("T")
        has_extrinsic = rotation is not None and len(rotation) == 9 and translation is not None and len(translation) == 3
        if has_extrinsic:
            calib["R_lidar_to_cam"] = np.array(rotation).reshape(3, 3).tolist()
            calib["T_lidar_to_cam"] = translation

        rect = next((camera[k] for k in ("R_rect_00", "R0_rect", "R_rect_02") if k in camera), None)
        rect_valid = rect is None or len(rect) == 9
        calib["R_rect_reference"] = np.array(rect).reshape(3, 3).tolist() if rect and rect_valid else np.eye(3).tolist()
        calib["projectable_images"] = []
        calib["projection_sizes"] = {}
        for suffix in ("01", "02"):
            proj = camera.get(f"P_rect_{suffix}")
            size = camera.get(f"S_rect_{suffix}", [])
            if len(size) == 2 and min(size) > 0:
                calib["projection_sizes"][f"image_{suffix}"] = size
            if proj is not None and len(proj) == 12:
                calib[f"P_rect_{suffix}"] = np.array(proj).reshape(3, 4).tolist()
                if has_extrinsic and rect_valid:
                    calib["projectable_images"].append(f"image_{suffix}")
        return calib

    def get_point_cloud(self, frame_id: str):
        for candidate in (
            os.path.join(self.path, "ouster_points", "data", f"{frame_id}.bin"),
            os.path.join(self.path, "ouster_points", f"{frame_id}.bin"),
        ):
            if os.path.isfile(candidate):
                return np.fromfile(candidate, dtype="<f4").reshape(-1, 4), {}
        return np.zeros((0, 4), dtype=np.float32), {}

    def get_image_path(self, modality: str, frame_id: str):
        for ext in ("png", "jpg", "tiff"):
            for sub in (os.path.join(modality, "data"), modality):
                candidate = os.path.join(self.path, sub, f"{frame_id}.{ext}")
                if os.path.isfile(candidate):
                    return candidate
        return None

    def get_telemetry(self, frame_id: str):
        res = {"frame_id": frame_id}
        for folder in ("gps", "imu", "lux"):
            for sub in (os.path.join(folder, "data"), folder):
                csv_path = os.path.join(self.path, sub, f"{frame_id}.csv")
                if os.path.isfile(csv_path):
                    with open(csv_path, newline="") as handle:
                        rows = list(csv.DictReader(handle))
                        if rows:
                            parsed = {}
                            for k, v in rows[0].items():
                                if k is None:
                                    continue
                                try:
                                    parsed[k.strip()] = float(v)
                                except (TypeError, ValueError):
                                    parsed[k.strip()] = v
                            res[folder] = parsed
                    break
        return res


def sample_resources(sequence_path=SAMPLE):
    sample = LocalSequence(str(sequence_path))
    if not sample.frames:
        raise SystemExit(f"No frames found in {sequence_path}.")
    prefix = f"/api/sequence/{sample.name}"
    resources = {
        "/api/sequences": ["application/json", {"sequences": [{"id": sample.name}]}],
        f"{prefix}/info": ["application/json", {
            "id": sample.name, "frames": sample.frames, "calibration": sample.calibration,
        }],
    }

    for frame in sample.frames:
        url = f"{prefix}/frame/{frame}"
        points, _ = sample.get_point_cloud(frame)
        points_bytes = gzip.compress(points.tobytes(), compresslevel=3, mtime=0)
        resources[f"{url}/points_bin"] = ["application/octet-stream", base64.b64encode(points_bytes).decode()]
        resources[f"{url}/telemetry"] = ["application/json", sample.get_telemetry(frame)]
        for modality in ("image_02", "image_01", "ambient", "reflec"):
            image = sample.get_image_path(modality, frame)
            if image:
                resources[f"{url}/image/{modality}"] = [
                    mimetypes.guess_type(image)[0], base64.b64encode(Path(image).read_bytes()).decode(),
                ]
    return resources


def export_viewer(output, sequence_path=SAMPLE):
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "viewer.css").read_text()
    core = (STATIC / "point-cloud.js").read_text().replace("export ", "")
    ui = re.sub(r"^import \{[^}]+\} from './point-cloud\.js(?:\?[^']*)?';\n", "", (STATIC / "viewer.js").read_text())
    data = json.dumps(sample_resources(sequence_path), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = re.sub(r'<link rel="stylesheet" href="/viewer\.css(?:\?[^"]*)?">', lambda _: f"<style>\n{css}</style>", html)
    html = re.sub(
        r'<script type="module" src="/viewer\.js(?:\?[^"]*)?"></script>',
        lambda _: (
            f'<script id="sample-data" type="application/json">{data}</script>\n'
            f'<script type="module">\n{core}\n{ui}</script>'
        ),
        html,
    )
    output.write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("durlar-viewer.html"))
    parser.add_argument("--sequence", type=Path, default=SAMPLE, help="Local DurLAR sequence directory")
    args = parser.parse_args()
    export_viewer(args.output, args.sequence)
