"""Package a local DurLAR sequence and viewer as a single offline HTML file."""

import argparse
import base64
import gzip
import json
import mimetypes
import re
from pathlib import Path

from durlar_loader import DurLARSequence

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLE = ROOT / "data" / "sample_durlar_sequence"


def sample_resources(sequence_path=SAMPLE):
    sample = DurLARSequence(str(sequence_path))
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
