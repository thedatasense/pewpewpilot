#!/usr/bin/env python3
"""
Export RT-DETR (PekingU/rtdetr_r50vd) and Depth Anything V2 Small
(depth-anything/Depth-Anything-V2-Small-hf) to static-shape ONNX models
for high-speed CoreML / ONNX Runtime inference in the Rust backend.
"""

import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation, RTDetrForObjectDetection

DETECTOR_ID = "PekingU/rtdetr_r50vd"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class RTDetrWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


class DepthWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values=pixel_values).predicted_depth


def consolidate_single_file(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    data_file = Path(f"{path}.data")
    onnx.save_model(model, str(path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()


def export_rtdetr(out_path: Path, local_only: bool) -> None:
    if out_path.is_file() and out_path.stat().st_size > 1_000_000:
        print(f"Skipping RT-DETR export (already exists at {out_path})")
        return
    print(f"Loading {DETECTOR_ID}...")
    base = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID, local_files_only=local_only).eval()
    wrapper = RTDetrWrapper(base).eval()
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting RT-DETR to {out_path}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    consolidate_single_file(out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MiB)")


def export_depth(out_path: Path, local_only: bool) -> None:
    if out_path.is_file() and out_path.stat().st_size > 1_000_000:
        print(f"Skipping Depth Anything V2 export (already exists at {out_path})")
        return
    print(f"Loading {DEPTH_ID}...")
    base = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID, local_files_only=local_only).eval()
    wrapper = DepthWrapper(base).eval()
    dummy = torch.zeros(1, 3, 518, 518, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting Depth Anything V2 to {out_path}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    consolidate_single_file(out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory to write exported ONNX models",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only locally cached Hugging Face weights",
    )
    args = parser.parse_args()

    export_rtdetr(args.models_dir / "rtdetr_r50vd.onnx", args.local_only)
    export_depth(args.models_dir / "depth_anything_v2_small.onnx", args.local_only)


if __name__ == "__main__":
    main()
