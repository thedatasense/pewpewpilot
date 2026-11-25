#!/usr/bin/env python3
"""Download paired DurLAR_S exemplar frames or assemble split DurLAR archives."""

import argparse
import glob
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

DEFAULT_REPO_ID = "l1997i/DurLAR_S"
DEFAULT_DRIVE = "DurLAR_20210716_S"
CALIBRATION = Path(__file__).resolve().parent / "calibration" / "DurLAR_calibs.zip"
FRAME_EXTENSIONS = {
    "image_01": {".png", ".jpg"},
    "image_02": {".png", ".jpg"},
    "ambient": {".png", ".jpg"},
    "reflec": {".png", ".jpg"},
    "ouster_points": {".bin"},
}


def select_frame_files(files, drive, max_frames):
    """Select files by shared frame ID in a single drive."""
    sensors = {name: {} for name in FRAME_EXTENSIONS}
    for filename in files:
        path = PurePosixPath(filename)
        parts = path.parts
        if len(parts) not in (3, 4) or parts[0] != drive:
            continue
        if len(parts) == 4 and parts[2] != "data":
            continue
        modality = parts[1]
        if path.suffix in FRAME_EXTENSIONS.get(modality, ()) and path.stem.isascii() and path.stem.isdecimal():
            sensors[modality][path.stem] = filename

    shared_ids = sorted(sensors["ouster_points"].keys() & sensors["image_02"].keys(), key=int)
    return [
        (modality, files_by_frame[frame])
        for frame in shared_ids[:max_frames]
        for modality, files_by_frame in sensors.items()
        if frame in files_by_frame
    ]


def download_drive(files, repo_id, local_dir, token, max_frames, drive):
    from huggingface_hub import hf_hub_download

    selected = select_frame_files(files, drive, max_frames)
    if not selected:
        raise ValueError(f"No matching LiDAR and RGB frames found in {repo_id}/{drive}.")

    frames = [
        list(frame_files)
        for _, frame_files in groupby(selected, lambda item: PurePosixPath(item[1]).stem)
    ]
    destination = Path(local_dir) / drive
    (destination / "ouster_points" / "data").mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".durlar-", dir=destination.parent) as work:
        work = Path(work)
        if repo_id == DEFAULT_REPO_ID:
            with ZipFile(CALIBRATION) as archive:
                for name in ("calib_cam_to_cam.txt", "calib_lidar_to_cam.txt", "calib_imu_to_lidar.txt"):
                    target = destination / name
                    if not target.exists():
                        staged = work / name
                        staged.write_bytes(archive.read(name))
                        staged.replace(target)

        def download_frame(frame_files):
            pending = []
            for modality, filename in frame_files:
                target = destination / modality / "data" / PurePosixPath(filename).name
                if target.is_file() and target.stat().st_size:
                    continue
                fetched = hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="dataset",
                    local_dir=work, token=token,
                )
                pending.append((Path(fetched), target))

            # Publish the scan last so the viewer only discovers paired frames.
            for fetched, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                fetched.replace(target)

        print(f"{drive}: checking {len(frames)} paired frames", flush=True)
        with ThreadPoolExecutor(max_workers=6) as workers:
            for index, _ in enumerate(workers.map(download_frame, frames), start=1):
                if index % 25 == 0 or index == len(frames):
                    print(f"  {index}/{len(frames)} available", flush=True)

    return destination


def download_exemplar(repo_id=DEFAULT_REPO_ID, local_dir="data", token=None, max_frames=12, drive=DEFAULT_DRIVE):
    """Download one drive, or all DurLAR_S drives when drive is None."""
    from huggingface_hub import list_repo_files

    if drive is not None and (not drive or Path(drive).name != drive or drive in (".", "..")):
        raise ValueError("Choose a single DurLAR drive folder name.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("Frame count must be at least one.")

    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    if drive is not None:
        return download_drive(files, repo_id, local_dir, token, max_frames, drive)

    drives = sorted({
        filename.split("/", 1)[0]
        for filename in files
        if re.fullmatch(r"DurLAR_[0-9]{8}_S", filename.split("/", 1)[0])
    })
    if not drives:
        raise ValueError(f"No DurLAR_S drives found in {repo_id}.")
    return [download_drive(files, repo_id, local_dir, token, max_frames, name) for name in drives]


def merge_and_extract_tar(tar_prefix, output_dir):
    """Merges split tar files (e.g. tar.aa, tar.ab or tar00, tar01) and extracts them."""
    tar_files = sorted(glob.glob(f"{tar_prefix}*"))
    if not tar_files:
        print(f"No split tar files found matching pattern: {tar_prefix}*")
        return False
        
    combined_tar = os.path.join(output_dir, "combined_archive.tar")
    print(f"Merging {len(tar_files)} parts into {combined_tar}...")
    with open(combined_tar, "wb") as outfile:
        for part in tar_files:
            print(f"  Appending {part}...")
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
                
    print(f"Extracting {combined_tar} to {output_dir}...")
    subprocess.run(["tar", "-xf", combined_tar, "-C", output_dir], check=True)
    print("Extraction complete.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DurLAR Dataset Downloader")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--local-dir", default="data", help="Output directory")
    parser.add_argument("--token", default=None, help="Hugging Face access token (optional)")
    parser.add_argument("--exemplar-only", action="store_true", help="Download only exemplar/sample frames")
    drives = parser.add_mutually_exclusive_group()
    drives.add_argument("--drive", default=DEFAULT_DRIVE, help="DurLAR_S sequence folder")
    drives.add_argument("--all-drives", action="store_true", help="Download every DurLAR_S drive")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument("--frames", type=int, default=12, help="Number of matched frames to fetch per drive")
    frames.add_argument("--all-frames", action="store_true", help="Download every matched frame in each selected drive")
    parser.add_argument("--merge-tar", default=None, help="Prefix of split tar files to merge and extract")
    
    args = parser.parse_args()
    
    if args.merge_tar:
        merge_and_extract_tar(args.merge_tar, args.local_dir)
    else:
        try:
            result = download_exemplar(
                args.repo_id, args.local_dir, args.token,
                max_frames=None if args.all_frames else args.frames,
                drive=None if args.all_drives else args.drive,
            )
        except ImportError:
            parser.exit(1, "Install the downloader dependency with python3 -m pip install huggingface_hub\n")
        except Exception as error:
            parser.exit(1, f"Could not download DurLAR: {error}\n")
        for path in (result if args.all_drives else [result]):
            print(f"Saved recorded DurLAR frames to {path}")
