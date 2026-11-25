# DurLAR Sensor Viewer

Compare the forward camera and the 360° LiDAR point cloud from the same frame. The included drive contains 50 synthetic frames. Each dot represents a LiDAR return. Its color can show distance, height relative to the sensor, or return intensity.

## Start the viewer

Install the Python packages if needed.

```sh
python3 -m pip install fastapi uvicorn numpy pandas pillow
./run_viewer.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). To use another port, pass it as the first argument to the launcher. If the sample is missing, run `python3 generate_sample_data.py` and reload.

To open the sample without a server, run `python3 export_sample_viewer.py` and open `durlar-viewer.html` in a current browser. The file contains the full synthetic drive. The server UI can open additional sequences from the project's `data/` directory.

## Read the views

| Control | What it shows |
| --- | --- |
| **Top** | Looking down at the scene. The arrow marks the front of the reference vehicle. |
| **3D**, **Ego**, **Front** | Orbit, look from the LiDAR origin, or look back at the vehicle from ahead. |
| **Project LiDAR** | Draw a sample of visible LiDAR returns on the camera frame using the sequence calibration. |
| **Inspect a return** | Click a point in the cloud or on the camera overlay to measure it and mark its matching position. A return can fall outside the forward camera view. |
| **Ambient**, **Reflectivity** | Switch to the 360° images from the LiDAR. |
| **Display settings** | Change range, height, intensity, point density, overlay appearance, or reference guides. |
| **Playback** | Use the controls, Space, or the arrow keys to move through the frames. The 1× target is 10 frames per second. |

The measurements use LiDAR coordinates. Positive forward, left, and height are the x, y, and z axes. The projected marker shows where a measured point lands in the image. Occlusion and viewpoint differences can affect what’s visible there.

## View recorded DurLAR frames

Download 12 paired frames from the first [DurLAR_S exemplar drive](https://huggingface.co/datasets/l1997i/DurLAR_S) and make a browser file.

```sh
python3 -m pip install huggingface_hub
python3 download_durlar.py --exemplar-only --frames 12
python3 export_sample_viewer.py durlar-recorded.html --sequence data/DurLAR_20210716_S
```

Open `durlar-recorded.html` in your browser. If you start the local server instead, it selects the recorded drive by default when it finds one. The downloader pairs images and point clouds by their frame IDs. Camera overlay uses the [authors' calibration](https://github.com/l1997i/DurLAR/blob/main/DurLAR_calibs.zip).

To download all paired LiDAR and image frames from every exemplar drive, run:

```sh
python3 download_durlar.py --all-drives --all-frames
```

Rerun the same command to resume. It skips files already present and saves each drive under `data/`. Use `--all-drives --frames 12` for a short clip from each drive. The server viewer lists each downloaded drive in the sequence menu.

The source is *DurLAR: A High-fidelity 128-channel LiDAR Dataset with Panoramic Ambient and Reflectivity Imagery for Multi-modal Autonomous Driving Applications*, Li Li, Khalid N. Ismail, Hubert P. H. Shum, and Toby P. Breckon, 3D Vision, 2021. [Paper](https://arxiv.org/abs/2406.10068).

## Load another sequence

Place an extracted sequence directly under `data/`. Keep the sensor folders and calibration files together. The server expects point records in little-endian `[x, y, z, intensity]` float32 format. The overlay requires a LiDAR-to-camera transform and the projection matrix for the chosen image.

See the [DurLAR dataset](https://huggingface.co/datasets/l1997i/DurLAR) and `python3 download_durlar.py --help` for the existing download options.

## Railway deployment

See [Railway setup](RAILWAY.md) for the one-drive Hobby deployment.
