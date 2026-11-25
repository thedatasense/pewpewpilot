#!/usr/bin/env python3
"""
Realistic DurLAR Multi-Modal Dataset Generator
Generates realistic, smooth 128-channel DurLAR dataset frames:
- Fixed world objects (buildings, trees, parked cars, curbs) that naturally
  flow past the ego vehicle as it drives forward down the street.
- 50 continuous frames (5.0 seconds at 10 Hz) for smooth, non-choppy playback.
- Matching forward camera scenes with aligned 3D LiDAR projection.
- Realistic 360° panoramic ambient and reflectivity returns.
- Continuous GPS, IMU, and Lux telemetry.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_durlar_sequence")

def create_calibration(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # LiDAR to Camera extrinsic:
    # Camera: X right, Y down, Z forward
    # LiDAR: X forward, Y left, Z up
    # Standard transformation:
    # X_cam = -Y_lidar
    # Y_cam = -Z_lidar
    # Z_cam =  X_lidar
    R = np.array([
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0]
    ], dtype=np.float64)
    
    # Camera is ~0.25m below LiDAR, ~0.10m forward
    T = np.array([0.0, 0.25, -0.10], dtype=np.float64) # meters

    with open(os.path.join(output_dir, "calib_lidar_to_cam.txt"), "w") as f:
        f.write("calib_time: 2021-08-15 14:30:00\n")
        f.write("R: " + " ".join(f"{x:.8e}" for x in R.flatten()) + "\n")
        f.write("T: " + " ".join(f"{x:.8e}" for x in T.flatten()) + "\n")
        
    # MultiSense S21 intrinsics (2048 x 1088)
    fx = 1750.0
    fy = 1750.0
    cx = 1024.0
    cy = 544.0
    
    P_rect_02 = np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float64)
    
    # Right camera (image_01, baseline 0.21m)
    P_rect_01 = np.array([
        [fx, 0.0, cx, -fx * 0.21],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float64)
    
    R_rect = np.eye(3, dtype=np.float64)

    with open(os.path.join(output_dir, "calib_cam_to_cam.txt"), "w") as f:
        f.write("calib_time: 2021-08-15 14:30:00\n")
        f.write("R_rect_00: " + " ".join(f"{x:.8e}" for x in R_rect.flatten()) + "\n")
        f.write("P_rect_01: " + " ".join(f"{x:.8e}" for x in P_rect_01.flatten()) + "\n")
        f.write("R_rect_02: " + " ".join(f"{x:.8e}" for x in R_rect.flatten()) + "\n")
        f.write("P_rect_02: " + " ".join(f"{x:.8e}" for x in P_rect_02.flatten()) + "\n")
        
    with open(os.path.join(output_dir, "calib_imu_to_lidar.txt"), "w") as f:
        f.write("calib_time: 2021-08-15 14:30:00\n")
        f.write("R: 1.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00\n")
        f.write("T: 0.00000000e+00 0.00000000e+00 -3.00000000e-01\n")
        
    return R, T, P_rect_02

# Define a persistent world of objects along the street
WORLD_BUILDINGS = [
    # (x_center, y_center, width_x, depth_y, height_z, base_reflectivity)
    (12.0,  9.0, 10.0, 6.0, 5.5, 0.45),
    (28.0,  9.5, 12.0, 7.0, 6.0, 0.50),
    (46.0,  9.0, 14.0, 6.0, 4.8, 0.42),
    (68.0, 10.0, 18.0, 8.0, 7.2, 0.55),
    (92.0,  9.2, 16.0, 6.5, 5.0, 0.48),
    
    (15.0, -9.0, 11.0, 6.0, 4.5, 0.46),
    (32.0, -9.5, 13.0, 7.0, 6.5, 0.52),
    (52.0, -9.0, 15.0, 6.0, 5.2, 0.44),
    (74.0, -10.0, 16.0, 8.0, 6.8, 0.50),
    (98.0, -9.2, 14.0, 6.5, 5.4, 0.47),
]

WORLD_TREES = [
    # (x_center, y_center, radius, height)
    (6.0,   5.8, 1.4, 4.2),
    (21.0,  5.8, 1.6, 4.8),
    (38.0,  6.0, 1.5, 4.5),
    (60.0,  5.8, 1.7, 5.0),
    (82.0,  6.0, 1.5, 4.6),
    
    (8.0,  -5.8, 1.5, 4.4),
    (24.0, -5.8, 1.6, 4.7),
    (43.0, -6.0, 1.4, 4.3),
    (64.0, -5.8, 1.8, 5.2),
    (86.0, -6.0, 1.5, 4.5),
]

WORLD_PARKED_CARS = [
    # (x_center, y_center, length_x, width_y, height_z, color_tuple)
    (18.0,  4.2, 4.4, 1.8, 1.45, (60, 90, 180)),   # Blue sedan
    (36.0,  4.2, 4.8, 1.9, 1.60, (220, 220, 225)), # Silver SUV
    (58.0,  4.2, 4.2, 1.8, 1.40, (180, 50, 40)),   # Red hatchback
    (26.0, -4.2, 4.5, 1.8, 1.50, (50, 50, 55)),    # Black sedan
    (48.0, -4.2, 4.6, 1.85, 1.55, (230, 230, 230)), # White car
    (78.0, -4.2, 4.3, 1.8, 1.45, (40, 120, 80)),   # Green car
]

def generate_scene_points(frame_idx, num_rings=128, azimuth_steps=1024):
    """
    Simulates a 128-channel Ouster OS1-128 LiDAR scan of a real urban street.
    Ego vehicle moves forward smoothly along X.
    All scenery is in WORLD coordinates, so objects glide past the vehicle naturally.
    """
    h_lidar = 1.7 # LiDAR mount height in meters
    speed = 7.0   # m/s (~25 km/h)
    dt = 0.1      # 10 Hz
    ego_x = frame_idx * speed * dt
    ego_y = 0.0
    
    # Lead vehicle ahead moves along with us at ~16m ahead
    lead_x = ego_x + 16.0 + 0.1 * math.sin(frame_idx * 0.2)
    lead_y = 0.2
    
    # Oncoming car in opposite lane (y ~ -2.0) moving towards us
    oncoming_x = 90.0 - frame_idx * 9.0 * dt
    oncoming_y = -2.2
    
    # LiDAR beam angles
    phi_min, phi_max = np.radians(-22.5), np.radians(22.5)
    elevations = np.linspace(phi_min, phi_max, num_rings)
    azimuths = np.linspace(-np.pi, np.pi, azimuth_steps, endpoint=False)
    
    phi_grid, theta_grid = np.meshgrid(elevations, azimuths)
    dx = np.cos(phi_grid) * np.cos(theta_grid)
    dy = np.cos(phi_grid) * np.sin(theta_grid)
    dz = np.sin(phi_grid)
    
    ranges = np.full_like(dx, np.nan)
    intensities = np.zeros_like(dx)
    
    # 1. Ground plane (road, curbs, sidewalks)
    ground_mask = dz < -0.015
    t_ground = -h_lidar / np.where(ground_mask, dz, -1.0)
    valid_ground = ground_mask & (t_ground > 0.8) & (t_ground < 70.0)
    
    # Compute ground intersection in WORLD coordinates
    gx_lidar = dx * t_ground
    gy_lidar = dy * t_ground
    
    gx_world = gx_lidar + ego_x
    gy_world = gy_lidar + ego_y
    
    # Road width is 7.0m (|y| < 3.5m)
    on_road = (np.abs(gy_world) < 3.5)
    # Center dashed lines (period 6m, length 3m)
    center_line = (np.abs(gy_world) < 0.12) & ((gx_world % 6.0) < 3.0) & on_road
    # Edge white lines at y = +/- 3.3m
    edge_lines = (np.abs(np.abs(gy_world) - 3.3) < 0.10)
    
    # Base asphalt reflectivity: ~0.20 - 0.25
    # Lane markings retro-reflectivity: ~0.90 - 0.98
    # Curbs and sidewalks at |y| between 3.5m and 6.5m: ~0.35 - 0.40
    # Grass verge at |y| > 6.5m: ~0.15 - 0.22
    g_intensity = np.full_like(gx_world, 0.22)
    g_intensity[on_road] = 0.25
    g_intensity[center_line | edge_lines] = 0.94
    
    sidewalk = (np.abs(gy_world) >= 3.5) & (np.abs(gy_world) < 6.5)
    g_intensity[sidewalk] = 0.38
    grass = (np.abs(gy_world) >= 6.5)
    g_intensity[grass] = 0.18
    
    ranges[valid_ground] = t_ground[valid_ground]
    intensities[valid_ground] = g_intensity[valid_ground]
    
    # 2. Lead Vehicle in front (relative pos: lead_x - ego_x, lead_y)
    rel_lead_x = lead_x - ego_x
    rel_lead_y = lead_y - ego_y
    if rel_lead_x > 1.0:
        # Box dimensions: length 4.5m, width 1.8m, height 1.5m
        bx_min, bx_max = rel_lead_x - 2.2, rel_lead_x + 2.2
        by_min, by_max = rel_lead_y - 0.9, rel_lead_y + 0.9
        bz_min, bz_max = -h_lidar, -h_lidar + 1.45
        
        # Rays hitting rear face
        rear_mask = (dx > 0.05)
        t_car = bx_min / np.where(rear_mask, dx, 1.0)
        cy = dy * t_car
        cz = dz * t_car
        hit_car = rear_mask & (t_car > 0.5) & (cy >= by_min) & (cy <= by_max) & (cz >= bz_min) & (cz <= bz_max)
        
        mask = hit_car & (np.isnan(ranges) | (t_car < ranges))
        ranges[mask] = t_car[mask]
        # License plate and taillights
        is_plate = (np.abs(cy - rel_lead_y) < 0.35) & (np.abs(cz - (bz_min + 0.35)) < 0.12)
        is_taillight = (np.abs(np.abs(cy - rel_lead_y) - 0.7) < 0.15) & (np.abs(cz - (bz_min + 0.5)) < 0.12)
        car_int = np.where(is_plate | is_taillight, 0.98, 0.65)
        intensities[mask] = car_int[mask]
        
    # 3. Oncoming Vehicle (opposite lane)
    rel_onc_x = oncoming_x - ego_x
    rel_onc_y = oncoming_y - ego_y
    if -15.0 < rel_onc_x < 60.0:
        bx_min, bx_max = rel_onc_x - 2.2, rel_onc_x + 2.2
        by_min, by_max = rel_onc_y - 0.9, rel_onc_y + 0.9
        bz_min, bz_max = -h_lidar, -h_lidar + 1.45
        
        # Front face
        front_mask = (dx > 0.05)
        t_onc = bx_max / np.where(front_mask, dx, 1.0)
        cy = dy * t_onc
        cz = dz * t_onc
        hit_onc = front_mask & (t_onc > 0.5) & (cy >= by_min) & (cy <= by_max) & (cz >= bz_min) & (cz <= bz_max)
        mask = hit_onc & (np.isnan(ranges) | (t_onc < ranges))
        ranges[mask] = t_onc[mask]
        intensities[mask] = 0.72

    # 4. Parked Cars along curbs
    for (pcx, pcy, plen, pwid, pheight, _) in WORLD_PARKED_CARS:
        rx = pcx - ego_x
        ry = pcy - ego_y
        if -10.0 < rx < 60.0:
            # Simple oriented bounding box check
            x_min, x_max = rx - plen/2, rx + plen/2
            y_min, y_max = ry - pwid/2, ry + pwid/2
            z_min, z_max = -h_lidar, -h_lidar + pheight
            
            # Check side facing road
            side_y = y_min if ry > 0 else y_max
            target_mask = (dy > 0.05) if ry > 0 else (dy < -0.05)
            t_side = side_y / np.where(target_mask, dy, 1.0)
            sx = dx * t_side
            sz = dz * t_side
            hit = target_mask & (t_side > 0.5) & (sx >= x_min) & (sx <= x_max) & (sz >= z_min) & (sz <= z_max)
            mask = hit & (np.isnan(ranges) | (t_side < ranges))
            ranges[mask] = t_side[mask]
            intensities[mask] = 0.60

    # 5. Persistent Buildings with gaps and varying heights
    for (bx, by, bw, bd, bh, b_ref) in WORLD_BUILDINGS:
        rx = bx - ego_x
        ry = by - ego_y
        if -15.0 < rx < 75.0:
            x_min, x_max = rx - bw/2, rx + bw/2
            front_y = ry - bd/2 if ry > 0 else ry + bd/2
            z_min, z_max = -h_lidar, -h_lidar + bh
            
            b_mask = (dy > 0.03) if ry > 0 else (dy < -0.03)
            t_bldg = front_y / np.where(b_mask, dy, 1.0)
            bx_pt = dx * t_bldg
            bz_pt = dz * t_bldg
            
            hit_b = b_mask & (t_bldg > 1.0) & (t_bldg < 80.0) & (bx_pt >= x_min) & (bx_pt <= x_max) & (bz_pt >= z_min) & (bz_pt <= z_max)
            mask = hit_b & (np.isnan(ranges) | (t_bldg < ranges))
            ranges[mask] = t_bldg[mask]
            # Facade texture with windows
            win_pattern = (np.sin(bx_pt * 2.0) > 0.2) & (np.sin(bz_pt * 2.5) > 0.0)
            intensities[mask] = np.where(win_pattern[mask], b_ref * 0.5, b_ref)

    # 6. Persistent Trees (foliage and trunks)
    for (tx, ty, trad, th) in WORLD_TREES:
        rx = tx - ego_x
        ry = ty - ego_y
        if -8.0 < rx < 65.0:
            dist_horiz = np.sqrt(rx**2 + ry**2)
            if dist_horiz > 2.0:
                t_approx = dist_horiz
                # Ray direction to tree center
                tree_dir = np.array([rx, ry, -h_lidar + th * 0.7])
                tree_dir = tree_dir / np.linalg.norm(tree_dir)
                cos_ang = dx * tree_dir[0] + dy * tree_dir[1] + dz * tree_dir[2]
                hit_tree = (cos_ang > 0.975) & (t_ground > dist_horiz - trad)
                mask = hit_tree & (np.isnan(ranges) | (dist_horiz < ranges))
                ranges[mask] = dist_horiz
                intensities[mask] = 0.35 + 0.1 * np.sin(rx + ry)

    valid = ~np.isnan(ranges) & (ranges > 0.8) & (ranges < 75.0)
    pts_x = (dx * ranges)[valid]
    pts_y = (dy * ranges)[valid]
    pts_z = (dz * ranges)[valid]
    pts_i = np.clip(intensities[valid], 0.0, 1.0)
    
    points = np.stack([pts_x, pts_y, pts_z, pts_i], axis=-1).astype(np.float32)
    
    # Panoramic ambient & reflectivity (2048 x 128)
    pano_width = 2048
    pano_height = 128
    col_ratio = pano_width // azimuth_steps
    
    ring_img_reflec = (intensities.T)[::-1, :]
    ring_img_range = (np.nan_to_num(ranges.T, nan=0.0))[::-1, :]
    
    expanded_reflec = np.repeat(ring_img_reflec, col_ratio, axis=1)
    expanded_range = np.repeat(ring_img_range, col_ratio, axis=1)
    
    pano_reflec = np.zeros((pano_height, pano_width), dtype=np.float32)
    pano_reflec[:, :expanded_reflec.shape[1]] = expanded_reflec
    
    sky_gradient = np.linspace(0.85, 0.15, pano_height)[:, None]
    pano_ambient = np.clip(sky_gradient * 0.35 + pano_reflec * 0.55 + (expanded_range > 0) * 0.1, 0.0, 1.0)
    
    return points, pano_ambient, pano_reflec, ego_x, lead_x, oncoming_x

def render_camera_images(frame_idx, ego_x, lead_x, oncoming_x, R, T, P_rect, output_w=2048, output_h=1088):
    """
    Renders realistic forward camera view aligning accurately with the 3D LiDAR point cloud.
    """
    img = Image.new("RGB", (output_w, output_h), (150, 190, 230)) # Daylight sky
    draw = ImageDraw.Draw(img)
    
    horizon_y = int(output_h * 0.51)
    
    # Distant tree skyline
    draw.rectangle([0, horizon_y - 60, output_w, horizon_y], fill=(95, 130, 115))
    # Road surface (asphalt)
    draw.rectangle([0, horizon_y, output_w, output_h], fill=(70, 72, 76))
    
    vp_x = output_w // 2
    vp_y = horizon_y
    
    # Sidewalks and curbs
    road_left = int(output_w * 0.22)
    road_right = int(output_w * 0.78)
    draw.polygon([(road_left, output_h), (vp_x - 20, vp_y), (vp_x + 20, vp_y), (road_right, output_h)], fill=(55, 57, 60))
    
    # Sidewalk paving
    draw.polygon([(0, output_h), (road_left, output_h), (vp_x - 20, vp_y), (int(output_w * 0.1), vp_y)], fill=(120, 118, 115))
    draw.polygon([(road_right, output_h), (output_w, output_h), (int(output_w * 0.9), vp_y), (vp_x + 20, vp_y)], fill=(120, 118, 115))

    # Center lane dashed markings (traveling with ego_x)
    dash_color = (240, 240, 240)
    # The dashed line world position:
    for world_marker_x in range(int(ego_x), int(ego_x + 60), 6):
        rel_x = world_marker_x - ego_x
        if rel_x > 2.0:
            # Perspective mapping
            t = 2.0 / rel_x
            w_dash = max(2, int(25 * t))
            y_dash = int(horizon_y + 540 * t)
            if y_dash < output_h:
                draw.rectangle([vp_x - w_dash//2, y_dash - 12, vp_x + w_dash//2, y_dash], fill=dash_color)

    # Lead Car (in front at rel_x = lead_x - ego_x)
    rel_lead_x = lead_x - ego_x
    if rel_lead_x > 3.0:
        # Project 3D bounding box corners into camera coordinates
        car_corners_lidar = np.array([
            [rel_lead_x - 2.2, -0.9, -1.7],
            [rel_lead_x - 2.2,  0.9, -1.7],
            [rel_lead_x - 2.2,  0.9, -0.25],
            [rel_lead_x - 2.2, -0.9, -0.25],
        ])
        corners_cam = (R @ car_corners_lidar.T).T + T
        corners_homo = np.hstack([corners_cam, np.ones((4, 1))])
        proj = (P_rect @ corners_homo.T).T
        pts_2d = proj[:, :2] / proj[:, 2:3]
        
        p0, p1, p2, p3 = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]
        car_poly = [(int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1]))]
        draw.polygon(car_poly, fill=(185, 30, 25)) # Red car body
        
        # Rear window
        win_w = abs(int(p1[0] - p0[0])) * 0.75
        win_h = abs(int(p3[1] - p0[1])) * 0.35
        c_x = (p0[0] + p1[0]) / 2.0
        c_y = p2[1] + (p1[1] - p2[1]) * 0.35
        draw.rectangle([c_x - win_w/2, c_y - win_h/2, c_x + win_w/2, c_y + win_h/2], fill=(40, 50, 60))
        
        # License plate
        plate_w = abs(int(p1[0] - p0[0])) * 0.35
        plate_h = abs(int(p3[1] - p0[1])) * 0.18
        plate_y = p0[1] - (p0[1] - p3[1]) * 0.32
        draw.rectangle([c_x - plate_w/2, plate_y - plate_h/2, c_x + plate_w/2, plate_y + plate_h/2], fill=(245, 245, 180), outline=(20, 20, 20))
        
        # Taillights
        draw.rectangle([p0[0] + 5, plate_y - plate_h/2, p0[0] + 5 + plate_w*0.5, plate_y + plate_h/2], fill=(255, 40, 40))
        draw.rectangle([p1[0] - 5 - plate_w*0.5, plate_y - plate_h/2, p1[0] - 5, plate_y + plate_h/2], fill=(255, 40, 40))

    # Oncoming Car (in opposite lane)
    rel_onc_x = oncoming_x - ego_x
    if 4.0 < rel_onc_x < 55.0:
        t = 2.0 / rel_onc_x
        onc_w = int(140 * t)
        onc_h = int(105 * t)
        onc_x = int(vp_x - 380 * t)
        onc_y = int(horizon_y + 260 * t)
        draw.rectangle([onc_x - onc_w//2, onc_y - onc_h, onc_x + onc_w//2, onc_y], fill=(45, 80, 160)) # Blue oncoming car
        # Headlights (if car is large enough)
        if onc_h >= 8 and onc_w >= 10:
            hl_w = max(2, int(15 * t))
            hy0 = onc_y - max(4, onc_h // 3)
            hy1 = onc_y - 2
            if hy0 < hy1:
                draw.rectangle([onc_x - onc_w//2 + 2, hy0, onc_x - onc_w//2 + 2 + hl_w, hy1], fill=(255, 255, 200))
                draw.rectangle([onc_x + onc_w//2 - 2 - hl_w, hy0, onc_x + onc_w//2 - 2, hy1], fill=(255, 255, 200))

    # Grayscale image_01
    gray_img = img.convert("L")
    return img, gray_img

def generate_telemetry(frame_idx, ego_x, base_time=1629037800.0):
    dt = 0.1
    t = base_time + frame_idx * dt
    speed_kmh = 25.2 + 0.3 * math.sin(frame_idx * 0.15)
    lat = 54.767820 + (ego_x * 0.0000089)
    lon = -1.574410 + (ego_x * 0.0000142)
    alt = 104.5 + 0.05 * math.sin(frame_idx * 0.1)
    heading = 62.4 + 0.05 * frame_idx
    
    gps_row = f"timestamp,latitude,longitude,altitude,speed_kmh,heading\n{t:.3f},{lat:.8f},{lon:.8f},{alt:.2f},{speed_kmh:.2f},{heading:.2f}\n"
    
    ax = 0.01 + 0.02 * math.sin(frame_idx * 0.3)
    ay = 0.00 + 0.01 * math.cos(frame_idx * 0.3)
    az = 9.81 + 0.03 * math.sin(frame_idx * 0.8)
    wx = 0.001 * math.sin(frame_idx * 0.2)
    wy = 0.002 * math.cos(frame_idx * 0.2)
    wz = 0.001 * math.sin(frame_idx * 0.1)
    imu_row = f"timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n{t:.3f},{ax:.4f},{ay:.4f},{az:.4f},{wx:.5f},{wy:.5f},{wz:.5f}\n"
    
    lux_val = 12450.0 + 50.0 * math.sin(frame_idx * 0.2)
    lux_row = f"timestamp,lux\n{t:.3f},{lux_val:.1f}\n"
    
    return gps_row, imu_row, lux_row

def main(num_frames=50):
    print(f"Generating {num_frames} high-fidelity DurLAR frames in {DATA_DIR}...")
    
    pts_dir = os.path.join(DATA_DIR, "ouster_points", "data")
    amb_dir = os.path.join(DATA_DIR, "ambient", "data")
    ref_dir = os.path.join(DATA_DIR, "reflec", "data")
    cam2_dir = os.path.join(DATA_DIR, "image_02", "data")
    cam1_dir = os.path.join(DATA_DIR, "image_01", "data")
    gps_dir = os.path.join(DATA_DIR, "gps", "data")
    imu_dir = os.path.join(DATA_DIR, "imu", "data")
    lux_dir = os.path.join(DATA_DIR, "lux", "data")
    
    for d in [pts_dir, amb_dir, ref_dir, cam2_dir, cam1_dir, gps_dir, imu_dir, lux_dir]:
        os.makedirs(d, exist_ok=True)
        
    R, T, P_rect = create_calibration(DATA_DIR)
    
    for f_idx in range(num_frames):
        fname = f"{f_idx:06d}"
        
        # 1. 3D Points & Panoramas
        points, pano_amb, pano_ref, ego_x, lead_x, oncoming_x = generate_scene_points(
            f_idx, num_rings=128, azimuth_steps=1024
        )
        
        # Save points
        bin_path = os.path.join(pts_dir, f"{fname}.bin")
        points.tofile(bin_path)
        
        # Save panoramas
        amb_img = Image.fromarray((pano_amb * 255.0).astype(np.uint8))
        amb_img.save(os.path.join(amb_dir, f"{fname}.png"))
        
        ref_img = Image.fromarray((pano_ref * 255.0).astype(np.uint8))
        ref_img.save(os.path.join(ref_dir, f"{fname}.png"))
        
        # 2. Camera views
        rgb_img, gray_img = render_camera_images(f_idx, ego_x, lead_x, oncoming_x, R, T, P_rect)
        rgb_img.save(os.path.join(cam2_dir, f"{fname}.png"))
        gray_img.save(os.path.join(cam1_dir, f"{fname}.png"))
        
        # 3. Telemetry
        gps_txt, imu_txt, lux_txt = generate_telemetry(f_idx, ego_x)
        with open(os.path.join(gps_dir, f"{fname}.csv"), "w") as f:
            f.write(gps_txt)
        with open(os.path.join(imu_dir, f"{fname}.csv"), "w") as f:
            f.write(imu_txt)
        with open(os.path.join(lux_dir, f"{fname}.csv"), "w") as f:
            f.write(lux_txt)
            
        if f_idx % 10 == 0 or f_idx == num_frames - 1:
            print(f" Frame {fname}: {points.shape[0]} points generated, saved successfully.")

    print(f"Realistic dataset generation complete ({num_frames} frames)! Stored at: {DATA_DIR}")

if __name__ == "__main__":
    main(50)
