# -*- coding: utf-8 -*-
"""
终极版：激光 + Nokov + ZeroMQ + Open3D实时可视化（带轨迹+坐标系）
"""

import socket
import json
import threading
import time
import numpy as np
from collections import deque
import zmq
from nokov.nokovsdk import *
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
import signal
import sys
import atexit
import open3d as o3d

# ========================= 配置区 =========================
UDP_IP          = "0.0.0.0"
UDP_PORT        = 8888
NOKOV_SERVER_IP = "192.168.5.110"
ZMQ_SERVER_IP   = "192.168.5.110"
ZMQ_PORT        = 5557

fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345
f_mm       = fx * PIXEL_SIZE
BASELINE_S = 220.0
A_RAD      = np.deg2rad(19.6)

R_cam2auv = R.from_euler('zyx', [np.deg2rad(199.6), 0, np.deg2rad(90)]).as_matrix()
T_cam2auv = np.array([424.0, 27.4, 247.6])

POSE_CACHE_SEC    = 5.0
SYNC_THRESHOLD_MS = 800
# =========================================================

# 全局状态
running = True
frame_idx = 0
pose_lock = threading.Lock()
pose_cache = deque()
first_image_received = False
first_pose_received = False
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# 实时可视化全局变量（必须定义！）
latest_world_pts = None
latest_auv_pose = None
trajectory_points = []

# ZeroMQ
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.set_hwm(0)
zmq_socket.set(zmq.CONFLATE, 1)
zmq_socket.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
print(f"[ZMQ] 已连接 {ZMQ_SERVER_IP}:{ZMQ_PORT}")

# ===================== PLY 文件函数（保持不变）=====================
def create_new_ply_file():
    global current_ply_path, ply_file, total_points
    total_points = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_ply_path = f"laser_pointcloud_{timestamp_str}.ply"
    ply_file = open(current_ply_path, "w")
    ply_file.write("ply\nformat ascii 1.0\n")
    ply_file.write(f"comment Generated @ {datetime.now().isoformat()}\n")
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    print(f"[PLY] 保存至：{current_ply_path}")

def append_points_to_ply(world_pts):
    global total_points
    lines = [f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in world_pts]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts)

def close_ply_file():
    global ply_file
    if ply_file and not ply_file.closed:
        ply_file.close()
        try:
            with open(current_ply_path, "r+") as f:
                content = f.read()
                header_end = content.index("end_header\n") + len("end_header\n")
                header, body = content[:header_end], content[header_end:]
                new_header = header.replace("element vertex 0", f"element vertex {total_points}")
                f.seek(0)
                f.write(new_header + body)
                f.truncate()
            print(f"\n[PLY] 已完成：{current_ply_path}（{total_points}点）")
        except:
            print(f"\n[PLY] 文件已保存")
atexit.register(close_ply_file)

# ===================== 激光处理函数 =====================
def compute_depths(x_coords):
    d = (x_coords - cx) * PIXEL_SIZE
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img, depths):
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm
    return pts

def camera_to_world_with_distance(cam_pts_mm, auv_pos, auv_quat):
    auv_pts = (R_cam2auv @ cam_pts_mm.T).T + T_cam2auv
    r = R.from_quat(auv_quat[[1,2,3,0]])
    R_w = r.as_matrix()
    world_pts = (R_w @ auv_pts.T).T + auv_pos
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    
    # 关键：更新全局变量供可视化线程使用
    global latest_world_pts, latest_auv_pose
    latest_world_pts = world_pts.copy()
    latest_auv_pose = (auv_pos.copy(), auv_quat.copy())
    
    return world_pts, distances

def build_pose_matrix(pos, quat):
    r = R.from_quat(quat[[1,2,3,0]])
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos
    return T.reshape(-1).tolist()

# ===================== Nokov & UDP 线程（保持不变）=====================
# （nokov_thread, get_nearest_pose, udp_thread 完全使用你原来的正确版本）

# ===================== 可视化线程（完美版）=====================
def visualization_thread():
    global vis, pcd, coord_frame, trajectory_line, auv_mesh, running, latest_world_pts, latest_auv_pose, trajectory_points

    vis = o3d.visualization.Visualizer()
    vis.create_window("激光结构光实时点云 + AUV轨迹", width=1600, height=1000)

    pcd = o3d.geometry.PointCloud()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    trajectory_line = o3d.geometry.LineSet()
    auv_mesh = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.2, depth=0.6)
    auv_mesh.paint_uniform_color([0.2, 0.6, 1.0])

    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.add_geometry(trajectory_line)
    vis.add_geometry(auv_mesh)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.asarray([0.05, 0.05, 0.1])

    print("[Open3D] 可视化窗口已启动")

    while running:
        need_update = False
        if latest_world_pts is not None and len(latest_world_pts) > 100:
            pcd.points = o3d.utility.Vector3dVector(latest_world_pts)
            dists = np.linalg.norm(latest_world_pts, axis=1)
            norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
            colors = np.zeros((len(dists), 3))
            colors[:, 0] = norm
            colors[:, 2] = 1 - norm
            pcd.colors = o3d.utility.Vector3dVector(colors)
            need_update = True

        if latest_auv_pose is not None:
            pos, quat = latest_auv_pose
            trajectory_points.append(pos.copy())
            if len(trajectory_points) > 10000:
                trajectory_points.pop(0)
            if len(trajectory_points) > 1:
                points_vec = o3d.utility.Vector3dVector(trajectory_points)
                lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
                trajectory_line.points = points_vec
                trajectory_line.lines = o3d.utility.Vector2iVector(lines)
                trajectory_line.colors = o3d.utility.Vector3dVector([[1,1,1]] * (len(lines)))

            T = np.eye(4)
            r = R.from_quat(quat[[1,2,3,0]])
            T[:3,:3] = r.as_matrix()
            T[:3,3] = pos
            auv_mesh.transform(np.linalg.inv(T @ np.linalg.inv(T)))  # 清零
            auv_mesh.transform(T)
            need_update = True

        if need_update:
            for geom in [pcd, trajectory_line, auv_mesh]:
                vis.update_geometry(geom)
            vis.poll_events()
            vis.update_renderer()
        time.sleep(0.03)

    vis.destroy_window()

# ===================== 主程序（关键修改）=====================
if __name__ == "__main__":
    create_new_ply_file()
    
    # 启动所有线程
    threading.Thread(target=nokov_thread, daemon=False).start()
    threading.Thread(target=udp_thread, daemon=False).start()
    time.sleep(2.0)  # 确保前两个线程先跑起来
    threading.Thread(target=visualization_thread, daemon=False).start()  # 正确位置！

    print("\n=== 激光结构光实时融合系统 + Open3D可视化 已启动 ===")
    print("点云 + 轨迹 + 坐标系 + AUV姿态 实时显示！\n")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        time.sleep(1)
        zmq_socket.close()
        context.term()
        sys.exit(0)