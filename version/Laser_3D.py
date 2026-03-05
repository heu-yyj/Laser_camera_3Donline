# -*- coding: utf-8 -*-
"""
激光结构光 + Nokov 动捕 + ZeroMQ 实时发布
每次程序启动生成一个独立、以启动时间命名的 PLY 文件
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

# ========================= 配置区 =========================
UDP_IP              = "0.0.0.0"
UDP_PORT            = 8888

NOKOV_SERVER_IP     = "192.168.5.105"

ZMQ_SERVER_IP       = "192.168.5.105"
ZMQ_PORT            = 5557

# 相机内参（请替换为实际值）
# fx, fy = 1500.0, 1500.0
# cx, cy = 960.0, 540.0
# K << 4308.8624, 0, 1379.5081, // fx, cx
#         0, 4302.9958, 1031.0359, // fy, cy
#         0, 0, 1;
fx , fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[4308.8624, 0, 1379.5081],
              [0, 4302.9958, 1031.0359],
              [0,  0,  1]], dtype=np.float64)

# 激光三角参数
PIXEL_SIZE = 0.00345
f_mm       = fx * PIXEL_SIZE
BASELINE_S = 220.0
A_RAD      = np.deg2rad(19.6)

# 相机 → AUV 固定外参
R_cam2auv = R.from_euler('zyx', [np.deg2rad(199.6), 0, np.deg2rad(90)]).as_matrix()
T_cam2auv = np.array([424.0, 27.4, 247.6])

POSE_CACHE_SEC = 3.0
# =========================================================

# 全局变量
pose_lock   = threading.Lock()
pose_cache  = deque()
frame_idx   = 0

# 当前运行的 PLY 文件路径（每次启动都重新生成）
current_ply_path = None
ply_file = None  # 文件句柄，用于快速追加写入

# ZeroMQ
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
print(f"[ZMQ] 已连接到 {ZMQ_SERVER_IP}:{ZMQ_PORT}")

# 创建本次运行的独立 PLY 文件（以启动时间命名）
def create_new_ply_file():
    global current_ply_path, ply_file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_ply_path = f"laser_pointcloud_{timestamp_str}.ply"
    
    ply_file = open(current_ply_path, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write(f"comment Generated at {datetime.now().isoformat()}\n")
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float distance\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    
    print(f"[PLY] 本次运行点云文件：{current_ply_path}")
    print(f"    → 程序关闭后该文件将自动完成，可直接用 CloudCompare 打开")

# 追加点到当前 PLY 文件（高效、线程安全）
ply_lock = threading.Lock()
total_points_in_current_ply = 0

def append_points_to_ply(world_pts, distances):
    global total_points_in_current_ply
    lines = [
        f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {dist:.6f}\n"
        for xyz, dist in zip(world_pts, distances)
    ]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points_in_current_ply += len(world_pts)

# 程序退出时更新 header 中的点数（可选美观，非必须）
import atexit
def close_ply_file():
    global ply_file
    if ply_file and not ply_file.closed:
        ply_file.close()
        # 可选：重写 header 更新点数（让文件更规范）
        try:
            with open(current_ply_path, "r+") as f:
                content = f.read()
                header_end = content.index("end_header\n") + len("end_header\n")
                header = content[:header_end]
                body = content[header_end:]
                new_header = header.replace(
                    "element vertex 0",
                    f"element vertex {total_points_in_current_ply}"
                )
                f.seek(0)
                f.write(new_header + body)
                f.truncate()
            print(f"\n[PLY] 已完成并更新点数：{current_ply_path} ({total_points_in_current_ply} 点)")
        except:
            print(f"\n[PLY] 文件已保存：{current_ply_path}")
atexit.register(close_ply_file)

# ===================== Nokov & 位姿查询 =====================
def nokov_thread():
    global client
    client = PySDKClient()
    ret = client.Initialize(bytes(NOKOV_SERVER_IP, encoding="utf-8"))
    if ret != 0:
        print(f"[NOKOV] 连接失败: {ret}")
        return
    print(f"[NOKOV] 已连接，开始接收位姿...")
    while True:
        frame = client.PyGetLastFrameOfMocapData()
        if frame:
            try:
                data = frame.contents
                ts_us = data.iTimeStamp
                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990: continue
                    pos = np.array([rb.x, rb.y, rb.z])
                    quat = np.array([rb.qw, rb.qx, rb.qy, rb.qz])
                    with pose_lock:
                        pose_cache.append((ts_us, pos.copy(), quat.copy()))
                        cutoff = ts_us - int(POSE_CACHE_SEC * 1e6)
                        while pose_cache and pose_cache[0][0] < cutoff:
                            pose_cache.popleft()
            finally:
                client.PyNokovFreeFrame(frame)
        else:
            time.sleep(0.001)

def get_nearest_pose(ts_ns: int):
    ts_us = ts_ns // 1000
    with pose_lock:
        if not pose_cache: return None
        ts_arr = np.array([t for t, _, _ in pose_cache])
        idx = np.argmin(np.abs(ts_arr - ts_us))
        if abs(ts_arr[idx] - ts_us) > 50_000: return None
        _, pos, quat = pose_cache[idx]
        return pos.copy(), quat.copy(), abs(ts_arr[idx] - ts_us) / 1000.0

# ===================== 激光处理 =====================
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
    return world_pts, distances

def build_pose_matrix(pos, quat):
    r = R.from_quat(quat[[1,2,3,0]])
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos
    return T.reshape(-1).tolist()

# ===================== UDP 主线程 =====================
def udp_thread():
    global frame_idx
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            msg = json.loads(data.decode('utf-8'))
            ts_ns = msg["timestamp"]
            points = msg.get("laser_points", [])
            if not points: continue

            pose_info = get_nearest_pose(ts_ns)
            if not pose_info:
                print(f"[Frame {frame_idx}] 无同步位姿")
                continue

            auv_pos, auv_quat, delay_ms = pose_info

            xs = np.array([p["x"] for p in points], dtype=np.float64)
            depths = compute_depths(xs)
            cam_pts = image_to_camera(points, depths)
            world_pts, distances = camera_to_world_with_distance(cam_pts, auv_pos, auv_quat)

            # 1. ZeroMQ 实时发布
            payload = {
                "header": {
                    "timestamp": time.time(),
                    "frame_id": "laser",
                    "frame_idx": frame_idx,
                    "pose": build_pose_matrix(auv_pos, auv_quat)
                },
                "points": world_pts.reshape(-1).tolist()
            }
            zmq_socket.send_json(payload, flags=zmq.NOBLOCK)

            # 2. 保存到本次运行的独立 PLY 文件
            append_points_to_ply(world_pts, distances)

            print(f"[→] Frame {frame_idx:04d} | {len(points)} pts | delay {delay_ms:.1f}ms | 已保存至 {os.path.basename(current_ply_path)}")

            frame_idx += 1

        except Exception as e:
            print("UDP 处理异常:", e)

# ===================== 主程序 =====================
if __name__ == "__main__":
    create_new_ply_file()  # 每次启动创建一个新文件

    threading.Thread(target=nokov_thread, daemon=True).start()
    time.sleep(1.5)
    threading.Thread(target=udp_thread, daemon=True).start()

    print("\n=== 激光结构光实时融合系统已启动 ===")
    print(f"本次运行点云独立保存为：{current_ply_path}")
    print("每次重启程序都会生成一个新文件，互不影响！\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在安全退出...")