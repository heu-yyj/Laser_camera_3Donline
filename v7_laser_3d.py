# x相较于v6版本，该版本增加了双向缓存主动配对逻辑，提升了数据同步的鲁棒性
# -*- coding: utf-8 -*- 

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
import sys
import atexit

# ========================= 配置区 =========================
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
NOKOV_SERVER_IP = "10.104.21.38"

# 添加需要接收点云数据的IP及端口
ZMQ_SERVERS = [
    {"ip": "10.104.21.145", "port": 5557},  
]

# 相机内参矩阵（像素单位）
fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345
f_mm = fx * PIXEL_SIZE
BASELINE_S = 220.0
A_RAD = np.deg2rad(19.6)

# AUV marker灯 长815mm 高80 另一个高40-45  2.8 3.2
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])
R_y = R.from_euler('y', angles_rad[1])
R_x = R.from_euler('x', angles_rad[2])
R_auv2cam = (R_z * R_y * R_x)
T_cam_in_auv = np.array([389.0, 39.8, 405.0])

# --- 修改配置 ---
# 因为两者频率不同，需要缓存足够长的时间窗口以容纳双方数据
POSE_CACHE_SEC = 1.0  # 缓存位姿1秒，足以包含多帧激光
LASER_CACHE_SEC = 1.0 # 缓存激光1秒，足以包含多帧位姿
SYNC_THRESHOLD_MS = 10 # 同步阈值，例如 10ms

# 全局状态
running = True
frame_idx = 0
pose_lock = threading.Lock()
laser_lock = threading.Lock()

# --- 修改：使用两个独立的缓存 ---
pose_cache = deque()  # [(timestamp_ms, pos_mm, quat), ...]
laser_cache = deque() # [(timestamp_ns, laser_points_list), ...]

first_image_received = False
first_pose_received = False
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# --- 修改 ZeroMQ 初始化 ---
context = zmq.Context()
zmq_sockets = []

for server in ZMQ_SERVERS:
    socket = context.socket(zmq.PUSH)
    socket.set_hwm(0)
    socket.set(zmq.CONFLATE, 1)
    connect_addr = f"tcp://{server['ip']}:{server['port']}"
    socket.connect(connect_addr)
    zmq_sockets.append(socket)
    print(f"[ZMQ] 已连接到服务端 {connect_addr}")

# ===================== 保存原始图像点数据 =====================
def save_raw_image_points(points, timestamp_ns):
    """保存原始图像坐标点到 PLY 文件"""
    global current_ply_path, ply_file, total_points

    if not points:
        return

    if not current_ply_path:
        current_ply_path = f"raw_image_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        ply_file = open(current_ply_path, "wb")
        # 写入 PLY 头部信息
        header = f"""ply
format binary_little_endian 1.0
element vertex {1000000}  # 预估顶点数
property float x
property float y
property float z
end_header
"""
        ply_file.write(header.encode('utf-8'))
        print(f"[PLY] 创建原始图像点 PLY 文件: {current_ply_path}")

    # 将图像坐标点转换为世界坐标点，这里我们暂时只保存图像坐标点，z=0
    # 注意：PLY 通常期望 float32
    image_pts_mm = np.array([(p["x"], p["y"], 0.0) for p in points], dtype=np.float32)
    ply_file.write(image_pts_mm.tobytes())
    total_points += len(image_pts_mm)
    print(f"[PLY] 已写入 {len(image_pts_mm)} 个原始图像点 (共 {total_points})")

# ===================== 保存世界坐标点云到 PLY 文件 =====================
def append_points_to_ply(points):
    """追加点云到 PLY 文件"""
    global current_ply_path, ply_file, total_points

    if not points.any():
        return

    if not current_ply_path:
        current_ply_path = f"world_point_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        ply_file = open(current_ply_path, "wb")
        # 写入 PLY 头部信息
        header = f"""ply
format binary_little_endian 1.0
element vertex {1000000}  # 预估顶点数
property float x
property float y
property float z
end_header
"""
        ply_file.write(header.encode('utf-8'))
        print(f"[PLY] 创建世界坐标点云 PLY 文件: {current_ply_path}")

    # 将点云数据转换为 float32 并写入
    points_float32 = points.astype(np.float32)
    ply_file.write(points_float32.tobytes())
    total_points += len(points_float32)
    print(f"[PLY] 已追加 {len(points_float32)} 个世界坐标点 (共 {total_points})")

def create_new_ply_file():
    """创建新的 PLY 文件 (如果需要)"""
    global current_ply_path, ply_file, total_points
    # 此函数用于初始化，实际的创建和写入在 append_points_to_ply 中完成
    pass

# ===================== 激光点处理相关函数 =====================
def compute_depths(xs):
    """根据图像坐标 x 计算深度"""
    # Convert pixel coordinate to angle
    angles = A_RAD - np.arctan(xs / f_mm)
    # Compute depth from angle and baseline
    depths = BASELINE_S / (np.cos(angles) + 1e-12)  # Add small epsilon to prevent division by zero
    return depths

def image_to_camera(image_points, depths):
    """将图像坐标和深度转换为相机坐标系下的点"""
    # Extract x and y from the list of dictionaries
    xs = np.array([p["x"] for p in image_points], dtype=np.float64)
    ys = np.array([p["y"] for p in image_points], dtype=np.float64)
    
    # Use the precomputed K matrix
    inv_K = np.linalg.inv(K)
    
    # Create homogeneous image points (u, v, 1)
    ones = np.ones_like(xs)
    image_coords_homo = np.vstack([xs, ys, ones]).T  # Shape: (N, 3)

    # Apply inverse intrinsic matrix to get normalized coordinates
    cam_coords_norm = (inv_K @ image_coords_homo.T).T  # Shape: (N, 3)

    # Multiply by depth to get 3D points in camera frame
    cam_points = cam_coords_norm * depths[:, np.newaxis]  # Broadcasting depth (N,) with (N, 3)

    return cam_points

def camera_to_world_with_distance(cam_points, auv_pos_mm, auv_quat):
    """将相机坐标系下的点转换到世界坐标系，并计算距离"""
    # 1. Transform from camera frame to AUV body frame
    r_cam_in_auv = R_auv2cam.apply(cam_points)
    pts_in_auv_body = r_cam_in_auv + T_cam_in_auv

    # 2. Transform from AUV body frame to world frame using AUV's pose
    R_w_in_auv = R.from_quat(auv_quat[[1, 2, 3, 0]]) # Convert from wxyz to xyzw for scipy
    world_points = R_w_in_auv.apply(pts_in_auv_body) + auv_pos_mm

    # 3. Calculate distances from the AUV origin to each point
    auv_origin_world = auv_pos_mm
    vectors = world_points - auv_origin_world
    distances = np.linalg.norm(vectors, axis=1)

    return world_points, distances

def build_pose_matrix(pos, quat):
    """构建位姿矩阵 (4x4)"""
    R_w_in_auv = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix() # Convert wxyz to xyzw
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R_w_in_auv
    pose_matrix[:3, 3] = pos
    return pose_matrix.tolist()


# ===================== 处理匹配帧的函数 =====================
def process_matched_frame(laser_ts_ns, laser_points, pose_ts_ms, auv_pos_mm, auv_quat, sync_delay_ms):
    """处理一对匹配好的激光和位姿数据"""
    global frame_idx
    
    # (修改) 在处理 Nokov 位姿之前，先保存原始图像点数据
    save_raw_image_points(laser_points, laser_ts_ns)

    xs = np.array([p["x"] for p in laser_points], dtype=np.float64)
    depths = compute_depths(xs)
    cam_pts_mm = image_to_camera(laser_points, depths)
    world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat)

    # 构建 ZMQ 消息
    pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat)
    points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)]
    
    payload = {
        "header": {
            "timestamp": time.time(),
            "frame_id": "lidar",
            "frame_idx": frame_idx,
            "pose": pose_matrix
        },
        "points": points_flat_mm
    }

    # ZMQ 发送逻辑 (保持不变)
    send_results = []
    for i, socket in enumerate(zmq_sockets):
        try:
            socket.send_json(payload, flags=zmq.NOBLOCK)
            send_results.append(f"成功发送到服务端 {i+1}")
        except zmq.Again:
            send_results.append(f"警告: 发送到服务端 {i+1} 失败 (可能队列满)")
        except Exception as e:
            send_results.append(f"错误: 发送到服务端 {i+1} 异常: {e}")
    
    success_count = sum(1 for res in send_results if res.startswith("成功"))
    total_count = len(zmq_sockets)
    print(f"[Published] Frame {frame_idx:05d} | {len(laser_points)} pts | "
          f"sync_delay: {sync_delay_ms:+.1f}ms | "
          f"发送结果: {success_count}/{total_count} 个服务端成功")
    
    append_points_to_ply(world_pts_mm)
    frame_idx += 1


# ===================== 配对逻辑 (核心) =====================
def match_and_process_data(new_data_type, new_ts, new_data_payload):
    """
    通用配对函数
    :param new_data_type: 'pose' or 'laser'
    :param new_ts: 时间戳 (ms for pose, ns for laser)
    :param new_data_payload: 对应数据 (pos, quat for pose; points for laser)
    """
    if new_data_type == 'pose':
        # New pose arrived, look for laser data
        pose_ts_ms, auv_pos_mm, auv_quat = new_data_payload
        target_cache = laser_cache
        target_lock = laser_lock
        target_time_converter = lambda x: x # Laser ts is already in ns
        source_time_converter = lambda x: x * 1_000_000 # Convert pose ts (ms) to ns
        data_processor = lambda l_ts, l_pts, p_ts, p_pos, p_q, delay: \
                         process_matched_frame(l_ts, l_pts, p_ts, p_pos, p_q, delay)
        
    elif new_data_type == 'laser':
        # New laser arrived, look for pose data
        laser_ts_ns, laser_points = new_data_payload
        target_cache = pose_cache
        target_lock = pose_lock
        target_time_converter = lambda x: x * 1_000_000 # Convert pose ts (ms) to ns
        source_time_converter = lambda x: x # Laser ts is already in ns
        data_processor = lambda l_ts, l_pts, p_ts, p_pos, p_q, delay: \
                         process_matched_frame(l_ts, l_pts, p_ts, p_pos, p_q, delay)
    else:
        return

    # Acquire lock for target cache (e.g., laser_cache if we have new pose)
    with target_lock:
        # Iterate through target cache to find matches
        matched_items = []
        for item_ts, item_data in target_cache:
            # Convert both timestamps to the same unit (nanoseconds) for comparison
            src_ts_ns = source_time_converter(new_ts)
            tgt_ts_ns = target_time_converter(item_ts)
            
            dt_ns = abs(src_ts_ns - tgt_ts_ns)
            dt_ms = dt_ns / 1_000_000

            if dt_ms <= SYNC_THRESHOLD_MS:
                matched_items.append((item_ts, item_data, dt_ms))

        # Remove matched items from target cache
        for ts, data, _ in matched_items:
            try:
                target_cache.remove((ts, data))
            except ValueError:
                pass # Might have been removed by another thread concurrently

    # Process matched pairs outside the lock to avoid blocking
    for matched_item_ts, matched_item_data, sync_delay_ms in matched_items:
        if new_data_type == 'pose':
            # We have new pose, matched with laser
            data_processor(matched_item_ts, matched_item_data, new_ts, *new_data_payload, sync_delay_ms)
        else: # new_data_type == 'laser'
            # We have new laser, matched with pose
            data_processor(new_ts, new_data_payload[1], matched_item_ts, *matched_item_data, sync_delay_ms)


# ===================== Nokov 线程 =====================
def nokov_thread():
    global client, first_pose_received
    client = PySDKClient()
    ret = client.Initialize(bytes(NOKOV_SERVER_IP, encoding="utf-8"))
    if ret != 0:
        print(f"[NOKOV] 连接失败: {ret}")
        return
    print("[NOKOV] 已连接，等待第一帧位姿...")
    while running:
        frame = client.PyGetLastFrameOfMocapData()
        if frame:
            try:
                data = frame.contents
                ts_ms = data.iTimeStamp

                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990: continue
                    
                    pos_mm = np.array([rb.x, rb.y, rb.z])
                    quat = np.array([rb.qx, rb.qy, rb.qz, rb.qw])
                    quat = quat / np.linalg.norm(quat)

                    # --- 核心改动：将新位姿加入缓存并尝试配对 ---
                    with pose_lock:
                        pose_cache.append((ts_ms, pos_mm.copy(), quat.copy()))
                        # 清理过期的位姿数据
                        cutoff_ts_ms = ts_ms - int(POSE_CACHE_SEC * 1e3)
                        while pose_cache and pose_cache[0][0] < cutoff_ts_ms:
                            pose_cache.popleft()
                    
                    if not first_pose_received:
                        first_pose_received = True
                        print(f"[NOKOV] 收到第一帧位姿，时间戳: {ts_ms} ms")

                    # 调用配对函数，处理新来的位姿
                    match_and_process_data('pose', ts_ms, (pos_mm, quat))

            finally:
                client.PyNokovFreeFrame(frame)
        else:
            time.sleep(0.001)

# ===================== UDP 主线程 =====================
def udp_thread():
    global first_image_received
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}（非阻塞 + 16MB缓冲）")

    while running:
        latest_msg = None
        while running:
            try:
                data, _ = sock.recvfrom(65535)
                latest_msg = json.loads(data.decode('utf-8'))
            except BlockingIOError:
                break
            except Exception as e:
                print("UDP解析异常:", e)

        if not latest_msg:
            time.sleep(0.001)
            continue

        points = latest_msg.get("laser_points", [])
        if not points:
            continue

        ts_ns = latest_msg["timestamp"]
        if not first_image_received:
            first_image_received = True
            print(f"[Laser] 收到第一帧激光点，时间戳: {ts_ns} ns")

        # (修改) 在处理 Nokov 位姿之前，先保存原始图像点数据
        save_raw_image_points(points, ts_ns)

        # --- 核心改动：将新激光数据加入缓存并尝试配对 ---
        with laser_lock:
            laser_cache.append((ts_ns, points))
            # 清理过期的激光数据
            cutoff_ts_ns = ts_ns - int(LASER_CACHE_SEC * 1e9)
            while laser_cache and laser_cache[0][0] < cutoff_ts_ns:
                laser_cache.popleft()

        # 调用配对函数，处理新来的激光
        match_and_process_data('laser', ts_ns, points)

    print("[UDP Thread] 退出")


# ===================== 主程序 (保持不变) =====================
if __name__ == "__main__":
    create_new_ply_file()
    
    threading.Thread(target=nokov_thread, daemon=True).start()
    threading.Thread(target=udp_thread, daemon=True).start()  
    print("\n=== 激光结构光实时融合系统 + ZMQ发布（双向缓存主动配对）已启动 ===")
    print(f"ZMQ 数据将发送到以下服务端:")
    for i, server in enumerate(ZMQ_SERVERS):
        print(f"  - 服务端 {i+1}: tcp://{server['ip']}:{server['port']}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
        running = False
        time.sleep(2) 
        for socket in zmq_sockets:
            socket.close()
        context.term()
        
        # Close PLY files on exit
        if ply_file:
            ply_file.close()
            print(f"[PLY] 已关闭文件: {current_ply_path}")
        print("资源已释放，程序退出。")
        sys.exit(0)