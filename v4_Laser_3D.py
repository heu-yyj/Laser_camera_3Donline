# -*- coding: utf-8 -*-
"""
激光 + Nokov + ZeroMQ 实时数据发布
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

# 相机相对于惯导中心的平移（mm）
T_cam2ins = np.array([424.0, 27.4, 247.6])   

# Marker灯/刚体 相对于惯导中心的平移（mm）
# 比如灯在惯导/INS前方3-4.5cm,取35mm，左边12.4mm，上方12.4mm+100+45mm
T_marker_to_ins = np.array([35, -12.4, -157.4])

# 最终外参：相机 → 动捕刚体原点（Marker灯）
T_cam2marker = T_cam2ins - T_marker_to_ins  
# 旋转部分通常不变（因为你刚体坐标轴和AUV一致） XYZ为AUV前右下
R_cam2marker =  R.from_euler('zyx', [np.deg2rad(199.6), 0, np.deg2rad(90)]).as_matrix() 
R_cam2auv = R_cam2marker
T_cam2auv = T_cam2marker # 单位：mm

POSE_CACHE_SEC    = 5.0
SYNC_THRESHOLD_MS = 100

# 单位转换常量
MM_TO_M = 0.001
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

def append_points_to_ply(world_pts_mm):
    global total_points
    # 转换为米并保留4位小数
    world_pts_m = world_pts_mm * MM_TO_M
    lines = [f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in world_pts_m]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts_mm) # 计数基于原始点数

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

def camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat):
    # 注意：这里的计算仍然使用毫米单位
    auv_pts_mm = (R_cam2auv @ cam_pts_mm.T).T + T_cam2auv
    r = R.from_quat(auv_quat[[1,2,3,0]])
    R_w = r.as_matrix()
    # auv_pos_mm 是毫米单位，auv_pts_mm 也是毫米单位，结果 world_pts_mm 也是毫米
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm 
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    
    return world_pts_mm, distances # 返回毫米单位的点云

def build_pose_matrix(pos_mm, quat):
    # 构建位姿矩阵时，也应使用米单位
    pos_m = pos_mm * MM_TO_M
    r = R.from_quat(quat[[1,2,3,0]])
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos_m # 使用米单位的位置
    return T.reshape(-1).tolist()

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
                ts_us = data.iTimeStamp
                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990: continue
                    # Nokov SDK 返回的 pos 是毫米单位
                    pos_mm = np.array([rb.x, rb.y, rb.z]) 
                    quat = np.array([rb.qw, rb.qx, rb.qy, rb.qz])
                    with pose_lock:
                        pose_cache.append((ts_us, pos_mm.copy(), quat.copy()))
                        cutoff = ts_us - int(POSE_CACHE_SEC * 1e6)
                        while pose_cache and pose_cache[0][0] < cutoff:
                            pose_cache.popleft()
                    if not first_pose_received:
                        first_pose_received = True
                        print(f"[NOKOV] 收到第一帧位姿，时间戳: {ts_us} μs")
            finally:
                client.PyNokovFreeFrame(frame)
        else:
            time.sleep(0.001)

# ===================== 时间戳匹配 ======================
def get_nearest_pose(ts_ns: int):
    ts_us = ts_ns // 1_000_000                    # ：纳秒 → 微秒 
    with pose_lock:
        if not pose_cache:
            return None, None
        ts_arr = np.array([t for t, _, _ in pose_cache])
        idx = np.argmin(np.abs(ts_arr - ts_us))
        dt_us = abs(ts_arr[idx] - ts_us)
        dt_ms = dt_us / 1000.0

        if dt_ms > SYNC_THRESHOLD_MS:
            print(f"[Warning] 时间戳偏差 {dt_ms:.1f}ms，仍使用最近位姿")

        _, pos_mm, quat = pose_cache[idx] # pos_mm 是毫米单位
        return (pos_mm.copy(), quat.copy(), dt_ms), dt_ms

# ===================== UDP 主线程 =====================
def udp_thread():
    global frame_idx, first_image_received
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

        if not first_pose_received:
            print("等待 Nokov 位姿就绪...")
            continue

        pose_info, dt_ms = get_nearest_pose(ts_ns)
        if pose_info is None:
            continue

        # auv_pos_mm 是毫米单位
        auv_pos_mm, auv_quat, sync_delay_ms = pose_info 

        xs = np.array([p["x"] for p in points], dtype=np.float64)
        depths = compute_depths(xs)
        cam_pts_mm = image_to_camera(points, depths)
        # world_pts_mm 是毫米单位的点云
        world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat) 

        # === 核心修改：单位转换与ZMQ发布 ===
        # 1. 转换单位：毫米 -> 米
        world_pts_m = world_pts_mm * MM_TO_M
        auv_pos_m = auv_pos_mm * MM_TO_M

        # 2. 构建ZMQ消息 (使用米单位)
        # 为了保留4位小数，我们在序列化时处理
        pose_flat_m = build_pose_matrix(auv_pos_mm, auv_quat) # 内部已转换为米
        # 扁平化点云并保留4位小数
        points_flat_m = [round(coord, 4) for coord in world_pts_m.reshape(-1)]

        payload = {
            "header": {
                "timestamp": time.time(),
                "frame_id": "laser",
                "frame_idx": frame_idx,
                # pose_flat_m 已经是米单位的列表
                "pose": pose_flat_m 
            },
            # points_flat_m 是保留4位小数的米单位坐标列表
            "points": points_flat_m 
        }
        zmq_socket.send_json(payload, flags=zmq.NOBLOCK)
        # ===================================

        # PLY 文件保存 (内部处理了单位转换)
        append_points_to_ply(world_pts_mm) 

        print(f"[Published] Frame {frame_idx:05d} | {len(points)} pts | "
              f"sync_delay: {sync_delay_ms:+.1f}ms | {os.path.basename(current_ply_path)}")

        frame_idx += 1

# ===================== 主程序 =====================
if __name__ == "__main__":
    create_new_ply_file()
    
    # 启动核心线程 (移除了 visualization_thread)
    threading.Thread(target=nokov_thread, daemon=True).start()
    threading.Thread(target=udp_thread, daemon=True).start()
    
    print("\n=== 激光结构光实时融合系统 + ZMQ发布  已启动 ===")
    print("点云数据 (米单位, 4位小数) 通过 ZMQ 发布！\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
        running = False
        # 等待线程结束 (简单等待，实际项目可使用 Event 或 Queue 控制)
        time.sleep(2) 
        zmq_socket.close()
        context.term()
        print("资源已释放，程序退出。")
        sys.exit(0)