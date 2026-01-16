# -*- coding: utf-8 -*-
"""
激光 + Nokov + ZeroMQ 实时数据发布（全系统单位：毫米 mm）
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
import sys
import atexit

# ========================= 配置区 =========================
UDP_IP          = "0.0.0.0"
UDP_PORT        = 8888
NOKOV_SERVER_IP = "10.104.21.38"  # 动捕系统位姿广播IP
ZMQ_SERVER_IP   = "10.104.21.145"  # 接收端 IP
ZMQ_PORT        = 5557

# 相机内参矩阵（像素单位）
fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345              # mm/pixel
f_mm       = fx * PIXEL_SIZE    # 焦距（mm）
BASELINE_S = 220.0              # 激光三角测距基线长度（mm）
A_RAD      = np.deg2rad(19.6)  # 激光发射角（弧度）

#AUV marker灯 长815mm 高80 另一个高40-45  2.8 3.2
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])      
R_y = R.from_euler('y', angles_rad[1])     
R_x = R.from_euler('x', angles_rad[2])      

# intrinsic zyx = R_z * R_y * R_x
R_auv2cam = (R_z * R_y * R_x)
#还是应该使用AUV到相机的矩阵
#R_cam2auv = R_A.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

# T_auv2cam = np.array([424.0, 27.4, 247.6]) - np.array([35, -12.4, -157.4]) # 从AUV到相机到的平移 (mm)
T_cam_in_auv =np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

POSE_CACHE_SEC    = 5.0         # 位姿缓存时间窗口（秒）
SYNC_THRESHOLD_MS = 10         # 时间戳同步阈值（毫秒）

# 全局状态
running = True
frame_idx = 0
pose_lock = threading.Lock()
pose_cache = deque()       # (timestamp_us, pos_mm, quat)
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

# ===================== 原始图像点数据保存函数 ======================
# (新增) 用于保存原始图像点坐标到文件
def save_raw_image_points(laser_points, timestamp_ns):
    """
    保存原始图像点坐标到一个以时间戳命名的文件中。
    每个时间戳对应一个文件。
    文件名格式: image_points_{timestamp_ns}.txt
    文件内容: 每行一个点的 x, y 坐标，以空格分隔。
    """
    # 确定保存目录
     # 生成当前日期的文件夹名 (例如: 20260116)
    date_str = datetime.now().strftime("%Y%m%d")
    
    # 生成当前时间的文件夹名 (例如: LaserRawImagePoints_171230)
    time_str = datetime.now().strftime("%H%M%S")
    subfolder_name = f"LaserRawImagePoints_{time_str}"
    
    # 构造完整的保存目录路径
    base_dir = "D:/工作/" # 基础目录
    raw_data_dir = os.path.join(base_dir, date_str, subfolder_name)
    
    # 创建目录（如果不存在）
    os.makedirs(raw_data_dir, exist_ok=True)

    # 构造文件名
    filename = f"image_points_{timestamp_ns}.txt"
    filepath = os.path.join(raw_data_dir, filename)

    # 写入文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for point in laser_points:
                # 假设 point 是一个字典，包含 'x' 和 'y' 键
                # 根据你提供的代码，格式是正确的
                f.write(f"{point['x']:.6f} {point['y']:.6f}\n") # 保留6位小数，可根据需要调整
        print(f"[RawData] 图像点已保存到: {filepath} (共 {len(laser_points)} 个点)")
    except Exception as e:
        print(f"[RawData] 保存图像点数据失败: {e} (时间戳: {timestamp_ns})")

# ===================== PLY 文件函数 ======================
def create_new_ply_file():
    global current_ply_path, ply_file, total_points
    total_points = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "D:/工作/LaserData"
    os.makedirs(output_dir, exist_ok=True)
    
    current_ply_path = os.path.join(output_dir, f"laser_pointcloud_{timestamp_str}.ply")
    ply_file = open(current_ply_path, "w", encoding='utf-8')
    
    ply_file.write("ply\nformat ascii 1.0\n")
    ply_file.write(f"comment Generated @ {datetime.now().isoformat()}\n")
    ply_file.write("comment Unit: millimeter\n")  # 明确标注单位
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    print(f"[PLY] 保存至：{current_ply_path}（单位：毫米）")

def append_points_to_ply(world_pts_mm):
    global total_points
    # 直接写入毫米单位（不转换）
    lines = [f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in world_pts_mm]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts_mm)

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
            print(f"\n[PLY] 已完成：{current_ply_path}（{total_points}点，单位：毫米）")
        except Exception as e:
            print(f"\n[PLY] 文件已保存（更新顶点数失败: {e})")
atexit.register(close_ply_file)

# ===================== 激光处理函数 =====================
def compute_depths(x_coords): 
    d = (x_coords - cx) * PIXEL_SIZE    # mm
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img, depths):
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm   # mm
    return pts

def camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat):
    auv_pts_mm = (R_auv2cam.as_matrix() @ cam_pts_mm.T).T + T_cam_in_auv  # 转到AUV坐标系下
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm 
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    return world_pts_mm, distances

def build_pose_matrix(pos_mm, quat):
    """返回 4x4 位姿矩阵（嵌套列表），单位：毫米"""
    r = R.from_quat(quat)
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos_mm  # mm
    # 转为 list of lists，保留 float 类型（JSON 可序列化）
    return T.tolist()  # [[...], [...], [...], [...]]

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
                    pos_mm = np.array([rb.x, rb.y, rb.z]) 
                    quat = np.array([rb.qx, rb.qy, rb.qz, rb.qw])
                    quat = quat / np.linalg.norm(quat)  # 归一化
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
    ts_us = ts_ns // 1_000_000
    with pose_lock:
        if not pose_cache:
            return None, None
        ts_arr = np.array([t for t, _, _ in pose_cache])
        idx = np.argmin(np.abs(ts_arr - ts_us))
        dt_us = abs(ts_arr[idx] - ts_us)
        dt_ms = dt_us / 1000.0
        if dt_ms > SYNC_THRESHOLD_MS:
            print(f"[Warning] 时间戳偏差 {dt_ms:.1f}ms，仍使用最近位姿")
        _, pos_mm, quat = pose_cache[idx]
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

        # (修改) 在处理 Nokov 位姿之前，先保存原始图像点数据
        # 这样即使没有位姿，原始数据也会被保存下来
        save_raw_image_points(points, ts_ns)

        if not first_pose_received:
            print("等待 Nokov 位姿就绪...")
            continue

        pose_info, dt_ms = get_nearest_pose(ts_ns)
        if pose_info is None:
            continue

        auv_pos_mm, auv_quat, sync_delay_ms = pose_info 
        
        xs = np.array([p["x"] for p in points], dtype=np.float64)
        depths = compute_depths(xs)
        cam_pts_mm = image_to_camera(points, depths)
        world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat) 


        # ======== 调试打印开始 ========
        print(f"\n[DEBUG Frame {frame_idx:05d}]")
        print(f"  Laser Points Count: {len(points)}")
        if points:
            xs = np.array([p["x"] for p in points], dtype=np.float64)
            print(f"  X coords - min: {xs.min():.2f}, max: {xs.max():.2f}, cx: {cx:.2f}")
        
        print(f"  Computed Depths - min: {depths.min():.2f}, max: {depths.max():.2f}, mean: {depths.mean():.2f}")
        print(f"  Depths - negative count: {(depths < 0).sum()}, zero count: {(depths == 0).sum()}, positive count: {(depths > 0).sum()}")
        
        print(f"  Camera Points - Z coords min: {cam_pts_mm[:, 2].min():.2f}, max: {cam_pts_mm[:, 2].max():.2f}, mean: {cam_pts_mm[:, 2].mean():.2f}")
        print(f"  Camera Points - Z negative count: {(cam_pts_mm[:, 2] < 0).sum()}")

        print(f"  AUV Pose - Position: [{auv_pos_mm[0]:.2f}, {auv_pos_mm[1]:.2f}, {auv_pos_mm[2]:.2f}]")
        # print(f"  AUV Quat: [{auv_quat[0]:.4f}, {auv_quat[1]:.4f}, {auv_quat[2]:.4f}, {auv_quat[3]:.4f}]") # 可选打印

        print(f"  World Points - Z coords min: {world_pts_mm[:, 2].min():.2f}, max: {world_pts_mm[:, 2].max():.2f}, mean: {world_pts_mm[:, 2].mean():.2f}")
        print(f"  World Points - Z relative to AUV (min, max): {world_pts_mm[:, 2].min() - auv_pos_mm[2]:.2f}, {world_pts_mm[:, 2].max() - auv_pos_mm[2]:.2f}")
        # ======== 调试打印结束 ========


        # 构建 ZMQ 消息（全部使用毫米单位）
        pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat)  # 4x4 嵌套列表
        points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)]

        payload = {
            "header": {
                "timestamp": time.time(),
                "frame_id": "lidar",
                "frame_idx": frame_idx,
                "pose": pose_matrix  # 格式: [[...], [...], [...], [...]]
            },
            "points": points_flat_mm  # 毫米单位，扁平列表
        }
        zmq_socket.send_json(payload, flags=zmq.NOBLOCK)

        append_points_to_ply(world_pts_mm) 

        print(f"[Published] Frame {frame_idx:05d} | {len(points)} pts | "
              f"sync_delay: {sync_delay_ms:+.1f}ms | {os.path.basename(current_ply_path)}")

        frame_idx += 1

# ===================== 主程序 =====================
if __name__ == "__main__":
    create_new_ply_file()
    
    threading.Thread(target=nokov_thread, daemon=True).start()
    threading.Thread(target=udp_thread, daemon=True).start()
    
    print("\n=== 激光结构光实时融合系统 + ZMQ发布（全系统单位：毫米）已启动 ===")
    print("点云与位姿均以毫米（mm）为单位通过 ZMQ 发布！\n")
    print("原始图像点数据将以时间戳命名保存到 'D:/工作/LaserRawImagePoints' 目录。")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
        running = False
        time.sleep(2) 
        zmq_socket.close()
        context.term()
        print("资源已释放，程序退出。")
        sys.exit(0)