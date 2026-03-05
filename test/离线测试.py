# -*- coding: utf-8 -*-
"""
激光图像点文件 + CSV位姿 + ZeroMQ 实时数据发布模拟（全系统单位：毫米 mm）
"""

import socket  # 为了兼容性保留，但本部分不再使用
import json
import threading
import time
import numpy as np
from collections import deque
import zmq
# from nokov.nokovsdk import * # 不再需要 Nokov SDK
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
import sys
import atexit
import glob
import pandas as pd # 需要安装: pip install pandas

# ========================= 配置区 =========================
# 输入路径配置
LASER_POINTS_DIR = "D:/工作/LaserImagePoints"  # 包含 image_points_*.txt 文件的目录
CSV_POSE_PATH    = "D:/工作/poses.csv"        # 包含位姿信息的 CSV 文件路径

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

angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])
R_y = R.from_euler('y', angles_rad[1])
R_x = R.from_euler('x', angles_rad[2])

# intrinsic zyx = R_z * R_y * R_x
R_A = (R_z * R_y * R_x)
R_cam2auv = R_A.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

T_cam_in_auv =np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

# 时间戳同步配置
SYNC_THRESHOLD_MS = 100         # 时间戳同步阈值（毫秒）。对于文件回放，此值可能需要调整或忽略。

# 全局状态
running = True
frame_idx = 0
pose_df = None # 存储加载的位姿数据
sorted_files = [] # 存储排序后的文件列表
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# ZeroMQ
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.set_hwm(0)
zmq_socket.set(zmq.CONFLATE, 1) # 注意：CONFLATE 在 PUSH/PULL 模式下可能不如 PUB/SUB 模式有效
zmq_socket.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
print(f"[ZMQ] 已连接 {ZMQ_SERVER_IP}:{ZMQ_PORT}")

# ===================== PLY 文件函数 ======================
def create_new_ply_file():
    global current_ply_path, ply_file, total_points
    total_points = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "D:/工作/LaserData"
    os.makedirs(output_dir, exist_ok=True)
    
    current_ply_path = os.path.join(output_dir, f"laser_pointcloud_from_file_{timestamp_str}.ply")
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
    # points_img 是一个包含 {'x': u, 'y': v} 字典的列表
    # 或者是一个形状为 (N, 2) 的 numpy 数组，其中每一行是 (u, v)
    K_inv = np.linalg.inv(K)
    # 将 (u, v) 坐标扩展为齐次坐标 (u, v, 1)
    if isinstance(points_img, list):
        uv_array = np.array([(pt['x'], pt['y']) for pt in points_img], dtype=np.float64)
    else:
        uv_array = points_img.astype(np.float64)
    
    ones_col = np.ones((uv_array.shape[0], 1))
    uv1_array = np.hstack((uv_array, ones_col)) # Shape: (N, 3)
    
    norms = (K_inv @ uv1_array.T).T # Shape: (N, 3)
    cam_pts_mm = depths[:, np.newaxis] * norms # Broadcasting: (N, 1) * (N, 3) -> (N, 3)
    return cam_pts_mm

def camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat):
    # 将相机坐标系下的点转换到 AUV 坐标系下
    auv_pts_mm = (R_cam2auv @ cam_pts_mm.T).T + T_cam_in_auv  # Shape: (N, 3)
    # 将 AUV 坐标系下的点转换到世界坐标系下
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm  # Shape: (N, 3)
    # 计算相机坐标系下的距离
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

# ===================== 位姿查找函数 =====================
def load_poses_from_csv(csv_path):
    """从CSV文件加载位姿数据"""
    print(f"[CSV] 正在加载位姿数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep='\t') # 假设是制表符分隔
        # 确保时间戳列是整数类型
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').astype('Int64')
        # 移除时间戳无效的行
        df = df.dropna(subset=['Timestamp'])
        # 按时间戳排序，确保查找效率
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        print(f"[CSV] 成功加载 {len(df)} 行位姿数据，时间戳范围: {df['Timestamp'].iloc[0]} - {df['Timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"[CSV] 加载失败: {e}")
        return None

def find_nearest_pose(timestamp_ns):
    """
    根据给定的时间戳(ns)查找最接近的位姿。
    CSV中的时间戳是ms，需要转换。
    """
    if pose_df is None or pose_df.empty:
        print("[ERROR] 位姿数据未加载或为空！")
        return None, None
    
    target_ts_ms = timestamp_ns // 1_000_000 # 转换ns到ms

    # 使用 pandas 进行查找
    # 找到第一个大于等于目标时间戳的索引
    idx_ge = pose_df['Timestamp'].searchsorted(target_ts_ms, side='left')
    
    candidates = []
    # 检查前一个索引
    if idx_ge > 0:
        candidates.append(idx_ge - 1)
    # 检查当前索引
    if idx_ge < len(pose_df):
        candidates.append(idx_ge)

    if not candidates:
        print(f"[WARNING] 无法找到时间戳 {target_ts_ms} ms 对应的位姿候选。")
        return None, None

    # 计算所有候选者的差值绝对值
    candidate_timestamps = pose_df.iloc[candidates]['Timestamp'].values
    time_diffs = np.abs(candidate_timestamps - target_ts_ms)
    closest_idx_offset = np.argmin(time_diffs)
    best_candidate_row_index = candidates[closest_idx_offset]
    
    row = pose_df.iloc[best_candidate_row_index]
    dt_ms = abs(row['Timestamp'] - target_ts_ms)
    
    if dt_ms > SYNC_THRESHOLD_MS:
        print(f"[WARNING] 时间戳偏差 {dt_ms:.1f}ms (File: {target_ts_ms}ms, CSV: {row['Timestamp']}ms)，仍使用最近位姿。")

    pos_mm = np.array([row['XToGlobal1'], row['YToGlobal1'], row['ZToGlobal1']], dtype=np.float64)
    quat = np.array([row['QxToGlobal1'], row['QyToGlobal1'], row['QzToGlobal1'], row['QwToGlobal1']], dtype=np.float64)
    
    # 归一化四元数（以防万一）
    quat = quat / np.linalg.norm(quat)
    
    return (pos_mm, quat, dt_ms), dt_ms

# ===================== 文件读取与处理主循环 ======================
def file_processing_loop():
    global frame_idx, running
    print(f"[FILES] 正在扫描激光点文件目录: {LASER_POINTS_DIR}")
    
    # 查找所有匹配的文件
    pattern = os.path.join(LASER_POINTS_DIR, "image_points_*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[ERROR] 在目录 {LASER_POINTS_DIR} 中未找到任何 'image_points_*.txt' 文件！")
        return

    # 从文件名中提取时间戳并排序
    def extract_timestamp(filename):
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        if len(parts) >= 3 and parts[0] == 'image' and parts[1] == 'points':
            try:
                # 提取 '1767862067962128341.txt' 中的数字部分
                ts_part = parts[2].split('.')[0]
                return int(ts_part)
            except ValueError:
                print(f"[WARNING] 无法从文件名 '{base_name}' 解析时间戳。")
                return float('inf') # 将无效文件排到最后
        return float('inf')

    sorted_files = sorted(files, key=extract_timestamp)
    valid_sorted_files = [f for f in sorted_files if extract_timestamp(f) != float('inf')]
    
    if not valid_sorted_files:
        print(f"[ERROR] 在目录 {LASER_POINTS_DIR} 中未找到任何有效的 'image_points_*.txt' 文件！")
        return

    print(f"[FILES] 找到 {len(valid_sorted_files)} 个有效激光点文件，按时间戳排序。")

    for laser_file_path in valid_sorted_files:
        if not running:
            break
        
        start_time_per_frame = time.time() # 记录处理单帧的开始时间

        # 1. 提取时间戳
        file_timestamp_ns = extract_timestamp(laser_file_path)
        print(f"\n[Processing File] {os.path.basename(laser_file_path)}, Timestamp (ns): {file_timestamp_ns}")

        # 2. 读取激光点
        points_img = []
        try:
            with open(laser_file_path, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split()
                        if len(parts) != 2:
                            print(f"[WARNING] 文件 {os.path.basename(laser_file_path)} 第 {line_num} 行格式错误: '{line.strip()}'")
                            continue
                        u, v = float(parts[0]), float(parts[1])
                        points_img.append({'x': u, 'y': v})
                    except ValueError:
                        print(f"[WARNING] 文件 {os.path.basename(laser_file_path)} 第 {line_num} 行包含非数字: '{line.strip()}'")
                        continue
        except IOError as e:
            print(f"[ERROR] 读取文件 {laser_file_path} 失败: {e}")
            continue

        if not points_img:
            print(f"[WARNING] 文件 {laser_file_path} 中没有有效的激光点数据。")
            continue

        # 3. 查找对应位姿
        pose_info, sync_delay_ms = find_nearest_pose(file_timestamp_ns)
        if pose_info is None:
            print(f"[ERROR] 文件 {laser_file_path} (ts={file_timestamp_ns}) 未能找到匹配的位姿。")
            continue

        auv_pos_mm, auv_quat, _ = pose_info

        # 4. 处理激光点到世界坐标系
        xs = np.array([p["x"] for p in points_img], dtype=np.float64)
        depths = compute_depths(xs)
        cam_pts_mm = image_to_camera(points_img, depths)
        world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat)

        # ======== 调试打印开始 (可选) ========
        print(f"[DEBUG Frame {frame_idx:05d}] File: {os.path.basename(laser_file_path)}")
        print(f"  Laser Points Count: {len(points_img)}")
        if points_img:
            print(f"  X coords - min: {xs.min():.2f}, max: {xs.max():.2f}, cx: {cx:.2f}")
        print(f"  Computed Depths - min: {depths.min():.2f}, max: {depths.max():.2f}, mean: {depths.mean():.2f}")
        print(f"  Camera Points - Z coords min: {cam_pts_mm[:, 2].min():.2f}, max: {cam_pts_mm[:, 2].max():.2f}, mean: {cam_pts_mm[:, 2].mean():.2f}")
        print(f"  AUV Pose - Position: [{auv_pos_mm[0]:.2f}, {auv_pos_mm[1]:.2f}, {auv_pos_mm[2]:.2f}]")
        print(f"  World Points - Z coords min: {world_pts_mm[:, 2].min():.2f}, max: {world_pts_mm[:, 2].max():.2f}, mean: {world_pts_mm[:, 2].mean():.2f}")
        # ======== 调试打印结束 ========

        # 5. 构建 ZMQ 消息并发送
        pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat)  # 4x4 嵌套列表
        points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)]

        payload = {
            "header": {
                "timestamp": time.time(), # 使用处理时刻的时间戳
                "frame_id": "lidar",
                "frame_idx": frame_idx,
                "pose": pose_matrix  # 格式: [[...], [...], [...], [...]]
            },
            "points": points_flat_mm  # 毫米单位，扁平列表
        }
        try:
            zmq_socket.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            print(f"[ZMQ] 发送队列满，丢弃帧 {frame_idx}。")

        # 6. 保存到 PLY 文件
        append_points_to_ply(world_pts_mm)

        print(f"[Published] Frame {frame_idx:05d} | File: {os.path.basename(laser_file_path)} | {len(points_img)} pts | "
              f"sync_delay: {sync_delay_ms:+.1f}ms | Total Points: {total_points}")

        frame_idx += 1

        # 7. 模拟实时性 (可选)
        # 你可以在这里添加 sleep 来控制处理速度，使其更像实时流
        # elapsed_time = time.time() - start_time_per_frame
        # desired_sleep_time = 0.1 - elapsed_time # 模拟 10Hz (0.1s per frame)
        # if desired_sleep_time > 0:
        #     time.sleep(desired_sleep_time)


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 加载位姿数据
    pose_df = load_poses_from_csv(CSV_POSE_PATH)
    if pose_df is None:
        print("[FATAL ERROR] 无法加载位姿数据，程序退出。")
        sys.exit(1)

    # 2. 创建输出 PLY 文件
    create_new_ply_file()

    print("\n=== 激光图像文件 + CSV位姿实时融合系统 + ZMQ发布（全系统单位：毫米）已启动 ===")
    print("程序将按时间戳顺序读取激光点文件，匹配位姿，并以毫米（mm）为单位通过 ZMQ 发布！\n")

    # 3. 启动文件处理循环
    try:
        file_processing_loop()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    finally:
        running = False
        time.sleep(0.5) # 给 ZMQ 一点时间清空
        zmq_socket.close()
        context.term()
        print("资源已释放，程序退出。")
        sys.exit(0)